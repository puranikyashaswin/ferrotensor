//! Matrix multiplication kernels.

use super::{grid, uniform, ComputeEngine, Kernel};
use crate::kernel_timing::KernelLabel;
use crate::{DType, UnifiedTensor, WgpuContext};

/// Operand layout understood by the `matmul_blocked` kernel.
#[derive(Clone, Copy)]
#[repr(u32)]
enum Layout {
    /// `a @ b` with `a: [M, K]`, `b: [K, N]`.
    Plain = 0,
    /// `a @ b^T` with `a: [M, K]`, `b: [N, K]`.
    TransposedB = 1,
    /// `a^T @ b` with `a: [K, M]`, `b: [K, N]`.
    TransposedA = 2,
}

impl Layout {
    fn name(self) -> &'static str {
        match self {
            Self::Plain => "a@b",
            Self::TransposedB => "a@bT",
            Self::TransposedA => "aT@b",
        }
    }
}

/// The operand the thin kernel reads as its thin side `t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Thin {
    /// `a`, for an output with few rows.
    A,
    /// `b`, for an output with few columns.
    B,
}

/// Most output rows (or columns) the thin kernel takes: past this the
/// 64x64-tile kernel launches more than one row of tiles.
const THIN_MAX_ROWS: u32 = 64;

/// The thin operand of an `[M, N]` product with `b` stored as `dtype`, if
/// the thin kernel runs it: the shorter output side when it is at most
/// [`THIN_MAX_ROWS`]. With f16 `b` only `a` can be thin, since the f16
/// kernel reads its wide operand as f16; 4-bit `b` keeps the tile kernel.
fn thin_side(dtype: DType, m: u32, n: u32) -> Option<Thin> {
    let (thin, rows, cols) = match dtype {
        DType::FP32 if n < m => (Thin::B, n, m),
        DType::FP32 | DType::FP16 => (Thin::A, m, n),
        DType::Q4 => return None,
    };
    // The grid's y dimension counts 32-column blocks.
    (rows <= THIN_MAX_ROWS && cols.div_ceil(32) <= 65_535).then_some(thin)
}

/// The thin kernel's params, `[rows, K, cols, mode]` then the strides
/// `[t row, t k, out row, out col]` (see matmul_thin.wgsl). With `b` thin
/// it computes `out^T = op(b)^T @ op(a)^T` and stores the result transposed.
fn thin_params(layout: Layout, (m, k, n): (u32, u32, u32), thin: Thin) -> [u32; 8] {
    match (thin, layout) {
        (Thin::A, Layout::Plain) => [m, k, n, 0, k, 1, n, 1],
        (Thin::A, Layout::TransposedB) => [m, k, n, 1, k, 1, n, 1],
        (Thin::A, Layout::TransposedA) => [m, k, n, 0, 1, m, n, 1],
        (Thin::B, Layout::Plain) => [n, k, m, 1, 1, n, 1, n],
        (Thin::B, Layout::TransposedB) => [n, k, m, 1, k, 1, 1, n],
        (Thin::B, Layout::TransposedA) => [n, k, m, 0, 1, n, 1, n],
    }
}

impl ComputeEngine {
    /// `out[M, N] = a[M, K] @ b[K, N]`.
    pub fn matmul(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let dims = (a.shape[0] as u32, a.shape[1] as u32, b.shape[1] as u32);
        self.matmul_run(ctx, Layout::Plain, dims, a, b, out);
    }

    /// `out[M, N] = a[M, K] @ b[N, K]^T` (e.g. the input gradient of a
    /// linear layer).
    pub fn matmul_bt(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let dims = (a.shape[0] as u32, a.shape[1] as u32, b.shape[0] as u32);
        self.matmul_run(ctx, Layout::TransposedB, dims, a, b, out);
    }

    /// `out[M, N] = a[K, M]^T @ b[K, N]` (e.g. the weight gradient of a
    /// linear layer).
    pub fn matmul_at(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let dims = (a.shape[1] as u32, a.shape[0] as u32, b.shape[1] as u32);
        self.matmul_run(ctx, Layout::TransposedA, dims, a, b, out);
    }

    /// [`matmul`](Self::matmul) with `b[K, N]` stored as f16
    /// ([`UnifiedTensor::from_f32_as_f16`]). Products and sums are f32.
    pub fn matmul_f16b(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        assert_eq!(b.dtype, DType::FP16, "matmul_f16b: b must be FP16");
        let dims = (a.shape[0] as u32, a.shape[1] as u32, b.shape[1] as u32);
        self.matmul_run(ctx, Layout::Plain, dims, a, b, out);
    }

    /// [`matmul_bt`](Self::matmul_bt) with `b[N, K]` stored as f16.
    pub fn matmul_bt_f16b(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        assert_eq!(b.dtype, DType::FP16, "matmul_bt_f16b: b must be FP16");
        let dims = (a.shape[0] as u32, a.shape[1] as u32, b.shape[0] as u32);
        self.matmul_run(ctx, Layout::TransposedB, dims, a, b, out);
    }

    /// [`matmul`](Self::matmul) with `b[K, N]` stored in the 4-bit format
    /// ([`UnifiedTensor::from_f32_as_q4`]). Products and sums are f32. This
    /// is the forward product `x · W` of a frozen 4-bit weight `W[K, N]`.
    pub fn matmul_q4b(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        assert_eq!(b.dtype, DType::Q4, "matmul_q4b: b must be Q4");
        let dims = (a.shape[0] as u32, a.shape[1] as u32, b.shape[1] as u32);
        self.matmul_run(ctx, Layout::Plain, dims, a, b, out);
    }

    /// [`matmul_bt`](Self::matmul_bt) with `b[N, K]` stored in the 4-bit
    /// format. With `b` a frozen 4-bit weight `W` of shape `[K', N']` and
    /// `a = dY[M, N']`, this is the input gradient `dX = dY · Wᵀ` of
    /// `Y = X · W`, read from the same tensor as [`matmul_q4b`](Self::matmul_q4b).
    pub fn matmul_bt_q4b(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        assert_eq!(b.dtype, DType::Q4, "matmul_bt_q4b: b must be Q4");
        let dims = (a.shape[0] as u32, a.shape[1] as u32, b.shape[0] as u32);
        self.matmul_run(ctx, Layout::TransposedB, dims, a, b, out);
    }

    /// Runs `out[M, N] = op(a) @ op(b)` with the kernel that suits the shape
    /// and `b`'s storage: the matrix-vector kernel for one row, the thin
    /// kernel when [`thin_side`] finds a thin operand, else the 64x64-tile
    /// kernel.
    fn matmul_run(
        &self,
        ctx: &WgpuContext,
        layout: Layout,
        (m, k, n): (u32, u32, u32),
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        if m == 1 && matches!(layout, Layout::Plain) {
            self.matvec(ctx, (k, n), a, b, out);
            return;
        }
        if let Some(thin) = thin_side(b.dtype, m, n) {
            let kernel = match b.dtype {
                DType::FP16 => &self.matmul_thin_h,
                _ => &self.matmul_thin,
            };
            let (t, w) = match thin {
                Thin::A => (a, b),
                Thin::B => (b, a),
            };
            let params = thin_params(layout, (m, k, n), thin);
            let (rows, cols) = (params[0], params[2]);
            let label = KernelLabel {
                entry: kernel.entry_point,
                layout: layout.name(),
                size_class: match thin {
                    Thin::A => "m<=64",
                    Thin::B => "n<=64",
                },
            };
            let params = uniform(ctx, bytemuck::cast_slice(&params));
            // One 16x16 workgroup per 8 rows by 32 columns of the thin product.
            self.run_labeled(
                ctx,
                kernel,
                label,
                &[
                    (0, t.buffer()),
                    (1, w.buffer()),
                    (2, out.buffer()),
                    (3, &params),
                ],
                (rows.div_ceil(8), cols.div_ceil(32), 1),
            );
            return;
        }
        let kernel = match b.dtype {
            DType::FP32 => &self.matmul,
            DType::FP16 => &self.matmul_h,
            DType::Q4 => &self.matmul_q4,
        };
        // The 4-bit kernel reads log2 of the group size above the layout.
        let mode = layout as u32 | b.q4_group().map_or(0, |g| g.size().trailing_zeros() << 8);
        let params = uniform(ctx, bytemuck::cast_slice(&[m, k, n, mode]));
        // One 16x16 workgroup per 64x64 output tile.
        self.run_labeled(
            ctx,
            kernel,
            matmul_label(kernel, layout, m, k, n),
            &[
                (0, a.buffer()),
                (1, b.buffer()),
                (2, out.buffer()),
                (3, &params),
            ],
            (n.div_ceil(64), m.div_ceil(64), 1),
        );
    }

    /// `out[1, N] = a[1, K] @ b[K, N]`, the shape of every linear layer when
    /// decoding one token: the threads of a workgroup split K, where the
    /// 64x64-tile kernel would compute one useful row of 64.
    fn matvec(
        &self,
        ctx: &WgpuContext,
        (k, n): (u32, u32),
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let (kernel, columns_per_group, log2_group) = match b.dtype {
            DType::FP32 => (&self.matvec, 32, 0),
            DType::FP16 => (&self.matvec_h, 32, 0),
            DType::Q4 => (
                &self.matvec_q4,
                8,
                b.q4_group().map_or(0, |g| g.size().trailing_zeros()),
            ),
        };
        let params = uniform(ctx, bytemuck::cast_slice(&[k, n, log2_group, 0]));
        self.run(
            ctx,
            kernel,
            &[
                (0, a.buffer()),
                (1, b.buffer()),
                (2, out.buffer()),
                (3, &params),
            ],
            grid(n.div_ceil(columns_per_group)),
        );
    }
}

/// The kernel timing label of a matmul: its layout and a size class, since
/// decode rows (`m <= 4`), LoRA-rank shapes and large products run
/// differently.
fn matmul_label(kernel: &Kernel, layout: Layout, m: u32, k: u32, n: u32) -> KernelLabel {
    let size_class = if m <= 4 {
        "m<=4"
    } else if m.min(k).min(n) <= 64 {
        "min dim<=64"
    } else if u64::from(m) * u64::from(n) < 1 << 20 {
        "out<1M"
    } else {
        "out>=1M"
    };
    KernelLabel {
        entry: kernel.entry_point,
        layout: layout.name(),
        size_class,
    }
}

#[cfg(test)]
mod tests {
    use super::{thin_params, thin_side, DType, Layout, Thin};

    /// The thin kernel's indexing on the host: `out(r, c) = sum t(r, k) w(k, c)`
    /// through the params' strides and layout mode.
    fn thin_on_host(params: [u32; 8], t: &[f32], w: &[f32], out_len: usize) -> Vec<f32> {
        let [rows, k, cols, mode, t_row, t_k, out_row, out_col] = params.map(|v| v as usize);
        let mut out = vec![f32::NAN; out_len];
        for r in 0..rows {
            for c in 0..cols {
                out[r * out_row + c * out_col] = (0..k)
                    .map(|kk| {
                        let wv = if mode == 0 {
                            w[kk * cols + c]
                        } else {
                            w[c * k + kk]
                        };
                        t[r * t_row + kk * t_k] * wv
                    })
                    .sum();
            }
        }
        out
    }

    fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| x[(i % rows) * cols + i / rows])
            .collect()
    }

    #[test]
    fn thin_params_index_every_layout_from_either_side() {
        let (m, k, n) = (3, 5, 4);
        // Small integers keep every sum exact.
        let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();
        let mut want = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                want[i * n + j] = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
            }
        }
        let a_t = transpose(&a, m, k);
        let b_t = transpose(&b, k, n);
        let stored = [
            (Layout::Plain, &a, &b),
            (Layout::TransposedB, &a, &b_t),
            (Layout::TransposedA, &a_t, &b),
        ];
        for (layout, a_s, b_s) in stored {
            for thin in [Thin::A, Thin::B] {
                let params = thin_params(layout, (m as u32, k as u32, n as u32), thin);
                let (t, w) = match thin {
                    Thin::A => (a_s, b_s),
                    Thin::B => (b_s, a_s),
                };
                assert_eq!(
                    thin_on_host(params, t, w, m * n),
                    want,
                    "{} with {thin:?} thin",
                    layout.name()
                );
            }
        }
    }

    #[test]
    fn thin_side_picks_the_short_output_side() {
        // A short pass against an f16 weight, forward and input gradient.
        assert_eq!(thin_side(DType::FP16, 40, 2048), Some(Thin::A));
        assert_eq!(thin_side(DType::FP16, 64, 8192), Some(Thin::A));
        assert_eq!(thin_side(DType::FP16, 65, 2048), None);
        // LoRA products of rank 8: x @ A has 8 columns, (x A) @ B and dB few rows.
        assert_eq!(thin_side(DType::FP32, 200, 8), Some(Thin::B));
        assert_eq!(thin_side(DType::FP32, 8, 2048), Some(Thin::A));
        assert_eq!(thin_side(DType::FP32, 64, 64), Some(Thin::A));
        assert_eq!(thin_side(DType::FP32, 100, 100), None);
        // f16 b can only be the wide side; 4-bit b keeps the tile kernel.
        assert_eq!(thin_side(DType::FP16, 200, 8), None);
        assert_eq!(thin_side(DType::Q4, 3, 2048), None);
        // More 32-column blocks than one grid dimension holds.
        assert_eq!(thin_side(DType::FP32, 8, 65_535 * 32 + 1), None);
    }
}
