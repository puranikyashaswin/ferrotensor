//! Matrix multiplication kernels.

use super::{uniform, ComputeEngine, Kernel};
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
        self.matmul_dispatch(ctx, Layout::Plain, dims, a, b, out);
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
        self.matmul_dispatch(ctx, Layout::TransposedB, dims, a, b, out);
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
        self.matmul_dispatch(ctx, Layout::TransposedA, dims, a, b, out);
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
        self.matmul_run(ctx, &self.matmul_h, Layout::Plain, dims, a, b, out);
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
        self.matmul_run(ctx, &self.matmul_h, Layout::TransposedB, dims, a, b, out);
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
        self.matmul_run(ctx, &self.matmul_q4, Layout::Plain, dims, a, b, out);
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
        self.matmul_run(ctx, &self.matmul_q4, Layout::TransposedB, dims, a, b, out);
    }

    fn matmul_dispatch(
        &self,
        ctx: &WgpuContext,
        layout: Layout,
        dims: (u32, u32, u32),
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.matmul_run(ctx, &self.matmul, layout, dims, a, b, out);
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_run(
        &self,
        ctx: &WgpuContext,
        kernel: &Kernel,
        layout: Layout,
        (m, k, n): (u32, u32, u32),
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
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
        layout: match layout {
            Layout::Plain => "a@b",
            Layout::TransposedB => "a@bT",
            Layout::TransposedA => "aT@b",
        },
        size_class,
    }
}
