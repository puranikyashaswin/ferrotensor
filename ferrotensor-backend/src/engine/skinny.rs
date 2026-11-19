//! The skinny matmul: `out[M, N] = alpha * a[M, K] @ op(b) + beta * out`
//! with f32 operands and few output columns (N up to [`SKINNY_MAX_COLS`]),
//! the shape of the LoRA products `x @ A` and `dY @ B^T`.
//!
//! The thin kernel runs these as the transposed product, N rows of M
//! columns, with one workgroup per 32 of the M columns: a 256-row product
//! ran as 8 workgroups on a GPU with dozens of compute units. Here a
//! workgroup of 256 threads takes `rows` rows of `a` and `cols` output
//! columns; the `256 / rows` lanes of a row split K between them, read `a`
//! along K (coalesced, four values at a time when K allows), and a tree in
//! workgroup memory adds the lanes' sums in a fixed order. One pass, no
//! second reduction dispatch, and the same bits on every run.

use super::{uniform, ComputeEngine, Epilogue, Kernel};
use crate::kernel_timing::KernelLabel;
use crate::{DType, UnifiedTensor, WgpuContext};
use std::borrow::Cow;
use std::sync::{Arc, PoisonError};

/// Most output columns the router sends to the skinny kernel.
pub(super) const SKINNY_MAX_COLS: u32 = 64;

/// Threads per workgroup.
const THREADS: u32 = 256;

/// Largest workgroup count per grid dimension.
const MAX_GROUPS: u32 = 65_535;

/// A skinny kernel's workgroup: `rows` rows of `a` by `cols` output columns.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SkinnyTile {
    /// Rows of `a` per workgroup; `256 / rows` lanes split each row's K.
    pub rows: u32,
    /// Output columns per workgroup, a multiple of 4.
    pub cols: u32,
}

impl SkinnyTile {
    /// Every tile the generator accepts, for tests and the tuning bench.
    pub const ALL: [Self; 13] = [
        Self::new(1, 4),
        Self::new(1, 8),
        Self::new(1, 16),
        Self::new(2, 4),
        Self::new(2, 8),
        Self::new(2, 16),
        Self::new(4, 4),
        Self::new(4, 8),
        Self::new(4, 16),
        Self::new(8, 4),
        Self::new(8, 8),
        Self::new(8, 16),
        Self::new(16, 8),
    ];

    const fn new(rows: u32, cols: u32) -> Self {
        Self { rows, cols }
    }

    /// Threads that split one row's K.
    pub fn lanes(self) -> u32 {
        THREADS / self.rows
    }

    /// Whether the generator accepts the tile: `rows` a power of two that
    /// leaves at least `cols` lanes per row, `cols` a positive multiple of 4.
    pub fn is_well_formed(self) -> bool {
        self.rows.is_power_of_two()
            && self.rows <= THREADS
            && self.cols > 0
            && self.cols.is_multiple_of(4)
            && self.lanes() >= self.cols
    }

    /// Bytes of workgroup memory: one f32 per thread and column.
    pub fn workgroup_bytes(self) -> u32 {
        self.cols * THREADS * 4
    }

    fn groups(self, m: u32, n: u32) -> (u32, u32) {
        (m.div_ceil(self.rows), n.div_ceil(self.cols))
    }
}

/// A compiled skinny kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct SkinnyKey {
    tile: SkinnyTile,
    /// `b` stored `[N, K]` (true) or `[K, N]`.
    transposed_b: bool,
    /// `a` and `b` read as `vec4<f32>`: K a multiple of 4, and for `b[K, N]`
    /// N too.
    vec4: bool,
}

/// The default tile for an `m x n` product: as many rows per workgroup as
/// keep the grid at `target` workgroups or more, and `cols` covering N in
/// one column block up to 16.
pub(super) fn default_tile(m: u32, n: u32, target: u32) -> SkinnyTile {
    let cols = if n <= 4 {
        4
    } else if n <= 8 {
        8
    } else {
        16
    };
    let rows = [8, 4, 2]
        .into_iter()
        .find(|&rows| {
            let (gx, gy) = SkinnyTile::new(rows, cols).groups(m, n);
            gx * gy >= target
        })
        .unwrap_or(1);
    SkinnyTile::new(rows, cols)
}

/// The K loop for `key`: each lane adds its share of every column's sum.
fn k_loop(key: SkinnyKey) -> String {
    let cols = key.tile.cols;
    let mut s = String::new();
    match (key.vec4, key.transposed_b) {
        (true, true) => {
            for j in 0..cols {
                s += &format!("  var acc{j} = 0.0;\n");
            }
            s += "  let k4 = k / 4u;\n  for (var q = lane; q < k4; q += LANES) {\n    let av = a[row * k4 + q];\n";
            for j in 0..cols {
                s += &format!("    acc{j} += dot(av, b[min(n0 + {j}u, n - 1u) * k4 + q]);\n");
            }
            s += "  }\n";
        }
        (true, false) => {
            for g in 0..cols / 4 {
                s += &format!("  var acc{g} = vec4<f32>(0.0);\n");
            }
            s += "  let k4 = k / 4u;\n  let n4 = n / 4u;\n  for (var q = lane; q < k4; q += LANES) {\n    let av = a[row * k4 + q];\n";
            for (e, c) in ["x", "y", "z", "w"].into_iter().enumerate() {
                for g in 0..cols / 4 {
                    // N is a multiple of 4 here, so a block past N reads the last one.
                    s += &format!(
                        "    acc{g} = fma(vec4<f32>(av.{c}), b[(q * 4u + {e}u) * n4 + min(n0 / 4u + {g}u, n4 - 1u)], acc{g});\n"
                    );
                }
            }
            s += "  }\n";
        }
        (false, transposed_b) => {
            for j in 0..cols {
                s += &format!("  var acc{j} = 0.0;\n");
            }
            s += "  for (var kk = lane; kk < k; kk += LANES) {\n    let av = a[row * k + kk];\n";
            for j in 0..cols {
                let idx = if transposed_b {
                    format!("min(n0 + {j}u, n - 1u) * k + kk")
                } else {
                    format!("kk * n + min(n0 + {j}u, n - 1u)")
                };
                s += &format!("    acc{j} = fma(av, b[{idx}], acc{j});\n");
            }
            s += "  }\n";
        }
    }
    // Column j of this thread's sums goes to red[j * 256 + li].
    for j in 0..cols {
        let v = if key.vec4 && !key.transposed_b {
            format!("acc{}.{}", j / 4, ["x", "y", "z", "w"][(j % 4) as usize])
        } else {
            format!("acc{j}")
        };
        s += &format!("  red[{j}u * 256u + li] = {v};\n");
    }
    s
}

/// The WGSL of the kernel for `key`, entry point `matmul_skinny`. Params are
/// `[M, K, N, 0]` and `[alpha bits, beta bits, 0, 0]`.
pub(super) fn source(key: SkinnyKey) -> String {
    let t = key.tile;
    let elem = if key.vec4 { "vec4<f32>" } else { "f32" };
    let adds: String = (0..t.cols)
        .map(|j| format!("      red[{j}u * 256u + li] += red[{j}u * 256u + li + s];\n"))
        .collect();
    format!(
        "@group(0) @binding(0) var<storage, read> a: array<{elem}>;
@group(0) @binding(1) var<storage, read> b: array<{elem}>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, 2>;

const ROWS: u32 = {rows}u;
const LANES: u32 = {lanes}u;
const COLS: u32 = {cols}u;

var<workgroup> red: array<f32, {red_len}>;

@compute @workgroup_size(256)
fn matmul_skinny(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
  let m = params[0].x;
  let k = params[0].y;
  let n = params[0].z;
  let r = li / LANES;
  let lane = li % LANES;
  // Rows and columns past the end read the last one; their sums are not stored.
  let row = min(wg.x * ROWS + r, m - 1u);
  let n0 = wg.y * COLS;
{k_loop}  workgroupBarrier();
  // Halving tree over the lanes of each row, the same order on every run.
  for (var s = LANES / 2u; s > 0u; s = s / 2u) {{
    if (lane < s) {{
{adds}    }}
    workgroupBarrier();
  }}
  let out_row = wg.x * ROWS + r;
  let col = n0 + lane;
  if (lane < COLS && out_row < m && col < n) {{
    let i = out_row * n + col;
    let beta = bitcast<f32>(params[1].y);
    var o = bitcast<f32>(params[1].x) * red[lane * 256u + r * LANES];
    if (beta != 0.0) {{ o = o + beta * c[i]; }}
    c[i] = o;
  }}
}}
",
        rows = t.rows,
        lanes = t.lanes(),
        cols = t.cols,
        red_len = t.cols * THREADS,
        k_loop = k_loop(key),
    )
}

impl ComputeEngine {
    /// Whether this device can run `tile`.
    pub(super) fn skinny_fits(&self, tile: SkinnyTile) -> bool {
        tile.is_well_formed()
            && THREADS <= self.small_m_max_threads
            && tile.workgroup_bytes() <= self.small_m_max_bytes
    }

    /// The tile the router runs an `m x n` product with, if one fits.
    pub(super) fn skinny_tile(&self, m: u32, n: u32) -> Option<SkinnyTile> {
        let tile = default_tile(m, n, self.split_target);
        let (gx, gy) = tile.groups(m, n);
        (self.skinny_fits(tile) && gx <= MAX_GROUPS && gy <= MAX_GROUPS).then_some(tile)
    }

    fn skinny_kernel(&self, ctx: &WgpuContext, key: SkinnyKey) -> Arc<Kernel> {
        let mut cache = self.skinny.lock().unwrap_or_else(PoisonError::into_inner);
        Arc::clone(cache.entry(key).or_insert_with(|| {
            let module = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("matmul_skinny"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Owned(source(key))),
                });
            Arc::new(Kernel::new(&ctx.device, &module, "matmul_skinny"))
        }))
    }

    /// Records the skinny kernel with `tile` for `out = alpha * a @ op(b) +
    /// beta * out` (`ep`), `b` f32 and stored `[N, K]` (`transposed_b`) or
    /// `[K, N]`. The caller checks the tile and the grid.
    pub(super) fn matmul_skinny(
        &self,
        ctx: &WgpuContext,
        transposed_b: bool,
        (m, k, n): (u32, u32, u32),
        [a, b, out]: [&UnifiedTensor; 3],
        tile: SkinnyTile,
        ep: Epilogue,
    ) {
        let vec4 = k.is_multiple_of(4) && (transposed_b || n.is_multiple_of(4));
        let kernel = self.skinny_kernel(
            ctx,
            SkinnyKey {
                tile,
                transposed_b,
                vec4,
            },
        );
        let params = uniform(
            ctx,
            bytemuck::cast_slice(&[m, k, n, 0, ep.alpha.to_bits(), ep.beta.to_bits(), 0, 0]),
        );
        let label = KernelLabel {
            entry: "matmul_skinny",
            layout: if transposed_b { "a@bT" } else { "a@b" },
            size_class: "n<=64",
        };
        let (gx, gy) = tile.groups(m, n);
        self.run_labeled(
            ctx,
            &kernel,
            label,
            &[
                (0, a.buffer()),
                (1, b.buffer()),
                (2, out.buffer()),
                (3, &params),
            ],
            (gx, gy, 1),
        );
    }

    /// Runs `out = alpha * a[M, K] @ op(b) + beta * out` (`ep`) on the
    /// skinny kernel with `tile`, or the tile the router would pick, with
    /// f32 `b` stored `[N, K]` (`transposed_b`) or `[K, N]`, and returns the
    /// tile it ran with.
    ///
    /// # Errors
    ///
    /// Returns an error when `b` is not f32, the shapes do not match, N is
    /// above 64, or the tile or its grid does not fit this device.
    #[doc(hidden)]
    pub fn matmul_skinny_into(
        &self,
        ctx: &WgpuContext,
        transposed_b: bool,
        [a, b, out]: [&UnifiedTensor; 3],
        tile: Option<SkinnyTile>,
        ep: Epilogue,
    ) -> Result<SkinnyTile, String> {
        if b.dtype != DType::FP32 {
            return Err("matmul_skinny_into: b must be f32".into());
        }
        let ([m, k], [b0, b1]) = (a.shape.as_slice(), b.shape.as_slice()) else {
            return Err(format!(
                "matmul_skinny_into: a and b must be 2-D, got {:?} and {:?}",
                a.shape, b.shape
            ));
        };
        let (b_k, n) = if transposed_b { (b1, b0) } else { (b0, b1) };
        if b_k != k || out.shape.iter().product::<usize>() != m * n {
            return Err(format!(
                "matmul_skinny_into: a {:?}, b {:?} (transposed_b {transposed_b}) and out {:?} do not match",
                a.shape, b.shape, out.shape
            ));
        }
        let (m, k, n) = (*m as u32, *k as u32, *n as u32);
        if m == 0 || k == 0 || n == 0 || n > SKINNY_MAX_COLS {
            return Err(format!(
                "matmul_skinny_into: {m} x {k} x {n} is empty or wider than {SKINNY_MAX_COLS} columns"
            ));
        }
        let tile = match tile {
            Some(tile) => tile,
            None => default_tile(m, n, self.split_target),
        };
        let (gx, gy) = tile.groups(m, n);
        if !self.skinny_fits(tile) || gx > MAX_GROUPS || gy > MAX_GROUPS {
            return Err(format!(
                "matmul_skinny_into: {tile:?} over {m} x {n} does not fit this device"
            ));
        }
        self.matmul_skinny(ctx, transposed_b, (m, k, n), [a, b, out], tile, ep);
        Ok(tile)
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::{default_tile, source, SkinnyKey, SkinnyTile};

    fn keys() -> Vec<SkinnyKey> {
        let mut keys = Vec::new();
        for tile in SkinnyTile::ALL {
            for transposed_b in [false, true] {
                for vec4 in [false, true] {
                    keys.push(SkinnyKey {
                        tile,
                        transposed_b,
                        vec4,
                    });
                }
            }
        }
        keys
    }

    #[test]
    fn every_kernel_validates() {
        for tile in SkinnyTile::ALL {
            assert!(tile.is_well_formed(), "{tile:?}");
            assert!(tile.workgroup_bytes() <= 16 * 1024, "{tile:?}");
        }
        for key in keys() {
            let src = source(key);
            let module = match naga::front::wgsl::parse_str(&src) {
                Ok(m) => m,
                Err(e) => panic!("{key:?}: WGSL parse error:\n{}", e.emit_to_string(&src)),
            };
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            );
            if let Err(e) = validator.validate(&module) {
                panic!(
                    "{key:?}: WGSL validation error:\n{}",
                    e.emit_to_string(&src)
                );
            }
        }
    }

    #[test]
    fn default_tile_reaches_the_target_grid() {
        // 135M LoRA at b1: 256 rows, rank 8; at b8: 2048 rows.
        assert_eq!(default_tile(256, 8, 160), SkinnyTile::new(1, 8));
        assert_eq!(default_tile(2048, 8, 160), SkinnyTile::new(8, 8));
        assert_eq!(default_tile(640, 8, 160), SkinnyTile::new(4, 8));
        assert_eq!(default_tile(2048, 64, 160), SkinnyTile::new(8, 16));
        assert_eq!(default_tile(100, 3, 160), SkinnyTile::new(1, 4));
        for m in [1, 7, 256, 2048, 100_000] {
            for n in [1, 4, 5, 8, 9, 16, 33, 64] {
                assert!(default_tile(m, n, 160).is_well_formed(), "{m} x {n}");
            }
        }
    }
}
