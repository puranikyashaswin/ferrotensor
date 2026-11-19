//! Matmuls on cooperative matrices: `out = alpha * a @ op(b) + beta * out`
//! with f32 `a`, f16 `b`, 16x16x16 f16 fragments and f32 accumulation.
//!
//! The kernel stages `a` (converted to f16) and `b` in workgroup memory, one
//! K step of `BK` at a time, and each subgroup of 32 threads multiplies its
//! own block of 16x16 fragments. The accumulators go back through workgroup
//! memory, so the epilogue and the edges of M and N that are not multiples
//! of 16 are handled per element.
//!
//! A value of `a` that f16 cannot hold (beyond +-65504, or NaN) is clamped
//! for the product, counted in [`ComputeEngine::coop_f16_overflows`], and the
//! first element of that workgroup's output tile is set to NaN, so a dynamic
//! loss scaler sees the overflow as a non-finite gradient.
//!
//! Every workgroup array is a multiple of 16 bytes long. The driver packs
//! them one after another, and a cooperative load or store from an address
//! that is not 16-byte aligned reads or writes the aligned address below it
//! (on the A10G a 4-byte atomic among them shifted the results by elements).

use super::{immediate_params, uniform, ComputeEngine, Epilogue, Kernel};
use crate::kernel_timing::KernelLabel;
use crate::{DType, ReadbackError, UnifiedTensor, WgpuContext};
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

/// K step staged in workgroup memory per iteration.
const BK: u32 = 32;
/// f16 elements of padding after each staged row, against bank conflicts.
const PAD: u32 = 8;
/// Threads per subgroup the kernel is written for.
const SUBGROUP: u32 = 32;
/// Fewest output rows routed to the kernel: decode steps and thin products
/// stay on the f32 kernels.
const MIN_ROWS: u32 = 64;
/// Largest workgroup count per grid dimension.
const MAX_GROUPS: u32 = 65_535;

/// A workgroup's output tile `bm x bn`, split into one `wm x wn` block per
/// subgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct CoopTile {
    bm: u32,
    bn: u32,
    wm: u32,
    wn: u32,
}

impl CoopTile {
    const fn new(bm: u32, bn: u32, wm: u32, wn: u32) -> Self {
        Self { bm, bn, wm, wn }
    }

    fn warps(self) -> u32 {
        (self.bm / self.wm) * (self.bn / self.wn)
    }

    fn threads(self) -> u32 {
        SUBGROUP * self.warps()
    }

    /// Workgroup memory in bytes: the staged `a` and `b` tiles in f16 and
    /// one 16x16 f32 staging fragment per subgroup.
    fn workgroup_bytes(self, transposed_b: bool) -> u32 {
        let a = self.bm * (BK + PAD) * 2;
        let b = if transposed_b {
            self.bn * (BK + PAD) * 2
        } else {
            BK * (self.bn + PAD) * 2
        };
        a + b + self.warps() * 256 * 4
    }

    fn groups(self, m: u32, n: u32) -> (u32, u32) {
        (n.div_ceil(self.bn), m.div_ceil(self.bm))
    }
}

/// The tiles, largest first: 128x64 with four 64x32 subgroup blocks was the
/// fastest shared-memory tile on the 135M shapes in the spike; the smaller
/// ones fill the GPU when M or N is short.
const TILES: [CoopTile; 3] = [
    CoopTile::new(128, 64, 64, 32),
    CoopTile::new(64, 64, 32, 32),
    CoopTile::new(32, 32, 16, 16),
];

/// What one compiled kernel is specialised for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct CoopKey {
    tile: CoopTile,
    /// `b` stored `[N, K]` (true) or `[K, N]`.
    transposed_b: bool,
    /// `a` read as `vec4<f32>`, which needs K to be a multiple of 4.
    a_vec4: bool,
    /// `b` read 8 halves at a time, which needs its row length (N, or K when
    /// transposed) to be a multiple of 8.
    b_vec8: bool,
}

/// The cooperative-matrix kernels of a device that has them.
pub(super) struct Coop {
    kernels: Mutex<HashMap<CoopKey, Arc<Kernel>>>,
    /// One `u32` counter of staging threads that clamped an `a` value.
    overflows: UnifiedTensor,
    /// Whether [`EXPERIMENTAL_ENV`](crate::adapters::EXPERIMENTAL_ENV) routes f16-weight matmuls here.
    routed: bool,
    /// Workgroups a grid aims for, as for the small-M kernel.
    target: u32,
    max_bytes: u32,
    max_threads: u32,
    /// Whether the kernel reads its parameters from immediates.
    immediates: bool,
}

/// `src` with its parameters moved to immediates. The `enable` directives
/// must stay first in the module, so only the text after them is rewritten.
fn immediate_source(src: &str) -> String {
    const LAST_ENABLE: &str = "enable wgpu_cooperative_matrix;\n";
    let split = src.find(LAST_ENABLE).map_or(0, |at| at + LAST_ENABLE.len());
    let (directives, body) = src.split_at(split);
    format!("{directives}{}", immediate_params(body))
}

/// Whether `ctx` can run the kernel: f16 in shaders, 16x16x16 f16
/// fragments accumulated in f32, and subgroups of exactly 32 threads.
fn device_supports(ctx: &WgpuContext) -> bool {
    let wanted = wgpu::Features::SHADER_F16 | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
    let info = ctx.adapter.get_info();
    ctx.device.features().contains(wanted)
        && info.subgroup_min_size == SUBGROUP
        && info.subgroup_max_size == SUBGROUP
        && ctx
            .cooperative_matrix()
            .iter()
            .any(|c| (c.m, c.n, c.k) == (16, 16, 16) && c.ab_type == "f16" && c.cr_type == "f32")
}

impl Coop {
    /// The kernels for `ctx`, or `None` when its device cannot run them.
    pub(super) fn new(ctx: &WgpuContext, target: u32) -> Option<Self> {
        if !device_supports(ctx) {
            return None;
        }
        let limits = ctx.device.limits();
        Some(Self {
            kernels: Mutex::new(HashMap::new()),
            overflows: UnifiedTensor::zeros(ctx, &[1]),
            routed: crate::adapters::experimental_opt_in(),
            target,
            max_bytes: limits.max_compute_workgroup_storage_size,
            max_threads: limits
                .max_compute_invocations_per_workgroup
                .min(limits.max_compute_workgroup_size_x),
            immediates: ctx.uses_immediates(),
        })
    }

    fn fits(&self, tile: CoopTile, transposed_b: bool) -> bool {
        tile.threads() <= self.max_threads && tile.workgroup_bytes(transposed_b) <= self.max_bytes
    }

    /// The largest tile whose grid over `m x n` reaches the target, else the
    /// smallest that fits.
    fn tile(&self, m: u32, n: u32, transposed_b: bool) -> Option<CoopTile> {
        let fitting = TILES.iter().copied().filter(|&t| {
            let (gx, gy) = t.groups(m, n);
            self.fits(t, transposed_b) && gx <= MAX_GROUPS && gy <= MAX_GROUPS
        });
        let mut last = None;
        for tile in fitting {
            let (gx, gy) = tile.groups(m, n);
            if gx * gy >= self.target {
                return Some(tile);
            }
            last = Some(tile);
        }
        last
    }

    fn kernel(&self, ctx: &WgpuContext, key: CoopKey) -> Arc<Kernel> {
        let mut cache = self.kernels.lock().unwrap_or_else(PoisonError::into_inner);
        Arc::clone(cache.entry(key).or_insert_with(|| {
            let mut src = source(key);
            if self.immediates {
                src = immediate_source(&src);
            }
            let module = ctx
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("matmul_coop"),
                    source: wgpu::ShaderSource::Wgsl(Cow::Owned(src)),
                });
            let pipeline = Arc::new(ctx.device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("matmul_coop"),
                    layout: None,
                    module: &module,
                    entry_point: Some("matmul_coop"),
                    // The kernel writes every workgroup value before reading it.
                    compilation_options: wgpu::PipelineCompilationOptions {
                        zero_initialize_workgroup_memory: false,
                        ..Default::default()
                    },
                    cache: None,
                },
            ));
            let layout = pipeline.get_bind_group_layout(0);
            Arc::new(Kernel {
                pipeline,
                layout,
                entry_point: "matmul_coop",
            })
        }))
    }
}

impl ComputeEngine {
    /// Whether this device can run matmuls on cooperative matrices (f16
    /// inputs, f32 accumulation), whether or not they are turned on.
    #[must_use]
    pub fn coop_matmul_available(&self) -> bool {
        self.coop.is_some()
    }

    /// Whether matmuls against f16 weights run on cooperative matrices: the
    /// device has them and [`EXPERIMENTAL_ENV`](crate::adapters::EXPERIMENTAL_ENV) is `1`.
    #[must_use]
    pub fn coop_matmul_active(&self) -> bool {
        self.coop.as_ref().is_some_and(|c| c.routed)
    }

    /// Staging threads of the cooperative-matrix matmul that met an input
    /// value beyond the f16 range (or NaN) since the engine was created; 0
    /// without the kernel. Reading it waits for the GPU.
    ///
    /// # Errors
    ///
    /// [`ReadbackError`] when wgpu cannot map the counter for reading.
    pub async fn coop_f16_overflows(&self, ctx: &WgpuContext) -> Result<u64, ReadbackError> {
        Ok(match self.coop.as_ref() {
            Some(coop) => coop
                .overflows
                .to_vec(ctx)
                .await?
                .first()
                .map_or(0, |v| u64::from(v.to_bits())),
            None => 0,
        })
    }

    /// Whether [`matmul_run`](Self::matmul_run) sends an `m x k x n` product
    /// against a `dtype` weight to the cooperative-matrix kernel.
    pub(super) fn coop_routes(&self, dtype: DType, (m, k, n): (u32, u32, u32)) -> bool {
        self.coop_matmul_active() && dtype == DType::FP16 && m >= MIN_ROWS && k >= 16 && n >= 16
    }

    /// Runs `out = alpha * a[M, K] @ op(b) + beta * out` (`ep`) on the
    /// cooperative-matrix kernel, with f16 `b` stored `[N, K]`
    /// (`transposed_b`) or `[K, N]`, whether or not the router would send
    /// the product there.
    ///
    /// # Errors
    ///
    /// Returns an error when the device has no cooperative matrices, `b` is
    /// not f16, the shapes do not match or no tile fits the device.
    #[doc(hidden)]
    pub fn matmul_coop_into(
        &self,
        ctx: &WgpuContext,
        transposed_b: bool,
        [a, b, out]: [&UnifiedTensor; 3],
        ep: Epilogue,
    ) -> Result<(), String> {
        if self.coop.is_none() {
            return Err("matmul_coop_into: this device has no f16 cooperative matrices".into());
        }
        if b.dtype != DType::FP16 {
            return Err("matmul_coop_into: b must be f16".into());
        }
        let ([m, k], [b0, b1]) = (a.shape.as_slice(), b.shape.as_slice()) else {
            return Err(format!(
                "matmul_coop_into: a and b must be 2-D, got {:?} and {:?}",
                a.shape, b.shape
            ));
        };
        let (b_k, n) = if transposed_b { (b1, b0) } else { (b0, b1) };
        if b_k != k || out.shape.iter().product::<usize>() != m * n {
            return Err(format!(
                "matmul_coop_into: a {:?}, b {:?} (transposed_b {transposed_b}) and out {:?} do not match",
                a.shape, b.shape, out.shape
            ));
        }
        if *m == 0 || *k == 0 || *n == 0 {
            return Err("matmul_coop_into: empty product".into());
        }
        let dims = (*m as u32, *k as u32, *n as u32);
        if self.matmul_coop(ctx, transposed_b, dims, [a, b, out], ep) {
            Ok(())
        } else {
            Err(format!(
                "matmul_coop_into: no tile fits {m} x {n} on this device"
            ))
        }
    }

    /// Records the cooperative-matrix matmul; false (recording nothing) when
    /// the device has no kernel or no tile fits.
    pub(super) fn matmul_coop(
        &self,
        ctx: &WgpuContext,
        transposed_b: bool,
        (m, k, n): (u32, u32, u32),
        [a, b, out]: [&UnifiedTensor; 3],
        ep: Epilogue,
    ) -> bool {
        let Some(coop) = self.coop.as_ref() else {
            return false;
        };
        let Some(tile) = coop.tile(m, n, transposed_b) else {
            return false;
        };
        let b_row = if transposed_b { k } else { n };
        let key = CoopKey {
            tile,
            transposed_b,
            a_vec4: k % 4 == 0,
            b_vec8: b_row % 8 == 0,
        };
        let kernel = coop.kernel(ctx, key);
        let params = uniform(
            ctx,
            bytemuck::cast_slice(&[m, k, n, 0, ep.alpha.to_bits(), ep.beta.to_bits(), 0, 0]),
        );
        let label = KernelLabel {
            entry: "matmul_coop",
            layout: if transposed_b { "a@bT" } else { "a@b" },
            size_class: if m <= 256 { "m<=256" } else { "m>256" },
        };
        self.run_labeled(
            ctx,
            &kernel,
            label,
            &[
                (0, a.buffer()),
                (1, b.buffer()),
                (2, out.buffer()),
                (3, &params),
                (4, coop.overflows.buffer()),
            ],
            {
                let (gx, gy) = tile.groups(m, n);
                (gx, gy, 1)
            },
        );
        true
    }
}

/// The 16x16 fragments of one K substep `kk`: loads of `a` and `b` from
/// workgroup memory, then every multiply-add of the subgroup's block.
fn fragment_step(key: CoopKey, kk: u32) -> String {
    let (ti, tj) = (key.tile.wm / 16, key.tile.wn / 16);
    let mut s = String::from("    {\n");
    for i in 0..ti {
        s.push_str(&format!(
            "      let fa{i} = coopLoadT<coop_mat16x16<f16, A>>(&sa[(wr + {}u) * AP + {}u], AP);\n",
            i * 16,
            kk * 16
        ));
    }
    for j in 0..tj {
        // b stored [N, K] is B column-major with stride AP; [K, N] is B
        // row-major with stride BP.
        let line = if key.transposed_b {
            format!(
                "      let fb{j} = coopLoad<coop_mat16x16<f16, B>>(&sb[(wc + {}u) * AP + {}u], AP);",
                j * 16,
                kk * 16
            )
        } else {
            format!(
                "      let fb{j} = coopLoadT<coop_mat16x16<f16, B>>(&sb[{}u * BP + wc + {}u], BP);",
                kk * 16,
                j * 16
            )
        };
        s.push_str(&line);
        s.push('\n');
    }
    for i in 0..ti {
        for j in 0..tj {
            s.push_str(&format!(
                "      acc{i}_{j} = coopMultiplyAdd(fa{i}, fb{j}, acc{i}_{j});\n"
            ));
        }
    }
    s.push_str("    }\n");
    s
}

/// Staging `a[m0.., k0..]` into `sa` as f16, flagging values f16 cannot hold.
fn load_a(key: CoopKey) -> String {
    let nt = key.tile.threads();
    if key.a_vec4 {
        let count = key.tile.bm * (BK / 4) / nt;
        format!(
            "    for (var i = 0u; i < {count}u; i++) {{
      let idx = li + i * {nt}u;
      let r = idx / {q}u;
      let q = idx % {q}u;
      let gr = m0 + r;
      let gk = k0 + q * 4u;
      var v = vec4<f32>(0.0);
      if (gr < m && gk < k) {{ v = a[(gr * k + gk) / 4u]; }}
      if (any(!(abs(v) <= vec4<f32>(F16_MAX)))) {{ overflow = true; }}
      let h = vec4<f16>(clamp(v, vec4<f32>(-F16_MAX), vec4<f32>(F16_MAX)));
      let o = r * AP + q * 4u;
      sa[o] = h.x;
      sa[o + 1u] = h.y;
      sa[o + 2u] = h.z;
      sa[o + 3u] = h.w;
    }}
",
            q = BK / 4
        )
    } else {
        let count = key.tile.bm * BK / nt;
        format!(
            "    for (var i = 0u; i < {count}u; i++) {{
      let idx = li + i * {nt}u;
      let r = idx / BK;
      let q = idx % BK;
      let gr = m0 + r;
      let gk = k0 + q;
      var v = 0.0;
      if (gr < m && gk < k) {{ v = a[gr * k + gk]; }}
      if (!(abs(v) <= F16_MAX)) {{ overflow = true; }}
      sa[r * AP + q] = f16(clamp(v, -F16_MAX, F16_MAX));
    }}
"
        )
    }
}

/// Staging the f16 `b` tile into `sb`: `[k][n]` rows of BP for `b[K, N]`,
/// `[n][k]` rows of AP for `b[N, K]`.
fn load_b(key: CoopKey) -> String {
    let nt = key.tile.threads();
    let bn = key.tile.bn;
    if key.b_vec8 {
        // (rows of the tile, 8-half chunks per row, global row, global
        // column of the chunk, word index, workgroup offset)
        let (rows_chunks, body) = if key.transposed_b {
            (
                (bn, BK / 8),
                "      let gn = n0 + r;\n      let gk = k0 + q * 8u;\n      if (gn < n && gk < k) { w = b[(gn * k + gk) / 8u]; }\n      let o = r * AP + q * 8u;\n",
            )
        } else {
            (
                (BK, bn / 8),
                "      let gk = k0 + r;\n      let gn = n0 + q * 8u;\n      if (gk < k && gn < n) { w = b[(gk * n + gn) / 8u]; }\n      let o = r * BP + q * 8u;\n",
            )
        };
        let (rows, chunks) = rows_chunks;
        let count = rows * chunks / nt;
        format!(
            "    for (var i = 0u; i < {count}u; i++) {{
      let idx = li + i * {nt}u;
      let r = idx / {chunks}u;
      let q = idx % {chunks}u;
      var w = vec4<u32>(0u);
{body}      let h0 = vec2<f16>(unpack2x16float(w.x));
      let h1 = vec2<f16>(unpack2x16float(w.y));
      let h2 = vec2<f16>(unpack2x16float(w.z));
      let h3 = vec2<f16>(unpack2x16float(w.w));
      sb[o] = h0.x;
      sb[o + 1u] = h0.y;
      sb[o + 2u] = h1.x;
      sb[o + 3u] = h1.y;
      sb[o + 4u] = h2.x;
      sb[o + 5u] = h2.y;
      sb[o + 6u] = h3.x;
      sb[o + 7u] = h3.y;
    }}
"
        )
    } else {
        let count = BK * bn / nt;
        let body = if key.transposed_b {
            "      let r = idx / BK;\n      let q = idx % BK;\n      let gn = n0 + r;\n      let gk = k0 + q;\n      let e = gn * k + gk;\n      let o = r * AP + q;\n"
        } else {
            "      let r = idx / BN;\n      let q = idx % BN;\n      let gk = k0 + r;\n      let gn = n0 + q;\n      let e = gk * n + gn;\n      let o = r * BP + q;\n"
        };
        format!(
            "    for (var i = 0u; i < {count}u; i++) {{
      let idx = li + i * {nt}u;
{body}      var v = 0.0;
      if (gk < k && gn < n) {{ v = unpack2x16float(b[e >> 1u])[e & 1u]; }}
      sb[o] = f16(v);
    }}
"
        )
    }
}

/// The epilogue of fragment `(i, j)`: through the subgroup's staging
/// fragment to `c`, guarded at the edges, with alpha and beta.
fn store_fragment(i: u32, j: u32) -> String {
    format!(
        "  coopStoreT(acc{i}_{j}, &cs[warp * 256u], 16u);
  workgroupBarrier();
  for (var e = 0u; e < 8u; e++) {{
    let idx = lane + e * 32u;
    let r = m0 + wr + {ri}u + idx / 16u;
    let col = n0 + wc + {cj}u + idx % 16u;
    if (r < m && col < n) {{
      var v = alpha * cs[warp * 256u + idx];
      if (beta != 0.0) {{ v += beta * c[r * n + col]; }}
      c[r * n + col] = v;
    }}
  }}
  workgroupBarrier();
",
        ri = i * 16,
        cj = j * 16
    )
}

/// The WGSL of the kernel for `key`, entry point `matmul_coop`. Params are
/// `[M, K, N, 0]` and `[alpha bits, beta bits, 0, 0]`.
pub(super) fn source(key: CoopKey) -> String {
    let t = key.tile;
    let (ti, tj) = (t.wm / 16, t.wn / 16);
    let warps = t.warps();
    let sb_len = if key.transposed_b {
        t.bn * (BK + PAD)
    } else {
        BK * (t.bn + PAD)
    };
    let mut init = String::new();
    let mut store = String::new();
    for i in 0..ti {
        for j in 0..tj {
            init.push_str(&format!(
                "  var acc{i}_{j} = coopLoadT<coop_mat16x16<f32, C>>(&cs[0], 16u);\n"
            ));
            store.push_str(&store_fragment(i, j));
        }
    }
    let steps: String = (0..BK / 16).map(|kk| fragment_step(key, kk)).collect();
    format!(
        "enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<{a_ty}>;
@group(0) @binding(1) var<storage, read> b: array<{b_ty}>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, 2>;
@group(0) @binding(4) var<storage, read_write> overflows: array<atomic<u32>>;

const BM: u32 = {bm}u;
const BN: u32 = {bn}u;
const BK: u32 = {BK}u;
const AP: u32 = {ap}u;
const BP: u32 = {bp}u;
const F16_MAX: f32 = 65504.0;

var<workgroup> sa: array<f16, {sa_len}>;
var<workgroup> sb: array<f16, {sb_len}>;
var<workgroup> cs: array<f32, {cs_len}>;

@compute @workgroup_size({nt})
fn matmul_coop(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
  let m = params[0].x;
  let k = params[0].y;
  let n = params[0].z;
  let alpha = bitcast<f32>(params[1].x);
  let beta = bitcast<f32>(params[1].y);
  let warp = li / 32u;
  let lane = li % 32u;
  let wr = (warp / {warps_n}u) * {wm}u;
  let wc = (warp % {warps_n}u) * {wn}u;
  let m0 = wg.y * BM;
  let n0 = wg.x * BN;
  for (var i = li; i < {cs_len}u; i += {nt}u) {{ cs[i] = 0.0; }}
  workgroupBarrier();
{init}  var overflow = false;
  for (var k0 = 0u; k0 < k; k0 += BK) {{
{load_a}{load_b}    workgroupBarrier();
{steps}    workgroupBarrier();
  }}
{store}  // After the epilogue's writes, so the poison is the last write to that element.
  storageBarrier();
  if (overflow) {{
    atomicAdd(&overflows[0], 1u);
    c[m0 * n + n0] = bitcast<f32>(0x7fc00000u);
  }}
}}
",
        a_ty = if key.a_vec4 { "vec4<f32>" } else { "f32" },
        b_ty = if key.b_vec8 { "vec4<u32>" } else { "u32" },
        bm = t.bm,
        bn = t.bn,
        ap = BK + PAD,
        bp = t.bn + PAD,
        sa_len = t.bm * (BK + PAD),
        cs_len = warps * 256,
        nt = t.threads(),
        warps_n = t.bn / t.wn,
        wm = t.wm,
        wn = t.wn,
        load_a = load_a(key),
        load_b = load_b(key),
    )
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::{source, CoopKey, CoopTile, BK, PAD, TILES};

    fn keys() -> Vec<CoopKey> {
        let mut keys = Vec::new();
        for tile in TILES {
            for transposed_b in [false, true] {
                for a_vec4 in [false, true] {
                    for b_vec8 in [false, true] {
                        keys.push(CoopKey {
                            tile,
                            transposed_b,
                            a_vec4,
                            b_vec8,
                        });
                    }
                }
            }
        }
        keys
    }

    #[test]
    fn every_kernel_validates_with_cooperative_matrices() {
        for key in keys() {
            let src = source(key);
            let module = match naga::front::wgsl::parse_str(&src) {
                Ok(m) => m,
                Err(e) => panic!("{key:?}: WGSL parse error:\n{}", e.emit_to_string(&src)),
            };
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16
                    | naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
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
    fn every_kernel_validates_with_immediates() {
        for key in keys() {
            let src = immediate_source(&source(key));
            assert!(!src.contains("var<uniform>"), "{key:?}");
            let module = match naga::front::wgsl::parse_str(&src) {
                Ok(m) => m,
                Err(e) => panic!("{key:?}: WGSL parse error:\n{}", e.emit_to_string(&src)),
            };
            assert_eq!(
                naga::valid::ImmediateSlots::size_for_module(&module),
                crate::engine::IMMEDIATE_BYTES,
                "{key:?}"
            );
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16
                    | naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32
                    | naga::valid::Capabilities::IMMEDIATES,
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
    fn tiles_split_evenly_over_their_threads() {
        for t in TILES {
            let nt = t.threads();
            assert_eq!(t.bm % t.wm, 0, "{t:?}");
            assert_eq!(t.bn % t.wn, 0, "{t:?}");
            assert_eq!(t.wm % 16, 0, "{t:?}");
            assert_eq!(t.wn % 16, 0, "{t:?}");
            // Every staging loop gives each thread the same count.
            assert_eq!(t.bm * (BK / 4) % nt, 0, "{t:?} a vec4");
            assert_eq!(t.bm * BK % nt, 0, "{t:?} a");
            assert_eq!(BK * t.bn / 8 % nt, 0, "{t:?} b vec8");
            assert_eq!(BK * t.bn % nt, 0, "{t:?} b");
            assert!(t.workgroup_bytes(true) <= 32 * 1024, "{t:?}");
            assert!(t.workgroup_bytes(false) <= 32 * 1024, "{t:?}");
            // Each workgroup array (sa, sb either way, cs) is a multiple of
            // 16 bytes, so the packed arrays all start 16-byte aligned.
            assert_eq!(t.bm * (BK + PAD) * 2 % 16, 0, "{t:?} sa");
            assert_eq!(t.bn * (BK + PAD) * 2 % 16, 0, "{t:?} sb transposed");
            assert_eq!(BK * (t.bn + PAD) * 2 % 16, 0, "{t:?} sb");
            assert_eq!(t.warps() * 256 * 4 % 16, 0, "{t:?} cs");
            // Row strides in bytes keep every fragment 16-byte aligned.
            assert_eq!((BK + PAD) * 2 % 16, 0);
            assert_eq!((t.bn + PAD) * 2 % 16, 0, "{t:?}");
        }
        assert_eq!(CoopTile::new(128, 64, 64, 32).threads(), 128);
    }
}
