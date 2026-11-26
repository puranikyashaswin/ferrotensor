//! The cooperative-matrix kernel with a fused loader or epilogue, and the
//! plain reference kernel the fused products are tested against.
//!
//! The fused kernels are the matmul template of `mod.rs` with another
//! [`LoaderKind`] and [`EpilogueKind`] in the key, so they share its tiles,
//! staging, K loop and f16 overflow check. With K split, the kernel stores
//! each slice's raw partial sums and a reduction pass (`reduce_source`) adds
//! them in order and runs the epilogue once: a column split, a residual or an
//! accumulate on a partial sum would be wrong.

use super::super::matmul_fused::Resolved;
use super::super::{grid, immediate_params, uniform, ComputeEngine, Kernel};
use super::plan::plan_fused;
use super::{
    epilogue, grid_fits, grouped_raster, loader, split_k, Coop, CoopConfig, CoopKey, EpilogueKind,
    LoaderKind,
};
use crate::kernel_timing::KernelLabel;
use crate::{UnifiedTensor, WgpuContext};
use std::sync::{Arc, PoisonError};
use std::time::Instant;

/// Vec4s of `params` a kernel for `loader` and `epilogue` declares: the sizes
/// and the scalars, then what the two keep there.
pub(in crate::engine) fn params_vec4s(loader: LoaderKind, epilogue: EpilogueKind) -> u32 {
    loader::params_vec4s(loader).max(epilogue::params_vec4s(epilogue))
}

/// The WGSL of the reference kernel, entry point `fused_ref`: one thread per
/// output element sums `a_load(r, p)` rounded to f16 times `b` over K in
/// order, and `ep_store` finishes the element. The bindings, `a_load` and
/// `ep_store` are the fused kernel's own, so the two cannot differ in what
/// they read and write.
pub(in crate::engine) fn reference_source(
    loader: LoaderKind,
    epilogue: EpilogueKind,
    transposed_b: bool,
) -> String {
    let b_index = if transposed_b {
        "col * k + p"
    } else {
        "p * n + col"
    };
    format!(
        "@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, {params_len}>;
{loader_bindings}{epilogue_bindings}
{loader_fn}{epilogue_fn}
fn round_f16(v: f32) -> f32 {{
  return unpack2x16float(pack2x16float(vec2<f32>(clamp(v, -65504.0, 65504.0), 0.0))).x;
}}

@compute @workgroup_size(64)
fn fused_ref(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let m = params[0].x;
  let k = params[0].y;
  let n = params[0].z;
  let idx = gid.x + gid.y * nwg.x * 64u;
  if (idx >= m * n) {{ return; }}
  let r = idx / n;
  let col = idx % n;
  var acc = 0.0;
  for (var p = 0u; p < k; p++) {{
    let e = {b_index};
    acc += round_f16(a_load(r, p)) * unpack2x16float(b[e >> 1u])[e & 1u];
  }}
  ep_store(r, col, acc);
}}
",
        params_len = params_vec4s(loader, epilogue),
        loader_bindings = loader::bindings(loader, false),
        epilogue_bindings = epilogue::bindings(epilogue),
        loader_fn = loader::element_fn(loader, false),
        epilogue_fn = epilogue::element_fn(epilogue),
    )
}

/// The WGSL of the reduction pass of a split fused product, entry point
/// `fused_reduce`: one thread per output element adds the `params[0].w`
/// slices' partial sums in order and finishes the element through the
/// epilogue's own `ep_store`, so the epilogue runs once, on the whole sum.
/// The partial sums are `parts[z * M * N ..]`, binding 0; the output and the
/// epilogue's buffers are bound as in the fused kernel.
pub(in crate::engine) fn reduce_source(epilogue: EpilogueKind) -> String {
    format!(
        "@group(0) @binding(0) var<storage, read> parts: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: array<vec4<u32>, {params_len}>;
{epilogue_bindings}
{epilogue_fn}
@compute @workgroup_size(256)
fn fused_reduce(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {{
  let n = params[0].z;
  let len = params[0].x * n;
  let idx = gid.x + gid.y * nwg.x * 256u;
  if (idx >= len) {{ return; }}
  var sum = parts[idx];
  for (var s = 1u; s < params[0].w; s++) {{
    sum += parts[s * len + idx];
  }}
  ep_store(idx / n, idx % n, sum);
}}
",
        params_len = epilogue::params_vec4s(epilogue),
        epilogue_bindings = epilogue::bindings(epilogue),
        epilogue_fn = epilogue::element_fn(epilogue),
    )
}

impl Coop {
    /// The reduction kernel for `epilogue`, compiled the first time it runs.
    fn fused_reduce_kernel(&self, ctx: &WgpuContext, epilogue: EpilogueKind) -> Arc<Kernel> {
        let mut cache = self
            .fused_reduce
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        Arc::clone(cache.entry(epilogue).or_insert_with(|| {
            let mut src = reduce_source(epilogue);
            if self.immediates {
                src = immediate_params(&src);
            }
            let module = ctx
                .shader_checks()
                .module(&ctx.device, "matmul_fused_reduce", src);
            Arc::new(Kernel::new(ctx, &module, "fused_reduce"))
        }))
    }
}

impl ComputeEngine {
    /// Records the fused product `r` on the cooperative-matrix kernel with the
    /// configuration the fused router picks, or the autotuner's.
    ///
    /// # Errors
    ///
    /// When the device has no cooperative matrices, no tile fits the product
    /// or its grid is past the dispatch limit; nothing is recorded then.
    pub(in crate::engine) fn matmul_fused_coop(
        &self,
        ctx: &WgpuContext,
        r: &Resolved<'_>,
    ) -> Result<(), String> {
        self.matmul_fused_coop_with(ctx, r, None).map(|_| ())
    }

    /// Records the fused product `r` with `config`, or with the configuration
    /// the router and the autotuner pick when it is `None`, and returns the
    /// configuration that ran.
    ///
    /// # Errors
    ///
    /// As [`matmul_fused_coop`](Self::matmul_fused_coop), and when `config`'s
    /// tile does not fit this device or its grid is past the dispatch limit.
    pub(in crate::engine) fn matmul_fused_coop_with(
        &self,
        ctx: &WgpuContext,
        r: &Resolved<'_>,
        config: Option<CoopConfig>,
    ) -> Result<CoopConfig, String> {
        let Some(coop) = self.coop.as_ref() else {
            return Err("matmul_fused: this device has no f16 cooperative matrices".into());
        };
        let (m, _, n) = r.dims;
        let transposed_b = r.transposed_b;
        let fits = |t| coop.fits(t, transposed_b);
        let (config, tune) = match config {
            Some(c) if fits(c.tile) && grid_fits(c.tile, m, n, c.split) => (c, None),
            Some(c) => {
                return Err(format!(
                    "matmul_fused: {} with split {} does not fit {m} x {n} on this device",
                    c.tile.name(),
                    c.split
                ));
            }
            None => {
                let Some(planned) =
                    plan_fused(r.dims, transposed_b, (coop.target, coop.tuned), fits)
                else {
                    return Err(format!(
                        "matmul_fused: no tile fits {m} x {n} on this device"
                    ));
                };
                coop.autotuned(ctx, transposed_b, (r.loader, r.epilogue), r.dims, planned)
            }
        };
        self.fused_coop_run(ctx, coop, r, (config, tune))
    }

    /// Records `config` for the fused product `r`: one pass that runs the
    /// epilogue, or with K split a pass of plain partial sums (every loader
    /// still reads its own slice) and a reduction that adds them in order and
    /// runs the epilogue once. The dispatches carry the autotuner's sample id
    /// `tune` when the run is a timing.
    fn fused_coop_run(
        &self,
        ctx: &WgpuContext,
        coop: &Coop,
        r: &Resolved<'_>,
        (config, tune): (CoopConfig, Option<u64>),
    ) -> Result<CoopConfig, String> {
        let (m, k, n) = r.dims;
        let transposed_b = r.transposed_b;
        let tile = config.tile;
        let (split, kps) = split_k(k, tile.bk, config.split);
        if !grid_fits(tile, m, n, split) {
            return Err(format!(
                "matmul_fused: {m} x {n} is past the dispatch limit of {}",
                tile.name()
            ));
        }
        let b_row = if transposed_b { k } else { n };
        let key = |epilogue| CoopKey {
            tile,
            transposed_b,
            a_vec4: r.a_vec4,
            b_vec8: b_row.is_multiple_of(8),
            loader: r.loader,
            epilogue,
            grouped: grouped_raster(r.loader, transposed_b, tile, n),
        };
        let label = KernelLabel {
            entry: "matmul_coop",
            layout: if transposed_b {
                "fused a@bT"
            } else {
                "fused a@b"
            },
            size_class: if m <= 256 { "m<=256" } else { "m>256" },
        };
        let (gx, gy) = tile.groups(m, n);
        if split == 1 {
            let kernel = self.fused_kernel(ctx, coop, key(r.epilogue), (m, tune));
            let words = r.params_words(kps, params_vec4s(r.loader, r.epilogue));
            let params = uniform(ctx, bytemuck::cast_slice(&words));
            self.run_tagged(
                ctx,
                &kernel,
                label,
                &r.bind(&params, Some(coop.overflows.buffer())),
                (gx, gy, 1),
                tune,
            );
            return Ok(CoopConfig { tile, split });
        }
        // The kernel stores each slice's raw sum (alpha 1, beta 0, the plain
        // epilogue) and the reduction applies the product's own epilogue.
        let kernel = self.fused_kernel(ctx, coop, key(EpilogueKind::AlphaBeta), (m, tune));
        let words = r.params_words_with(
            kps,
            params_vec4s(r.loader, EpilogueKind::AlphaBeta),
            (1.0, 0.0),
        );
        let params = uniform(ctx, bytemuck::cast_slice(&words));
        let parts = UnifiedTensor::empty(ctx, &[split as usize, m as usize, n as usize]);
        self.run_tagged(
            ctx,
            &kernel,
            label,
            &r.bind_partial(&params, coop.overflows.buffer(), parts.buffer()),
            (gx, gy, split),
            tune,
        );
        let reduce = coop.fused_reduce_kernel(ctx, r.epilogue);
        let words = r.params_words(split, epilogue::params_vec4s(r.epilogue));
        let params = uniform(ctx, bytemuck::cast_slice(&words));
        let label = KernelLabel {
            entry: "fused_reduce",
            layout: "",
            size_class: "",
        };
        self.run_tagged(
            ctx,
            &reduce,
            label,
            &r.bind_reduce(&params, parts.buffer()),
            grid((m * n).div_ceil(256)),
            tune,
        );
        Ok(CoopConfig { tile, split })
    }

    /// The compiled kernel for `key` and a product of `rows` rows; with a
    /// `tune` sample its compile time is charged to the autotuner, as a
    /// trial's cost.
    fn fused_kernel(
        &self,
        ctx: &WgpuContext,
        coop: &Coop,
        key: CoopKey,
        (rows, tune): (u32, Option<u64>),
    ) -> Arc<Kernel> {
        let compiling = Instant::now();
        let kernel = coop.kernel(ctx, key, rows);
        if tune.is_some() {
            if let Some(mut tuner) = coop.tuner() {
                tuner.charge(compiling.elapsed());
            }
        }
        kernel
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::super::{immediate_source, source, CoopKey, CoopTile, EpilogueKind, LoaderKind};
    use super::{reduce_source, reference_source};

    /// Every loader and epilogue, including each count of sources and
    /// segments and both bias settings.
    fn kinds() -> Vec<(LoaderKind, EpilogueKind)> {
        let loaders = [
            LoaderKind::Plain,
            LoaderKind::Sources { count: 2 },
            LoaderKind::Sources { count: 3 },
            LoaderKind::Sources { count: 4 },
            LoaderKind::SwiGlu,
        ];
        let mut epilogues = vec![
            EpilogueKind::AlphaBeta,
            EpilogueKind::Residual,
            EpilogueKind::SwiGluBackward,
        ];
        for segments in 1..=4 {
            for bias in [false, true] {
                epilogues.push(EpilogueKind::ColumnSplit { segments, bias });
            }
        }
        let mut kinds = Vec::new();
        for l in loaders {
            for e in &epilogues {
                kinds.push((l, *e));
            }
        }
        kinds
    }

    /// A spread of tiles: single and double buffered, K 32 and 64.
    fn tiles() -> Vec<CoopTile> {
        CoopTile::ALL.iter().copied().step_by(5).collect()
    }

    fn keys() -> Vec<CoopKey> {
        let mut keys = Vec::new();
        for (loader, epilogue) in kinds() {
            for tile in tiles() {
                for transposed_b in [false, true] {
                    for a_vec4 in [false, true] {
                        keys.push(CoopKey {
                            tile,
                            transposed_b,
                            a_vec4,
                            b_vec8: true,
                            loader,
                            epilogue,
                            grouped: false,
                        });
                    }
                }
            }
        }
        keys
    }

    fn validate(label: &str, src: &str, immediates: bool) {
        let module = match naga::front::wgsl::parse_str(src) {
            Ok(m) => m,
            Err(e) => panic!("{label}: WGSL parse error:\n{}", e.emit_to_string(src)),
        };
        let mut caps = naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32;
        if immediates {
            caps |= naga::valid::Capabilities::IMMEDIATES;
        }
        let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps);
        if let Err(e) = validator.validate(&module) {
            panic!("{label}: WGSL validation error:\n{}", e.emit_to_string(src));
        }
    }

    #[test]
    fn every_fused_kernel_validates() {
        for key in keys() {
            validate(&format!("{key:?}"), &source(key), false);
        }
    }

    #[test]
    fn every_fused_kernel_validates_with_immediates() {
        for key in keys() {
            let src = immediate_source(&source(key));
            assert!(!src.contains("var<uniform>"), "{key:?}");
            validate(&format!("{key:?}"), &src, true);
        }
    }

    #[test]
    fn every_reference_kernel_validates() {
        for (loader, epilogue) in kinds() {
            for transposed_b in [false, true] {
                let src = reference_source(loader, epilogue, transposed_b);
                validate(
                    &format!("{loader:?} {epilogue:?} {transposed_b}"),
                    &src,
                    false,
                );
                let src = crate::engine::immediate_params(&src);
                validate(
                    &format!("{loader:?} {epilogue:?} {transposed_b}"),
                    &src,
                    true,
                );
            }
        }
    }

    /// Every epilogue the fused kernels have, as the key lists them.
    fn epilogues() -> Vec<EpilogueKind> {
        kinds()
            .into_iter()
            .map(|(_, e)| e)
            .fold(Vec::new(), |mut v, e| {
                if !v.contains(&e) {
                    v.push(e);
                }
                v
            })
    }

    #[test]
    fn every_reduce_kernel_validates() {
        assert_eq!(epilogues().len(), 11);
        for epilogue in epilogues() {
            let src = reduce_source(epilogue);
            validate(&format!("{epilogue:?}"), &src, false);
            let src = crate::engine::immediate_params(&src);
            validate(&format!("{epilogue:?}"), &src, true);
        }
    }

    #[test]
    fn the_reduce_kernel_sums_the_slices_before_the_epilogue() {
        let src = reduce_source(EpilogueKind::Residual);
        assert!(src.contains("sum += parts[s * len + idx];"));
        assert!(src.contains("res: array<f32>"));
        // The sum is complete before `ep_store` sees it, and nothing else
        // writes `c`.
        let sum = src.find("sum += parts").expect("the sum loop");
        let store = src
            .find("  ep_store(idx / n, idx % n, sum);")
            .expect("the store");
        assert!(sum < store);
        assert_eq!(src.matches("ep_store(").count(), 2);
    }

    #[test]
    fn fused_kernels_declare_the_bindings_their_loader_and_epilogue_need() {
        let key = |loader, epilogue| CoopKey {
            tile: CoopTile::ALL[0],
            transposed_b: false,
            a_vec4: false,
            b_vec8: true,
            loader,
            epilogue,
            grouped: false,
        };
        let src = source(key(
            LoaderKind::Sources { count: 4 },
            EpilogueKind::ColumnSplit {
                segments: 4,
                bias: true,
            },
        ));
        for name in ["a1", "a2", "a3", "c1", "c2", "c3", "bias"] {
            assert!(
                src.contains(&format!(" {name}: array<f32>;")),
                "{name} is not declared"
            );
        }
        assert!(src.contains("array<vec4<u32>, 4>"));
        let plain = source(key(LoaderKind::Plain, EpilogueKind::AlphaBeta));
        assert!(plain.contains("array<vec4<u32>, 2>"));
        assert!(!plain.contains("ep_store") && !plain.contains("a_load"));
    }
}
