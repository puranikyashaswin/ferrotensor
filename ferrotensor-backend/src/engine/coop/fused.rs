//! The cooperative-matrix kernel with a fused loader or epilogue, and the
//! plain reference kernel the fused products are tested against.
//!
//! The fused kernels are the matmul template of `mod.rs` with another
//! [`LoaderKind`] and [`EpilogueKind`] in the key, so they share its tiles,
//! staging, K loop and f16 overflow check. They run with K whole: the
//! partial sums of a split product cannot take a column split or a residual
//! before the reduction adds them.

use super::super::matmul_fused::Resolved;
use super::super::{uniform, ComputeEngine};
use super::{epilogue, grid_fits, loader, plan, split_k, CoopKey, EpilogueKind, LoaderKind};
use crate::kernel_timing::KernelLabel;
use crate::WgpuContext;

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

impl ComputeEngine {
    /// Records the fused product `r` on the cooperative-matrix kernel with the
    /// tile the router picks and K whole.
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
        let Some(coop) = self.coop.as_ref() else {
            return Err("matmul_fused: this device has no f16 cooperative matrices".into());
        };
        let (m, k, n) = r.dims;
        let transposed_b = r.transposed_b;
        let fits = |t| coop.fits(t, transposed_b);
        let Some(config) = plan(r.dims, transposed_b, (coop.target, coop.tuned), fits) else {
            return Err(format!(
                "matmul_fused: no tile fits {m} x {n} on this device"
            ));
        };
        let tile = config.tile;
        if !grid_fits(tile, m, n, 1) {
            return Err(format!(
                "matmul_fused: {m} x {n} is past the dispatch limit of {}",
                tile.name()
            ));
        }
        let (_, kps) = split_k(k, tile.bk, 1);
        let b_row = if transposed_b { k } else { n };
        let key = CoopKey {
            tile,
            transposed_b,
            a_vec4: r.a_vec4,
            b_vec8: b_row.is_multiple_of(8),
            loader: r.loader,
            epilogue: r.epilogue,
        };
        let kernel = coop.kernel(ctx, key);
        let words = r.params_words(kps, params_vec4s(r.loader, r.epilogue));
        let params = uniform(ctx, bytemuck::cast_slice(&words));
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
        self.run_labeled(
            ctx,
            &kernel,
            label,
            &r.bind(&params, Some(coop.overflows.buffer())),
            (gx, gy, 1),
        );
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::super::{immediate_source, source, CoopKey, CoopTile, EpilogueKind, LoaderKind};
    use super::reference_source;

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

    #[test]
    fn fused_kernels_declare_the_bindings_their_loader_and_epilogue_need() {
        let key = |loader, epilogue| CoopKey {
            tile: CoopTile::ALL[0],
            transposed_b: false,
            a_vec4: false,
            b_vec8: true,
            loader,
            epilogue,
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
