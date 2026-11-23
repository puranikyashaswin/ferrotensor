//! The cooperative-matrix matmul and flash attention against CPU references.
//!
//! These need a GPU with f16 cooperative matrices (16x16x16, f32
//! accumulation) and subgroups of 32 threads, and `FERROTENSOR_EXPERIMENTAL=1`
//! so the device asks for them; otherwise each test prints why it skipped and
//! passes. Run them with:
//!
//! ```text
//! FERROTENSOR_EXPERIMENTAL=1 cargo test -p ferrotensor-backend --test gpu_coop -- --ignored --test-threads=1
//! ```

#![allow(clippy::unwrap_used, clippy::panic)]

mod common;

use common::attention::{cpu_attention, cpu_attention_backward, cpu_attention_lse};
use ferrotensor_backend::{
    AttentionOptions, ComputeEngine, CoopConfig, CoopTile, Epilogue, MatmulPath, UnifiedTensor,
    WgpuContext,
};
use std::sync::OnceLock;

struct Gpu {
    ctx: WgpuContext,
    engine: ComputeEngine,
}

fn gpu() -> &'static Gpu {
    static GPU: OnceLock<Gpu> = OnceLock::new();
    GPU.get_or_init(|| {
        let ctx = pollster::block_on(WgpuContext::new());
        let engine = ComputeEngine::new(&ctx);
        Gpu { ctx, engine }
    })
}

/// The shared GPU when it has the kernel; prints why the test skips if not.
fn coop_gpu(test: &str) -> Option<&'static Gpu> {
    let g = gpu();
    if g.engine.coop_matmul_available() {
        Some(g)
    } else {
        eprintln!(
            "{test}: skipped, {} has no f16 cooperative matrices with f32 accumulation \
             (or FERROTENSOR_EXPERIMENTAL is not 1)",
            g.ctx.adapter_info()
        );
        None
    }
}

/// Deterministic pseudo-random values in [-1, 1).
fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn f16_round(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect()
}

/// `a[M, K] @ b[K, N]` in f64.
fn cpu_matmul(a: &[f32], b: &[f32], (m, k, n): (usize, usize, usize)) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = f64::from(a[i * k + p]);
            for j in 0..n {
                out[i * n + j] += av * f64::from(b[p * n + j]);
            }
        }
    }
    out
}

fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0; x.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = x[r * cols + c];
        }
    }
    t
}

/// Largest |got - want| over the largest |want|.
fn max_rel_error(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let mut max_err = 0.0f64;
    let mut max_ref = 0.0f64;
    for (&g, &w) in got.iter().zip(want) {
        assert!(g.is_finite(), "non-finite output {g}");
        max_err = max_err.max((f64::from(g) - w).abs());
        max_ref = max_ref.max(w.abs());
    }
    max_err / max_ref.max(f64::MIN_POSITIVE)
}

/// Runs the coop kernel on `a[M, K]` and `b[K, N]` (uploaded as `[N, K]`
/// when `transposed_b`) with `ep` over `out0`, and returns the output.
fn run_coop(
    g: &Gpu,
    dims: (usize, usize, usize),
    a: &[f32],
    b: &[f32],
    transposed_b: bool,
    epilogue: (Epilogue, &[f32]),
) -> Vec<f32> {
    run_coop_config(g, dims, a, b, transposed_b, epilogue, None)
        .unwrap()
        .0
}

/// [`run_coop`] with `config` (`None`: the router's choice); also returns
/// the configuration that ran, or the error when it does not fit.
fn run_coop_config(
    g: &Gpu,
    (m, k, n): (usize, usize, usize),
    a: &[f32],
    b: &[f32],
    transposed_b: bool,
    (ep, out0): (Epilogue, &[f32]),
    config: Option<CoopConfig>,
) -> Result<(Vec<f32>, CoopConfig), String> {
    let a_t = UnifiedTensor::new(&g.ctx, a, &[m, k]);
    let b_t = if transposed_b {
        UnifiedTensor::from_f32_as_f16(&g.ctx, &transpose(b, k, n), &[n, k])
    } else {
        UnifiedTensor::from_f32_as_f16(&g.ctx, b, &[k, n])
    };
    let out = UnifiedTensor::new(&g.ctx, out0, &[m, n]);
    let (engine, tensors) = (&g.engine, [&a_t, &b_t, &out]);
    let ran = engine.matmul_coop_config_into(&g.ctx, transposed_b, tensors, config, ep)?;
    Ok((pollster::block_on(out.to_vec(&g.ctx)).unwrap(), ran))
}

/// Every tile, the register-prefetch ones included, with K whole and split
/// in 3 (fewer when K has fewer steps).
fn configs() -> Vec<CoopConfig> {
    CoopTile::ALL
        .iter()
        .chain(&CoopTile::PIPELINED)
        .flat_map(|&tile| [1, 3].map(|split| CoopConfig { tile, split }))
        .collect()
}

/// Runs `config`; `None` (after saying so) when its tile does not fit this
/// device, which is no failure: the router never picks such a tile.
fn run_or_skip(
    g: &Gpu,
    dims: (usize, usize, usize),
    (a, b): (&[f32], &[f32]),
    transposed_b: bool,
    epilogue: (Epilogue, &[f32]),
    config: CoopConfig,
) -> Option<Vec<f32>> {
    match run_coop_config(g, dims, a, b, transposed_b, epilogue, Some(config)) {
        Ok((got, ran)) => {
            assert_eq!(ran.tile, config.tile);
            assert!(ran.split >= 1 && ran.split <= config.split, "{ran:?}");
            Some(got)
        }
        Err(e) if e.contains("does not fit") => {
            eprintln!(
                "{} split {}: skipped, {e}",
                config.tile.name(),
                config.split
            );
            None
        }
        Err(e) => panic!("{} split {}: {e}", config.tile.name(), config.split),
    }
}

/// The 135M training shapes at batch 1 x 256 tokens, then shapes whose M,
/// N and K are not multiples of 16 (and K not of 4, N not of 8, which take
/// the scalar staging loads).
const SHAPES: &[(usize, usize, usize)] = &[
    (256, 576, 576),
    (256, 576, 1536),
    (256, 1536, 576),
    (512, 576, 192),
    (64, 16, 16),
    (77, 50, 33),
    (100, 37, 70),
    (130, 576, 200),
    (65, 1000, 17),
];

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn coop_matmul_matches_cpu_on_f16_rounded_inputs() {
    let Some(g) = coop_gpu("coop_matmul_matches_cpu_on_f16_rounded_inputs") else {
        return;
    };
    for (seed, &(m, k, n)) in SHAPES.iter().enumerate() {
        let a = rand_vec(m * k, seed as u64);
        let b = rand_vec(k * n, 100 + seed as u64);
        let (a16, b16) = (f16_round(&a), f16_round(&b));
        let want16 = cpu_matmul(&a16, &b16, (m, k, n));
        let want32 = cpu_matmul(&a, &b, (m, k, n));
        for transposed_b in [false, true] {
            let got = run_coop(
                g,
                (m, k, n),
                &a,
                &b,
                transposed_b,
                (Epilogue::STORE, &vec![0.0; m * n]),
            );
            let e16 = max_rel_error(&got, &want16);
            let e32 = max_rel_error(&got, &want32);
            eprintln!(
                "coop {m}x{k}x{n} transposed_b {transposed_b}: max rel err {e16:.2e} against f16-rounded inputs, {e32:.2e} against f32 inputs"
            );
            // f16 products are exact in f32; what is left is f32 summation.
            assert!(
                e16 < 2e-5,
                "{m}x{k}x{n} transposed_b {transposed_b}: {e16:e}"
            );
            // Rounding a to f16 costs about 2^-11 relative per element.
            assert!(
                e32 < 3e-3,
                "{m}x{k}x{n} transposed_b {transposed_b}: {e32:e}"
            );
        }
    }
}

/// Shapes of 1000 and 1352 rows, neither a multiple of any tile height, so the
/// last row of tiles is partial. They are the large-M products (256 rows or
/// more) that run in the kernel built without bounds checks while
/// `FERROTENSOR_SHADER_CHECKS` is unset. The last shape has a K that is not a
/// multiple of 4 and an N that is not a multiple of 8, which take the scalar
/// staging loads.
const UNCHECKED_SHAPES: &[(usize, usize, usize)] = &[
    (1000, 576, 1536),
    (1352, 1024, 1160),
    (1352, 2048, 2048),
    (1000, 1002, 71),
];

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn coop_large_m_without_bounds_checks_matches_f64_off_the_tile() {
    let Some(g) = coop_gpu("coop_large_m_without_bounds_checks_matches_f64_off_the_tile") else {
        return;
    };
    let checks = std::env::var("FERROTENSOR_SHADER_CHECKS").unwrap_or_else(|_| "unset".into());
    for (seed, &(m, k, n)) in UNCHECKED_SHAPES.iter().enumerate() {
        let a = rand_vec(m * k, 300 + seed as u64);
        let b = rand_vec(k * n, 400 + seed as u64);
        // The reference is f64 on the f16-rounded operands, so what is left
        // is f32 summation (below 2e-5 of the largest output) and not
        // the 2^-11 rounding of the inputs.
        let want = cpu_matmul(&f16_round(&a), &f16_round(&b), (m, k, n));
        let zeros = vec![0.0; m * n];
        for transposed_b in [false, true] {
            let got = run_coop(
                g,
                (m, k, n),
                &a,
                &b,
                transposed_b,
                (Epilogue::STORE, &zeros),
            );
            let e = max_rel_error(&got, &want);
            eprintln!(
                "coop {m}x{k}x{n} transposed_b {transposed_b} (shader checks {checks}): max rel err {e:.2e}"
            );
            assert!(e < 2e-5, "{m}x{k}x{n} transposed_b {transposed_b}: {e:e}");
        }
    }
    // Every register-prefetch tile, K whole and split in 3, on the first
    // shape: the router picks only some of them, and each is its own kernel.
    let (m, k, n) = UNCHECKED_SHAPES[0];
    let a = rand_vec(m * k, 500);
    let b = rand_vec(k * n, 501);
    let want = cpu_matmul(&f16_round(&a), &f16_round(&b), (m, k, n));
    let zeros = vec![0.0; m * n];
    let mut ran = 0;
    for &tile in &CoopTile::PIPELINED {
        for split in [1, 3] {
            for transposed_b in [false, true] {
                let config = CoopConfig { tile, split };
                let Some(got) = run_or_skip(
                    g,
                    (m, k, n),
                    (&a, &b),
                    transposed_b,
                    (Epilogue::STORE, &zeros),
                    config,
                ) else {
                    continue;
                };
                let e = max_rel_error(&got, &want);
                assert!(
                    e < 2e-5,
                    "{} split {split} transposed_b {transposed_b}: {e:e}",
                    tile.name()
                );
                ran += 1;
            }
        }
    }
    assert!(ran > 0, "no register-prefetch tile fits this device");
}

/// Products of 32 to 63 rows go to the cooperative kernel, not the small-M one.
#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn products_of_32_to_63_rows_take_the_coop_kernel() {
    let Some(g) = coop_gpu("products_of_32_to_63_rows_take_the_coop_kernel") else {
        return;
    };
    if !g.engine.coop_matmul_active() {
        eprintln!(
            "products_of_32_to_63_rows_take_the_coop_kernel: skipped, the kernel is off here"
        );
        return;
    }
    let (k, n) = (576, 1536);
    for (seed, m) in [32usize, 48, 63].into_iter().enumerate() {
        let a = rand_vec(m * k, seed as u64);
        let b = rand_vec(k * n, 100 + seed as u64);
        // The kernel rounds a and b to f16; the small-M kernel keeps a in f32,
        // so only the first meets this bound against f16-rounded inputs.
        let want = cpu_matmul(&f16_round(&a), &f16_round(&b), (m, k, n));
        for transposed_b in [false, true] {
            let a_t = UnifiedTensor::new(&g.ctx, &a, &[m, k]);
            let b_t = if transposed_b {
                UnifiedTensor::from_f32_as_f16(&g.ctx, &transpose(&b, k, n), &[n, k])
            } else {
                UnifiedTensor::from_f32_as_f16(&g.ctx, &b, &[k, n])
            };
            let out = UnifiedTensor::zeros(&g.ctx, &[m, n]);
            let launch = g
                .engine
                .matmul_with_path(&g.ctx, transposed_b, &a_t, &b_t, &out, MatmulPath::Auto)
                .unwrap();
            assert!(launch.is_none(), "{m} rows took the small-M kernel");
            let got = pollster::block_on(out.to_vec(&g.ctx)).unwrap();
            let e = max_rel_error(&got, &want);
            assert!(e < 2e-5, "{m}x{k}x{n} transposed_b {transposed_b}: {e:e}");
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn coop_matmul_applies_alpha_and_beta() {
    let Some(g) = coop_gpu("coop_matmul_applies_alpha_and_beta") else {
        return;
    };
    let (m, k, n) = (96, 80, 72);
    let a = f16_round(&rand_vec(m * k, 7));
    let b = f16_round(&rand_vec(k * n, 8));
    let old = rand_vec(m * n, 9);
    let product = cpu_matmul(&a, &b, (m, k, n));
    for (alpha, beta) in [(1.0f32, 1.0f32), (0.5, 0.0), (-2.0, 0.25), (1.0, 0.0)] {
        let want: Vec<f64> = product
            .iter()
            .zip(&old)
            .map(|(&p, &o)| f64::from(alpha) * p + f64::from(beta) * f64::from(o))
            .collect();
        for transposed_b in [false, true] {
            let got = run_coop(
                g,
                (m, k, n),
                &a,
                &b,
                transposed_b,
                (Epilogue { alpha, beta }, &old),
            );
            let e = max_rel_error(&got, &want);
            assert!(
                e < 2e-5,
                "alpha {alpha} beta {beta} transposed_b {transposed_b}: {e:e}"
            );
        }
    }
    // beta = 0 never reads out: NaN there must not leak into the result.
    let got = run_coop(
        g,
        (m, k, n),
        &a,
        &b,
        false,
        (Epilogue::STORE, &vec![f32::NAN; m * n]),
    );
    assert!(max_rel_error(&got, &product) < 2e-5);
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn coop_matmul_counts_f16_overflow_and_poisons_its_tile() {
    let Some(g) = coop_gpu("coop_matmul_counts_f16_overflow_and_poisons_its_tile") else {
        return;
    };
    // N = 16 is one column of workgroups whatever the tile, so exactly one
    // staging thread loads the value set below.
    let (m, k, n) = (256, 64, 16);
    let before = pollster::block_on(g.engine.coop_f16_overflows(&g.ctx)).unwrap();
    let b = rand_vec(k * n, 11);
    let fine = rand_vec(m * k, 12);
    let got = run_coop(
        g,
        (m, k, n),
        &fine,
        &b,
        false,
        (Epilogue::STORE, &vec![0.0; m * n]),
    );
    assert!(got.iter().all(|v| v.is_finite()));
    assert_eq!(
        pollster::block_on(g.engine.coop_f16_overflows(&g.ctx)).unwrap(),
        before
    );
    // One value past the f16 range in row 3: the one thread that stages it counts it.
    let mut big = fine.clone();
    big[3 * k + 5] = 70_000.0;
    let got = run_coop(
        g,
        (m, k, n),
        &big,
        &b,
        false,
        (Epilogue::STORE, &vec![0.0; m * n]),
    );
    assert_eq!(
        pollster::block_on(g.engine.coop_f16_overflows(&g.ctx)).unwrap(),
        before + 1
    );
    assert!(got[0].is_nan(), "the tile's first element is poisoned");
    assert!(got.iter().any(|v| v.is_nan()));
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn coop_matmul_times_identity_returns_a_unshifted() {
    let Some(g) = coop_gpu("coop_matmul_times_identity_returns_a_unshifted") else {
        return;
    };
    // Integers below 2048 are exact in f16, so `a @ I` must equal `a`
    // element for element; a fragment read or written off its 16-byte
    // alignment shows up as values shifted along the rows.
    let (m, k, n) = (64, 16, 16);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 1000) as f32).collect();
    let mut eye = vec![0.0f32; k * n];
    for i in 0..k {
        eye[i * n + i] = 1.0;
    }
    for transposed_b in [false, true] {
        let got = run_coop(
            g,
            (m, k, n),
            &a,
            &eye,
            transposed_b,
            (Epilogue::STORE, &vec![0.0; m * n]),
        );
        assert_eq!(&got[..n], &a[..n], "transposed_b {transposed_b} row 0");
        assert_eq!(got, a, "transposed_b {transposed_b}");
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn every_coop_tile_and_split_matches_cpu() {
    let Some(g) = coop_gpu("every_coop_tile_and_split_matches_cpu") else {
        return;
    };
    for (seed, &(m, k, n)) in SHAPES.iter().enumerate() {
        let a = f16_round(&rand_vec(m * k, 200 + seed as u64));
        let b = f16_round(&rand_vec(k * n, 300 + seed as u64));
        let want = cpu_matmul(&a, &b, (m, k, n));
        let zeros = vec![0.0; m * n];
        for config in configs() {
            for transposed_b in [false, true] {
                let ep = (Epilogue::STORE, zeros.as_slice());
                let got = run_or_skip(g, (m, k, n), (&a, &b), transposed_b, ep, config);
                let Some(got) = got else {
                    continue;
                };
                // Slices of K are summed in another order than one pass: the
                // bound is the f32 summation error, not equality.
                let e = max_rel_error(&got, &want);
                assert!(
                    e < 2e-5,
                    "{} split {} {m}x{k}x{n} transposed_b {transposed_b}: {e:e}",
                    config.tile.name(),
                    config.split
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn every_coop_tile_and_split_times_identity_returns_a() {
    let Some(g) = coop_gpu("every_coop_tile_and_split_times_identity_returns_a") else {
        return;
    };
    // M not a multiple of 16; K = 160 is 5 steps of 32 or 3 of 64, so with a
    // split of 3 every slice but one adds exact zeros.
    let (m, k, n) = (77, 160, 160);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 1000) as f32).collect();
    let mut eye = vec![0.0f32; k * n];
    for i in 0..k {
        eye[i * n + i] = 1.0;
    }
    let zeros = vec![0.0; m * n];
    for config in configs() {
        for transposed_b in [false, true] {
            let ep = (Epilogue::STORE, zeros.as_slice());
            let got = run_or_skip(g, (m, k, n), (&a, &eye), transposed_b, ep, config);
            let Some(got) = got else {
                continue;
            };
            assert_eq!(
                got,
                a,
                "{} split {} transposed_b {transposed_b}",
                config.tile.name(),
                config.split
            );
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn split_k_applies_alpha_beta_and_keeps_the_overflow_poison() {
    let Some(g) = coop_gpu("split_k_applies_alpha_beta_and_keeps_the_overflow_poison") else {
        return;
    };
    let (m, k, n) = (96, 200, 72);
    let a = f16_round(&rand_vec(m * k, 17));
    let b = f16_round(&rand_vec(k * n, 18));
    let old = rand_vec(m * n, 19);
    let product = cpu_matmul(&a, &b, (m, k, n));
    let split = CoopConfig {
        tile: CoopTile::ALL[4],
        split: 3,
    };
    for (alpha, beta) in [(1.0f32, 1.0f32), (-2.0, 0.25), (0.5, 0.0)] {
        let want: Vec<f64> = product
            .iter()
            .zip(&old)
            .map(|(&p, &o)| f64::from(alpha) * p + f64::from(beta) * f64::from(o))
            .collect();
        for transposed_b in [false, true] {
            let ep = (Epilogue { alpha, beta }, old.as_slice());
            let got = run_or_skip(g, (m, k, n), (&a, &b), transposed_b, ep, split);
            let Some(got) = got else {
                return;
            };
            let e = max_rel_error(&got, &want);
            assert!(
                e < 2e-5,
                "alpha {alpha} beta {beta} transposed_b {transposed_b}: {e:e}"
            );
        }
    }
    // An overflow in the first slice of K poisons that slice's partial sums,
    // and the reduction carries the NaN to the output.
    let before = pollster::block_on(g.engine.coop_f16_overflows(&g.ctx)).unwrap();
    let mut big = a.clone();
    big[5] = 70_000.0;
    let zeros = vec![0.0; m * n];
    let ep = (Epilogue::STORE, zeros.as_slice());
    let got = run_or_skip(g, (m, k, n), (&big, &b), false, ep, split);
    let Some(got) = got else {
        return;
    };
    // Each column workgroup of row block 0 stages a[5] in slice 0, and each counts it once.
    let gx = n.div_ceil(64) as u64;
    assert_eq!(
        pollster::block_on(g.engine.coop_f16_overflows(&g.ctx)).unwrap(),
        before + gx
    );
    assert!(got[0].is_nan(), "the tile's first element is poisoned");
    assert!(
        got[64].is_nan(),
        "the second column workgroup's tile is poisoned too"
    );
    assert!(
        got[64 * n].is_finite(),
        "row block 1 never staged a[5], so its tile stays finite"
    );
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn coop_overflow_count_is_per_column_workgroup() {
    let Some(g) = coop_gpu("coop_overflow_count_is_per_column_workgroup") else {
        return;
    };
    // The counter counts staging threads per workgroup, so one bad value of
    // `a` is counted once for every column workgroup that stages it.
    let (m, k) = (64, 64);
    for tile in [CoopTile::ALL[3], CoopTile::ALL[4]] {
        for n in [16, 64, 72, 200] {
            let a = rand_vec(m * k, 21);
            let b = rand_vec(k * n, 22);
            let mut big = a.clone();
            big[5] = 70_000.0;
            let config = CoopConfig { tile, split: 1 };
            let zeros = vec![0.0; m * n];
            let before = pollster::block_on(g.engine.coop_f16_overflows(&g.ctx)).unwrap();
            let ep = (Epilogue::STORE, zeros.as_slice());
            let Some(got) = run_or_skip(g, (m, k, n), (&big, &b), false, ep, config) else {
                continue;
            };
            let after = pollster::block_on(g.engine.coop_f16_overflows(&g.ctx)).unwrap();
            let columns = n.div_ceil(tile.bn as usize);
            assert_eq!(after - before, columns as u64, "{} n {n}", tile.name());
            for j in 0..columns {
                let first = j * tile.bn as usize;
                assert!(
                    got[first].is_nan(),
                    "{} n {n}: column tile {j} is not poisoned",
                    tile.name()
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn coop_rounds_unrepresentable_a_within_one_ulp() {
    let Some(g) = coop_gpu("coop_rounds_unrepresentable_a_within_one_ulp") else {
        return;
    };
    // The kernel converts `a` to f16 on load, so with f32 `a` the output is
    // that of the rounded values. Each product of `a` in [-1, 1) is off by at
    // most 2^-11 of itself; the bound below allows twice that, so it holds
    // under either neighbour the driver may round to.
    for (m, k, n) in [(64, 64, 64), (77, 50, 33), (100, 37, 70), (130, 200, 72)] {
        let a = rand_vec(m * k, 31);
        let b = f16_round(&rand_vec(k * n, 32));
        let want = cpu_matmul(&a, &b, (m, k, n));
        let abs_a: Vec<f32> = a.iter().map(|v| v.abs()).collect();
        let abs_b: Vec<f32> = b.iter().map(|v| v.abs()).collect();
        let scale = cpu_matmul(&abs_a, &abs_b, (m, k, n));
        let got = run_coop(
            g,
            (m, k, n),
            &a,
            &b,
            false,
            (Epilogue::STORE, &vec![0.0; m * n]),
        );
        for (i, ((&x, &w), &s)) in got.iter().zip(&want).zip(&scale).enumerate() {
            let bound = s / 1024.0 + 1e-6;
            assert!(
                (f64::from(x) - w).abs() <= bound,
                "{m}x{k}x{n} element {i}: got {x}, want {w}, bound {bound}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Flash attention on cooperative matrices
// ---------------------------------------------------------------------------

/// The shared GPU with the attention kernels switched on; prints why the
/// test skips if the device lacks them.
fn flash_gpu(test: &str) -> Option<&'static Gpu> {
    let g = gpu();
    if g.engine.flash_coop_available() {
        g.engine.set_flash_coop(true);
        Some(g)
    } else {
        eprintln!(
            "{test}: skipped, {} has no f16 cooperative matrices with f32 accumulation \
             (or FERROTENSOR_EXPERIMENTAL is not 1)",
            g.ctx.adapter_info()
        );
        None
    }
}

fn upload(g: &Gpu, data: &[f32], shape: &[usize]) -> UnifiedTensor {
    UnifiedTensor::new(&g.ctx, data, shape)
}

fn zeros(g: &Gpu, shape: &[usize]) -> UnifiedTensor {
    UnifiedTensor::zeros(&g.ctx, shape)
}

fn read(g: &Gpu, t: &UnifiedTensor) -> Vec<f32> {
    pollster::block_on(t.to_vec(&g.ctx)).unwrap()
}

fn max_abs(x: &[f32]) -> f32 {
    x.iter().fold(0.0, |m, v| m.max(v.abs()))
}

#[track_caller]
fn assert_near(name: &str, got: &[f32], want: &[f32], (atol, rtol): (f32, f32)) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = atol + rtol * w.abs();
        assert!(
            g.is_finite() && (g - w).abs() <= tol,
            "{name}: index {i}: got {g}, want {w}, tol {tol}"
        );
    }
}

/// An attention call: batch, query and key lengths, heads, KV heads, causal
/// masking and the position of the first query. The head size is 64.
struct FlashCase {
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hkv: usize,
    causal: bool,
    offset: u32,
}

const HEAD: usize = 64;

/// Lengths that are and are not multiples of 64, GQA with groups of 3, 3 and
/// 1, a non-causal call with different query and key lengths, and a causal
/// one that starts at an offset.
const FLASH_CASES: &[FlashCase] = &[
    FlashCase {
        b: 1,
        sq: 64,
        sk: 64,
        h: 1,
        hkv: 1,
        causal: true,
        offset: 0,
    },
    FlashCase {
        b: 1,
        sq: 256,
        sk: 256,
        h: 9,
        hkv: 3,
        causal: true,
        offset: 0,
    },
    FlashCase {
        b: 2,
        sq: 100,
        sk: 100,
        h: 15,
        hkv: 5,
        causal: true,
        offset: 0,
    },
    FlashCase {
        b: 1,
        sq: 130,
        sk: 130,
        h: 32,
        hkv: 32,
        causal: true,
        offset: 0,
    },
    FlashCase {
        b: 1,
        sq: 192,
        sk: 192,
        h: 3,
        hkv: 3,
        causal: false,
        offset: 0,
    },
    FlashCase {
        b: 1,
        sq: 70,
        sk: 150,
        h: 2,
        hkv: 1,
        causal: false,
        offset: 0,
    },
    FlashCase {
        b: 1,
        sq: 17,
        sk: 17,
        h: 2,
        hkv: 2,
        causal: true,
        offset: 0,
    },
    FlashCase {
        b: 1,
        sq: 50,
        sk: 80,
        h: 4,
        hkv: 2,
        causal: true,
        offset: 30,
    },
];

impl FlashCase {
    fn dims(&self) -> (usize, usize, usize, usize, usize, usize) {
        (self.b, self.sq, self.sk, self.h, self.hkv, HEAD)
    }

    fn visible(&self) -> impl Fn(usize, usize) -> bool {
        let (causal, offset) = (self.causal, self.offset as usize);
        move |i, j| !causal || j <= i + offset
    }

    fn options(&self) -> AttentionOptions {
        AttentionOptions {
            causal: self.causal,
            causal_offset: self.offset,
            ..AttentionOptions::default()
        }
    }

    fn name(&self) -> String {
        format!(
            "b{} q{} k{} h{}/{} causal={} offset={}",
            self.b, self.sq, self.sk, self.h, self.hkv, self.causal, self.offset
        )
    }
}

/// f16-representable inputs for `case`: q, k, v and dO.
fn flash_inputs(case: &FlashCase, seed: u64) -> [Vec<f32>; 4] {
    let q = case.b * case.sq * case.h * HEAD;
    let kv = case.b * case.sk * case.hkv * HEAD;
    [
        f16_round(&rand_vec(q, seed)),
        f16_round(&rand_vec(kv, seed + 1)),
        f16_round(&rand_vec(kv, seed + 2)),
        f16_round(&rand_vec(q, seed + 3)),
    ]
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn coop_flash_attention_matches_cpu_on_f16_rounded_inputs() {
    let Some(g) = flash_gpu("coop_flash_attention_matches_cpu_on_f16_rounded_inputs") else {
        return;
    };
    for (n, case) in FLASH_CASES.iter().enumerate() {
        let name = case.name();
        let (b, sq, sk, h, hkv) = (case.b, case.sq, case.sk, case.h, case.hkv);
        let dims = case.dims();
        let [q, k, v, dout] = flash_inputs(case, 200 + 4 * n as u64);
        // The reference sees the same f16-rounded values the kernel stages.
        let (want, probs) = cpu_attention(&q, &k, &v, None, dims, case.visible());
        let (want_dq, want_dk, want_dv) = cpu_attention_backward(&q, &k, &v, &dout, &probs, dims);
        let want_lse = cpu_attention_lse(&q, &k, None, dims, case.visible());

        let (tq, tk, tv, tdo) = (
            upload(g, &q, &[b, sq, h, HEAD]),
            upload(g, &k, &[b, sk, hkv, HEAD]),
            upload(g, &v, &[b, sk, hkv, HEAD]),
            upload(g, &dout, &[b, sq, h, HEAD]),
        );
        let opts = case.options();
        let out = zeros(g, &[b, sq, h, HEAD]);
        let stats = zeros(g, &[b, h, sq, 2]);
        let runs = g.engine.flash_coop_runs();
        g.engine
            .flash_attention_v2_with_stats(&g.ctx, &tq, &tk, &tv, &out, &stats, &opts, None);
        assert_eq!(g.engine.flash_coop_runs().0, runs.0 + 1, "{name}: routed");
        let got = read(g, &out);
        // P is rounded to f16 before the PV product: 2^-11 relative per weight.
        assert_near(&format!("fwd {name}"), &got, &want, (2e-3, 1e-2));
        let lse: Vec<f32> = read(g, &stats).iter().skip(1).step_by(2).copied().collect();
        assert_near(&format!("lse {name}"), &lse, &want_lse, (5e-4, 1e-4));

        let (gq, gk, gv) = (
            zeros(g, &[b, sq, h, HEAD]),
            zeros(g, &[b, sk, hkv, HEAD]),
            zeros(g, &[b, sk, hkv, HEAD]),
        );
        g.engine.flash_attention_v2_backward_with_stats(
            &g.ctx, &tq, &tk, &tv, &out, &tdo, &gq, &gk, &gv, &stats, &opts, None,
        );
        assert_eq!(
            g.engine.flash_coop_runs().1,
            runs.1 + 1,
            "{name}: bwd routed"
        );
        for (label, got, want) in [
            ("dq", read(g, &gq), want_dq),
            ("dk", read(g, &gk), want_dk),
            ("dv", read(g, &gv), want_dv),
        ] {
            // P and dS are f16 in the products, so each term is off by about
            // 2^-11 of itself; one percent of the largest gradient bounds the sums.
            let atol = 0.01 * max_abs(&want);
            assert_near(&format!("{label} {name}"), &got, &want, (atol, 2e-2));
        }
        // Delta = dO . O, from the kernel's own f32 output.
        let delta = read(g, &stats);
        for row in 0..b * h * sq {
            let (bi, rest) = (row / (h * sq), row % (h * sq));
            let (hi, i) = (rest / sq, rest % sq);
            let at = ((bi * sq + i) * h + hi) * HEAD;
            let want: f64 = (0..HEAD)
                .map(|x| f64::from(dout[at + x]) * f64::from(got[at + x]))
                .sum();
            let d = f64::from(delta[2 * row]);
            assert!(
                (d - want).abs() <= 1e-4,
                "{name}: Delta of row {row}: got {d}, want {want}"
            );
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn coop_flash_attention_agrees_with_a_numerical_gradient() {
    let Some(g) = flash_gpu("coop_flash_attention_agrees_with_a_numerical_gradient") else {
        return;
    };
    let case = FlashCase {
        b: 1,
        sq: 48,
        sk: 48,
        h: 2,
        hkv: 1,
        causal: true,
        offset: 0,
    };
    let (b, sq, sk, h, hkv) = (case.b, case.sq, case.sk, case.h, case.hkv);
    let [q, k, v, dout] = flash_inputs(&case, 900);
    let (tq, tk, tv, tdo) = (
        upload(g, &q, &[b, sq, h, HEAD]),
        upload(g, &k, &[b, sk, hkv, HEAD]),
        upload(g, &v, &[b, sk, hkv, HEAD]),
        upload(g, &dout, &[b, sq, h, HEAD]),
    );
    let opts = case.options();
    let out = zeros(g, &[b, sq, h, HEAD]);
    let stats = zeros(g, &[b, h, sq, 2]);
    g.engine
        .flash_attention_v2_with_stats(&g.ctx, &tq, &tk, &tv, &out, &stats, &opts, None);
    let (gq, gk, gv) = (
        zeros(g, &[b, sq, h, HEAD]),
        zeros(g, &[b, sk, hkv, HEAD]),
        zeros(g, &[b, sk, hkv, HEAD]),
    );
    g.engine.flash_attention_v2_backward_with_stats(
        &g.ctx, &tq, &tk, &tv, &out, &tdo, &gq, &gk, &gv, &stats, &opts, None,
    );
    let (gq, gk, gv) = (read(g, &gq), read(g, &gk), read(g, &gv));

    // L = sum(dO * attention(q, k, v)) from the CPU forward pass, which has
    // no f16 in it; the kernel's gradients must be its derivative.
    let loss = |q: &[f32], k: &[f32], v: &[f32]| -> f64 {
        let (o, _) = cpu_attention(q, k, v, None, case.dims(), case.visible());
        o.iter()
            .zip(&dout)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum()
    };
    let eps = 1e-2f32;
    let numeric = |which: usize, at: usize| -> f32 {
        let bump = |sign: f32| {
            let (mut q2, mut k2, mut v2) = (q.clone(), k.clone(), v.clone());
            match which {
                0 => q2[at] += sign * eps,
                1 => k2[at] += sign * eps,
                _ => v2[at] += sign * eps,
            }
            loss(&q2, &k2, &v2)
        };
        ((bump(1.0) - bump(-1.0)) / (2.0 * f64::from(eps))) as f32
    };
    for (which, label, grad) in [(0, "dq", &gq), (1, "dk", &gk), (2, "dv", &gv)] {
        let atol = 0.02 * max_abs(grad) + 2e-3;
        for at in [0, 1, 37, 63, 64, 500, 1000, grad.len() / 2, grad.len() - 1] {
            let want = numeric(which, at);
            assert!(
                grad[at].is_finite() && (grad[at] - want).abs() <= atol,
                "{label}[{at}]: kernel {}, numerical {want}, tol {atol}",
                grad[at]
            );
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn coop_flash_attention_counts_f16_overflow_and_poisons_its_output() {
    let Some(g) = flash_gpu("coop_flash_attention_counts_f16_overflow_and_poisons_its_output")
    else {
        return;
    };
    let case = FlashCase {
        b: 1,
        sq: 64,
        sk: 64,
        h: 1,
        hkv: 1,
        causal: true,
        offset: 0,
    };
    let shape = [1, 64, 1, HEAD];
    let count = || pollster::block_on(g.engine.coop_f16_overflows(&g.ctx)).unwrap();
    let [q, k, v, dout] = flash_inputs(&case, 300);
    let run = |q: &[f32], dout: &[f32]| {
        let (tq, tk, tv, tdo) = (
            upload(g, q, &shape),
            upload(g, &k, &shape),
            upload(g, &v, &shape),
            upload(g, dout, &shape),
        );
        let opts = case.options();
        let (out, stats) = (zeros(g, &shape), zeros(g, &[1, 1, 64, 2]));
        g.engine
            .flash_attention_v2_with_stats(&g.ctx, &tq, &tk, &tv, &out, &stats, &opts, None);
        let (gq, gk, gv) = (zeros(g, &shape), zeros(g, &shape), zeros(g, &shape));
        g.engine.flash_attention_v2_backward_with_stats(
            &g.ctx, &tq, &tk, &tv, &out, &tdo, &gq, &gk, &gv, &stats, &opts, None,
        );
        (read(g, &out), read(g, &gq), read(g, &gk))
    };

    let before = count();
    let (out, gq, gk) = run(&q, &dout);
    assert_eq!(count(), before, "finite inputs count nothing");
    assert!(out.iter().chain(&gq).chain(&gk).all(|v| v.is_finite()));

    // One q value past the f16 range: the one thread that stages it counts it.
    let mut big = q.clone();
    big[5] = 70_000.0;
    let before = count();
    let (out, _, _) = run(&big, &dout);
    assert!(count() > before, "the overflow is counted");
    assert!(out[0].is_nan(), "the workgroup's first output is poisoned");

    // A dO value past the range poisons dQ and dK of the workgroups that stage it.
    let mut big = dout.clone();
    big[7] = 70_000.0;
    let before = count();
    let (out, gq, gk) = run(&q, &big);
    assert!(count() > before, "the dO overflow is counted");
    assert!(
        out.iter().all(|v| v.is_finite()),
        "the forward pass is clean"
    );
    assert!(gq[0].is_nan(), "dQ is poisoned");
    assert!(gk[0].is_nan(), "dK is poisoned");
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn flash_attention_keeps_the_scalar_kernels_for_what_the_coop_kernels_do_not_cover() {
    let Some(g) = flash_gpu(
        "flash_attention_keeps_the_scalar_kernels_for_what_the_coop_kernels_do_not_cover",
    ) else {
        return;
    };
    // (name, query length, head size, mask, window, dropout, switch on)
    let calls = [
        ("head size 128", 32, 128, false, 0, 0.0, true),
        ("mask", 32, 64, true, 0, 0.0, true),
        ("dropout", 32, 64, false, 0, 0.1, true),
        ("window", 32, 64, false, 8, 0.0, true),
        ("one query", 1, 64, false, 0, 0.0, true),
        ("switched off", 32, 64, false, 0, 0.0, false),
    ];
    for (name, sq, d, masked, window, dropout, on) in calls {
        g.engine.set_flash_coop(on);
        let (b, sk, h, hkv) = (1, 32, 2, 2);
        let q = rand_vec(b * sq * h * d, 70);
        let k = rand_vec(b * sk * hkv * d, 71);
        let v = rand_vec(b * sk * hkv * d, 72);
        let mask = rand_vec(b * h * sq * sk, 73);
        let opts = AttentionOptions {
            causal: sq > 1,
            window,
            dropout_prob: dropout,
            seed: 5,
            ..AttentionOptions::default()
        };
        let out = zeros(g, &[b, sq, h, d]);
        let stats = zeros(g, &[b, h, sq, 2]);
        let tmask = upload(g, &mask, &[b, h, sq, sk]);
        let runs = g.engine.flash_coop_runs();
        g.engine.flash_attention_v2_with_stats(
            &g.ctx,
            &upload(g, &q, &[b, sq, h, d]),
            &upload(g, &k, &[b, sk, hkv, d]),
            &upload(g, &v, &[b, sk, hkv, d]),
            &out,
            &stats,
            &opts,
            masked.then_some(&tmask),
        );
        assert_eq!(
            g.engine.flash_coop_runs(),
            runs,
            "{name}: stays on the scalar kernel"
        );
        let got = read(g, &out);
        assert!(got.iter().all(|v| v.is_finite()), "{name}");
        // Without dropout or a window the scalar kernel is f32 throughout.
        if dropout == 0.0 && window == 0 {
            let causal = opts.causal;
            let (want, _) = cpu_attention(
                &q,
                &k,
                &v,
                masked.then_some(mask.as_slice()),
                (b, sq, sk, h, hkv, d),
                |i, j| !causal || j <= i,
            );
            assert_near(name, &got, &want, (1e-5, 1e-3));
        }
    }
    g.engine.set_flash_coop(true);
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn pipelined_tiles_match_cpu_at_large_m() {
    let Some(g) = coop_gpu("pipelined_tiles_match_cpu_at_large_m") else {
        return;
    };
    // M just under and at a multiple of 128 and 256 and well past 256, N not
    // a multiple of 128 (or 8 or 16), K not a multiple of 32 or 64.
    for (seed, &(m, k, n)) in [
        (2047, 288, 200),
        (1000, 1000, 520),
        (2048, 256, 384),
        (513, 100, 129),
    ]
    .iter()
    .enumerate()
    {
        let a = f16_round(&rand_vec(m * k, 400 + seed as u64));
        let b = f16_round(&rand_vec(k * n, 500 + seed as u64));
        let want = cpu_matmul(&a, &b, (m, k, n));
        let zeros = vec![0.0; m * n];
        for tile in CoopTile::PIPELINED {
            for split in [1, 3] {
                for transposed_b in [false, true] {
                    let config = CoopConfig { tile, split };
                    let ep = (Epilogue::STORE, zeros.as_slice());
                    let Some(got) = run_or_skip(g, (m, k, n), (&a, &b), transposed_b, ep, config)
                    else {
                        continue;
                    };
                    let e = max_rel_error(&got, &want);
                    assert!(
                        e < 2e-5,
                        "{} split {split} {m}x{k}x{n} transposed_b {transposed_b}: {e:e}",
                        tile.name()
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn pipelined_tiles_times_identity_return_a_at_large_m() {
    let Some(g) = coop_gpu("pipelined_tiles_times_identity_return_a_at_large_m") else {
        return;
    };
    // Integers below 2048 are exact in f16 and so are their sums: `a @ I`
    // must equal `a` element for element, at a K of 9 steps of 32 (5 of 64)
    // and an M that is no multiple of any tile.
    let (m, k, n) = (2047, 288, 288);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 1000) as f32).collect();
    let mut eye = vec![0.0f32; k * n];
    for i in 0..k {
        eye[i * n + i] = 1.0;
    }
    let zeros = vec![0.0; m * n];
    for tile in CoopTile::PIPELINED {
        for split in [1, 2] {
            for transposed_b in [false, true] {
                let config = CoopConfig { tile, split };
                let ep = (Epilogue::STORE, zeros.as_slice());
                let Some(got) = run_or_skip(g, (m, k, n), (&a, &eye), transposed_b, ep, config)
                else {
                    continue;
                };
                assert_eq!(
                    got,
                    a,
                    "{} split {split} transposed_b {transposed_b}",
                    tile.name()
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn pipelined_tiles_apply_alpha_beta_and_count_overflow() {
    let Some(g) = coop_gpu("pipelined_tiles_apply_alpha_beta_and_count_overflow") else {
        return;
    };
    let (m, k, n) = (300, 96, 136);
    let a = f16_round(&rand_vec(m * k, 600));
    let b = f16_round(&rand_vec(k * n, 601));
    let old = rand_vec(m * n, 602);
    let product = cpu_matmul(&a, &b, (m, k, n));
    let want: Vec<f64> = product
        .iter()
        .zip(&old)
        .map(|(&p, &o)| -2.0 * p + 0.25 * f64::from(o))
        .collect();
    let ep = Epilogue {
        alpha: -2.0,
        beta: 0.25,
    };
    for tile in CoopTile::PIPELINED {
        for transposed_b in [false, true] {
            let config = CoopConfig { tile, split: 1 };
            let Some(got) = run_or_skip(g, (m, k, n), (&a, &b), transposed_b, (ep, &old), config)
            else {
                continue;
            };
            let e = max_rel_error(&got, &want);
            assert!(
                e < 2e-5,
                "{} transposed_b {transposed_b}: {e:e}",
                tile.name()
            );
        }
    }
    // One value of `a` past the f16 range is counted once, by the one thread
    // that stages it, and poisons its output tile: N = 16 is one column of
    // workgroups whatever the tile.
    let (m, k, n) = (256, 64, 16);
    let b = rand_vec(k * n, 603);
    let mut big = rand_vec(m * k, 604);
    big[3 * k + 5] = 70_000.0;
    let zeros = vec![0.0; m * n];
    for tile in CoopTile::PIPELINED {
        let before = pollster::block_on(g.engine.coop_f16_overflows(&g.ctx)).unwrap();
        let config = CoopConfig { tile, split: 1 };
        let ep = (Epilogue::STORE, zeros.as_slice());
        let Some(got) = run_or_skip(g, (m, k, n), (&big, &b), false, ep, config) else {
            continue;
        };
        assert_eq!(
            pollster::block_on(g.engine.coop_f16_overflows(&g.ctx)).unwrap(),
            before + 1,
            "{}",
            tile.name()
        );
        assert!(
            got[0].is_nan(),
            "{}: the tile's first element is poisoned",
            tile.name()
        );
    }
}

/// Rows around the grouped raster of the kernel (8 row tiles of the tile's
/// rows per group): 1352 rows is 11 row tiles of 128 (a last group of 3), 1100
/// is 9 (a last group of 1), 1000 is 8 (one whole group, last tile partial in
/// rows) and 1024 is 8 whole tiles. N is not a multiple of every tile's width,
/// so the last column block is partial too.
const RASTER_SHAPES: &[(usize, usize, usize)] = &[
    (1352, 96, 264),
    (1100, 72, 136),
    (1000, 80, 200),
    (1024, 64, 1032),
];

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn coop_grouped_raster_matches_f64_with_partial_groups() {
    let Some(g) = coop_gpu("coop_grouped_raster_matches_f64_with_partial_groups") else {
        return;
    };
    for (seed, &(m, k, n)) in RASTER_SHAPES.iter().enumerate() {
        let a = f16_round(&rand_vec(m * k, 700 + seed as u64));
        let b = f16_round(&rand_vec(k * n, 710 + seed as u64));
        let old = rand_vec(m * n, 720 + seed as u64);
        let product = cpu_matmul(&a, &b, (m, k, n));
        // A tile that ran twice or not at all would show in `0.5 * a @ b + 0.25 * old`.
        let want: Vec<f64> = product
            .iter()
            .zip(&old)
            .map(|(&p, &o)| 0.5 * p + 0.25 * f64::from(o))
            .collect();
        for transposed_b in [false, true] {
            let ep = (
                Epilogue {
                    alpha: 0.5,
                    beta: 0.25,
                },
                old.as_slice(),
            );
            // The router's own pick, then every tile with K whole and split.
            let got = run_coop(g, (m, k, n), &a, &b, transposed_b, ep);
            let e = max_rel_error(&got, &want);
            assert!(
                e < 2e-5,
                "router {m}x{k}x{n} transposed_b {transposed_b}: {e:e}"
            );
            for config in configs() {
                let Some(got) = run_or_skip(g, (m, k, n), (&a, &b), transposed_b, ep, config)
                else {
                    continue;
                };
                let e = max_rel_error(&got, &want);
                assert!(
                    e < 2e-5,
                    "{} split {} {m}x{k}x{n} transposed_b {transposed_b}: {e:e}",
                    config.tile.name(),
                    config.split
                );
            }
        }
    }
}

/// Environment variable that makes the next test write its outputs to a file
/// and stop; it is how the test runs itself with the earlier raster order.
const RASTER_DUMP: &str = "FERROTENSOR_TEST_RASTER_DUMP";

/// The bits of the router's pick and of a tile of each size on the raster
/// shapes, both layouts, K whole and split.
fn raster_output_bytes(g: &Gpu) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (seed, &(m, k, n)) in RASTER_SHAPES.iter().enumerate() {
        let a = f16_round(&rand_vec(m * k, 800 + seed as u64));
        let b = f16_round(&rand_vec(k * n, 810 + seed as u64));
        let old = rand_vec(m * n, 820 + seed as u64);
        let ep = (
            Epilogue {
                alpha: 0.5,
                beta: 0.25,
            },
            old.as_slice(),
        );
        for transposed_b in [false, true] {
            let mut outputs = vec![run_coop(g, (m, k, n), &a, &b, transposed_b, ep)];
            for config in configs() {
                if let Some(got) = run_or_skip(g, (m, k, n), (&a, &b), transposed_b, ep, config) {
                    outputs.push(got);
                }
            }
            for got in outputs {
                bytes.extend(got.iter().flat_map(|v| v.to_bits().to_le_bytes()));
            }
        }
    }
    bytes
}

/// Any one-to-one map of workgroups to tiles leaves every tile's arithmetic
/// alone, so the grouped raster must give the very bits that group size 1 (the
/// column-fastest order of before) gives. The group size is read once per
/// process, so the test runs itself again with `FERROTENSOR_RASTER_GROUP=1`.
#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn coop_grouped_raster_is_bit_identical_to_the_column_fastest_order() {
    let name = "coop_grouped_raster_is_bit_identical_to_the_column_fastest_order";
    let Some(g) = coop_gpu(name) else {
        return;
    };
    let bytes = raster_output_bytes(g);
    if let Ok(path) = std::env::var(RASTER_DUMP) {
        std::fs::write(path, &bytes).unwrap();
        return;
    }
    if std::env::var("FERROTENSOR_RASTER_GROUP").is_ok_and(|v| v == "1") {
        eprintln!("{name}: skipped, this run already uses group size 1");
        return;
    }
    let dump = std::env::temp_dir().join(format!("ferrotensor-raster-{}.bin", std::process::id()));
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            name,
            "--ignored",
            "--test-threads=1",
            "--nocapture",
        ])
        .env("FERROTENSOR_RASTER_GROUP", "1")
        .env(RASTER_DUMP, &dump)
        .status()
        .unwrap();
    assert!(status.success(), "the child run with group size 1 failed");
    let column_fastest = std::fs::read(&dump).unwrap();
    std::fs::remove_file(&dump).unwrap();
    assert_eq!(column_fastest.len(), bytes.len());
    let differing = column_fastest
        .iter()
        .zip(&bytes)
        .filter(|(x, y)| x != y)
        .count();
    assert_eq!(differing, 0, "{differing} bytes differ from group size 1");
    eprintln!("{name}: {} bytes identical", bytes.len());
}
