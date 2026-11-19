//! The cooperative-matrix matmul against CPU references.
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

use ferrotensor_backend::{ComputeEngine, Epilogue, UnifiedTensor, WgpuContext};
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
    (m, k, n): (usize, usize, usize),
    a: &[f32],
    b: &[f32],
    transposed_b: bool,
    (ep, out0): (Epilogue, &[f32]),
) -> Vec<f32> {
    let a_t = UnifiedTensor::new(&g.ctx, a, &[m, k]);
    let b_t = if transposed_b {
        UnifiedTensor::from_f32_as_f16(&g.ctx, &transpose(b, k, n), &[n, k])
    } else {
        UnifiedTensor::from_f32_as_f16(&g.ctx, b, &[k, n])
    };
    let out = UnifiedTensor::new(&g.ctx, out0, &[m, n]);
    g.engine
        .matmul_coop_into(&g.ctx, transposed_b, [&a_t, &b_t, &out], ep)
        .unwrap();
    pollster::block_on(out.to_vec(&g.ctx)).unwrap()
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
            assert!(e16 < 2e-5, "{m}x{k}x{n} transposed_b {transposed_b}: {e16:e}");
            // Rounding a to f16 costs about 2^-11 relative per element.
            assert!(e32 < 3e-3, "{m}x{k}x{n} transposed_b {transposed_b}: {e32:e}");
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
    let (m, k, n) = (256, 64, 64);
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
    // One value past the f16 range in row 3: one workgroup counts it.
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
