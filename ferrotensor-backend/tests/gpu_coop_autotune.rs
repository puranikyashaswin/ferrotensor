//! The cooperative-matrix autotuner on a GPU: every trial run computes the
//! product, and the tuner settles and saves its choice.
//!
//! This needs a GPU with f16 cooperative matrices and timestamp queries, and
//! the tuner turned on at the start of the process; otherwise the test
//! prints why it skipped and passes. Run it with:
//!
//! ```text
//! FERROTENSOR_EXPERIMENTAL=1 FERROTENSOR_AUTOTUNE=1 FERROTENSOR_AUTOTUNE_CACHE=/tmp/autotune-test.txt \
//!   cargo test -p ferrotensor-backend --test gpu_coop_autotune -- --ignored --test-threads=1
//! ```

#![allow(clippy::unwrap_used, clippy::panic)]

use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext, AUTOTUNE_ENV};

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

/// `a[M, K] @ b[N, K]^T` in f64.
fn cpu_matmul_bt(a: &[f32], b: &[f32], (m, k, n): (usize, usize, usize)) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k)
                .map(|p| f64::from(a[i * k + p]) * f64::from(b[j * k + p]))
                .sum();
        }
    }
    out
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

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn trials_compute_the_product_and_the_tuner_settles() {
    if std::env::var(AUTOTUNE_ENV).as_deref() != Ok("1") {
        eprintln!("skipped: {AUTOTUNE_ENV} is not 1");
        return;
    }
    let ctx = pollster::block_on(WgpuContext::new());
    let engine = ComputeEngine::new(&ctx);
    // The tuner exists only on a device that routes matmuls to the kernel
    // and times dispatches.
    if !engine.coop_matmul_active() || engine.autotune_finish(&ctx).is_none() {
        eprintln!(
            "skipped: {} has no routed f16 cooperative matrices with timestamp queries",
            ctx.adapter_info()
        );
        return;
    }
    let (m, k, n) = (128, 1024, 1024);
    let a = rand_vec(m * k, 1);
    let b = rand_vec(n * k, 2);
    let want = cpu_matmul_bt(&f16_round(&a), &f16_round(&b), (m, k, n));
    let a_t = UnifiedTensor::new(&ctx, &a, &[m, k]);
    let b_t = UnifiedTensor::from_f32_as_f16(&ctx, &b, &[n, k]);
    let out = UnifiedTensor::empty(&ctx, &[m, n]);
    // Four candidates or more, three timings each, one product per round.
    for round in 0..60 {
        engine.matmul_bt(&ctx, &a_t, &b_t, &out);
        let got = pollster::block_on(out.to_vec(&ctx)).unwrap();
        let err = max_rel_error(&got, &want);
        assert!(err < 2e-5, "round {round}: max rel err {err:e}");
    }
    let report = engine.autotune_finish(&ctx).expect("the tuner is on");
    eprintln!("{report:?}");
    assert_eq!(report.shapes_tuned + report.shapes_open, 1, "{report:?}");
    assert_eq!(report.shapes_cached, 0, "{report:?}");
    assert!(report.seconds > 0.0, "{report:?}");
    assert!(!report.budget_exhausted, "{report:?}");
}
