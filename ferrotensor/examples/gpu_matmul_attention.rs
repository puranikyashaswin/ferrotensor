//! Micro-demo: the same ops on the CPU and the GPU, checked against each
//! other and timed.
//!
//! Timings include reading the result back to the host (`to_vec`), which is
//! what forces the GPU to finish. They are a single-machine snapshot, not a
//! benchmark; see `benchmarks/` for the real harness.
//!
//! Run with: cargo run --release -p ferrotensor --example gpu_matmul_attention
//!
//! The example exits with an error when the CPU and GPU results differ by more
//! than 1e-3. Two environment variables are for CI, where the GPU is software
//! Vulkan on the CPU:
//!
//! - FERROTENSOR_EXAMPLE_QUICK=1 runs only the smaller sizes.
//! - FERROTENSOR_REQUIRE_GPU=1 makes a missing GPU an error instead of a skip.

use std::time::Instant;

use ferrotensor::prelude::*;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        // A NaN difference must stay NaN so the check below fails on it.
        .fold(0.0, |m, d| if d.is_nan() || d > m { d } else { m })
}

/// Largest allowed difference between the CPU and GPU results.
const TOLERANCE: f32 = 1e-3;

fn env_set(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

/// Best-of-`runs` wall time in milliseconds, after one warm-up call.
fn time_ms(runs: usize, mut f: impl FnMut() -> Vec<f32>) -> (f64, Vec<f32>) {
    let mut out = f();
    let mut best = f64::MAX;
    for _ in 0..runs {
        let t = Instant::now();
        out = f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    (best, out)
}

fn main() {
    let Some(gpu) = Device::gpu() else {
        assert!(
            !env_set("FERROTENSOR_REQUIRE_GPU"),
            "FERROTENSOR_REQUIRE_GPU=1 but no GPU adapter is available"
        );
        println!("no GPU adapter available; nothing to compare");
        return;
    };
    let quick = env_set("FERROTENSOR_EXAMPLE_QUICK");
    let cpu = Device::cpu();
    println!("gpu: {} ({})\n", gpu.name(), gpu.backend());

    println!("matmul [n, n] @ [n, n]");
    println!(
        "{:>6} {:>10} {:>10} {:>9} {:>12}",
        "n", "cpu ms", "gpu ms", "speedup", "max |diff|"
    );
    let matmul_sizes: &[usize] = if quick {
        &[256, 512]
    } else {
        &[256, 512, 1024, 2048]
    };
    for &n in matmul_sizes {
        let a = Tensor::randn(&[n, n], &cpu);
        let b = Tensor::randn(&[n, n], &cpu);
        let (ga, gb) = (a.to_device(&gpu), b.to_device(&gpu));
        let (t_cpu, r_cpu) = time_ms(2, || a.matmul(&b).to_vec());
        let (t_gpu, r_gpu) = time_ms(5, || ga.matmul(&gb).to_vec());
        let diff = max_abs_diff(&r_cpu, &r_gpu);
        println!(
            "{n:>6} {t_cpu:>10.1} {t_gpu:>10.1} {:>8.1}x {diff:>12.2e}",
            t_cpu / t_gpu
        );
        assert!(
            diff < TOLERANCE,
            "matmul n={n}: CPU and GPU differ by {diff}"
        );
    }

    println!("\ncausal scaled-dot-product attention, [1, 8 heads, seq, 64]");
    println!(
        "{:>6} {:>10} {:>10} {:>9} {:>12}",
        "seq", "cpu ms", "gpu ms", "speedup", "max |diff|"
    );
    let attention_lengths: &[usize] = if quick { &[128] } else { &[128, 512] };
    for &seq in attention_lengths {
        let shape = [1, 8, seq, 64];
        let (q, k, v) = (
            Tensor::randn(&shape, &cpu),
            Tensor::randn(&shape, &cpu),
            Tensor::randn(&shape, &cpu),
        );
        let (gq, gk, gv) = (q.to_device(&gpu), k.to_device(&gpu), v.to_device(&gpu));
        let (t_cpu, r_cpu) = time_ms(2, || {
            F::scaled_dot_product_attention(&q, &k, &v, true).to_vec()
        });
        let (t_gpu, r_gpu) = time_ms(5, || {
            F::scaled_dot_product_attention(&gq, &gk, &gv, true).to_vec()
        });
        let diff = max_abs_diff(&r_cpu, &r_gpu);
        println!(
            "{seq:>6} {t_cpu:>10.1} {t_gpu:>10.1} {:>8.1}x {diff:>12.2e}",
            t_cpu / t_gpu
        );
        assert!(
            diff < TOLERANCE,
            "attention seq={seq}: CPU and GPU differ by {diff}"
        );
    }
}
