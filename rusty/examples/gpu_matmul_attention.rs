//! Micro-demo: the same ops on the CPU and the GPU, checked against each
//! other and timed.
//!
//! Timings include reading the result back to the host (`to_vec`), which is
//! what forces the GPU to finish. They are a single-machine snapshot, not a
//! benchmark; see `benchmarks/` for the real harness.
//!
//! Run with: cargo run --release -p rusty --example gpu_matmul_attention

use std::time::Instant;

use rusty::prelude::*;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
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
        println!("no GPU adapter available; nothing to compare");
        return;
    };
    let cpu = Device::cpu();
    println!("gpu: {} ({})\n", gpu.name(), gpu.backend());

    println!("matmul [n, n] @ [n, n]");
    println!(
        "{:>6} {:>10} {:>10} {:>9} {:>12}",
        "n", "cpu ms", "gpu ms", "speedup", "max |diff|"
    );
    for n in [256usize, 512, 1024, 2048] {
        let a = Tensor::randn(&[n, n], &cpu);
        let b = Tensor::randn(&[n, n], &cpu);
        let (ga, gb) = (a.to_device(&gpu), b.to_device(&gpu));
        let (t_cpu, r_cpu) = time_ms(2, || a.matmul(&b).to_vec());
        let (t_gpu, r_gpu) = time_ms(5, || ga.matmul(&gb).to_vec());
        println!(
            "{n:>6} {t_cpu:>10.1} {t_gpu:>10.1} {:>8.1}x {:>12.2e}",
            t_cpu / t_gpu,
            max_abs_diff(&r_cpu, &r_gpu)
        );
    }

    println!("\ncausal scaled-dot-product attention, [1, 8 heads, seq, 64]");
    println!(
        "{:>6} {:>10} {:>10} {:>9} {:>12}",
        "seq", "cpu ms", "gpu ms", "speedup", "max |diff|"
    );
    for seq in [128usize, 512] {
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
        println!(
            "{seq:>6} {t_cpu:>10.1} {t_gpu:>10.1} {:>8.1}x {:>12.2e}",
            t_cpu / t_gpu,
            max_abs_diff(&r_cpu, &r_gpu)
        );
    }
}
