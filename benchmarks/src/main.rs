//! Rusty GPU Performance Benchmarks
//!
//! Measures performance of key GPU operations and compares against theoretical peaks.
//! Run with: cargo run -p rusty-benchmarks --release

use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use std::time::Instant;

const WARMUP_ITERS: usize = 5;
const BENCH_ITERS: usize = 50;

fn main() {
    println!("======================================================================");
    println!("             RUSTY ML FRAMEWORK - GPU BENCHMARKS                      ");
    println!("======================================================================");
    println!();

    // Initialize GPU
    let ctx = pollster::block_on(async { WgpuContext::new().await });
    let engine = ComputeEngine::new(&ctx);

    // Print GPU info
    print_gpu_info(&ctx);

    println!();
    println!("[BENCH] Running benchmarks...");
    println!();
    println!("----------------------------------------------------------------------");

    // MatMul benchmarks
    bench_matmul(&ctx, &engine, 512, 512, 512);
    bench_matmul(&ctx, &engine, 1024, 1024, 1024);
    bench_matmul(&ctx, &engine, 2048, 2048, 2048);
    bench_matmul(&ctx, &engine, 4096, 4096, 4096);

    println!("----------------------------------------------------------------------");

    // Element-wise operations
    bench_elementwise(&ctx, &engine, 1_000_000, "1M");
    bench_elementwise(&ctx, &engine, 10_000_000, "10M");
    bench_elementwise(&ctx, &engine, 100_000_000, "100M");

    println!("----------------------------------------------------------------------");

    // Softmax (attention-like)
    bench_softmax(&ctx, &engine, 512, 512);
    bench_softmax(&ctx, &engine, 1024, 1024);
    bench_softmax(&ctx, &engine, 2048, 2048);

    println!("----------------------------------------------------------------------");

    // RMSNorm
    bench_rmsnorm(&ctx, &engine, 4096, 512);
    bench_rmsnorm(&ctx, &engine, 4096, 1024);
    bench_rmsnorm(&ctx, &engine, 4096, 2048);

    println!("----------------------------------------------------------------------");

    // SiLU activation
    bench_silu(&ctx, &engine, 1_000_000, "1M");
    bench_silu(&ctx, &engine, 10_000_000, "10M");

    println!();
    println!("======================================================================");
    println!("                     BENCHMARK COMPLETE                               ");
    println!("======================================================================");
    println!();
}

fn print_gpu_info(ctx: &WgpuContext) {
    let info = ctx.adapter.get_info();
    println!("[GPU] Information:");
    println!("      Name:    {}", info.name);
    println!("      Backend: {:?}", info.backend);
    println!("      Type:    {:?}", info.device_type);
}

/// Sync GPU by submitting empty and polling
fn sync_gpu(ctx: &WgpuContext) {
    ctx.queue.submit([]);
    ctx.device.poll(wgpu::Maintain::Wait);
}

fn bench_matmul(ctx: &WgpuContext, engine: &ComputeEngine, m: usize, k: usize, n: usize) {
    let a_data: Vec<f32> = (0..m * k).map(|i| (i % 100) as f32 / 100.0).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i % 100) as f32 / 100.0).collect();

    let a = UnifiedTensor::new(ctx, &a_data, &[m, k]);
    let b = UnifiedTensor::new(ctx, &b_data, &[k, n]);
    let c = UnifiedTensor::empty(ctx, &[m, n]);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        engine.matmul(ctx, &a, &b, &c);
        sync_gpu(ctx);
    }

    // Bench
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        engine.matmul(ctx, &a, &b, &c);
        sync_gpu(ctx);
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / BENCH_ITERS as f64;

    // Calculate GFLOPS (2*M*N*K operations for matmul)
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let gflops = flops / (avg_ms / 1000.0) / 1e9;

    println!(
        "  MatMul [{:>4}x{:>4}x{:>4}]:  {:>8.3} ms  |  {:>8.2} GFLOPS",
        m, k, n, avg_ms, gflops
    );
}

fn bench_elementwise(ctx: &WgpuContext, engine: &ComputeEngine, size: usize, label: &str) {
    let data: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();

    let a = UnifiedTensor::new(ctx, &data, &[size]);
    let b = UnifiedTensor::new(ctx, &data, &[size]);
    let c = UnifiedTensor::empty(ctx, &[size]);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        engine.add(ctx, &a, &b, &c);
        sync_gpu(ctx);
    }

    // Bench
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        engine.add(ctx, &a, &b, &c);
        sync_gpu(ctx);
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / BENCH_ITERS as f64;

    // Calculate bandwidth (2 reads + 1 write = 3 * size * 4 bytes)
    let bytes = 3.0 * size as f64 * 4.0;
    let gb_s = bytes / (avg_ms / 1000.0) / 1e9;

    println!(
        "  Add [{:>5} elements]:    {:>8.3} ms  |  {:>8.2} GB/s",
        label, avg_ms, gb_s
    );
}

fn bench_softmax(ctx: &WgpuContext, engine: &ComputeEngine, batch: usize, seq_len: usize) {
    let data: Vec<f32> = (0..batch * seq_len)
        .map(|i| (i % 100) as f32 / 10.0)
        .collect();

    let input = UnifiedTensor::new(ctx, &data, &[batch, seq_len]);
    let output = UnifiedTensor::empty(ctx, &[batch, seq_len]);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        engine.softmax(ctx, &input, &output);
        sync_gpu(ctx);
    }

    // Bench
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        engine.softmax(ctx, &input, &output);
        sync_gpu(ctx);
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / BENCH_ITERS as f64;

    let throughput = (batch * seq_len) as f64 / (avg_ms / 1000.0) / 1e6;

    println!(
        "  Softmax [{:>4}x{:>4}]:      {:>8.3} ms  |  {:>8.2} M elem/s",
        batch, seq_len, avg_ms, throughput
    );
}

fn bench_rmsnorm(ctx: &WgpuContext, engine: &ComputeEngine, hidden_dim: usize, seq_len: usize) {
    let data: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| (i % 100) as f32 / 100.0)
        .collect();
    let weight: Vec<f32> = (0..hidden_dim).map(|_| 1.0).collect();

    let input = UnifiedTensor::new(ctx, &data, &[seq_len, hidden_dim]);
    let w = UnifiedTensor::new(ctx, &weight, &[hidden_dim]);
    let output = UnifiedTensor::empty(ctx, &[seq_len, hidden_dim]);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        engine.rms_norm(ctx, &input, &w, &output);
        sync_gpu(ctx);
    }

    // Bench
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        engine.rms_norm(ctx, &input, &w, &output);
        sync_gpu(ctx);
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / BENCH_ITERS as f64;

    let throughput = (seq_len * hidden_dim) as f64 / (avg_ms / 1000.0) / 1e6;

    println!(
        "  RMSNorm [seq={:>4}, dim={:>4}]: {:>6.3} ms  |  {:>8.2} M elem/s",
        seq_len, hidden_dim, avg_ms, throughput
    );
}

fn bench_silu(ctx: &WgpuContext, engine: &ComputeEngine, size: usize, label: &str) {
    let data: Vec<f32> = (0..size)
        .map(|i| (i as f32 / size as f32) * 2.0 - 1.0)
        .collect();

    let input = UnifiedTensor::new(ctx, &data, &[size]);
    let output = UnifiedTensor::empty(ctx, &[size]);

    // Warmup
    for _ in 0..WARMUP_ITERS {
        engine.silu(ctx, &input, &output);
        sync_gpu(ctx);
    }

    // Bench
    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        engine.silu(ctx, &input, &output);
        sync_gpu(ctx);
    }
    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_secs_f64() * 1000.0 / BENCH_ITERS as f64;

    let throughput = size as f64 / (avg_ms / 1000.0) / 1e9;

    println!(
        "  SiLU [{:>5} elements]:    {:>8.3} ms  |  {:>8.2} G elem/s",
        label, avg_ms, throughput
    );
}
