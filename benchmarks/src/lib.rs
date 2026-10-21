//! GPU kernel benchmarks for the Ferrotensor backend.
//!
//! Run with `cargo run -p ferrotensor-benchmarks --release` or
//! `cargo bench -p ferrotensor-benchmarks`. Pass `--quick` for smaller sizes and
//! `--filter <text>` to run only cases whose name contains `text`.
//!
//! Each case dispatches the kernel `batch` times back to back and then waits
//! for the GPU, so per-call CPU overhead (bind group and uniform creation,
//! submit) overlaps with GPU work. The reported time is the median per-call
//! time over several such samples.

use ferrotensor_backend::{AttentionOptions, ComputeEngine, UnifiedTensor, WgpuContext};
use std::time::{Duration, Instant};

/// Benchmark settings parsed from the command line.
#[derive(Debug, Clone, Default)]
pub struct Config {
    /// Use smaller problem sizes (about 8 s in total).
    pub quick: bool,
    /// Only run cases whose name contains this string.
    pub filter: Option<String>,
}

impl Config {
    /// Parses `--quick` and `--filter <text>`; other arguments (such as the
    /// `--bench` flag that `cargo bench` passes) are ignored.
    pub fn from_args() -> Self {
        let mut cfg = Config::default();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--quick" => cfg.quick = true,
                "--filter" => cfg.filter = args.next(),
                _ => {}
            }
        }
        cfg
    }

    fn wants(&self, name: &str) -> bool {
        self.filter.as_deref().map_or(true, |f| name.contains(f))
    }
}

/// One measured case.
struct Measurement {
    name: String,
    shape: String,
    median: Duration,
    /// (value, unit), e.g. GFLOPS or GB/s.
    rate: (f64, &'static str),
}

/// Times `op` and returns the median per-call duration.
fn time_op(ctx: &WgpuContext, mut op: impl FnMut()) -> Duration {
    let sync = || {
        ctx.queue.submit([]);
        ctx.device.poll(wgpu::Maintain::Wait);
    };
    // Warm up and estimate the cost of one call.
    let start = Instant::now();
    for _ in 0..3 {
        op();
    }
    sync();
    let one = start.elapsed() / 3;
    // Batch calls so each sample takes about 20 ms (at most 64 calls).
    let batch = (Duration::from_millis(20).as_secs_f64() / one.as_secs_f64().max(1e-6))
        .clamp(1.0, 64.0) as u32;
    let mut samples = Vec::new();
    let budget = Instant::now();
    while samples.len() < 5 || (samples.len() < 30 && budget.elapsed() < Duration::from_secs(1)) {
        let t = Instant::now();
        for _ in 0..batch {
            op();
        }
        sync();
        samples.push(t.elapsed() / batch);
    }
    samples.sort();
    samples[samples.len() / 2]
}

fn rand_tensor(ctx: &WgpuContext, shape: &[usize], seed: u32) -> UnifiedTensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| {
            ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) >> 8) as f32
                / (1u32 << 24) as f32
                - 0.5
        })
        .collect();
    UnifiedTensor::new(ctx, &data, shape)
}

fn gflops(flops: f64, t: Duration) -> (f64, &'static str) {
    (flops / t.as_secs_f64() / 1e9, "GFLOPS")
}

fn gbps(bytes: f64, t: Duration) -> (f64, &'static str) {
    (bytes / t.as_secs_f64() / 1e9, "GB/s")
}

#[derive(Clone, Copy)]
enum MatmulKind {
    Plain,
    TransposedB,
    TransposedA,
}

fn bench_matmul(
    ctx: &WgpuContext,
    engine: &ComputeEngine,
    kind: MatmulKind,
    (m, k, n): (usize, usize, usize),
) -> Measurement {
    let (name, a_shape, b_shape) = match kind {
        MatmulKind::Plain => ("matmul", [m, k], [k, n]),
        MatmulKind::TransposedB => ("matmul_bt", [m, k], [n, k]),
        MatmulKind::TransposedA => ("matmul_at", [k, m], [k, n]),
    };
    let a = rand_tensor(ctx, &a_shape, 1);
    let b = rand_tensor(ctx, &b_shape, 2);
    let c = UnifiedTensor::empty(ctx, &[m, n]);
    let t = time_op(ctx, || match kind {
        MatmulKind::Plain => engine.matmul(ctx, &a, &b, &c),
        MatmulKind::TransposedB => engine.matmul_bt(ctx, &a, &b, &c),
        MatmulKind::TransposedA => engine.matmul_at(ctx, &a, &b, &c),
    });
    Measurement {
        name: name.into(),
        shape: format!("{m}x{k}x{n}"),
        median: t,
        rate: gflops(2.0 * (m * k * n) as f64, t),
    }
}

/// FLOPs of one attention forward pass (QK^T and PV), halved when causal.
fn attention_flops(b: usize, h: usize, s: usize, d: usize, causal: bool) -> f64 {
    let full = 4.0 * (b * h * s * s * d) as f64;
    if causal {
        full / 2.0
    } else {
        full
    }
}

fn bench_attention(
    ctx: &WgpuContext,
    engine: &ComputeEngine,
    (b, h, s, d): (usize, usize, usize, usize),
    backward: bool,
) -> Measurement {
    let shape = [b, s, h, d];
    let q = rand_tensor(ctx, &shape, 3);
    let k = rand_tensor(ctx, &shape, 4);
    let v = rand_tensor(ctx, &shape, 5);
    let out = UnifiedTensor::empty(ctx, &shape);
    let opts = AttentionOptions {
        causal: true,
        ..AttentionOptions::default()
    };
    engine.flash_attention_v2_with(ctx, &q, &k, &v, &out, &opts, None);
    let flops = attention_flops(b, h, s, d, true);
    let (name, t, flops) = if backward {
        let dout = rand_tensor(ctx, &shape, 6);
        let (gq, gk, gv) = (
            UnifiedTensor::empty(ctx, &shape),
            UnifiedTensor::empty(ctx, &shape),
            UnifiedTensor::empty(ctx, &shape),
        );
        let t = time_op(ctx, || {
            engine.flash_attention_v2_backward_with(
                ctx, &q, &k, &v, &out, &dout, &gq, &gk, &gv, &opts, None,
            )
        });
        // Backward does 2.5x the forward matmul work (FlashAttention-2 convention).
        ("attention_bwd", t, 2.5 * flops)
    } else {
        let t = time_op(ctx, || {
            engine.flash_attention_v2_with(ctx, &q, &k, &v, &out, &opts, None)
        });
        ("attention_fwd", t, flops)
    };
    Measurement {
        name: name.into(),
        shape: format!("b{b} h{h} s{s} d{d} causal"),
        median: t,
        rate: gflops(flops, t),
    }
}

fn bench_rows(
    ctx: &WgpuContext,
    engine: &ComputeEngine,
    name: &str,
    (rows, len): (usize, usize),
) -> Measurement {
    let x = rand_tensor(ctx, &[rows, len], 7);
    let w = rand_tensor(ctx, &[len], 8);
    let out = UnifiedTensor::empty(ctx, &[rows, len]);
    let t = match name {
        "softmax" => time_op(ctx, || engine.softmax(ctx, &x, &out)),
        _ => time_op(ctx, || engine.rms_norm(ctx, &x, &w, &out)),
    };
    Measurement {
        name: name.into(),
        shape: format!("{rows}x{len}"),
        median: t,
        // One read and one write per element.
        rate: gbps(8.0 * (rows * len) as f64, t),
    }
}

fn bench_elementwise(
    ctx: &WgpuContext,
    engine: &ComputeEngine,
    name: &str,
    n: usize,
) -> Measurement {
    let a = rand_tensor(ctx, &[n], 9);
    let b = rand_tensor(ctx, &[n], 10);
    let out = UnifiedTensor::empty(ctx, &[n]);
    let (t, bytes) = match name {
        "add" => (time_op(ctx, || engine.add(ctx, &a, &b, &out)), 12.0),
        _ => (time_op(ctx, || engine.silu(ctx, &a, &out)), 8.0),
    };
    Measurement {
        name: name.into(),
        shape: format!("{}M", n / 1_000_000),
        median: t,
        rate: gbps(bytes * n as f64, t),
    }
}

/// Runs every benchmark selected by `cfg` and prints a Markdown table.
pub fn run(cfg: &Config) {
    let ctx = pollster::block_on(WgpuContext::new());
    let engine = ComputeEngine::new(&ctx);
    let info = ctx.adapter.get_info();
    println!(
        "GPU: {} ({:?}, {:?})",
        info.name, info.backend, info.device_type
    );
    println!("Mode: {}\n", if cfg.quick { "quick" } else { "full" });
    println!("| kernel | shape | median time | throughput |");
    println!("|---|---|---:|---:|");

    let mut cases: Vec<(String, Box<dyn Fn() -> Measurement + '_>)> = Vec::new();
    let (ctx, engine) = (&ctx, &engine);
    let squares: &[usize] = if cfg.quick {
        &[512, 1024]
    } else {
        &[512, 1024, 2048, 4096]
    };
    for &s in squares {
        cases.push((
            "matmul".into(),
            Box::new(move || bench_matmul(ctx, engine, MatmulKind::Plain, (s, s, s))),
        ));
    }
    // A 512-token MLP up-projection and an odd-sized product.
    for dims in [(512, 2048, 8192), (1000, 1000, 1000)] {
        cases.push((
            "matmul".into(),
            Box::new(move || bench_matmul(ctx, engine, MatmulKind::Plain, dims)),
        ));
    }
    for kind in [MatmulKind::TransposedB, MatmulKind::TransposedA] {
        let name = match kind {
            MatmulKind::TransposedB => "matmul_bt",
            _ => "matmul_at",
        };
        cases.push((
            name.into(),
            Box::new(move || bench_matmul(ctx, engine, kind, (1024, 1024, 1024))),
        ));
    }
    let seqs: &[usize] = if cfg.quick {
        &[512]
    } else {
        &[512, 1024, 2048]
    };
    for &s in seqs {
        cases.push((
            "attention_fwd".into(),
            Box::new(move || bench_attention(ctx, engine, (1, 16, s, 64), false)),
        ));
    }
    if !cfg.quick {
        cases.push((
            "attention_fwd".into(),
            Box::new(move || bench_attention(ctx, engine, (1, 8, 1024, 128), false)),
        ));
    }
    for &s in if cfg.quick {
        &[512][..]
    } else {
        &[512, 1024][..]
    } {
        cases.push((
            "attention_bwd".into(),
            Box::new(move || bench_attention(ctx, engine, (1, 16, s, 64), true)),
        ));
    }
    cases.push((
        "softmax".into(),
        Box::new(move || bench_rows(ctx, engine, "softmax", (4096, 1024))),
    ));
    cases.push((
        "rms_norm".into(),
        Box::new(move || bench_rows(ctx, engine, "rms_norm", (2048, 4096))),
    ));
    cases.push((
        "add".into(),
        Box::new(move || bench_elementwise(ctx, engine, "add", 4_000_000)),
    ));
    cases.push((
        "silu".into(),
        Box::new(move || bench_elementwise(ctx, engine, "silu", 4_000_000)),
    ));

    for (name, case) in &cases {
        if !cfg.wants(name) {
            continue;
        }
        let m = case();
        println!(
            "| {} | {} | {:.3} ms | {:.1} {} |",
            m.name,
            m.shape,
            m.median.as_secs_f64() * 1e3,
            m.rate.0,
            m.rate.1
        );
    }
}
