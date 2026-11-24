//! Times every cooperative-matrix tile and K split, and every skinny tile,
//! on the matmul shapes of LoRA fine-tuning of SmolLM2 at 256 and 2048 rows:
//! `cargo run -p ferrotensor-benchmarks --release --bin matmul-tune`.
//!
//! The base-weight products are the forward `x @ W^T` (`a@bT`, f16 `W`
//! `[out, in]`) and the input gradient `dY @ W` (`a@b`) of the attention and
//! MLP projections and the vocabulary head. The LoRA products have N = rank
//! and K = the width of the layer they adapt, in both layouts.
//!
//! Every candidate runs `--calls` products back to back, `--reps` times; the
//! time is the median GPU time per product from timestamp queries (wall time
//! when the adapter has none), including the split-K reduction. Stdout gets
//! one Markdown table row per shape (the previous kernel, the router's pick
//! and the fastest candidate); stderr gets every candidate, with its largest
//! difference from the previous kernel's result.
//!
//! Flags: `--models 135m,360m,1.7b`, `--rows 256,2048`, `--only coop|skinny`,
//! `--tiles old|new|both` (the coop tiles to try: `old` is `CoopTile::ALL`,
//! the default; `new` is the register-prefetch tiles of
//! `CoopTile::PIPELINED`), `--calls N` (default 10), `--reps N` (default 3),
//! `--quick` (135M at 256 rows, one rep of two calls).

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use ferrotensor_backend::{
    wgpu, ComputeEngine, CoopConfig, CoopTile, Epilogue, KernelTiming, MatmulPath, SkinnyTile,
    UnifiedTensor, WgpuContext,
};
use std::time::Instant;

/// Workgroups below which a coop tile is also tried with K split.
const SPLIT_BELOW: u32 = 160;
/// Largest partial-sum buffer a split candidate may allocate.
const MAX_PARTS_BYTES: usize = 512 << 20;
/// LoRA rank of the default fine-tuning recipe.
const RANK: usize = 8;

struct Model {
    name: &'static str,
    hidden: usize,
    kv: usize,
    mlp: usize,
    vocab: usize,
}

const MODELS: [Model; 3] = [
    Model {
        name: "135m",
        hidden: 576,
        kv: 192,
        mlp: 1536,
        vocab: 49152,
    },
    Model {
        name: "360m",
        hidden: 960,
        kv: 320,
        mlp: 2560,
        vocab: 49152,
    },
    Model {
        name: "1.7b",
        hidden: 2048,
        kv: 2048,
        mlp: 8192,
        vocab: 49152,
    },
];

struct Options {
    models: Vec<String>,
    rows: Vec<usize>,
    only: Option<String>,
    tiles: String,
    calls: usize,
    reps: usize,
}

fn options() -> Options {
    let mut o = Options {
        models: MODELS.iter().map(|m| m.name.to_string()).collect(),
        rows: vec![256, 2048],
        only: None,
        tiles: "old".to_string(),
        calls: 10,
        reps: 3,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{arg} takes a value"));
        match arg.as_str() {
            "--models" => o.models = value().split(',').map(str::to_string).collect(),
            "--rows" => {
                o.rows = value()
                    .split(',')
                    .map(|v| v.parse().expect("--rows takes numbers"))
                    .collect();
            }
            "--only" => o.only = Some(value()),
            "--tiles" => o.tiles = value(),
            "--calls" => o.calls = value().parse().expect("--calls takes a number"),
            "--reps" => o.reps = value().parse().expect("--reps takes a number"),
            "--quick" => {
                o.models = vec!["135m".into()];
                o.rows = vec![256];
                o.calls = 2;
                o.reps = 1;
            }
            other => panic!("unknown flag {other}"),
        }
    }
    assert!(
        ["old", "new", "both"].contains(&o.tiles.as_str()),
        "--tiles takes old, new or both"
    );
    assert!(
        o.calls > 0 && o.reps > 0,
        "--calls and --reps must be positive"
    );
    o
}

/// Deterministic values in [-1, 1).
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// One product `out[m, n] = a[m, k] @ op(w)`.
#[derive(Clone, Copy, PartialEq)]
struct Shape {
    op: &'static str,
    transposed_b: bool,
    m: usize,
    k: usize,
    n: usize,
}

impl Shape {
    fn layout(self) -> &'static str {
        if self.transposed_b {
            "a@bT"
        } else {
            "a@b"
        }
    }

    fn flops(self) -> f64 {
        2.0 * self.m as f64 * self.k as f64 * self.n as f64
    }
}

/// The base-weight products of `model` at `m` rows, each `(n, k)` once.
fn coop_shapes(model: &Model, m: usize) -> Vec<Shape> {
    let (h, kv, mlp, vocab) = (model.hidden, model.kv, model.mlp, model.vocab);
    let mut out: Vec<Shape> = Vec::new();
    for (op, out_dim, in_dim) in [
        ("q,o", h, h),
        ("k,v", kv, h),
        ("gate,up", mlp, h),
        ("down", h, mlp),
        ("head", vocab, h),
    ] {
        // The forward reads W [out, in] as b^T; the input gradient reads it
        // as b [K = out, N = in].
        let forward = Shape {
            op,
            transposed_b: true,
            m,
            k: in_dim,
            n: out_dim,
        };
        let input_grad = Shape {
            op,
            transposed_b: false,
            m,
            k: out_dim,
            n: in_dim,
        };
        for s in [forward, input_grad] {
            let key = (s.transposed_b, s.k, s.n);
            if !out.iter().any(|o| (o.transposed_b, o.k, o.n) == key) {
                out.push(s);
            }
        }
    }
    out
}

/// The LoRA products of rank [`RANK`] at `m` rows: K of every adapted width.
fn skinny_shapes(model: &Model, m: usize) -> Vec<Shape> {
    let mut widths = vec![model.hidden, model.kv, model.mlp];
    widths.sort_unstable();
    widths.dedup();
    let mut out = Vec::new();
    for k in widths {
        for transposed_b in [true, false] {
            out.push(Shape {
                op: "lora",
                transposed_b,
                m,
                k,
                n: RANK,
            });
        }
    }
    out
}

struct Bench<'a> {
    ctx: &'a WgpuContext,
    engine: &'a ComputeEngine,
    calls: usize,
    reps: usize,
    /// The coop tiles to try, from `--tiles`.
    tiles: Vec<CoopTile>,
}

impl Bench<'_> {
    /// Median microseconds per call of `run`, and what its first call
    /// returned.
    fn time<T>(&self, mut run: impl FnMut() -> Result<T, String>) -> Result<(f64, T), String> {
        let first = run()?;
        for _ in 0..2 {
            run()?;
        }
        self.ctx.synchronize();
        let mut samples = Vec::new();
        for _ in 0..self.reps {
            let before = self.ctx.kernel_times();
            let start = Instant::now();
            for _ in 0..self.calls {
                run()?;
            }
            self.ctx.synchronize();
            let calls = self.calls as f64;
            let gpu = self.ctx.kernel_times().since(&before).total().as_secs_f64();
            let secs = if self.ctx.kernel_timing_mode().is_some() {
                gpu
            } else {
                start.elapsed().as_secs_f64()
            };
            samples.push(secs * 1e6 / calls);
        }
        Ok((median(samples), first))
    }

    fn read(&self, t: &UnifiedTensor) -> Vec<f32> {
        pollster::block_on(t.to_vec(self.ctx)).expect("GPU readback")
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |m, d| {
            // f32::max drops a NaN; a NaN anywhere is the result.
            if m.is_nan() || d.is_nan() {
                f32::NAN
            } else {
                m.max(d)
            }
        })
}

/// One timed candidate.
struct Timing {
    name: String,
    us: f64,
}

fn report(model: &str, s: Shape, previous: f64, router: &Timing, all: &[Timing]) {
    let best = all
        .iter()
        .min_by(|a, b| a.us.total_cmp(&b.us))
        .expect("at least the router ran");
    println!(
        "| {model} | {} | {} | {} | {} | {} | {previous:.1} | {} | {:.1} | {} | {:.1} | {:.2} |",
        s.m,
        s.op,
        s.layout(),
        s.k,
        s.n,
        router.name,
        router.us,
        best.name,
        best.us,
        s.flops() / (best.us * 1e-6) / 1e12,
    );
}

fn bench_coop(b: &Bench<'_>, model: &str, s: Shape) {
    let ctx = b.ctx;
    let a = UnifiedTensor::new(ctx, &values(s.m * s.k, 1), &[s.m, s.k]);
    let scale = 1.0 / (s.k as f32).sqrt();
    let w_data: Vec<f32> = values(s.n * s.k, 2).iter().map(|v| v * scale).collect();
    let w_shape = if s.transposed_b {
        [s.n, s.k]
    } else {
        [s.k, s.n]
    };
    let w = UnifiedTensor::from_f32_as_f16(ctx, &w_data, &w_shape);
    drop(w_data);
    let out = UnifiedTensor::zeros(ctx, &[s.m, s.n]);
    let t = [&a, &w, &out];
    let (previous, _) = b
        .time(|| {
            b.engine
                .matmul_with_path(ctx, s.transposed_b, &a, &w, &out, MatmulPath::Previous)
        })
        .expect("the previous kernels take every shape");
    let reference = b.read(&out);
    let flops = s.flops();
    let line = |name: &str, us: f64, diff: f32| {
        eprintln!(
            "  {model} m{} {} {} k{} n{}: {name:<24} {us:>10.1} us {:>7.2} TFLOPS  diff {diff:.2e}{}",
            s.m,
            s.op,
            s.layout(),
            s.k,
            s.n,
            flops / (us * 1e-6) / 1e12,
            if diff.is_nan() || diff > 0.05 { "  MISMATCH" } else { "" },
        );
    };
    line("previous", previous, 0.0);
    let name = |c: CoopConfig| format!("{} s{}", c.tile.name(), c.split);
    let Ok((us, picked)) = b.time(|| {
        b.engine
            .matmul_coop_config_into(ctx, s.transposed_b, t, None, Epilogue::STORE)
    }) else {
        eprintln!("  no cooperative-matrix tile runs {}x{}x{}", s.m, s.k, s.n);
        return;
    };
    line("router", us, max_diff(&b.read(&out), &reference));
    let router = Timing {
        name: name(picked),
        us,
    };
    // With `--tiles new` the best column is the best of the new tiles alone.
    let mut all = if b.tiles.iter().any(|t| CoopTile::ALL.contains(t)) {
        vec![Timing {
            name: router.name.clone(),
            us,
        }]
    } else {
        Vec::new()
    };
    let mut seen = vec![picked];
    for &tile in &b.tiles {
        let groups = (s.n as u32).div_ceil(tile.bn) * (s.m as u32).div_ceil(tile.bm);
        let splits: &[u32] = if groups < SPLIT_BELOW {
            &[1, 2, 3, 4, 6, 8]
        } else {
            &[1]
        };
        for &split in splits {
            if split > 1 && s.m * s.n * split as usize * 4 > MAX_PARTS_BYTES {
                continue;
            }
            let config = CoopConfig { tile, split };
            let result = b.time(|| {
                b.engine.matmul_coop_config_into(
                    ctx,
                    s.transposed_b,
                    t,
                    Some(config),
                    Epilogue::STORE,
                )
            });
            let Ok((us, ran)) = result else {
                continue;
            };
            // A split can come out smaller; time each configuration once.
            if seen.contains(&ran) {
                continue;
            }
            seen.push(ran);
            line(&name(ran), us, max_diff(&b.read(&out), &reference));
            all.push(Timing {
                name: name(ran),
                us,
            });
        }
    }
    if all.is_empty() {
        eprintln!("  no new tile runs {}x{}x{}", s.m, s.k, s.n);
        return;
    }
    report(model, s, previous, &router, &all);
}

fn bench_skinny(b: &Bench<'_>, model: &str, s: Shape) {
    let ctx = b.ctx;
    let a = UnifiedTensor::new(ctx, &values(s.m * s.k, 3), &[s.m, s.k]);
    let w_shape = if s.transposed_b {
        [s.n, s.k]
    } else {
        [s.k, s.n]
    };
    let w = UnifiedTensor::new(ctx, &values(s.n * s.k, 4), &w_shape);
    let out = UnifiedTensor::zeros(ctx, &[s.m, s.n]);
    let t = [&a, &w, &out];
    let (previous, _) = b
        .time(|| {
            b.engine
                .matmul_with_path(ctx, s.transposed_b, &a, &w, &out, MatmulPath::Previous)
        })
        .expect("the previous kernels take every shape");
    let reference = b.read(&out);
    let line = |name: &str, us: f64, diff: f32| {
        eprintln!(
            "  {model} m{} lora {} k{} n{}: {name:<10} {us:>8.1} us  diff {diff:.2e}{}",
            s.m,
            s.layout(),
            s.k,
            s.n,
            if diff.is_nan() || diff > 1e-3 {
                "  MISMATCH"
            } else {
                ""
            },
        );
    };
    line("previous", previous, 0.0);
    let name = |t: SkinnyTile| format!("{}x{}", t.rows, t.cols);
    let (us, picked) = b
        .time(|| {
            b.engine
                .matmul_skinny_into(ctx, s.transposed_b, t, None, Epilogue::STORE)
        })
        .expect("the skinny kernel takes every LoRA shape");
    line("router", us, max_diff(&b.read(&out), &reference));
    let router = Timing {
        name: name(picked),
        us,
    };
    let mut all = Vec::new();
    for tile in SkinnyTile::ALL {
        let result = b.time(|| {
            b.engine
                .matmul_skinny_into(ctx, s.transposed_b, t, Some(tile), Epilogue::STORE)
        });
        let Ok((us, _)) = result else {
            continue;
        };
        line(&name(tile), us, max_diff(&b.read(&out), &reference));
        all.push(Timing {
            name: name(tile),
            us,
        });
    }
    report(model, s, previous, &router, &all);
}

fn main() {
    let o = options();
    let ctx = pollster::block_on(WgpuContext::try_new_with_timing(
        wgpu::PowerPreference::HighPerformance,
        false,
        KernelTiming::On,
    ))
    .expect("no GPU adapter");
    let engine = ComputeEngine::new(&ctx);
    eprintln!(
        "adapter: {}; kernel timing: {}; cooperative matrices: {}",
        ctx.adapter_info(),
        ctx.kernel_timing_mode().map_or("off", |m| m.name()),
        engine.coop_matmul_available(),
    );
    let tiles: Vec<CoopTile> = match o.tiles.as_str() {
        "old" => CoopTile::ALL.to_vec(),
        "new" => CoopTile::PIPELINED.to_vec(),
        _ => CoopTile::ALL
            .into_iter()
            .chain(CoopTile::PIPELINED)
            .collect(),
    };
    let bench = Bench {
        ctx: &ctx,
        engine: &engine,
        calls: o.calls,
        reps: o.reps,
        tiles,
    };
    println!(
        "| model | M | op | layout | K | N | previous us | router | router us | best | best us | best TFLOPS |"
    );
    println!("|---|---:|---|---|---:|---:|---:|---|---:|---|---:|---:|");
    let coop = engine.coop_matmul_available() && o.only.as_deref() != Some("skinny");
    let skinny = o.only.as_deref() != Some("coop");
    for model in MODELS
        .iter()
        .filter(|m| o.models.iter().any(|n| n == m.name))
    {
        for &m in &o.rows {
            if skinny {
                for s in skinny_shapes(model, m) {
                    bench_skinny(&bench, model.name, s);
                }
            }
            if coop {
                for s in coop_shapes(model, m) {
                    bench_coop(&bench, model.name, s);
                }
            }
        }
    }
}
