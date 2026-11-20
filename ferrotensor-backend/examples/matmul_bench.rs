//! Times the small-M matmul against the kernels from before it, on the
//! products of LoRA fine-tuning with a micro-batch of 32 to 160 tokens.
//!
//! Run with: `cargo run --release -p ferrotensor-backend --example matmul_bench`
//!
//! Flags: `--quick` runs one small shape and checks that both kernels agree;
//! `--sweep` adds other tiles and split counts; `--calls N` (default 20) and
//! `--reps N` (default 5) set the timing loop: `reps` batches of `calls`
//! back-to-back products, each batch ending with a wait for the GPU.
//!
//! Output: one JSON object per line and kernel run, with `kernel`
//! (`"previous"` or `"small_m"`), `auto` (the kernel `matmul` picks for the
//! shape), `layout` (`"a@bT"`: `b` is `[N, K]`; `"a@b"`: `b` is `[K, N]`),
//! `w` (`"f16"` or `"f32"`), `m`, `n`, `k`, `tile` and `split` (null for
//! `previous`), `ms` (median wall time per product), `gpu_ms` (median
//! kernel GPU time per product, null without timestamp queries), `tflops`
//! (from `ms`) and `max_abs_diff` (against `previous`). A tile the device
//! cannot run gives a line with `error` instead.
//!
//! `--fused` times the base-weight products of a SmolLM2 training step
//! instead (135M, 360M and 1.7B; `--model NAME` keeps one): the fused QKV
//! (and, at 135M, gate and up) forward with a column-split epilogue, their
//! backward over four sources, and the plain products, at 32, 48, 64, 80, 96,
//! 128 and 1024 rows (`--rows A,B,C` sets others). Every product is timed on
//! a context without kernel timestamps (`us`, wall time per product over
//! `--calls` back-to-back products, `--reps` batches, median), then again on
//! one with them for `gpu_us` (the kernels' GPU time) and `reduce_us` (the
//! part of it that is a split-K reduction pass). One JSON object per line:
//! `path` (`fused`; `router`, the kernel `matmul` picks, or `small_m` when
//! it picks that; `coop`, the cooperative kernel whatever the router says),
//! `model`, `op`, `layout`, `m`, `k`, `n`, `tile` and `split` (the
//! cooperative configuration, null for `router`), `us`, `tflops`, `gbs`
//! (weight, `a` and output bytes over `us`), `gpu_us` and `reduce_us`. The
//! fused products and `coop` need `FERROTENSOR_EXPERIMENTAL=1`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use ferrotensor_backend::{
    wgpu, ComputeEngine, CoopConfig, Epilogue, FusedA, FusedOut, FusedSpec, KernelTiming,
    MatmulPath, SmallMLaunch, SmallMTile, UnifiedTensor, WgpuContext,
};
use std::time::Instant;

struct Options {
    quick: bool,
    sweep: bool,
    fused: bool,
    model: Option<String>,
    rows: Vec<usize>,
    calls: usize,
    reps: usize,
}

fn options() -> Options {
    let mut o = Options {
        quick: false,
        sweep: false,
        fused: false,
        model: None,
        rows: vec![32, 48, 64, 80, 96, 128, 1024],
        calls: 20,
        reps: 5,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--quick" => o.quick = true,
            "--sweep" => o.sweep = true,
            "--fused" => o.fused = true,
            "--model" => o.model = Some(args.next().expect("--model takes a size")),
            "--rows" => {
                o.rows = args
                    .next()
                    .expect("--rows takes a comma-separated list")
                    .split(',')
                    .map(|r| r.parse().expect("--rows takes numbers"))
                    .collect();
            }
            "--calls" | "--reps" => {
                let v: usize = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .filter(|&v| v > 0)
                    .unwrap_or_else(|| panic!("{arg} takes a positive number"));
                if arg == "--calls" {
                    o.calls = v;
                } else {
                    o.reps = v;
                }
            }
            other => panic!("unknown flag {other}"),
        }
    }
    if o.quick {
        o.calls = o.calls.min(2);
        o.reps = o.reps.min(1);
        o.rows.truncate(1);
    }
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

/// One product shape: `out[m, n] = a[m, k] @ op(w)`.
#[derive(Clone, Copy)]
struct Shape {
    transposed_b: bool,
    half: bool,
    m: usize,
    n: usize,
    k: usize,
}

fn shapes(quick: bool) -> Vec<Shape> {
    let mut out = Vec::new();
    let mut add = |transposed_b, half, m, n, k| {
        out.push(Shape {
            transposed_b,
            half,
            m,
            n,
            k,
        });
    };
    if quick {
        for transposed_b in [true, false] {
            for half in [true, false] {
                add(transposed_b, half, 64, 320, 288);
            }
        }
        return out;
    }
    for half in [true, false] {
        for m in [32, 64, 96, 128, 160] {
            // Forward: a linear weight [N, K] in Hugging Face layout, and the
            // vocabulary projection.
            for (n, k) in [(2048, 2048), (8192, 2048), (2048, 8192), (49152, 2048)] {
                add(true, half, m, n, k);
            }
            // Input gradient: the same weights read as [K, N].
            for (n, k) in [(2048, 2048), (2048, 8192), (8192, 2048), (2048, 49152)] {
                add(false, half, m, n, k);
            }
        }
    }
    // LoRA rank 8: x @ A^T with A [8, K], f32.
    for k in [2048, 8192] {
        add(true, false, 64, 8, k);
    }
    out
}

/// The tiles `--sweep` adds to the default `d`.
fn sweep_tiles(d: SmallMTile) -> Vec<SmallMTile> {
    let mut tiles = vec![
        // Half the rows per thread, twice the threads along M.
        SmallMTile {
            tm: d.tm.div_ceil(2),
            ty: d.ty * 2,
            ..d
        },
        // 8 columns per thread.
        SmallMTile { tn: 8, ..d },
        // 32 threads along N.
        SmallMTile { tx: 32, ..d },
        // The other K step.
        SmallMTile {
            bk: if d.bk == 32 { 16 } else { 32 },
            ..d
        },
    ];
    tiles.retain(|&t| t != d);
    tiles
}

struct Bench<'a> {
    ctx: &'a WgpuContext,
    engine: &'a ComputeEngine,
    calls: usize,
    reps: usize,
}

impl Bench<'_> {
    /// Median wall and GPU milliseconds per product for `path`, and how the
    /// small-M kernel ran.
    fn time(
        &self,
        s: Shape,
        [a, w, out]: [&UnifiedTensor; 3],
        path: MatmulPath,
    ) -> Result<(f64, Option<f64>, Option<SmallMLaunch>), String> {
        let run = || {
            self.engine
                .matmul_with_path(self.ctx, s.transposed_b, a, w, out, path)
        };
        let launch = run()?;
        for _ in 1..3 {
            run()?;
        }
        self.ctx.synchronize();
        let (mut wall, mut gpu) = (Vec::new(), Vec::new());
        for _ in 0..self.reps {
            let before = self.ctx.kernel_times();
            let start = Instant::now();
            for _ in 0..self.calls {
                run()?;
            }
            self.ctx.synchronize();
            let calls = self.calls as f64;
            wall.push(start.elapsed().as_secs_f64() * 1e3 / calls);
            let kernels = self.ctx.kernel_times().since(&before);
            gpu.push(kernels.total().as_secs_f64() * 1e3 / calls);
        }
        let gpu = self.ctx.kernel_timing_mode().map(|_| median(gpu));
        Ok((median(wall), gpu, launch))
    }
}

fn read(ctx: &WgpuContext, t: &UnifiedTensor) -> Vec<f32> {
    pollster::block_on(t.to_vec(ctx)).expect("GPU readback")
}

fn json_f64(v: Option<f64>) -> String {
    v.filter(|v| v.is_finite())
        .map_or_else(|| "null".to_string(), |v| format!("{v:.6}"))
}

fn main() {
    env_logger::init();
    let o = options();
    if o.fused {
        fused_main(&o);
        return;
    }
    let ctx = pollster::block_on(WgpuContext::try_new_with_timing(
        wgpu::PowerPreference::HighPerformance,
        false,
        KernelTiming::On,
    ))
    .expect("no GPU adapter");
    let engine = ComputeEngine::new(&ctx);
    eprintln!(
        "adapter: {}; kernel timing: {}",
        ctx.adapter_info(),
        ctx.kernel_timing_mode().map_or("off", |m| m.name())
    );
    let bench = Bench {
        ctx: &ctx,
        engine: &engine,
        calls: o.calls,
        reps: o.reps,
    };

    for s in shapes(o.quick) {
        let a = UnifiedTensor::new(&ctx, &values(s.m * s.k, 1), &[s.m, s.k]);
        let w_shape = if s.transposed_b {
            [s.n, s.k]
        } else {
            [s.k, s.n]
        };
        // Scaled so a sum over K stays near 1 and differences read as
        // relative errors.
        let scale = 1.0 / (s.k as f32).sqrt();
        let w_data: Vec<f32> = values(s.n * s.k, 2).iter().map(|v| v * scale).collect();
        let w = if s.half {
            UnifiedTensor::from_f32_as_f16(&ctx, &w_data, &w_shape)
        } else {
            UnifiedTensor::new(&ctx, &w_data, &w_shape)
        };
        drop(w_data);
        let out = UnifiedTensor::zeros(&ctx, &[s.m, s.n]);
        let auto = engine
            .matmul_with_path(&ctx, s.transposed_b, &a, &w, &out, MatmulPath::Auto)
            .expect("the router takes every shape");
        let auto = if auto.is_some() {
            "small_m"
        } else {
            "previous"
        };
        let head = format!(
            "\"auto\":\"{auto}\",\"layout\":\"{}\",\"w\":\"{}\",\"m\":{},\"n\":{},\"k\":{}",
            if s.transposed_b { "a@bT" } else { "a@b" },
            if s.half { "f16" } else { "f32" },
            s.m,
            s.n,
            s.k
        );
        let flops = 2.0 * s.m as f64 * s.n as f64 * s.k as f64;
        let buffers = [&a, &w, &out];

        let (ms, gpu_ms, _) = bench
            .time(s, buffers, MatmulPath::Previous)
            .expect("the previous kernels take every shape");
        let reference = read(&ctx, &out);
        println!(
            "{{\"kernel\":\"previous\",{head},\"tile\":null,\"split\":null,\"ms\":{},\"gpu_ms\":{},\"tflops\":{},\"max_abs_diff\":null}}",
            json_f64(Some(ms)),
            json_f64(gpu_ms),
            json_f64(Some(flops / (ms * 1e-3) / 1e12)),
        );

        let mut paths = vec![MatmulPath::SmallM {
            tile: None,
            split: None,
        }];
        let mut i = 0;
        while i < paths.len() {
            let path = paths[i];
            i += 1;
            let result = bench.time(s, buffers, path);
            let (ms, gpu_ms, launch) = match result {
                Ok((ms, gpu_ms, Some(launch))) => (ms, gpu_ms, launch),
                Ok((_, _, None)) => unreachable!("a small-M path reports its launch"),
                Err(e) => {
                    println!(
                        "{{\"kernel\":\"small_m\",{head},\"error\":\"{}\"}}",
                        e.replace('"', "'")
                    );
                    continue;
                }
            };
            let got = read(&ctx, &out);
            let diff = got
                .iter()
                .zip(&reference)
                .map(|(g, r)| (g - r).abs())
                .fold(0.0f32, f32::max);
            let SmallMTile { tm, tn, tx, ty, bk } = launch.tile;
            println!(
                "{{\"kernel\":\"small_m\",{head},\"tile\":{{\"tm\":{tm},\"tn\":{tn},\"tx\":{tx},\"ty\":{ty},\"bk\":{bk}}},\"split\":{},\"ms\":{},\"gpu_ms\":{},\"tflops\":{},\"max_abs_diff\":{}}}",
                launch.split,
                json_f64(Some(ms)),
                json_f64(gpu_ms),
                json_f64(Some(flops / (ms * 1e-3) / 1e12)),
                json_f64(Some(f64::from(diff))),
            );
            if o.quick {
                assert!(
                    diff <= 1e-3,
                    "small_m differs from previous by {diff} on {head}"
                );
            }
            // The default run picks the tile and split the sweep varies.
            if o.sweep && i == 1 {
                let split = launch.split;
                for tile in sweep_tiles(launch.tile) {
                    paths.push(MatmulPath::SmallM {
                        tile: Some(tile),
                        split: None,
                    });
                }
                for other in [1, split * 2] {
                    if other != split {
                        paths.push(MatmulPath::SmallM {
                            tile: Some(launch.tile),
                            split: Some(other),
                        });
                    }
                }
            }
        }
    }
}

/// How a product of a training step reaches the kernels.
#[derive(Clone)]
enum Op {
    /// `a[m, k] @ op(w)`, one matrix in and one out.
    Plain { transposed_b: bool },
    /// The fused forward `x[m, k] @ w[k, n]` whose columns go to one output
    /// per width (the Q, K, V and LoRA `U` matrices, or gate, up and `U`).
    Split { widths: Vec<usize> },
    /// The fused backward: the matrices of `widths` side by side make `a`
    /// (`k` is their sum), against `w[n, k]`, accumulated into `[m, n]`.
    Sources { widths: Vec<usize> },
}

/// One base-weight product of a training step, at `k` and `n`.
struct Case {
    model: &'static str,
    name: String,
    op: Op,
    k: usize,
    n: usize,
}

/// The products of SmolLM2-135M, 360M and 1.7B (`only` keeps one size) at
/// LoRA rank 8 on every attention projection.
fn cases(only: Option<&str>) -> Vec<Case> {
    const RANK: usize = 8;
    const VOCAB: usize = 49_152;
    // Name, hidden size, query width, key and value width, feed-forward
    // width, and whether gate and up run as one fused product (their summed
    // width is at most 4,096).
    let models = [
        ("135M", 576, 576, 192, 1536, true),
        ("360M", 960, 960, 320, 2560, false),
        ("1.7B", 2048, 2048, 2048, 8192, false),
    ];
    let mut out = Vec::new();
    for (model, hidden, q, kv, inter, gate_up) in models {
        if only.is_some_and(|o| o != model) {
            continue;
        }
        let mut add = |name: &str, op: Op, k: usize, n: usize| {
            out.push(Case {
                model,
                name: name.to_string(),
                op,
                k,
                n,
            });
        };
        let u = 3 * RANK;
        let qkv = q + 2 * kv + u;
        add(
            "qkv fwd",
            Op::Split {
                widths: vec![q, kv, kv, u],
            },
            hidden,
            qkv,
        );
        add(
            "qkv dX",
            Op::Sources {
                widths: vec![q, kv, kv, u],
            },
            qkv,
            hidden,
        );
        if gate_up {
            let gu = 2 * inter + 2 * RANK;
            let widths = vec![inter, inter, 2 * RANK];
            add(
                "gate_up fwd",
                Op::Split {
                    widths: widths.clone(),
                },
                hidden,
                gu,
            );
            add("gate_up dX", Op::Sources { widths }, gu, hidden);
        }
        // The weight is stored [N, K] of the forward (Hugging Face layout);
        // the input gradient reads it as [K, N] with the sizes swapped.
        for (name, k, n) in [
            ("o", q, hidden),
            ("gate", hidden, inter),
            ("down", inter, hidden),
        ] {
            add(
                &format!("{name} fwd"),
                Op::Plain { transposed_b: true },
                k,
                n,
            );
            add(
                &format!("{name} dX"),
                Op::Plain {
                    transposed_b: false,
                },
                n,
                k,
            );
        }
        add("head fwd", Op::Plain { transposed_b: true }, hidden, VOCAB);
        add(
            "head dX",
            Op::Plain {
                transposed_b: false,
            },
            VOCAB,
            hidden,
        );
    }
    out
}

/// Which kernel a case runs on.
#[derive(Clone, Copy, PartialEq)]
enum Variant {
    /// The fused cooperative kernel with the fused router's configuration.
    Fused,
    /// What `matmul` picks for a plain product.
    Router,
    /// The cooperative kernel for a plain product, whatever `matmul` picks.
    Coop,
}

struct Rig {
    ctx: WgpuContext,
    engine: ComputeEngine,
}

fn rig(timing: KernelTiming) -> Rig {
    let ctx = pollster::block_on(WgpuContext::try_new_with_timing(
        wgpu::PowerPreference::HighPerformance,
        false,
        timing,
    ))
    .expect("no GPU adapter");
    let engine = ComputeEngine::new(&ctx);
    Rig { ctx, engine }
}

/// Runs a fused product on the cooperative kernel and returns the
/// configuration it ran with. A tree from before the fused products took a
/// configuration has only `matmul_fused_coop_into`, which returns nothing: to
/// time that tree, call it here and return `Ok(None)`.
fn run_fused(
    engine: &ComputeEngine,
    ctx: &WgpuContext,
    spec: &FusedSpec<'_>,
) -> Result<Option<CoopConfig>, String> {
    engine
        .matmul_fused_coop_config_into(ctx, spec, None)
        .map(Some)
}

/// What one timed case measured.
struct Run {
    path: &'static str,
    config: Option<CoopConfig>,
    /// Median wall microseconds per product.
    us: f64,
    /// Median kernel GPU microseconds per product, and the part of them in
    /// split-K reduction passes; null without timestamp queries.
    gpu_us: Option<f64>,
    reduce_us: Option<f64>,
}

fn measure(rig: &Rig, c: &Case, m: usize, variant: Variant, o: &Options) -> Result<Run, String> {
    let (ctx, engine) = (&rig.ctx, &rig.engine);
    let (transposed_b, w_shape) = match &c.op {
        Op::Plain { transposed_b } => (
            *transposed_b,
            if *transposed_b {
                [c.n, c.k]
            } else {
                [c.k, c.n]
            },
        ),
        Op::Split { .. } => (false, [c.k, c.n]),
        Op::Sources { .. } => (true, [c.n, c.k]),
    };
    let scale = 1.0 / (c.k as f32).sqrt();
    let w_data: Vec<f32> = values(c.n * c.k, 2).iter().map(|v| v * scale).collect();
    let w = UnifiedTensor::from_f32_as_f16(ctx, &w_data, &w_shape);
    drop(w_data);
    let a_widths = match &c.op {
        Op::Sources { widths } => widths.clone(),
        _ => vec![c.k],
    };
    let a_parts: Vec<UnifiedTensor> = a_widths
        .iter()
        .zip(1u64..)
        .map(|(&w, seed)| UnifiedTensor::new(ctx, &values(m * w, seed), &[m, w]))
        .collect();
    let a_refs: Vec<&UnifiedTensor> = a_parts.iter().collect();
    let out_widths = match &c.op {
        Op::Split { widths } => widths.clone(),
        _ => vec![c.n],
    };
    let outs: Vec<UnifiedTensor> = out_widths
        .iter()
        .map(|&w| UnifiedTensor::zeros(ctx, &[m, w]))
        .collect();
    let out_refs: Vec<&UnifiedTensor> = outs.iter().collect();
    let run = || -> Result<(&'static str, Option<CoopConfig>), String> {
        match (variant, &c.op) {
            (Variant::Fused, Op::Split { .. }) => {
                let spec = FusedSpec {
                    a: FusedA::Plain(&a_parts[0]),
                    b: &w,
                    transposed_b,
                    out: FusedOut::Split {
                        outs: &out_refs,
                        bias: None,
                        alpha: 1.0,
                    },
                };
                run_fused(engine, ctx, &spec).map(|config| ("fused", config))
            }
            (Variant::Fused, Op::Sources { .. }) => {
                let spec = FusedSpec {
                    a: FusedA::Sources(&a_refs),
                    b: &w,
                    transposed_b,
                    out: FusedOut::AlphaBeta {
                        out: &outs[0],
                        // The backward accumulates into the gradient.
                        epilogue: Epilogue {
                            alpha: 1.0,
                            beta: 1.0,
                        },
                    },
                };
                run_fused(engine, ctx, &spec).map(|config| ("fused", config))
            }
            (Variant::Router, Op::Plain { .. }) => engine
                .matmul_with_path(
                    ctx,
                    transposed_b,
                    &a_parts[0],
                    &w,
                    &outs[0],
                    MatmulPath::Auto,
                )
                .map(|launch| {
                    (
                        if launch.is_some() {
                            "small_m"
                        } else {
                            "router"
                        },
                        None,
                    )
                }),
            (Variant::Coop, Op::Plain { .. }) => engine
                .matmul_coop_config_into(
                    ctx,
                    transposed_b,
                    [&a_parts[0], &w, &outs[0]],
                    None,
                    Epilogue::STORE,
                )
                .map(|config| ("coop", Some(config))),
            _ => Err("that kernel does not run this product".to_string()),
        }
    };
    let (path, config) = run()?;
    for _ in 1..3 {
        run()?;
    }
    ctx.synchronize();
    let (mut wall, mut gpu, mut reduce) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..o.reps {
        let before = ctx.kernel_times();
        let start = Instant::now();
        for _ in 0..o.calls {
            run()?;
        }
        ctx.synchronize();
        let calls = o.calls as f64;
        wall.push(start.elapsed().as_secs_f64() * 1e6 / calls);
        let kernels = ctx.kernel_times().since(&before);
        gpu.push(kernels.total().as_secs_f64() * 1e6 / calls);
        let in_reduce: f64 = kernels
            .by_gpu_time()
            .iter()
            .filter(|(name, _)| {
                name.starts_with("fused_reduce") || name.starts_with("matmul_split_reduce")
            })
            .map(|(_, t)| t.gpu_time.as_secs_f64())
            .sum();
        reduce.push(in_reduce * 1e6 / calls);
    }
    let timed = ctx.kernel_timing_mode().is_some();
    Ok(Run {
        path,
        config,
        us: median(wall),
        gpu_us: timed.then(|| median(gpu)),
        reduce_us: timed.then(|| median(reduce)),
    })
}

/// The `--fused` mode: every case at every row count, on every kernel that
/// runs it.
fn fused_main(o: &Options) {
    // Wall times come from a context without timestamp queries, which
    // inflate short kernels; the GPU times from one with them.
    let wall = rig(KernelTiming::Off);
    let timed = rig(KernelTiming::On);
    eprintln!(
        "adapter: {}; kernel timing: {}",
        wall.ctx.adapter_info(),
        timed.ctx.kernel_timing_mode().map_or("off", |m| m.name())
    );
    for c in cases(o.model.as_deref()) {
        let variants: &[Variant] = match c.op {
            Op::Plain { .. } => &[Variant::Router, Variant::Coop],
            _ => &[Variant::Fused],
        };
        for &m in &o.rows {
            for &variant in variants {
                let head =
                    format!(
                    "\"model\":\"{}\",\"op\":\"{}\",\"layout\":\"{}\",\"m\":{m},\"k\":{},\"n\":{}",
                    c.model,
                    c.name,
                    match &c.op {
                        Op::Plain { transposed_b: false } | Op::Split { .. } => "a@b",
                        _ => "a@bT",
                    },
                    c.k,
                    c.n
                );
                let run = match measure(&wall, &c, m, variant, o) {
                    Ok(run) => run,
                    Err(e) => {
                        println!("{{{head},\"error\":\"{}\"}}", e.replace('"', "'"));
                        continue;
                    }
                };
                let counted = measure(&timed, &c, m, variant, o).ok();
                let flops = 2.0 * m as f64 * c.n as f64 * c.k as f64;
                // Weight halves, `a` and the output in f32.
                let bytes =
                    2.0 * (c.n * c.k) as f64 + 4.0 * (m * c.k) as f64 + 4.0 * (m * c.n) as f64;
                let (tile, split) = run.config.map_or_else(
                    || ("null".to_string(), "null".to_string()),
                    |cfg| (format!("\"{}\"", cfg.tile.name()), cfg.split.to_string()),
                );
                println!(
                    "{{\"path\":\"{}\",{head},\"tile\":{tile},\"split\":{split},\"us\":{},\"tflops\":{},\"gbs\":{},\"gpu_us\":{},\"reduce_us\":{}}}",
                    run.path,
                    json_f64(Some(run.us)),
                    json_f64(Some(flops / (run.us * 1e-6) / 1e12)),
                    json_f64(Some(bytes / (run.us * 1e-6) / 1e9)),
                    json_f64(counted.as_ref().and_then(|r| r.gpu_us)),
                    json_f64(counted.as_ref().and_then(|r| r.reduce_us)),
                );
            }
        }
    }
}
