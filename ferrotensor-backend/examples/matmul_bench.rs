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

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use ferrotensor_backend::{
    wgpu, ComputeEngine, KernelTiming, MatmulPath, SmallMLaunch, SmallMTile, UnifiedTensor,
    WgpuContext,
};
use std::time::Instant;

struct Options {
    quick: bool,
    sweep: bool,
    calls: usize,
    reps: usize,
}

fn options() -> Options {
    let mut o = Options {
        quick: false,
        sweep: false,
        calls: 20,
        reps: 5,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--quick" => o.quick = true,
            "--sweep" => o.sweep = true,
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
