//! Times the fused cooperative-matrix products against the separate kernels
//! they replace, on the projections of SmolLM2-135M and SmolLM2-1.7B.
//!
//! Run with:
//! `FERROTENSOR_EXPERIMENTAL=1 cargo run --release -p ferrotensor-backend --example fused_bench`
//!
//! Flags: `--quick` runs one small model, one batch and one repetition;
//! `--calls N` (default 20) and `--reps N` (default 5) set the timing loop:
//! `reps` batches of `calls` back-to-back runs, each batch ending with a wait
//! for the GPU. Without f16 cooperative matrices it says so and exits.
//!
//! Output: one JSON object per line and variant, with `model`, `case`, `m`
//! (rows: tokens per step), `variant` (`"separate"`: the kernels the fused
//! product replaces; `"fused"`), `ms` (median wall time per run), `gpu_ms`
//! (median kernel GPU time per run, null without timestamp queries) and
//! `kernels` (kernel launches per run, 0 without timestamp queries).
//!
//! The cases are the fused QKV projection (with and without the three LoRA
//! `A` matrices folded in as columns), gate and up, the down projection that
//! reads `silu(gate) * up` and adds its residual, the backward `[dQ | dK |
//! dV | dU] @ W^T` over four sources, and the down projection's backward.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use ferrotensor_backend::{
    wgpu, ComputeEngine, Epilogue, FusedA, FusedOut, FusedSpec, KernelTiming, UnifiedTensor,
    WgpuContext,
};
use std::time::Instant;

struct Options {
    quick: bool,
    calls: usize,
    reps: usize,
}

fn options() -> Options {
    let mut o = Options {
        quick: false,
        calls: 20,
        reps: 5,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--quick" => o.quick = true,
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
        o.reps = 1;
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

/// The sizes of one model's projections.
struct Model {
    name: &'static str,
    hidden: usize,
    q: usize,
    kv: usize,
    inter: usize,
}

/// LoRA rank of the folded `A` matrices.
const RANK: usize = 16;

const MODELS: [Model; 2] = [
    Model {
        name: "135M",
        hidden: 576,
        q: 576,
        kv: 192,
        inter: 1536,
    },
    Model {
        name: "1.7B",
        hidden: 2048,
        q: 2048,
        kv: 2048,
        inter: 8192,
    },
];

struct Bench<'a> {
    ctx: &'a WgpuContext,
    engine: &'a ComputeEngine,
    calls: usize,
    reps: usize,
}

type Run<'r> = &'r dyn Fn() -> Result<(), String>;

impl Bench<'_> {
    fn f32(&self, n: usize, shape: &[usize], seed: u64) -> UnifiedTensor {
        UnifiedTensor::new(self.ctx, &values(n, seed), shape)
    }

    fn f16(&self, shape: &[usize], seed: u64) -> UnifiedTensor {
        let n = shape.iter().product();
        UnifiedTensor::from_f32_as_f16(self.ctx, &values(n, seed), shape)
    }

    fn zeros(&self, shape: &[usize]) -> UnifiedTensor {
        UnifiedTensor::zeros(self.ctx, shape)
    }

    /// Median wall and GPU milliseconds per run of `run`, and the kernels it
    /// launched.
    fn time(&self, run: Run<'_>) -> Result<(f64, Option<f64>, u64), String> {
        for _ in 0..3 {
            run()?;
        }
        self.ctx.synchronize();
        let (mut wall, mut gpu, mut kernels) = (Vec::new(), Vec::new(), 0u64);
        for _ in 0..self.reps {
            let before = self.ctx.kernel_times();
            let start = Instant::now();
            for _ in 0..self.calls {
                run()?;
            }
            self.ctx.synchronize();
            let calls = self.calls as f64;
            wall.push(start.elapsed().as_secs_f64() * 1e3 / calls);
            let total = self.ctx.kernel_times().since(&before);
            gpu.push(total.total().as_secs_f64() * 1e3 / calls);
            kernels = total
                .by_gpu_time()
                .iter()
                .map(|(_, t)| t.dispatches)
                .sum::<u64>()
                / self.calls as u64;
        }
        let gpu = self.ctx.kernel_timing_mode().map(|_| median(gpu));
        Ok((median(wall), gpu, kernels))
    }

    fn report(&self, model: &str, case: &str, m: usize, variant: &str, run: Run<'_>) {
        match self.time(run) {
            Ok((ms, gpu_ms, kernels)) => println!(
                "{{\"model\":\"{model}\",\"case\":\"{case}\",\"m\":{m},\"variant\":\"{variant}\",\
                 \"ms\":{ms:.6},\"gpu_ms\":{},\"kernels\":{kernels}}}",
                gpu_ms.map_or_else(|| "null".to_string(), |v| format!("{v:.6}"))
            ),
            Err(e) => println!(
                "{{\"model\":\"{model}\",\"case\":\"{case}\",\"m\":{m},\"variant\":\"{variant}\",\
                 \"error\":{e:?}}}"
            ),
        }
    }

    fn model(&self, md: &Model, m: usize) {
        let (h, q, kv, inter) = (md.hidden, md.q, md.kv, md.inter);
        let name = md.name;
        let eng = self.engine;
        let ctx = self.ctx;
        let store = Epilogue::STORE;
        let x = self.f32(m * h, &[m, h], 1);

        // Fused QKV: one product against [Wq | Wk | Wv], three outputs, with
        // the bias of the Qwen2 projections.
        {
            let wq = self.f16(&[h, q], 2);
            let wk = self.f16(&[h, kv], 3);
            let wv = self.f16(&[h, kv], 4);
            let (oq, ok, ov) = (
                self.zeros(&[m, q]),
                self.zeros(&[m, kv]),
                self.zeros(&[m, kv]),
            );
            let w = self.f16(&[h, q + 2 * kv], 5);
            let bias = self.f32(q + 2 * kv, &[q + 2 * kv], 6);
            self.report(name, "qkv", m, "separate", &|| {
                for (w, o) in [(&wq, &oq), (&wk, &ok), (&wv, &ov)] {
                    eng.matmul_coop_into(ctx, false, [&x, w, o], store)?;
                }
                Ok(())
            });
            self.report(name, "qkv", m, "fused", &|| {
                eng.matmul_fused_coop_into(
                    ctx,
                    &FusedSpec {
                        a: FusedA::Plain(&x),
                        b: &w,
                        transposed_b: false,
                        out: FusedOut::Split {
                            outs: &[&oq, &ok, &ov],
                            bias: Some(&bias),
                            alpha: 1.0,
                        },
                    },
                )
            });
            // With the three A matrices as more columns, whose product `U`
            // the separate path would compute in three more kernels.
            let a_cols = 3 * RANK;
            let wa = self.f16(&[h, q + 2 * kv + a_cols], 7);
            let u = self.zeros(&[m, a_cols]);
            let wa_each = [
                self.f16(&[h, RANK], 8),
                self.f16(&[h, RANK], 9),
                self.f16(&[h, RANK], 10),
            ];
            let u_each = [
                self.zeros(&[m, RANK]),
                self.zeros(&[m, RANK]),
                self.zeros(&[m, RANK]),
            ];
            self.report(name, "qkv_lora_a", m, "separate", &|| {
                for (w, o) in [(&wq, &oq), (&wk, &ok), (&wv, &ov)] {
                    eng.matmul_coop_into(ctx, false, [&x, w, o], store)?;
                }
                for (w, o) in wa_each.iter().zip(&u_each) {
                    eng.matmul_coop_into(ctx, false, [&x, w, o], store)?;
                }
                Ok(())
            });
            self.report(name, "qkv_lora_a", m, "fused", &|| {
                eng.matmul_fused_coop_into(
                    ctx,
                    &FusedSpec {
                        a: FusedA::Plain(&x),
                        b: &wa,
                        transposed_b: false,
                        out: FusedOut::Split {
                            outs: &[&oq, &ok, &ov, &u],
                            bias: None,
                            alpha: 1.0,
                        },
                    },
                )
            });
        }

        // Gate and up: one product with two outputs.
        {
            let wg = self.f16(&[h, inter], 11);
            let wu = self.f16(&[h, inter], 12);
            let (og, ou) = (self.zeros(&[m, inter]), self.zeros(&[m, inter]));
            let w = self.f16(&[h, 2 * inter], 13);
            self.report(name, "gate_up", m, "separate", &|| {
                eng.matmul_coop_into(ctx, false, [&x, &wg, &og], store)?;
                eng.matmul_coop_into(ctx, false, [&x, &wu, &ou], store)
            });
            self.report(name, "gate_up", m, "fused", &|| {
                eng.matmul_fused_coop_into(
                    ctx,
                    &FusedSpec {
                        a: FusedA::Plain(&x),
                        b: &w,
                        transposed_b: false,
                        out: FusedOut::Split {
                            outs: &[&og, &ou],
                            bias: None,
                            alpha: 1.0,
                        },
                    },
                )
            });
        }

        // Down projection: silu(gate) * up, the product and the residual add.
        {
            let gate = self.f32(m * inter, &[m, inter], 14);
            let up = self.f32(m * inter, &[m, inter], 15);
            let gu = self.f32(m * 2 * inter, &[m, 2 * inter], 16);
            let wd = self.f16(&[inter, h], 17);
            let residual = self.f32(m * h, &[m, h], 18);
            let (act, prod, out) = (
                self.zeros(&[m, inter]),
                self.zeros(&[m, h]),
                self.zeros(&[m, h]),
            );
            self.report(name, "down_swiglu_residual", m, "separate", &|| {
                eng.silu_gate_fused(ctx, &gate, &up, &act);
                eng.matmul_coop_into(ctx, false, [&act, &wd, &prod], store)?;
                eng.add(ctx, &prod, &residual, &out);
                Ok(())
            });
            self.report(name, "down_swiglu_residual", m, "fused", &|| {
                eng.matmul_fused_coop_into(
                    ctx,
                    &FusedSpec {
                        a: FusedA::SwiGlu(&gu),
                        b: &wd,
                        transposed_b: false,
                        out: FusedOut::Residual {
                            out: &out,
                            residual: &residual,
                            alpha: 1.0,
                        },
                    },
                )
            });
        }

        // QKV backward: dX = [dQ | dK | dV | dU] @ W^T, four sources against
        // one weight, against four products accumulating into dX.
        {
            let widths = [q, kv, kv, 3 * RANK];
            let k: usize = widths.iter().sum();
            let grads: Vec<UnifiedTensor> = widths
                .iter()
                .enumerate()
                .map(|(i, &w)| self.f32(m * w, &[m, w], 20 + i as u64))
                .collect();
            let sources: Vec<&UnifiedTensor> = grads.iter().collect();
            let w = self.f16(&[h, k], 30);
            let parts: Vec<UnifiedTensor> = widths
                .iter()
                .enumerate()
                .map(|(i, &wd)| self.f16(&[h, wd], 31 + i as u64))
                .collect();
            let dx = self.zeros(&[m, h]);
            self.report(name, "qkv_backward", m, "separate", &|| {
                for (i, (g, p)) in grads.iter().zip(&parts).enumerate() {
                    let beta = if i == 0 { 0.0 } else { 1.0 };
                    eng.matmul_coop_into(ctx, true, [g, p, &dx], Epilogue { alpha: 1.0, beta })?;
                }
                Ok(())
            });
            self.report(name, "qkv_backward", m, "fused", &|| {
                eng.matmul_fused_coop_into(
                    ctx,
                    &FusedSpec {
                        a: FusedA::Sources(&sources),
                        b: &w,
                        transposed_b: true,
                        out: FusedOut::AlphaBeta {
                            out: &dx,
                            epilogue: store,
                        },
                    },
                )
            });
        }

        // Down projection backward: dH = dY @ W^T, then the gradients of the
        // gate and up outputs.
        {
            let dy = self.f32(m * h, &[m, h], 40);
            let wd = self.f16(&[inter, h], 41);
            let gate = self.f32(m * inter, &[m, inter], 42);
            let up = self.f32(m * inter, &[m, inter], 43);
            let gu = self.f32(m * 2 * inter, &[m, 2 * inter], 44);
            let (dh, d_gate, d_up, d_gu) = (
                self.zeros(&[m, inter]),
                self.zeros(&[m, inter]),
                self.zeros(&[m, inter]),
                self.zeros(&[m, 2 * inter]),
            );
            self.report(name, "swiglu_backward", m, "separate", &|| {
                eng.matmul_coop_into(ctx, true, [&dy, &wd, &dh], store)?;
                eng.silu_gate_backward(ctx, &up, &gate, &dh, &d_up, &d_gate);
                Ok(())
            });
            self.report(name, "swiglu_backward", m, "fused", &|| {
                eng.matmul_fused_coop_into(
                    ctx,
                    &FusedSpec {
                        a: FusedA::Plain(&dy),
                        b: &wd,
                        transposed_b: true,
                        out: FusedOut::SwiGluBackward {
                            d_gu: &d_gu,
                            gu: &gu,
                        },
                    },
                )
            });
        }
    }
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
    if !engine.coop_matmul_available() {
        eprintln!(
            "fused_bench: this device has no f16 cooperative matrices (or FERROTENSOR_EXPERIMENTAL \
             is not 1); nothing to time"
        );
        return;
    }
    let bench = Bench {
        ctx: &ctx,
        engine: &engine,
        calls: o.calls,
        reps: o.reps,
    };
    let (models, rows): (&[Model], &[usize]) = if o.quick {
        (&MODELS[..1], &[256])
    } else {
        (&MODELS, &[256, 1024])
    };
    for model in models {
        for &m in rows {
            bench.model(model, m);
        }
    }
}
