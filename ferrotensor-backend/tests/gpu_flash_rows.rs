//! Flash attention with 32-row and 64-row workgroups, and with RoPE applied
//! inside the kernels, against CPU references.
//!
//! The cooperative kernels need a GPU with f16 cooperative matrices (16x16x16,
//! f32 accumulation), subgroups of 32 threads and `FERROTENSOR_EXPERIMENTAL=1`;
//! without them those tests print why they skipped and pass. The RoPE
//! fallback test runs on any GPU. Run them with:
//!
//! ```text
//! FERROTENSOR_EXPERIMENTAL=1 cargo test -p ferrotensor-backend --test gpu_flash_rows -- --ignored --test-threads=1
//! ```

#![allow(clippy::unwrap_used, clippy::panic, clippy::too_many_arguments)]

mod common;

use common::attention::{cpu_attention, cpu_attention_backward, cpu_attention_lse};
use ferrotensor_backend::{AttentionOptions, ComputeEngine, RopeFreqs, UnifiedTensor, WgpuContext};
use std::sync::OnceLock;

const HEAD: usize = 64;

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

/// The shared GPU with the attention kernels switched on; prints why the
/// test skips if the device lacks them.
fn flash_gpu(test: &str) -> Option<&'static Gpu> {
    let g = gpu();
    if g.engine.flash_coop_available() {
        g.engine.set_flash_coop(true);
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

fn upload(g: &Gpu, data: &[f32], shape: &[usize]) -> UnifiedTensor {
    UnifiedTensor::new(&g.ctx, data, shape)
}

fn zeros(g: &Gpu, shape: &[usize]) -> UnifiedTensor {
    UnifiedTensor::zeros(&g.ctx, shape)
}

fn read(g: &Gpu, t: &UnifiedTensor) -> Vec<f32> {
    pollster::block_on(t.to_vec(&g.ctx)).unwrap()
}

fn max_abs(x: &[f32]) -> f32 {
    x.iter().fold(0.0, |m, v| m.max(v.abs()))
}

#[track_caller]
fn assert_near(name: &str, got: &[f32], want: &[f32], (atol, rtol): (f32, f32)) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = atol + rtol * w.abs();
        assert!(
            g.is_finite() && (g - w).abs() <= tol,
            "{name}: index {i}: got {g}, want {w}, tol {tol}"
        );
    }
}

/// An attention call: batch, query and key lengths, heads, KV heads and
/// causal masking. The head size is 64.
#[derive(Clone, Copy)]
struct Case {
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hkv: usize,
    causal: bool,
}

impl Case {
    fn dims(&self) -> (usize, usize, usize, usize, usize, usize) {
        (self.b, self.sq, self.sk, self.h, self.hkv, HEAD)
    }

    fn visible(&self) -> impl Fn(usize, usize) -> bool {
        let causal = self.causal;
        move |i, j| !causal || j <= i
    }

    fn options(&self) -> AttentionOptions {
        AttentionOptions {
            causal: self.causal,
            ..AttentionOptions::default()
        }
    }

    fn name(&self) -> String {
        format!(
            "b{} q{} k{} h{}/{} causal={}",
            self.b, self.sq, self.sk, self.h, self.hkv, self.causal
        )
    }

    /// f16-representable q, k, v and dO.
    fn inputs(&self, seed: u64) -> [Vec<f32>; 4] {
        let q = self.b * self.sq * self.h * HEAD;
        let kv = self.b * self.sk * self.hkv * HEAD;
        [
            f16_round(&rand_vec(q, seed)),
            f16_round(&rand_vec(kv, seed + 1)),
            f16_round(&rand_vec(kv, seed + 2)),
            f16_round(&rand_vec(q, seed + 3)),
        ]
    }
}

const fn case(b: usize, (sq, sk): (usize, usize), (h, hkv): (usize, usize), causal: bool) -> Case {
    Case {
        b,
        sq,
        sk,
        h,
        hkv,
        causal,
    }
}

/// Lengths around the 16-row blocks and the 32- and 64-row tiles (1 to 17,
/// 33, 65, 255, 256), GQA with groups of 3, 3 and 1 (9/3, 15/5, 32/32 heads),
/// and non-causal calls with other key lengths.
const CASES: &[Case] = &[
    case(1, (17, 17), (2, 2), true),
    case(1, (33, 33), (2, 2), true),
    case(1, (65, 65), (3, 3), true),
    case(1, (255, 255), (9, 3), true),
    case(1, (256, 256), (9, 3), true),
    case(2, (100, 100), (15, 5), true),
    case(1, (130, 130), (32, 32), true),
    case(1, (192, 192), (3, 3), false),
    case(1, (70, 150), (2, 1), false),
];

/// What a call returns: the output, the log-sum-exp of every row, and the
/// gradients of q, k and v.
struct Outputs {
    out: Vec<f32>,
    lse: Vec<f32>,
    grads: [Vec<f32>; 3],
}

/// Runs forward and backward on the engine as it is set up.
fn run(g: &Gpu, c: &Case, [q, k, v, dout]: &[Vec<f32>; 4]) -> Outputs {
    let (b, sq, sk, h, hkv) = (c.b, c.sq, c.sk, c.h, c.hkv);
    let (tq, tk, tv, tdo) = (
        upload(g, q, &[b, sq, h, HEAD]),
        upload(g, k, &[b, sk, hkv, HEAD]),
        upload(g, v, &[b, sk, hkv, HEAD]),
        upload(g, dout, &[b, sq, h, HEAD]),
    );
    let opts = c.options();
    let out = zeros(g, &[b, sq, h, HEAD]);
    let stats = zeros(g, &[b, h, sq, 2]);
    g.engine
        .flash_attention_v2_with_stats(&g.ctx, &tq, &tk, &tv, &out, &stats, &opts, None);
    let lse = read(g, &stats).iter().skip(1).step_by(2).copied().collect();
    let (gq, gk, gv) = (
        zeros(g, &[b, sq, h, HEAD]),
        zeros(g, &[b, sk, hkv, HEAD]),
        zeros(g, &[b, sk, hkv, HEAD]),
    );
    g.engine.flash_attention_v2_backward_with_stats(
        &g.ctx, &tq, &tk, &tv, &out, &tdo, &gq, &gk, &gv, &stats, &opts, None,
    );
    Outputs {
        out: read(g, &out),
        lse,
        grads: [read(g, &gq), read(g, &gk), read(g, &gv)],
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn flash_rows_32_and_64_agree_with_each_other_and_with_cpu() {
    let Some(g) = flash_gpu("flash_rows_32_and_64_agree_with_each_other_and_with_cpu") else {
        return;
    };
    let overflows = || pollster::block_on(g.engine.coop_f16_overflows(&g.ctx)).unwrap();
    let before = overflows();
    for (n, c) in CASES.iter().enumerate() {
        let name = c.name();
        let inputs = c.inputs(500 + 4 * n as u64);
        let [q, k, v, dout] = &inputs;
        let (want, probs) = cpu_attention(q, k, v, None, c.dims(), c.visible());
        let (want_dq, want_dk, want_dv) = cpu_attention_backward(q, k, v, dout, &probs, c.dims());
        let want_lse = cpu_attention_lse(q, k, None, c.dims(), c.visible());

        let mut by_rows = Vec::new();
        for rows in [32u32, 64] {
            g.engine.set_flash_rows(rows);
            let runs = g.engine.flash_coop_runs();
            let narrow = g.engine.flash_coop_narrow_runs();
            let got = run(g, c, &inputs);
            assert_eq!(
                g.engine.flash_coop_runs(),
                (runs.0 + 1, runs.1 + 1),
                "{name}: routed"
            );
            let narrow_now = g.engine.flash_coop_narrow_runs();
            let step = u64::from(rows == 32);
            assert_eq!(
                narrow_now,
                (narrow.0 + step, narrow.1 + step),
                "{name}: {rows} rows"
            );

            assert_near(
                &format!("fwd {name} rows {rows}"),
                &got.out,
                &want,
                (2e-3, 1e-2),
            );
            assert_near(
                &format!("lse {name} rows {rows}"),
                &got.lse,
                &want_lse,
                (5e-4, 1e-4),
            );
            for (label, got, want) in [
                ("dq", &got.grads[0], &want_dq),
                ("dk", &got.grads[1], &want_dk),
                ("dv", &got.grads[2], &want_dv),
            ] {
                let atol = 0.01 * max_abs(want);
                assert_near(
                    &format!("{label} {name} rows {rows}"),
                    got,
                    want,
                    (atol, 2e-2),
                );
            }
            by_rows.push(got);
        }
        // The same f16 products in the same order: the layouts agree closely.
        let (narrow, wide) = (&by_rows[0], &by_rows[1]);
        assert_near(
            &format!("out 32 vs 64 {name}"),
            &narrow.out,
            &wide.out,
            (1e-3, 1e-3),
        );
        assert_near(
            &format!("lse 32 vs 64 {name}"),
            &narrow.lse,
            &wide.lse,
            (1e-3, 1e-3),
        );
        for (label, a, b) in [
            ("dq", &narrow.grads[0], &wide.grads[0]),
            ("dk", &narrow.grads[1], &wide.grads[1]),
            ("dv", &narrow.grads[2], &wide.grads[2]),
        ] {
            let atol = 1e-3 * max_abs(b).max(1.0);
            assert_near(&format!("{label} 32 vs 64 {name}"), a, b, (atol, 1e-3));
        }
    }
    g.engine.set_flash_rows(0);
    assert_eq!(overflows(), before, "finite inputs count no f16 overflow");
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn flash_picks_32_rows_only_when_64_would_leave_sms_idle() {
    let Some(g) = flash_gpu("flash_picks_32_rows_only_when_64_would_leave_sms_idle") else {
        return;
    };
    g.engine.set_flash_rows(0);
    // 1 x 9 heads x 4 tiles of 64 is 36 workgroups: 32 rows. 2 x 15 x 4 is
    // 120: 32 rows. 8 x 9 x 4 is 288, past the 160 that fill the device: 64.
    for (c, narrow) in [
        (case(1, (256, 256), (9, 3), true), true),
        (case(2, (256, 256), (15, 5), true), true),
        (case(8, (256, 256), (9, 3), true), false),
    ] {
        let name = c.name();
        let inputs = c.inputs(700);
        let before = g.engine.flash_coop_narrow_runs();
        run(g, &c, &inputs);
        let after = g.engine.flash_coop_narrow_runs();
        let step = u64::from(narrow);
        assert_eq!(after, (before.0 + step, before.1 + step), "{name}");
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn a_single_query_ignores_the_row_size() {
    let g = gpu();
    let (sq, sk, h) = (1, 1, 2);
    let q = rand_vec(sq * h * HEAD, 11);
    let k = rand_vec(sk * h * HEAD, 12);
    let v = rand_vec(sk * h * HEAD, 13);
    let runs = g.engine.flash_coop_runs();
    let out = zeros(g, &[1, sq, h, HEAD]);
    let stats = zeros(g, &[1, h, sq, 2]);
    g.engine.flash_attention_v2_with_stats(
        &g.ctx,
        &upload(g, &q, &[1, sq, h, HEAD]),
        &upload(g, &k, &[1, sk, h, HEAD]),
        &upload(g, &v, &[1, sk, h, HEAD]),
        &out,
        &stats,
        &AttentionOptions::default(),
        None,
    );
    assert_eq!(g.engine.flash_coop_runs(), runs, "one query stays scalar");
    // One key has all the softmax weight, so the output is its value.
    assert_near("one query", &read(g, &out), &v, (1e-6, 1e-5));
}

// ---------------------------------------------------------------------------
// RoPE inside the kernels
// ---------------------------------------------------------------------------

/// The inverse frequencies of `rope_type: "llama3"` (factor 32, low and high
/// frequency factors 1 and 4, 8,192 original positions: the Llama-3.2-1B
/// config) for 64 dimensions, in f64 from the formula of transformers'
/// `_compute_llama3_parameters`.
fn llama3_freqs(theta: f64) -> Vec<f32> {
    let (factor, low, high, original) = (32.0_f64, 1.0_f64, 4.0_f64, 8192.0_f64);
    (0..HEAD / 2)
        .map(|d| {
            let inv = 1.0 / theta.powf(2.0 * d as f64 / HEAD as f64);
            let wavelen = 2.0 * std::f64::consts::PI / inv;
            let scaled = if wavelen > original / low {
                inv / factor
            } else if wavelen < original / high {
                inv
            } else {
                let smooth = (original / wavelen - low) / (high - low);
                (1.0 - smooth) * inv / factor + smooth * inv
            };
            scaled as f32
        })
        .collect()
}

/// Rotate-half RoPE of a `[batch, seq, heads, 64]` tensor in f64: row `t` is
/// position `pos + t`, and `inverse` is the transposed rotation. `freqs`, when
/// given, are the inverse frequencies in place of those of `theta`.
fn rope_host(
    x: &[f32],
    (b, s, h): (usize, usize, usize),
    (theta, pos, freqs): (f32, u32, Option<&[f32]>),
    inverse: bool,
) -> Vec<f32> {
    let half = HEAD / 2;
    let mut out = x.to_vec();
    for bi in 0..b {
        for t in 0..s {
            for hi in 0..h {
                let at = ((bi * s + t) * h + hi) * HEAD;
                for d in 0..half {
                    let inv = match freqs {
                        Some(table) => f64::from(table[d]),
                        None => f64::from(theta).powf(-2.0 * d as f64 / HEAD as f64),
                    };
                    let angle = f64::from(pos + t as u32) * inv;
                    let (sin, cos) = angle.sin_cos();
                    let sin = if inverse { -sin } else { sin };
                    let (x1, x2) = (f64::from(x[at + d]), f64::from(x[at + d + half]));
                    out[at + d] = (x1 * cos - x2 * sin) as f32;
                    out[at + d + half] = (x1 * sin + x2 * cos) as f32;
                }
            }
        }
    }
    out
}

/// Calls with RoPE: a base frequency and the position of the first row. The
/// positions start at 0 and past it; the bases are the common ones.
const ROPE_CASES: &[(Case, f32, u32)] = &[
    (case(1, (17, 17), (2, 2), true), 10_000.0, 0),
    (case(1, (100, 100), (3, 3), true), 10_000.0, 5),
    (case(1, (256, 256), (9, 3), true), 100_000.0, 0),
    (case(2, (130, 130), (15, 5), true), 100_000.0, 37),
    (case(1, (96, 96), (32, 32), true), 10_000.0, 1_000),
    (case(1, (192, 192), (3, 3), false), 10_000.0, 3),
];

/// Runs every RoPE case through the RoPE entry points and compares with a CPU
/// reference that rotates first. `coop` says the calls run on the
/// cooperative kernels, which stage the rotated values as f16.
fn check_rope(g: &Gpu, rows: u32, coop: bool) {
    g.engine.set_flash_rows(rows);
    for (n, (c, theta, pos), scaled) in ROPE_CASES
        .iter()
        .enumerate()
        .flat_map(|(n, case)| [false, true].map(|scaled| (n, case, scaled)))
    {
        let name = format!(
            "{} theta={theta} pos={pos} rows={rows} scaled={scaled}",
            c.name()
        );
        // The scaled calls give the kernels a table that no theta reproduces.
        let table = scaled.then(|| llama3_freqs(f64::from(*theta)));
        let device_table = table
            .as_deref()
            .map(|t| RopeFreqs::new(&g.ctx, t, HEAD).unwrap());
        let [q, k, v, dout] = c.inputs(900 + 4 * n as u64);
        let (b, sq, sk, h, hkv) = (c.b, c.sq, c.sk, c.h, c.hkv);
        let rope = (*theta, *pos, table.as_deref());
        let round = |x: Vec<f32>| if coop { f16_round(&x) } else { x };
        let q_rot = round(rope_host(&q, (b, sq, h), rope, false));
        let k_rot = round(rope_host(&k, (b, sk, hkv), rope, false));
        let (want, probs) = cpu_attention(&q_rot, &k_rot, &v, None, c.dims(), c.visible());
        let (dq_rot, dk_rot, want_dv) =
            cpu_attention_backward(&q_rot, &k_rot, &v, &dout, &probs, c.dims());
        let want_dq = rope_host(&dq_rot, (b, sq, h), rope, true);
        let want_dk = rope_host(&dk_rot, (b, sk, hkv), rope, true);
        let want_lse = cpu_attention_lse(&q_rot, &k_rot, None, c.dims(), c.visible());

        let (tq, tk, tv, tdo) = (
            upload(g, &q, &[b, sq, h, HEAD]),
            upload(g, &k, &[b, sk, hkv, HEAD]),
            upload(g, &v, &[b, sk, hkv, HEAD]),
            upload(g, &dout, &[b, sq, h, HEAD]),
        );
        let opts = c.options();
        let out = zeros(g, &[b, sq, h, HEAD]);
        let stats = zeros(g, &[b, h, sq, 2]);
        let runs = g.engine.flash_coop_runs();
        g.engine.flash_attention_v2_rope_with_stats(
            &g.ctx,
            &tq,
            &tk,
            &tv,
            &out,
            Some(&stats),
            &opts,
            (*theta, *pos, device_table.as_ref()),
        );
        let got = read(g, &out);
        assert_near(&format!("fwd {name}"), &got, &want, (2e-3, 1e-2));
        let lse: Vec<f32> = read(g, &stats).iter().skip(1).step_by(2).copied().collect();
        assert_near(&format!("lse {name}"), &lse, &want_lse, (1e-3, 1e-3));

        let (gq, gk, gv) = (
            zeros(g, &[b, sq, h, HEAD]),
            zeros(g, &[b, sk, hkv, HEAD]),
            zeros(g, &[b, sk, hkv, HEAD]),
        );
        g.engine.flash_attention_v2_rope_backward_with_stats(
            &g.ctx,
            &tq,
            &tk,
            &tv,
            &out,
            &tdo,
            [&gq, &gk, &gv],
            &stats,
            &opts,
            (*theta, *pos, device_table.as_ref()),
        );
        let step = u64::from(coop);
        assert_eq!(
            g.engine.flash_coop_runs(),
            (runs.0 + step, runs.1 + step),
            "{name}: routed={coop}"
        );
        for (label, got, want) in [
            ("dq", read(g, &gq), want_dq),
            ("dk", read(g, &gk), want_dk),
            ("dv", read(g, &gv), want_dv),
        ] {
            let atol = 0.01 * max_abs(&want);
            assert_near(&format!("{label} {name}"), &got, &want, (atol, 2e-2));
        }
    }
    g.engine.set_flash_rows(0);
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn coop_flash_rope_matches_cpu_at_both_row_sizes() {
    let Some(g) = flash_gpu("coop_flash_rope_matches_cpu_at_both_row_sizes") else {
        return;
    };
    for rows in [32, 64] {
        check_rope(g, rows, true);
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn flash_rope_on_the_scalar_kernels_rotates_copies() {
    let g = gpu();
    g.engine.set_flash_coop(false);
    check_rope(g, 0, false);
    if g.engine.flash_coop_available() {
        g.engine.set_flash_coop(true);
    }
}
