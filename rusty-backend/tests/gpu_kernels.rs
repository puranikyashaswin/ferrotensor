//! GPU-vs-CPU reference tests for every kernel in `ComputeEngine`.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p rusty-backend --test gpu_kernels -- --ignored --test-threads=1
//! ```
//!
//! All tests share one device (creating several wgpu devices concurrently is
//! unreliable on some drivers).

use rusty_backend::{AdamParams, ComputeEngine, UnifiedTensor, WgpuContext};
use std::sync::OnceLock;

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

fn upload(data: &[f32], shape: &[usize]) -> UnifiedTensor {
    UnifiedTensor::new(&gpu().ctx, data, shape)
}

fn zeros(shape: &[usize]) -> UnifiedTensor {
    UnifiedTensor::zeros(&gpu().ctx, shape)
}

fn read(t: &UnifiedTensor) -> Vec<f32> {
    pollster::block_on(t.to_vec(&gpu().ctx))
}

#[track_caller]
fn assert_close(name: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        let tol = atol + rtol * w.abs();
        assert!(
            err <= tol && g.is_finite() == w.is_finite(),
            "{name}: index {i}: got {g}, want {w} (err {err}, tol {tol})"
        );
    }
}

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                out[i * n + j] += av * b[p * n + j];
            }
        }
    }
    out
}

fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0; a.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = a[r * cols + c];
        }
    }
    t
}

fn cpu_softmax_rows(x: &[f32], len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(len) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        out.extend(exps.iter().map(|e| e / sum));
    }
    out
}

fn cpu_rms_norm(x: &[f32], w: &[f32], len: usize, eps: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(len) {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / len as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        out.extend(row.iter().zip(w).map(|(v, w)| v * inv * w));
    }
    out
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Reference attention for [batch, seq, heads, dim] tensors with GQA, an
/// optional additive mask of shape [batch, heads, seq_q, seq_k] and causal
/// masking where key j is visible to query i when j <= i.
#[allow(clippy::too_many_arguments)]
fn cpu_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    (b, sq, sk, h, hkv, d): (usize, usize, usize, usize, usize, usize),
    causal: bool,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0.0f32; b * sq * h * d];
    // Attention probabilities per (b, h, i), used by the backward reference.
    let mut probs = Vec::new();
    for bi in 0..b {
        for hi in 0..h {
            let kh = hi / (h / hkv);
            for i in 0..sq {
                let qo = ((bi * sq + i) * h + hi) * d;
                let mut s = vec![f32::NEG_INFINITY; sk];
                for (j, sj) in s.iter_mut().enumerate() {
                    if causal && j > i {
                        continue;
                    }
                    let ko = ((bi * sk + j) * hkv + kh) * d;
                    let mut dot = 0.0;
                    for x in 0..d {
                        dot += q[qo + x] * k[ko + x];
                    }
                    *sj = dot * scale;
                    if let Some(m) = mask {
                        *sj += m[((bi * h + hi) * sq + i) * sk + j];
                    }
                }
                let max = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = s.iter().map(|x| (x - max).exp()).collect();
                let sum: f32 = e.iter().sum();
                let p: Vec<f32> = e.iter().map(|x| x / sum).collect();
                for (j, pj) in p.iter().enumerate() {
                    let vo = ((bi * sk + j) * hkv + kh) * d;
                    for x in 0..d {
                        out[qo + x] += pj * v[vo + x];
                    }
                }
                probs.push(p);
            }
        }
    }
    (out, probs)
}

/// Reference gradients of attention (no dropout) given dO.
#[allow(clippy::type_complexity)]
fn cpu_attention_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dout: &[f32],
    probs: &[Vec<f32>],
    (b, sq, sk, h, hkv, d): (usize, usize, usize, usize, usize, usize),
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let scale = 1.0 / (d as f32).sqrt();
    let mut dq = vec![0.0f32; q.len()];
    let mut dk = vec![0.0f32; k.len()];
    let mut dv = vec![0.0f32; v.len()];
    let mut row = 0;
    for bi in 0..b {
        for hi in 0..h {
            let kh = hi / (h / hkv);
            for i in 0..sq {
                let p = &probs[row];
                row += 1;
                let qo = ((bi * sq + i) * h + hi) * d;
                // dP_ij = dO_i . V_j ; dS = P * (dP - sum_j P_ij dP_ij)
                let dp: Vec<f32> = (0..sk)
                    .map(|j| {
                        let vo = ((bi * sk + j) * hkv + kh) * d;
                        (0..d).map(|x| dout[qo + x] * v[vo + x]).sum()
                    })
                    .collect();
                let rowsum: f32 = p.iter().zip(&dp).map(|(a, b)| a * b).sum();
                for j in 0..sk {
                    let ko = ((bi * sk + j) * hkv + kh) * d;
                    let ds = p[j] * (dp[j] - rowsum) * scale;
                    for x in 0..d {
                        dv[ko + x] += p[j] * dout[qo + x];
                        dq[qo + x] += ds * k[ko + x];
                        dk[ko + x] += ds * q[qo + x];
                    }
                }
            }
        }
    }
    (dq, dk, dv)
}

// ---------------------------------------------------------------------------
// Matmul
// ---------------------------------------------------------------------------

const MATMUL_SHAPES: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (1, 7, 1),
    (3, 5, 7),
    (16, 16, 16),
    (17, 33, 9),
    (64, 64, 64),
    (65, 1, 130),
    (127, 129, 65),
    (1, 512, 300),
    (300, 512, 1),
    (256, 384, 128),
];

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn matmul_matches_cpu() {
    let g = gpu();
    for (seed, &(m, k, n)) in MATMUL_SHAPES.iter().enumerate() {
        let a = rand_vec(m * k, seed as u64);
        let b = rand_vec(k * n, 100 + seed as u64);
        let out = zeros(&[m, n]);
        g.engine
            .matmul(&g.ctx, &upload(&a, &[m, k]), &upload(&b, &[k, n]), &out);
        let want = cpu_matmul(&a, &b, m, k, n);
        assert_close(
            &format!("matmul {m}x{k}x{n}"),
            &read(&out),
            &want,
            1e-4,
            1e-4,
        );
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn matmul_bt_matches_cpu() {
    let g = gpu();
    for (seed, &(m, k, n)) in MATMUL_SHAPES.iter().enumerate() {
        let a = rand_vec(m * k, seed as u64);
        let b = rand_vec(k * n, 200 + seed as u64);
        let bt = transpose(&b, k, n); // [n, k]
        let out = zeros(&[m, n]);
        g.engine
            .matmul_bt(&g.ctx, &upload(&a, &[m, k]), &upload(&bt, &[n, k]), &out);
        let want = cpu_matmul(&a, &b, m, k, n);
        assert_close(
            &format!("matmul_bt {m}x{k}x{n}"),
            &read(&out),
            &want,
            1e-4,
            1e-4,
        );
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn matmul_at_matches_cpu() {
    let g = gpu();
    for (seed, &(m, k, n)) in MATMUL_SHAPES.iter().enumerate() {
        let a = rand_vec(m * k, seed as u64);
        let b = rand_vec(k * n, 300 + seed as u64);
        let at = transpose(&a, m, k); // [k, m]
        let out = zeros(&[m, n]);
        g.engine
            .matmul_at(&g.ctx, &upload(&at, &[k, m]), &upload(&b, &[k, n]), &out);
        let want = cpu_matmul(&a, &b, m, k, n);
        assert_close(
            &format!("matmul_at {m}x{k}x{n}"),
            &read(&out),
            &want,
            1e-4,
            1e-4,
        );
    }
}

// ---------------------------------------------------------------------------
// Element-wise ops and their backward passes
// ---------------------------------------------------------------------------

const ELEMWISE_LEN: usize = 1000; // not a multiple of the workgroup size

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn binary_ops_match_cpu() {
    let g = gpu();
    let n = ELEMWISE_LEN;
    let a = rand_vec(n, 1);
    let b = rand_vec(n, 2);
    let (ta, tb) = (upload(&a, &[n]), upload(&b, &[n]));
    type Op = fn(&ComputeEngine, &WgpuContext, &UnifiedTensor, &UnifiedTensor, &UnifiedTensor);
    let cases: [(&str, Op, fn(f32, f32) -> f32); 5] = [
        ("add", ComputeEngine::add, |x, y| x + y),
        ("sub", ComputeEngine::sub, |x, y| x - y),
        ("mul", ComputeEngine::mul, |x, y| x * y),
        ("silu_gate", ComputeEngine::silu_gate_fused, |u, g| {
            u * sigmoid(u) * g
        }),
        ("mse_loss", ComputeEngine::mse_loss, |p, t| {
            0.5 * (p - t) * (p - t)
        }),
    ];
    for (name, op, f) in cases {
        let out = zeros(&[n]);
        op(&g.engine, &g.ctx, &ta, &tb, &out);
        let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| f(*x, *y)).collect();
        assert_close(name, &read(&out), &want, 1e-6, 1e-5);
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn unary_ops_match_cpu() {
    let g = gpu();
    let n = ELEMWISE_LEN;
    let x: Vec<f32> = rand_vec(n, 3).iter().map(|v| v * 8.0).collect();
    let tx = upload(&x, &[n]);
    type Op = fn(&ComputeEngine, &WgpuContext, &UnifiedTensor, &UnifiedTensor);
    let cases: [(&str, Op, fn(f32) -> f32); 3] = [
        ("silu", ComputeEngine::silu, |v| v * sigmoid(v)),
        ("relu", ComputeEngine::relu, |v| v.max(0.0)),
        ("grad_norm_sq", ComputeEngine::gradient_norm_squared, |v| {
            v * v
        }),
    ];
    for (name, op, f) in cases {
        let out = zeros(&[n]);
        op(&g.engine, &g.ctx, &tx, &out);
        let want: Vec<f32> = x.iter().map(|v| f(*v)).collect();
        assert_close(name, &read(&out), &want, 1e-5, 1e-5);
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn elementwise_backward_matches_cpu() {
    let g = gpu();
    let n = ELEMWISE_LEN;
    let go = rand_vec(n, 4);
    let a: Vec<f32> = rand_vec(n, 5).iter().map(|v| v * 4.0).collect();
    let b = rand_vec(n, 6);
    let (tgo, ta, tb) = (upload(&go, &[n]), upload(&a, &[n]), upload(&b, &[n]));

    let (ga, gb) = (zeros(&[n]), zeros(&[n]));
    g.engine.add_backward(&g.ctx, &tgo, &ga, &gb);
    assert_close("add_bw a", &read(&ga), &go, 0.0, 0.0);
    assert_close("add_bw b", &read(&gb), &go, 0.0, 0.0);

    g.engine.sub_backward(&g.ctx, &tgo, &ga, &gb);
    let neg: Vec<f32> = go.iter().map(|v| -v).collect();
    assert_close("sub_bw a", &read(&ga), &go, 0.0, 0.0);
    assert_close("sub_bw b", &read(&gb), &neg, 0.0, 0.0);

    g.engine.mul_backward(&g.ctx, &tgo, &ta, &tb, &ga, &gb);
    let want_a: Vec<f32> = go.iter().zip(&b).map(|(g, b)| g * b).collect();
    let want_b: Vec<f32> = go.iter().zip(&a).map(|(g, a)| g * a).collect();
    assert_close("mul_bw a", &read(&ga), &want_a, 1e-6, 1e-6);
    assert_close("mul_bw b", &read(&gb), &want_b, 1e-6, 1e-6);

    let gi = zeros(&[n]);
    g.engine.silu_backward(&g.ctx, &tgo, &ta, &gi);
    let want: Vec<f32> = go
        .iter()
        .zip(&a)
        .map(|(g, x)| {
            let s = sigmoid(*x);
            g * s * (1.0 + x * (1.0 - s))
        })
        .collect();
    assert_close("silu_bw", &read(&gi), &want, 1e-5, 1e-5);

    g.engine.relu_backward(&g.ctx, &tgo, &ta, &gi);
    let want: Vec<f32> = go
        .iter()
        .zip(&a)
        .map(|(g, x)| if *x > 0.0 { *g } else { 0.0 })
        .collect();
    assert_close("relu_bw", &read(&gi), &want, 0.0, 0.0);

    g.engine
        .silu_gate_backward(&g.ctx, &ta, &tb, &tgo, &ga, &gb);
    let want_up: Vec<f32> = (0..n)
        .map(|i| {
            let s = sigmoid(a[i]);
            go[i] * b[i] * s * (1.0 + a[i] * (1.0 - s))
        })
        .collect();
    let want_gate: Vec<f32> = (0..n).map(|i| go[i] * a[i] * sigmoid(a[i])).collect();
    assert_close("silu_gate_bw up", &read(&ga), &want_up, 1e-5, 1e-5);
    assert_close("silu_gate_bw gate", &read(&gb), &want_gate, 1e-5, 1e-5);

    g.engine.mse_loss_backward(&g.ctx, &ta, &tb, &gi);
    let want: Vec<f32> = a.iter().zip(&b).map(|(p, t)| p - t).collect();
    assert_close("mse_bw", &read(&gi), &want, 1e-6, 1e-6);
}

// ---------------------------------------------------------------------------
// Softmax and normalisation
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn softmax_matches_cpu() {
    let g = gpu();
    for &(rows, len, scale, offset) in &[
        (1, 1, 1.0, 0.0),
        (3, 7, 1.0, 0.0),
        (5, 1000, 10.0, 0.0),
        (2, 4096, 1.0, 0.0),
        (4, 33, 1.0, -5000.0), // large negative logits
        (4, 33, 30.0, 2000.0), // large positive logits
    ] {
        let x: Vec<f32> = rand_vec(rows * len, len as u64)
            .iter()
            .map(|v| v * scale + offset)
            .collect();
        let out = zeros(&[rows, len]);
        g.engine.softmax(&g.ctx, &upload(&x, &[rows, len]), &out);
        let want = cpu_softmax_rows(&x, len);
        assert_close(
            &format!("softmax {rows}x{len}"),
            &read(&out),
            &want,
            1e-6,
            1e-4,
        );
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn softmax_backward_matches_cpu() {
    let g = gpu();
    let (rows, len) = (6, 300);
    let s = cpu_softmax_rows(&rand_vec(rows * len, 7), len);
    let go = rand_vec(rows * len, 8);
    let gi = zeros(&[rows, len]);
    g.engine.softmax_backward(
        &g.ctx,
        &upload(&go, &[rows, len]),
        &upload(&s, &[rows, len]),
        &gi,
    );
    let mut want = Vec::new();
    for (sr, gr) in s.chunks(len).zip(go.chunks(len)) {
        let dot: f32 = sr.iter().zip(gr).map(|(a, b)| a * b).sum();
        want.extend(sr.iter().zip(gr).map(|(s, g)| s * (g - dot)));
    }
    assert_close("softmax_bw", &read(&gi), &want, 1e-6, 1e-4);
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn rms_norm_matches_cpu() {
    let g = gpu();
    for &(rows, len) in &[(1, 1), (5, 37), (3, 64), (2, 4096), (130, 256)] {
        let x = rand_vec(rows * len, len as u64);
        let w = rand_vec(len, 1 + len as u64);
        let out = zeros(&[rows, len]);
        g.engine
            .rms_norm(&g.ctx, &upload(&x, &[rows, len]), &upload(&w, &[len]), &out);
        let want = cpu_rms_norm(&x, &w, len, 1e-5);
        assert_close(
            &format!("rms_norm {rows}x{len}"),
            &read(&out),
            &want,
            1e-5,
            1e-4,
        );
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn rms_norm_residual_matches_cpu() {
    let g = gpu();
    let len = 777;
    let x = rand_vec(len, 9);
    let r = rand_vec(len, 10);
    let w = rand_vec(len, 11);
    let out = zeros(&[len]);
    g.engine.rms_norm_residual(
        &g.ctx,
        &upload(&x, &[len]),
        &upload(&r, &[len]),
        &upload(&w, &[len]),
        &out,
    );
    let sum: Vec<f32> = x.iter().zip(&r).map(|(a, b)| a + b).collect();
    let want = cpu_rms_norm(&sum, &w, len, 1e-5);
    assert_close("rms_norm_residual", &read(&out), &want, 1e-5, 1e-4);
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

/// Rotate-half RoPE over [batch, seq, heads * head_dim]; position of token s
/// is `pos + s` for every head.
fn cpu_rope(x: &[f32], batch: usize, seq: usize, heads: usize, hd: usize, pos: usize) -> Vec<f32> {
    let mut out = x.to_vec();
    let half = hd / 2;
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..heads {
                let base = ((b * seq + s) * heads + h) * hd;
                let p = (pos + s) as f32;
                for i in 0..half {
                    let freq = 10000f32.powf(-2.0 * i as f32 / hd as f32);
                    let (sin, cos) = (p * freq).sin_cos();
                    let (x1, x2) = (x[base + i], x[base + i + half]);
                    out[base + i] = x1 * cos - x2 * sin;
                    out[base + i + half] = x1 * sin + x2 * cos;
                }
            }
        }
    }
    out
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn rope_matches_cpu() {
    let g = gpu();
    for &(batch, seq, heads, hd, pos) in &[(1, 1, 1, 8, 0), (1, 5, 1, 64, 3), (2, 7, 4, 32, 11)] {
        let x = rand_vec(batch * seq * heads * hd, 12);
        let t = upload(&x, &[batch, seq, heads * hd]);
        g.engine.rope(&g.ctx, &t, hd, pos);
        let want = cpu_rope(&x, batch, seq, heads, hd, pos);
        assert_close(
            &format!("rope b{batch} s{seq} h{heads} d{hd}"),
            &read(&t),
            &want,
            1e-4,
            1e-4,
        );
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn flash_attention_single_head_matches_cpu() {
    let g = gpu();
    for &(seq, d, causal) in &[
        (1, 16, false),
        (37, 64, true),
        (100, 64, false),
        (65, 128, true),
    ] {
        let q = rand_vec(seq * d, 13);
        let k = rand_vec(seq * d, 14);
        let v = rand_vec(seq * d, 15);
        let out = zeros(&[seq, d]);
        g.engine.flash_attention(
            &g.ctx,
            &upload(&q, &[seq, d]),
            &upload(&k, &[seq, d]),
            &upload(&v, &[seq, d]),
            &out,
            causal,
        );
        let (want, _) = cpu_attention(&q, &k, &v, None, (1, seq, seq, 1, 1, d), causal);
        assert_close(
            &format!("flash_attention s{seq} d{d} causal={causal}"),
            &read(&out),
            &want,
            1e-5,
            1e-3,
        );
    }
}

/// (batch, seq_q, seq_k, heads, kv_heads, head_dim, causal)
const ATTN_CASES: &[(usize, usize, usize, usize, usize, usize, bool)] = &[
    (1, 1, 1, 1, 1, 16, false),
    (1, 8, 8, 2, 2, 32, true),
    (2, 37, 37, 4, 2, 64, true),
    (1, 70, 70, 2, 1, 64, false),
    (1, 33, 50, 4, 4, 48, false),
    (1, 130, 130, 2, 2, 128, true),
];

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn flash_attention_v2_matches_cpu() {
    let g = gpu();
    for &(b, sq, sk, h, hkv, d, causal) in ATTN_CASES {
        let q = rand_vec(b * sq * h * d, 16);
        let k = rand_vec(b * sk * hkv * d, 17);
        let v = rand_vec(b * sk * hkv * d, 18);
        let out = zeros(&[b, sq, h, d]);
        g.engine.flash_attention_v2(
            &g.ctx,
            &upload(&q, &[b, sq, h, d]),
            &upload(&k, &[b, sk, hkv, d]),
            &upload(&v, &[b, sk, hkv, d]),
            &out,
            causal,
            0.0,
            0,
            None,
        );
        let (want, _) = cpu_attention(&q, &k, &v, None, (b, sq, sk, h, hkv, d), causal);
        assert_close(
            &format!("flash_v2 b{b} q{sq} k{sk} h{h}/{hkv} d{d} causal={causal}"),
            &read(&out),
            &want,
            1e-5,
            1e-3,
        );
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn flash_attention_v2_mask_matches_cpu() {
    let g = gpu();
    let (b, sq, sk, h, d) = (2, 19, 23, 2, 32);
    let q = rand_vec(b * sq * h * d, 19);
    let k = rand_vec(b * sk * h * d, 20);
    let v = rand_vec(b * sk * h * d, 21);
    // Padding-style mask: hide the last few keys of batch 1, plus a bias.
    let mut mask = vec![0.0f32; b * h * sq * sk];
    for bi in 0..b {
        for hi in 0..h {
            for i in 0..sq {
                for j in 0..sk {
                    let idx = ((bi * h + hi) * sq + i) * sk + j;
                    mask[idx] = if bi == 1 && j >= 15 {
                        -1e9
                    } else {
                        0.1 * ((i + j) % 3) as f32
                    };
                }
            }
        }
    }
    let out = zeros(&[b, sq, h, d]);
    g.engine.flash_attention_v2(
        &g.ctx,
        &upload(&q, &[b, sq, h, d]),
        &upload(&k, &[b, sk, h, d]),
        &upload(&v, &[b, sk, h, d]),
        &out,
        false,
        0.0,
        0,
        Some(&upload(&mask, &[b, h, sq, sk])),
    );
    let (want, _) = cpu_attention(&q, &k, &v, Some(&mask), (b, sq, sk, h, h, d), false);
    assert_close("flash_v2 mask", &read(&out), &want, 1e-5, 1e-3);
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn flash_attention_v2_dropout_keeps_expected_fraction() {
    let g = gpu();
    // With V = 1 everywhere, each output equals sum_j P_ij * keep_ij / (1 - p),
    // whose mean over many rows is 1.
    let (b, s, h, d) = (1, 256, 2, 16);
    let q = rand_vec(b * s * h * d, 22);
    let k = rand_vec(b * s * h * d, 23);
    let v = vec![1.0f32; b * s * h * d];
    let out = zeros(&[b, s, h, d]);
    g.engine.flash_attention_v2(
        &g.ctx,
        &upload(&q, &[b, s, h, d]),
        &upload(&k, &[b, s, h, d]),
        &upload(&v, &[b, s, h, d]),
        &out,
        false,
        0.25,
        1234,
        None,
    );
    let got = read(&out);
    let mean = got.iter().sum::<f32>() / got.len() as f32;
    assert!((mean - 1.0).abs() < 0.05, "dropout mean {mean}");
    assert!(
        got.iter().any(|x| (x - 1.0).abs() > 1e-3),
        "dropout had no effect"
    );
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn flash_attention_v2_backward_matches_cpu() {
    let g = gpu();
    for &(b, sq, sk, h, hkv, d, causal) in ATTN_CASES {
        let dims = (b, sq, sk, h, hkv, d);
        let q = rand_vec(b * sq * h * d, 24);
        let k = rand_vec(b * sk * hkv * d, 25);
        let v = rand_vec(b * sk * hkv * d, 26);
        let dout = rand_vec(b * sq * h * d, 27);
        let (out, probs) = cpu_attention(&q, &k, &v, None, dims, causal);
        let (want_dq, want_dk, want_dv) = cpu_attention_backward(&q, &k, &v, &dout, &probs, dims);

        let (gq, gk, gv) = (
            zeros(&[b, sq, h, d]),
            zeros(&[b, sk, hkv, d]),
            zeros(&[b, sk, hkv, d]),
        );
        g.engine.flash_attention_v2_backward(
            &g.ctx,
            &upload(&q, &[b, sq, h, d]),
            &upload(&k, &[b, sk, hkv, d]),
            &upload(&v, &[b, sk, hkv, d]),
            &upload(&out, &[b, sq, h, d]),
            &upload(&dout, &[b, sq, h, d]),
            &gq,
            &gk,
            &gv,
            causal,
        );
        let name = format!("b{b} q{sq} k{sk} h{h}/{hkv} d{d} causal={causal}");
        assert_close(&format!("dq {name}"), &read(&gq), &want_dq, 1e-4, 1e-3);
        assert_close(&format!("dk {name}"), &read(&gk), &want_dk, 1e-4, 1e-3);
        assert_close(&format!("dv {name}"), &read(&gv), &want_dv, 1e-4, 1e-3);
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn flash_attention_single_head_backward_matches_cpu() {
    let g = gpu();
    let (s, d) = (45, 32);
    let dims = (1, s, s, 1, 1, d);
    let q = rand_vec(s * d, 28);
    let k = rand_vec(s * d, 29);
    let v = rand_vec(s * d, 30);
    let dout = rand_vec(s * d, 31);
    let (out, probs) = cpu_attention(&q, &k, &v, None, dims, true);
    let (want_dq, want_dk, want_dv) = cpu_attention_backward(&q, &k, &v, &dout, &probs, dims);
    let (gq, gk, gv) = (zeros(&[s, d]), zeros(&[s, d]), zeros(&[s, d]));
    g.engine.flash_attention_backward(
        &g.ctx,
        &upload(&q, &[s, d]),
        &upload(&k, &[s, d]),
        &upload(&v, &[s, d]),
        &upload(&out, &[s, d]),
        &upload(&dout, &[s, d]),
        &gq,
        &gk,
        &gv,
        true,
    );
    assert_close("v1 dq", &read(&gq), &want_dq, 1e-4, 1e-3);
    assert_close("v1 dk", &read(&gk), &want_dk, 1e-4, 1e-3);
    assert_close("v1 dv", &read(&gv), &want_dv, 1e-4, 1e-3);
}

// ---------------------------------------------------------------------------
// Quantisation
// ---------------------------------------------------------------------------

fn bytes_tensor(bytes: &[u8], shape: &[usize]) -> UnifiedTensor {
    UnifiedTensor::from_bytes(&gpu().ctx, bytes, shape)
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn dequantize_u8_matches_cpu() {
    let g = gpu();
    let n = 4 * 300;
    let bytes: Vec<u8> = (0..n).map(|i| (i * 37 % 256) as u8).collect();
    let out = zeros(&[n]);
    g.engine
        .dequantize(&g.ctx, &bytes_tensor(&bytes, &[n]), &out);
    let want: Vec<f32> = bytes.iter().map(|b| *b as f32).collect();
    assert_close("dequantize", &read(&out), &want, 0.0, 0.0);
}

fn check_affine_dequant(n: usize) {
    let g = gpu();
    let groups = n.div_ceil(64);
    let bytes: Vec<u8> = (0..n).map(|i| (i * 131 % 256) as u8).collect();
    let scales = rand_vec(groups, 32);
    let biases = rand_vec(groups, 33);
    let w = bytes_tensor(&bytes, &[n / 4]);
    let s = upload(&scales, &[groups]);
    let bi = upload(&biases, &[groups]);
    let out = UnifiedTensor::empty(&g.ctx, &[n]);
    g.engine
        .affine_dequantize(&g.ctx, w.buffer(), s.buffer(), bi.buffer(), &out);
    let want: Vec<f32> = (0..n)
        .map(|i| bytes[i] as f32 * scales[i / 64] + biases[i / 64])
        .collect();
    assert_close(
        &format!("affine_dequantize n={n}"),
        &read(&out),
        &want,
        1e-5,
        1e-5,
    );
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn affine_dequantize_matches_cpu() {
    check_affine_dequant(64 * 37);
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn affine_dequantize_large_uses_2d_dispatch() {
    // More than 65535 workgroups of 64 threads.
    check_affine_dequant(65535 * 64 + 64 * 10);
}

// ---------------------------------------------------------------------------
// Losses, reductions, optimisers and misc
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn cross_entropy_matches_cpu() {
    let g = gpu();
    for &(vocab, target) in &[(1usize, 0u32), (10, 3), (5000, 4321)] {
        let logits: Vec<f32> = rand_vec(vocab, vocab as u64)
            .iter()
            .map(|v| v * 5.0)
            .collect();
        let tl = upload(&logits, &[vocab]);
        let loss = zeros(&[1]);
        g.engine.cross_entropy_loss(&g.ctx, &tl, target, &loss);
        let p = cpu_softmax_rows(&logits, vocab);
        let want_loss = -p[target as usize].ln();
        assert_close("ce loss", &read(&loss), &[want_loss], 1e-4, 1e-4);

        let grad = zeros(&[vocab]);
        g.engine.cross_entropy_backward(&g.ctx, &tl, target, &grad);
        let mut want = p.clone();
        want[target as usize] -= 1.0;
        assert_close("ce grad", &read(&grad), &want, 1e-6, 1e-4);
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn top_k_matches_cpu() {
    let g = gpu();
    let x = rand_vec(500, 34);
    let k = 5;
    let out = zeros(&[k]);
    g.engine.top_k(&g.ctx, &upload(&x, &[500]), &out, k as u32);
    let got: Vec<u32> = read(&out).iter().map(|v| v.to_bits()).collect();
    let mut idx: Vec<u32> = (0..500).collect();
    idx.sort_by(|a, b| x[*b as usize].total_cmp(&x[*a as usize]));
    assert_eq!(got, idx[..k].to_vec());
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn conv1d_matches_cpu() {
    let g = gpu();
    let (dim, seq) = (6, 21);
    let x = rand_vec(dim * seq, 35);
    let w = rand_vec(dim * 3, 36);
    let bias = rand_vec(dim, 37);
    let out = zeros(&[dim, seq]);
    g.engine.conv1d(
        &g.ctx,
        &upload(&x, &[dim, seq]),
        &upload(&w, &[dim, 3]),
        &upload(&bias, &[dim]),
        &out,
    );
    // Depthwise, kernel 3, padding 1: y[d,s] = b[d] + sum_k w[d,k] x[d, s+k-1]
    let mut want = vec![0.0; dim * seq];
    for d in 0..dim {
        for s in 0..seq {
            let mut acc = bias[d];
            for kk in 0..3 {
                let i = s as isize + kk as isize - 1;
                if i >= 0 && (i as usize) < seq {
                    acc += w[d * 3 + kk] * x[d * seq + i as usize];
                }
            }
            want[d * seq + s] = acc;
        }
    }
    assert_close("conv1d", &read(&out), &want, 1e-5, 1e-5);
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn optimizers_match_cpu() {
    let g = gpu();
    let n = ELEMWISE_LEN;
    let w = rand_vec(n, 38);
    let grad = rand_vec(n, 39);

    let tw = upload(&w, &[n]);
    g.engine.sgd_update(&g.ctx, &tw, &upload(&grad, &[n]), 0.1);
    let want: Vec<f32> = w.iter().zip(&grad).map(|(w, g)| w - 0.1 * g).collect();
    assert_close("sgd", &read(&tw), &want, 1e-6, 1e-6);

    let m0 = rand_vec(n, 40);
    let v0: Vec<f32> = rand_vec(n, 41).iter().map(|v| v.abs()).collect();
    let p = AdamParams {
        lr: 1e-2,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
        t: 3.0,
        _pad: [0.0; 2],
    };
    let (tw, tm, tv) = (upload(&w, &[n]), upload(&m0, &[n]), upload(&v0, &[n]));
    g.engine
        .adamw_update(&g.ctx, &tw, &upload(&grad, &[n]), &tm, &tv, &p);
    let (mut ww, mut wm, mut wv) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let m = p.beta1 * m0[i] + (1.0 - p.beta1) * grad[i];
        let v = p.beta2 * v0[i] + (1.0 - p.beta2) * grad[i] * grad[i];
        let mh = m / (1.0 - p.beta1.powf(p.t));
        let vh = v / (1.0 - p.beta2.powf(p.t));
        ww[i] = w[i] - p.lr * (mh / (vh.sqrt() + p.eps) + p.weight_decay * w[i]);
        wm[i] = m;
        wv[i] = v;
    }
    assert_close("adamw m", &read(&tm), &wm, 1e-6, 1e-5);
    assert_close("adamw v", &read(&tv), &wv, 1e-6, 1e-5);
    assert_close("adamw w", &read(&tw), &ww, 1e-5, 1e-4);
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn gradient_utilities_match_cpu() {
    let g = gpu();
    let n = ELEMWISE_LEN;
    let a = rand_vec(n, 42);
    let b = rand_vec(n, 43);

    let ta = upload(&a, &[n]);
    g.engine.gradient_accumulate(&g.ctx, &ta, &upload(&b, &[n]));
    let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_close("grad_accumulate", &read(&ta), &sum, 1e-6, 1e-6);

    g.engine.gradient_scale(&g.ctx, &ta, 0.5);
    let half: Vec<f32> = sum.iter().map(|v| v * 0.5).collect();
    assert_close("grad_scale", &read(&ta), &half, 1e-6, 1e-6);

    g.engine.clip_grad_by_norm(&g.ctx, &ta, 0.25);
    let q: Vec<f32> = half.iter().map(|v| v * 0.25).collect();
    assert_close("clip_grad", &read(&ta), &q, 1e-6, 1e-6);

    let dst = zeros(&[2 * n]);
    g.engine.copy_tensor(&g.ctx, &upload(&a, &[n]), &dst, n * 4);
    let mut want = vec![0.0; n];
    want.extend_from_slice(&a);
    assert_close("copy_tensor", &read(&dst), &want, 0.0, 0.0);
}
