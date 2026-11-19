//! GPU-vs-CPU reference tests for every kernel in `ComputeEngine`.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-backend --test gpu_kernels -- --ignored --test-threads=1
//! ```
//!
//! All tests share one device (creating several wgpu devices concurrently is
//! unreliable on some drivers).

#![allow(clippy::unwrap_used)]

use ferrotensor_backend::{
    q4, AdamParams, AttentionOptions, ComputeEngine, DType, Epilogue, MatmulPath, MemoryCategory,
    SkinnyTile, SmallMTile, UnifiedTensor, WgpuContext, MASKED_CE_SKIP, SUM_SQUARES_CHUNK,
};
use q4::Q4Group;
use std::io::Write as _;
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
    pollster::block_on(t.to_vec(&gpu().ctx)).unwrap()
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
/// optional additive mask of shape [batch, heads, seq_q, seq_k] and a
/// visibility rule `visible(i, j)` for query i and key j.
#[allow(clippy::too_many_arguments)]
fn cpu_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    (b, sq, sk, h, hkv, d): (usize, usize, usize, usize, usize, usize),
    visible: impl Fn(usize, usize) -> bool,
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
                    if !visible(i, j) {
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

fn causal_rule(causal: bool) -> impl Fn(usize, usize) -> bool {
    move |i, j| !causal || j <= i
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

/// `x` rounded to f16 and back, as the `*_f16b` matmuls see it.
fn f16_round(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect()
}

/// `x` as the f16-weight matmuls see it: with the cooperative-matrix kernel
/// routed, `a` is converted to f16, so tests feed values f16 holds exactly.
fn maybe_f16(x: Vec<f32>) -> Vec<f32> {
    if gpu().engine.coop_matmul_active() {
        f16_round(&x)
    } else {
        x
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn matmul_with_f16_b_matches_cpu() {
    // Odd k * n exercises the padded last word; shapes off the 64x64 tile
    // exercise the edges.
    let g = gpu();
    for (seed, &(m, k, n)) in MATMUL_SHAPES
        .iter()
        .chain(&[(3, 5, 7), (65, 33, 129)])
        .enumerate()
    {
        let a = maybe_f16(rand_vec(m * k, 300 + seed as u64));
        let b = rand_vec(k * n, 400 + seed as u64);
        let b_h = UnifiedTensor::from_f32_as_f16(&g.ctx, &b, &[k, n]);
        assert!(b_h.size_in_bytes() <= ((k * n * 2).div_ceil(4) * 4).max(4) as u64);
        let out = zeros(&[m, n]);
        g.engine
            .matmul_f16b(&g.ctx, &upload(&a, &[m, k]), &b_h, &out);
        let want = cpu_matmul(&a, &f16_round(&b), m, k, n);
        assert_close(
            &format!("matmul_f16b {m}x{k}x{n}"),
            &read(&out),
            &want,
            1e-4,
            1e-4,
        );

        let bt = transpose(&b, k, n); // [n, k]
        let bt_h = UnifiedTensor::from_f32_as_f16(&g.ctx, &bt, &[n, k]);
        let out = zeros(&[m, n]);
        g.engine
            .matmul_bt_f16b(&g.ctx, &upload(&a, &[m, k]), &bt_h, &out);
        assert_close(
            &format!("matmul_bt_f16b {m}x{k}x{n}"),
            &read(&out),
            &want,
            1e-4,
            1e-4,
        );
    }
}

/// Largest `|got - want|`.
fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max)
}

/// `w` through the 4-bit round trip on the host, as the `*_q4b` matmuls
/// see it.
fn q4_round(w: &[f32], k: usize, n: usize, group: Q4Group) -> Vec<f32> {
    q4::dequantize(&q4::quantize(w, k, n, group).unwrap(), k, n, group)
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn matmul_with_q4_b_matches_cpu() {
    // A frozen weight W[k, n] is used twice: forward `y = x W` (mode 0,
    // `matmul_q4b`) and the input gradient `dx = dy W^T` (mode 1,
    // `matmul_bt_q4b`, the same tensor). Both are compared with the same
    // product on the host-dequantised copy of W, never the original W.
    // k * n off a multiple of 64 exercises the padded last group; shapes
    // off the 64x64 tile exercise the edges; (2, 64, 3) has one group per
    // output column.
    let g = gpu();
    let (mut fwd_diff, mut bwd_diff, mut quant_err, mut quant_effect) = (0f32, 0f32, 0f32, 0f32);
    for (seed, &(m, k, n)) in MATMUL_SHAPES
        .iter()
        .chain(&[(3, 5, 7), (65, 33, 129), (2, 64, 3)])
        .enumerate()
    {
        let x = rand_vec(m * k, 500 + seed as u64);
        let w: Vec<f32> = rand_vec(k * n, 600 + seed as u64)
            .iter()
            .map(|v| v * 0.1)
            .collect();
        let w_q = UnifiedTensor::from_f32_as_q4(&g.ctx, &w, &[k, n], Q4Group::DEFAULT).unwrap();
        assert_eq!(w_q.dtype, DType::Q4);
        assert_eq!(w_q.shape, vec![k, n]);
        assert_eq!(
            w_q.size_in_bytes(),
            4 * q4::packed_words(k, n, Q4Group::DEFAULT) as u64
        );
        let w_dq = q4_round(&w, k, n, Q4Group::DEFAULT);
        quant_err = quant_err.max(max_abs_diff(&w_dq, &w));

        let y = zeros(&[m, n]);
        g.engine.matmul_q4b(&g.ctx, &upload(&x, &[m, k]), &w_q, &y);
        let want = cpu_matmul(&x, &w_dq, m, k, n);
        let got = read(&y);
        assert_close(&format!("matmul_q4b {m}x{k}x{n}"), &got, &want, 1e-4, 1e-4);
        fwd_diff = fwd_diff.max(max_abs_diff(&got, &want));
        quant_effect = quant_effect.max(max_abs_diff(&got, &cpu_matmul(&x, &w, m, k, n)));

        let dy = rand_vec(m * n, 700 + seed as u64);
        let dx = zeros(&[m, k]);
        g.engine
            .matmul_bt_q4b(&g.ctx, &upload(&dy, &[m, n]), &w_q, &dx);
        let want = cpu_matmul(&dy, &transpose(&w_dq, k, n), m, n, k);
        let got = read(&dx);
        assert_close(
            &format!("matmul_bt_q4b {m}x{n}x{k}"),
            &got,
            &want,
            1e-4,
            1e-4,
        );
        bwd_diff = bwd_diff.max(max_abs_diff(&got, &want));
    }
    let _ = writeln!(
        std::io::stderr(),
        "q4 matmul vs host-dequantised f32: forward x W max |diff| {fwd_diff:.3e}, \
         input gradient dy W^T max |diff| {bwd_diff:.3e} (tolerance 1e-4 + 1e-4 |want|); \
         weights in [-0.1, 0.1): max |W - dequant(quant(W))| {quant_err:.3e}, \
         max |x W_q - x W| {quant_effect:.3e} (quantisation effect, not asserted)"
    );
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn matmul_with_q4_b_matches_cpu_for_every_group_size() {
    // Each size moves the scale words and the group boundaries, so a kernel
    // that ignored the size would fail every size but 64. Shapes with k a
    // multiple of every size, k off all of them, and k * n off the largest
    // group, in both modes.
    let g = gpu();
    for size in Q4Group::SIZES {
        let group = Q4Group::new(size).unwrap();
        for (seed, &(m, k, n)) in [
            (1, 1, 1),
            (3, 5, 7),
            (17, 128, 9),
            (65, 33, 129),
            (64, 256, 64),
        ]
        .iter()
        .enumerate()
        {
            let x = rand_vec(m * k, 1500 + seed as u64);
            let w: Vec<f32> = rand_vec(k * n, 1600 + seed as u64)
                .iter()
                .map(|v| v * 0.1)
                .collect();
            let w_q = UnifiedTensor::from_f32_as_q4(&g.ctx, &w, &[k, n], group).unwrap();
            assert_eq!(w_q.q4_group(), Some(group));
            assert_eq!(
                w_q.size_in_bytes(),
                4 * q4::packed_words(k, n, group) as u64
            );
            let w_dq = q4_round(&w, k, n, group);

            let y = zeros(&[m, n]);
            g.engine.matmul_q4b(&g.ctx, &upload(&x, &[m, k]), &w_q, &y);
            assert_close(
                &format!("matmul_q4b G={size} {m}x{k}x{n}"),
                &read(&y),
                &cpu_matmul(&x, &w_dq, m, k, n),
                1e-4,
                1e-4,
            );

            let dy = rand_vec(m * n, 1700 + seed as u64);
            let dx = zeros(&[m, k]);
            g.engine
                .matmul_bt_q4b(&g.ctx, &upload(&dy, &[m, n]), &w_q, &dx);
            assert_close(
                &format!("matmul_bt_q4b G={size} {m}x{n}x{k}"),
                &read(&dx),
                &cpu_matmul(&dy, &transpose(&w_dq, k, n), m, n, k),
                1e-4,
                1e-4,
            );
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn q4_weights_take_four_and_a_half_bits_each() {
    // 4-bit codes plus one 32-bit scale/min word per 64 weights, counted by
    // the memory tracker under the current category, apart from f16.
    let g = gpu();
    let (k, n) = (2048, 2048);
    let w = rand_vec(k * n, 800);
    let before = *g
        .ctx
        .memory()
        .current()
        .category(MemoryCategory::BaseWeights);
    let w_q = {
        let _tag = g.ctx.memory().scope(MemoryCategory::BaseWeights);
        UnifiedTensor::from_f32_as_q4(&g.ctx, &w, &[k, n], Q4Group::DEFAULT).unwrap()
    };
    let bytes = (k * n / 2 + 4 * (k * n / 64)) as u64;
    assert_eq!(bytes, 2_359_296);
    assert_eq!(w_q.size_in_bytes(), bytes);
    assert_eq!(w_q.memory_category(), MemoryCategory::BaseWeights);
    let after = *g
        .ctx
        .memory()
        .current()
        .category(MemoryCategory::BaseWeights);
    assert_eq!(after.allocated_bytes - before.allocated_bytes, bytes);
    assert_eq!(after.logical_bytes - before.logical_bytes, bytes);
    assert_eq!(after.q4_bytes - before.q4_bytes, bytes);
    assert_eq!(after.f16_bytes, before.f16_bytes);
    drop(w_q);
    let freed = *g
        .ctx
        .memory()
        .current()
        .category(MemoryCategory::BaseWeights);
    assert_eq!(freed.q4_bytes, before.q4_bytes);
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn q4_upload_rejects_non_finite_weights() {
    let g = gpu();
    let live = g.ctx.memory().live_bytes();
    let err = UnifiedTensor::from_f32_as_q4(
        &g.ctx,
        &[0.0, f32::NAN, 1.0, 2.0],
        &[2, 2],
        Q4Group::DEFAULT,
    )
    .unwrap_err();
    assert!(
        matches!(err, q4::Q4Error::NonFinite { row: 0, col: 1, .. }),
        "{err:?}"
    );
    assert_eq!(g.ctx.memory().live_bytes(), live, "nothing was uploaded");
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

/// `(m, k, n)` of products with a short output side, which run the thin
/// kernel: M of 2, 3, 63 and 64 against wide N; K of 1 and off the 32 a
/// workgroup steps by; N of 1 and wider than a vocabulary block; the LoRA
/// shapes of rank 8 and rank 1 (`x @ A` with 8 or 1 columns, `(x A) @ B`
/// with K of 8 or 1, `dB` with 8 rows).
const THIN_SHAPES: &[(usize, usize, usize)] = &[
    (2, 37, 1),
    (3, 1000, 5000),
    (2, 2049, 7),
    (63, 33, 129),
    (64, 576, 192),
    (200, 576, 8),
    (40, 8, 576),
    (300, 64, 1),
    (40, 1, 576),
    (8, 63, 2048),
];

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn thin_matmul_matches_cpu_in_every_layout() {
    // f32 in all three layouts, and f16 `b` forward and transposed. With f32
    // the short side may be N (the kernel then runs on the transposed
    // product); f16 `b` with few columns takes the tile kernel.
    let g = gpu();
    for (seed, &(m, k, n)) in THIN_SHAPES.iter().enumerate() {
        let a = maybe_f16(rand_vec(m * k, 3000 + seed as u64));
        let b: Vec<f32> = rand_vec(k * n, 3100 + seed as u64)
            .iter()
            .map(|v| v * 0.1)
            .collect();
        let check = |name: &str, out: &UnifiedTensor, b_seen: &[f32]| {
            assert_close(
                &format!("{name} {m}x{k}x{n}"),
                &read(out),
                &cpu_matmul(&a, b_seen, m, k, n),
                1e-4,
                1e-4,
            );
        };
        let (bt, at) = (transpose(&b, k, n), transpose(&a, m, k));
        let a_up = upload(&a, &[m, k]);

        let out = zeros(&[m, n]);
        g.engine.matmul(&g.ctx, &a_up, &upload(&b, &[k, n]), &out);
        check("matmul", &out, &b);
        let out = zeros(&[m, n]);
        g.engine
            .matmul_bt(&g.ctx, &a_up, &upload(&bt, &[n, k]), &out);
        check("matmul_bt", &out, &b);
        let out = zeros(&[m, n]);
        g.engine
            .matmul_at(&g.ctx, &upload(&at, &[k, m]), &upload(&b, &[k, n]), &out);
        check("matmul_at", &out, &b);

        let b_h = UnifiedTensor::from_f32_as_f16(&g.ctx, &b, &[k, n]);
        let out = zeros(&[m, n]);
        g.engine.matmul_f16b(&g.ctx, &a_up, &b_h, &out);
        check("matmul_f16b", &out, &f16_round(&b));
        let bt_h = UnifiedTensor::from_f32_as_f16(&g.ctx, &bt, &[n, k]);
        let out = zeros(&[m, n]);
        g.engine.matmul_bt_f16b(&g.ctx, &a_up, &bt_h, &out);
        check("matmul_bt_f16b", &out, &f16_round(&b));
    }
}

/// `(m, k, n)` for the skinny kernel: the LoRA `x @ A` of rank 8 at 256 and
/// 2048 rows and at the 1.7B MLP's K of 8192, then M, K and N off every
/// tile's multiple, one column, and the widest N it takes.
const SKINNY_SHAPES: &[(usize, usize, usize)] = &[
    (256, 576, 8),
    (2048, 576, 8),
    (256, 8192, 8),
    (200, 40, 8),
    (77, 37, 5),
    (300, 64, 1),
    (130, 1000, 64),
];

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn every_skinny_tile_matches_cpu_in_both_layouts() {
    let g = gpu();
    for (seed, &(m, k, n)) in SKINNY_SHAPES.iter().enumerate() {
        let a = rand_vec(m * k, 3400 + seed as u64);
        let b: Vec<f32> = rand_vec(k * n, 3500 + seed as u64)
            .iter()
            .map(|v| v * 0.1)
            .collect();
        let old = rand_vec(m * n, 3600 + seed as u64);
        let product = cpu_matmul(&a, &b, m, k, n);
        let a_up = upload(&a, &[m, k]);
        let layouts = [
            (false, upload(&b, &[k, n])),
            (true, upload(&transpose(&b, k, n), &[n, k])),
        ];
        for (transposed_b, b_up) in &layouts {
            for tile in SkinnyTile::ALL {
                for (alpha, beta) in [(1.0f32, 0.0f32), (-0.5, 1.0)] {
                    let out = upload(&old, &[m, n]);
                    let ep = Epilogue { alpha, beta };
                    let tensors = [&a_up, b_up, &out];
                    let ran =
                        g.engine
                            .matmul_skinny_into(&g.ctx, *transposed_b, tensors, Some(tile), ep);
                    assert_eq!(ran, Ok(tile));
                    let want: Vec<f32> = product
                        .iter()
                        .zip(&old)
                        .map(|(&p, &o)| alpha * p + beta * o)
                        .collect();
                    // Names the tile, shape, layout, alpha and beta.
                    let name = format!("{tile:?} {m}x{k}x{n} {transposed_b} {alpha} {beta}");
                    assert_close(&name, &read(&out), &want, 1e-4, 1e-4);
                }
            }
        }
    }
}

/// `(m, k, n)` of products whose `b` is stored along k: `a @ b^T` in f32 and
/// f16, and `x @ W` with 4-bit `W`. K is off the 16 the tile kernel stages
/// at a time and N off its 64 columns. M and N above 64 (for f16 and 4-bit
/// `b`, M above 64 alone) take the 64x64-tile kernel; (37, 70, 45) and
/// (1, 33, 17) take the thin and matrix-vector kernels, checked against the
/// same reference.
const ALONG_K_SHAPES: &[(usize, usize, usize)] = &[
    (66, 2048, 96),
    (67, 70, 65),
    (129, 33, 17),
    (65, 1, 130),
    (100, 31, 191),
    (37, 70, 45),
    (1, 33, 17),
];

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn matmul_with_b_along_k_matches_cpu_at_odd_shapes() {
    let g = gpu();
    for (seed, &(m, k, n)) in ALONG_K_SHAPES.iter().enumerate() {
        let a = maybe_f16(rand_vec(m * k, 3200 + seed as u64));
        // Weight-sized values keep the k = 2048 sums near 1, so the
        // tolerance below measures the kernel, not f32 rounding.
        let b: Vec<f32> = rand_vec(k * n, 3300 + seed as u64)
            .iter()
            .map(|v| v * 0.1)
            .collect();
        let check = |name: &str, out: &UnifiedTensor, b_seen: &[f32]| {
            assert_close(
                &format!("{name} {m}x{k}x{n}"),
                &read(out),
                &cpu_matmul(&a, b_seen, m, k, n),
                1e-4,
                1e-4,
            );
        };
        let bt = transpose(&b, k, n); // [n, k]
        let a_up = upload(&a, &[m, k]);

        let out = zeros(&[m, n]);
        g.engine
            .matmul_bt(&g.ctx, &a_up, &upload(&bt, &[n, k]), &out);
        check("matmul_bt", &out, &b);

        let bt_h = UnifiedTensor::from_f32_as_f16(&g.ctx, &bt, &[n, k]);
        let out = zeros(&[m, n]);
        g.engine.matmul_bt_f16b(&g.ctx, &a_up, &bt_h, &out);
        check("matmul_bt_f16b", &out, &f16_round(&b));

        // The 4-bit format stores W[k, n] in [n, k] order, so the forward
        // product is the one that reads it along k.
        let w_q = UnifiedTensor::from_f32_as_q4(&g.ctx, &b, &[k, n], Q4Group::DEFAULT).unwrap();
        let out = zeros(&[m, n]);
        g.engine.matmul_q4b(&g.ctx, &a_up, &w_q, &out);
        check("matmul_q4b", &out, &q4_round(&b, k, n, Q4Group::DEFAULT));
    }
}

/// `(m, k, n)` of products the small-M kernel takes: M of one and of two row
/// blocks, K and N off the 64 columns and 16 or 32 of K a workgroup steps by.
const SMALL_M_SHAPES: &[(usize, usize, usize)] = &[
    (17, 259, 300),
    (37, 1000, 257),
    (64, 512, 256),
    (100, 333, 321),
    (130, 289, 263),
    (256, 260, 512),
];

/// `b` of a small-M test as `[K, N]` and as `[N, K]`, in f32 and in f16, with
/// the reference product for f32 and for f16 `b`.
struct SmallMWeights {
    weights: Vec<(&'static str, bool, UnifiedTensor)>,
    want: Vec<f32>,
    want_h: Vec<f32>,
}

impl SmallMWeights {
    fn new(a: &[f32], (m, k, n): (usize, usize, usize), seed: u64) -> Self {
        let g = gpu();
        let b: Vec<f32> = rand_vec(k * n, seed).iter().map(|v| v * 0.1).collect();
        let bt = transpose(&b, k, n);
        Self {
            weights: vec![
                ("f32 a@b", false, upload(&b, &[k, n])),
                ("f32 a@bT", true, upload(&bt, &[n, k])),
                (
                    "f16 a@b",
                    false,
                    UnifiedTensor::from_f32_as_f16(&g.ctx, &b, &[k, n]),
                ),
                (
                    "f16 a@bT",
                    true,
                    UnifiedTensor::from_f32_as_f16(&g.ctx, &bt, &[n, k]),
                ),
            ],
            want: cpu_matmul(a, &b, m, k, n),
            want_h: cpu_matmul(a, &f16_round(&b), m, k, n),
        }
    }

    fn want(&self, b: &UnifiedTensor) -> &[f32] {
        if b.dtype == DType::FP16 {
            &self.want_h
        } else {
            &self.want
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn small_m_matmul_matches_cpu_for_every_split() {
    // Both layouts, f32 and f16 b, the split the heuristic picks and 1, 2, 3
    // and 7 slices of K.
    let g = gpu();
    for (seed, &(m, k, n)) in SMALL_M_SHAPES.iter().enumerate() {
        let a = rand_vec(m * k, 5000 + seed as u64);
        let a_up = upload(&a, &[m, k]);
        let w = SmallMWeights::new(&a, (m, k, n), 5100 + seed as u64);
        for (name, transposed_b, b) in &w.weights {
            for split in [None, Some(1), Some(2), Some(3), Some(7)] {
                let out = zeros(&[m, n]);
                let launch = g
                    .engine
                    .matmul_with_path(
                        &g.ctx,
                        *transposed_b,
                        &a_up,
                        b,
                        &out,
                        MatmulPath::SmallM { tile: None, split },
                    )
                    .unwrap()
                    .unwrap();
                // K of 256 or more is at least 8 steps, so 2 slices of whole
                // steps always cover it; 3 and 7 may round down.
                match split {
                    Some(1) => assert_eq!(launch.split, 1),
                    Some(2) => assert_eq!(launch.split, 2),
                    Some(s) => assert!((1..=s).contains(&launch.split), "{launch:?}"),
                    None => assert!(launch.split >= 1),
                }
                assert_close(
                    &format!("small_m {name} {m}x{k}x{n} {launch:?}"),
                    &read(&out),
                    w.want(b),
                    1e-4,
                    1e-4,
                );
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn small_m_matmul_matches_cpu_with_odd_tiles() {
    // Two threads along M with K steps of 32; 5 threads along M by 8 along
    // N with 8 columns each, whose 40 threads leave the last round of staged
    // values short; and one thread. All fit in 16 KiB of workgroup memory.
    let g = gpu();
    let tiles = [
        SmallMTile {
            tm: 4,
            tn: 4,
            tx: 8,
            ty: 2,
            bk: 32,
        },
        SmallMTile {
            tm: 3,
            tn: 8,
            tx: 8,
            ty: 5,
            bk: 16,
        },
        SmallMTile {
            tm: 1,
            tn: 4,
            tx: 1,
            ty: 1,
            bk: 4,
        },
    ];
    for (seed, &(m, k, n)) in [(17, 259, 300), (130, 289, 263)].iter().enumerate() {
        let a = rand_vec(m * k, 5200 + seed as u64);
        let a_up = upload(&a, &[m, k]);
        let w = SmallMWeights::new(&a, (m, k, n), 5300 + seed as u64);
        for (name, transposed_b, b) in &w.weights {
            for tile in tiles {
                for split in [Some(1), Some(3)] {
                    let out = zeros(&[m, n]);
                    let path = MatmulPath::SmallM {
                        tile: Some(tile),
                        split,
                    };
                    let launch = g
                        .engine
                        .matmul_with_path(&g.ctx, *transposed_b, &a_up, b, &out, path)
                        .unwrap()
                        .unwrap();
                    assert_eq!(launch.tile, tile);
                    assert_close(
                        &format!("small_m {name} {m}x{k}x{n} {launch:?}"),
                        &read(&out),
                        w.want(b),
                        1e-4,
                        1e-4,
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn small_m_matmul_is_deterministic() {
    // The split's partial sums are added in a fixed order, so two runs give
    // the same bits.
    let g = gpu();
    let (m, k, n) = (130, 1000, 263);
    let a = rand_vec(m * k, 5400);
    let a_up = upload(&a, &[m, k]);
    let w = SmallMWeights::new(&a, (m, k, n), 5401);
    for (name, transposed_b, b) in &w.weights {
        for split in [Some(1), Some(5)] {
            let path = MatmulPath::SmallM { tile: None, split };
            let run = || {
                let out = zeros(&[m, n]);
                g.engine
                    .matmul_with_path(&g.ctx, *transposed_b, &a_up, b, &out, path)
                    .unwrap();
                read(&out).iter().map(|v| v.to_bits()).collect::<Vec<u32>>()
            };
            assert_eq!(run(), run(), "small_m {name} split {split:?}");
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn matmul_routes_few_rows_against_wide_weights_to_small_m() {
    let g = gpu();
    let route = |(m, k, n): (usize, usize, usize), transposed_b: bool, dtype: DType| {
        let a = upload(&rand_vec(m * k, 5500), &[m, k]);
        let shape = if transposed_b { [n, k] } else { [k, n] };
        let data = rand_vec(k * n, 5501);
        let b = match dtype {
            DType::FP16 => UnifiedTensor::from_f32_as_f16(&g.ctx, &data, &shape),
            _ => upload(&data, &shape),
        };
        let out = zeros(&[m, n]);
        g.engine
            .matmul_with_path(&g.ctx, transposed_b, &a, &b, &out, MatmulPath::Auto)
            .unwrap()
            .map(|launch| launch.tile)
    };
    // 64 rows: 8 threads along M with 8 rows each, 16 along N with 4
    // columns each. K steps of 32 need 36 KiB of workgroup memory (NVIDIA's
    // Vulkan driver has 48), steps of 16 need 20 KiB (Metal and software
    // Vulkan have 32); the router takes no smaller step, so with 16 KiB
    // (DX12 on WARP) the previous kernels run.
    let bytes = g.ctx.device.limits().max_compute_workgroup_storage_size;
    // With the cooperative-matrix kernel routed, f16 weights with 64 or more rows go there first.
    let coop = g.engine.coop_matmul_active();
    let tile = |bk| SmallMTile {
        tm: 8,
        tn: 4,
        tx: 16,
        ty: 8,
        bk,
    };
    let want = if bytes >= 36 * 1024 {
        Some(tile(32))
    } else if bytes >= 20 * 1024 {
        Some(tile(16))
    } else {
        None
    };
    assert_eq!(route((64, 512, 256), true, DType::FP32), want, "{bytes} B");
    assert_eq!(
        route((64, 512, 256), false, DType::FP16),
        if coop { None } else { want },
        "{bytes} B"
    );
    // 160 rows: two row blocks of 80, with K steps of 16 in 22.5 KiB.
    assert_eq!(
        route((160, 256, 300), true, DType::FP16).map(SmallMTile::bm),
        (bytes >= 23_040 && !coop).then_some(80),
        "{bytes} B"
    );
    // Too narrow, too shallow, too few or too many rows: the previous kernels.
    assert_eq!(route((64, 512, 192), true, DType::FP32), None);
    assert_eq!(route((64, 192, 512), false, DType::FP32), None);
    assert_eq!(route((15, 512, 512), true, DType::FP32), None);
    assert_eq!(route((257, 512, 512), true, DType::FP32), None);
    assert_eq!(route((64, 2048, 8), true, DType::FP32), None);
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn small_m_matmul_refuses_what_it_cannot_run() {
    let g = gpu();
    let (m, k, n) = (64, 256, 256);
    let a = upload(&rand_vec(m * k, 5600), &[m, k]);
    let data = rand_vec(k * n, 5601);
    let out = zeros(&[m, n]);
    let small_m = |tile| MatmulPath::SmallM { tile, split: None };
    // A 4-bit weight.
    let b_q4 = UnifiedTensor::from_f32_as_q4(&g.ctx, &data, &[k, n], Q4Group::DEFAULT).unwrap();
    assert!(g
        .engine
        .matmul_with_path(&g.ctx, false, &a, &b_q4, &out, small_m(None))
        .is_err());
    let b = upload(&data, &[k, n]);
    // Columns per thread not a multiple of 4, and more threads than a
    // workgroup can have.
    for tile in [
        SmallMTile {
            tm: 8,
            tn: 6,
            tx: 16,
            ty: 8,
            bk: 32,
        },
        SmallMTile {
            tm: 1,
            tn: 4,
            tx: 64,
            ty: 64,
            bk: 4,
        },
    ] {
        assert!(g
            .engine
            .matmul_with_path(&g.ctx, false, &a, &b, &out, small_m(Some(tile)))
            .is_err());
    }
    // Shapes that do not match.
    assert!(g
        .engine
        .matmul_with_path(&g.ctx, true, &a, &b, &zeros(&[m, n + 1]), small_m(None))
        .is_err());
}

// ---------------------------------------------------------------------------
// Element-wise ops and their backward passes
// ---------------------------------------------------------------------------

const ELEMWISE_LEN: usize = 1000; // not a multiple of the workgroup size

type BinaryOp = fn(&ComputeEngine, &WgpuContext, &UnifiedTensor, &UnifiedTensor, &UnifiedTensor);
type UnaryOp = fn(&ComputeEngine, &WgpuContext, &UnifiedTensor, &UnifiedTensor);
/// (name, GPU op, CPU reference)
type BinaryCase = (&'static str, BinaryOp, fn(f32, f32) -> f32);
type UnaryCase = (&'static str, UnaryOp, fn(f32) -> f32);

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn binary_ops_match_cpu() {
    let g = gpu();
    let n = ELEMWISE_LEN;
    let a = rand_vec(n, 1);
    let b = rand_vec(n, 2);
    let (ta, tb) = (upload(&a, &[n]), upload(&b, &[n]));
    let cases: [BinaryCase; 5] = [
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
    let cases: [UnaryCase; 3] = [
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
    let (rows, len) = (4, 96);
    let x: Vec<f32> = rand_vec(rows * len, 60).iter().map(|v| v * 0.01).collect();
    let w = rand_vec(len, 61);
    let out = zeros(&[rows, len]);
    g.engine.rms_norm_eps(
        &g.ctx,
        &upload(&x, &[rows, len]),
        &upload(&w, &[len]),
        &out,
        1e-2,
    );
    let want = cpu_rms_norm(&x, &w, len, 1e-2);
    assert_close("rms_norm_eps", &read(&out), &want, 1e-5, 1e-4);
}

/// Gradients of RMSNorm over rows of `len`: `(grad_x, grad_w)`.
fn cpu_rms_norm_backward(
    x: &[f32],
    g: &[f32],
    w: &[f32],
    len: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0f32; x.len()];
    let mut gw = vec![0.0f32; len];
    for ((xr, gr), gxr) in x.chunks(len).zip(g.chunks(len)).zip(gx.chunks_mut(len)) {
        let inv = 1.0 / (xr.iter().map(|v| v * v).sum::<f32>() / len as f32 + eps).sqrt();
        let dot: f32 = (0..len).map(|i| gr[i] * w[i] * xr[i]).sum();
        for i in 0..len {
            gw[i] += gr[i] * xr[i] * inv;
            gxr[i] = gr[i] * w[i] * inv - xr[i] * dot * inv.powi(3) / len as f32;
        }
    }
    (gx, gw)
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn rms_norm_backward_matches_cpu() {
    let g = gpu();
    for &(rows, len, eps) in &[
        (1, 1, 1e-5),
        (5, 37, 1e-5),
        (3, 64, 0.1),
        (2, 4096, 1e-5),
        (130, 576, 1e-5),
    ] {
        let x = rand_vec(rows * len, 70 + len as u64);
        let w = rand_vec(len, 71 + len as u64);
        let go = rand_vec(rows * len, 72 + len as u64);
        let gx = zeros(&[rows, len]);
        let gw = zeros(&[len]);
        let (tx, tw, tgo) = (
            upload(&x, &[rows, len]),
            upload(&w, &[len]),
            upload(&go, &[rows, len]),
        );
        let e = &g.engine;
        e.rms_norm_backward(&g.ctx, &tx, &tw, &tgo, &gx, Some(&gw), eps);
        let (want_x, want_w) = cpu_rms_norm_backward(&x, &go, &w, len, eps);
        let name = format!("rms_norm_backward {rows}x{len}");
        assert_close(&format!("{name} grad_x"), &read(&gx), &want_x, 1e-4, 1e-3);
        assert_close(&format!("{name} grad_w"), &read(&gw), &want_w, 1e-4, 1e-3);
        // A frozen weight gets no gradient: the input gradient is the same.
        let gx_only = zeros(&[rows, len]);
        e.rms_norm_backward(&g.ctx, &tx, &tw, &tgo, &gx_only, None, eps);
        let bits = |t: &UnifiedTensor| read(t).iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&gx_only), bits(&gx), "{name} grad_x without grad_w");
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

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn masked_cross_entropy_matches_cpu() {
    let g = gpu();
    for &(rows, vocab) in &[(1, 1), (4, 7), (9, 300), (3, 49152)] {
        let logits: Vec<f32> = rand_vec(rows * vocab, 80 + vocab as u64)
            .iter()
            .map(|v| v * 8.0)
            .collect();
        // Every third row is skipped; the others target a varying id.
        let targets: Vec<u32> = (0..rows)
            .map(|r| {
                if r % 3 == 1 {
                    MASKED_CE_SKIP
                } else {
                    ((r * 7919) % vocab) as u32
                }
            })
            .collect();
        let supervised = targets.iter().filter(|&&t| t != MASKED_CE_SKIP).count();
        let inv = 1.0 / supervised as f32;
        let target_buf = UnifiedTensor::from_bytes(&g.ctx, bytemuck::cast_slice(&targets), &[rows]);
        let grad = zeros(&[rows, vocab]);
        let loss = zeros(&[rows]);
        g.engine.masked_cross_entropy(
            &g.ctx,
            &upload(&logits, &[rows, vocab]),
            &target_buf,
            inv,
            &grad,
            &loss,
        );

        let probs = cpu_softmax_rows(&logits, vocab);
        let mut want_grad = vec![0.0f32; rows * vocab];
        let mut want_loss = vec![0.0f32; rows];
        for (r, &t) in targets.iter().enumerate() {
            if t == MASKED_CE_SKIP {
                continue;
            }
            let t = t as usize;
            want_loss[r] = -probs[r * vocab + t].ln();
            for i in 0..vocab {
                want_grad[r * vocab + i] = probs[r * vocab + i] * inv;
            }
            want_grad[r * vocab + t] -= inv;
        }
        let name = format!("masked_cross_entropy {rows}x{vocab}");
        assert_close(
            &format!("{name} loss"),
            &read(&loss),
            &want_loss,
            1e-4,
            1e-4,
        );
        assert_close(
            &format!("{name} grad"),
            &read(&grad),
            &want_grad,
            1e-6,
            1e-3,
        );
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn masked_cross_entropy_weighted_scales_each_row() {
    let g = gpu();
    let (rows, vocab) = (6, 97);
    let logits = rand_vec(rows * vocab, 500)
        .iter()
        .map(|v| v * 4.0)
        .collect::<Vec<_>>();
    let targets: Vec<u32> = vec![3, MASKED_CE_SKIP, 50, 96, 0, MASKED_CE_SKIP];
    let weights: Vec<f32> = vec![0.5, 0.5, 0.25, 0.25, 0.25, 1.0];
    let target_buf = UnifiedTensor::from_bytes(&g.ctx, bytemuck::cast_slice(&targets), &[rows]);
    let grad = zeros(&[rows, vocab]);
    let loss = zeros(&[rows]);
    g.engine.masked_cross_entropy_weighted(
        &g.ctx,
        &upload(&logits, &[rows, vocab]),
        &target_buf,
        &upload(&weights, &[rows]),
        &grad,
        &loss,
    );
    let probs = cpu_softmax_rows(&logits, vocab);
    let mut want_grad = vec![0.0f32; rows * vocab];
    let mut want_loss = vec![0.0f32; rows];
    for (r, &t) in targets.iter().enumerate() {
        if t == MASKED_CE_SKIP {
            continue;
        }
        let t = t as usize;
        want_loss[r] = -probs[r * vocab + t].ln();
        for i in 0..vocab {
            want_grad[r * vocab + i] = probs[r * vocab + i] * weights[r];
        }
        want_grad[r * vocab + t] -= weights[r];
    }
    assert_close("weighted loss", &read(&loss), &want_loss, 1e-4, 1e-4);
    assert_close("weighted grad", &read(&grad), &want_grad, 1e-6, 1e-3);
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn sum_squares_partial_adds_up_to_the_squared_norm() {
    let g = gpu();
    let sizes = [
        1,
        SUM_SQUARES_CHUNK - 1,
        SUM_SQUARES_CHUNK,
        3 * SUM_SQUARES_CHUNK + 5,
    ];
    let inputs: Vec<Vec<f32>> = sizes.iter().map(|&n| rand_vec(n, 90 + n as u64)).collect();
    let chunks: usize = sizes.iter().map(|n| n.div_ceil(SUM_SQUARES_CHUNK)).sum();
    let out = zeros(&[chunks]);
    let mut offset = 0;
    let mut want = Vec::new();
    for x in &inputs {
        let written = g
            .engine
            .sum_squares_partial(&g.ctx, &upload(x, &[x.len()]), &out, offset);
        assert_eq!(written, x.len().div_ceil(SUM_SQUARES_CHUNK));
        offset += written;
        want.extend(
            x.chunks(SUM_SQUARES_CHUNK)
                .map(|c| c.iter().map(|v| v * v).sum::<f32>()),
        );
    }
    assert_close("sum_squares_partial", &read(&out), &want, 1e-4, 1e-4);
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

/// Rotate-half RoPE over [batch, seq, heads * head_dim]; position of token s
/// is `pos + s` for every head.
fn cpu_rope(x: &[f32], batch: usize, seq: usize, heads: usize, hd: usize, pos: usize) -> Vec<f32> {
    cpu_rope_dir(x, batch, seq, heads, hd, pos, 1.0)
}

/// [`cpu_rope`] rotating by `dir` times the angle (-1.0 is the inverse).
fn cpu_rope_dir(
    x: &[f32],
    batch: usize,
    seq: usize,
    heads: usize,
    hd: usize,
    pos: usize,
    dir: f32,
) -> Vec<f32> {
    let mut out = x.to_vec();
    let half = hd / 2;
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..heads {
                let base = ((b * seq + s) * heads + h) * hd;
                let p = (pos + s) as f32;
                for i in 0..half {
                    let freq = 10000f32.powf(-2.0 * i as f32 / hd as f32);
                    let (sin, cos) = (dir * p * freq).sin_cos();
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
    // 2-D [seq, heads * head_dim] input: rows are tokens.
    let (seq, heads, hd, pos) = (6, 3, 16, 2);
    let x = rand_vec(seq * heads * hd, 62);
    let t = upload(&x, &[seq, heads * hd]);
    g.engine.rope(&g.ctx, &t, hd, pos);
    let want = cpu_rope(&x, 1, seq, heads, hd, pos);
    assert_close("rope 2-D", &read(&t), &want, 1e-4, 1e-4);
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn rope_inverse_matches_cpu_and_undoes_rope() {
    let g = gpu();
    for &(batch, seq, heads, hd, pos) in &[(1, 1, 1, 8, 0), (1, 5, 1, 64, 3), (2, 7, 4, 32, 11)] {
        let x = rand_vec(batch * seq * heads * hd, 13);
        let shape = [batch, seq, heads * hd];
        let name = format!("rope inverse b{batch} s{seq} h{heads} d{hd}");

        let t = upload(&x, &shape);
        g.engine.rope_theta_inverse(&g.ctx, &t, hd, pos, 10000.0);
        let want = cpu_rope_dir(&x, batch, seq, heads, hd, pos, -1.0);
        assert_close(&name, &read(&t), &want, 1e-4, 1e-4);

        let t = upload(&x, &shape);
        g.engine.rope(&g.ctx, &t, hd, pos);
        g.engine.rope_theta_inverse(&g.ctx, &t, hd, pos, 10000.0);
        assert_close(&format!("{name} round trip"), &read(&t), &x, 1e-4, 1e-4);
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn rope_into_a_new_buffer_matches_in_place_and_keeps_the_input() {
    // Against the in-place kernel bit for bit, and the CPU rotation.
    let g = gpu();
    let e = &g.engine;
    let bits = |t: &UnifiedTensor| read(t).iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for &(batch, seq, heads, hd, pos) in &[(1, 1, 1, 8, 0), (1, 5, 1, 64, 3), (2, 7, 4, 32, 11)] {
        let x = rand_vec(batch * seq * heads * hd, 14);
        let shape = [batch, seq, heads * hd];
        let input = upload(&x, &shape);
        let x_bits = bits(&input);
        for inverse in [false, true] {
            let name = format!("rope_to {shape:?} inverse {inverse}");
            let in_place = upload(&x, &shape);
            let out = zeros(&shape);
            if inverse {
                e.rope_theta_inverse(&g.ctx, &in_place, hd, pos, 10000.0);
                e.rope_theta_inverse_to(&g.ctx, (&input, &out), hd, pos, 10000.0);
            } else {
                e.rope_theta(&g.ctx, &in_place, hd, pos, 10000.0);
                e.rope_theta_to(&g.ctx, (&input, &out), hd, pos, 10000.0);
            }
            assert_eq!(bits(&out), bits(&in_place), "{name}");
            assert_eq!(bits(&input), x_bits, "{name}: input written");
            let dir = if inverse { -1.0 } else { 1.0 };
            let want = cpu_rope_dir(&x, batch, seq, heads, hd, pos, dir);
            assert_close(&name, &read(&out), &want, 1e-4, 1e-4);
        }
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
        let (want, _) = cpu_attention(
            &q,
            &k,
            &v,
            None,
            (1, seq, seq, 1, 1, d),
            causal_rule(causal),
        );
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
        let (want, _) = cpu_attention(
            &q,
            &k,
            &v,
            None,
            (b, sq, sk, h, hkv, d),
            causal_rule(causal),
        );
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
    let (want, _) = cpu_attention(
        &q,
        &k,
        &v,
        Some(&mask),
        (b, sq, sk, h, h, d),
        causal_rule(false),
    );
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
        let (out, probs) = cpu_attention(&q, &k, &v, None, dims, causal_rule(causal));
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

/// Backward with groups of 3 query heads per KV head (the 135M, 360M and
/// 1.7B head counts, and a batch), where dK and dV go through per-head
/// partials and the group sum.
#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn flash_attention_v2_backward_sums_the_heads_of_a_group() {
    let g = gpu();
    // (batch, seq, heads, kv heads, head_dim)
    let cases: &[(usize, usize, usize, usize, usize)] = &[
        (1, 256, 9, 3, 64),
        (1, 100, 15, 5, 64),
        (2, 45, 6, 2, 64),
        (1, 33, 4, 1, 128),
        (1, 40, 8, 8, 64),
    ];
    for (n, &(b, s, h, hkv, d)) in cases.iter().enumerate() {
        let dims = (b, s, s, h, hkv, d);
        let seed = 200 + 4 * n as u64;
        let q = rand_vec(b * s * h * d, seed);
        let k = rand_vec(b * s * hkv * d, seed + 1);
        let v = rand_vec(b * s * hkv * d, seed + 2);
        let dout = rand_vec(b * s * h * d, seed + 3);
        let (out, probs) = cpu_attention(&q, &k, &v, None, dims, causal_rule(true));
        let (want_dq, want_dk, want_dv) = cpu_attention_backward(&q, &k, &v, &dout, &probs, dims);
        let (gq, gk, gv) = (
            zeros(&[b, s, h, d]),
            zeros(&[b, s, hkv, d]),
            zeros(&[b, s, hkv, d]),
        );
        g.engine.flash_attention_v2_backward(
            &g.ctx,
            &upload(&q, &[b, s, h, d]),
            &upload(&k, &[b, s, hkv, d]),
            &upload(&v, &[b, s, hkv, d]),
            &upload(&out, &[b, s, h, d]),
            &upload(&dout, &[b, s, h, d]),
            &gq,
            &gk,
            &gv,
            true,
        );
        let name = format!("b{b} s{s} h{h}/{hkv} d{d}");
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
    let (out, probs) = cpu_attention(&q, &k, &v, None, dims, causal_rule(true));
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

/// Log-sum-exp of the scaled (and masked) scores of every (batch, head,
/// query) row, in the order of the attention stats.
fn cpu_attention_lse(
    q: &[f32],
    k: &[f32],
    mask: Option<&[f32]>,
    (b, sq, sk, h, hkv, d): (usize, usize, usize, usize, usize, usize),
    visible: impl Fn(usize, usize) -> bool,
) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    let mut lse = Vec::with_capacity(b * h * sq);
    for bi in 0..b {
        for hi in 0..h {
            let kh = hi / (h / hkv);
            for i in 0..sq {
                let qo = ((bi * sq + i) * h + hi) * d;
                let s: Vec<f64> = (0..sk)
                    .filter(|&j| visible(i, j))
                    .map(|j| {
                        let ko = ((bi * sk + j) * hkv + kh) * d;
                        let dot: f64 = (0..d).map(|x| (q[qo + x] * k[ko + x]) as f64).sum();
                        let m = mask.map_or(0.0, |m| m[((bi * h + hi) * sq + i) * sk + j] as f64);
                        dot * scale as f64 + m
                    })
                    .collect();
                let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = s.iter().map(|x| (x - max).exp()).sum();
                lse.push((max + sum.ln()) as f32);
            }
        }
    }
    lse
}

/// (batch, seq, heads, kv_heads, head_dim) for the saved-stats tests: every
/// length is causal, lengths 17, 66 and 130 end inside a 16-row block and
/// 256 fills its blocks exactly.
const SAVED_STATS_CASES: &[(usize, usize, usize, usize, usize)] = &[
    (1, 1, 2, 2, 64),
    (1, 17, 2, 1, 64),
    (1, 66, 4, 4, 64),
    (2, 130, 4, 2, 64),
    (1, 66, 2, 2, 128),
    (1, 130, 2, 2, 96),
    (1, 256, 2, 2, 64),
    (1, 256, 2, 1, 128),
];

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn flash_attention_v2_saved_stats_match_cpu_and_the_recomputing_backward() {
    let g = gpu();
    for (seed, &(b, s, h, hkv, d)) in SAVED_STATS_CASES.iter().enumerate() {
        let dims = (b, s, s, h, hkv, d);
        let seed = 60 + 4 * seed as u64;
        let q = rand_vec(b * s * h * d, seed);
        let k = rand_vec(b * s * hkv * d, seed + 1);
        let v = rand_vec(b * s * hkv * d, seed + 2);
        let dout = rand_vec(b * s * h * d, seed + 3);
        let (want, probs) = cpu_attention(&q, &k, &v, None, dims, causal_rule(true));
        let (want_dq, want_dk, want_dv) = cpu_attention_backward(&q, &k, &v, &dout, &probs, dims);
        let want_lse = cpu_attention_lse(&q, &k, None, dims, causal_rule(true));

        let (tq, tk, tv, tdo) = (
            upload(&q, &[b, s, h, d]),
            upload(&k, &[b, s, hkv, d]),
            upload(&v, &[b, s, hkv, d]),
            upload(&dout, &[b, s, h, d]),
        );
        let opts = AttentionOptions {
            causal: true,
            ..AttentionOptions::default()
        };
        let out = zeros(&[b, s, h, d]);
        let stats = zeros(&[b, h, s, 2]);
        g.engine
            .flash_attention_v2_with_stats(&g.ctx, &tq, &tk, &tv, &out, &stats, &opts, None);
        let name = format!("b{b} s{s} h{h}/{hkv} d{d}");
        assert_close(&format!("fwd {name}"), &read(&out), &want, 1e-5, 1e-3);
        let lse: Vec<f32> = read(&stats).iter().skip(1).step_by(2).copied().collect();
        assert_close(&format!("lse {name}"), &lse, &want_lse, 1e-5, 1e-5);

        let grads = || {
            (
                zeros(&[b, s, h, d]),
                zeros(&[b, s, hkv, d]),
                zeros(&[b, s, hkv, d]),
            )
        };
        let (gq, gk, gv) = grads();
        g.engine.flash_attention_v2_backward_with_stats(
            &g.ctx, &tq, &tk, &tv, &out, &tdo, &gq, &gk, &gv, &stats, &opts, None,
        );
        let (rq, rk, rv) = grads();
        g.engine.flash_attention_v2_backward_with(
            &g.ctx, &tq, &tk, &tv, &out, &tdo, &rq, &rk, &rv, &opts, None,
        );
        let (gq, gk, gv) = (read(&gq), read(&gk), read(&gv));
        assert_close(&format!("dq {name}"), &gq, &want_dq, 1e-4, 1e-3);
        assert_close(&format!("dk {name}"), &gk, &want_dk, 1e-4, 1e-3);
        assert_close(&format!("dv {name}"), &gv, &want_dv, 1e-4, 1e-3);
        // The two backward paths differ only in how the log-sum-exp was
        // summed, so they agree far more closely than either with the CPU.
        assert_close(
            &format!("dq vs recompute {name}"),
            &gq,
            &read(&rq),
            1e-5,
            1e-4,
        );
        assert_close(
            &format!("dk vs recompute {name}"),
            &gk,
            &read(&rk),
            1e-5,
            1e-4,
        );
        assert_close(
            &format!("dv vs recompute {name}"),
            &gv,
            &read(&rv),
            1e-5,
            1e-4,
        );
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn flash_attention_v2_saved_stats_keep_mask_dropout_and_window() {
    let g = gpu();
    let (b, s, h, hkv, d) = (2, 45, 4, 2, 64);
    let q = rand_vec(b * s * h * d, 80);
    let k = rand_vec(b * s * hkv * d, 81);
    let v = rand_vec(b * s * hkv * d, 82);
    let dout = rand_vec(b * s * h * d, 83);
    // [batch, 1, 1, seq] padding mask: batch 1 hides its last 6 keys, so
    // its first rows still see keys and its later rows lose some.
    let mask: Vec<f32> = (0..b * s)
        .map(|i| {
            if i / s == 1 && i % s >= s - 6 {
                -1e9
            } else {
                0.0
            }
        })
        .collect();
    let (tq, tk, tv, tdo, tmask) = (
        upload(&q, &[b, s, h, d]),
        upload(&k, &[b, s, hkv, d]),
        upload(&v, &[b, s, hkv, d]),
        upload(&dout, &[b, s, h, d]),
        upload(&mask, &[b, 1, 1, s]),
    );
    let opts = AttentionOptions {
        causal: true,
        window: 20,
        dropout_prob: 0.2,
        seed: 0x1234_5678_9ABC_DEF0,
        ..AttentionOptions::default()
    };
    let out = zeros(&[b, s, h, d]);
    let stats = zeros(&[b, h, s, 2]);
    g.engine.flash_attention_v2_with_stats(
        &g.ctx,
        &tq,
        &tk,
        &tv,
        &out,
        &stats,
        &opts,
        Some(&tmask),
    );
    let plain = zeros(&[b, s, h, d]);
    g.engine
        .flash_attention_v2_with(&g.ctx, &tq, &tk, &tv, &plain, &opts, Some(&tmask));
    assert_eq!(read(&out), read(&plain), "saving stats changed the output");

    let mask_full: Vec<f32> = (0..b * h * s * s)
        .map(|i| mask[(i / (h * s * s)) * s + i % s])
        .collect();
    let visible = |i: usize, j: usize| j <= i && i - j < 20;
    let want_lse = cpu_attention_lse(&q, &k, Some(&mask_full), (b, s, s, h, hkv, d), visible);
    let lse: Vec<f32> = read(&stats).iter().skip(1).step_by(2).copied().collect();
    assert_close("lse mask window", &lse, &want_lse, 1e-5, 1e-5);

    let grads = || {
        (
            zeros(&[b, s, h, d]),
            zeros(&[b, s, hkv, d]),
            zeros(&[b, s, hkv, d]),
        )
    };
    let (gq, gk, gv) = grads();
    g.engine.flash_attention_v2_backward_with_stats(
        &g.ctx,
        &tq,
        &tk,
        &tv,
        &out,
        &tdo,
        &gq,
        &gk,
        &gv,
        &stats,
        &opts,
        Some(&tmask),
    );
    let (rq, rk, rv) = grads();
    g.engine.flash_attention_v2_backward_with(
        &g.ctx,
        &tq,
        &tk,
        &tv,
        &out,
        &tdo,
        &rq,
        &rk,
        &rv,
        &opts,
        Some(&tmask),
    );
    assert_close("dq mask dropout window", &read(&gq), &read(&rq), 1e-5, 1e-4);
    assert_close("dk mask dropout window", &read(&gk), &read(&rk), 1e-5, 1e-4);
    assert_close("dv mask dropout window", &read(&gv), &read(&rv), 1e-5, 1e-4);
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

/// CPU copy of the WGSL `hash3` used for attention dropout.
fn hash3(x: u32, y: u32, z: u32) -> f32 {
    let lcg = |v: u32| v.wrapping_mul(1664525).wrapping_add(1013904223);
    let (mut a, mut b, mut c) = (lcg(x), lcg(y), lcg(z));
    a = a.wrapping_add(b.wrapping_mul(c));
    b = b.wrapping_add(c.wrapping_mul(a));
    c = c.wrapping_add(a.wrapping_mul(b));
    a ^= a >> 16;
    b ^= b >> 16;
    c ^= c >> 16;
    a.wrapping_add(b).wrapping_add(c) as f32 * 2.328_306_4e-10
}

/// CPU copy of the WGSL `fmix32` (Murmur3 finalizer).
fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

/// CPU copy of the WGSL `dropout_scale` keep decision.
fn dropout_keep(prob: f32, seed: u64, bh: u32, q: u32, k: u32) -> bool {
    let key = fmix32(seed as u32 ^ fmix32((seed >> 32) as u32 ^ 0x9e37_79b9));
    hash3(bh ^ key, q, k) >= prob
}

/// The `[seq_q, seq_k]` keep mask the forward kernel draws for `seed`,
/// read back through the output: zero scores give uniform weights, and
/// one-hot values (`head_dim == seq_k`) put weight j of row i in out[i, j].
fn dropout_mask(prob: f32, seed: u64, sq: usize, sk: usize) -> Vec<bool> {
    let g = gpu();
    let q = vec![0.0f32; sq * sk];
    let k = vec![0.0f32; sk * sk];
    let v: Vec<f32> = (0..sk * sk)
        .map(|i| if i / sk == i % sk { 1.0 } else { 0.0 })
        .collect();
    let out = zeros(&[1, sq, 1, sk]);
    g.engine.flash_attention_v2(
        &g.ctx,
        &upload(&q, &[1, sq, 1, sk]),
        &upload(&k, &[1, sk, 1, sk]),
        &upload(&v, &[1, sk, 1, sk]),
        &out,
        false,
        prob,
        seed,
        None,
    );
    read(&out).iter().map(|&x| x > 0.0).collect()
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn flash_attention_v2_dropout_masks_of_nearby_seeds_are_unrelated() {
    // Masks used to hash (bh, q, k + seed): seed + 1 gave the mask of seed
    // shifted by one key, and seeds equal in their low 32 bits gave the same
    // mask. At p = 0.5 two unrelated masks agree at about half the positions.
    let (prob, sq, sk) = (0.5f32, 64, 32);
    let seed = 1000u64;
    let a = dropout_mask(prob, seed, sq, sk);
    for (idx, &kept) in a.iter().enumerate() {
        let want = dropout_keep(prob, seed, 0, (idx / sk) as u32, (idx % sk) as u32);
        assert_eq!(kept, want, "mask differs from the CPU reference at {idx}");
    }
    let kept = a.iter().filter(|&&x| x).count() as f32 / a.len() as f32;
    assert!((kept - 0.5).abs() < 0.1, "kept fraction {kept}");

    let agree = |pairs: &mut dyn Iterator<Item = (bool, bool)>| {
        let (mut same, mut n) = (0usize, 0usize);
        for (x, y) in pairs {
            same += (x == y) as usize;
            n += 1;
        }
        same as f32 / n as f32
    };
    let b = dropout_mask(prob, seed + 1, sq, sk);
    let shifted = agree(
        &mut (0..sq)
            .flat_map(|i| (1..sk).map(move |j| (i, j)))
            .map(|(i, j)| (a[i * sk + j], b[i * sk + j - 1])),
    );
    assert!(
        (shifted - 0.5).abs() < 0.1,
        "mask(seed + 1) is a shifted copy of mask(seed): agreement {shifted}"
    );
    let same_pos = agree(&mut a.iter().copied().zip(b.iter().copied()));
    assert!(
        (same_pos - 0.5).abs() < 0.1,
        "seed + 1 agreement {same_pos}"
    );
    let high = dropout_mask(prob, seed + (1 << 32), sq, sk);
    let high_agree = agree(&mut a.iter().copied().zip(high.iter().copied()));
    assert!(
        (high_agree - 0.5).abs() < 0.1,
        "high seed bits ignored: agreement {high_agree}"
    );
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn flash_attention_v2_mask_and_dropout_forward_backward_match_cpu() {
    let g = gpu();
    let (b, sq, sk, h, hkv, d) = (2, 21, 21, 4, 2, 32);
    // Both seed halves non-zero, so the kernels must read all 64 bits.
    let (prob, seed) = (0.2f32, 0x9E37_79B9_0000_004Du64);
    let q = rand_vec(b * sq * h * d, 44);
    let k = rand_vec(b * sk * hkv * d, 45);
    let v = rand_vec(b * sk * hkv * d, 46);
    let dout = rand_vec(b * sq * h * d, 47);
    // [batch, 1, 1, seq_k] padding mask, broadcast over heads and queries.
    let mask: Vec<f32> = (0..b * sk)
        .map(|i| {
            if i / sk == 1 && i % sk >= 17 {
                -1e9
            } else {
                0.0
            }
        })
        .collect();
    let mask_full: Vec<f32> = (0..b * h * sq * sk)
        .map(|i| mask[(i / (h * sq * sk)) * sk + i % sk])
        .collect();

    let (_, probs) = cpu_attention(
        &q,
        &k,
        &v,
        Some(&mask_full),
        (b, sq, sk, h, hkv, d),
        causal_rule(true),
    );
    let scale = 1.0 / (d as f32).sqrt();
    let keep = |bh: usize, i: usize, j: usize| {
        if dropout_keep(prob, seed, bh as u32, i as u32, j as u32) {
            1.0 / (1.0 - prob)
        } else {
            0.0
        }
    };
    let mut out = vec![0.0f32; q.len()];
    let (mut dq, mut dk, mut dv) = (
        vec![0.0f32; q.len()],
        vec![0.0; k.len()],
        vec![0.0; v.len()],
    );
    let mut row = 0;
    for bi in 0..b {
        for hi in 0..h {
            let kh = hi / (h / hkv);
            for i in 0..sq {
                let p = &probs[row];
                row += 1;
                let qo = ((bi * sq + i) * h + hi) * d;
                let kvo = |j: usize| ((bi * sk + j) * hkv + kh) * d;
                for (j, pj) in p.iter().enumerate() {
                    let m = keep(bi * h + hi, i, j);
                    for x in 0..d {
                        out[qo + x] += pj * m * v[kvo(j) + x];
                    }
                }
                let delta: f32 = (0..d).map(|x| dout[qo + x] * out[qo + x]).sum();
                for (j, pj) in p.iter().enumerate() {
                    let m = keep(bi * h + hi, i, j);
                    let dov: f32 = (0..d).map(|x| dout[qo + x] * v[kvo(j) + x]).sum();
                    let ds = pj * (m * dov - delta) * scale;
                    for x in 0..d {
                        dv[kvo(j) + x] += pj * m * dout[qo + x];
                        dq[qo + x] += ds * k[kvo(j) + x];
                        dk[kvo(j) + x] += ds * q[qo + x];
                    }
                }
            }
        }
    }

    let (tq, tk, tv) = (
        upload(&q, &[b, sq, h, d]),
        upload(&k, &[b, sk, hkv, d]),
        upload(&v, &[b, sk, hkv, d]),
    );
    let tmask = upload(&mask, &[b, 1, 1, sk]);
    let tout = zeros(&[b, sq, h, d]);
    g.engine
        .flash_attention_v2(&g.ctx, &tq, &tk, &tv, &tout, true, prob, seed, Some(&tmask));
    assert_close("dropout fwd", &read(&tout), &out, 1e-5, 1e-3);

    let (gq, gk, gv) = (
        zeros(&[b, sq, h, d]),
        zeros(&[b, sk, hkv, d]),
        zeros(&[b, sk, hkv, d]),
    );
    let opts = AttentionOptions {
        causal: true,
        dropout_prob: prob,
        seed,
        ..AttentionOptions::default()
    };
    g.engine.flash_attention_v2_backward_with(
        &g.ctx,
        &tq,
        &tk,
        &tv,
        &tout,
        &upload(&dout, &[b, sq, h, d]),
        &gq,
        &gk,
        &gv,
        &opts,
        Some(&tmask),
    );
    assert_close("dropout dq", &read(&gq), &dq, 1e-4, 1e-3);
    assert_close("dropout dk", &read(&gk), &dk, 1e-4, 1e-3);
    assert_close("dropout dv", &read(&gv), &dv, 1e-4, 1e-3);
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn flash_attention_v2_offset_and_window_match_cpu() {
    let g = gpu();
    // (seq_q, seq_k, causal, causal_offset, window)
    let cases = [
        (1, 40, true, 39, 0),   // decode one token against a filled cache
        (5, 64, true, 20, 0),   // chunk at positions 20..25 of a 64-slot cache
        (50, 50, true, 0, 8),   // causal sliding window
        (37, 37, false, 0, 5),  // two-sided window
        (70, 90, true, 10, 33), // offset + window, several key tiles
    ];
    let (b, h, hkv, d) = (1, 4, 2, 32);
    for (sq, sk, causal, offset, window) in cases {
        let dims = (b, sq, sk, h, hkv, d);
        let q = rand_vec(b * sq * h * d, 48);
        let k = rand_vec(b * sk * hkv * d, 49);
        let v = rand_vec(b * sk * hkv * d, 50);
        let dout = rand_vec(b * sq * h * d, 51);
        let visible = |i: usize, j: usize| {
            let qa = i + offset as usize;
            (!causal || j <= qa) && (window == 0 || qa.abs_diff(j) < window as usize)
        };
        let (want, probs) = cpu_attention(&q, &k, &v, None, dims, visible);
        let (want_dq, want_dk, want_dv) = cpu_attention_backward(&q, &k, &v, &dout, &probs, dims);

        let opts = AttentionOptions {
            causal,
            causal_offset: offset,
            window,
            ..AttentionOptions::default()
        };
        let (tq, tk, tv) = (
            upload(&q, &[b, sq, h, d]),
            upload(&k, &[b, sk, hkv, d]),
            upload(&v, &[b, sk, hkv, d]),
        );
        let out = zeros(&[b, sq, h, d]);
        g.engine
            .flash_attention_v2_with(&g.ctx, &tq, &tk, &tv, &out, &opts, None);
        let name = format!("q{sq} k{sk} causal={causal} offset={offset} window={window}");
        assert_close(&format!("fwd {name}"), &read(&out), &want, 1e-5, 1e-3);

        let (gq, gk, gv) = (
            zeros(&[b, sq, h, d]),
            zeros(&[b, sk, hkv, d]),
            zeros(&[b, sk, hkv, d]),
        );
        g.engine.flash_attention_v2_backward_with(
            &g.ctx,
            &tq,
            &tk,
            &tv,
            &out,
            &upload(&dout, &[b, sq, h, d]),
            &gq,
            &gk,
            &gv,
            &opts,
            None,
        );
        assert_close(&format!("dq {name}"), &read(&gq), &want_dq, 1e-4, 1e-3);
        assert_close(&format!("dk {name}"), &read(&gk), &want_dk, 1e-4, 1e-3);
        assert_close(&format!("dv {name}"), &read(&gv), &want_dv, 1e-4, 1e-3);
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn large_dispatches_use_a_2d_grid() {
    let g = gpu();
    // More than 65535 workgroups of 64 threads.
    let n = 65535 * 64 * 2 + 3;
    let a = rand_vec(n, 52);
    let b = rand_vec(n, 53);
    let out = zeros(&[n]);
    g.engine
        .add(&g.ctx, &upload(&a, &[n]), &upload(&b, &[n]), &out);
    let got = read(&out);
    for i in [0, n / 2, n - 1] {
        assert_eq!(got[i], a[i] + b[i], "add[{i}]");
    }
    g.engine.silu(&g.ctx, &upload(&a, &[n]), &out);
    let got = read(&out);
    for i in [0, n / 3, n - 1] {
        assert!((got[i] - a[i] * sigmoid(a[i])).abs() < 1e-6, "silu[{i}]");
    }

    // More than 65535 rows for the row-wise kernels.
    let (rows, len) = (70_000, 3);
    let x = rand_vec(rows * len, 54);
    let out = zeros(&[rows, len]);
    g.engine.softmax(&g.ctx, &upload(&x, &[rows, len]), &out);
    assert_close(
        "softmax 70k rows",
        &read(&out),
        &cpu_softmax_rows(&x, len),
        1e-6,
        1e-4,
    );
    let w = vec![1.0f32; len];
    g.engine
        .rms_norm(&g.ctx, &upload(&x, &[rows, len]), &upload(&w, &[len]), &out);
    assert_close(
        "rms_norm 70k rows",
        &read(&out),
        &cpu_rms_norm(&x, &w, len, 1e-5),
        1e-5,
        1e-4,
    );
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn self_check_passes_and_an_adapter_is_selected() {
    let g = gpu();
    for r in pollster::block_on(ferrotensor_backend::adapters::self_check(&g.ctx)).unwrap() {
        assert!(
            r.passed(),
            "{}: max error {} > {}",
            r.name,
            r.max_abs_err,
            r.tolerance
        );
    }
    let adapters = ferrotensor_backend::adapters::list_adapters();
    assert!(!adapters.is_empty());
    assert_eq!(
        adapters.iter().filter(|a| a.selected).count(),
        1,
        "{adapters:#?}"
    );
}

// ---------------------------------------------------------------------------
// Batched submission
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn write_data_runs_after_recorded_reads() {
    let g = gpu();
    let a = upload(&[1.0, 2.0, 3.0], &[3]);
    let b = zeros(&[3]);
    // The copy is only recorded; the write must not overtake it.
    g.engine.copy_tensor(&g.ctx, &a, &b, 0);
    a.write_data(&g.ctx, bytemuck::cast_slice(&[9.0f32, 9.0, 9.0]));
    assert_eq!(read(&b), vec![1.0, 2.0, 3.0]);
    assert_eq!(read(&a), vec![9.0, 9.0, 9.0]);
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn long_op_chains_keep_their_order() {
    // More ops than one submission holds, each depending on the last.
    let g = gpu();
    let one = upload(&[1.0; 64], &[64]);
    let mut x = zeros(&[64]);
    for _ in 0..1000 {
        let y = UnifiedTensor::empty(&g.ctx, &[64]);
        g.engine.add(&g.ctx, &x, &one, &y);
        x = y;
    }
    assert_eq!(read(&x), vec![1000.0; 64]);
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn q4_upload_rejects_non_matrix_shapes() {
    let g = gpu();
    let err =
        UnifiedTensor::from_f32_as_q4(&g.ctx, &[0.0; 8], &[2, 2, 2], Q4Group::DEFAULT).unwrap_err();
    assert_eq!(err, q4::Q4Error::NotMatrix { rank: 3 });
}

// ---------------------------------------------------------------------------
// Decoding one token
// ---------------------------------------------------------------------------

/// `(k, n)` of one-row products: K off the 16 threads that split it, N of
/// 1, odd N, and N as wide as a vocabulary.
const MATVEC_SHAPES: &[(usize, usize)] = &[
    (1, 1),
    (7, 1),
    (5, 7),
    (33, 129),
    (576, 576),
    (1000, 3),
    (64, 49_152),
];

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn matvec_matches_cpu_for_every_storage_kind() {
    // One-row products run the matrix-vector kernels. Each storage kind is
    // compared with the host product on the weights as the kernel sees
    // them: rounded to f16, or through the 4-bit round trip.
    let g = gpu();
    for (seed, &(k, n)) in MATVEC_SHAPES.iter().enumerate() {
        let x = rand_vec(k, 2000 + seed as u64);
        let w: Vec<f32> = rand_vec(k * n, 2100 + seed as u64)
            .iter()
            .map(|v| v * 0.1)
            .collect();
        let xt = upload(&x, &[1, k]);
        let check = |kind: &str, w_stored: &UnifiedTensor, w_seen: &[f32]| {
            let y = zeros(&[1, n]);
            match w_stored.dtype {
                DType::FP32 => g.engine.matmul(&g.ctx, &xt, w_stored, &y),
                DType::FP16 => g.engine.matmul_f16b(&g.ctx, &xt, w_stored, &y),
                DType::Q4 => g.engine.matmul_q4b(&g.ctx, &xt, w_stored, &y),
            }
            assert_close(
                &format!("matvec {kind} 1x{k}x{n}"),
                &read(&y),
                &cpu_matmul(&x, w_seen, 1, k, n),
                1e-4,
                1e-4,
            );
        };
        check("f32", &upload(&w, &[k, n]), &w);
        let w_h = UnifiedTensor::from_f32_as_f16(&g.ctx, &w, &[k, n]);
        check("f16", &w_h, &f16_round(&w));
        for size in Q4Group::SIZES {
            let group = Q4Group::new(size).unwrap();
            let w_q = UnifiedTensor::from_f32_as_q4(&g.ctx, &w, &[k, n], group).unwrap();
            check(&format!("q4 G={size}"), &w_q, &q4_round(&w, k, n, group));
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn one_query_attention_matches_the_many_query_kernel() {
    // One query runs the decode kernel. The same query as the second row of
    // a two-query call, one position earlier, runs the kernel that gives
    // each thread a query; both are also compared with the host. Cases:
    // (batch, cache slots, position, heads, kv heads, head_dim), with
    // caches longer than one 128-key tile, GQA and a head size off 4.
    let g = gpu();
    let cases = [
        (1, 8, 3, 1, 1, 16),
        (2, 37, 36, 4, 2, 64),
        (1, 300, 299, 2, 1, 64),
        (1, 300, 140, 9, 3, 64),
        (1, 130, 129, 2, 2, 128),
        (1, 20, 19, 2, 2, 6),
    ];
    for (i, &(b, sk, pos, h, hkv, d)) in cases.iter().enumerate() {
        let seed = 3000 + 10 * i as u64;
        let q = rand_vec(b * h * d, seed);
        let k = rand_vec(b * sk * hkv * d, seed + 1);
        let v = rand_vec(b * sk * hkv * d, seed + 2);
        let (tk, tv) = (upload(&k, &[b, sk, hkv, d]), upload(&v, &[b, sk, hkv, d]));
        let opts = |offset: usize| AttentionOptions {
            causal: true,
            causal_offset: offset as u32,
            ..AttentionOptions::default()
        };
        let one = zeros(&[b, 1, h, d]);
        let tq = upload(&q, &[b, 1, h, d]);
        g.engine
            .flash_attention_v2_with(&g.ctx, &tq, &tk, &tv, &one, &opts(pos), None);
        let got = read(&one);

        let filler = rand_vec(b * h * d, seed + 3);
        let mut two_q = Vec::new();
        for row in 0..b {
            two_q.extend_from_slice(&filler[row * h * d..(row + 1) * h * d]);
            two_q.extend_from_slice(&q[row * h * d..(row + 1) * h * d]);
        }
        let two = zeros(&[b, 2, h, d]);
        let tq2 = upload(&two_q, &[b, 2, h, d]);
        g.engine
            .flash_attention_v2_with(&g.ctx, &tq2, &tk, &tv, &two, &opts(pos - 1), None);
        let second: Vec<f32> = read(&two)
            .chunks(h * d)
            .skip(1)
            .step_by(2)
            .flatten()
            .copied()
            .collect();

        let name = format!("b{b} cache {sk} pos {pos} h{h}/{hkv} d{d}");
        assert_close(
            &format!("decode vs two-query kernel, {name}"),
            &got,
            &second,
            1e-5,
            1e-3,
        );
        let dims = (b, 1, sk, h, hkv, d);
        let (want, _) = cpu_attention(&q, &k, &v, None, dims, |_, j| j <= pos);
        assert_close(&format!("decode vs host, {name}"), &got, &want, 1e-5, 1e-3);
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn one_query_attention_keeps_the_window_and_the_mask() {
    let g = gpu();
    let (b, sk, h, hkv, d, pos, window) = (2, 200, 2, 1, 32, 150, 60);
    let q = rand_vec(b * h * d, 3100);
    let k = rand_vec(b * sk * hkv * d, 3101);
    let v = rand_vec(b * sk * hkv * d, 3102);
    // [batch, heads, 1, keys]: batch 1 hides every 7th key, the rest a bias.
    let mut mask = vec![0.0f32; b * h * sk];
    for (i, m) in mask.iter_mut().enumerate() {
        let (batch, j) = (i / (h * sk), i % sk);
        *m = if batch == 1 && j % 7 == 0 {
            -1e9
        } else {
            0.1 * (j % 3) as f32
        };
    }
    for causal in [true, false] {
        let opts = AttentionOptions {
            causal,
            causal_offset: pos as u32,
            window: window as u32,
            ..AttentionOptions::default()
        };
        let out = zeros(&[b, 1, h, d]);
        g.engine.flash_attention_v2_with(
            &g.ctx,
            &upload(&q, &[b, 1, h, d]),
            &upload(&k, &[b, sk, hkv, d]),
            &upload(&v, &[b, sk, hkv, d]),
            &out,
            &opts,
            Some(&upload(&mask, &[b, h, 1, sk])),
        );
        let visible = |_: usize, j: usize| (!causal || j <= pos) && pos.abs_diff(j) < window;
        let dims = (b, 1, sk, h, hkv, d);
        let (want, _) = cpu_attention(&q, &k, &v, Some(&mask), dims, visible);
        assert_close(
            &format!("decode causal={causal}"),
            &read(&out),
            &want,
            1e-5,
            1e-3,
        );
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn argmax_rows_matches_the_host_argmax_including_ties() {
    // The host rule (`ferrotensor_loader::argmax`): the largest value in
    // `f32::total_cmp` order, the last index of equal maxima.
    let g = gpu();
    let host_argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i as u32)
    };
    let mut tied = rand_vec(259, 41);
    tied[17] = 2.0;
    tied[200] = 2.0;
    let mut with_nan = rand_vec(300, 42);
    with_nan[123] = f32::NAN;
    let rows = [
        (vec![-3.0], Some(0)),
        (tied, Some(200)),
        (vec![-0.0, 0.0, -0.0, -1.0], Some(1)),
        (with_nan, Some(123)),
        (vec![0.5; 513], Some(512)),
        (rand_vec(49_152, 43), None),
    ];
    for (row, expected) in &rows {
        let out = zeros(&[1]);
        g.engine
            .argmax_rows(&g.ctx, &upload(row, &[1, row.len()]), &out);
        let got = read(&out)[0].to_bits();
        assert_eq!(got, host_argmax(&row[..]), "row of {}", row.len());
        if let Some(expected) = expected {
            assert_eq!(got, *expected, "row of {}", row.len());
        }
    }

    let (n_rows, len) = (5, 1000);
    let x = rand_vec(n_rows * len, 44);
    let out = zeros(&[n_rows]);
    g.engine
        .argmax_rows(&g.ctx, &upload(&x, &[n_rows, len]), &out);
    let got: Vec<u32> = read(&out).iter().map(|v| v.to_bits()).collect();
    let want: Vec<u32> = x.chunks(len).map(host_argmax).collect();
    assert_eq!(got, want);
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// `alpha * op(a) @ op(b) + beta * init` on the CPU in f64, with `b` stored
/// `[N, K]` when `transposed_b`.
fn cpu_epilogue(
    a: &[f32],
    b: &[f32],
    init: &[f32],
    (m, k, n): (usize, usize, usize),
    transposed_b: bool,
    ep: Epilogue,
) -> Vec<f32> {
    (0..m * n)
        .map(|i| {
            let (r, c) = (i / n, i % n);
            let dot: f64 = (0..k)
                .map(|p| {
                    let bv = if transposed_b {
                        b[c * k + p]
                    } else {
                        b[p * n + c]
                    };
                    f64::from(a[r * k + p]) * f64::from(bv)
                })
                .sum();
            let mut v = f64::from(ep.alpha) * dot;
            if ep.beta != 0.0 {
                v += f64::from(ep.beta) * f64::from(init[i]);
            }
            v as f32
        })
        .collect()
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn matmul_epilogue_matches_cpu_on_every_path() {
    let g = gpu();
    // One shape per kernel: one row (the matrix-vector kernel for a@b, the
    // thin kernel for a@bT), a thin a, a thin b (64x64 tiles with an f16 b),
    // the router's skinny kernel for a thin f32 b, 64x64 tiles, and the
    // small-M kernel with one slice of K and with three.
    let small_m = |split| MatmulPath::SmallM {
        tile: None,
        split: Some(split),
    };
    let cases = [
        ("one row", (1, 96, 80), MatmulPath::Previous),
        ("thin a", (8, 96, 200), MatmulPath::Previous),
        ("thin b", (200, 40, 8), MatmulPath::Previous),
        ("skinny", (200, 576, 8), MatmulPath::Auto),
        ("tiles", (100, 72, 130), MatmulPath::Previous),
        ("small-M, 1 slice", (64, 512, 256), small_m(1)),
        ("small-M, 3 slices", (64, 512, 256), small_m(3)),
    ];
    let epilogues = [
        Epilogue::STORE,
        Epilogue::scaled(0.375),
        Epilogue::ACCUMULATE,
        Epilogue {
            alpha: -2.0,
            beta: 0.25,
        },
    ];
    for (seed, (name, (m, k, n), path)) in cases.into_iter().enumerate() {
        let seed = 9000 + 10 * seed as u64;
        let a = rand_vec(m * k, seed);
        // Multiples of 1/64 in [-1, 1] are exact in f16, so the f16 weight
        // holds the same values as the f32 one.
        let b: Vec<f32> = rand_vec(k * n, seed + 1)
            .iter()
            .map(|v| (v * 64.0).round() / 64.0)
            .collect();
        let init = rand_vec(m * n, seed + 2);
        let a_up = upload(&a, &[m, k]);
        for transposed_b in [false, true] {
            let (b_stored, shape) = if transposed_b {
                (transpose(&b, k, n), [n, k])
            } else {
                (b.clone(), [k, n])
            };
            let weights = [
                ("f32", upload(&b_stored, &shape)),
                (
                    "f16",
                    UnifiedTensor::from_f32_as_f16(&g.ctx, &b_stored, &shape),
                ),
            ];
            for (dtype, b_up) in &weights {
                let run = |ep: Epilogue, out: &UnifiedTensor| {
                    g.engine
                        .matmul_with_path_into(&g.ctx, transposed_b, [&a_up, b_up, out], path, ep)
                        .unwrap();
                };
                let product = zeros(&[m, n]);
                run(Epilogue::STORE, &product);
                let product = read(&product);
                for ep in epilogues {
                    let label = format!("{name} {dtype} transposed_b {transposed_b} {ep:?}");
                    // An out of NaN shows that beta = 0 never reads it.
                    let start = if ep.beta == 0.0 {
                        vec![f32::NAN; m * n]
                    } else {
                        init.clone()
                    };
                    let out = upload(&start, &[m, n]);
                    run(ep, &out);
                    let got = read(&out);
                    let want = cpu_epilogue(&a, &b_stored, &init, (m, k, n), transposed_b, ep);
                    assert_close(&label, &got, &want, 1e-4, 1e-4);
                    // The product is rounded before the epilogue, so scaling
                    // or adding the stored product gives the same bits.
                    let from_product: Vec<f32> = product
                        .iter()
                        .zip(&init)
                        .map(|(p, i)| {
                            let v = ep.alpha * p;
                            if ep.beta == 0.0 {
                                v
                            } else {
                                v + ep.beta * i
                            }
                        })
                        .collect();
                    assert_eq!(bits(&got), bits(&from_product), "{label}");
                }
            }
        }
    }
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn transposed_a_matmul_accumulates_into_out() {
    // The weight-gradient product a^T @ b added into an earlier gradient,
    // as a second micro-batch adds its own: the thin kernel for a LoRA-rank
    // output and 64x64 tiles for a wide one.
    let g = gpu();
    for (seed, (m, k, n)) in [(8, 200, 96), (72, 50, 90)].into_iter().enumerate() {
        let seed = 9100 + 10 * seed as u64;
        let a = rand_vec(k * m, seed);
        let b = rand_vec(k * n, seed + 1);
        let init = rand_vec(m * n, seed + 2);
        let (a_up, b_up) = (upload(&a, &[k, m]), upload(&b, &[k, n]));
        let out = upload(&init, &[m, n]);
        g.engine
            .matmul_at_into(&g.ctx, &a_up, &b_up, &out, Epilogue::ACCUMULATE);
        let got = read(&out);
        let want = cpu_epilogue(
            &transpose(&a, k, m),
            &b,
            &init,
            (m, k, n),
            false,
            Epilogue::ACCUMULATE,
        );
        assert_close(&format!("aT@b {m}x{k}x{n}"), &got, &want, 1e-4, 1e-4);
        // The same bits as a separate product and add.
        let product = zeros(&[m, n]);
        g.engine.matmul_at(&g.ctx, &a_up, &b_up, &product);
        let sum = zeros(&[m, n]);
        g.engine
            .add(&g.ctx, &upload(&init, &[m, n]), &product, &sum);
        assert_eq!(bits(&got), bits(&read(&sum)), "aT@b {m}x{k}x{n}");
    }
}
