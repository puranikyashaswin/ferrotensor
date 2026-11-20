//! CPU references for attention: the forward pass, its log-sum-exp and its
//! gradients, in f32 with f64 sums for the log-sum-exp.

// Each test file uses some of these.
#![allow(dead_code)]

/// Reference attention for [batch, seq, heads, dim] tensors with GQA, an
/// optional additive mask of shape [batch, heads, seq_q, seq_k] and a
/// visibility rule `visible(i, j)` for query i and key j.
#[allow(clippy::too_many_arguments)]
pub fn cpu_attention(
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
pub fn cpu_attention_backward(
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

pub fn causal_rule(causal: bool) -> impl Fn(usize, usize) -> bool {
    move |i, j| !causal || j <= i
}

/// Log-sum-exp of the scaled (and masked) scores of every (batch, head,
/// query) row, in the order of the attention stats.
pub fn cpu_attention_lse(
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
