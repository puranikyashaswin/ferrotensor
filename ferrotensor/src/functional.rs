//! Functional Module
//!
//! Stateless functions for common operations (like PyTorch's `F` namespace).
//! Every function is built from differentiable tensor ops, so gradients flow
//! through all of them; `functional::tests` checks each against finite
//! differences.

use crate::tensor::Tensor;
use crate::Device;

// ============================================================================
// ACTIVATIONS
// ============================================================================

/// ReLU: `max(0, x)`.
pub fn relu(x: &Tensor) -> Tensor {
    x.relu()
}

/// GELU (tanh approximation).
pub fn gelu(x: &Tensor) -> Tensor {
    x.gelu()
}

/// SiLU (swish): `x * sigmoid(x)`.
pub fn silu(x: &Tensor) -> Tensor {
    x.silu()
}

/// Logistic sigmoid.
pub fn sigmoid(x: &Tensor) -> Tensor {
    x.sigmoid()
}

/// Softmax over `dim`.
pub fn softmax(x: &Tensor, dim: isize) -> Tensor {
    x.softmax(dim)
}

/// Log-softmax over `dim`.
pub fn log_softmax(x: &Tensor, dim: isize) -> Tensor {
    x.log_softmax(dim)
}

// ============================================================================
// LAYERS
// ============================================================================

/// Affine map `x @ weight + bias` with `weight` of shape `[in, out]`.
pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let out = x.matmul(weight);
    match bias {
        Some(b) => out.add(b),
        None => out,
    }
}

/// Embedding lookup: rows of `weight` (`[vocab, dim]`) for each id.
///
/// The result has shape `[ids.len(), dim]`; the gradient scatters back into
/// the looked-up rows.
pub fn embedding(weight: &Tensor, ids: &[usize]) -> Tensor {
    weight.index_select(0, ids)
}

/// Layer normalization over the last dimension, with optional affine
/// `weight` and `bias` of that dimension's size.
pub fn layer_norm(x: &Tensor, weight: Option<&Tensor>, bias: Option<&Tensor>, eps: f32) -> Tensor {
    let mean = x.mean_dim(-1, true);
    let centered = x.sub(&mean);
    let var = centered.square().mean_dim(-1, true);
    let mut out = centered.mul(&var.add_scalar(eps).rsqrt());
    if let Some(w) = weight {
        out = out.mul(w);
    }
    if let Some(b) = bias {
        out = out.add(b);
    }
    out
}

/// RMS normalization over the last dimension (Llama-style):
/// `x / sqrt(mean(x^2) + eps) * weight`.
pub fn rms_norm(x: &Tensor, weight: Option<&Tensor>, eps: f32) -> Tensor {
    let inv_rms = x.square().mean_dim(-1, true).add_scalar(eps).rsqrt();
    let out = x.mul(&inv_rms);
    match weight {
        Some(w) => out.mul(w),
        None => out,
    }
}

/// Dropout: zero each element with probability `p` and scale the rest by
/// `1 / (1 - p)`. Identity when `training` is false or `p` is 0.
///
/// # Panics
/// Panics unless `0 <= p < 1`.
pub fn dropout(x: &Tensor, p: f32, training: bool) -> Tensor {
    assert!(
        (0.0..1.0).contains(&p),
        "dropout probability must be in [0, 1), got {p}"
    );
    if !training || p == 0.0 {
        return x.clone();
    }
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let scale = 1.0 / (1.0 - p);
    let mask: Vec<f32> = (0..x.numel())
        .map(|_| if rng.gen::<f32>() < p { 0.0 } else { scale })
        .collect();
    x.mul(&Tensor::from_vec(mask, &x.shape, x.device()))
}

/// Scaled dot-product attention: `softmax(q k^T / sqrt(d) + mask) v`.
///
/// `q` is `[.., Lq, d]`, `k` is `[.., Lk, d]`, `v` is `[.., Lk, dv]` with
/// equal leading dims. With `causal`, query `i` attends only to keys
/// `j <= i + (Lk - Lq)` (so a query block at the end of a longer key
/// sequence still sees its own past).
pub fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Tensor {
    let d = *q.shape.last().expect("attention inputs need a feature dim");
    let lq = q.shape[q.ndim() - 2];
    let lk = k.shape[k.ndim() - 2];
    let mut scores = q.matmul(&k.transpose()).mul_scalar(1.0 / (d as f32).sqrt());
    if causal {
        let offset = lk.saturating_sub(lq) as isize;
        let future = Tensor::ones(&[lq, lk], q.device()).triu(offset + 1);
        scores = scores.masked_fill(&future, -1e9);
    }
    scores.softmax(-1).matmul(v)
}

// ============================================================================
// LOSSES
// ============================================================================

/// Mean squared error: `mean((pred - target)^2)`.
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    assert_eq!(pred.shape, target.shape, "shapes must match for MSE loss");
    pred.sub(target).square().mean()
}

/// Mean absolute error: `mean(|pred - target|)`.
pub fn l1_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    assert_eq!(pred.shape, target.shape, "shapes must match for L1 loss");
    pred.sub(target).abs().mean()
}

/// Smooth L1 (Huber) loss, averaged: `0.5 d^2 / beta` where `|d| < beta`,
/// else `|d| - 0.5 beta`. `beta = 0` gives the L1 loss.
pub fn smooth_l1_loss(pred: &Tensor, target: &Tensor, beta: f32) -> Tensor {
    assert_eq!(
        pred.shape, target.shape,
        "shapes must match for Smooth L1 loss"
    );
    assert!(beta >= 0.0, "beta must be non-negative");
    let abs = pred.sub(target).abs();
    if beta == 0.0 {
        return abs.mean();
    }
    // With m = min(|d|, beta): 0.5 m^2 / beta + (|d| - m) equals the
    // piecewise definition on both sides of beta.
    let m = abs.clamp(0.0, beta);
    m.square().mul_scalar(0.5 / beta).add(&abs.sub(&m)).mean()
}

/// Cross-entropy between `logits` (`[.., C]`) and `targets`, averaged.
///
/// `targets` is either:
/// - class indices with the shape of `logits` minus its last dim (stored as
///   `f32`); negative indices (e.g. `-100`) are ignored, as for padding or
///   prompt tokens, and the mean is over the remaining positions;
/// - or a distribution (one-hot or soft labels) with the shape of `logits`.
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Tensor {
    let log_probs = logits.log_softmax(-1);
    if targets.shape == logits.shape {
        let rows = (logits.numel() / logits.shape[logits.ndim() - 1]).max(1);
        return log_probs.mul(targets).sum().div_scalar(-(rows as f32));
    }
    assert_eq!(
        targets.shape,
        logits.shape[..logits.ndim() - 1],
        "class-index targets must have the logits shape without the class dim"
    );
    let indices: Vec<i64> = targets.to_vec().iter().map(|&t| t as i64).collect();
    nll_loss_indices(&log_probs, &indices)
}

/// Negative log-likelihood of class `targets` under `log_probs` (`[.., C]`),
/// averaged over non-ignored (non-negative) targets.
pub fn nll_loss(log_probs: &Tensor, targets: &[i64]) -> Tensor {
    nll_loss_indices(log_probs, targets)
}

fn nll_loss_indices(log_probs: &Tensor, targets: &[i64]) -> Tensor {
    let classes = log_probs.shape[log_probs.ndim() - 1];
    let rows = log_probs.numel() / classes.max(1);
    assert_eq!(
        targets.len(),
        rows,
        "one target per row of log-probabilities"
    );
    let kept = targets.iter().filter(|&&t| t >= 0).count();
    let mut weights = vec![0.0f32; rows * classes];
    if kept > 0 {
        for (row, &t) in targets.iter().enumerate() {
            if t >= 0 {
                let t = t as usize;
                assert!(
                    t < classes,
                    "target class {t} out of range for {classes} classes"
                );
                weights[row * classes + t] = -1.0 / kept as f32;
            }
        }
    }
    let weights = Tensor::from_vec(weights, &log_probs.shape, log_probs.device());
    log_probs.mul(&weights).sum()
}

/// Binary cross-entropy on probabilities, averaged:
/// `-mean(t log p + (1 - t) log(1 - p))`. `pred` is clamped to
/// `[1e-7, 1 - 1e-7]`; prefer [`binary_cross_entropy_with_logits`].
pub fn binary_cross_entropy(pred: &Tensor, target: &Tensor) -> Tensor {
    assert_eq!(pred.shape, target.shape, "shapes must match for BCE loss");
    let p = pred.clamp(1e-7, 1.0 - 1e-7);
    let pos = target.mul(&p.log());
    let neg = target.rsub_scalar(1.0).mul(&p.rsub_scalar(1.0).log());
    pos.add(&neg).mean().neg()
}

/// Binary cross-entropy on logits (numerically stable), averaged:
/// `mean(max(x, 0) - x t + log(1 + exp(-|x|)))`.
pub fn binary_cross_entropy_with_logits(logits: &Tensor, target: &Tensor) -> Tensor {
    assert_eq!(
        logits.shape, target.shape,
        "shapes must match for BCE with logits"
    );
    let zero = Tensor::zeros(&[], logits.device());
    let softplus_neg_abs = logits.abs().neg().exp().add_scalar(1.0).log();
    logits
        .maximum(&zero)
        .sub(&logits.mul(target))
        .add(&softplus_neg_abs)
        .mean()
}

// ============================================================================
// METRICS AND HELPERS
// ============================================================================

/// Cosine similarity along `dim`: `sum(a b) / (|a| |b| + eps)`.
pub fn cosine_similarity(a: &Tensor, b: &Tensor, dim: isize) -> Tensor {
    let dot = a.mul(b).sum_dim(dim, false);
    let norm_a = a.square().sum_dim(dim, false).sqrt();
    let norm_b = b.square().sum_dim(dim, false).sqrt();
    dot.div(&norm_a.mul(&norm_b).add_scalar(1e-8))
}

/// One-hot rows for `indices` (`[indices.len(), num_classes]`); indices out
/// of range give an all-zero row.
pub fn one_hot(indices: &[usize], num_classes: usize, device: &Device) -> Tensor {
    let mut data = vec![0.0f32; indices.len() * num_classes];
    for (i, &idx) in indices.iter().enumerate() {
        if idx < num_classes {
            data[i * num_classes + idx] = 1.0;
        }
    }
    Tensor::from_vec(data, &[indices.len(), num_classes], device)
}

/// Fraction of rows of `logits` (`[N, C]`) whose argmax equals the target
/// class.
pub fn accuracy(logits: &Tensor, targets: &[usize]) -> f32 {
    let pred = logits.argmax_dim(-1).to_vec();
    assert_eq!(pred.len(), targets.len(), "one target per row");
    if targets.is_empty() {
        return 0.0;
    }
    let correct = pred
        .iter()
        .zip(targets)
        .filter(|(p, &t)| **p as usize == t)
        .count();
    correct as f32 / targets.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::tests::{check_gpu_parity, check_gradients, Case, Init};

    const U: Init = Init::Uniform(-2.0, 2.0);
    const PROB: Init = Init::Uniform(0.05, 0.95);

    fn case(name: &'static str, inputs: &[(&[usize], Init)], f: fn(&[Tensor]) -> Tensor) -> Case {
        Case {
            name,
            inputs: inputs.iter().map(|(s, i)| (s.to_vec(), *i)).collect(),
            f,
        }
    }

    fn cases() -> Vec<Case> {
        vec![
            case("linear", &[(&[2, 3], U), (&[3, 4], U), (&[4], U)], |x| {
                linear(&x[0], &x[1], Some(&x[2]))
            }),
            case("embedding", &[(&[5, 3], U)], |x| {
                embedding(&x[0], &[1, 4, 1])
            }),
            case("layer_norm", &[(&[3, 5], U), (&[5], U), (&[5], U)], |x| {
                layer_norm(&x[0], Some(&x[1]), Some(&x[2]), 1e-5)
            }),
            case("rms_norm", &[(&[2, 3, 4], U), (&[4], U)], |x| {
                rms_norm(&x[0], Some(&x[1]), 1e-6)
            }),
            case(
                "attention",
                &[(&[2, 3, 4], U), (&[2, 3, 4], U), (&[2, 3, 2], U)],
                |x| scaled_dot_product_attention(&x[0], &x[1], &x[2], false),
            ),
            case(
                "attention_causal",
                &[(&[2, 4], U), (&[3, 4], U), (&[3, 2], U)],
                |x| scaled_dot_product_attention(&x[0], &x[1], &x[2], true),
            ),
            case("mse", &[(&[2, 3], U), (&[2, 3], U)], |x| {
                mse_loss(&x[0], &x[1])
            }),
            case("l1", &[(&[6], Init::AwayFromZero(-2.0, 2.0))], |x| {
                l1_loss(&x[0], &x[0].zeros_like())
            }),
            case("smooth_l1", &[(&[8], U)], |x| {
                smooth_l1_loss(&x[0], &x[0].zeros_like().add_scalar(0.1), 0.7)
            }),
            case("cross_entropy_indices", &[(&[3, 4], U)], |x| {
                let t = Tensor::from_data(&[2.0, -100.0, 0.0], &[3], x[0].device());
                cross_entropy(&x[0], &t)
            }),
            case("cross_entropy_soft", &[(&[2, 3], U)], |x| {
                let t = Tensor::from_data(&[0.2, 0.3, 0.5, 1.0, 0.0, 0.0], &[2, 3], x[0].device());
                cross_entropy(&x[0], &t)
            }),
            case("bce", &[(&[5], PROB), (&[5], PROB)], |x| {
                binary_cross_entropy(&x[0], &x[1])
            }),
            case("bce_logits", &[(&[5], U), (&[5], PROB)], |x| {
                binary_cross_entropy_with_logits(&x[0], &x[1])
            }),
            case("cosine", &[(&[2, 4], U), (&[2, 4], U)], |x| {
                cosine_similarity(&x[0], &x[1], -1)
            }),
            case("dropout_eval", &[(&[4], U)], |x| {
                dropout(&x[0], 0.5, false).square()
            }),
        ]
    }

    #[test]
    fn gradients_match_finite_differences() {
        for case in cases() {
            check_gradients(&case);
        }
    }

    #[test]
    fn gpu_matches_cpu() {
        check_gpu_parity(&cases());
    }

    fn cpu(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_data(data, shape, &Device::cpu())
    }

    #[test]
    fn cross_entropy_matches_hand_computation() {
        let logits = cpu(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]);
        let targets = cpu(&[2.0, 0.0], &[2]);
        let lse0 = (1f32.exp() + 2f32.exp() + 3f32.exp()).ln();
        let expected = ((lse0 - 3.0) + 3f32.ln()) / 2.0;
        assert!((cross_entropy(&logits, &targets).item() - expected).abs() < 1e-5);
        // Same loss from one-hot targets.
        let one_hot = one_hot(&[2, 0], 3, &Device::cpu());
        assert!((cross_entropy(&logits, &one_hot).item() - expected).abs() < 1e-5);
    }

    #[test]
    fn ignored_targets_do_not_count() {
        let logits = cpu(&[1.0, 2.0, 3.0, 9.0, 0.0, 0.0], &[2, 3]);
        let only_first = cross_entropy(&logits.narrow(0, 0, 1), &cpu(&[2.0], &[1])).item();
        let masked = cross_entropy(&logits, &cpu(&[2.0, -100.0], &[2])).item();
        assert!((only_first - masked).abs() < 1e-6);
    }

    #[test]
    fn smooth_l1_is_piecewise() {
        let pred = cpu(&[0.1, 2.0], &[2]);
        let loss = smooth_l1_loss(&pred, &cpu(&[0.0, 0.0], &[2]), 1.0).item();
        let expected = (0.5 * 0.01 + (2.0 - 0.5)) / 2.0;
        assert!((loss - expected).abs() < 1e-6);
    }

    #[test]
    fn bce_matches_bce_with_logits() {
        let logits = cpu(&[-1.0, 0.5, 2.0], &[3]);
        let target = cpu(&[0.0, 1.0, 1.0], &[3]);
        let a = binary_cross_entropy(&logits.sigmoid(), &target).item();
        let b = binary_cross_entropy_with_logits(&logits, &target).item();
        assert!((a - b).abs() < 1e-5);
    }

    #[test]
    fn layer_norm_normalizes_rows() {
        let x = cpu(&[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], &[2, 4]);
        let y = layer_norm(&x, None, None, 1e-5);
        for row in y.to_vec().chunks(4) {
            let mean: f32 = row.iter().sum::<f32>() / 4.0;
            let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-5 && (var - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn dropout_keeps_expectation_and_respects_eval() {
        let x = Tensor::ones(&[10_000], &Device::cpu());
        let y = dropout(&x, 0.3, true).to_vec();
        let zeros = y.iter().filter(|&&v| v == 0.0).count() as f32 / 10_000.0;
        assert!((zeros - 0.3).abs() < 0.03);
        let mean: f32 = y.iter().sum::<f32>() / 10_000.0;
        assert!((mean - 1.0).abs() < 0.05);
        assert_eq!(dropout(&x, 0.3, false).to_vec(), x.to_vec());
    }

    #[test]
    fn causal_attention_ignores_future_keys() {
        let d = Device::cpu();
        let q = Tensor::ones(&[3, 2], &d);
        let k = Tensor::ones(&[3, 2], &d);
        let v = cpu(&[1.0, 0.0, 0.0, 1.0, 5.0, 5.0], &[3, 2]);
        let out = scaled_dot_product_attention(&q, &k, &v, true).to_vec();
        // Row 0 sees only key 0; row 1 averages keys 0 and 1.
        assert!((out[0] - 1.0).abs() < 1e-5 && out[1].abs() < 1e-5);
        assert!((out[2] - 0.5).abs() < 1e-5 && (out[3] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn accuracy_counts_argmax_hits() {
        let logits = cpu(&[0.1, 0.9, 0.8, 0.2], &[2, 2]);
        assert_eq!(accuracy(&logits, &[1, 1]), 0.5);
    }
}
