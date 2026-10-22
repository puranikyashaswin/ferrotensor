//! Losses and functional layers behind `ferrotensor.nn.functional`.

use ferrotensor::functional as F;
use ferrotensor::Tensor;
use pyo3::prelude::*;

use crate::device::resolve_device;
use crate::error::{guard, index_error, value_error};
use crate::tensor::{check_same_device, parse_data, PyTensor};

fn same_shape(op: &str, a: &Tensor, b: &Tensor) -> PyResult<()> {
    check_same_device(a, b)?;
    if a.shape == b.shape {
        Ok(())
    } else {
        Err(value_error(format!(
            "{op}: input shape {:?} does not match target shape {:?}",
            a.shape, b.shape
        )))
    }
}

/// `target` as a tensor on `like`'s device (accepts lists of numbers).
fn target_tensor(target: &Bound<'_, PyAny>, like: &Tensor) -> PyResult<Tensor> {
    if let Ok(t) = target.cast::<PyTensor>() {
        let t = t.borrow().inner.clone();
        check_same_device(like, &t)?;
        return Ok(t);
    }
    let (values, shape) = parse_data(target)?;
    guard(|| Tensor::from_vec(values, &shape, like.device()))
}

/// Check class-index targets for `[.., classes]` scores; negatives are ignored.
fn class_indices(target: &Tensor, classes: usize) -> PyResult<Vec<i64>> {
    guard(|| target.to_vec())?
        .into_iter()
        .map(|v| {
            if v.fract() != 0.0 || v >= classes as f32 {
                Err(index_error(format!(
                    "target {v} is not a class index in [0, {classes}) (negative = ignored)"
                )))
            } else {
                Ok(v as i64)
            }
        })
        .collect()
}

fn scores_and_classes(op: &str, input: &Tensor) -> PyResult<usize> {
    input
        .shape
        .last()
        .copied()
        .filter(|&c| c > 0)
        .ok_or_else(|| value_error(format!("{op}: input needs a non-empty class dimension")))
}

#[pyfunction]
pub(crate) fn mse_loss(
    input: PyRef<'_, PyTensor>,
    target: &Bound<'_, PyAny>,
) -> PyResult<PyTensor> {
    let target = target_tensor(target, &input.inner)?;
    same_shape("mse_loss", &input.inner, &target)?;
    guard(|| F::mse_loss(&input.inner, &target)).map(Into::into)
}

#[pyfunction]
pub(crate) fn l1_loss(input: PyRef<'_, PyTensor>, target: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
    let target = target_tensor(target, &input.inner)?;
    same_shape("l1_loss", &input.inner, &target)?;
    guard(|| F::l1_loss(&input.inner, &target)).map(Into::into)
}

#[pyfunction]
#[pyo3(signature = (input, target, beta = 1.0))]
pub(crate) fn smooth_l1_loss(
    input: PyRef<'_, PyTensor>,
    target: &Bound<'_, PyAny>,
    beta: f32,
) -> PyResult<PyTensor> {
    if beta <= 0.0 {
        return Err(value_error("smooth_l1_loss: beta must be positive"));
    }
    let target = target_tensor(target, &input.inner)?;
    same_shape("smooth_l1_loss", &input.inner, &target)?;
    guard(|| F::smooth_l1_loss(&input.inner, &target, beta)).map(Into::into)
}

/// Cross-entropy of logits `[.., C]` against class indices `[..]` (negative
/// ids are ignored) or class probabilities `[.., C]`, averaged.
#[pyfunction]
pub(crate) fn cross_entropy(
    input: PyRef<'_, PyTensor>,
    target: &Bound<'_, PyAny>,
) -> PyResult<PyTensor> {
    let logits = &input.inner;
    let classes = scores_and_classes("cross_entropy", logits)?;
    let target = target_tensor(target, logits)?;
    if target.shape != logits.shape {
        if target.shape[..] != logits.shape[..logits.ndim() - 1] {
            return Err(value_error(format!(
                "cross_entropy: target shape {:?} must be {:?} (class indices) or {:?} (probabilities)",
                target.shape,
                &logits.shape[..logits.ndim() - 1],
                logits.shape
            )));
        }
        class_indices(&target, classes)?;
    }
    guard(|| F::cross_entropy(logits, &target)).map(Into::into)
}

/// Negative log-likelihood of class indices under log-probabilities `[.., C]`.
#[pyfunction]
pub(crate) fn nll_loss(
    input: PyRef<'_, PyTensor>,
    target: &Bound<'_, PyAny>,
) -> PyResult<PyTensor> {
    let log_probs = &input.inner;
    let classes = scores_and_classes("nll_loss", log_probs)?;
    let target = target_tensor(target, log_probs)?;
    if target.shape[..] != log_probs.shape[..log_probs.ndim() - 1] {
        return Err(value_error(format!(
            "nll_loss: target shape {:?} must be {:?}",
            target.shape,
            &log_probs.shape[..log_probs.ndim() - 1]
        )));
    }
    let indices = class_indices(&target, classes)?;
    guard(|| F::nll_loss(log_probs, &indices)).map(Into::into)
}

#[pyfunction]
pub(crate) fn binary_cross_entropy(
    input: PyRef<'_, PyTensor>,
    target: &Bound<'_, PyAny>,
) -> PyResult<PyTensor> {
    let target = target_tensor(target, &input.inner)?;
    same_shape("binary_cross_entropy", &input.inner, &target)?;
    guard(|| F::binary_cross_entropy(&input.inner, &target)).map(Into::into)
}

#[pyfunction]
pub(crate) fn binary_cross_entropy_with_logits(
    input: PyRef<'_, PyTensor>,
    target: &Bound<'_, PyAny>,
) -> PyResult<PyTensor> {
    let target = target_tensor(target, &input.inner)?;
    same_shape("binary_cross_entropy_with_logits", &input.inner, &target)?;
    guard(|| F::binary_cross_entropy_with_logits(&input.inner, &target)).map(Into::into)
}

/// Inverted dropout: zero entries with probability `p`, scale the rest.
#[pyfunction]
#[pyo3(signature = (input, p = 0.5, training = true))]
pub(crate) fn dropout(input: PyRef<'_, PyTensor>, p: f32, training: bool) -> PyResult<PyTensor> {
    if !(0.0..1.0).contains(&p) {
        return Err(value_error(format!(
            "dropout probability must be in [0, 1), got {p}"
        )));
    }
    guard(|| F::dropout(&input.inner, p, training)).map(Into::into)
}

fn check_last_dim(op: &str, x: &Tensor, w: Option<&PyRef<'_, PyTensor>>) -> PyResult<()> {
    let Some(w) = w else { return Ok(()) };
    check_same_device(x, &w.inner)?;
    if x.shape.last() != Some(&w.inner.numel()) || w.inner.ndim() != 1 {
        return Err(value_error(format!(
            "{op}: weight shape {:?} does not match the last dimension of {:?}",
            w.inner.shape, x.shape
        )));
    }
    Ok(())
}

/// Layer normalization over the last dimension.
#[pyfunction]
#[pyo3(signature = (input, weight = None, bias = None, eps = 1e-5))]
pub(crate) fn layer_norm(
    input: PyRef<'_, PyTensor>,
    weight: Option<PyRef<'_, PyTensor>>,
    bias: Option<PyRef<'_, PyTensor>>,
    eps: f32,
) -> PyResult<PyTensor> {
    check_last_dim("layer_norm", &input.inner, weight.as_ref())?;
    check_last_dim("layer_norm", &input.inner, bias.as_ref())?;
    guard(|| {
        F::layer_norm(
            &input.inner,
            weight.as_ref().map(|w| &w.inner),
            bias.as_ref().map(|b| &b.inner),
            eps,
        )
    })
    .map(Into::into)
}

/// RMS normalization over the last dimension.
#[pyfunction]
#[pyo3(signature = (input, weight = None, eps = 1e-5))]
pub(crate) fn rms_norm(
    input: PyRef<'_, PyTensor>,
    weight: Option<PyRef<'_, PyTensor>>,
    eps: f32,
) -> PyResult<PyTensor> {
    check_last_dim("rms_norm", &input.inner, weight.as_ref())?;
    guard(|| F::rms_norm(&input.inner, weight.as_ref().map(|w| &w.inner), eps)).map(Into::into)
}

/// `input @ weight + bias` with `weight` of shape `[in, out]`.
#[pyfunction]
#[pyo3(signature = (input, weight, bias = None))]
pub(crate) fn linear(
    input: PyRef<'_, PyTensor>,
    weight: PyRef<'_, PyTensor>,
    bias: Option<PyRef<'_, PyTensor>>,
) -> PyResult<PyTensor> {
    let (x, w) = (&input.inner, &weight.inner);
    check_same_device(x, w)?;
    if x.ndim() < 2 || w.ndim() != 2 || x.shape.last() != w.shape.first() {
        return Err(value_error(format!(
            "linear: input {:?} needs at least 2 dims ending in weight's first dim, weight {:?}",
            x.shape, w.shape
        )));
    }
    if let Some(b) = &bias {
        check_same_device(x, &b.inner)?;
        if b.inner.shape[..] != w.shape[1..] {
            return Err(value_error(format!(
                "linear: bias shape {:?} must be [{}]",
                b.inner.shape, w.shape[1]
            )));
        }
    }
    guard(|| F::linear(x, w, bias.as_ref().map(|b| &b.inner))).map(Into::into)
}

/// `softmax(q k^T / sqrt(d)) v` for q `[.., Lq, d]`, k `[.., Lk, d]`, v `[.., Lk, dv]`.
#[pyfunction]
#[pyo3(signature = (query, key, value, is_causal = false))]
pub(crate) fn scaled_dot_product_attention(
    query: PyRef<'_, PyTensor>,
    key: PyRef<'_, PyTensor>,
    value: PyRef<'_, PyTensor>,
    is_causal: bool,
) -> PyResult<PyTensor> {
    let (q, k, v) = (&query.inner, &key.inner, &value.inner);
    check_same_device(q, k)?;
    check_same_device(q, v)?;
    let n = q.ndim();
    let ok = n >= 2
        && k.ndim() == n
        && v.ndim() == n
        && q.shape[..n - 2] == k.shape[..n - 2]
        && k.shape[..n - 1] == v.shape[..n - 1]
        && q.shape[n - 1] == k.shape[n - 1];
    if !ok {
        return Err(value_error(format!(
            "scaled_dot_product_attention expects q [.., Lq, d], k [.., Lk, d] and \
             v [.., Lk, dv] with equal leading dims, got {:?}, {:?}, {:?}",
            q.shape, k.shape, v.shape
        )));
    }
    guard(|| F::scaled_dot_product_attention(q, k, v, is_causal)).map(Into::into)
}

/// One-hot rows `[len(indices), num_classes]`.
#[pyfunction]
#[pyo3(signature = (indices, num_classes, *, device = None))]
pub(crate) fn one_hot(
    indices: &Bound<'_, PyAny>,
    num_classes: usize,
    device: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyTensor> {
    let ids = crate::tensor::parse_indices(indices, num_classes)?;
    let device = resolve_device(device)?;
    guard(|| F::one_hot(&ids, num_classes, &device)).map(Into::into)
}

/// Fraction of rows of `logits` whose argmax equals the target class; rows
/// with a negative (ignored) target do not count.
#[pyfunction]
pub(crate) fn accuracy(logits: PyRef<'_, PyTensor>, target: &Bound<'_, PyAny>) -> PyResult<f32> {
    let l = &logits.inner;
    let classes = scores_and_classes("accuracy", l)?;
    let ids = class_indices(&target_tensor(target, l)?, classes)?;
    if l.ndim() != 2 || ids.len() != l.shape[0] {
        return Err(value_error(format!(
            "accuracy: logits must be [N, C] with N targets, got {:?} and {} targets",
            l.shape,
            ids.len()
        )));
    }
    guard(|| F::accuracy(l, &ids))
}

/// Cosine similarity along `dim`.
#[pyfunction]
#[pyo3(signature = (x1, x2, dim = -1))]
pub(crate) fn cosine_similarity(
    x1: PyRef<'_, PyTensor>,
    x2: PyRef<'_, PyTensor>,
    dim: isize,
) -> PyResult<PyTensor> {
    same_shape("cosine_similarity", &x1.inner, &x2.inner)?;
    crate::tensor::normalize_dim(dim, x1.inner.ndim())?;
    guard(|| F::cosine_similarity(&x1.inner, &x2.inner, dim)).map(Into::into)
}
