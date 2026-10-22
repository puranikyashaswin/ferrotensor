//! The Rust layers behind `ferrotensor.nn`.
//!
//! Every layer is a `_Layer` wrapping a boxed [`Module`]; the Python classes
//! in `ferrotensor/nn/__init__.py` give them their constructors and names.

use ferrotensor::nn::{Dropout, Embedding, LayerNorm, Linear, Module, MultiHeadAttention, RMSNorm};
use pyo3::prelude::*;

use crate::device::resolve_device;
use crate::error::{guard, index_error, value_error};
use crate::tensor::{parse_data, PyTensor};

/// What `forward` checks before calling into Rust.
enum Expect {
    /// Last dimension of this size (Linear also accepts 1-D input).
    Features(usize),
    /// Integer ids below this vocabulary size.
    Ids(usize),
    /// Trailing dimensions equal to this shape.
    Trailing(Vec<usize>),
    /// `[seq, dim]` or `[batch, seq, dim]`.
    Sequence(usize),
    Any,
}

/// A ferrotensor layer.
#[pyclass(name = "_Layer", module = "ferrotensor.nn")]
pub(crate) struct PyLayer {
    inner: Box<dyn Module>,
    expect: Expect,
    linear: bool,
}

impl PyLayer {
    fn new(inner: impl Module + 'static, expect: Expect) -> Self {
        Self {
            inner: Box::new(inner),
            expect,
            linear: false,
        }
    }
}

fn positive(name: &str, value: usize) -> PyResult<usize> {
    if value == 0 {
        Err(value_error(format!("{name} must be positive")))
    } else {
        Ok(value)
    }
}

#[pymethods]
impl PyLayer {
    #[staticmethod]
    #[pyo3(signature = (in_features, out_features, bias = true, device = None))]
    fn linear(
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        positive("in_features", in_features)?;
        positive("out_features", out_features)?;
        let device = resolve_device(device)?;
        let layer = guard(|| {
            if bias {
                Linear::new(in_features, out_features, &device)
            } else {
                Linear::new_no_bias(in_features, out_features, &device)
            }
        })?;
        let mut l = Self::new(layer, Expect::Features(in_features));
        l.linear = true;
        Ok(l)
    }

    #[staticmethod]
    #[pyo3(signature = (num_embeddings, embedding_dim, device = None))]
    fn embedding(
        num_embeddings: usize,
        embedding_dim: usize,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        positive("num_embeddings", num_embeddings)?;
        positive("embedding_dim", embedding_dim)?;
        let device = resolve_device(device)?;
        let layer = guard(|| Embedding::new(num_embeddings, embedding_dim, &device))?;
        Ok(Self::new(layer, Expect::Ids(num_embeddings)))
    }

    #[staticmethod]
    #[pyo3(signature = (normalized_shape, eps = 1e-5, device = None))]
    fn layer_norm(
        normalized_shape: Vec<usize>,
        eps: f32,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if normalized_shape.is_empty() || normalized_shape.contains(&0) {
            return Err(value_error(
                "normalized_shape must be non-empty and positive",
            ));
        }
        let device = resolve_device(device)?;
        let layer = guard(|| LayerNorm::with_eps(&normalized_shape, eps, &device))?;
        Ok(Self::new(layer, Expect::Trailing(normalized_shape)))
    }

    #[staticmethod]
    #[pyo3(signature = (dim, eps = 1e-5, device = None))]
    fn rms_norm(dim: usize, eps: f32, device: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        positive("dim", dim)?;
        let device = resolve_device(device)?;
        let layer = guard(|| RMSNorm::with_eps(dim, eps, &device))?;
        Ok(Self::new(layer, Expect::Features(dim)))
    }

    #[staticmethod]
    #[pyo3(signature = (p = 0.5))]
    fn dropout(p: f32) -> PyResult<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(value_error(format!(
                "dropout probability must be in [0, 1), got {p}"
            )));
        }
        Ok(Self::new(Dropout::new(p), Expect::Any))
    }

    #[staticmethod]
    #[pyo3(signature = (embed_dim, num_heads, causal = false, device = None))]
    fn attention(
        embed_dim: usize,
        num_heads: usize,
        causal: bool,
        device: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        positive("num_heads", num_heads)?;
        if embed_dim == 0 || !embed_dim.is_multiple_of(num_heads) {
            return Err(value_error(format!(
                "embed_dim ({embed_dim}) must be a positive multiple of num_heads ({num_heads})"
            )));
        }
        let device = resolve_device(device)?;
        let layer =
            guard(|| MultiHeadAttention::new(embed_dim, num_heads, &device).causal(causal))?;
        Ok(Self::new(layer, Expect::Sequence(embed_dim)))
    }

    /// Run the layer; ids for an embedding may be a tensor or a list of ints.
    fn forward(&self, x: &Bound<'_, PyAny>) -> PyResult<PyTensor> {
        if let Expect::Ids(vocab) = self.expect {
            return self.forward_ids(x, vocab);
        }
        let x = x.cast::<PyTensor>()?.borrow().inner.clone();
        let shape = &x.shape;
        let ok = match &self.expect {
            Expect::Features(n) => shape.last() == Some(n),
            Expect::Trailing(s) => {
                shape.len() >= s.len() && shape[shape.len() - s.len()..] == s[..]
            }
            Expect::Sequence(n) => matches!(shape.len(), 2 | 3) && shape.last() == Some(n),
            Expect::Ids(_) | Expect::Any => true,
        };
        if !ok {
            let want = match &self.expect {
                Expect::Features(n) => format!("last dimension {n}"),
                Expect::Trailing(s) => format!("trailing dimensions {s:?}"),
                Expect::Sequence(n) => format!("shape [seq, {n}] or [batch, seq, {n}]"),
                Expect::Ids(_) | Expect::Any => String::new(),
            };
            return Err(value_error(format!(
                "expected input with {want}, got shape {shape:?}"
            )));
        }
        guard(|| {
            if self.linear && x.ndim() == 1 {
                self.inner.forward(&x.unsqueeze(0)).squeeze_dim(0)
            } else {
                self.inner.forward(&x)
            }
        })
        .map(Into::into)
    }

    /// `(name, parameter)` pairs; the tensors share storage with the layer.
    fn named_parameters(&self) -> Vec<(String, PyTensor)> {
        self.inner
            .named_parameters()
            .into_iter()
            .map(|(n, p)| (n, p.into()))
            .collect()
    }

    fn set_training(&mut self, training: bool) {
        self.inner.set_training(training);
    }
}

impl PyLayer {
    fn forward_ids(&self, x: &Bound<'_, PyAny>, vocab: usize) -> PyResult<PyTensor> {
        let (values, shape) = parse_data(x)?;
        for &v in &values {
            if v.fract() != 0.0 || v < 0.0 || v as usize >= vocab {
                return Err(index_error(format!(
                    "embedding id {v} is not an integer in [0, {vocab})"
                )));
            }
        }
        let device = self
            .inner
            .parameters()
            .first()
            .map(|p| p.device().clone())
            .unwrap_or_else(ferrotensor::Device::cpu);
        let ids = guard(|| ferrotensor::Tensor::from_vec(values, &shape, &device))?;
        guard(|| self.inner.forward(&ids)).map(Into::into)
    }
}
