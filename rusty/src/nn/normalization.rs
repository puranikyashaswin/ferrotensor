//! Normalization Layers
//!
//! Layer normalization and RMS normalization, both differentiable in their
//! input and parameters.

use super::Module;
use crate::functional;
use crate::tensor::Tensor;
use crate::Device;

/// Layer Normalization over the trailing `normalized_shape` dimensions,
/// with learnable scale (gamma) and shift (beta).
pub struct LayerNorm {
    /// Gamma (scale) parameter, flattened to `[prod(normalized_shape)]`
    pub gamma: Tensor,
    /// Beta (shift) parameter, flattened to `[prod(normalized_shape)]`
    pub beta: Tensor,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Normalized shape (the trailing dims of the input)
    pub normalized_shape: Vec<usize>,
}

impl LayerNorm {
    /// Create a LayerNorm layer (gamma = 1, beta = 0, eps = 1e-5).
    pub fn new(normalized_shape: &[usize], device: &Device) -> Self {
        let size: usize = normalized_shape.iter().product();
        Self {
            gamma: Tensor::ones(&[size], device).requires_grad_(),
            beta: Tensor::zeros(&[size], device).requires_grad_(),
            eps: 1e-5,
            normalized_shape: normalized_shape.to_vec(),
        }
    }

    /// Create LayerNorm with custom epsilon.
    pub fn with_eps(normalized_shape: &[usize], eps: f32, device: &Device) -> Self {
        let mut ln = Self::new(normalized_shape, device);
        ln.eps = eps;
        ln
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        let k = self.normalized_shape.len();
        assert!(
            x.ndim() >= k && x.shape[x.ndim() - k..] == self.normalized_shape[..],
            "LayerNorm over {:?} got input of shape {:?}",
            self.normalized_shape,
            x.shape
        );
        let size = self.gamma.numel();
        let flat = [&x.shape[..x.ndim() - k], &[size]].concat();
        functional::layer_norm(
            &x.reshape(&flat),
            Some(&self.gamma),
            Some(&self.beta),
            self.eps,
        )
        .reshape(&x.shape)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![
            ("weight".to_string(), self.gamma.clone()),
            ("bias".to_string(), self.beta.clone()),
        ]
    }
}

impl std::fmt::Debug for LayerNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerNorm")
            .field("normalized_shape", &self.normalized_shape)
            .field("eps", &self.eps)
            .finish()
    }
}

/// RMS Normalization over the last dimension (used in Llama, etc.).
///
/// Normalizes by root mean square without centering, then scales.
pub struct RMSNorm {
    /// Weight (scale) parameter
    pub weight: Tensor,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Hidden dimension
    pub hidden_size: usize,
}

impl RMSNorm {
    /// Create an RMSNorm layer (weight = 1, eps = 1e-5).
    pub fn new(hidden_size: usize, device: &Device) -> Self {
        Self {
            weight: Tensor::ones(&[hidden_size], device).requires_grad_(),
            eps: 1e-5,
            hidden_size,
        }
    }

    /// Create RMSNorm with custom epsilon.
    pub fn with_eps(hidden_size: usize, eps: f32, device: &Device) -> Self {
        let mut rms = Self::new(hidden_size, device);
        rms.eps = eps;
        rms
    }

    /// Create from existing weights.
    pub fn from_weights(weight: Tensor, eps: f32) -> Self {
        let hidden_size = weight.numel();
        Self {
            weight,
            eps,
            hidden_size,
        }
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(
            x.shape.last(),
            Some(&self.hidden_size),
            "RMSNorm of size {} got input of shape {:?}",
            self.hidden_size,
            x.shape
        );
        functional::rms_norm(x, Some(&self.weight), self.eps)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".to_string(), self.weight.clone())]
    }
}

impl std::fmt::Debug for RMSNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RMSNorm")
            .field("hidden_size", &self.hidden_size)
            .field("eps", &self.eps)
            .finish()
    }
}
