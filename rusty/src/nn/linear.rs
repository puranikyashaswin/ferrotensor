//! Linear Layer
//!
//! Fully connected layer: y = xW + b

use super::Module;
use crate::tensor::Tensor;
use crate::Device;

/// Linear (fully connected) layer.
///
/// Applies a linear transformation: y = xW + b
///
/// # Example
/// ```rust,no_run
/// use rusty::prelude::*;
///
/// let device = Device::new();
/// let linear = nn::Linear::new(784, 256, &device);
/// let x = Tensor::randn(&[32, 784], &device);
/// let y = linear.forward(&x); // [32, 256]
/// ```
pub struct Linear {
    /// Weight matrix [in_features, out_features]
    pub weight: Tensor,
    /// Bias vector [out_features] (optional)
    pub bias: Option<Tensor>,
}

impl Linear {
    /// Create a new Linear layer with Xavier initialization.
    pub fn new(in_features: usize, out_features: usize, device: &Device) -> Self {
        // Xavier/Glorot initialization: scale = sqrt(2 / (in + out))
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();
        let weight = Tensor::randn(&[in_features, out_features], device)
            .mul_scalar(scale)
            .requires_grad_();
        let bias = Tensor::zeros(&[out_features], device).requires_grad_();

        Self {
            weight,
            bias: Some(bias),
        }
    }

    /// Create a Linear layer without bias.
    pub fn new_no_bias(in_features: usize, out_features: usize, device: &Device) -> Self {
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();
        let weight = Tensor::randn(&[in_features, out_features], device)
            .mul_scalar(scale)
            .requires_grad_();

        Self { weight, bias: None }
    }

    /// Create a Linear layer from existing weights.
    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        // x: [..., in_features] @ weight: [in_features, out_features] = [..., out_features]
        let out = x.matmul(&self.weight);

        if let Some(ref bias) = self.bias {
            // Broadcasting add - for now assume bias matches last dim
            // In production we'd have proper broadcasting
            out.add(bias)
        } else {
            out
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

impl std::fmt::Debug for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let in_features = self.weight.shape[0];
        let out_features = self.weight.shape[1];
        f.debug_struct("Linear")
            .field("in_features", &in_features)
            .field("out_features", &out_features)
            .field("bias", &self.bias.is_some())
            .finish()
    }
}
