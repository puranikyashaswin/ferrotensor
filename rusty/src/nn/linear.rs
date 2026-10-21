//! Linear Layer
//!
//! Fully connected layer: y = xW + b

use super::Module;
use crate::tensor::Tensor;
use crate::Device;

/// Linear (fully connected) layer.
///
/// Applies `y = x W + b` to the last dimension of `x` (`[.., in]` to
/// `[.., out]`). `W` is stored as `[in, out]`.
///
/// # Example
/// ```rust
/// use rusty::prelude::*;
///
/// let device = Device::cpu();
/// let linear = nn::Linear::new(784, 256, &device);
/// let x = Tensor::randn(&[32, 784], &device);
/// let y = linear.forward(&x);
/// assert_eq!(y.shape, vec![32, 256]);
/// ```
pub struct Linear {
    /// Weight matrix [in_features, out_features]
    pub weight: Tensor,
    /// Bias vector [out_features] (optional)
    pub bias: Option<Tensor>,
}

impl Linear {
    /// Create a Linear layer with Xavier-normal weights and zero bias.
    pub fn new(in_features: usize, out_features: usize, device: &Device) -> Self {
        let bias = Tensor::zeros(&[out_features], device).requires_grad_();
        Self {
            weight: xavier(in_features, out_features, device),
            bias: Some(bias),
        }
    }

    /// Create a Linear layer without bias.
    pub fn new_no_bias(in_features: usize, out_features: usize, device: &Device) -> Self {
        Self {
            weight: xavier(in_features, out_features, device),
            bias: None,
        }
    }

    /// Create a Linear layer from existing weights (`[in, out]`).
    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    /// Input feature count.
    pub fn in_features(&self) -> usize {
        self.weight.shape[0]
    }

    /// Output feature count.
    pub fn out_features(&self) -> usize {
        self.weight.shape[1]
    }
}

/// Xavier/Glorot normal init: std = sqrt(2 / (in + out)).
fn xavier(in_features: usize, out_features: usize, device: &Device) -> Tensor {
    let scale = (2.0 / (in_features + out_features) as f32).sqrt();
    Tensor::randn(&[in_features, out_features], device)
        .mul_scalar(scale)
        .requires_grad_()
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        crate::functional::linear(x, &self.weight, self.bias.as_ref())
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.named_parameters()
            .into_iter()
            .map(|(_, p)| p)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = vec![("weight".to_string(), self.weight.clone())];
        if let Some(ref bias) = self.bias {
            params.push(("bias".to_string(), bias.clone()));
        }
        params
    }
}

impl std::fmt::Debug for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Linear")
            .field("in_features", &self.in_features())
            .field("out_features", &self.out_features())
            .field("bias", &self.bias.is_some())
            .finish()
    }
}
