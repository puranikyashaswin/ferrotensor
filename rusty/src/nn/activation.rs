//! Activation Functions
//!
//! Non-linear activation functions as neural network modules.

use super::Module;
use crate::tensor::Tensor;

/// ReLU activation: max(0, x)
#[derive(Debug, Clone, Copy)]
pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// SiLU (Swish) activation: x * sigmoid(x)
#[derive(Debug, Clone, Copy)]
pub struct SiLU;

impl Module for SiLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.silu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// GELU activation: x * Φ(x) (Gaussian Error Linear Unit)
#[derive(Debug, Clone, Copy)]
pub struct GELU;

impl Module for GELU {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.gelu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Sigmoid activation: 1 / (1 + exp(-x))
#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.sigmoid()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Tanh activation
#[derive(Debug, Clone, Copy)]
pub struct Tanh;

/// Leaky ReLU: `x` for `x > 0`, else `negative_slope * x`.
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU {
    /// Slope for negative inputs.
    pub negative_slope: f32,
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self {
            negative_slope: 0.01,
        }
    }
}

impl Module for LeakyReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.leaky_relu(self.negative_slope)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

impl Module for Tanh {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.tanh()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Softmax activation along a dimension
#[derive(Debug, Clone, Copy)]
pub struct Softmax {
    /// Dimension to normalize over (negative counts from the end).
    pub dim: isize,
}

impl Softmax {
    /// Softmax over `dim`.
    pub fn new(dim: isize) -> Self {
        Self { dim }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self { dim: -1 }
    }
}

impl Module for Softmax {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.softmax(self.dim)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}
