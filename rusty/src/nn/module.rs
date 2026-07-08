//! Module Trait
//!
//! Base trait for all neural network modules.

use crate::tensor::Tensor;
use crate::Device;

/// Base trait for all neural network modules.
///
/// # Example
/// ```rust,no_run
/// use rusty::prelude::*;
/// use rusty::nn::Module;
///
/// struct MyModel {
///     fc1: nn::Linear,
///     fc2: nn::Linear,
/// }
///
/// impl Module for MyModel {
///     fn forward(&self, x: &Tensor) -> Tensor {
///         let x = self.fc1.forward(x).relu();
///         self.fc2.forward(&x)
///     }
///     
///     fn parameters(&self) -> Vec<Tensor> {
///         let mut p = self.fc1.parameters();
///         p.extend(self.fc2.parameters());
///         p
///     }
/// }
/// ```
pub trait Module: Send + Sync {
    /// Forward pass.
    fn forward(&self, x: &Tensor) -> Tensor;

    /// Get all trainable parameters.
    fn parameters(&self) -> Vec<Tensor>;

    /// Set training mode.
    fn train(&mut self) {}

    /// Set evaluation mode.
    fn eval(&mut self) {}

    /// Zero gradients for all parameters.
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }

    /// Count total number of parameters.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

/// Trait for modules that can be initialized with a device.
pub trait ModuleInit {
    fn new(device: &Device) -> Self;
}
