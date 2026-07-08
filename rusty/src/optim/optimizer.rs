//! Optimizer Trait
//!
//! Base trait for all optimizers.

use crate::tensor::Tensor;

/// Base trait for all optimizers.
pub trait Optimizer {
    /// Perform one optimization step (update parameters).
    fn step(&mut self);

    /// Zero all parameter gradients.
    fn zero_grad(&self);

    /// Get the learning rate.
    fn lr(&self) -> f32;

    /// Set the learning rate.
    fn set_lr(&mut self, lr: f32);
}

/// Helper to extract parameters from a module.
pub fn collect_parameters<M: crate::nn::Module>(module: &M) -> Vec<Tensor> {
    module.parameters()
}
