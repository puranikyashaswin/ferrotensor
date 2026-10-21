//! Optimizer Trait and gradient clipping.

use crate::tensor::{no_grad, Tensor};

/// Base trait for all optimizers.
pub trait Optimizer {
    /// Perform one optimization step (update parameters from their grads).
    fn step(&mut self);

    /// Zero all parameter gradients.
    fn zero_grad(&self);

    /// Get the learning rate.
    fn lr(&self) -> f32;

    /// Set the learning rate.
    fn set_lr(&mut self, lr: f32);
}

/// Scale gradients so their global L2 norm is at most `max_norm`.
///
/// Returns the norm before clipping (useful for logging). Parameters without
/// a gradient are skipped.
///
/// # Example
/// ```rust
/// use rusty::prelude::*;
///
/// let w = Tensor::from_data(&[3.0, 4.0], &[2], &Device::cpu()).requires_grad_();
/// w.mul(&w).sum().mul_scalar(0.5).backward(); // grad = w = [3, 4], norm 5
/// let norm = optim::clip_grad_norm(&[w.clone()], 1.0);
/// assert!((norm - 5.0).abs() < 1e-6);
/// let g = w.grad().unwrap().to_vec();
/// assert!((g[0] - 0.6).abs() < 1e-5 && (g[1] - 0.8).abs() < 1e-5);
/// ```
pub fn clip_grad_norm(params: &[Tensor], max_norm: f32) -> f32 {
    no_grad(|| {
        let grads: Vec<Tensor> = params.iter().filter_map(Tensor::grad).collect();
        let total: f32 = grads
            .iter()
            .map(|g| g.square().sum().item())
            .sum::<f32>()
            .sqrt();
        if total > max_norm {
            let scale = max_norm / (total + 1e-6);
            for g in &grads {
                g.assign(&g.mul_scalar(scale));
            }
        }
        total
    })
}

/// Clamp every gradient element to `[-clip, clip]`.
pub fn clip_grad_value(params: &[Tensor], clip: f32) {
    no_grad(|| {
        for g in params.iter().filter_map(Tensor::grad) {
            g.assign(&g.clamp(-clip, clip));
        }
    })
}
