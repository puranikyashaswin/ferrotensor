//! Optimizers
//!
//! Gradient descent optimizers, gradient clipping and learning-rate
//! schedules.

mod adam;
mod optimizer;
mod scheduler;
mod sgd;

#[cfg(test)]
mod tests;

pub use adam::AdamW;
pub use optimizer::{clip_grad_norm, clip_grad_value, Optimizer};
pub use scheduler::{
    ConstantLr, CosineAnnealingLr, LrScheduler, StepLr, WarmupCosineLr, WarmupLinearLr,
};
pub use sgd::SGD;
