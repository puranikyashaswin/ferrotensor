//! Tensor Module
//!
//! The [`Tensor`] type: storage on a CPU or GPU [`Device`](crate::Device),
//! differentiable ops, and the autograd engine.

mod autograd;
mod core;
mod cpu;
mod ops;
mod raw;
mod shape;

#[cfg(test)]
mod tests;

pub use autograd::{is_grad_enabled, no_grad, NoGradGuard};
pub use core::Tensor;
