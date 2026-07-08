//! Tensor Module
//!
//! Core tensor data structure with GPU storage and automatic differentiation.

mod autograd;
mod core;
mod ops;

pub use core::Tensor;
