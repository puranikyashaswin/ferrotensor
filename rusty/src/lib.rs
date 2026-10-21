//! # Rusty - GPU-Accelerated Machine Learning Library
//!
//! Rusty is a high-performance machine learning library written in pure Rust,
//! designed to replace Python/PyTorch for AI training and inference.
//!
//! ## Features
//!
//! - **GPU Acceleration**: Native Metal (macOS) and Vulkan support via wgpu
//! - **Automatic Differentiation**: Partial — see project status in README
//! - **Neural Network Modules**: PyTorch-like `nn` module system (experimental)
//! - **Modern Architectures**: Transformer, Attention, LoRA (experimental)
//! - **Quantization**: Int8 dequantization on GPU
//!
//! ## Quick Start
//!
//! See `examples/basic_tensor.rs` for a working GPU matmul demo.
//! For the full stack, use `rusty-backend` + `rusty-graph` via `rusty-cli`.

#![warn(missing_docs)]

pub mod backend;
pub mod data;
pub mod functional;
pub mod nn;
pub mod optim;
pub mod serialize;
pub mod tensor;

mod device;

pub use device::Device;

/// Prelude module - import everything you need with `use rusty::prelude::*`
pub mod prelude {
    pub use crate::data::{DataLoader, Dataset};
    pub use crate::functional as F;
    pub use crate::nn;
    pub use crate::nn::Module;
    pub use crate::optim;
    pub use crate::optim::Optimizer;
    pub use crate::tensor::Tensor;
    pub use crate::Device;
}
