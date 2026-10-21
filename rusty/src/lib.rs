//! # Rusty - GPU-Accelerated Machine Learning Library
//!
//! A PyTorch-style tensor and neural-network API in pure Rust. Tensors live
//! on a [`Device`]: a GPU through wgpu (Metal, Vulkan, DX12) with hand-written
//! WGSL kernels, or the CPU, which needs no GPU and runs the same ops.
//!
//! ## Features
//!
//! - **Tensors**: broadcasting element-wise math, reductions over any dim,
//!   batched matmul, softmax/log-softmax, permute, slicing, concatenation,
//!   gathers.
//! - **Automatic differentiation**: every op records its backward pass;
//!   gradients are checked against finite differences on the CPU and
//!   against the CPU on the GPU.
//! - **`nn`**: Linear, Embedding, LayerNorm, RMSNorm, Dropout, multi-head
//!   (causal) attention, activations, Sequential.
//! - **`optim`**: SGD (momentum, Nesterov), AdamW, gradient clipping,
//!   warmup + cosine and other LR schedules.
//! - **`serialize`**: safetensors save/load and module state dicts.
//!
//! ## Quick Start
//!
//! ```rust
//! use rusty::prelude::*;
//!
//! let device = Device::cpu(); // or Device::best() for a GPU when present
//! let model = nn::Sequential::new(&device)
//!     .add(nn::Linear::new(2, 16, &device))
//!     .add(nn::Tanh)
//!     .add(nn::Linear::new(16, 2, &device));
//! let mut opt = optim::AdamW::new(model.parameters(), 0.05).weight_decay(0.0);
//!
//! // XOR
//! let x = Tensor::from_data(&[0., 0., 0., 1., 1., 0., 1., 1.], &[4, 2], &device);
//! let y = Tensor::from_data(&[0., 1., 1., 0.], &[4], &device);
//! for _ in 0..200 {
//!     opt.zero_grad();
//!     let loss = F::cross_entropy(&model.forward(&x), &y);
//!     loss.backward();
//!     opt.step();
//! }
//! ```
//!
//! The LLM fine-tuning stack (LoRA, LFM2/Llama blocks, the CLI) is built on
//! `rusty-backend`, `rusty-autograd` and `rusty-graph`.

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
pub use tensor::{no_grad, Tensor};

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
