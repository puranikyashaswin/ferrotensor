//! Neural Network Module
//!
//! PyTorch-style neural network building blocks.

mod activation;
mod attention;
mod container;
mod embedding;
mod linear;
mod module;
mod normalization;

pub use activation::{ReLU, SiLU, Sigmoid, Softmax, Tanh, GELU};
pub use attention::MultiHeadAttention;
pub use container::{ModuleList, Sequential};
pub use embedding::Embedding;
pub use linear::Linear;
pub use module::Module;
pub use normalization::{LayerNorm, RMSNorm};
