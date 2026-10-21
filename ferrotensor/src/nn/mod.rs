//! Neural Network Module
//!
//! PyTorch-style neural network building blocks.

mod activation;
mod attention;
mod container;
mod dropout;
mod embedding;
mod linear;
mod module;
mod normalization;

pub use activation::{LeakyReLU, ReLU, SiLU, Sigmoid, Softmax, Tanh, GELU};
pub use attention::MultiHeadAttention;
pub use container::{Layer, ModuleList, Sequential};
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use linear::Linear;
pub use module::{prefixed, Module, ModuleInit};
pub use normalization::{LayerNorm, RMSNorm};
