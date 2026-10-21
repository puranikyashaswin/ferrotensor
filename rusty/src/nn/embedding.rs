//! Embedding Layer
//!
//! Token embedding lookup table.

use super::Module;
use crate::tensor::Tensor;
use crate::Device;

/// Embedding layer: maps integer token ids to learned dense vectors.
///
/// # Example
/// ```rust
/// use rusty::prelude::*;
///
/// let device = Device::cpu();
/// let emb = nn::Embedding::new(100, 16, &device);
/// let out = emb.forward_ids(&[3, 7, 3]);
/// assert_eq!(out.shape, vec![3, 16]);
/// ```
pub struct Embedding {
    /// Embedding weights [vocab_size, embedding_dim]
    pub weight: Tensor,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
}

impl Embedding {
    /// Create an Embedding with N(0, 0.02^2) weights.
    pub fn new(vocab_size: usize, embedding_dim: usize, device: &Device) -> Self {
        let weight = Tensor::randn(&[vocab_size, embedding_dim], device)
            .mul_scalar(0.02)
            .requires_grad_();
        Self {
            weight,
            vocab_size,
            embedding_dim,
        }
    }

    /// Create an Embedding from existing weights (`[vocab, dim]`).
    pub fn from_weights(weight: Tensor) -> Self {
        let vocab_size = weight.shape[0];
        let embedding_dim = weight.shape[1];
        Self {
            weight,
            vocab_size,
            embedding_dim,
        }
    }

    /// Look up `ids`; returns `[ids.len(), embedding_dim]` (differentiable
    /// in the weights).
    pub fn forward_ids(&self, ids: &[usize]) -> Tensor {
        crate::functional::embedding(&self.weight, ids)
    }

    /// Look up `u32` token ids.
    pub fn forward_indices(&self, token_ids: &[u32]) -> Tensor {
        let ids: Vec<usize> = token_ids.iter().map(|&id| id as usize).collect();
        self.forward_ids(&ids)
    }
}

impl Module for Embedding {
    /// `x` holds token ids stored as `f32`; the output has shape
    /// `x.shape + [embedding_dim]`.
    fn forward(&self, x: &Tensor) -> Tensor {
        let ids: Vec<usize> = x.to_vec().iter().map(|&v| v as usize).collect();
        let out_shape = [&x.shape[..], &[self.embedding_dim]].concat();
        self.forward_ids(&ids).reshape(&out_shape)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        vec![("weight".to_string(), self.weight.clone())]
    }
}

impl std::fmt::Debug for Embedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedding")
            .field("vocab_size", &self.vocab_size)
            .field("embedding_dim", &self.embedding_dim)
            .finish()
    }
}
