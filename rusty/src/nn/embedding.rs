//! Embedding Layer
//!
//! Token embedding lookup table.

use super::Module;
use crate::tensor::Tensor;
use crate::Device;

/// Embedding layer for token lookup.
///
/// Maps integer token IDs to dense vectors.
pub struct Embedding {
    /// Embedding weights [vocab_size, embedding_dim]
    pub weight: Tensor,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
}

impl Embedding {
    /// Create a new Embedding layer with random initialization.
    pub fn new(vocab_size: usize, embedding_dim: usize, device: &Device) -> Self {
        // Initialize with small random values
        let weight = Tensor::randn(&[vocab_size, embedding_dim], device)
            .mul_scalar(0.02)
            .requires_grad_();

        Self {
            weight,
            vocab_size,
            embedding_dim,
        }
    }

    /// Create Embedding from existing weights.
    pub fn from_weights(weight: Tensor) -> Self {
        let vocab_size = weight.shape[0];
        let embedding_dim = weight.shape[1];
        Self {
            weight,
            vocab_size,
            embedding_dim,
        }
    }

    /// Lookup embeddings for given token IDs.
    pub fn forward_indices(&self, token_ids: &[u32]) -> Tensor {
        let weight_data = self.weight.to_vec();
        let seq_len = token_ids.len();

        // Gather embeddings
        let mut result = Vec::with_capacity(seq_len * self.embedding_dim);
        for &id in token_ids {
            let start = (id as usize) * self.embedding_dim;
            let end = start + self.embedding_dim;
            result.extend_from_slice(&weight_data[start..end]);
        }

        Tensor::from_data(&result, &[seq_len, self.embedding_dim], &self.weight.device)
    }
}

impl Module for Embedding {
    fn forward(&self, x: &Tensor) -> Tensor {
        // Extract token IDs from tensor and forward
        // Assumes x contains integer token IDs as floats
        let ids: Vec<u32> = x.to_vec().iter().map(|&v| v as u32).collect();
        self.forward_indices(&ids)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
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
