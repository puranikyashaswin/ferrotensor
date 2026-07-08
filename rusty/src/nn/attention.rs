//! Multi-Head Attention
//!
//! Self-attention mechanism used in transformers.

use super::{Linear, Module};
use crate::tensor::Tensor;
use crate::Device;

/// Multi-Head Attention layer.
///
/// Implements scaled dot-product attention with multiple heads.
pub struct MultiHeadAttention {
    /// Query projection
    pub q_proj: Linear,
    /// Key projection
    pub k_proj: Linear,
    /// Value projection
    pub v_proj: Linear,
    /// Output projection
    pub o_proj: Linear,
    /// Number of attention heads
    pub n_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
}

impl MultiHeadAttention {
    /// Create a new MultiHeadAttention layer.
    pub fn new(hidden_dim: usize, n_heads: usize, device: &Device) -> Self {
        assert!(
            hidden_dim % n_heads == 0,
            "hidden_dim must be divisible by n_heads"
        );
        let head_dim = hidden_dim / n_heads;

        Self {
            q_proj: Linear::new_no_bias(hidden_dim, hidden_dim, device),
            k_proj: Linear::new_no_bias(hidden_dim, hidden_dim, device),
            v_proj: Linear::new_no_bias(hidden_dim, hidden_dim, device),
            o_proj: Linear::new_no_bias(hidden_dim, hidden_dim, device),
            n_heads,
            head_dim,
            hidden_dim,
        }
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        // Project to Q, K, V
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // For now, simplified single-head attention (no reshape/split)
        // TODO: Implement proper multi-head attention with reshape

        // Attention scores: Q @ K^T / sqrt(d)
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scores = q.matmul(&k).mul_scalar(scale);

        // Softmax
        let attn = scores.softmax(-1);

        // Apply to values
        let out = attn.matmul(&v);

        // Output projection
        self.o_proj.forward(&out)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.q_proj.parameters();
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.o_proj.parameters());
        params
    }
}

impl std::fmt::Debug for MultiHeadAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiHeadAttention")
            .field("hidden_dim", &self.hidden_dim)
            .field("n_heads", &self.n_heads)
            .field("head_dim", &self.head_dim)
            .finish()
    }
}
