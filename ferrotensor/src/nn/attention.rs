//! Multi-Head Attention
//!
//! Self-attention mechanism used in transformers.

use super::module::prefixed;
use super::{Linear, Module};
use crate::functional::scaled_dot_product_attention;
use crate::tensor::Tensor;
use crate::Device;

/// Multi-head self-attention.
///
/// Input is `[seq, hidden]` or `[batch, seq, hidden]`; the output has the
/// same shape. Heads attend independently with scaled dot-product attention,
/// optionally causal (each position sees only itself and earlier ones).
///
/// # Example
/// ```rust
/// use ferrotensor::prelude::*;
///
/// let device = Device::cpu();
/// let attn = nn::MultiHeadAttention::new(32, 4, &device).causal(true);
/// let x = Tensor::randn(&[2, 5, 32], &device);
/// assert_eq!(attn.forward(&x).shape, vec![2, 5, 32]);
/// ```
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
    /// Mask future positions
    pub is_causal: bool,
}

impl MultiHeadAttention {
    /// Create a (non-causal) MultiHeadAttention layer.
    ///
    /// # Panics
    /// Panics unless `hidden_dim` is divisible by `n_heads`.
    pub fn new(hidden_dim: usize, n_heads: usize, device: &Device) -> Self {
        assert!(
            n_heads > 0 && hidden_dim.is_multiple_of(n_heads),
            "hidden_dim must be divisible by n_heads"
        );
        Self {
            q_proj: Linear::new_no_bias(hidden_dim, hidden_dim, device),
            k_proj: Linear::new_no_bias(hidden_dim, hidden_dim, device),
            v_proj: Linear::new_no_bias(hidden_dim, hidden_dim, device),
            o_proj: Linear::new_no_bias(hidden_dim, hidden_dim, device),
            n_heads,
            head_dim: hidden_dim / n_heads,
            hidden_dim,
            is_causal: false,
        }
    }

    /// Enable or disable the causal mask.
    pub fn causal(mut self, causal: bool) -> Self {
        self.is_causal = causal;
        self
    }

    /// `[b, l, hidden]` to `[b, heads, l, head_dim]`.
    fn split_heads(&self, x: &Tensor, b: usize, l: usize) -> Tensor {
        x.reshape(&[b, l, self.n_heads, self.head_dim])
            .permute(&[0, 2, 1, 3])
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        let batched = x.ndim() == 3;
        assert!(
            (x.ndim() == 2 || batched) && x.shape[x.ndim() - 1] == self.hidden_dim,
            "MultiHeadAttention({}) expects [seq, {0}] or [batch, seq, {0}], got {:?}",
            self.hidden_dim,
            x.shape
        );
        let (b, l) = if batched {
            (x.shape[0], x.shape[1])
        } else {
            (1, x.shape[0])
        };
        let q = self.split_heads(&self.q_proj.forward(x), b, l);
        let k = self.split_heads(&self.k_proj.forward(x), b, l);
        let v = self.split_heads(&self.v_proj.forward(x), b, l);
        let heads = scaled_dot_product_attention(&q, &k, &v, self.is_causal);
        let merged = heads.permute(&[0, 2, 1, 3]).reshape(&x.shape);
        self.o_proj.forward(&merged)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.named_parameters()
            .into_iter()
            .map(|(_, p)| p)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut params = prefixed("q_proj", self.q_proj.named_parameters());
        params.extend(prefixed("k_proj", self.k_proj.named_parameters()));
        params.extend(prefixed("v_proj", self.v_proj.named_parameters()));
        params.extend(prefixed("o_proj", self.o_proj.named_parameters()));
        params
    }
}

impl std::fmt::Debug for MultiHeadAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiHeadAttention")
            .field("hidden_dim", &self.hidden_dim)
            .field("n_heads", &self.n_heads)
            .field("head_dim", &self.head_dim)
            .field("is_causal", &self.is_causal)
            .finish()
    }
}
