//! Multi-Head Attention
//!
//! Self-attention mechanism used in transformers.

use super::module::prefixed;
use super::{Linear, Module};
use crate::functional::scaled_dot_product_attention;
use crate::tensor::Tensor;
use crate::Device;

/// Keys and values of the positions a [`MultiHeadAttention`] layer has seen,
/// for incremental decoding with
/// [`forward_cached`](MultiHeadAttention::forward_cached).
///
/// Use one cache per layer and per sequence batch; [`clear`](KvCache::clear)
/// it to start a new sequence. Run decoding under
/// [`no_grad`](crate::no_grad) so the cache does not keep autograd graphs
/// alive.
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    /// Cached keys and values, each `[batch, heads, len, head_dim]`.
    kv: Option<(Tensor, Tensor)>,
}

impl KvCache {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of cached positions.
    pub fn len(&self) -> usize {
        self.kv.as_ref().map_or(0, |(k, _)| k.shape[2])
    }

    /// True when no position is cached.
    pub fn is_empty(&self) -> bool {
        self.kv.is_none()
    }

    /// Drop every cached position.
    pub fn clear(&mut self) {
        self.kv = None;
    }

    /// Cached keys and values, each `[batch, heads, len, head_dim]`.
    pub fn keys_values(&self) -> Option<(&Tensor, &Tensor)> {
        self.kv.as_ref().map(|(k, v)| (k, v))
    }
}

/// Multi-head self-attention.
///
/// Input is `[seq, hidden]` or `[batch, seq, hidden]`; the output has the
/// same shape. Heads attend independently with scaled dot-product attention,
/// optionally causal (each position sees only itself and earlier ones).
/// [`forward_cached`](Self::forward_cached) decodes incrementally with a
/// [`KvCache`].
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

    /// Attend from the new positions `x` (`[seq, hidden]` or
    /// `[batch, seq, hidden]`) to the positions in `cache` and to `x`
    /// itself, then append `x`'s keys and values to `cache`.
    ///
    /// For a causal layer, feeding a sequence in pieces (one token at a time
    /// or in chunks) gives the same outputs as one [`forward`](Module::forward)
    /// over the whole sequence, while each step projects only the new
    /// positions. A non-causal layer lets the new positions see every cached
    /// one.
    ///
    /// # Panics
    /// Panics when `x` has the wrong hidden size, or its batch size differs
    /// from the cached one.
    ///
    /// # Example
    /// ```rust
    /// use ferrotensor::no_grad;
    /// use ferrotensor::prelude::*;
    ///
    /// let device = Device::cpu();
    /// let attn = nn::MultiHeadAttention::new(16, 2, &device).causal(true);
    /// let x = Tensor::randn(&[1, 5, 16], &device);
    /// let full = attn.forward(&x);
    ///
    /// let mut cache = nn::KvCache::new();
    /// let steps: Vec<Tensor> = (0..5)
    ///     .map(|t| no_grad(|| attn.forward_cached(&x.narrow(1, t, 1), &mut cache)))
    ///     .collect();
    /// assert_eq!(cache.len(), 5);
    /// let step_by_step = Tensor::cat(&steps, 1);
    /// for (a, b) in full.to_vec().iter().zip(step_by_step.to_vec()) {
    ///     assert!((a - b).abs() < 1e-5);
    /// }
    /// ```
    pub fn forward_cached(&self, x: &Tensor, cache: &mut KvCache) -> Tensor {
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
        let mut k = self.split_heads(&self.k_proj.forward(x), b, l);
        let mut v = self.split_heads(&self.v_proj.forward(x), b, l);
        if let Some((past_k, past_v)) = cache.kv.take() {
            assert_eq!(
                past_k.shape[0], b,
                "KvCache holds batch size {}, got {b}",
                past_k.shape[0]
            );
            k = Tensor::cat(&[past_k, k], 2);
            v = Tensor::cat(&[past_v, v], 2);
        }
        let heads = scaled_dot_product_attention(&q, &k, &v, self.is_causal);
        cache.kv = Some((k, v));
        let merged = heads.permute(&[0, 2, 1, 3]).reshape(&x.shape);
        self.o_proj.forward(&merged)
    }

    /// `[b, l, hidden]` to `[b, heads, l, head_dim]`.
    fn split_heads(&self, x: &Tensor, b: usize, l: usize) -> Tensor {
        x.reshape(&[b, l, self.n_heads, self.head_dim])
            .permute(&[0, 2, 1, 3])
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_cached(x, &mut KvCache::new())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::no_grad;

    fn max_diff(a: &Tensor, b: &Tensor) -> f32 {
        assert_eq!(a.shape, b.shape);
        a.to_vec()
            .iter()
            .zip(b.to_vec())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Feed `x` (`[batch, seq, hidden]`) through the cache in pieces of the
    /// given sizes and concatenate the outputs along the sequence.
    fn decode(attn: &MultiHeadAttention, x: &Tensor, pieces: &[usize]) -> Tensor {
        let mut cache = KvCache::new();
        let mut start = 0;
        let outs: Vec<Tensor> = pieces
            .iter()
            .map(|&len| {
                let out = no_grad(|| attn.forward_cached(&x.narrow(1, start, len), &mut cache));
                start += len;
                assert_eq!(cache.len(), start);
                out
            })
            .collect();
        assert_eq!(start, x.shape[1]);
        Tensor::cat(&outs, 1)
    }

    fn check_incremental_matches_full(device: &Device) {
        let attn = MultiHeadAttention::new(16, 4, device).causal(true);
        let x = Tensor::randn(&[2, 7, 16], device);
        let full = no_grad(|| attn.forward(&x));
        for pieces in [&[1, 1, 1, 1, 1, 1, 1][..], &[3, 1, 2, 1], &[7]] {
            let diff = max_diff(&decode(&attn, &x, pieces), &full);
            assert!(diff < 1e-5, "pieces {pieces:?}: max diff {diff}");
        }
    }

    #[test]
    fn incremental_decoding_matches_full_sequence() {
        check_incremental_matches_full(&Device::cpu());
    }

    #[test]
    fn unbatched_input_uses_the_cache_too() {
        let d = Device::cpu();
        let attn = MultiHeadAttention::new(8, 2, &d).causal(true);
        let x = Tensor::randn(&[4, 8], &d);
        let full = attn.forward(&x);
        let mut cache = KvCache::new();
        let first = attn.forward_cached(&x.narrow(0, 0, 3), &mut cache);
        let last = attn.forward_cached(&x.narrow(0, 3, 1), &mut cache);
        let (k, v) = cache.keys_values().unwrap();
        assert_eq!(k.shape, vec![1, 2, 4, 4]);
        assert_eq!(v.shape, vec![1, 2, 4, 4]);
        assert!(max_diff(&Tensor::cat(&[first, last], 0), &full) < 1e-5);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn non_causal_cache_attends_to_every_cached_position() {
        // The last position of a non-causal layer sees the whole sequence,
        // with or without the cache.
        let d = Device::cpu();
        let attn = MultiHeadAttention::new(8, 2, &d);
        let x = Tensor::randn(&[1, 5, 8], &d);
        let full = attn.forward(&x).narrow(1, 4, 1);
        let mut cache = KvCache::new();
        attn.forward_cached(&x.narrow(1, 0, 4), &mut cache);
        let last = attn.forward_cached(&x.narrow(1, 4, 1), &mut cache);
        assert!(max_diff(&last, &full) < 1e-5);
    }

    #[test]
    #[should_panic(expected = "KvCache holds batch size 2")]
    fn cache_rejects_a_different_batch() {
        let d = Device::cpu();
        let attn = MultiHeadAttention::new(8, 2, &d).causal(true);
        let mut cache = KvCache::new();
        attn.forward_cached(&Tensor::randn(&[2, 1, 8], &d), &mut cache);
        attn.forward_cached(&Tensor::randn(&[3, 1, 8], &d), &mut cache);
    }

    /// With `FERROTENSOR_RUN_GPU_TESTS=1`: the cache works the same on the GPU.
    #[test]
    fn gpu_incremental_decoding_matches_full_sequence() {
        if std::env::var("FERROTENSOR_RUN_GPU_TESTS").as_deref() != Ok("1") {
            eprintln!("skipping GPU KV cache test; set FERROTENSOR_RUN_GPU_TESTS=1 to enable");
            return;
        }
        let gpu = Device::gpu().expect("FERROTENSOR_RUN_GPU_TESTS=1 but no GPU is available");
        check_incremental_matches_full(&gpu);
    }
}
