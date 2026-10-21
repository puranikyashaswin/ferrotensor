//! Model building blocks on top of `ferrotensor-autograd`: linear layers with
//! LoRA, RMSNorm, RoPE grouped-query attention, SwiGLU MLPs, Llama and LFM2
//! decoders, and the fused attention layers in [`flash_attention`].

use ferrotensor_autograd::Tensor;
use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use std::collections::HashMap;
use std::sync::Arc;

pub mod flash_attention;
/// LFM2 (Liquid foundation model) blocks.
pub mod lfm;
pub mod model_builder;

pub use flash_attention::{
    AttentionMask, FlashAttention, FlashAttentionGrads, GroupedQueryAttention,
    KVCache as FlashKVCache, MultiQueryAttention, SlidingWindowAttention,
};
pub use lfm::Lfm2Model;

// --- LoRA Adapter ---
/// Low-rank adapter added to a [`Linear`]: `x @ a @ b * alpha / rank`.
#[derive(Debug, Clone)]
pub struct LoRAAdapter {
    /// Down projection `[in, rank]`.
    pub a: Tensor, // [In, Rank]
    /// Up projection `[rank, out]` (zero-initialised, so training starts
    /// from the base model).
    pub b: Tensor, // [Rank, Out]
    /// Scale numerator; the update is scaled by `alpha / rank`.
    pub alpha: f32,
    /// Adapter rank.
    pub rank: usize,
}

impl LoRAAdapter {
    /// An adapter for an `in_dim -> out_dim` projection. `a` is drawn
    /// uniformly from `±1/sqrt(in_dim)` (Kaiming-uniform, as in PEFT) with a
    /// deterministic generator seeded from the shape, and `b` is zero.
    ///
    /// A constant `a` would make every rank component identical, so the
    /// adapter could never learn more than a rank-1 update.
    pub fn new(
        ctx: Arc<WgpuContext>,
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f32,
    ) -> Self {
        Self::new_seeded(ctx, in_dim, out_dim, rank, alpha, 0)
    }

    /// [`new`](Self::new) with an extra `seed`, so adapters of equal shape
    /// (e.g. every layer's `q_proj`) start from different matrices.
    pub fn new_seeded(
        ctx: Arc<WgpuContext>,
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f32,
        seed: u64,
    ) -> Self {
        let bound = 1.0 / (in_dim.max(1) as f32).sqrt();
        let mut state = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((in_dim as u64) << 32 | (out_dim as u64) << 8 | rank as u64)
            | 1;
        let a_data: Vec<f32> = (0..in_dim * rank)
            .map(|_| {
                // xorshift64*
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
                let unit = bits as f32 / (1u64 << 24) as f32;
                (unit * 2.0 - 1.0) * bound
            })
            .collect();
        Self {
            a: Tensor::new(
                ctx.clone(),
                UnifiedTensor::new(&ctx, &a_data, &[in_dim, rank]),
            ),
            b: Tensor::new(ctx.clone(), UnifiedTensor::zeros(&ctx, &[rank, out_dim])),
            alpha,
            rank,
        }
    }

    /// The adapter tensors `[a, b]`.
    pub fn params(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// Bias-free linear layer `y = x @ weight` with an optional LoRA adapter.
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight `[in, out]` (already transposed from the PyTorch `[out, in]`).
    pub weight: Tensor,
    /// Optional low-rank adapter added to the output.
    pub lora: Option<LoRAAdapter>,
}

impl Linear {
    fn project_2d(&self, engine: &ComputeEngine, x: &Tensor) -> Tensor {
        let mut out = Tensor::matmul(x, &self.weight, engine);

        if let Some(lora) = &self.lora {
            let temp = Tensor::matmul(x, &lora.a, engine);
            let lora_out = Tensor::matmul(&temp, &lora.b, engine);
            let scale = lora.alpha / lora.rank as f32;
            let scale_tensor = Tensor::new(
                x.ctx.clone(),
                UnifiedTensor::new(
                    &x.ctx,
                    &vec![scale; lora_out.data.shape.iter().product()],
                    &lora_out.data.shape,
                ),
            );
            let scaled_lora = Tensor::mul(&lora_out, &scale_tensor, engine);
            out = Tensor::add(&out, &scaled_lora, engine);
        }

        out
    }

    /// A layer without LoRA.
    pub fn new(weight: Tensor) -> Self {
        Self { weight, lora: None }
    }

    /// A layer with a LoRA adapter.
    pub fn with_lora(weight: Tensor, lora: LoRAAdapter) -> Self {
        Self {
            weight,
            lora: Some(lora),
        }
    }

    /// Projects `[tokens, in]` or `[batch, seq, in]` to the same leading
    /// shape with `out` features.
    pub fn forward(&self, engine: &ComputeEngine, x: &Tensor) -> Tensor {
        match x.data.shape.as_slice() {
            [_, _] => self.project_2d(engine, x),
            [batch, seq, _hidden] => {
                let flat = Tensor::reshape_2d(x);
                let projected = self.project_2d(engine, &flat);
                Tensor::unflatten_last(&projected, *batch, *seq)
            }
            shape => panic!("Linear::forward expects 2D or 3D tensor, got {shape:?}"),
        }
    }

    /// Trainable tensors (weights, then any LoRA adapters).
    pub fn params(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(lora) = &self.lora {
            p.extend(lora.params());
        }
        p
    }

    /// Parameters keyed by Hugging Face-style names under `prefix`.
    pub fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(format!("{}.weight", prefix), self.weight.clone());
        if let Some(lora) = &self.lora {
            map.insert(format!("{}.lora_A.weight", prefix), lora.a.clone());
            map.insert(format!("{}.lora_B.weight", prefix), lora.b.clone());
        }
        map
    }

    /// Only the LoRA adapter tensors (what LoRA fine-tuning updates).
    pub fn lora_params(&self) -> Vec<Tensor> {
        if let Some(lora) = &self.lora {
            lora.params()
        } else {
            vec![]
        }
    }
}

/// Token embedding table.
pub struct Embedding {
    /// Embedding table `[vocab, hidden]`.
    pub weight: Tensor,
}

impl Embedding {
    /// Wraps an embedding table.
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    /// Looks up `input_ids` and returns `[1, seq, hidden]`. The lookup runs
    /// on the CPU (the table is read back from the GPU).
    pub fn forward(&self, ctx: &WgpuContext, _engine: &ComputeEngine, input_ids: &[u32]) -> Tensor {
        let hidden_dim = self.weight.data.shape[1];
        let seq_len = input_ids.len();
        let mut out_data = Vec::with_capacity(seq_len * hidden_dim);
        let w_data = pollster::block_on(self.weight.data.to_vec(ctx));
        for &id in input_ids {
            let start = (id as usize) * hidden_dim;
            let end = start + hidden_dim;
            out_data.extend_from_slice(&w_data[start..end]);
        }
        Tensor::new(
            self.weight.ctx.clone(),
            UnifiedTensor::new(ctx, &out_data, &[1, seq_len, hidden_dim]),
        )
    }

    /// Trainable tensors (weights, then any LoRA adapters).
    pub fn params(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    /// Parameters keyed by Hugging Face-style names under `prefix`.
    pub fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(format!("{}.weight", prefix), self.weight.clone());
        map
    }
}

/// Root-mean-square layer norm over the last dimension.
pub struct RMSNorm {
    /// Per-feature scale `[hidden]`.
    pub weight: Tensor,
    /// Epsilon added to the mean square.
    pub eps: f32,
}

impl RMSNorm {
    /// Wraps a scale vector and epsilon.
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Normalises `input` over its last dimension.
    pub fn forward(&self, engine: &ComputeEngine, input: &Tensor) -> Tensor {
        Tensor::rms_norm(input, &self.weight, self.eps, engine)
    }

    /// Trainable tensors (weights, then any LoRA adapters).
    pub fn params(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    /// Parameters keyed by Hugging Face-style names under `prefix`.
    pub fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(format!("{}.weight", prefix), self.weight.clone());
        map
    }
}

/// Causal grouped-query self-attention with RoPE (Llama style).
pub struct Attention {
    /// Query projection `[hidden, n_heads * head_dim]`.
    pub q_proj: Linear,
    /// Key projection `[hidden, num_kv_heads * head_dim]`.
    pub k_proj: Linear,
    /// Value projection `[hidden, num_kv_heads * head_dim]`.
    pub v_proj: Linear,
    /// Output projection `[n_heads * head_dim, hidden]`.
    pub o_proj: Linear,
    /// Number of query heads.
    pub n_heads: usize,
    /// Dimension of each head (at most 128).
    pub head_dim: usize,
    /// Number of key/value heads; divides `n_heads`.
    pub num_kv_heads: usize,
    /// Attention dropout probability.
    pub dropout_prob: f32,
    /// RoPE base frequency (`rope_theta` in Hugging Face configs).
    pub rope_theta: f32,
}

impl Attention {
    /// Attends over `x` (`[batch, seq, hidden]`) whose first token is at
    /// position `pos`. With `cache`, the new keys/values are written at `pos`
    /// and the queries attend to the whole cache up to their own position.
    pub fn forward(
        &self,
        engine: &ComputeEngine,
        x: &Tensor,
        pos: usize,
        cache: Option<&KVCacheLayer>,
        mask: Option<&UnifiedTensor>,
    ) -> Tensor {
        let q = self.q_proj.forward(engine, x);
        let k = self.k_proj.forward(engine, x);
        let v = self.v_proj.forward(engine, x);

        let q = Tensor::rope_theta(&q, self.head_dim, pos, self.rope_theta, engine);
        let k = Tensor::rope_theta(&k, self.head_dim, pos, self.rope_theta, engine);

        // Reshape for GQA [batch, seq, n_heads, head_dim]
        let batch = q.data.shape[0];
        let seq_len = q.data.shape[1];

        let Some(c) = cache else {
            // Differentiable path (training and full-sequence evaluation).
            let q_4d = Tensor::reshape(&q, &[batch, seq_len, self.n_heads, self.head_dim]);
            let k_4d = Tensor::reshape(&k, &[batch, seq_len, self.num_kv_heads, self.head_dim]);
            let v_4d = Tensor::reshape(&v, &[batch, seq_len, self.num_kv_heads, self.head_dim]);
            let mask = mask.map(|m| Tensor::frozen(x.ctx.clone(), m.clone()));
            let out = Tensor::flash_attention(
                &q_4d,
                &k_4d,
                &v_4d,
                mask.as_ref(),
                true,
                self.dropout_prob,
                0,
                engine,
            );
            let out = Tensor::reshape(&out, &[batch, seq_len, self.n_heads * self.head_dim]);
            return self.o_proj.forward(engine, &out);
        };

        // Incremental decoding against a KV cache (inference only, no autograd).
        let q_4d = q
            .data
            .reshaped(&[batch, seq_len, self.n_heads, self.head_dim]);
        let gqa = GroupedQueryAttention::new(
            self.n_heads * self.head_dim,
            self.n_heads,
            self.num_kv_heads,
            true, // causal
            self.dropout_prob,
        );

        // Write the new keys/values at `pos` (cache is `[1, max_seq, kv * head_dim]`).
        let offset = pos * self.num_kv_heads * self.head_dim * 4;
        engine.copy_tensor(&x.ctx, &k.data, &c.key, offset);
        engine.copy_tensor(&x.ctx, &v.data, &c.value, offset);

        let cache_shape = &c.key.shape;
        let kv_shape = [
            cache_shape[0],
            cache_shape[1],
            self.num_kv_heads,
            self.head_dim,
        ];
        let c_k_4d = c.key.reshaped(&kv_shape);
        let c_v_4d = c.value.reshaped(&kv_shape);

        // Queries sit at positions pos..pos + seq_len; causal masking
        // hides the cache slots that have not been written yet.
        let attn_out_4d = gqa.forward_at(&q_4d, &c_k_4d, &c_v_4d, mask, pos, &x.ctx, engine);
        let attn_out = attn_out_4d.reshaped(&[batch, seq_len, self.n_heads * self.head_dim]);
        let attn_tensor = Tensor::new(x.ctx.clone(), attn_out);
        self.o_proj.forward(engine, &attn_tensor)
    }

    /// Trainable tensors (weights, then any LoRA adapters).
    pub fn params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.q_proj.params());
        p.extend(self.k_proj.params());
        p.extend(self.v_proj.params());
        p.extend(self.o_proj.params());
        p
    }

    /// Parameters keyed by Hugging Face-style names under `prefix`.
    pub fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.extend(self.q_proj.named_params(&format!("{}.q_proj", prefix)));
        map.extend(self.k_proj.named_params(&format!("{}.k_proj", prefix)));
        map.extend(self.v_proj.named_params(&format!("{}.v_proj", prefix)));
        map.extend(self.o_proj.named_params(&format!("{}.o_proj", prefix)));
        map
    }

    /// Only the LoRA adapter tensors (what LoRA fine-tuning updates).
    pub fn lora_params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.q_proj.lora_params());
        p.extend(self.k_proj.lora_params());
        p.extend(self.v_proj.lora_params());
        p.extend(self.o_proj.lora_params());
        p
    }
}

/// SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`.
pub struct MLP {
    /// Gate projection (passed through SiLU).
    pub gate_proj: Linear,
    /// Up projection.
    pub up_proj: Linear,
    /// Down projection back to the hidden size.
    pub down_proj: Linear,
}

impl MLP {
    /// Applies the feed-forward to `[tokens, hidden]` or `[batch, seq, hidden]`.
    pub fn forward(&self, engine: &ComputeEngine, input: &Tensor) -> Tensor {
        let up = self.up_proj.forward(engine, input);
        let gate = self.gate_proj.forward(engine, input);
        // silu_gate_fused(a, b) = silu(a) * b; SwiGLU applies SiLU to the gate.
        let fused = Tensor::silu_gate_fused(&gate, &up, engine);
        self.down_proj.forward(engine, &fused)
    }

    /// Trainable tensors (weights, then any LoRA adapters).
    pub fn params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.gate_proj.params());
        p.extend(self.up_proj.params());
        p.extend(self.down_proj.params());
        p
    }

    /// Parameters keyed by Hugging Face-style names under `prefix`.
    pub fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.extend(
            self.gate_proj
                .named_params(&format!("{}.gate_proj", prefix)),
        );
        map.extend(self.up_proj.named_params(&format!("{}.up_proj", prefix)));
        map.extend(
            self.down_proj
                .named_params(&format!("{}.down_proj", prefix)),
        );
        map
    }

    /// Only the LoRA adapter tensors (what LoRA fine-tuning updates).
    pub fn lora_params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.gate_proj.lora_params());
        p.extend(self.up_proj.lora_params());
        p.extend(self.down_proj.lora_params());
        p
    }
}

/// Pre-norm Llama decoder block: attention and MLP with residuals.
pub struct LlamaBlock {
    /// Norm before attention (`input_layernorm`).
    pub norm1: RMSNorm,
    /// Self-attention.
    pub attn: Attention,
    /// Norm before the MLP (`post_attention_layernorm`).
    pub norm2: RMSNorm,
    /// Feed-forward.
    pub mlp: MLP,
}

impl LlamaBlock {
    /// Runs the block on `input` (`[batch, seq, hidden]`); see
    /// [`Attention::forward`] for `pos`, `cache` and `mask`.
    pub fn forward(
        &self,
        engine: &ComputeEngine,
        input: &Tensor,
        pos: usize,
        cache: Option<&KVCacheLayer>,
        mask: Option<&UnifiedTensor>,
    ) -> Tensor {
        let normalized1 = self.norm1.forward(engine, input);
        let attn_out = self.attn.forward(engine, &normalized1, pos, cache, mask);
        let h = Tensor::add(input, &attn_out, engine);
        let normalized2 = self.norm2.forward(engine, &h);
        let mlp_out = self.mlp.forward(engine, &normalized2);
        Tensor::add(&h, &mlp_out, engine)
    }

    /// Trainable tensors (weights, then any LoRA adapters).
    pub fn params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.norm1.params());
        p.extend(self.attn.params());
        p.extend(self.norm2.params());
        p.extend(self.mlp.params());
        p
    }

    /// Parameters keyed by Hugging Face-style names under `prefix`.
    pub fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.extend(
            self.norm1
                .named_params(&format!("{}.input_layernorm", prefix)),
        );
        map.extend(self.attn.named_params(&format!("{}.self_attn", prefix)));
        map.extend(
            self.norm2
                .named_params(&format!("{}.post_attention_layernorm", prefix)),
        );
        map.extend(self.mlp.named_params(&format!("{}.mlp", prefix)));
        map
    }

    /// Only the LoRA adapter tensors (what LoRA fine-tuning updates).
    pub fn lora_params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.attn.lora_params());
        p.extend(self.mlp.lora_params());
        p
    }
}

/// A Llama-style decoder: embedding, blocks, final norm and LM head.
pub struct LlamaModel {
    /// Token embeddings.
    pub embed_tokens: Embedding,
    /// Decoder blocks.
    pub layers: Vec<LlamaBlock>,
    /// Final norm.
    pub norm: RMSNorm,
    /// Projection to vocabulary logits.
    pub lm_head: Linear,
}

impl LlamaModel {
    /// Logits `[1, seq, vocab]` for `input_ids`, the first of which is at
    /// `start_pos`.
    pub fn forward(
        &self,
        ctx: &WgpuContext,
        engine: &ComputeEngine,
        input_ids: &[u32],
        start_pos: usize,
        cache: Option<&mut KVCache>,
        mask: Option<&UnifiedTensor>,
    ) -> Tensor {
        let mut x = self.embed_tokens.forward(ctx, engine, input_ids);
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = cache.as_ref().map(|c| &c.layers[i]);
            x = layer.forward(engine, &x, start_pos, layer_cache, mask);
        }
        let x_norm = self.norm.forward(engine, &x);
        self.lm_head.forward(engine, &x_norm)
    }

    /// Trainable tensors (weights, then any LoRA adapters).
    pub fn params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.embed_tokens.params());
        for layer in &self.layers {
            p.extend(layer.params());
        }
        p.extend(self.norm.params());
        p.extend(self.lm_head.params());
        p
    }

    /// Parameters keyed by Hugging Face-style names under `prefix`.
    pub fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.extend(
            self.embed_tokens
                .named_params(&format!("{}.embed_tokens", prefix)),
        );
        for (i, layer) in self.layers.iter().enumerate() {
            map.extend(layer.named_params(&format!("{}.layers.{}", prefix, i)));
        }
        map.extend(self.norm.named_params(&format!("{}.norm", prefix)));
        map.extend(self.lm_head.named_params("lm_head"));
        map
    }

    /// Only the LoRA adapter tensors (what LoRA fine-tuning updates).
    pub fn lora_params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for layer in &self.layers {
            p.extend(layer.lora_params());
        }
        p.extend(self.lm_head.lora_params());
        p
    }
}

/// Cached keys and values of one layer for incremental decoding.
pub struct KVCacheLayer {
    /// Keys `[batch, max_seq, num_kv_heads * head_dim]`.
    pub key: UnifiedTensor,
    /// Values, same shape as `key`.
    pub value: UnifiedTensor,
}
/// Per-layer KV caches of a model.
pub struct KVCache {
    /// One cache per decoder layer.
    pub layers: Vec<KVCacheLayer>,
}

/// Helper to create a padding mask for Flash Attention V2 (additive mask)
///
/// Returns a tensor of shape [batch_size, 1, 1, max_seq_len] (or broadcastable).
/// - 0.0 for valid positions
/// - -1e9 for padded positions
pub fn create_padding_mask(
    ctx: &WgpuContext,
    lengths: &[usize],
    max_seq_len: usize,
) -> UnifiedTensor {
    let batch_size = lengths.len();
    let mut mask_data = vec![0.0f32; batch_size * max_seq_len];

    for (b, &len) in lengths.iter().enumerate() {
        for s in 0..max_seq_len {
            if s >= len {
                mask_data[b * max_seq_len + s] = -1e9;
            }
        }
    }

    UnifiedTensor::new(ctx, &mask_data, &[batch_size, 1, 1, max_seq_len])
}
