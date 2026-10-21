use rusty_autograd::Tensor;
use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};
use std::collections::HashMap;
use std::sync::Arc;

pub mod flash_attention;
pub mod lfm;
pub mod model_builder;

pub use flash_attention::{
    AttentionMask, FlashAttention, FlashAttentionGrads, GroupedQueryAttention,
    KVCache as FlashKVCache, MultiQueryAttention, SlidingWindowAttention,
};
pub use lfm::Lfm2Model;

// --- LoRA Adapter ---
#[derive(Debug, Clone)]
pub struct LoRAAdapter {
    pub a: Tensor, // [In, Rank]
    pub b: Tensor, // [Rank, Out]
    pub alpha: f32,
    pub rank: usize,
}

impl LoRAAdapter {
    pub fn new(
        ctx: Arc<WgpuContext>,
        in_dim: usize,
        out_dim: usize,
        rank: usize,
        alpha: f32,
    ) -> Self {
        let mk_tensor = |shape: &[usize], val: f32| {
            Tensor::new(
                ctx.clone(),
                UnifiedTensor::new(&ctx, &vec![val; shape.iter().product()], shape),
            )
        };
        Self {
            a: mk_tensor(&[in_dim, rank], 0.01),
            b: mk_tensor(&[rank, out_dim], 0.0),
            alpha,
            rank,
        }
    }

    pub fn params(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
}

#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Tensor,
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

    pub fn new(weight: Tensor) -> Self {
        Self { weight, lora: None }
    }

    pub fn with_lora(weight: Tensor, lora: LoRAAdapter) -> Self {
        Self {
            weight,
            lora: Some(lora),
        }
    }

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

    pub fn params(&self) -> Vec<Tensor> {
        let mut p = vec![self.weight.clone()];
        if let Some(lora) = &self.lora {
            p.extend(lora.params());
        }
        p
    }

    pub fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(format!("{}.weight", prefix), self.weight.clone());
        if let Some(lora) = &self.lora {
            map.insert(format!("{}.lora_A.weight", prefix), lora.a.clone());
            map.insert(format!("{}.lora_B.weight", prefix), lora.b.clone());
        }
        map
    }

    pub fn lora_params(&self) -> Vec<Tensor> {
        if let Some(lora) = &self.lora {
            lora.params()
        } else {
            vec![]
        }
    }
}

pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }

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

    pub fn params(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    pub fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(format!("{}.weight", prefix), self.weight.clone());
        map
    }
}

pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, engine: &ComputeEngine, input: &Tensor) -> Tensor {
        Tensor::rms_norm(input, &self.weight, self.eps, engine)
    }

    pub fn params(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }

    pub fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(format!("{}.weight", prefix), self.weight.clone());
        map
    }
}

pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub n_heads: usize,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    pub dropout_prob: f32,
}

impl Attention {
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

        let q = Tensor::rope(&q, self.head_dim, pos, engine);
        let k = Tensor::rope(&k, self.head_dim, pos, engine);

        // Reshape for GQA [batch, seq, n_heads, head_dim]
        let batch = q.data.shape[0];
        let seq_len = q.data.shape[1];

        let q_4d = q
            .data
            .reshaped(&[batch, seq_len, self.n_heads, self.head_dim]);
        let k_4d = k
            .data
            .reshaped(&[batch, seq_len, self.num_kv_heads, self.head_dim]);
        let v_4d = v
            .data
            .reshaped(&[batch, seq_len, self.num_kv_heads, self.head_dim]);

        // Initialize GQA helper
        let gqa = GroupedQueryAttention::new(
            self.n_heads * self.head_dim,
            self.n_heads,
            self.num_kv_heads,
            true, // causal
            self.dropout_prob,
        );

        // Use cached K,V if available (for incremental decoding)
        if let Some(c) = cache {
            // Copy current K,V to cache at position
            // Important: Use num_kv_heads for offset calculation!
            engine.copy_tensor(
                &x.ctx,
                &k.data,
                &c.key,
                pos * self.num_kv_heads * self.head_dim * 4,
            );
            engine.copy_tensor(
                &x.ctx,
                &v.data,
                &c.value,
                pos * self.num_kv_heads * self.head_dim * 4,
            );

            // Reshape cache for GQA (assuming cache is [B, MaxSeq, H_KV, D])
            // Note: cache.key is likely [B, MaxSeq, H_KV * D]
            let cache_shape = &c.key.shape;
            let c_k_4d = c.key.reshaped(&[
                cache_shape[0],
                cache_shape[1],
                self.num_kv_heads,
                self.head_dim,
            ]);
            let c_v_4d = c.value.reshaped(&[
                cache_shape[0],
                cache_shape[1],
                self.num_kv_heads,
                self.head_dim,
            ]);

            // Compute attention using cached K,V
            // Queries sit at positions pos..pos + seq_len; causal masking
            // hides the cache slots that have not been written yet.
            let attn_out_4d = gqa.forward_at(&q_4d, &c_k_4d, &c_v_4d, mask, pos, &x.ctx, engine);

            // Reshape back to 3D [B, S, H*D]
            let attn_out = attn_out_4d.reshaped(&[batch, seq_len, self.n_heads * self.head_dim]);
            let attn_tensor = Tensor::new(x.ctx.clone(), attn_out);
            self.o_proj.forward(engine, &attn_tensor)
        } else {
            // No cache, use current K,V only
            let attn_out_4d = gqa.forward(&q_4d, &k_4d, &v_4d, mask, &x.ctx, engine);

            // Reshape back to 3D
            let attn_out = attn_out_4d.reshaped(&[batch, seq_len, self.n_heads * self.head_dim]);
            let attn_tensor = Tensor::new(x.ctx.clone(), attn_out);
            self.o_proj.forward(engine, &attn_tensor)
        }
    }

    pub fn params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.q_proj.params());
        p.extend(self.k_proj.params());
        p.extend(self.v_proj.params());
        p.extend(self.o_proj.params());
        p
    }

    pub fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.extend(self.q_proj.named_params(&format!("{}.q_proj", prefix)));
        map.extend(self.k_proj.named_params(&format!("{}.k_proj", prefix)));
        map.extend(self.v_proj.named_params(&format!("{}.v_proj", prefix)));
        map.extend(self.o_proj.named_params(&format!("{}.o_proj", prefix)));
        map
    }

    pub fn lora_params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.q_proj.lora_params());
        p.extend(self.k_proj.lora_params());
        p.extend(self.v_proj.lora_params());
        p.extend(self.o_proj.lora_params());
        p
    }
}

pub struct MLP {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl MLP {
    pub fn forward(&self, engine: &ComputeEngine, input: &Tensor) -> Tensor {
        let up = self.up_proj.forward(engine, input);
        let gate = self.gate_proj.forward(engine, input);
        // silu_gate_fused(a, b) = silu(a) * b; SwiGLU applies SiLU to the gate.
        let fused = Tensor::silu_gate_fused(&gate, &up, engine);
        self.down_proj.forward(engine, &fused)
    }

    pub fn params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.gate_proj.params());
        p.extend(self.up_proj.params());
        p.extend(self.down_proj.params());
        p
    }

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

    pub fn lora_params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.gate_proj.lora_params());
        p.extend(self.up_proj.lora_params());
        p.extend(self.down_proj.lora_params());
        p
    }
}

pub struct LlamaBlock {
    pub norm1: RMSNorm,
    pub attn: Attention,
    pub norm2: RMSNorm,
    pub mlp: MLP,
}

impl LlamaBlock {
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

    pub fn params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.norm1.params());
        p.extend(self.attn.params());
        p.extend(self.norm2.params());
        p.extend(self.mlp.params());
        p
    }

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

    pub fn lora_params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        p.extend(self.attn.lora_params());
        p.extend(self.mlp.lora_params());
        p
    }
}

pub struct LlamaModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<LlamaBlock>,
    pub norm: RMSNorm,
    pub lm_head: Linear,
}

impl LlamaModel {
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

    pub fn lora_params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for layer in &self.layers {
            p.extend(layer.lora_params());
        }
        p.extend(self.lm_head.lora_params());
        p
    }
}

pub struct KVCacheLayer {
    pub key: UnifiedTensor,
    pub value: UnifiedTensor,
}
pub struct KVCache {
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
