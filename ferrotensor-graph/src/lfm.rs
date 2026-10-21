use crate::{Attention, KVCacheLayer, Linear, RMSNorm, Tensor, MLP};
use ferrotensor_backend::{ComputeEngine, UnifiedTensor};
use std::cell::OnceCell;
use std::collections::HashMap;
use std::sync::Arc;

/// LFM2 short-convolution operator: projection, depthwise conv, projection.
pub struct Lfm2Conv {
    /// Input projection.
    pub in_proj: Linear,
    /// Depthwise kernel `[dim, 3]`.
    pub conv_weight: Tensor, // [dim, 3]
    /// Bias `[dim]`.
    pub conv_bias: Tensor, // [dim]
    /// Output projection.
    pub out_proj: Linear,
}

impl Lfm2Conv {
    /// Applies the operator to `x`.
    pub fn forward(&self, engine: &ComputeEngine, x: &Tensor) -> Tensor {
        let projected = self.in_proj.forward(engine, x);

        let out_data = UnifiedTensor::empty(&x.ctx, &projected.data.shape);
        engine.conv1d(
            &x.ctx,
            &projected.data,
            &self.conv_weight.data,
            &self.conv_bias.data,
            &out_data,
        );

        let conv_out = Tensor::new(x.ctx.clone(), out_data);
        self.out_proj.forward(engine, &conv_out)
    }

    /// Trainable tensors (weights, then any LoRA adapters).
    pub fn params(&self) -> Vec<Tensor> {
        let mut p = self.in_proj.params();
        p.push(self.conv_weight.clone());
        p.push(self.conv_bias.clone());
        p.extend(self.out_proj.params());
        p
    }

    /// Only the LoRA adapter tensors (what LoRA fine-tuning updates).
    pub fn lora_params(&self) -> Vec<Tensor> {
        let mut p = self.in_proj.lora_params();
        p.extend(self.out_proj.lora_params());
        p
    }

    /// Parameters keyed by Hugging Face-style names under `prefix`.
    pub fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.extend(self.in_proj.named_params(&format!("{}.in_proj", prefix)));
        map.insert(format!("{}.conv.weight", prefix), self.conv_weight.clone());
        map.insert(format!("{}.conv.bias", prefix), self.conv_bias.clone());
        map.extend(self.out_proj.named_params(&format!("{}.out_proj", prefix)));
        map
    }

    /// LoRA tensors keyed by name under `prefix`.
    pub fn named_lora_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.extend(
            self.in_proj
                .named_params(&format!("{}.in_proj", prefix))
                .into_iter()
                .filter(|(name, _)| name.contains("lora_")),
        );
        map.extend(
            self.out_proj
                .named_params(&format!("{}.out_proj", prefix))
                .into_iter()
                .filter(|(name, _)| name.contains("lora_")),
        );
        map
    }
}

/// The token-mixing operator of an LFM2 block.
#[allow(clippy::large_enum_variant)]
pub enum Lfm2LayerOp {
    /// Short convolution.
    Conv(Lfm2Conv),
    /// Grouped-query attention.
    Attention(Attention),
}

/// LFM2 block: `x + op(norm(x))` then `h + mlp(norm(h))`.
pub struct Lfm2Block {
    /// Norm before the operator.
    pub operator_norm: RMSNorm,
    /// Convolution or attention.
    pub op: Lfm2LayerOp,
    /// Norm before the MLP.
    pub ffn_norm: RMSNorm,
    /// Feed-forward.
    pub mlp: MLP,
}

impl Lfm2Block {
    /// Runs the block; `pos`, `cache` and `mask` are used by attention layers.
    pub fn forward(
        &self,
        engine: &ComputeEngine,
        x: &Tensor,
        pos: usize,
        cache: Option<&KVCacheLayer>,
        mask: Option<&UnifiedTensor>,
    ) -> Tensor {
        // Residual 1: x + Op(Norm(x))
        let h = self.operator_norm.forward(engine, x);
        let h = match &self.op {
            Lfm2LayerOp::Conv(conv) => conv.forward(engine, &h),
            Lfm2LayerOp::Attention(attn) => attn.forward(engine, &h, pos, cache, mask),
        };
        let h = Tensor::add(x, &h, engine);

        // Residual 2: h + MLP(Norm(h))
        let m = self.ffn_norm.forward(engine, &h);
        let m = self.mlp.forward(engine, &m);
        Tensor::add(&h, &m, engine)
    }

    /// Trainable tensors (weights, then any LoRA adapters).
    pub fn params(&self) -> Vec<Tensor> {
        let mut p = self.operator_norm.params();
        match &self.op {
            Lfm2LayerOp::Conv(c) => p.extend(c.params()),
            Lfm2LayerOp::Attention(a) => p.extend(a.params()),
        }
        p.extend(self.ffn_norm.params());
        p.extend(self.mlp.params());
        p
    }

    /// Only the LoRA adapter tensors (what LoRA fine-tuning updates).
    pub fn lora_params(&self) -> Vec<Tensor> {
        let mut p = match &self.op {
            Lfm2LayerOp::Conv(c) => c.lora_params(),
            Lfm2LayerOp::Attention(a) => a.lora_params(),
        };
        p.extend(self.mlp.lora_params());
        p
    }

    /// Parameters keyed by Hugging Face-style names under `prefix`.
    pub fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.extend(
            self.operator_norm
                .named_params(&format!("{}.operator_norm", prefix)),
        );
        match &self.op {
            Lfm2LayerOp::Conv(c) => map.extend(c.named_params(&format!("{}.conv", prefix))),
            Lfm2LayerOp::Attention(a) => {
                map.extend(a.named_params(&format!("{}.self_attn", prefix)))
            }
        }
        map.extend(self.ffn_norm.named_params(&format!("{}.ffn_norm", prefix)));
        map.extend(self.mlp.named_params(&format!("{}.feed_forward", prefix)));
        map
    }

    /// LoRA tensors keyed by name under `prefix`.
    pub fn named_lora_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = match &self.op {
            Lfm2LayerOp::Conv(c) => c.named_lora_params(&format!("{}.conv", prefix)),
            Lfm2LayerOp::Attention(a) => a
                .named_params(&format!("{}.self_attn", prefix))
                .into_iter()
                .filter(|(name, _)| name.contains("lora_"))
                .collect(),
        };
        map.extend(
            self.mlp
                .named_params(&format!("{}.feed_forward", prefix))
                .into_iter()
                .filter(|(name, _)| name.contains("lora_")),
        );
        map
    }
}

/// A decoder-only language model: LFM2 (mixed conv/attention blocks) or
/// Llama-style (every block is attention), as built by
/// [`Lfm2ModelBuilder`](crate::model_builder::Lfm2ModelBuilder).
pub struct Lfm2Model {
    /// Embedding table `[vocab, hidden]`.
    pub embed_tokens: Tensor, // Embedding weight
    /// Optional norm applied right after the embedding (neither Llama nor
    /// Hugging Face LFM2 use one; LFM2's `embedding_norm` is its final norm).
    pub embedding_norm: Option<RMSNorm>,
    /// Blocks.
    pub layers: Vec<Lfm2Block>,
    /// Final norm.
    pub norm_f: RMSNorm,
    /// Projection to vocabulary logits.
    pub lm_head: Linear,
    /// Host copy of `embed_tokens`, read once on first use (the embedding
    /// lookup runs on the CPU).
    embed_host: OnceCell<Vec<f32>>,
}

impl Lfm2Model {
    /// Assemble a model from its parts.
    pub fn new(
        embed_tokens: Tensor,
        embedding_norm: Option<RMSNorm>,
        layers: Vec<Lfm2Block>,
        norm_f: RMSNorm,
        lm_head: Linear,
    ) -> Self {
        Self {
            embed_tokens,
            embedding_norm,
            layers,
            norm_f,
            lm_head,
            embed_host: OnceCell::new(),
        }
    }

    /// Whether every block is attention, so incremental decoding with a
    /// [`KVCache`](crate::KVCache) is exact (conv blocks keep no state).
    pub fn supports_kv_cache(&self) -> bool {
        self.layers
            .iter()
            .all(|l| matches!(l.op, Lfm2LayerOp::Attention(_)))
    }

    /// An empty KV cache for sequences of up to `max_seq` tokens, or `None`
    /// when [`supports_kv_cache`](Self::supports_kv_cache) is false.
    pub fn new_kv_cache(&self, max_seq: usize) -> Option<crate::KVCache> {
        if !self.supports_kv_cache() {
            return None;
        }
        let ctx = &self.embed_tokens.ctx;
        let layers = self
            .layers
            .iter()
            .filter_map(|l| match &l.op {
                Lfm2LayerOp::Attention(a) => {
                    let shape = [1, max_seq, a.num_kv_heads * a.head_dim];
                    Some(crate::KVCacheLayer {
                        key: UnifiedTensor::zeros(ctx, &shape),
                        value: UnifiedTensor::zeros(ctx, &shape),
                    })
                }
                Lfm2LayerOp::Conv(_) => None,
            })
            .collect();
        Some(crate::KVCache { layers })
    }

    /// Logits `[1, seq, vocab]` for `input_ids`, the first of which is at `pos`.
    pub fn forward(
        &self,
        engine: &ComputeEngine,
        input_ids: &[u32],
        pos: usize,
        cache: Option<&mut crate::KVCache>,
        mask: Option<&UnifiedTensor>,
    ) -> Tensor {
        let ctx = self.embed_tokens.ctx.clone();

        // 1. Embedding
        // Note: Generic Embedding.forward usually handles this, manual here for LFM specific
        let mut x = self.lookup_embeddings(&ctx, engine, input_ids);
        if let Some(norm) = &self.embedding_norm {
            x = norm.forward(engine, &x);
        }

        // 2. Layers
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = cache.as_ref().map(|c| &c.layers[i]);
            x = layer.forward(engine, &x, pos, layer_cache, mask);
        }

        // 3. Final Norm & Head
        x = self.norm_f.forward(engine, &x);
        self.lm_head.forward(engine, &x)
    }

    fn lookup_embeddings(
        &self,
        ctx: &Arc<ferrotensor_backend::WgpuContext>,
        _engine: &ComputeEngine,
        ids: &[u32],
    ) -> Tensor {
        let hidden_dim = self.embed_tokens.data.shape[1];
        let seq_len = ids.len();
        let mut data = Vec::with_capacity(seq_len * hidden_dim);

        // Host-side lookup against a cached copy of the table (there is no
        // GPU gather kernel yet). Out-of-range ids map to zeros.
        let w = self
            .embed_host
            .get_or_init(|| pollster::block_on(self.embed_tokens.data.to_vec(ctx)));
        for &id in ids {
            let start = (id as usize) * hidden_dim;
            match w.get(start..start + hidden_dim) {
                Some(row) => data.extend_from_slice(row),
                None => data.extend(std::iter::repeat(0.0).take(hidden_dim)),
            }
        }
        // The lookup is not differentiable, so its output is a constant.
        Tensor::frozen(
            ctx.clone(),
            UnifiedTensor::new(ctx, &data, &[1, seq_len, hidden_dim]),
        )
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

    /// All trainable tensors.
    pub fn params(&self) -> Vec<Tensor> {
        let mut p = vec![self.embed_tokens.clone()];
        if let Some(norm) = &self.embedding_norm {
            p.extend(norm.params());
        }
        for layer in &self.layers {
            p.extend(layer.params());
        }
        p.extend(self.norm_f.params());
        p.extend(self.lm_head.params());
        p
    }

    /// All parameters keyed by Hugging Face-style names.
    pub fn named_params(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert(
            "model.embed_tokens.weight".to_string(),
            self.embed_tokens.clone(),
        );
        if let Some(norm) = &self.embedding_norm {
            map.extend(norm.named_params("model.embedding_norm"));
        }
        for (idx, layer) in self.layers.iter().enumerate() {
            map.extend(layer.named_params(&format!("model.layers.{}", idx)));
        }
        map.extend(self.norm_f.named_params("model.norm"));
        map.extend(self.lm_head.named_params("lm_head"));
        map
    }

    /// LoRA tensors keyed by name.
    pub fn named_lora_params(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        for (idx, layer) in self.layers.iter().enumerate() {
            map.extend(layer.named_lora_params(&format!("model.layers.{}", idx)));
        }
        map.extend(
            self.lm_head
                .named_params("lm_head")
                .into_iter()
                .filter(|(name, _)| name.contains("lora_")),
        );
        map
    }
}
