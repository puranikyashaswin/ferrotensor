use crate::model_builder::Architecture;
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

    /// Parameters keyed by LFM2 names under `prefix` (`operator_norm`,
    /// `ffn_norm`, `feed_forward`); see [`named_params_as`](Self::named_params_as).
    pub fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        self.named_params_as(prefix, Architecture::Lfm2)
    }

    /// Parameters keyed by the names of `arch` under `prefix`. For
    /// [`Architecture::Llama`] these are the Hugging Face Llama names
    /// (`input_layernorm`, `post_attention_layernorm`, `mlp.*`).
    pub fn named_params_as(&self, prefix: &str, arch: Architecture) -> HashMap<String, Tensor> {
        let [op_norm, mlp_norm, mlp] = block_names(arch);
        let mut map = HashMap::new();
        map.extend(
            self.operator_norm
                .named_params(&format!("{}.{op_norm}", prefix)),
        );
        match &self.op {
            Lfm2LayerOp::Conv(c) => map.extend(c.named_params(&format!("{}.conv", prefix))),
            Lfm2LayerOp::Attention(a) => {
                map.extend(a.named_params(&format!("{}.self_attn", prefix)))
            }
        }
        map.extend(
            self.ffn_norm
                .named_params(&format!("{}.{mlp_norm}", prefix)),
        );
        map.extend(self.mlp.named_params(&format!("{}.{mlp}", prefix)));
        map
    }

    /// LoRA tensors keyed by LFM2 names under `prefix`; see
    /// [`named_lora_params_as`](Self::named_lora_params_as).
    pub fn named_lora_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        self.named_lora_params_as(prefix, Architecture::Lfm2)
    }

    /// LoRA tensors keyed by the names of `arch` under `prefix`.
    pub fn named_lora_params_as(
        &self,
        prefix: &str,
        arch: Architecture,
    ) -> HashMap<String, Tensor> {
        let [_, _, mlp] = block_names(arch);
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
                .named_params(&format!("{}.{mlp}", prefix))
                .into_iter()
                .filter(|(name, _)| name.contains("lora_")),
        );
        map
    }
}

/// Block-level module names of `arch`: operator norm, MLP norm, MLP.
fn block_names(arch: Architecture) -> [&'static str; 3] {
    match arch {
        Architecture::Llama => ["input_layernorm", "post_attention_layernorm", "mlp"],
        Architecture::Lfm2 => ["operator_norm", "ffn_norm", "feed_forward"],
    }
}

/// Pairs of (Hugging Face Llama name, name earlier releases used) for the
/// block modules whose names changed for Llama-style models.
const LLAMA_RENAMES: [(&str, &str); 3] = [
    (".input_layernorm.", ".operator_norm."),
    (".post_attention_layernorm.", ".ffn_norm."),
    (".mlp.", ".feed_forward."),
];

/// The name a Llama-style parameter had before Ferrotensor used Hugging
/// Face names for Llama blocks, or `None` when the name did not change
/// (attention projections, `lm_head`, embeddings, the final norm, and every
/// LFM2 name).
///
/// `model.layers.0.mlp.gate_proj.lora_A.weight` was
/// `model.layers.0.feed_forward.gate_proj.lora_A.weight`.
pub fn legacy_llama_name(name: &str) -> Option<String> {
    if !name.starts_with("model.layers.") {
        return None;
    }
    LLAMA_RENAMES
        .iter()
        .find(|(new, _)| name.contains(new))
        .map(|(new, old)| name.replacen(new, old, 1))
}

/// The inverse of [`legacy_llama_name`]: the Hugging Face Llama name of a
/// parameter named the old way, or `None` when `name` is not an old
/// Llama-block name.
pub fn llama_name_from_legacy(name: &str) -> Option<String> {
    if !name.starts_with("model.layers.") {
        return None;
    }
    LLAMA_RENAMES
        .iter()
        .find(|(_, old)| name.contains(old))
        .map(|(new, old)| name.replacen(old, new, 1))
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
    /// Naming family of [`named_params`](Self::named_params).
    architecture: Architecture,
}

impl Lfm2Model {
    /// Assemble a model from its parts. Parameters get LFM2 names until
    /// [`with_architecture`](Self::with_architecture) says otherwise.
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
            architecture: Architecture::Lfm2,
        }
    }

    /// Set the family whose parameter names [`named_params`](Self::named_params)
    /// and [`named_lora_params`](Self::named_lora_params) use: Hugging Face
    /// Llama names for [`Architecture::Llama`], LFM2 names otherwise.
    /// [`Lfm2ModelBuilder`](crate::model_builder::Lfm2ModelBuilder) sets it
    /// from the checkpoint's config.
    pub fn with_architecture(mut self, architecture: Architecture) -> Self {
        self.architecture = architecture;
        self
    }

    /// The naming family set by [`with_architecture`](Self::with_architecture).
    pub fn architecture(&self) -> Architecture {
        self.architecture
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
        let mut x = self.lookup_embeddings(&ctx, &[input_ids]);
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

    /// Whether [`forward_batch`](Self::forward_batch) can run: every block is
    /// attention (conv blocks would mix the sequences of a batch).
    pub fn supports_batching(&self) -> bool {
        self.supports_kv_cache()
    }

    /// Logits `[batch, seq, vocab]` for equally long `rows` of token ids,
    /// each starting at position 0, in one pass. Pad shorter sequences on
    /// the right: attention is causal, so real tokens never see the padding.
    ///
    /// # Panics
    /// If the rows differ in length, or if
    /// [`supports_batching`](Self::supports_batching) is false.
    pub fn forward_batch(&self, engine: &ComputeEngine, rows: &[&[u32]]) -> Tensor {
        assert!(
            self.supports_batching(),
            "forward_batch needs an all-attention model"
        );
        let seq_len = rows.first().map_or(0, |r| r.len());
        assert!(
            rows.iter().all(|r| r.len() == seq_len),
            "forward_batch: rows must have the same length"
        );
        let ctx = self.embed_tokens.ctx.clone();
        let mut x = self.lookup_embeddings(&ctx, rows);
        if let Some(norm) = &self.embedding_norm {
            x = norm.forward(engine, &x);
        }
        for layer in &self.layers {
            x = layer.forward(engine, &x, 0, None, None);
        }
        x = self.norm_f.forward(engine, &x);
        self.lm_head.forward(engine, &x)
    }

    /// Hidden states `[batch, seq, hidden]` after the blocks listed in
    /// `taps`, in that order, for equally long `rows` of token ids starting
    /// at position 0. Tap `l` is the output of block `l` (1-based), before
    /// the final norm; tap 0 is the embedding output. Blocks after the
    /// deepest tap are not run, so tapping only early blocks also truncates
    /// the model.
    ///
    /// Pad shorter sequences on the right, as for
    /// [`forward_batch`](Self::forward_batch).
    ///
    /// # Panics
    /// If the rows differ in length, a tap exceeds the number of blocks, or
    /// [`supports_batching`](Self::supports_batching) is false.
    pub fn hidden_states_batch(
        &self,
        engine: &ComputeEngine,
        rows: &[&[u32]],
        taps: &[usize],
    ) -> Vec<Tensor> {
        assert!(
            self.supports_batching(),
            "hidden_states_batch needs an all-attention model"
        );
        let seq_len = rows.first().map_or(0, |r| r.len());
        assert!(
            rows.iter().all(|r| r.len() == seq_len),
            "hidden_states_batch: rows must have the same length"
        );
        let deepest = taps.iter().copied().max().unwrap_or(0);
        assert!(
            deepest <= self.layers.len(),
            "hidden_states_batch: tap {deepest} exceeds {} blocks",
            self.layers.len()
        );
        let ctx = self.embed_tokens.ctx.clone();
        let mut x = self.lookup_embeddings(&ctx, rows);
        if let Some(norm) = &self.embedding_norm {
            x = norm.forward(engine, &x);
        }
        let mut states = Vec::with_capacity(deepest + 1);
        states.push(x.clone());
        for layer in &self.layers[..deepest] {
            x = layer.forward(engine, &x, 0, None, None);
            states.push(x.clone());
        }
        taps.iter().map(|&tap| states[tap].clone()).collect()
    }

    /// Embeddings `[rows, seq, hidden]` for equally long rows of ids.
    fn lookup_embeddings(
        &self,
        ctx: &Arc<ferrotensor_backend::WgpuContext>,
        rows: &[&[u32]],
    ) -> Tensor {
        let hidden_dim = self.embed_tokens.data.shape[1];
        let seq_len = rows.first().map_or(0, |r| r.len());
        let mut data = Vec::with_capacity(rows.len() * seq_len * hidden_dim);

        // Host-side lookup against a cached copy of the table (there is no
        // GPU gather kernel yet). Out-of-range ids map to zeros.
        let w = self
            .embed_host
            .get_or_init(|| pollster::block_on(self.embed_tokens.data.to_vec(ctx)));
        for &id in rows.iter().flat_map(|r| r.iter()) {
            let start = (id as usize) * hidden_dim;
            match w.get(start..start + hidden_dim) {
                Some(row) => data.extend_from_slice(row),
                None => data.extend(std::iter::repeat_n(0.0, hidden_dim)),
            }
        }
        // The lookup is not differentiable, so its output is a constant.
        Tensor::frozen(
            ctx.clone(),
            UnifiedTensor::new(ctx, &data, &[rows.len(), seq_len, hidden_dim]),
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
            map.extend(layer.named_params_as(&format!("model.layers.{}", idx), self.architecture));
        }
        map.extend(self.norm_f.named_params("model.norm"));
        map.extend(self.lm_head.named_params("lm_head"));
        map
    }

    /// LoRA tensors keyed by name.
    pub fn named_lora_params(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        for (idx, layer) in self.layers.iter().enumerate() {
            map.extend(
                layer.named_lora_params_as(&format!("model.layers.{}", idx), self.architecture),
            );
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

#[cfg(test)]
mod tests {
    use super::{legacy_llama_name, llama_name_from_legacy};

    #[test]
    fn llama_block_names_map_to_and_from_the_old_names() {
        let pairs = [
            (
                "model.layers.3.input_layernorm.weight",
                "model.layers.3.operator_norm.weight",
            ),
            (
                "model.layers.3.post_attention_layernorm.weight",
                "model.layers.3.ffn_norm.weight",
            ),
            (
                "model.layers.0.mlp.gate_proj.lora_A.weight",
                "model.layers.0.feed_forward.gate_proj.lora_A.weight",
            ),
            (
                "model.layers.11.mlp.down_proj.weight",
                "model.layers.11.feed_forward.down_proj.weight",
            ),
        ];
        for (new, old) in pairs {
            assert_eq!(legacy_llama_name(new).as_deref(), Some(old));
            assert_eq!(llama_name_from_legacy(old).as_deref(), Some(new));
        }
    }

    #[test]
    fn unchanged_names_have_no_legacy_form() {
        for name in [
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "lm_head.lora_B.weight",
            "model.norm.weight",
            "model.embed_tokens.weight",
        ] {
            assert_eq!(legacy_llama_name(name), None, "{name}");
            assert_eq!(llama_name_from_legacy(name), None, "{name}");
        }
    }
}
