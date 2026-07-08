use crate::{Attention, KVCacheLayer, Linear, RMSNorm, Tensor, MLP};
use rusty_backend::{ComputeEngine, UnifiedTensor};
use std::collections::HashMap;
use std::sync::Arc;

pub struct Lfm2Conv {
    pub in_proj: Linear,
    pub conv_weight: Tensor, // [dim, 3]
    pub conv_bias: Tensor,   // [dim]
    pub out_proj: Linear,
}

impl Lfm2Conv {
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

    pub fn params(&self) -> Vec<Tensor> {
        let mut p = self.in_proj.params();
        p.push(self.conv_weight.clone());
        p.push(self.conv_bias.clone());
        p.extend(self.out_proj.params());
        p
    }

    pub fn lora_params(&self) -> Vec<Tensor> {
        let mut p = self.in_proj.lora_params();
        p.extend(self.out_proj.lora_params());
        p
    }

    pub fn named_params(&self, prefix: &str) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.extend(self.in_proj.named_params(&format!("{}.in_proj", prefix)));
        map.insert(format!("{}.conv.weight", prefix), self.conv_weight.clone());
        map.insert(format!("{}.conv.bias", prefix), self.conv_bias.clone());
        map.extend(self.out_proj.named_params(&format!("{}.out_proj", prefix)));
        map
    }

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

pub enum Lfm2LayerOp {
    Conv(Lfm2Conv),
    Attention(Attention),
}

pub struct Lfm2Block {
    pub operator_norm: RMSNorm,
    pub op: Lfm2LayerOp,
    pub ffn_norm: RMSNorm,
    pub mlp: MLP,
}

impl Lfm2Block {
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

    pub fn lora_params(&self) -> Vec<Tensor> {
        let mut p = match &self.op {
            Lfm2LayerOp::Conv(c) => c.lora_params(),
            Lfm2LayerOp::Attention(a) => a.lora_params(),
        };
        p.extend(self.mlp.lora_params());
        p
    }

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

pub struct Lfm2Model {
    pub embed_tokens: Tensor, // Embedding weight
    pub embedding_norm: RMSNorm,
    pub layers: Vec<Lfm2Block>,
    pub norm_f: RMSNorm,
    pub lm_head: Linear,
}

impl Lfm2Model {
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
        x = self.embedding_norm.forward(engine, &x);

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
        ctx: &Arc<rusty_backend::WgpuContext>,
        _engine: &ComputeEngine,
        ids: &[u32],
    ) -> Tensor {
        let hidden_dim = self.embed_tokens.data.shape[1];
        let seq_len = ids.len();
        let mut data = Vec::with_capacity(seq_len * hidden_dim);

        // Host-side lookup (Mocking for now as we don't have a high-perf GPU embedding kernel yet)
        let w = pollster::block_on(self.embed_tokens.data.to_vec(ctx));
        for &id in ids {
            let start = (id as usize) * hidden_dim;
            data.extend_from_slice(&w[start..start + hidden_dim]);
        }
        Tensor::new(
            ctx.clone(),
            UnifiedTensor::new(ctx, &data, &[1, seq_len, hidden_dim]),
        )
    }

    pub fn lora_params(&self) -> Vec<Tensor> {
        let mut p = Vec::new();
        for layer in &self.layers {
            p.extend(layer.lora_params());
        }
        p.extend(self.lm_head.lora_params());
        p
    }

    pub fn params(&self) -> Vec<Tensor> {
        let mut p = vec![self.embed_tokens.clone()];
        p.extend(self.embedding_norm.params());
        for layer in &self.layers {
            p.extend(layer.params());
        }
        p.extend(self.norm_f.params());
        p.extend(self.lm_head.params());
        p
    }

    pub fn named_params(&self) -> HashMap<String, Tensor> {
        let mut map = HashMap::new();
        map.insert("model.embed_tokens.weight".to_string(), self.embed_tokens.clone());
        map.extend(self.embedding_norm.named_params("model.embedding_norm"));
        for (idx, layer) in self.layers.iter().enumerate() {
            map.extend(layer.named_params(&format!("model.layers.{}", idx)));
        }
        map.extend(self.norm_f.named_params("model.norm"));
        map.extend(self.lm_head.named_params("lm_head"));
        map
    }

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
