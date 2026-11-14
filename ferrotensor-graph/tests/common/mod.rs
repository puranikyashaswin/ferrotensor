//! Shared fixture: a tiny randomly initialised Llama-style checkpoint
//! (grouped-query attention, untied LM head) written to a temporary
//! directory, plus helpers to load it on the GPU.

#![allow(dead_code)]

use ferrotensor_backend::{ComputeEngine, WgpuContext};
use ferrotensor_graph::lfm::Lfm2Model;
use ferrotensor_graph::model_builder::{Lfm2ModelBuilder, LoRaConfig};
use ferrotensor_loader::{ModelLoader, WeightsContext};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const VOCAB: usize = 32;
pub const HIDDEN: usize = 32;
pub const HEADS: usize = 4;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = HIDDEN / HEADS;
pub const LAYERS: usize = 2;
pub const FFN: usize = 64;
/// Not the 1e-5 default, so a builder that ignores `rms_norm_eps` is caught.
pub const EPS: f32 = 1e-6;
/// Not the 10000 default, so a builder that ignores `rope_theta` is caught.
pub const THETA: f32 = 500.0;

/// Weights in Hugging Face layout (`[out, in]` for linear layers), by name:
/// `(shape, values)`.
pub type Weights = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// A fresh directory under the system temp dir, removed on drop.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "ferrotensor-graph-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        Self(dir)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Deterministic values in `[-scale, scale)`.
pub fn pseudo_random(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let u = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32;
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Random weights of the tiny model in Hugging Face layout. Norm scales are
/// near 1 but not exactly 1, so a norm that drops its weight is caught.
pub fn tiny_llama_weights() -> Weights {
    let kv = KV_HEADS * HEAD_DIM;
    let mut weights = Weights::new();
    let mut seed = 1;
    let mut add = |name: String, shape: Vec<usize>, scale: f32, offset: f32| {
        let n = shape.iter().product();
        seed += 1;
        let values = pseudo_random(n, seed, scale)
            .into_iter()
            .map(|v| v + offset)
            .collect();
        weights.insert(name, (shape, values));
    };
    add(
        "model.embed_tokens.weight".into(),
        vec![VOCAB, HIDDEN],
        0.5,
        0.0,
    );
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        let q = HEADS * HEAD_DIM;
        add(
            format!("{p}.self_attn.q_proj.weight"),
            vec![q, HIDDEN],
            0.2,
            0.0,
        );
        add(
            format!("{p}.self_attn.k_proj.weight"),
            vec![kv, HIDDEN],
            0.2,
            0.0,
        );
        add(
            format!("{p}.self_attn.v_proj.weight"),
            vec![kv, HIDDEN],
            0.2,
            0.0,
        );
        add(
            format!("{p}.self_attn.o_proj.weight"),
            vec![HIDDEN, q],
            0.2,
            0.0,
        );
        add(
            format!("{p}.mlp.gate_proj.weight"),
            vec![FFN, HIDDEN],
            0.2,
            0.0,
        );
        add(
            format!("{p}.mlp.up_proj.weight"),
            vec![FFN, HIDDEN],
            0.2,
            0.0,
        );
        add(
            format!("{p}.mlp.down_proj.weight"),
            vec![HIDDEN, FFN],
            0.2,
            0.0,
        );
        add(
            format!("{p}.input_layernorm.weight"),
            vec![HIDDEN],
            0.25,
            1.0,
        );
        add(
            format!("{p}.post_attention_layernorm.weight"),
            vec![HIDDEN],
            0.25,
            1.0,
        );
    }
    add("model.norm.weight".into(), vec![HIDDEN], 0.25, 1.0);
    add("lm_head.weight".into(), vec![VOCAB, HIDDEN], 0.2, 0.0);
    weights
}

/// Write `config.json` and `model.safetensors` for `weights` into `dir`.
pub fn write_tiny_llama(dir: &Path, weights: &Weights) {
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "intermediate_size": FFN,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "num_hidden_layers": LAYERS,
        "vocab_size": VOCAB,
        "rms_norm_eps": EPS,
        "rope_theta": THETA,
        "tie_word_embeddings": false,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).expect("write config");

    let bytes: Vec<(&String, &Vec<usize>, Vec<u8>)> = weights
        .iter()
        .map(|(n, (s, v))| (n, s, v.iter().flat_map(|x| x.to_le_bytes()).collect()))
        .collect();
    let views: HashMap<String, safetensors::tensor::TensorView<'_>> = bytes
        .iter()
        .map(|(n, s, b)| {
            (
                n.to_string(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.to_vec(), b)
                    .expect("tensor view"),
            )
        })
        .collect();
    let data = safetensors::tensor::serialize(views, &None).expect("serialize");
    std::fs::write(dir.join("model.safetensors"), data).expect("write weights");
}

pub struct Gpu {
    pub ctx: Arc<WgpuContext>,
    pub engine: Arc<ComputeEngine>,
}

pub fn gpu() -> Gpu {
    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));
    let engine = Arc::new(ComputeEngine::new(&ctx));
    Gpu { ctx, engine }
}

/// LoRA on every linear layer of the model, including the LM head.
pub fn lora_everywhere() -> LoRaConfig {
    LoRaConfig {
        rank: 4,
        alpha: 8.0,
        target_modules: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
        .map(String::from)
        .to_vec(),
    }
}

/// Load the checkpoint in `dir`, optionally with LoRA adapters and with the
/// frozen linear weights stored as f16.
pub fn build_model(
    gpu: &Gpu,
    dir: &Path,
    lora: Option<LoRaConfig>,
    f16_weights: bool,
) -> Lfm2Model {
    let config = ModelLoader::load_config(dir.join("config.json")).expect("config");
    let mut weights = WeightsContext::new();
    weights.load_model_dir(dir).expect("weights");
    let mut builder = Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &weights, config)
        .with_f16_weights(f16_weights);
    if let Some(lora) = lora {
        builder = builder.with_lora(lora);
    }
    builder.build().expect("build")
}
