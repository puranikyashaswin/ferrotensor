//! Shared fixtures: a tiny randomly initialised Llama-style checkpoint
//! written to a temporary directory, and helpers to train it.

#![allow(dead_code)]

use ferrotensor_autograd::{AdamW, Tensor};
use ferrotensor_backend::{ComputeEngine, WgpuContext};
use ferrotensor_graph::lfm::Lfm2Model;
use ferrotensor_graph::model_builder::{Lfm2ModelBuilder, LoRaConfig};
use ferrotensor_loader::{ModelLoader, WeightsContext};
use ferrotensor_trainer::Trainer;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const VOCAB: usize = 64;
pub const HIDDEN: usize = 32;
pub const HEADS: usize = 4;
pub const KV_HEADS: usize = 2;
pub const LAYERS: usize = 2;
pub const FFN: usize = 64;

/// A fresh directory under the system temp dir, removed on drop.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!(
            "ferrotensor-trainer-{tag}-{}-{nanos}",
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

/// Write `config.json` and `model.safetensors` of a tiny Llama-style model
/// (tied embeddings, grouped-query attention) into `dir`.
pub fn write_tiny_llama(dir: &Path) {
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "intermediate_size": FFN,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "num_hidden_layers": LAYERS,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).expect("write config");

    let kv = HIDDEN / HEADS * KV_HEADS;
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut seed = 1;
    // scale 0 means "all ones" (norm weights).
    let mut add = |name: &str, shape: Vec<usize>, scale: f32| {
        let n = shape.iter().product();
        seed += 1;
        let values = if scale == 0.0 {
            vec![1.0; n]
        } else {
            pseudo_random(n, seed, scale)
        };
        tensors.push((name.to_string(), shape, values));
    };
    add("model.embed_tokens.weight", vec![VOCAB, HIDDEN], 0.5);
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        add(
            &format!("{p}.self_attn.q_proj.weight"),
            vec![HIDDEN, HIDDEN],
            0.2,
        );
        add(
            &format!("{p}.self_attn.k_proj.weight"),
            vec![kv, HIDDEN],
            0.2,
        );
        add(
            &format!("{p}.self_attn.v_proj.weight"),
            vec![kv, HIDDEN],
            0.2,
        );
        add(
            &format!("{p}.self_attn.o_proj.weight"),
            vec![HIDDEN, HIDDEN],
            0.2,
        );
        add(&format!("{p}.mlp.gate_proj.weight"), vec![FFN, HIDDEN], 0.2);
        add(&format!("{p}.mlp.up_proj.weight"), vec![FFN, HIDDEN], 0.2);
        add(&format!("{p}.mlp.down_proj.weight"), vec![HIDDEN, FFN], 0.2);
        add(&format!("{p}.input_layernorm.weight"), vec![HIDDEN], 0.0);
        add(
            &format!("{p}.post_attention_layernorm.weight"),
            vec![HIDDEN],
            0.0,
        );
    }
    add("model.norm.weight", vec![HIDDEN], 0.0);

    let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
        .into_iter()
        .map(|(n, s, v)| (n, s, v.iter().flat_map(|x| x.to_le_bytes()).collect()))
        .collect();
    let views: HashMap<String, safetensors::tensor::TensorView<'_>> = bytes
        .iter()
        .map(|(n, s, b)| {
            (
                n.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.clone(), b)
                    .expect("tensor view"),
            )
        })
        .collect();
    let data = safetensors::tensor::serialize(views, &None).expect("serialize");
    std::fs::write(dir.join("model.safetensors"), data).expect("write weights");
}

/// Per-head dimension of the tiny Qwen3-style model; `HEADS * QWEN3_HEAD_DIM`
/// differs from `HIDDEN`, as in the released Qwen3 models.
pub const QWEN3_HEAD_DIM: usize = 16;

/// Write a tiny Qwen3-style model into `dir`: the shapes of
/// [`write_tiny_llama`] except an explicit `head_dim`, plus per-head
/// `q_norm`/`k_norm` weights, non-unit norm weights, an untied `lm_head`, and
/// the `layer_types` list recent `transformers` versions write into Qwen3
/// configs.
pub fn write_tiny_qwen3(dir: &Path) {
    let layer_types = vec!["full_attention"; LAYERS];
    let config = serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "intermediate_size": FFN,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": QWEN3_HEAD_DIM,
        "num_hidden_layers": LAYERS,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_bias": false,
        "tie_word_embeddings": false,
        "layer_types": layer_types,
    });
    std::fs::write(dir.join("config.json"), config.to_string()).expect("write config");

    let q = HEADS * QWEN3_HEAD_DIM;
    let kv = KV_HEADS * QWEN3_HEAD_DIM;
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut seed = 1000;
    // Norm weights are 1 + noise so that a skipped or misplaced norm changes
    // the logits.
    let mut add = |name: String, shape: Vec<usize>, scale: f32, norm: bool| {
        let n = shape.iter().product();
        seed += 1;
        let mut values = pseudo_random(n, seed, scale);
        if norm {
            values.iter_mut().for_each(|v| *v += 1.0);
        }
        tensors.push((name, shape, values));
    };
    add(
        "model.embed_tokens.weight".into(),
        vec![VOCAB, HIDDEN],
        0.5,
        false,
    );
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        add(
            format!("{p}.self_attn.q_proj.weight"),
            vec![q, HIDDEN],
            0.3,
            false,
        );
        add(
            format!("{p}.self_attn.k_proj.weight"),
            vec![kv, HIDDEN],
            0.3,
            false,
        );
        add(
            format!("{p}.self_attn.v_proj.weight"),
            vec![kv, HIDDEN],
            0.2,
            false,
        );
        add(
            format!("{p}.self_attn.o_proj.weight"),
            vec![HIDDEN, q],
            0.2,
            false,
        );
        add(
            format!("{p}.self_attn.q_norm.weight"),
            vec![QWEN3_HEAD_DIM],
            0.5,
            true,
        );
        add(
            format!("{p}.self_attn.k_norm.weight"),
            vec![QWEN3_HEAD_DIM],
            0.5,
            true,
        );
        add(
            format!("{p}.mlp.gate_proj.weight"),
            vec![FFN, HIDDEN],
            0.2,
            false,
        );
        add(
            format!("{p}.mlp.up_proj.weight"),
            vec![FFN, HIDDEN],
            0.2,
            false,
        );
        add(
            format!("{p}.mlp.down_proj.weight"),
            vec![HIDDEN, FFN],
            0.2,
            false,
        );
        add(
            format!("{p}.input_layernorm.weight"),
            vec![HIDDEN],
            0.3,
            true,
        );
        add(
            format!("{p}.post_attention_layernorm.weight"),
            vec![HIDDEN],
            0.3,
            true,
        );
    }
    add("model.norm.weight".into(), vec![HIDDEN], 0.3, true);
    add("lm_head.weight".into(), vec![VOCAB, HIDDEN], 0.3, false);
    write_f32_safetensors(&dir.join("model.safetensors"), tensors);
}

/// Serialize `(name, shape, values)` f32 tensors to a safetensors file.
pub fn write_f32_safetensors(path: &Path, tensors: Vec<(String, Vec<usize>, Vec<f32>)>) {
    let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
        .into_iter()
        .map(|(n, s, v)| (n, s, v.iter().flat_map(|x| x.to_le_bytes()).collect()))
        .collect();
    let views: HashMap<String, safetensors::tensor::TensorView<'_>> = bytes
        .iter()
        .map(|(n, s, b)| {
            (
                n.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, s.clone(), b)
                    .expect("tensor view"),
            )
        })
        .collect();
    let data = safetensors::tensor::serialize(views, &None).expect("serialize");
    std::fs::write(path, data).expect("write weights");
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

pub fn lora_config() -> LoRaConfig {
    LoRaConfig {
        rank: 4,
        alpha: 8.0,
        target_modules: vec![
            "q_proj".into(),
            "k_proj".into(),
            "v_proj".into(),
            "o_proj".into(),
        ],
    }
}

/// Build the tiny model in `dir` with LoRA adapters.
pub fn build_model(gpu: &Gpu, dir: &Path) -> Lfm2Model {
    build_model_with(gpu, dir, false)
}

/// [`build_model`], optionally with frozen weights stored as f16.
pub fn build_model_with(gpu: &Gpu, dir: &Path, f16_weights: bool) -> Lfm2Model {
    let config = ModelLoader::load_config(dir.join("config.json")).expect("config");
    let mut weights = WeightsContext::new();
    weights.load_model_dir(dir).expect("weights");
    Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &weights, config)
        .with_lora(lora_config())
        .with_f16_weights(f16_weights)
        .build()
        .expect("build")
}

/// LoRA parameters in a stable (name-sorted) order.
pub fn sorted_lora_params(model: &Lfm2Model) -> (Vec<String>, Vec<Tensor>) {
    let named = model.named_lora_params();
    let mut names: Vec<String> = named.keys().cloned().collect();
    names.sort();
    let params = names.iter().map(|n| named[n].clone()).collect();
    (names, params)
}

pub fn trainer(gpu: &Gpu, model: Lfm2Model, lr: f32) -> Trainer {
    Trainer::new(
        model,
        Box::new(AdamW::new(gpu.engine.clone(), lr)),
        gpu.ctx.clone(),
        gpu.engine.clone(),
    )
}

/// A short "sentence" over the tiny vocabulary: the model has to learn a
/// fixed cyclic pattern.
pub fn pattern_sequence() -> Vec<u32> {
    (0..24).map(|i| ((i * 7 + 3) % VOCAB) as u32).collect()
}

pub fn host(gpu: &Gpu, t: &Tensor) -> Vec<f32> {
    pollster::block_on(t.data.to_vec(&gpu.ctx))
}
