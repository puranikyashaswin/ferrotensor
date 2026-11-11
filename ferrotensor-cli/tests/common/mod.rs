#![allow(dead_code)]
//! Shared fixtures: a tiny randomly initialised Llama-style model directory
//! (config, weights, tokenizer) and a token dataset for the CLI tests.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

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
            "ferrotensor-cli-{tag}-{}-{nanos}",
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

/// Write a word-level `tokenizer.json` with `VOCAB` entries into `dir`:
/// ids 0..=3 are `<pad>`, `<s>`, `</s>`, `<unk>`; id `n >= 4` is the word `w{n}`.
pub fn write_tiny_tokenizer(dir: &Path) {
    let mut vocab = serde_json::Map::new();
    for (id, tok) in ["<pad>", "<s>", "</s>", "<unk>"].iter().enumerate() {
        vocab.insert((*tok).to_string(), serde_json::json!(id));
    }
    for id in 4..VOCAB {
        vocab.insert(format!("w{id}"), serde_json::json!(id));
    }
    let tokenizer = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<unk>" }
    });
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).expect("write tokenizer");
}

/// Write `count` token ids (cycling through 4..VOCAB) as little-endian u32
/// to `path`: a `--task pretrain` dataset.
pub fn write_token_bin(path: &Path, count: usize) {
    let bytes: Vec<u8> = (0..count)
        .flat_map(|i| ((4 + i % (VOCAB - 4)) as u32).to_le_bytes())
        .collect();
    std::fs::write(path, bytes).expect("write tokens");
}
