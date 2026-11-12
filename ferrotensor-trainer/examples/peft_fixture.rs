//! Fixture for `scripts/check_peft_export.py`: a tiny random Llama-style
//! model, a LoRA adapter trained on it with Ferrotensor, and the logits the
//! adapted model gives for fixed token ids.
//!
//! ```text
//! cargo run --release -p ferrotensor-trainer --example peft_fixture -- OUT
//! ```
//!
//! writes `OUT/model/` (config.json + model.safetensors),
//! `OUT/adapter.safetensors` (Ferrotensor format) and `OUT/logits.json`.
//! Needs a GPU (software Vulkan works).

#[path = "../tests/common/mod.rs"]
mod common;

use common::*;
use ferrotensor_graph::model_builder::LoRaConfig;
use ferrotensor_trainer::checkpoint::{save_lora_checkpoint, LoraMetadata};
use ferrotensor_trainer::peft::PEFT_TARGET_MODULES;
use std::path::PathBuf;

const STEPS: usize = 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: peft_fixture OUT_DIR")?,
    );
    let model_dir = out.join("model");
    std::fs::create_dir_all(&model_dir)?;
    write_tiny_llama(&model_dir);

    let gpu = gpu();
    let meta = LoraMetadata {
        rank: 4,
        alpha: 8.0,
        target_modules: PEFT_TARGET_MODULES.iter().map(|t| t.to_string()).collect(),
        trained_modules: None,
    };
    let config = ferrotensor_loader::ModelLoader::load_config(model_dir.join("config.json"))?;
    let mut weights = ferrotensor_loader::WeightsContext::new();
    weights.load_model_dir(&model_dir)?;
    let model = ferrotensor_graph::model_builder::Lfm2ModelBuilder::new(
        gpu.ctx.clone(),
        gpu.engine.clone(),
        &weights,
        config,
    )
    .with_lora(LoRaConfig {
        rank: meta.rank,
        alpha: meta.alpha,
        target_modules: meta.target_modules.clone(),
    })
    .build()?;

    // Train until every lora_B is clearly non-zero.
    let (_, params) = sorted_lora_params(&model);
    let mut t = trainer(&gpu, model, 1e-2);
    let seq = pattern_sequence();
    let mut last = 0.0;
    for _ in 0..STEPS {
        last = t
            .train_step(&seq[..seq.len() - 1], &seq[1..], &params)?
            .loss;
    }
    let mut min_b = f32::INFINITY;
    for (name, tensor) in t.model.named_lora_params() {
        if name.contains("lora_B") {
            let max = host(&gpu, &tensor)
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()));
            min_b = min_b.min(max);
        }
    }
    if min_b <= 1e-4 {
        return Err(format!("a lora_B is still near zero after {STEPS} steps ({min_b})").into());
    }

    let adapter = out.join("adapter.safetensors");
    pollster::block_on(save_lora_checkpoint(
        &gpu.ctx,
        &adapter,
        &t.model.named_lora_params(),
        Some(&meta),
    ))?;

    // Token ids the adapter was not trained on.
    let ids: Vec<u32> = (0..16).map(|i| ((i * 5 + 2) % VOCAB) as u32).collect();
    let logits = host(&gpu, &t.model.forward(&gpu.engine, &ids, 0, None, None));
    let json = serde_json::json!({
        "input_ids": ids,
        "vocab_size": VOCAB,
        "logits": logits,
    });
    std::fs::write(out.join("logits.json"), json.to_string())?;
    println!(
        "wrote {}: {STEPS} LoRA steps (final loss {last:.4}, smallest max |lora_B| {min_b:.3e}), {} logits",
        out.display(),
        logits.len()
    );
    Ok(())
}
