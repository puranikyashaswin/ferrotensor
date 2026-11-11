//! LoRA fine-tuning end to end on a tiny model, with no download.
//!
//! Writes a tiny random Llama-style model (64-token vocabulary, 2 layers) to
//! a temporary directory, attaches rank-4 LoRA adapters to the attention
//! projections, trains them on one fixed 16-token pattern, saves the
//! adapter, loads it into a fresh copy of the model and checks that the
//! copy gives the same loss. The program exits non-zero if the loss does not
//! fall by at least half, or if the reloaded model scores differently.
//!
//! Run with: `cargo run --release -p ferrotensor-trainer --example lora_finetune_tiny`
//!
//! Output from the CI job "GPU tests (software Vulkan)" (Mesa lavapipe):
//!
//! ```text
//! (filled in from the CI log)
//! ```

#[path = "common/mod.rs"]
mod common;

use common::*;
use ferrotensor_autograd::AdamW;
use ferrotensor_trainer::checkpoint::{
    apply_checkpoint, load_lora_checkpoint, save_lora_checkpoint,
};

const STEPS: usize = 40;
const LR: f32 = 1e-2;

fn main() {
    let dir = TempDir::new("lora-finetune");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let seq = pattern();
    println!("task: continue the pattern {seq:?}");

    let model = build_model(&gpu, dir.path());
    let params = sorted_lora_params(&model);
    let trainable: usize = params
        .iter()
        .map(|p| p.data.shape.iter().product::<usize>())
        .sum();
    println!(
        "model: {LAYERS} layers, vocabulary {VOCAB}; LoRA rank {} on q/k/v/o, {trainable} trainable values",
        lora_config().rank
    );

    let before = eval_loss(&gpu, &model, &seq);
    println!("loss before training: {before:.4}");

    let mut trainer = trainer(&gpu, model, AdamW::new(gpu.engine.clone(), LR));
    let (input, target) = (&seq[..SEQ - 1], &seq[1..]);
    for step in 0..STEPS {
        let stats = trainer
            .train_step(input, target, &params)
            .expect("train step");
        assert!(
            stats.loss.is_finite(),
            "loss is {} at step {step}",
            stats.loss
        );
        if step % 10 == 0 || step == STEPS - 1 {
            println!("step {step:2}  loss {:.4}", stats.loss);
        }
    }
    let after = eval_loss(&gpu, &trainer.model, &seq);
    println!("loss after {STEPS} steps: {after:.4}");
    assert!(
        after < 0.5 * before,
        "the loss should at least halve: {before} -> {after}"
    );

    // Save the trained adapters (the base weights are not written).
    let adapter = dir.path().join("adapter.safetensors");
    let named = trainer.model.named_lora_params();
    pollster::block_on(save_lora_checkpoint(
        &gpu.ctx,
        &adapter,
        &named,
        Some(&lora_metadata()),
    ))
    .expect("save adapter");
    let size = std::fs::metadata(&adapter).map(|m| m.len()).unwrap_or(0);
    println!("saved {} adapter tensors ({size} bytes)", named.len());

    // A fresh copy of the model scores like the untrained one ...
    let fresh = build_model(&gpu, dir.path());
    let fresh_loss = eval_loss(&gpu, &fresh, &seq);
    println!("fresh copy, no adapter: loss {fresh_loss:.4}");

    // ... and like the trained one once the adapter is loaded.
    let checkpoint = load_lora_checkpoint(&adapter).expect("load adapter");
    assert_eq!(checkpoint.metadata, Some(lora_metadata()));
    let restored =
        apply_checkpoint(&gpu.ctx, &checkpoint, &fresh.named_lora_params()).expect("apply");
    let reloaded = eval_loss(&gpu, &fresh, &seq);
    println!("fresh copy, adapter loaded ({restored} tensors): loss {reloaded:.4}");
    let diff = (reloaded - after).abs();
    println!("|reloaded - trained| = {diff:.2e}");
    assert!(
        diff <= 1e-4,
        "the reloaded model should give the trained loss: {after} vs {reloaded}"
    );
}
