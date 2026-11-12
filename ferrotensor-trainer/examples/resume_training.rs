//! Stop a LoRA run halfway, save it, resume it, and land where an
//! uninterrupted run lands.
//!
//! On the same tiny random model as `lora_finetune_tiny` (no download):
//!
//! 1. trains 12 steps without stopping;
//! 2. trains 6 steps, saves the adapter and the AdamW state (moments and
//!    step count), builds a fresh model, loads both and trains 6 more;
//! 3. for contrast, resumes from the same adapter with a new, empty AdamW.
//!
//! The program exits non-zero unless run 2 matches run 1 at every resumed
//! step (relative difference at most 1e-5). Run 3 shows why the optimizer
//! state is saved: without it the losses after the resume differ.
//!
//! Run with: `cargo run --release -p ferrotensor-trainer --example resume_training`
//!
//! Output from the CI job "GPU tests (software Vulkan)" (Mesa lavapipe):
//!
//! ```text
//! device: llvmpipe (LLVM 20.1.2, 256 bits) (Vulkan)
//! 12 steps straight:           5.9385 5.7171 5.4485 5.1535 4.8705 4.6353 4.4604 4.3264 4.2083 4.0905 3.9635 3.8199
//! saved adapter and optimizer state after 6 steps
//! 6 + 6 steps, resumed:        5.9385 5.7171 5.4485 5.1535 4.8705 4.6353 4.4604 4.3264 4.2083 4.0905 3.9635 3.8199
//! 6 + 6, new optimizer:        5.9385 5.7171 5.4485 5.1535 4.8705 4.6353 4.4604 4.2503 4.0582 3.8699 3.6729 3.4597
//! largest relative difference, resumed vs straight: 0.00e0 (allowed 1e-5)
//! final loss without the optimizer state differs by 0.3602
//! ```

#[path = "common/mod.rs"]
mod common;

use common::*;
use ferrotensor_autograd::{AdamW, Optimizer};
use ferrotensor_trainer::checkpoint::{
    apply_checkpoint, load_lora_checkpoint, save_lora_checkpoint, TrainingCheckpoint, TrainingState,
};
use ferrotensor_trainer::Trainer;
use std::path::Path;

const STEPS: usize = 12;
const HALF: usize = STEPS / 2;
const LR: f32 = 5e-3;
const TOLERANCE: f32 = 1e-5;

/// Run `steps` training steps and return the loss of each.
fn train(trainer: &mut Trainer, seq: &[u32], steps: usize) -> Vec<f32> {
    let params = sorted_lora_params(&trainer.model);
    (0..steps)
        .map(|_| {
            trainer
                .train_step(&seq[..SEQ - 1], &seq[1..], &params)
                .expect("train step")
                .loss
        })
        .collect()
}

fn show(name: &str, losses: &[f32]) {
    let s: Vec<String> = losses.iter().map(|l| format!("{l:.4}")).collect();
    println!("{name:<28} {}", s.join(" "));
}

/// A fresh model with the adapter at `adapter` loaded, and a trainer whose
/// optimizer state comes from `state` (or a new AdamW when `None`).
fn resume(gpu: &Gpu, dir: &Path, adapter: &Path, state: Option<&Path>) -> Trainer {
    let model = build_model(gpu, dir);
    let named = model.named_lora_params();
    let checkpoint = load_lora_checkpoint(adapter).expect("load adapter");
    apply_checkpoint(&gpu.ctx, &checkpoint, &named).expect("apply adapter");
    let mut optimizer = AdamW::new(gpu.engine.clone(), LR);
    let mut steps_done = 0;
    if let Some(path) = state {
        let saved = TrainingCheckpoint::load(path).expect("load training state");
        let moments = saved.optimizer_state.as_ref().expect("optimizer state");
        assert!(
            optimizer.load_state(&gpu.ctx, &named, moments),
            "the optimizer state did not match the model"
        );
        steps_done = saved.training_state.step;
    }
    let mut t = trainer(gpu, model, optimizer);
    t.set_optimizer_steps(steps_done);
    t
}

fn main() {
    let dir = TempDir::new("resume");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let seq = pattern();

    // 1. Straight through.
    let mut straight = trainer(
        &gpu,
        build_model(&gpu, dir.path()),
        AdamW::new(gpu.engine.clone(), LR),
    );
    let straight_losses = train(&mut straight, &seq, STEPS);
    show(&format!("{STEPS} steps straight:"), &straight_losses);

    // 2. Half, save, reload, the other half.
    let mut first = trainer(
        &gpu,
        build_model(&gpu, dir.path()),
        AdamW::new(gpu.engine.clone(), LR),
    );
    let first_losses = train(&mut first, &seq, HALF);
    let adapter = dir.path().join("adapter.safetensors");
    let named = first.model.named_lora_params();
    pollster::block_on(save_lora_checkpoint(
        &gpu.ctx,
        &adapter,
        &named,
        Some(&lora_metadata()),
    ))
    .expect("save adapter");
    let mut training_state = TrainingState::new(STEPS);
    training_state.step = first.optimizer_steps();
    training_state.lr = LR;
    training_state.sequence_length = SEQ;
    let state_path = dir.path().join("adapter.safetensors.state.json");
    TrainingCheckpoint {
        training_state,
        optimizer_state: first.optimizer.save_state(&gpu.ctx, &named),
        grad_scaler_state: None,
    }
    .save(&state_path)
    .expect("save training state");
    println!("saved adapter and optimizer state after {HALF} steps");

    let mut resumed = resume(&gpu, dir.path(), &adapter, Some(&state_path));
    let resumed_losses = train(&mut resumed, &seq, STEPS - HALF);
    show(
        &format!("{HALF} + {} steps, resumed:", STEPS - HALF),
        &[&first_losses[..], &resumed_losses[..]].concat(),
    );

    // 3. Same adapter, but the optimizer starts over.
    let mut no_state = resume(&gpu, dir.path(), &adapter, None);
    let no_state_losses = train(&mut no_state, &seq, STEPS - HALF);
    show(
        &format!("{HALF} + {}, new optimizer:", STEPS - HALF),
        &[&first_losses[..], &no_state_losses[..]].concat(),
    );

    let worst = straight_losses[HALF..]
        .iter()
        .zip(&resumed_losses)
        .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
        .fold(0.0f32, f32::max);
    println!(
        "largest relative difference, resumed vs straight: {worst:.2e} (allowed {TOLERANCE:.0e})"
    );
    assert!(
        worst <= TOLERANCE,
        "the resumed run should match the straight run: {straight_losses:?} vs {resumed_losses:?}"
    );
    let gap = (straight_losses[STEPS - 1] - no_state_losses[STEPS - HALF - 1]).abs();
    println!("final loss without the optimizer state differs by {gap:.4}");
}
