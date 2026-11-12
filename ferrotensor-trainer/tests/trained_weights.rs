//! Base weights trained by full or partial fine-tuning, saved and loaded
//! back, on a tiny Llama-style model. The tests train the attention
//! projections, the set `ferrotensor-cli train --train-modules attn` selects.
//!
//! These need a GPU, so they are `#[ignore]`d by default:
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test trained_weights -- --ignored --test-threads=1
//! ```

mod common;

use common::*;
use ferrotensor_autograd::{AdamW, Optimizer, Tensor};
use ferrotensor_graph::lfm::Lfm2Model;
use ferrotensor_graph::model_builder::Lfm2ModelBuilder;
use ferrotensor_loader::{ModelLoader, WeightsContext};
use ferrotensor_trainer::checkpoint::{
    apply_checkpoint, load_lora_checkpoint, save_lora_checkpoint, LoraMetadata, TrainingCheckpoint,
    TrainingState,
};
use ferrotensor_trainer::Trainer;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

/// A model without adapters. `trainable` is how `train` builds it when base
/// weights train; with `f16_weights`, frozen linear weights are stored as f16,
/// which is how `eval` and `generate` build a model by default.
fn build_base(gpu: &Gpu, dir: &Path, trainable: bool, f16_weights: bool) -> Lfm2Model {
    let config = ModelLoader::load_config(dir.join("config.json")).expect("config");
    let mut weights = WeightsContext::new();
    weights.load_model_dir(dir).expect("weights");
    Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &weights, config)
        .with_trainable_base(trainable)
        .with_f16_weights(f16_weights)
        .build()
        .expect("build")
}

/// The tensors `--train-modules attn` trains.
fn attn_params(model: &Lfm2Model) -> HashMap<String, Tensor> {
    model
        .named_params()
        .into_iter()
        .filter(|(name, _)| name.contains("self_attn"))
        .collect()
}

fn sorted(named: &HashMap<String, Tensor>) -> Vec<Tensor> {
    let mut names: Vec<&String> = named.keys().collect();
    names.sort();
    names.into_iter().map(|n| named[n].clone()).collect()
}

fn probe_ids() -> Vec<u32> {
    (0..12).map(|i| ((i * 5 + 2) % VOCAB) as u32).collect()
}

fn logits(gpu: &Gpu, model: &Lfm2Model) -> Vec<f32> {
    host(
        gpu,
        &model.forward(&gpu.engine, &probe_ids(), 0, None, None),
    )
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

fn train_steps(trainer: &mut Trainer, params: &[Tensor], steps: usize) -> Vec<f32> {
    let seq = pattern_sequence();
    (0..steps)
        .map(|_| {
            let stats = trainer
                .train_step(&seq[..seq.len() - 1], &seq[1..], params)
                .expect("step");
            assert!(stats.loss.is_finite());
            stats.loss
        })
        .collect()
}

#[test]
#[ignore = "needs a GPU"]
fn trained_base_weights_save_and_load_into_a_fresh_model() {
    let dir = TempDir::new("weights-roundtrip");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let untrained = logits(&gpu, &build_base(&gpu, dir.path(), false, false));

    let model = build_base(&gpu, dir.path(), true, false);
    let named = attn_params(&model);
    assert_eq!(named.len(), LAYERS * 4, "q/k/v/o of every layer");
    let mut trained = trainer(&gpu, model, 1e-2);
    train_steps(&mut trained, &sorted(&named), 6);
    let trained_logits = logits(&gpu, &trained.model);
    assert!(
        max_abs_diff(&trained_logits, &untrained) > 1e-4,
        "training changed nothing"
    );

    let path = dir.path().join("trained_weights.safetensors");
    let meta = LoraMetadata::trained_weights("attn");
    pollster::block_on(save_lora_checkpoint(&gpu.ctx, &path, &named, Some(&meta))).expect("save");

    // The file holds exactly the trained tensors, with their values.
    let file = load_lora_checkpoint(&path).expect("load");
    assert_eq!(file.metadata.as_ref(), Some(&meta));
    let saved: BTreeSet<&String> = file.tensors.keys().collect();
    let wanted: BTreeSet<&String> = named.keys().collect();
    assert_eq!(saved, wanted);
    for (name, tensor) in &named {
        let (shape, values) = &file.tensors[name];
        assert_eq!(*shape, tensor.data.shape, "{name}");
        assert_eq!(*values, host(&gpu, tensor), "{name}");
    }

    // A fresh frozen f32 model takes the file and gives the same logits.
    let fresh = build_base(&gpu, dir.path(), false, false);
    assert!(max_abs_diff(&logits(&gpu, &fresh), &trained_logits) > 1e-4);
    let n = apply_checkpoint(&gpu.ctx, &file, &attn_params(&fresh)).expect("apply");
    assert_eq!(n, named.len());
    let restored = logits(&gpu, &fresh);
    assert!(
        max_abs_diff(&restored, &trained_logits) <= 1e-6,
        "restored logits differ by {}",
        max_abs_diff(&restored, &trained_logits)
    );
    assert!(max_abs_diff(&restored, &untrained) > 1e-4);

    // The file is strict: another selection of the same model does not fit.
    let all: HashMap<String, Tensor> = fresh
        .named_params()
        .into_iter()
        .filter(|(name, _)| !name.contains("embed_tokens"))
        .collect();
    let err = apply_checkpoint(&gpu.ctx, &file, &all).expect_err("more tensors than the file has");
    assert!(err.contains("checkpoint has no tensor"), "{err}");
    let mlp: HashMap<String, Tensor> = fresh
        .named_params()
        .into_iter()
        .filter(|(name, _)| name.contains(".mlp."))
        .collect();
    let err = apply_checkpoint(&gpu.ctx, &file, &mlp).expect_err("other tensors");
    assert!(err.contains("checkpoint has no tensor"), "{err}");
}

#[test]
#[ignore = "needs a GPU"]
fn a_lora_save_keeps_its_tensors_and_metadata() {
    let dir = TempDir::new("weights-lora-unchanged");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    let (names, params) = sorted_lora_params(&model);
    let mut t = trainer(&gpu, model, 1e-2);
    train_steps(&mut t, &params, 2);

    let meta = LoraMetadata {
        rank: 4,
        alpha: 8.0,
        target_modules: lora_config().target_modules,
        trained_modules: None,
    };
    let path = dir.path().join("lora_adapters.safetensors");
    let named = t.model.named_lora_params();
    pollster::block_on(save_lora_checkpoint(&gpu.ctx, &path, &named, Some(&meta))).expect("save");

    let file = load_lora_checkpoint(&path).expect("load");
    assert_eq!(file.metadata.as_ref(), Some(&meta));
    let keys: Vec<&String> = {
        let mut k: Vec<&String> = file.tensors.keys().collect();
        k.sort();
        k
    };
    assert_eq!(keys, names.iter().collect::<Vec<_>>());
    assert!(names.iter().all(|n| n.contains("lora_")));

    // The header keys are the ones earlier versions wrote.
    let bytes = std::fs::read(&path).expect("read");
    let (_, header) = safetensors::tensor::SafeTensors::read_metadata(&bytes).expect("header");
    let header = header.metadata().clone().expect("metadata");
    let keys: BTreeSet<&str> = header.keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        BTreeSet::from(["alpha", "format", "rank", "target_modules"])
    );
    assert_eq!(header["format"], "ferrotensor-lora");
}

#[test]
#[ignore = "needs a GPU"]
fn trained_weights_do_not_load_into_a_model_that_cannot_take_them() {
    let dir = TempDir::new("weights-cannot-load");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build_base(&gpu, dir.path(), true, false);
    let named = attn_params(&model);
    let mut trained = trainer(&gpu, model, 1e-2);
    train_steps(&mut trained, &sorted(&named), 3);
    let path = dir.path().join("trained_weights.safetensors");
    let meta = LoraMetadata::trained_weights("attn");
    pollster::block_on(save_lora_checkpoint(&gpu.ctx, &path, &named, Some(&meta))).expect("save");
    let file = load_lora_checkpoint(&path).expect("load");

    // Frozen f16 weights (the default for eval and generate) cannot hold
    // trained f32 values: a clear error, and nothing is written.
    let half = build_base(&gpu, dir.path(), false, true);
    let before = logits(&gpu, &half);
    let err = apply_checkpoint(&gpu.ctx, &file, &attn_params(&half)).expect_err("f16 model");
    assert!(
        err.contains("FP16") && err.contains("f32 weights"),
        "unclear error: {err}"
    );
    assert_eq!(logits(&gpu, &half), before, "a failed load wrote something");

    // A model with LoRA adapters has no use for them either.
    let lora = build_model(&gpu, dir.path());
    let err = apply_checkpoint(&gpu.ctx, &file, &lora.named_lora_params()).expect_err("lora model");
    assert!(err.contains("checkpoint has no tensor"), "{err}");

    // And f16 tensors cannot be saved: an error, not a garbled file.
    let too_small = dir.path().join("f16.safetensors");
    let err = pollster::block_on(save_lora_checkpoint(
        &gpu.ctx,
        &too_small,
        &attn_params(&half),
        None,
    ))
    .expect_err("f16 tensors")
    .to_string();
    assert!(err.contains("FP16"), "{err}");
    assert!(!too_small.exists());
}

#[test]
#[ignore = "needs a GPU"]
fn trained_weights_resume_continues_identically() {
    let dir = TempDir::new("weights-resume");
    write_tiny_llama(dir.path());
    let gpu = gpu();

    // Run A: 4 steps, save weights and optimizer state, 3 more steps.
    let model = build_base(&gpu, dir.path(), true, false);
    let named = attn_params(&model);
    let params = sorted(&named);
    let mut a = trainer(&gpu, model, 5e-3);
    train_steps(&mut a, &params, 4);
    let weights = dir.path().join("trained_weights.safetensors");
    let meta = LoraMetadata::trained_weights("attn");
    pollster::block_on(save_lora_checkpoint(
        &gpu.ctx,
        &weights,
        &named,
        Some(&meta),
    ))
    .expect("save weights");
    let mut state = TrainingState::new(7);
    state.step = a.optimizer_steps();
    state.train_modules = "attn".into();
    let state_path = dir.path().join("trained_weights.safetensors.state.json");
    TrainingCheckpoint {
        training_state: state,
        optimizer_state: a.optimizer.save_state(&gpu.ctx, &named),
        grad_scaler_state: None,
    }
    .save(&state_path)
    .expect("save state");
    let a_losses = train_steps(&mut a, &params, 3);

    // Run B: a fresh model built the way `train --resume` builds it.
    let model = build_base(&gpu, dir.path(), true, false);
    let named_b = attn_params(&model);
    let file = load_lora_checkpoint(&weights).expect("load weights");
    assert_eq!(file.metadata.as_ref(), Some(&meta));
    apply_checkpoint(&gpu.ctx, &file, &named_b).expect("apply");
    let loaded = TrainingCheckpoint::load(&state_path).expect("load state");
    let mut opt = AdamW::new(gpu.engine.clone(), 5e-3);
    let load = opt
        .load_state(
            &gpu.ctx,
            &named_b,
            loaded.optimizer_state.as_ref().expect("optimizer state"),
        )
        .expect("load optimizer state");
    assert_eq!(load.matched, named_b.len());
    assert!(load.is_complete(), "{load:?}");
    let params_b = sorted(&named_b);
    let mut b = Trainer::new(model, Box::new(opt), gpu.ctx.clone(), gpu.engine.clone());
    b.set_optimizer_steps(loaded.training_state.step);
    let b_losses = train_steps(&mut b, &params_b, 3);

    for (x, y) in a_losses.iter().zip(&b_losses) {
        assert!(
            (x - y).abs() <= 1e-5 * x.abs().max(1.0),
            "resumed run diverged: {a_losses:?} vs {b_losses:?}"
        );
    }
}
