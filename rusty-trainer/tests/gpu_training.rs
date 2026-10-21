//! End-to-end trainer tests on a tiny randomly initialised Llama-style
//! model written to a temporary directory.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p rusty-trainer --test gpu_training -- --ignored --test-threads=1
//! ```

mod common;

use common::*;
use rusty_autograd::{AdamW, Optimizer};
use rusty_trainer::checkpoint::{
    apply_checkpoint, load_lora_checkpoint, save_lora_checkpoint, LoraMetadata, TrainingCheckpoint,
    TrainingState,
};

#[test]
#[ignore = "needs a GPU"]
fn lora_training_reduces_loss_on_a_tiny_model() {
    let dir = TempDir::new("loss");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    let (_, params) = sorted_lora_params(&model);
    assert_eq!(
        params.len(),
        LAYERS * 4 * 2,
        "q/k/v/o adapters, A and B each"
    );
    let mut trainer = trainer(&gpu, model, 1e-2);

    let seq = pattern_sequence();
    let (input, target) = (&seq[..seq.len() - 1], &seq[1..]);
    let mut losses = Vec::new();
    for _ in 0..40 {
        let stats = trainer.train_step(input, target, &params).expect("step");
        assert!(stats.loss.is_finite());
        assert!(stats.optimizer_step);
        losses.push(stats.loss);
    }
    let first = losses[0];
    let last = losses[losses.len() - 1];
    assert!(
        last < 0.5 * first,
        "loss should at least halve on a memorisable sequence: {losses:?}"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn frozen_base_weights_receive_no_gradient() {
    let dir = TempDir::new("frozen");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    let (_, lora) = sorted_lora_params(&model);
    let seq = pattern_sequence();
    let logits = model.forward(&gpu.engine, &seq[..8], 0, None, None);
    let loss = rusty_trainer::masked_cross_entropy(&logits, &seq[1..9], &[1; 8]).expect("loss");
    loss.tensor.backward(&gpu.engine);

    for (name, t) in model.named_params() {
        let has_grad = t.grad.borrow().is_some();
        if name.contains("lora_") {
            // B starts at zero, so A's gradient is zero but present.
            assert!(has_grad, "{name} should get a gradient");
        } else {
            assert!(!has_grad, "frozen {name} got a gradient");
        }
    }
    let b_grad_norm: f32 = lora
        .iter()
        .filter_map(|t| t.grad.borrow().clone())
        .map(|g| {
            pollster::block_on(g.to_vec(&gpu.ctx))
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
        })
        .sum();
    assert!(b_grad_norm > 0.0, "LoRA gradients are all zero");
}

#[test]
#[ignore = "needs a GPU"]
fn gradient_accumulation_defers_the_step_and_is_deterministic() {
    // With grad_accum = 2 the first sequence only accumulates and the second
    // applies the update; repeating the run gives bit-identical adapters,
    // and stepping after every sequence gives different ones.
    let dir = TempDir::new("accum");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let seq = pattern_sequence();
    let a = (&seq[..12], &seq[1..13]);
    let b = (&seq[6..18], &seq[7..19]);

    let run = |accum: usize| {
        let model = build_model(&gpu, dir.path());
        let (_, params) = sorted_lora_params(&model);
        let mut t = trainer(&gpu, model, 1e-2).with_grad_accum_steps(accum);
        let s1 = t.train_step(a.0, a.1, &params).expect("step");
        let s2 = t.train_step(b.0, b.1, &params).expect("step");
        let values: Vec<f32> = params.iter().flat_map(|p| host(&gpu, p)).collect();
        (s1, s2, values)
    };
    let (s1, s2, accum_values) = run(2);
    assert!(!s1.optimizer_step && s2.optimizer_step);
    let (_, _, accum_again) = run(2);
    assert_eq!(accum_values, accum_again, "training is not deterministic");
    let (p1, p2, sequential) = run(1);
    assert!(p1.optimizer_step && p2.optimizer_step);
    assert_ne!(accum_values, sequential);
}

#[test]
#[ignore = "needs a GPU"]
fn checkpoint_resume_round_trip_continues_identically() {
    let dir = TempDir::new("resume");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let seq = pattern_sequence();
    let (input, target) = (&seq[..seq.len() - 1], &seq[1..]);
    let meta = LoraMetadata {
        rank: 4,
        alpha: 8.0,
        target_modules: lora_config().target_modules,
    };

    // Run A: 5 steps, checkpoint, 3 more steps.
    let model = build_model(&gpu, dir.path());
    let (_, params) = sorted_lora_params(&model);
    let mut a = trainer(&gpu, model, 5e-3);
    for _ in 0..5 {
        a.train_step(input, target, &params).expect("step");
    }
    let adapter = dir.path().join("adapter.safetensors");
    let named = a.model.named_lora_params();
    pollster::block_on(save_lora_checkpoint(
        &gpu.ctx,
        &adapter,
        &named,
        Some(&meta),
    ))
    .expect("save adapter");
    let mut state = TrainingState::new(8);
    state.step = a.optimizer_steps();
    let ckpt = TrainingCheckpoint {
        training_state: state,
        optimizer_state: a.optimizer.save_state(&gpu.ctx, &named),
        grad_scaler_state: None,
    };
    let state_path = dir.path().join("adapter.safetensors.state.json");
    ckpt.save(&state_path).expect("save state");
    let a_losses: Vec<f32> = (0..3)
        .map(|_| a.train_step(input, target, &params).expect("step").loss)
        .collect();

    // Run B: fresh model, restore adapter + optimizer, 3 steps.
    let model = build_model(&gpu, dir.path());
    let restored = load_lora_checkpoint(&adapter).expect("load adapter");
    assert_eq!(restored.metadata.as_ref(), Some(&meta));
    let n = apply_checkpoint(&gpu.ctx, &restored, &model.named_lora_params()).expect("apply");
    assert_eq!(n, named.len());
    let (_, params_b) = sorted_lora_params(&model);
    let named_b = model.named_lora_params();
    let mut opt = AdamW::new(gpu.engine.clone(), 5e-3);
    let loaded = TrainingCheckpoint::load(&state_path).expect("load state");
    assert_eq!(loaded.training_state.step, 5);
    assert!(opt.load_state(
        &gpu.ctx,
        &named_b,
        loaded.optimizer_state.as_ref().expect("optimizer state")
    ));
    let mut b =
        rusty_trainer::Trainer::new(model, Box::new(opt), gpu.ctx.clone(), gpu.engine.clone());
    b.set_optimizer_steps(loaded.training_state.step);
    let b_losses: Vec<f32> = (0..3)
        .map(|_| b.train_step(input, target, &params_b).expect("step").loss)
        .collect();

    for (x, y) in a_losses.iter().zip(&b_losses) {
        assert!(
            (x - y).abs() <= 1e-5 * x.abs().max(1.0),
            "resumed run diverged: {a_losses:?} vs {b_losses:?}"
        );
    }
}

#[test]
#[ignore = "needs a GPU"]
fn adapter_with_wrong_rank_is_rejected() {
    let dir = TempDir::new("rank");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    let adapter = dir.path().join("a.safetensors");
    pollster::block_on(save_lora_checkpoint(
        &gpu.ctx,
        &adapter,
        &model.named_lora_params(),
        None,
    ))
    .expect("save");

    // A rank-2 model cannot take rank-4 tensors.
    let config =
        rusty_loader::ModelLoader::load_config(dir.path().join("config.json")).expect("cfg");
    let mut weights = rusty_loader::WeightsContext::new();
    weights.load_model_dir(dir.path()).expect("weights");
    let other = rusty_graph::model_builder::Lfm2ModelBuilder::new(
        gpu.ctx.clone(),
        gpu.engine.clone(),
        &weights,
        config,
    )
    .with_lora(rusty_graph::model_builder::LoRaConfig {
        rank: 2,
        ..lora_config()
    })
    .build()
    .expect("build");
    let ckpt = load_lora_checkpoint(&adapter).expect("load");
    assert!(ckpt.metadata.is_none());
    let err = apply_checkpoint(&gpu.ctx, &ckpt, &other.named_lora_params()).expect_err("mismatch");
    assert!(err.contains("shape"), "{err}");
}

#[test]
#[ignore = "needs a GPU"]
fn kv_cache_decoding_matches_full_recompute() {
    let dir = TempDir::new("kv");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    let seq = pattern_sequence();
    let full = host(&gpu, &model.forward(&gpu.engine, &seq[..10], 0, None, None));

    let mut cache = model.new_kv_cache(16).expect("llama supports a kv cache");
    let mut last_rows = Vec::new();
    let prefix = host(
        &gpu,
        &model.forward(&gpu.engine, &seq[..6], 0, Some(&mut cache), None),
    );
    last_rows.extend_from_slice(&prefix[5 * VOCAB..6 * VOCAB]);
    for pos in 6..10 {
        let step = host(
            &gpu,
            &model.forward(&gpu.engine, &seq[pos..pos + 1], pos, Some(&mut cache), None),
        );
        last_rows.extend_from_slice(&step);
    }
    let want = &full[5 * VOCAB..10 * VOCAB];
    for (i, (x, y)) in want.iter().zip(&last_rows).enumerate() {
        assert!((x - y).abs() < 1e-3, "logit {i}: full {x} vs cached {y}");
    }
}
