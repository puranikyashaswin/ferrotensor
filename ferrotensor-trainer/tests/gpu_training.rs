//! End-to-end trainer tests on a tiny randomly initialised Llama-style
//! model written to a temporary directory.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test gpu_training -- --ignored --test-threads=1
//! ```

mod common;

use common::*;
use ferrotensor_autograd::{AdamW, Optimizer};
use ferrotensor_backend::DType;
use ferrotensor_trainer::checkpoint::{
    apply_checkpoint, load_lora_checkpoint, save_lora_checkpoint, LoraMetadata, TrainingCheckpoint,
    TrainingState,
};
use ferrotensor_trainer::Batch;

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
fn f16_base_weights_halve_memory_and_train_like_f32() {
    let dir = TempDir::new("f16");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let full = build_model(&gpu, dir.path());
    let half = build_model_with(&gpu, dir.path(), true);

    let w = &half.lm_head.weight.data;
    let numel: usize = w.shape.iter().product();
    assert_eq!(full.lm_head.weight.data.dtype, DType::FP32);
    assert_eq!(w.dtype, DType::FP16);
    assert_eq!(w.size_in_bytes(), ((numel * 2).div_ceil(4) * 4) as u64);

    let seq = pattern_sequence();
    let (input, target) = (&seq[..seq.len() - 1], &seq[1..]);
    let mut losses = [Vec::new(), Vec::new()];
    for (i, model) in [full, half].into_iter().enumerate() {
        let (_, params) = sorted_lora_params(&model);
        let mut trainer = trainer(&gpu, model, 1e-2);
        for _ in 0..40 {
            let stats = trainer.train_step(input, target, &params).expect("step");
            assert!(stats.loss.is_finite());
            losses[i].push(stats.loss);
        }
    }
    let (f32_losses, f16_losses) = (&losses[0], &losses[1]);
    // Same adapters and data; only the base weights are rounded to f16.
    assert!(
        (f16_losses[0] - f32_losses[0]).abs() <= 1e-2 * f32_losses[0],
        "first loss: f32 {} vs f16 {}",
        f32_losses[0],
        f16_losses[0]
    );
    assert!(
        f16_losses[39] < 0.5 * f16_losses[0],
        "f16 weights should still learn the sequence: {f16_losses:?}"
    );
}

/// A `Batch` of `(inputs, targets, loss mask)` sequences, right-padded.
fn make_batch(seqs: &[(Vec<u32>, Vec<u32>, Vec<u8>)]) -> Batch {
    let max = seqs.iter().map(|s| s.0.len()).max().unwrap_or(0);
    let pad = |v: &Vec<u32>| {
        let mut p = v.clone();
        p.resize(max, 0);
        p
    };
    Batch {
        input_ids: seqs.iter().map(|s| s.0.clone()).collect(),
        target_ids: seqs.iter().map(|s| s.1.clone()).collect(),
        padded_input_ids: seqs.iter().map(|s| pad(&s.0)).collect(),
        padded_target_ids: seqs.iter().map(|s| pad(&s.1)).collect(),
        attention_mask: seqs
            .iter()
            .map(|s| (0..max).map(|i| u8::from(i < s.0.len())).collect())
            .collect(),
        loss_masks: seqs
            .iter()
            .map(|s| {
                let mut m = s.2.clone();
                m.resize(max, 0);
                m
            })
            .collect(),
        batch_size: seqs.len(),
        max_seq_len: max,
    }
}

#[test]
#[ignore = "needs a GPU"]
fn batched_training_matches_one_sequence_at_a_time() {
    let dir = TempDir::new("batched");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let seq = pattern_sequence();
    // Three lengths, one with a prompt that is not supervised.
    let seqs: Vec<(Vec<u32>, Vec<u32>, Vec<u8>)> =
        [(0usize, 10usize, 0usize), (2, 17, 3), (5, 12, 0)]
            .iter()
            .map(|&(start, end, unsupervised)| {
                let input = seq[start..end - 1].to_vec();
                let target = seq[start + 1..end].to_vec();
                let mask = (0..input.len())
                    .map(|i| u8::from(i >= unsupervised))
                    .collect();
                (input, target, mask)
            })
            .collect();

    let model = build_model(&gpu, dir.path());
    assert!(model.supports_batching());
    let (_, params) = sorted_lora_params(&model);
    let mut batched = trainer(&gpu, model, 1e-2).with_grad_accum_steps(3);
    let batched_stats = batched
        .train_batch(&make_batch(&seqs), &params)
        .expect("batched step");

    let model = build_model(&gpu, dir.path());
    let (_, params) = sorted_lora_params(&model);
    let mut sequential = trainer(&gpu, model, 1e-2).with_grad_accum_steps(3);
    let sequential_stats: Vec<_> = seqs
        .iter()
        .map(|(i, t, m)| {
            sequential
                .train_step_masked(i, t, Some(m), &params)
                .expect("sequential step")
        })
        .collect();

    assert_eq!(batched_stats.len(), 3);
    for (b, s) in batched_stats.iter().zip(&sequential_stats) {
        assert_eq!(b.tokens, s.tokens);
        assert!(
            (b.loss - s.loss).abs() <= 1e-4 * s.loss.abs().max(1.0),
            "loss {} vs {}",
            b.loss,
            s.loss
        );
        assert_eq!(b.optimizer_step, s.optimizer_step);
    }
    // The optimizer steps after the third sequence in both, on the same
    // accumulated gradient.
    let (bn, sn) = (
        batched_stats[2].grad_norm.expect("batched stepped"),
        sequential_stats[2].grad_norm.expect("sequential stepped"),
    );
    assert!((bn - sn).abs() <= 1e-4 * sn, "grad norm {bn} vs {sn}");
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
    let loss = ferrotensor_trainer::masked_cross_entropy(&logits, &seq[1..9], &[1; 8], &gpu.engine)
        .expect("loss");
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
    let mut b = ferrotensor_trainer::Trainer::new(
        model,
        Box::new(opt),
        gpu.ctx.clone(),
        gpu.engine.clone(),
    );
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
        ferrotensor_loader::ModelLoader::load_config(dir.path().join("config.json")).expect("cfg");
    let mut weights = ferrotensor_loader::WeightsContext::new();
    weights.load_model_dir(dir.path()).expect("weights");
    let other = ferrotensor_graph::model_builder::Lfm2ModelBuilder::new(
        gpu.ctx.clone(),
        gpu.engine.clone(),
        &weights,
        config,
    )
    .with_lora(ferrotensor_graph::model_builder::LoRaConfig {
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

/// LoRA on all seven projections of a Llama layer, plus the tied `lm_head`.
fn all_projections_lora() -> ferrotensor_graph::model_builder::LoRaConfig {
    ferrotensor_graph::model_builder::LoRaConfig {
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

/// An adapter tensor's `(name, shape, values)`.
type Named = (String, Vec<usize>, Vec<f32>);

/// Every adapter tensor, `lora_A` and `lora_B` alike, sorted by name.
fn adapter_matrices(gpu: &Gpu, model: &ferrotensor_graph::lfm::Lfm2Model) -> Vec<Named> {
    let (names, params) = sorted_lora_params(model);
    names
        .into_iter()
        .zip(params)
        .map(|(name, t)| (name, t.data.shape.clone(), host(gpu, &t)))
        .collect()
}

#[test]
#[ignore = "needs a GPU"]
fn lora_a_matrices_all_start_different() {
    let dir = TempDir::new("lora-init");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build_model_with_lora(&gpu, dir.path(), all_projections_lora());
    let all = adapter_matrices(&gpu, &model);
    let a: Vec<&Named> = all.iter().filter(|m| m.0.contains("lora_A")).collect();
    let b: Vec<&Named> = all.iter().filter(|m| m.0.contains("lora_B")).collect();
    assert_eq!(
        a.len(),
        LAYERS * 7 + 1,
        "seven projections per layer + lm_head"
    );
    assert_eq!(b.len(), a.len());

    let mut equal_shape_pairs = 0;
    for (i, x) in a.iter().enumerate() {
        for y in &a[i + 1..] {
            if x.1 == y.1 {
                equal_shape_pairs += 1;
                assert_ne!(x.2, y.2, "{} and {} start from the same A", x.0, y.0);
            }
        }
    }
    // q/o, k/v and gate/up (and the lm_head) share shapes in every layer, so
    // the comparison above covers the cases that used to collide.
    assert!(equal_shape_pairs >= LAYERS * 3, "{equal_shape_pairs} pairs");

    for (name, shape, values) in &a {
        let bound = 1.0 / (shape[0] as f32).sqrt();
        assert!(
            values.iter().all(|v| v.abs() <= bound),
            "{name} outside +-1/sqrt(in)"
        );
        assert!(values.iter().any(|v| *v != 0.0), "{name} is all zero");
    }
    for (name, _, values) in &b {
        assert!(
            values.iter().all(|v| *v == 0.0),
            "{name} does not start at 0"
        );
    }
}

#[test]
#[ignore = "needs a GPU"]
fn lora_init_is_identical_across_builds() {
    let dir = TempDir::new("lora-init-det");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let first = adapter_matrices(
        &gpu,
        &build_model_with_lora(&gpu, dir.path(), all_projections_lora()),
    );
    let second = adapter_matrices(
        &gpu,
        &build_model_with_lora(&gpu, dir.path(), all_projections_lora()),
    );
    assert_eq!(first.len(), second.len());
    for (x, y) in first.iter().zip(&second) {
        assert_eq!(x.0, y.0);
        assert_eq!(x.2, y.2, "{} differs between builds", x.0);
    }
}
