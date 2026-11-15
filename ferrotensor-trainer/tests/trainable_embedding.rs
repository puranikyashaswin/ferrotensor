//! A trainable token embedding on a tiny Llama-style model: its gradient
//! against finite differences, optimizer steps reaching the lookup, block
//! checkpointing, and saving and loading the trained table. Also that the
//! table stays frozen when it is not selected.
//!
//! These need a GPU, so they are `#[ignore]`d by default:
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test trainable_embedding -- --ignored --test-threads=1
//! ```

#![allow(clippy::expect_used)]

mod common;

use common::*;
use ferrotensor_autograd::{no_grad, Tensor};
use ferrotensor_graph::lfm::Lfm2Model;
use ferrotensor_graph::model_builder::Lfm2ModelBuilder;
use ferrotensor_loader::{ModelLoader, WeightsContext};
use ferrotensor_trainer::checkpoint::{
    apply_checkpoint, load_lora_checkpoint, save_lora_checkpoint, LoraMetadata,
};
use ferrotensor_trainer::masked_cross_entropy;
use std::collections::HashMap;
use std::path::Path;

const EMBED: &str = "model.embed_tokens.weight";

/// Ids 5 and 9 repeat, so their rows sum the gradients of several positions.
const INPUT: [u32; 7] = [5, 9, 5, 12, 9, 5, 20];
const TARGET: [u32; 7] = [9, 5, 12, 9, 5, 20, 3];

/// The tiny model with every base weight in f32; `trainable` is how `train`
/// builds it when base weights train.
fn build_base(gpu: &Gpu, dir: &Path, trainable: bool) -> Lfm2Model {
    let config = ModelLoader::load_config(dir.join("config.json")).expect("config");
    let mut weights = WeightsContext::new();
    weights.load_model_dir(dir).expect("weights");
    Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &weights, config)
        .with_trainable_base(trainable)
        .with_f16_weights(false)
        .build()
        .expect("build")
}

/// The model `--train-modules embed` trains: a trainable base whose
/// embedding is selected.
fn embed_model(gpu: &Gpu, dir: &Path) -> Lfm2Model {
    build_base(gpu, dir, true).with_trainable_embedding(true)
}

/// The loss tensor and value of [`INPUT`] against [`TARGET`], every position
/// supervised.
fn loss(gpu: &Gpu, model: &Lfm2Model) -> (Tensor, f32) {
    let logits = model.forward(&gpu.engine, &INPUT, 0, None, None);
    let loss = masked_cross_entropy(&logits, &TARGET, &[1; 7], &gpu.engine).expect("loss");
    (loss.tensor, loss.value)
}

/// The embedding output (tap 0) for `rows`.
fn embedded(gpu: &Gpu, model: &Lfm2Model, rows: &[&[u32]]) -> Tensor {
    model
        .hidden_states_batch(&gpu.engine, rows, &[0])
        .pop()
        .expect("one tap")
}

fn write(gpu: &Gpu, tensor: &Tensor, values: &[f32]) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    tensor.data.write_data(&gpu.ctx, &bytes);
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

/// Central differences of the loss against the table, for the rows of a
/// repeated id (5), an id used once (12) and an unused id (40), at three
/// columns each.
///
/// The loss is an f32 mean of about ln(64) = 4.2, so one evaluation rounds
/// by a few times 5e-7 and the difference quotient over 2h = 1e-2 by a few
/// times 1e-4; the step error h^2 f'''/6 adds a term that grows with the
/// gradient. The bound is therefore 1e-3 absolute plus 2e-2 relative, the
/// relative part as in `ferrotensor-autograd/tests/grad_check.rs`. The exact
/// accumulation over repeated ids is checked against hand-computed values in
/// `ferrotensor-autograd/tests/embedding.rs`; this test checks the gradient
/// that reaches the table through the whole model.
#[test]
#[ignore = "needs a GPU"]
fn embedding_gradient_matches_finite_differences() {
    let dir = TempDir::new("embed-fd");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    // A frozen base, so only the table records a gradient.
    let model = build_base(&gpu, dir.path(), false).with_trainable_embedding(true);
    let (loss_tensor, _) = loss(&gpu, &model);
    loss_tensor.backward(&gpu.engine);
    let grad = model
        .embed_tokens
        .grad
        .borrow()
        .clone()
        .expect("the table has a gradient");
    let grad = pollster::block_on(grad.to_vec(&gpu.ctx));
    let table = host(&gpu, &model.embed_tokens);

    let h = 5e-3f32;
    let loss_at = |index: usize, delta: f32| {
        let mut perturbed = table.clone();
        perturbed[index] += delta;
        write(&gpu, &model.embed_tokens, &perturbed);
        let value = no_grad(|| loss(&gpu, &model).1);
        write(&gpu, &model.embed_tokens, &table);
        f64::from(value)
    };
    for id in [5usize, 12, 40] {
        for col in [0usize, 7, HIDDEN - 1] {
            let index = id * HIDDEN + col;
            let numeric = (loss_at(index, h) - loss_at(index, -h)) / (2.0 * f64::from(h));
            let analytic = f64::from(grad[index]);
            assert!(
                (analytic - numeric).abs() <= 1e-3 + 2e-2 * numeric.abs(),
                "row {id} col {col}: analytic {analytic:.6} vs numeric {numeric:.6}"
            );
        }
    }
    // The unused row gets exactly nothing; the used rows do get a gradient.
    let unused = &grad[40 * HIDDEN..41 * HIDDEN];
    assert!(unused.iter().all(|&g| g == 0.0), "row 40: {unused:?}");
    for id in [5usize, 12] {
        let row = &grad[id * HIDDEN..(id + 1) * HIDDEN];
        assert!(row.iter().any(|g| g.abs() > 1e-4), "row {id}: {row:?}");
    }
}

/// One optimizer step changes the table, the next lookup returns the new
/// rows exactly, and the loss on one repeated sequence falls over a few
/// steps. A tied `lm_head` is a separate copy and does not move.
#[test]
#[ignore = "needs a GPU"]
fn optimizer_steps_reach_the_lookup_and_lower_the_loss() {
    let dir = TempDir::new("embed-steps");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = embed_model(&gpu, dir.path());
    let params = vec![model.embed_tokens.clone()];
    let before = host(&gpu, &model.embed_tokens);
    let head_before = host(&gpu, &model.lm_head.weight);
    let mut t = trainer(&gpu, model, 1e-2);

    let mut losses = Vec::new();
    let stats = t.train_step(&INPUT, &TARGET, &params).expect("step");
    assert!(stats.optimizer_step);
    losses.push(stats.loss);
    let after = host(&gpu, &t.model.embed_tokens);
    assert!(
        max_abs_diff(&before, &after) > 1e-4,
        "the table did not move"
    );

    let rows: [&[u32]; 1] = [&INPUT];
    let looked_up = host(&gpu, &embedded(&gpu, &t.model, &rows));
    let expected: Vec<f32> = INPUT
        .iter()
        .flat_map(|&id| after[id as usize * HIDDEN..(id as usize + 1) * HIDDEN].to_vec())
        .collect();
    assert_eq!(looked_up, expected, "the lookup used a stale table");

    for _ in 0..7 {
        losses.push(t.train_step(&INPUT, &TARGET, &params).expect("step").loss);
    }
    assert!(
        losses[7] < losses[0] - 1e-3,
        "loss did not fall: {losses:?}"
    );
    assert_eq!(host(&gpu, &t.model.lm_head.weight), head_before);
}

/// With block checkpointing the embedding output feeds the first replayed
/// block; its gradient must come out as without checkpointing (the
/// tolerance of `block_checkpointing.rs`: 1e-6 plus 1e-4 relative).
#[test]
#[ignore = "needs a GPU"]
fn block_checkpointing_gives_the_same_embedding_gradient() {
    let dir = TempDir::new("embed-ckpt");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let grad = |checkpointing: bool| {
        let model = embed_model(&gpu, dir.path()).with_block_checkpointing(checkpointing);
        loss(&gpu, &model).0.backward(&gpu.engine);
        let grad = model.embed_tokens.grad.borrow().clone().expect("gradient");
        pollster::block_on(grad.to_vec(&gpu.ctx))
    };
    let (on, off) = (grad(true), grad(false));
    assert_eq!(on.len(), VOCAB * HIDDEN);
    for (i, (a, b)) in on.iter().zip(&off).enumerate() {
        assert!(
            (a - b).abs() <= 1e-6 + 1e-4 * b.abs(),
            "[{i}]: {a} (checkpointed) vs {b}"
        );
    }
    assert!(off.iter().any(|g| g.abs() > 1e-4));
}

/// Without `with_trainable_embedding` the lookup is a constant, whether
/// the other base weights train (`--train-modules all`) or LoRA does, and
/// a trainable table records nothing under `no_grad`.
#[test]
#[ignore = "needs a GPU"]
fn the_embedding_stays_frozen_unless_selected() {
    let dir = TempDir::new("embed-frozen");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let rows: [&[u32]; 1] = [&INPUT];
    for model in [
        build_base(&gpu, dir.path(), true),
        build_model(&gpu, dir.path()),
    ] {
        assert!(!model.embed_tokens.requires_grad);
        let out = embedded(&gpu, &model, &rows);
        assert!(out.creator.is_none() && !out.requires_grad);
        loss(&gpu, &model).0.backward(&gpu.engine);
        assert!(model.embed_tokens.grad.borrow().is_none());
    }
    let model = embed_model(&gpu, dir.path());
    let out = no_grad(|| embedded(&gpu, &model, &rows));
    assert!(out.creator.is_none() && !out.requires_grad);
    let out = embedded(&gpu, &model, &rows);
    assert!(out.creator.is_some() && out.requires_grad);
}

/// A trained table saved as trained weights holds the trained values and,
/// loaded into a fresh frozen model, gives the trained model's logits once
/// the build-time host copy is cleared.
#[test]
#[ignore = "needs a GPU"]
fn a_trained_table_saves_and_loads_into_a_fresh_model() {
    let dir = TempDir::new("embed-roundtrip");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = embed_model(&gpu, dir.path());
    let named = HashMap::from([(EMBED.to_string(), model.embed_tokens.clone())]);
    let params = vec![model.embed_tokens.clone()];
    let mut t = trainer(&gpu, model, 1e-2);
    for _ in 0..3 {
        t.train_step(&INPUT, &TARGET, &params).expect("step");
    }
    let logits = |model: &Lfm2Model| {
        let out = no_grad(|| model.forward(&gpu.engine, &INPUT, 0, None, None));
        host(&gpu, &out)
    };
    let trained_logits = logits(&t.model);

    let path = dir.path().join("trained_weights.safetensors");
    let meta = LoraMetadata::trained_weights("embed");
    pollster::block_on(save_lora_checkpoint(&gpu.ctx, &path, &named, Some(&meta))).expect("save");
    let file = load_lora_checkpoint(&path).expect("load");
    assert_eq!(file.metadata.as_ref(), Some(&meta));
    assert_eq!(file.tensors.len(), 1);
    let (shape, values) = &file.tensors[EMBED];
    assert_eq!(*shape, [VOCAB, HIDDEN]);
    assert_eq!(*values, host(&gpu, &t.model.embed_tokens));

    let fresh = build_base(&gpu, dir.path(), false);
    let untrained_logits = logits(&fresh);
    assert!(max_abs_diff(&untrained_logits, &trained_logits) > 1e-4);
    let fresh_named = HashMap::from([(EMBED.to_string(), fresh.embed_tokens.clone())]);
    assert_eq!(
        apply_checkpoint(&gpu.ctx, &file, &fresh_named).expect("apply"),
        1
    );
    fresh.clear_host_embeddings();
    let restored = logits(&fresh);
    assert!(
        max_abs_diff(&restored, &trained_logits) <= 1e-6,
        "restored logits differ by {}",
        max_abs_diff(&restored, &trained_logits)
    );
}
