//! Fused q/k/v and gate/up projections (`Lfm2Model::enable_fused_projections`)
//! against the unfused layers: the same loss and gradients, an f16 mirror of
//! every LoRA `A` that equals `f16(A)` bit for bit after a step and after a
//! resume, and a model that never enabled them left exactly as it was.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test fused_projections -- --ignored --test-threads=1
//! ```

#![allow(clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use ferrotensor_autograd::{FusedProjections, Tensor};
use ferrotensor_graph::lfm::{Lfm2LayerOp, Lfm2Model};
use ferrotensor_graph::model_builder::{Lfm2ModelBuilder, LoRaConfig};
use ferrotensor_graph::FusedMode;
use ferrotensor_loader::{ModelLoader, WeightsContext};
use ferrotensor_trainer::checkpoint::{
    apply_checkpoint, load_lora_checkpoint, save_lora_checkpoint,
};
use std::path::Path;
use std::sync::Arc;

/// The tiny model with f16 frozen weights and rank-4 adapters on all seven
/// projections of every block.
fn build(gpu: &Gpu, dir: &Path) -> Lfm2Model {
    let config = ModelLoader::load_config(dir.join("config.json")).expect("config");
    let mut weights = WeightsContext::new();
    weights.load_model_dir(dir).expect("weights");
    let lora = LoRaConfig {
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
        ]
        .map(String::from)
        .to_vec(),
    };
    Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &weights, config)
        .with_lora(lora)
        .with_f16_weights(true)
        .build()
        .expect("build")
}

/// Non-zero LoRA B matrices, so every LoRA A gradient is non-zero too.
fn set_lora_b(gpu: &Gpu, model: &Lfm2Model) {
    let (names, params) = sorted_lora_params(model);
    for (i, (name, t)) in names.iter().zip(&params).enumerate() {
        if !name.ends_with("lora_B.weight") {
            continue;
        }
        let numel = t.data.shape.iter().product();
        let bytes: Vec<u8> = pseudo_random(numel, 1000 + i as u64, 0.1)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        t.data.write_data(&gpu.ctx, &bytes);
    }
}

/// The loss and the gradient of every LoRA tensor, by name, after one
/// forward and backward pass.
fn loss_and_grads(gpu: &Gpu, model: &Lfm2Model) -> (f32, Vec<(String, Vec<f32>)>) {
    let seq = pattern_sequence();
    let (input, target) = (&seq[..seq.len() - 1], &seq[1..]);
    let logits = model.forward_with_dropout(&gpu.engine, input, 0, None, None, None);
    let mask = vec![1u8; target.len()];
    let loss = ferrotensor_trainer::masked_cross_entropy(&logits, target, &mask, &gpu.engine)
        .expect("loss");
    loss.tensor.backward(&gpu.engine);
    let (names, params) = sorted_lora_params(model);
    let grads = names
        .into_iter()
        .zip(params)
        .map(|(name, t)| {
            let g = t.grad.borrow();
            let g = g
                .as_ref()
                .unwrap_or_else(|| panic!("{name} has no gradient"));
            let values = pollster::block_on(g.to_vec(&gpu.ctx)).expect("GPU readback");
            (name, values)
        })
        .collect();
    (loss.value, grads)
}

/// Every fused group of `model` with the adapters' `A` tensors it mirrors.
fn groups(model: &Lfm2Model) -> Vec<(Arc<FusedProjections>, Vec<Tensor>)> {
    let mut out = Vec::new();
    for layer in &model.layers {
        let a_of = |lins: &[&ferrotensor_graph::Linear]| -> Vec<Tensor> {
            lins.iter()
                .map(|l| l.lora.as_ref().expect("adapter").a.clone())
                .collect()
        };
        if let Lfm2LayerOp::Attention(attn) = &layer.op {
            if let Some(g) = attn.fused.get() {
                let a = a_of(&[&attn.q_proj, &attn.k_proj, &attn.v_proj]);
                out.push((g, a));
            }
        }
        if let Some(g) = layer.mlp.fused.get() {
            let a = a_of(&[&layer.mlp.gate_proj, &layer.mlp.up_proj]);
            out.push((g, a));
        }
    }
    out
}

/// How many mirrored `A` matrices differ from `f16(A)`, and how many there
/// are.
fn mirror_mismatches(gpu: &Gpu, model: &Lfm2Model) -> (usize, usize) {
    let (mut bad, mut total) = (0, 0);
    for (group, a_list) in groups(model) {
        let mirrored = group.mirror_halves(&gpu.ctx).expect("mirror readback");
        assert_eq!(mirrored.len(), a_list.len());
        for (halves, a) in mirrored.iter().zip(&a_list) {
            let want: Vec<u16> = host(gpu, a)
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect();
            total += 1;
            bad += usize::from(*halves != want);
        }
    }
    (bad, total)
}

#[test]
#[ignore = "needs a GPU"]
fn fused_projections_match_the_unfused_layers() {
    let dir = TempDir::new("fused-match");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let off = build(&gpu, dir.path());
    let on = build(&gpu, dir.path());
    set_lora_b(&gpu, &off);
    set_lora_b(&gpu, &on);
    let built = on
        .enable_fused_projections(&gpu.engine, FusedMode::Force)
        .expect("enable");
    assert_eq!(
        built,
        LAYERS * 2,
        "one attention and one MLP group per block"
    );

    let (loss_off, grads_off) = loss_and_grads(&gpu, &off);
    let (loss_on, grads_on) = loss_and_grads(&gpu, &on);
    assert!(
        (loss_on - loss_off).abs() <= 1e-2,
        "loss {loss_on} (fused) vs {loss_off}"
    );
    assert_eq!(
        grads_on.len(),
        LAYERS * 7 * 2,
        "A and B of seven projections"
    );
    for ((name, got), (_, want)) in grads_on.iter().zip(&grads_off) {
        assert_eq!(got.len(), want.len(), "{name}: gradient length");
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(scale > 0.0, "{name}: the reference gradient is zero");
        let worst = got
            .iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 0.05 * scale,
            "{name}: largest difference {worst} against a gradient scale of {scale}"
        );
    }
}

#[test]
#[ignore = "needs a GPU"]
fn fused_projections_train_like_the_unfused_layers() {
    let dir = TempDir::new("fused-train");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let off = build(&gpu, dir.path());
    let on = build(&gpu, dir.path());
    set_lora_b(&gpu, &off);
    set_lora_b(&gpu, &on);
    on.enable_fused_projections(&gpu.engine, FusedMode::Force)
        .expect("enable");
    let (_, params_off) = sorted_lora_params(&off);
    let (_, params_on) = sorted_lora_params(&on);
    let mut trainer_off = trainer(&gpu, off, 1e-2);
    let mut trainer_on = trainer(&gpu, on, 1e-2);

    let seq = pattern_sequence();
    let (input, target) = (&seq[..seq.len() - 1], &seq[1..]);
    for step in 0..6 {
        let a = trainer_on
            .train_step(input, target, &params_on)
            .expect("fused step")
            .loss;
        let b = trainer_off
            .train_step(input, target, &params_off)
            .expect("unfused step")
            .loss;
        assert!(
            (a - b).abs() <= 2e-2,
            "step {step}: loss {a} (fused) vs {b} (unfused)"
        );
    }
    let (bad, total) = mirror_mismatches(&gpu, &trainer_on.model);
    assert_eq!((bad, total), (0, LAYERS * 5), "mirrors after six steps");
}

#[test]
#[ignore = "needs a GPU"]
fn the_f16_mirror_equals_f16_of_a_after_a_step_and_after_a_resume() {
    let dir = TempDir::new("fused-mirror");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build(&gpu, dir.path());
    set_lora_b(&gpu, &model);
    model
        .enable_fused_projections(&gpu.engine, FusedMode::Force)
        .expect("enable");
    assert_eq!(
        mirror_mismatches(&gpu, &model),
        (0, LAYERS * 5),
        "a fresh model"
    );
    let (_, params) = sorted_lora_params(&model);
    let mut trained = trainer(&gpu, model, 1e-2);
    let seq = pattern_sequence();
    let (input, target) = (&seq[..seq.len() - 1], &seq[1..]);
    for _ in 0..3 {
        trained.train_step(input, target, &params).expect("step");
    }
    assert_eq!(
        mirror_mismatches(&gpu, &trained.model),
        (0, LAYERS * 5),
        "after a step"
    );

    // A resume: a fresh model takes the adapter of the trained one.
    let adapter = dir.path().join("adapter.safetensors");
    let named = trained.model.named_lora_params();
    pollster::block_on(save_lora_checkpoint(&gpu.ctx, &adapter, &named, None))
        .expect("save adapter");
    let fresh = build(&gpu, dir.path());
    fresh
        .enable_fused_projections(&gpu.engine, FusedMode::Force)
        .expect("enable");
    let ckpt = load_lora_checkpoint(&adapter).expect("load adapter");
    apply_checkpoint(&gpu.ctx, &ckpt, &fresh.named_lora_params()).expect("apply");
    let (stale, total) = mirror_mismatches(&gpu, &fresh);
    assert!(
        stale > 0,
        "the adapter changed the A matrices, so the mirror is stale until it is refreshed ({total} groups)"
    );
    fresh.refresh_lora_mirrors(&gpu.engine).expect("refresh");
    assert_eq!(
        mirror_mismatches(&gpu, &fresh),
        (0, LAYERS * 5),
        "after a resume"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn a_model_without_fused_projections_runs_exactly_as_before() {
    let dir = TempDir::new("fused-off");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let plain = build(&gpu, dir.path());
    let auto = build(&gpu, dir.path());
    set_lora_b(&gpu, &plain);
    set_lora_b(&gpu, &auto);
    let built = auto
        .enable_fused_projections(&gpu.engine, FusedMode::Auto)
        .expect("enable");
    if gpu.engine.coop_matmul_active() {
        eprintln!("cooperative matmul is on here: Auto builds groups, nothing to compare");
        assert_eq!(built, LAYERS * 2);
        return;
    }
    assert_eq!(built, 0, "Auto builds nothing without cooperative matmul");
    assert!(groups(&auto).is_empty());
    let (loss_a, grads_a) = loss_and_grads(&gpu, &plain);
    let (loss_b, grads_b) = loss_and_grads(&gpu, &auto);
    assert_eq!(loss_a.to_bits(), loss_b.to_bits(), "loss");
    for ((name, a), (_, b)) in grads_a.iter().zip(&grads_b) {
        let same = a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(same, "{name}: gradient differs");
    }
}
