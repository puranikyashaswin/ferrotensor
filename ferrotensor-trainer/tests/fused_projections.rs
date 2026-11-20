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
use ferrotensor_graph::{FuseChoice, FuseSwitch, FusedMode};
use ferrotensor_loader::{ModelLoader, WeightsContext};
use ferrotensor_trainer::checkpoint::{
    apply_checkpoint, load_lora_checkpoint, save_lora_checkpoint, LoraMetadata,
};
use ferrotensor_trainer::merge::{merge_adapter, MERGED_WEIGHTS_FILE};
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
    let mut on = build(&gpu, dir.path());
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
    let mut on = build(&gpu, dir.path());
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
    let mut model = build(&gpu, dir.path());
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
    let mut fresh = build(&gpu, dir.path());
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
    let mut auto = build(&gpu, dir.path());
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

/// The frozen weight of every q, k, v, gate and up projection of `model` as
/// `(name, values)`, in layer order. A fused model's own layers hold a
/// placeholder instead, so this is for a model that never fused.
fn base_weights(gpu: &Gpu, model: &Lfm2Model) -> Vec<(String, Vec<f32>)> {
    let mut out = Vec::new();
    for (i, layer) in model.layers.iter().enumerate() {
        if let Lfm2LayerOp::Attention(attn) = &layer.op {
            for (n, l) in [
                ("q", &attn.q_proj),
                ("k", &attn.k_proj),
                ("v", &attn.v_proj),
            ] {
                out.push((format!("{i}.{n}"), host(gpu, &l.weight)));
            }
        }
        for (n, l) in [("gate", &layer.mlp.gate_proj), ("up", &layer.mlp.up_proj)] {
            out.push((format!("{i}.{n}"), host(gpu, &l.weight)));
        }
    }
    out
}

fn bits(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect()
}

#[test]
#[ignore = "needs a GPU"]
fn switches_off_run_exactly_the_unfused_path() {
    let dir = TempDir::new("fused-switch-off");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let plain = build(&gpu, dir.path());
    let mut off = build(&gpu, dir.path());
    set_lora_b(&gpu, &plain);
    set_lora_b(&gpu, &off);
    let built = off
        .enable_fused_projections_with(&gpu.engine, FusedMode::Force, FuseChoice::OFF, 64)
        .expect("enable");
    assert_eq!(built, (0, 0), "both switches off build no group");
    assert!(groups(&off).is_empty());

    let (loss_a, grads_a) = loss_and_grads(&gpu, &plain);
    let (loss_b, grads_b) = loss_and_grads(&gpu, &off);
    assert_eq!(loss_a.to_bits(), loss_b.to_bits(), "loss");
    for ((name, a), (_, b)) in grads_a.iter().zip(&grads_b) {
        let same = a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(same, "{name}: gradient differs");
    }
    let before = base_weights(&gpu, &plain);
    let after = base_weights(&gpu, &off);
    assert_eq!(before.len(), after.len());
    for ((name, a), (_, b)) in before.iter().zip(&after) {
        assert_eq!(a, b, "{name}: the weight changed with the switches off");
    }
}

#[test]
#[ignore = "needs a GPU"]
fn each_switch_builds_only_its_own_groups() {
    let dir = TempDir::new("fused-switch-one");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let mut qkv_only = build(&gpu, dir.path());
    let choice = FuseChoice {
        qkv: FuseSwitch::On,
        gate_up: FuseSwitch::Off,
    };
    let built = qkv_only
        .enable_fused_projections_with(&gpu.engine, FusedMode::Force, choice, 64)
        .expect("enable");
    assert_eq!(built, (LAYERS, 0));
    let mut gate_up_only = build(&gpu, dir.path());
    let choice = FuseChoice {
        qkv: FuseSwitch::Off,
        gate_up: FuseSwitch::On,
    };
    let built = gate_up_only
        .enable_fused_projections_with(&gpu.engine, FusedMode::Force, choice, 64)
        .expect("enable");
    assert_eq!(built, (0, LAYERS));
}

#[test]
#[ignore = "needs a GPU"]
fn the_fused_weight_equals_the_concatenation_of_the_originals() {
    let dir = TempDir::new("fused-concat");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let reference = build(&gpu, dir.path());
    let mut fused = build(&gpu, dir.path());
    set_lora_b(&gpu, &reference);
    set_lora_b(&gpu, &fused);
    fused
        .enable_fused_projections(&gpu.engine, FusedMode::Force)
        .expect("enable");
    let originals = base_weights(&gpu, &reference);

    let mut checked = 0;
    for (i, layer) in fused.layers.iter().enumerate() {
        let mut sets: Vec<(Arc<FusedProjections>, Vec<String>)> = Vec::new();
        if let Lfm2LayerOp::Attention(attn) = &layer.op {
            if let Some(g) = attn.fused.get() {
                sets.push((g, ["q", "k", "v"].map(|n| format!("{i}.{n}")).to_vec()));
            }
        }
        if let Some(g) = layer.mlp.fused.get() {
            sets.push((g, ["gate", "up"].map(|n| format!("{i}.{n}")).to_vec()));
        }
        for (group, names) in sets {
            let halves = group.weight_halves(&gpu.ctx).expect("weight readback");
            let rows = group.input_size();
            let total = halves.len() / rows;
            assert_eq!(halves.len(), rows * total);
            let mut offset = 0;
            for (index, (name, width)) in names.iter().zip(group.widths()).enumerate() {
                let (_, want) = originals
                    .iter()
                    .find(|(n, _)| n == name)
                    .unwrap_or_else(|| panic!("no original {name}"));
                assert_eq!(want.len(), rows * width, "{name}: original size");
                let want = bits(want);
                for r in 0..rows {
                    let got = &halves[r * total + offset..r * total + offset + width];
                    assert_eq!(got, &want[r * width..(r + 1) * width], "{name}, row {r}");
                }
                // The copy taken back out of the group is the original too.
                let back = group.projection(&gpu.ctx, index).expect("projection");
                assert_eq!(bits(&host(&gpu, &back)), want, "{name}: restored weight");
                offset += width;
                checked += 1;
            }
        }
    }
    assert_eq!(checked, LAYERS * 5, "q, k, v, gate and up of every block");
}

#[test]
#[ignore = "needs a GPU"]
fn a_fused_model_decodes_with_a_cache_like_the_unfused_one() {
    let dir = TempDir::new("fused-cache");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let reference = build(&gpu, dir.path());
    let mut fused = build(&gpu, dir.path());
    set_lora_b(&gpu, &reference);
    set_lora_b(&gpu, &fused);
    fused
        .enable_fused_projections(&gpu.engine, FusedMode::Force)
        .expect("enable");
    let seq = pattern_sequence();
    let run = |model: &Lfm2Model| {
        let mut cache = model
            .new_kv_cache(seq.len())
            .expect("an attention-only model");
        let logits = model.forward(&gpu.engine, &seq[..8], 0, Some(&mut cache), None);
        let first = host(&gpu, &logits);
        let logits = model.forward(&gpu.engine, &seq[8..9], 8, Some(&mut cache), None);
        (first, host(&gpu, &logits))
    };
    // A pass with a cache cannot use the group, so it runs the layers' own
    // code on weights copied back out of it: the same values, bit for bit.
    let (a1, a2) = run(&reference);
    let (b1, b2) = run(&fused);
    let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    assert!(same(&a1, &b1), "prefill logits differ");
    assert!(same(&a2, &b2), "decode logits differ");
}

#[test]
#[ignore = "needs a GPU"]
fn save_and_merge_write_the_original_layout_from_a_fused_model() {
    let dir = TempDir::new("fused-merge");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let reference = build(&gpu, dir.path());
    let mut fused = build(&gpu, dir.path());
    set_lora_b(&gpu, &reference);
    set_lora_b(&gpu, &fused);
    fused
        .enable_fused_projections(&gpu.engine, FusedMode::Force)
        .expect("enable");

    let meta = LoraMetadata {
        rank: 4,
        alpha: 8.0,
        target_modules: Vec::new(),
        trained_modules: None,
    };
    let save = |model: &Lfm2Model, file: &str| {
        let path = dir.path().join(file);
        let named = model.named_lora_params();
        pollster::block_on(save_lora_checkpoint(&gpu.ctx, &path, &named, Some(&meta)))
            .expect("save adapter");
        load_lora_checkpoint(&path).expect("load adapter")
    };
    let ckpt_ref = save(&reference, "ref.safetensors");
    let ckpt_fused = save(&fused, "fused.safetensors");
    assert_eq!(ckpt_ref.tensors.len(), LAYERS * 7 * 2);
    assert_eq!(ckpt_fused.tensors.len(), ckpt_ref.tensors.len());
    for (name, (shape, values)) in &ckpt_ref.tensors {
        let (fshape, fvalues) = &ckpt_fused.tensors[name];
        assert_eq!(shape, fshape, "{name}: shape");
        assert_eq!(values, fvalues, "{name}: values");
    }

    let out_ref = TempDir::new("fused-merge-ref");
    let out_fused = TempDir::new("fused-merge-fused");
    let a = merge_adapter(dir.path(), &ckpt_ref, out_ref.path()).expect("merge reference");
    let b = merge_adapter(dir.path(), &ckpt_fused, out_fused.path()).expect("merge fused");
    assert_eq!((a.merged, a.copied), (b.merged, b.copied));
    assert_eq!(a.merged, LAYERS * 7, "seven adapted projections per block");
    let bytes_a = std::fs::read(out_ref.path().join(MERGED_WEIGHTS_FILE)).expect("read");
    let bytes_b = std::fs::read(out_fused.path().join(MERGED_WEIGHTS_FILE)).expect("read");
    assert!(bytes_a == bytes_b, "the merged files differ");
}
