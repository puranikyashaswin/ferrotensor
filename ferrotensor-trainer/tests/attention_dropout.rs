//! Attention dropout in the decoder and the trainer: masks follow
//! `(run seed, step, layer)`, evaluation and generation apply none, and a
//! dropout probability of 0 leaves every output bit-identical.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test attention_dropout -- --ignored --test-threads=1
//! ```

mod common;

use common::*;
use ferrotensor_autograd::{AdamW, Tensor};
use ferrotensor_backend::UnifiedTensor;
use ferrotensor_graph::dropout::{derive_seed, layer_seed};
use ferrotensor_graph::lfm::{Lfm2LayerOp, Lfm2Model};
use ferrotensor_trainer::Trainer;
use std::path::Path;

const P: f32 = 0.5;

/// Set `attention_dropout` in the tiny model's `config.json`.
fn set_attention_dropout(dir: &Path, p: f32) {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).expect("read config");
    let mut cfg: serde_json::Value = serde_json::from_str(&text).expect("parse config");
    cfg["attention_dropout"] = serde_json::json!(p);
    std::fs::write(&path, cfg.to_string()).expect("write config");
}

/// The same weights twice: without attention dropout and with `P`.
fn models(gpu: &Gpu, dir: &Path) -> (Lfm2Model, Lfm2Model) {
    write_tiny_llama(dir);
    let plain = build_model(gpu, dir);
    set_attention_dropout(dir, P);
    let dropped = build_model(gpu, dir);
    for layer in &dropped.layers {
        match &layer.op {
            Lfm2LayerOp::Attention(a) => assert_eq!(a.dropout_prob, P),
            Lfm2LayerOp::Conv(_) => panic!("tiny Llama has no conv blocks"),
        }
    }
    (plain, dropped)
}

fn ids() -> Vec<u32> {
    pattern_sequence()[..16].to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn logits(gpu: &Gpu, model: &Lfm2Model, seed: Option<u64>) -> Vec<u32> {
    let out = model.forward_with_dropout(&gpu.engine, &ids(), 0, None, None, seed);
    bits(&host(gpu, &out))
}

/// The model's forward written out by hand, with layer `i` seeded by
/// `seed_of(i)`.
fn manual_logits(gpu: &Gpu, model: &Lfm2Model, seed_of: impl Fn(usize) -> u64) -> Vec<u32> {
    let ids = ids();
    let table = host(gpu, &model.embed_tokens);
    let data: Vec<f32> = ids
        .iter()
        .flat_map(|&id| table[id as usize * HIDDEN..(id as usize + 1) * HIDDEN].to_vec())
        .collect();
    let mut x = Tensor::frozen(
        gpu.ctx.clone(),
        UnifiedTensor::new(&gpu.ctx, &data, &[1, ids.len(), HIDDEN]),
    );
    assert!(model.embedding_norm.is_none());
    for (i, layer) in model.layers.iter().enumerate() {
        x = layer.forward_with_dropout(&gpu.engine, &x, 0, None, None, Some(seed_of(i)));
    }
    let x = model.norm_f.forward(&gpu.engine, &x);
    bits(&host(gpu, &model.lm_head.forward(&gpu.engine, &x)))
}

#[test]
#[ignore = "needs a GPU"]
fn dropout_masks_are_a_pure_function_of_the_step_seed() {
    let dir = TempDir::new("dropout-seed");
    let gpu = gpu();
    let (_, dropped) = models(&gpu, dir.path());
    let step0 = derive_seed(&[1, 0, 0, 0]);
    let step1 = derive_seed(&[1, 0, 1, 0]);

    let a = logits(&gpu, &dropped, Some(step0));
    assert_eq!(
        a,
        logits(&gpu, &dropped, Some(step0)),
        "same seed, same masks"
    );
    assert_ne!(a, logits(&gpu, &dropped, Some(step1)), "steps share a mask");
    assert_ne!(a, logits(&gpu, &dropped, None), "dropout had no effect");
}

#[test]
#[ignore = "needs a GPU"]
fn each_layer_draws_its_own_mask() {
    let dir = TempDir::new("dropout-layers");
    let gpu = gpu();
    let (_, dropped) = models(&gpu, dir.path());
    let step = derive_seed(&[3, 0, 5, 0]);

    // The model seeds layer i with layer_seed(step, i) ...
    let model = logits(&gpu, &dropped, Some(step));
    assert_eq!(
        model,
        manual_logits(&gpu, &dropped, |i| layer_seed(step, i))
    );
    // ... and that is not one seed shared by every layer.
    assert_ne!(
        model,
        manual_logits(&gpu, &dropped, |_| layer_seed(step, 0))
    );

    // One block, two layer seeds: two masks.
    let x = Tensor::frozen(
        gpu.ctx.clone(),
        UnifiedTensor::new(
            &gpu.ctx,
            &pseudo_random(16 * HIDDEN, 11, 1.0),
            &[1, 16, HIDDEN],
        ),
    );
    let block = &dropped.layers[0];
    let run = |layer| {
        let out = block.forward_with_dropout(
            &gpu.engine,
            &x,
            0,
            None,
            None,
            Some(layer_seed(step, layer)),
        );
        bits(&host(&gpu, &out))
    };
    assert_ne!(run(0), run(1), "layers share a mask");
}

#[test]
#[ignore = "needs a GPU"]
fn zero_dropout_is_bit_identical_with_or_without_a_seed() {
    let dir = TempDir::new("dropout-zero");
    let gpu = gpu();
    let (plain, _) = models(&gpu, dir.path());
    let eval = bits(&host(
        &gpu,
        &plain.forward(&gpu.engine, &ids(), 0, None, None),
    ));
    for seed in [0, 7, derive_seed(&[1, 0, 0, 0])] {
        assert_eq!(eval, logits(&gpu, &plain, Some(seed)), "seed {seed}");
    }

    let seq = ids();
    let rows: Vec<&[u32]> = vec![&seq[..8], &seq[8..]];
    let batch = bits(&host(&gpu, &plain.forward_batch(&gpu.engine, &rows)));
    let seeded = plain.forward_batch_with_dropout(&gpu.engine, &rows, Some(derive_seed(&[2])));
    assert_eq!(batch, bits(&host(&gpu, &seeded)));
}

#[test]
#[ignore = "needs a GPU"]
fn evaluation_and_generation_apply_no_dropout() {
    let dir = TempDir::new("dropout-eval");
    let gpu = gpu();
    let (plain, dropped) = models(&gpu, dir.path());
    let seq = ids();

    // Full-sequence evaluation.
    let full = |m: &Lfm2Model| bits(&host(&gpu, &m.forward(&gpu.engine, &seq, 0, None, None)));
    assert_eq!(full(&plain), full(&dropped));
    let rows: Vec<&[u32]> = vec![&seq[..8], &seq[8..]];
    let batch = |m: &Lfm2Model| bits(&host(&gpu, &m.forward_batch(&gpu.engine, &rows)));
    assert_eq!(batch(&plain), batch(&dropped));

    // Generation: prefill into a KV cache, then one decode step.
    let generate = |m: &Lfm2Model| {
        let mut cache = m.new_kv_cache(32).expect("all-attention model");
        let prefill = m.forward(&gpu.engine, &seq[..15], 0, Some(&mut cache), None);
        let step = m.forward(&gpu.engine, &seq[15..], 15, Some(&mut cache), None);
        (bits(&host(&gpu, &prefill)), bits(&host(&gpu, &step)))
    };
    assert_eq!(generate(&plain), generate(&dropped));
}

fn zero_lr_trainer(gpu: &Gpu, model: Lfm2Model) -> Trainer {
    let mut opt = AdamW::new(gpu.engine.clone(), 0.0);
    opt.weight_decay = 0.0;
    Trainer::new(model, Box::new(opt), gpu.ctx.clone(), gpu.engine.clone())
}

#[test]
#[ignore = "needs a GPU"]
fn trainer_masks_change_per_step_and_resume_identically() {
    let dir = TempDir::new("dropout-trainer");
    let gpu = gpu();
    let seq = pattern_sequence();
    let (input, target) = (&seq[..seq.len() - 1], &seq[1..]);
    let (plain, dropped) = models(&gpu, dir.path());

    // With lr 0 the weights never change, so a step's loss depends only on
    // its dropout masks.
    let (_, params) = sorted_lora_params(&dropped);
    let mut a = zero_lr_trainer(&gpu, dropped).with_dropout_seed(9);
    let a_losses: Vec<u32> = (0..4)
        .map(|_| {
            a.train_step(input, target, &params)
                .expect("step")
                .loss
                .to_bits()
        })
        .collect();
    let mut distinct = a_losses.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), 4, "steps share a mask: {a_losses:?}");

    // Resuming at step 2 draws the masks of steps 2 and 3 again.
    let model = build_model(&gpu, dir.path());
    let (_, params_b) = sorted_lora_params(&model);
    let mut b = zero_lr_trainer(&gpu, model).with_dropout_seed(9);
    b.set_optimizer_steps(2);
    let b_losses: Vec<u32> = (0..2)
        .map(|_| {
            b.train_step(input, target, &params_b)
                .expect("step")
                .loss
                .to_bits()
        })
        .collect();
    assert_eq!(b_losses, a_losses[2..]);

    // Another run seed, other masks.
    let model = build_model(&gpu, dir.path());
    let (_, params_c) = sorted_lora_params(&model);
    let mut c = zero_lr_trainer(&gpu, model).with_dropout_seed(10);
    let c0 = c.train_step(input, target, &params_c).expect("step").loss;
    assert_ne!(c0.to_bits(), a_losses[0]);

    // Without dropout the seed changes nothing.
    let (_, params_p) = sorted_lora_params(&plain);
    let mut p = zero_lr_trainer(&gpu, plain).with_dropout_seed(9);
    let p_losses: Vec<u32> = (0..2)
        .map(|_| {
            p.train_step(input, target, &params_p)
                .expect("step")
                .loss
                .to_bits()
        })
        .collect();
    assert_eq!(p_losses[0], p_losses[1]);
}
