//! Inference under `no_grad` on the tiny model: the logits are bit-identical
//! to a recorded forward (full recompute and KV cache, with and without a
//! trained LoRA adapter), no activation outlives the forward, and training
//! afterwards on the same thread matches training without it.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test no_grad_inference -- --ignored --test-threads=1
//! ```

#![allow(clippy::expect_used)]

mod common;

use common::*;
use ferrotensor_autograd::{no_grad, Tensor};
use ferrotensor_backend::memory::wgpu_buffer_size;
use ferrotensor_graph::lfm::Lfm2Model;
use ferrotensor_graph::model_builder::Lfm2ModelBuilder;
use ferrotensor_loader::{ModelLoader, WeightsContext};
use std::io::Write as _;

fn build_base_model(gpu: &Gpu, dir: &std::path::Path) -> Lfm2Model {
    let config = ModelLoader::load_config(dir.join("config.json")).expect("config");
    let mut weights = WeightsContext::new();
    weights.load_model_dir(dir).expect("weights");
    Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &weights, config)
        .build()
        .expect("build")
}

fn bits(gpu: &Gpu, t: &Tensor) -> Vec<u32> {
    host(gpu, t).iter().map(|v| v.to_bits()).collect()
}

/// Logits of the full sequence, then of a prefill and one decode step
/// against a KV cache.
fn logits_full_and_cached(gpu: &Gpu, model: &Lfm2Model, seq: &[u32]) -> [Vec<u32>; 3] {
    let (prefix, last) = seq.split_at(seq.len() - 1);
    let full = model.forward(&gpu.engine, seq, 0, None, None);
    let mut cache = model.new_kv_cache(seq.len());
    assert!(cache.is_some(), "the tiny model is all attention");
    let prefill = model.forward(&gpu.engine, prefix, 0, cache.as_mut(), None);
    let step = model.forward(&gpu.engine, last, prefix.len(), cache.as_mut(), None);
    [bits(gpu, &full), bits(gpu, &prefill), bits(gpu, &step)]
}

#[test]
#[ignore = "needs a GPU"]
fn no_grad_logits_match_recorded_logits_bit_for_bit() {
    let dir = TempDir::new("no_grad_logits");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let seq = pattern_sequence();

    let base = build_base_model(&gpu, dir.path());
    // One step makes the zero-initialised LoRA B non-zero.
    let mut lora_trainer = trainer(&gpu, build_model(&gpu, dir.path()), 1e-2);
    let (_, params) = sorted_lora_params(&lora_trainer.model);
    let stats = lora_trainer
        .train_step(&seq[..seq.len() - 1], &seq[1..], &params)
        .expect("step");
    assert!(stats.optimizer_step);

    for (name, model) in [("base", &base), ("lora", &lora_trainer.model)] {
        let recorded = logits_full_and_cached(&gpu, model, &seq);
        let unrecorded = no_grad(|| logits_full_and_cached(&gpu, model, &seq));
        for (path, (r, u)) in ["full", "prefill", "decode"]
            .iter()
            .zip(recorded.iter().zip(&unrecorded))
        {
            assert_eq!(r, u, "{name} model, {path} logits");
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn no_grad_forward_keeps_only_the_logits_alive() {
    let dir = TempDir::new("no_grad_memory");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    let seq = pattern_sequence();
    let memory = gpu.ctx.memory();

    let measure = |forward: &dyn Fn() -> Tensor| {
        let before = memory.live_bytes();
        memory.reset_peak();
        let logits = forward();
        let live = memory.live_bytes() - before;
        let peak = memory.peak_live_bytes() - before;
        drop(logits);
        (live, peak)
    };
    let (recorded_live, recorded_peak) =
        measure(&|| model.forward(&gpu.engine, &seq, 0, None, None));
    let (unrecorded_live, unrecorded_peak) =
        measure(&|| no_grad(|| model.forward(&gpu.engine, &seq, 0, None, None)));

    // Written to the stream directly so the numbers show in the CI log.
    writeln!(
        std::io::stderr(),
        "tiny model forward, {} tokens: live after forward {recorded_live} B recorded, \
         {unrecorded_live} B under no_grad; peak {recorded_peak} B recorded, \
         {unrecorded_peak} B under no_grad",
        seq.len()
    )
    .expect("write to stderr");
    let logits_bytes = wgpu_buffer_size((seq.len() * VOCAB * 4) as u64);
    assert_eq!(unrecorded_live, logits_bytes, "only the logits stay live");
    assert!(unrecorded_live < recorded_live);
    assert!(unrecorded_peak < recorded_peak);
}

#[test]
#[ignore = "needs a GPU"]
fn training_after_no_grad_matches_training_without_it() {
    let dir = TempDir::new("no_grad_train");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let seq = pattern_sequence();
    let (input, target) = (&seq[..seq.len() - 1], &seq[1..]);

    let train_once = |model: Lfm2Model| {
        let mut t = trainer(&gpu, model, 1e-2);
        let (_, params) = sorted_lora_params(&t.model);
        let before: Vec<Vec<u32>> = params.iter().map(|p| bits(&gpu, p)).collect();
        let loss = t.train_step(input, target, &params).expect("step").loss;
        let after: Vec<Vec<u32>> = params.iter().map(|p| bits(&gpu, p)).collect();
        assert_ne!(before, after, "the step updated the adapters");
        (loss.to_bits(), after)
    };

    let reference = train_once(build_model(&gpu, dir.path()));
    let model = build_model(&gpu, dir.path());
    let logits = no_grad(|| model.forward(&gpu.engine, input, 0, None, None));
    assert!(logits.creator.is_none());
    let after_no_grad = train_once(model);

    assert_eq!(after_no_grad.0, reference.0, "loss");
    assert_eq!(after_no_grad.1, reference.1, "LoRA weights after one step");
}
