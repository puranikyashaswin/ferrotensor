//! GPU memory accounting on the tiny model: the report's categories add up,
//! the frozen weights come out at their exact size, turning the report on
//! changes nothing about training, and dropped buffers are counted as
//! awaiting the GPU until the context waits for it.
//!
//! Needs a GPU, so `#[ignore]`d by default:
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test gpu_memory -- --ignored --test-threads=1
//! ```

mod common;

use common::*;
use ferrotensor_backend::memory::{wgpu_buffer_size, MemorySnapshot};
use ferrotensor_backend::pool::DEFAULT_MAX_POOLED_BYTES;
use ferrotensor_backend::{MemoryCategory, UnifiedTensor};
use ferrotensor_trainer::memory::{tag_trainable, MemoryRecorder};
use std::io::Write as _;

/// Bytes wgpu allocates for `n` values stored as f32.
fn f32_bytes(n: usize) -> u64 {
    wgpu_buffer_size(n as u64 * 4)
}

/// Bytes wgpu allocates for `n` values stored as f16: packed two per word,
/// padded to whole 32-bit words (`UnifiedTensor::from_f32_as_f16`).
fn f16_bytes(n: usize) -> u64 {
    wgpu_buffer_size((n as u64 * 2).next_multiple_of(4).max(4))
}

/// Frozen weights of `write_tiny_llama` built with f16 weights:
/// `(total, of which f16)`. Linear layers (and the tied head) are f16; the
/// embedding and the norms stay f32.
fn tiny_base_weight_bytes() -> (u64, u64) {
    let kv = HIDDEN / HEADS * KV_HEADS;
    let linears = [
        HIDDEN * HIDDEN, // q
        kv * HIDDEN,     // k
        kv * HIDDEN,     // v
        HIDDEN * HIDDEN, // o
        FFN * HIDDEN,    // gate
        FFN * HIDDEN,    // up
        HIDDEN * FFN,    // down
    ];
    let f16: u64 = LAYERS as u64 * linears.iter().map(|&n| f16_bytes(n)).sum::<u64>()
        + f16_bytes(VOCAB * HIDDEN);
    let f32 = f32_bytes(VOCAB * HIDDEN) // embedding
        + LAYERS as u64 * 2 * f32_bytes(HIDDEN) // two norms per layer
        + f32_bytes(HIDDEN); // final norm
    (f16 + f32, f16)
}

/// LoRA A `[in, r]` and B `[r, out]` on q/k/v/o, f32.
fn tiny_lora_bytes() -> u64 {
    let r = lora_config().rank;
    let kv = HIDDEN / HEADS * KV_HEADS;
    let per_layer: u64 = [
        (HIDDEN, HIDDEN),
        (HIDDEN, kv),
        (HIDDEN, kv),
        (HIDDEN, HIDDEN),
    ]
    .iter()
    .map(|&(i, o)| f32_bytes(i * r) + f32_bytes(r * o))
    .sum();
    LAYERS as u64 * per_layer
}

fn assert_consistent(s: &MemorySnapshot) {
    let sum: u64 = s.categories.iter().map(|c| c.allocated_bytes).sum();
    assert_eq!(sum, s.live_bytes, "categories must add up to the total");
    assert!(s.logical_bytes <= s.live_bytes);
}

fn print(text: &str) {
    // Written to the stream directly so it shows in the test log.
    let _ = std::io::stderr().write_all(text.as_bytes());
}

#[test]
#[ignore = "needs a GPU"]
fn memory_report_adds_up_and_does_not_change_training() {
    let dir = TempDir::new("memory");
    write_tiny_llama(dir.path());
    let seq = pattern_sequence();
    let (input, target) = (&seq[..seq.len() - 1], &seq[1..]);

    // With the report.
    let gpu_on = gpu();
    let mut recorder = MemoryRecorder::start(gpu_on.ctx.clone());
    let model = {
        let _tag = gpu_on.ctx.memory().scope(MemoryCategory::BaseWeights);
        build_model_with(&gpu_on, dir.path(), true)
    };
    let (_, params) = sorted_lora_params(&model);
    tag_trainable(&params);
    let load = recorder.record("after model load").clone();
    print(&load.to_table());

    let (base, base_f16) = tiny_base_weight_bytes();
    let lora = tiny_lora_bytes();
    for s in [&load.peak, &load.live] {
        assert_consistent(s);
        let b = s.category(MemoryCategory::BaseWeights);
        assert_eq!(b.allocated_bytes, base, "frozen weights");
        assert_eq!(b.f16_bytes, base_f16, "frozen f16 weights");
        assert_eq!(s.category(MemoryCategory::Trainable).allocated_bytes, lora);
    }
    assert_eq!(load.live.live_bytes, base + lora);
    assert!(!load.peak.largest.is_empty());

    let mut trainer_on = trainer(&gpu_on, model, 1e-2);
    let mut losses_on = Vec::new();
    for step in 1..=2 {
        let stats = trainer_on.train_step(input, target, &params).expect("step");
        assert!(stats.optimizer_step);
        losses_on.push(stats.loss);
        let r = recorder.record(format!("optimizer step {step}")).clone();
        print(&r.to_table());
        for s in [&r.peak, &r.overall_peak, &r.live] {
            assert_consistent(s);
            assert!(s.live_bytes > 0);
        }
        assert!(r.peak.live_bytes >= r.live.live_bytes);
        assert!(r.overall_peak.live_bytes >= r.peak.live_bytes);
        assert!(r.peak_held_bytes >= r.peak.live_bytes);
        assert!(r.overall_peak_held_bytes >= r.peak_held_bytes);
        assert!(r.overall_peak_held_bytes >= r.overall_peak.live_bytes);
        // Every dispatch drops a uniform buffer before the GPU runs it.
        assert!(
            r.peak_awaiting_gpu_bytes > 0,
            "dispatch buffers not counted"
        );
        assert!(r.overall_peak_awaiting_gpu_bytes >= r.peak_awaiting_gpu_bytes);
        assert!(r.peak_total_bytes >= r.peak_held_bytes);
        assert!(
            r.peak_total_bytes
                >= r.peak.live_bytes + r.peak.pool_idle_bytes + r.peak.awaiting_gpu_bytes
        );
        assert!(r.overall_peak_total_bytes >= r.peak_total_bytes);
        assert_eq!(r.live.awaiting_gpu_bytes, 0, "record waits for the GPU");
        assert_eq!(r.pool_cap_bytes, DEFAULT_MAX_POOLED_BYTES);
        for c in [
            MemoryCategory::Activations,
            MemoryCategory::Loss,
            MemoryCategory::BackwardScratch,
        ] {
            assert!(r.peak.category(c).allocated_bytes > 0, "{c:?} at the peak");
        }
        // Between steps only the weights, adapters and AdamW moments
        // (m and v, f32, one pair per adapter tensor) stay live.
        let live = &r.live;
        assert_eq!(
            live.category(MemoryCategory::BaseWeights).allocated_bytes,
            base
        );
        assert_eq!(
            live.category(MemoryCategory::Trainable).allocated_bytes,
            lora
        );
        assert_eq!(
            live.category(MemoryCategory::OptimizerState)
                .allocated_bytes,
            2 * lora
        );
        assert_eq!(
            live.live_bytes,
            base + 3 * lora,
            "something else stayed live"
        );
    }
    let peak = recorder.reports[1].peak.clone();
    let grads = peak.category(MemoryCategory::Gradients).allocated_bytes;
    assert!(
        grads <= lora,
        "parameter gradients {grads} exceed the adapters {lora}"
    );
    print(&format!(
        "{}\n",
        serde_json::to_string(&recorder.to_json()).expect("json")
    ));
    recorder.stop();

    // Without it.
    let gpu_off = gpu();
    let model = build_model_with(&gpu_off, dir.path(), true);
    let (_, params) = sorted_lora_params(&model);
    let mut trainer_off = trainer(&gpu_off, model, 1e-2);
    assert!(!gpu_off.ctx.memory().is_detailed());
    let losses_off: Vec<f32> = (0..2)
        .map(|_| {
            trainer_off
                .train_step(input, target, &params)
                .expect("step")
                .loss
        })
        .collect();

    let bits = |l: &[f32]| l.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&losses_on),
        bits(&losses_off),
        "the memory report changed the loss"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn dropped_buffers_await_the_gpu_until_synchronize() {
    let gpu = gpu();
    let ctx = &gpu.ctx;
    let mem = ctx.memory();
    let shape = [1024];
    let bytes = f32_bytes(1024);
    // Room in the pool for exactly one buffer of this size.
    ctx.pool.clear();
    ctx.pool.set_max_pooled_bytes(bytes);
    ctx.synchronize();
    assert_eq!(mem.awaiting_gpu_bytes(), 0);
    mem.reset_peak();

    let src = UnifiedTensor::new(ctx, &[1.0; 1024], &shape);
    let kept = UnifiedTensor::empty(ctx, &shape);
    let past_cap = UnifiedTensor::empty(ctx, &shape);
    // Work in flight that reads both buffers.
    gpu.engine.copy_tensor(ctx, &src, &kept, 0);
    gpu.engine.copy_tensor(ctx, &src, &past_cap, 0);
    let (live, idle) = (mem.live_bytes(), mem.pool_idle_bytes());

    drop(kept);
    assert_eq!(mem.pool_idle_bytes(), idle + bytes, "kept by the pool");
    assert_eq!(mem.awaiting_gpu_bytes(), 0);
    drop(past_cap);
    assert_eq!(mem.pool_idle_bytes(), idle + bytes);
    assert_eq!(mem.awaiting_gpu_bytes(), bytes, "dropped past the cap");
    assert_eq!(mem.live_bytes(), live - 2 * bytes);

    // A dispatch's uniform buffer is counted from the start.
    gpu.engine.gradient_scale(ctx, &src, 1.0);
    assert_eq!(mem.awaiting_gpu_bytes(), bytes + 4);

    ctx.flush();
    assert_eq!(mem.awaiting_gpu_bytes(), bytes + 4, "flush does not poll");
    ctx.synchronize();
    assert_eq!(
        mem.awaiting_gpu_bytes(),
        0,
        "synchronize lets wgpu free them"
    );
    assert_eq!(mem.peak_awaiting_gpu_bytes(), bytes + 4);
    // Live + idle + awaiting stayed at `live + idle` through the drops,
    // then the uniform added 4 bytes.
    assert_eq!(mem.peak_total_bytes(), live + idle + 4);

    // The readback copy is staging while it runs, then awaits the GPU.
    let staging_before = mem.current().category(MemoryCategory::Staging).buffers;
    let values = pollster::block_on(src.to_vec(ctx));
    assert_eq!(values.len(), 1024);
    assert_eq!(
        mem.current().category(MemoryCategory::Staging).buffers,
        staging_before
    );
    assert_eq!(mem.awaiting_gpu_bytes(), bytes);
    ctx.synchronize();
    assert_eq!(mem.awaiting_gpu_bytes(), 0);

    let s = mem.current();
    assert_consistent(&s);
    assert_eq!(s.live_bytes, mem.live_bytes());
}
