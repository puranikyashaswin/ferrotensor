//! Pool overflow (buffers freed beyond the pool's cap stay reusable until
//! the GPU has finished with them) changes no training result, creates fewer
//! buffers and lowers the peak of live + pool idle + awaiting-GPU bytes.
//!
//! The tiny model never fills the default 512 MiB cap, so both runs set the
//! cap to 0: every buffer the pool keeps is then kept by overflow.
//!
//! Needs a GPU, so `#[ignore]`d by default:
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test pool_overflow -- --ignored --test-threads=1
//! ```

mod common;

use common::*;
use std::io::Write as _;

struct Run {
    loss_bits: Vec<u32>,
    peak_total_bytes: u64,
    buffers_created: u64,
    buffers_reused: u64,
}

fn train(dir: &std::path::Path, overflow: bool) -> Run {
    let seq = pattern_sequence();
    let (input, target) = (&seq[..seq.len() - 1], &seq[1..]);
    let gpu = gpu();
    gpu.ctx.pool.set_overflow(overflow);
    let model = build_model_with(&gpu, dir, true);
    let (_, params) = sorted_lora_params(&model);
    gpu.ctx.pool.set_max_pooled_bytes(0);
    gpu.ctx.synchronize();
    let mem = gpu.ctx.memory();
    mem.reset_peak();
    let (created, reused) = (
        gpu.ctx.pool.buffers_created(),
        gpu.ctx.pool.buffers_reused(),
    );

    let mut run_trainer = trainer(&gpu, model, 1e-2);
    let loss_bits = (0..3)
        .map(|_| {
            run_trainer
                .train_step(input, target, &params)
                .expect("step")
                .loss
                .to_bits()
        })
        .collect();
    gpu.ctx.synchronize();
    assert_eq!(mem.awaiting_gpu_bytes(), 0);
    assert_eq!(gpu.ctx.pool.pooled_bytes(), 0, "trimmed to the cap");
    Run {
        loss_bits,
        peak_total_bytes: mem.peak_total_bytes(),
        buffers_created: gpu.ctx.pool.buffers_created() - created,
        buffers_reused: gpu.ctx.pool.buffers_reused() - reused,
    }
}

#[test]
#[ignore = "needs a GPU"]
fn pool_overflow_keeps_losses_and_lowers_the_peak() {
    let dir = TempDir::new("pool-overflow");
    write_tiny_llama(dir.path());
    let off = train(dir.path(), false);
    let on = train(dir.path(), true);
    let _ = writeln!(
        std::io::stderr(),
        "pool overflow off: peak total {} B, {} buffers created, {} reused\n\
         pool overflow on:  peak total {} B, {} buffers created, {} reused",
        off.peak_total_bytes,
        off.buffers_created,
        off.buffers_reused,
        on.peak_total_bytes,
        on.buffers_created,
        on.buffers_reused,
    );

    assert_eq!(on.loss_bits, off.loss_bits, "overflow changed the loss");
    assert!(on.buffers_reused > 0, "overflow reused nothing");
    assert!(on.buffers_created < off.buffers_created);
    assert!(
        on.peak_total_bytes < off.peak_total_bytes,
        "peak total {} with overflow, {} without",
        on.peak_total_bytes,
        off.peak_total_bytes
    );
}
