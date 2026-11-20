//! Buffer creations stay flat when the padded length changes on almost every
//! micro-batch, as exact padding (`length_bucket` 1) makes it.
//!
//! The pool serves a request from the size class of its byte size, so a
//! length it has not seen before usually lands on a buffer an earlier length
//! left behind. The sizes of one step depend only on its length, so once the
//! pool holds every class a length range needs, no step creates a buffer.
//!
//! Needs a GPU, so `#[ignore]`d by default:
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test pool_size_classes -- --ignored --test-threads=1
//! ```

#![allow(clippy::expect_used)]

mod common;

use common::*;
use std::io::Write as _;

/// A training step on a sequence of `len` tokens of the cyclic pattern.
fn step(
    trainer: &mut ferrotensor_trainer::Trainer,
    params: &[ferrotensor_autograd::Tensor],
    len: usize,
) -> f32 {
    let seq: Vec<u32> = (0..=len).map(|i| ((i * 7 + 3) % VOCAB) as u32).collect();
    trainer
        .train_step(&seq[..len], &seq[1..], params)
        .expect("step")
        .loss
}

#[test]
#[ignore = "needs a GPU"]
fn buffer_creations_stay_flat_as_the_length_changes_every_step() {
    let dir = TempDir::new("pool-size-classes");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    let (_, params) = sorted_lora_params(&model);
    let mut trainer = trainer(&gpu, model, 1e-3).with_length_bucket(1);

    let created_over = |trainer: &mut ferrotensor_trainer::Trainer, lengths: &[usize]| {
        let before = gpu.ctx.pool.buffers_created();
        for &len in lengths {
            let loss = step(trainer, &params, len);
            assert!(loss.is_finite(), "length {len}: loss {loss}");
        }
        gpu.ctx.synchronize();
        gpu.ctx.pool.buffers_created() - before
    };

    // Warm up on the even lengths, then train on the odd ones, which the pool
    // has not seen, in an order that changes the length on every step.
    let warm: Vec<usize> = (100..=160).step_by(2).collect();
    let odd: Vec<usize> = (0..30).map(|i| 101 + 2 * ((i * 11) % 30)).collect();
    let created_warm = created_over(&mut trainer, &warm);
    let created_odd = created_over(&mut trainer, &odd);
    let created_again = created_over(&mut trainer, &odd);
    let _ = writeln!(
        std::io::stderr(),
        "buffers created: {created_warm} over {} warm-up steps, {created_odd} over {} steps of \
         lengths not seen before, {created_again} over the same steps again",
        warm.len(),
        odd.len(),
    );

    assert!(created_warm > 0, "the warm-up created nothing");
    assert!(
        created_odd * 10 <= created_warm,
        "{created_odd} buffers over {} new lengths after {created_warm} while warming up",
        odd.len()
    );
    assert_eq!(created_again, 0, "the same lengths again");
}
