//! Training steps after the first reuse the bind groups and uniform buffers
//! the first one created, instead of making new ones for every dispatch.
//!
//! Needs a GPU, so `#[ignore]`d by default:
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test dispatch_reuse -- --ignored --test-threads=1
//! ```

#![allow(clippy::expect_used)]

mod common;

use common::*;
use std::io::Write as _;

#[test]
#[ignore = "needs a GPU"]
fn steps_after_the_first_reuse_bind_groups_and_uniforms() {
    let dir = TempDir::new("dispatch-reuse");
    write_tiny_llama(dir.path());
    let seq = pattern_sequence();
    let (input, target) = (&seq[..seq.len() - 1], &seq[1..]);
    let gpu = gpu();
    let model = build_model_with(&gpu, dir.path(), true);
    let (_, params) = sorted_lora_params(&model);
    let mut run_trainer = trainer(&gpu, model, 1e-2);

    let steps: Vec<_> = (0..4)
        .map(|_| {
            let before = gpu.ctx.gpu_counters();
            run_trainer
                .train_step(input, target, &params)
                .expect("step");
            gpu.ctx.gpu_counters().since(&before)
        })
        .collect();
    let mut log = std::io::stderr();
    for (i, s) in steps.iter().enumerate() {
        writeln!(
            log,
            "tiny model step {}: {} dispatches, bind groups {} created {} reused, \
             uniform buffers {} created {} reused",
            i + 1,
            s.dispatches,
            s.bind_groups,
            s.bind_groups_reused,
            s.uniform_buffers,
            s.uniform_buffers_reused,
        )
        .expect("write to stderr");
    }

    for s in &steps[1..] {
        assert_eq!(s.bind_groups + s.bind_groups_reused, s.dispatches);
        // On main every dispatch created a bind group and every dispatch with
        // parameters a uniform buffer (190 and 148 per step). What a later
        // step still creates is tied to data new in that step: the uploaded
        // token embeddings and loss weights, and the AdamW step number.
        assert!(
            s.bind_groups * 4 < s.dispatches,
            "bind groups created in a later step: {s:?}"
        );
        assert!(
            s.uniform_buffers * 4 < s.uniform_buffers + s.uniform_buffers_reused,
            "uniform buffers created in a later step: {s:?}"
        );
    }
}
