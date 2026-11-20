//! Running `lm_head` and the loss on the supervised positions alone, and
//! padding each micro-batch to its longest sequence, must give the losses and
//! gradients of the full computation over bucket-padded sequences.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test supervised_head_rows -- --ignored --test-threads=1
//! ```

#![allow(clippy::expect_used, clippy::panic)]

mod common;

use common::*;
use ferrotensor_graph::lfm::Lfm2Model;
use ferrotensor_trainer::{Batch, StepStats};
use std::path::Path;

/// A batch of `(input, target, supervised from)` sequences: the loss mask is
/// 0 before the third field, as for the prompt of an SFT example, and 0 on
/// the padding.
fn masked_batch(seqs: &[(Vec<u32>, Vec<u32>, usize)]) -> Batch {
    let max = seqs.iter().map(|s| s.0.len()).max().unwrap_or(0);
    let pad = |v: &[u32]| {
        let mut p = v.to_vec();
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
                (0..max)
                    .map(|i| u8::from(i >= s.2 && i < s.0.len()))
                    .collect()
            })
            .collect(),
        batch_size: seqs.len(),
        max_seq_len: max,
    }
}

/// Sequences of different lengths with prompts of different lengths, one of
/// them with no supervised token.
fn sequences() -> Vec<(Vec<u32>, Vec<u32>, usize)> {
    let seq = pattern_sequence();
    vec![
        (seq[0..9].to_vec(), seq[1..10].to_vec(), 4),
        (seq[2..18].to_vec(), seq[3..19].to_vec(), 7),
        (seq[5..16].to_vec(), seq[6..17].to_vec(), 11),
        (seq[1..21].to_vec(), seq[2..22].to_vec(), 12),
        (seq[3..13].to_vec(), seq[4..14].to_vec(), 0),
    ]
}

/// How a run pads and where it runs the head.
#[derive(Clone, Copy)]
struct Setup {
    bucket: usize,
    head_rows: bool,
}

/// Per-step stats and the final LoRA parameters of a short run: one-sequence
/// steps and batched steps, with accumulation and clipping.
fn run(gpu: &Gpu, dir: &Path, setup: Setup) -> (Vec<StepStats>, Vec<Vec<f32>>) {
    let model = build_model(gpu, dir);
    let (_, params) = sorted_lora_params(&model);
    let mut trainer = trainer(gpu, model, 1e-2)
        .with_grad_accum_steps(2)
        .with_clip_grad_norm(0.5)
        .with_length_bucket(setup.bucket)
        .with_supervised_head_rows(setup.head_rows);
    let seqs = sequences();
    let batch = masked_batch(&seqs);
    let mut stats = Vec::new();
    for _ in 0..3 {
        for (input, target, from) in &seqs[..2] {
            let mask: Vec<u8> = (0..input.len()).map(|i| u8::from(i >= *from)).collect();
            stats.push(
                trainer
                    .train_step_masked(input, target, Some(&mask), &params)
                    .expect("step"),
            );
        }
        stats.extend(trainer.train_batch(&batch, &params).expect("batch"));
    }
    let values = params.iter().map(|p| host(gpu, p)).collect();
    (stats, values)
}

fn assert_same_run(
    (a_stats, a_params): &(Vec<StepStats>, Vec<Vec<f32>>),
    (b_stats, b_params): &(Vec<StepStats>, Vec<Vec<f32>>),
    what: &str,
) {
    assert_eq!(a_stats.len(), b_stats.len(), "{what}: step count");
    for (i, (a, b)) in a_stats.iter().zip(b_stats).enumerate() {
        assert_eq!(a.tokens, b.tokens, "{what}, step {i}: tokens");
        assert_eq!(
            a.optimizer_step, b.optimizer_step,
            "{what}, step {i}: optimizer step"
        );
        assert!(
            (a.loss - b.loss).abs() <= 1e-4 * b.loss.abs().max(1.0),
            "{what}, step {i}: loss {} vs {}",
            a.loss,
            b.loss
        );
        if let (Some(x), Some(y)) = (a.grad_norm, b.grad_norm) {
            assert!(
                (x - y).abs() <= 1e-3 * y.max(1e-3),
                "{what}, step {i}: grad norm {x} vs {y}"
            );
        }
    }
    assert!(
        a_stats.iter().any(|s| s.tokens > 0 && s.loss > 0.0),
        "{what}: no loss was measured"
    );
    for (p, (a, b)) in a_params.iter().zip(b_params).enumerate() {
        let worst = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 1e-4,
            "{what}: LoRA parameter {p} differs by {worst}"
        );
    }
}

/// The reference is today's path: the head on every position of sequences
/// padded to a multiple of 32.
#[test]
#[ignore = "needs a GPU"]
fn head_on_supervised_rows_matches_the_full_head() {
    let dir = TempDir::new("head-rows");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let full = run(
        &gpu,
        dir.path(),
        Setup {
            bucket: 32,
            head_rows: false,
        },
    );
    let gathered = run(
        &gpu,
        dir.path(),
        Setup {
            bucket: 32,
            head_rows: true,
        },
    );
    assert_same_run(&gathered, &full, "gathered head");
}

#[test]
#[ignore = "needs a GPU"]
fn exact_padding_matches_bucket_padding() {
    let dir = TempDir::new("exact-pad");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    for head_rows in [false, true] {
        let bucketed = run(
            &gpu,
            dir.path(),
            Setup {
                bucket: 32,
                head_rows,
            },
        );
        let exact = run(
            &gpu,
            dir.path(),
            Setup {
                bucket: 1,
                head_rows,
            },
        );
        assert_same_run(&exact, &bucketed, "exact padding");
    }
}

/// The default pads to the longest sequence and runs the head on the
/// supervised rows, and still gives today's numbers.
#[test]
#[ignore = "needs a GPU"]
fn defaults_match_the_bucketed_full_head() {
    let dir = TempDir::new("defaults");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let reference = run(
        &gpu,
        dir.path(),
        Setup {
            bucket: 32,
            head_rows: false,
        },
    );
    let model = build_model(&gpu, dir.path());
    let (_, params) = sorted_lora_params(&model);
    let mut trainer = trainer(&gpu, model, 1e-2)
        .with_grad_accum_steps(2)
        .with_clip_grad_norm(0.5);
    assert_eq!(trainer.length_bucket, 0);
    assert!(trainer.head_on_supervised_rows);
    let seqs = sequences();
    let batch = masked_batch(&seqs);
    let mut stats = Vec::new();
    for _ in 0..3 {
        for (input, target, from) in &seqs[..2] {
            let mask: Vec<u8> = (0..input.len()).map(|i| u8::from(i >= *from)).collect();
            stats.push(
                trainer
                    .train_step_masked(input, target, Some(&mask), &params)
                    .expect("step"),
            );
        }
        stats.extend(trainer.train_batch(&batch, &params).expect("batch"));
    }
    let values = params.iter().map(|p| host(&gpu, p)).collect();
    assert_same_run(&(stats, values), &reference, "defaults");
}

/// Row `i` of the gathered logits is row `positions[i]` of the full ones.
fn assert_rows_match(full: &[f32], gathered: &[f32], positions: &[u32], what: &str) {
    assert_eq!(full.len() % VOCAB, 0);
    assert_eq!(gathered.len(), positions.len() * VOCAB, "{what}: shape");
    for (i, &p) in positions.iter().enumerate() {
        let want = &full[p as usize * VOCAB..(p as usize + 1) * VOCAB];
        let got = &gathered[i * VOCAB..(i + 1) * VOCAB];
        for (v, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                "{what}: row {i} (position {p}), logit {v}: {g} vs {w}"
            );
        }
    }
}

fn logits_of(gpu: &Gpu, model: &Lfm2Model, rows: &[&[u32]], head: Option<&[u32]>) -> Vec<f32> {
    let logits = match (rows, head) {
        ([row], Some(head)) => model.forward_head_rows(&gpu.engine, row, None, head),
        ([row], None) => model.forward_with_dropout(&gpu.engine, row, 0, None, None, None),
        (rows, Some(head)) => model.forward_batch_head_rows(&gpu.engine, rows, None, head),
        (rows, None) => model.forward_batch_with_dropout(&gpu.engine, rows, None),
    };
    host(gpu, &logits)
}

#[test]
#[ignore = "needs a GPU"]
fn head_rows_logits_are_rows_of_the_full_logits() {
    let dir = TempDir::new("head-logits");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    let seq = pattern_sequence();

    let row = &seq[..20];
    let positions = [3u32, 4, 9, 10, 11, 19];
    let full = logits_of(&gpu, &model, &[row], None);
    let gathered = logits_of(&gpu, &model, &[row], Some(&positions));
    assert_rows_match(&full, &gathered, &positions, "one sequence");

    let rows: Vec<&[u32]> = vec![&seq[0..16], &seq[3..19], &seq[6..22]];
    let positions = [0u32, 5, 15, 16, 20, 31, 32, 47];
    let full = logits_of(&gpu, &model, &rows, None);
    let gathered = logits_of(&gpu, &model, &rows, Some(&positions));
    assert_rows_match(&full, &gathered, &positions, "batch of three");
}
