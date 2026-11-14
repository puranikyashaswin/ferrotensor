//! `ClassifierTrainer` on a tiny Llama-style model: learning, determinism,
//! gradient accumulation, the two learning rates, skipped steps, saving and
//! loading, and the first-token-skipping pooling.

mod common;

use common::*;
use ferrotensor_autograd::WarmupCosineScheduler;
use ferrotensor_graph::lfm::Lfm2Model;
use ferrotensor_graph::model_builder::{Lfm2ModelBuilder, LoRaConfig};
use ferrotensor_graph::{mean_pool_matrix, ClassifierHead};
use ferrotensor_loader::{ModelLoader, WeightsContext};
use ferrotensor_trainer::checkpoint::{
    apply_checkpoint, load_lora_checkpoint, save_lora_checkpoint, LoraMetadata,
};
use ferrotensor_trainer::classify::{
    load_classifier_head, plan_epoch, predict_rows, predicted_class, save_classifier_head,
};
use ferrotensor_trainer::ClassifierTrainer;
use std::path::Path;

const CLASSES: usize = 3;
const TARGETS: [&str; 7] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];
const TAPS: [usize; 2] = [1, 2];

fn lora_meta() -> LoraMetadata {
    LoraMetadata {
        rank: 4,
        alpha: 8.0,
        target_modules: TARGETS.iter().map(|t| t.to_string()).collect(),
        trained_modules: None,
    }
}

/// The tiny model with adapters on all seven projections, optionally cut
/// after `layers` blocks (as `classify-train` does for its deepest tap).
fn build(gpu: &Gpu, dir: &Path, layers: Option<usize>) -> Lfm2Model {
    let mut config = ModelLoader::load_config(dir.join("config.json")).expect("config");
    if let Some(layers) = layers {
        config.num_hidden_layers = layers;
    }
    let mut weights = WeightsContext::new();
    weights.load_model_dir(dir).expect("weights");
    let meta = lora_meta();
    Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &weights, config)
        .with_lora(LoRaConfig {
            rank: meta.rank,
            alpha: meta.alpha,
            target_modules: meta.target_modules,
        })
        .build()
        .expect("build")
}

fn classifier(gpu: &Gpu, dir: &Path, lr: f32, head_lr: f32, seed: u64) -> ClassifierTrainer {
    let model = build(gpu, dir, None);
    let head = ClassifierHead::new_seeded(gpu.ctx.clone(), HIDDEN, CLASSES, &TAPS, seed)
        .with_skip_first(true);
    ClassifierTrainer::new(
        model,
        head,
        gpu.ctx.clone(),
        gpu.engine.clone(),
        lr,
        head_lr,
    )
}

/// `per_class` sequences of different lengths per class; class `c` draws
/// its tokens from its own third of the vocabulary.
fn labelled_rows(per_class: u32) -> (Vec<Vec<u32>>, Vec<u32>) {
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for class in 0..CLASSES as u32 {
        for variant in 0..per_class {
            let len = 6 + ((class + 2 * variant) % 7) as usize;
            let row = (0..len as u32)
                .map(|i| class * 20 + (i * (3 + variant) + variant) % 20)
                .collect();
            rows.push(row);
            labels.push(class);
        }
    }
    (rows, labels)
}

fn refs(rows: &[Vec<u32>]) -> Vec<&[u32]> {
    rows.iter().map(Vec::as_slice).collect()
}

fn values(gpu: &Gpu, params: &[ferrotensor_autograd::Tensor]) -> Vec<Vec<f32>> {
    params.iter().map(|p| host(gpu, p)).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn relative_diff(a: f32, b: f32) -> f32 {
    (a - b).abs() / a.abs().max(b.abs()).max(1e-12)
}

#[test]
#[ignore = "needs a GPU"]
fn classifier_trainer_learns_separable_labels_with_seven_lora_targets() {
    let dir = TempDir::new("classify-learn");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let mut trainer = classifier(&gpu, dir.path(), 1e-2, 1e-2, 0);
    let (rows, labels) = labelled_rows(2);
    let rows = refs(&rows);

    let named = trainer.model.named_lora_params();
    for target in TARGETS {
        assert!(
            named.keys().any(|n| n.contains(target)),
            "no adapter on {target}"
        );
    }
    let before: Vec<(String, Vec<f32>)> = named
        .iter()
        .map(|(n, t)| (n.clone(), host(&gpu, t)))
        .collect();

    let mut losses = Vec::new();
    for _ in 0..60 {
        let stats = trainer.train_batch(&rows, &labels).expect("train");
        let update = stats.update.expect("grad_accum 1 steps after every batch");
        assert!(update.applied && update.grad_norm.is_finite());
        assert!(stats.loss.is_finite());
        losses.push(stats.loss);
    }
    assert_eq!(trainer.optimizer_steps(), 60);
    assert_eq!(trainer.skipped_steps(), 0);
    assert!(
        losses[losses.len() - 1] < 0.25 * losses[0],
        "the loss should fall on six separable sequences: {losses:?}"
    );

    let probs = trainer.predict(&rows).expect("predict");
    for (row, (p, &label)) in probs.iter().zip(&labels).enumerate() {
        assert_eq!(predicted_class(p), label, "row {row}: {p:?}");
    }

    // Every target's B matrix (zero at the start) has moved.
    for target in TARGETS {
        let moved = before.iter().any(|(name, old)| {
            name.contains(target)
                && name.contains("lora_B")
                && max_abs_diff(old, &host(&gpu, &named[name])) > 0.0
        });
        assert!(moved, "the {target} adapters did not train");
    }
}

#[test]
#[ignore = "needs a GPU"]
fn the_same_seed_gives_identical_runs() {
    let dir = TempDir::new("classify-seed");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let (rows, labels) = labelled_rows(3);
    let lengths: Vec<usize> = rows.iter().map(Vec::len).collect();

    let run = |seed: u64| -> Vec<f32> {
        let mut trainer = classifier(&gpu, dir.path(), 1e-2, 1e-2, seed)
            .with_grad_accum(2)
            .with_lr_scheduler(Box::new(WarmupCosineScheduler::new(1.0, 0.1, 2, 6)));
        let mut trace = Vec::new();
        for epoch in 0..2 {
            for batch in plan_epoch(&lengths, 2, seed, epoch) {
                let r: Vec<&[u32]> = batch.iter().map(|&i| rows[i].as_slice()).collect();
                let l: Vec<u32> = batch.iter().map(|&i| labels[i]).collect();
                let stats = trainer.train_batch(&r, &l).expect("train");
                trace.push(stats.loss);
                if let Some(u) = stats.update {
                    trace.extend([u.grad_norm, u.lr, u.head_lr]);
                }
            }
        }
        if let Some(u) = trainer.flush() {
            trace.extend([u.grad_norm, u.lr, u.head_lr]);
        }
        trace.extend(trainer.predict(&refs(&rows)).expect("predict").concat());
        trace
    };
    let first = run(5);
    assert_eq!(first, run(5), "same seed, same losses and probabilities");
    assert_ne!(first, run(6), "a different seed changes the run");
}

#[test]
#[ignore = "needs a GPU"]
fn saved_head_and_adapter_give_the_same_probabilities() {
    let dir = TempDir::new("classify-save-model");
    write_tiny_llama(dir.path());
    let out = TempDir::new("classify-save-out");
    let gpu = gpu();
    let mut trainer = classifier(&gpu, dir.path(), 1e-2, 1e-2, 2);
    let (rows, labels) = labelled_rows(2);
    for _ in 0..5 {
        trainer.train_batch(&refs(&rows), &labels).expect("train");
    }
    let probs =
        predict_rows(&gpu.engine, &trainer.model, &trainer.head, &rows, 4, 0).expect("predict");
    for p in &probs {
        assert_eq!(p.len(), CLASSES);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probabilities sum to {sum}");
    }

    let adapter = out.path().join("adapter.safetensors");
    let head_path = out.path().join("head.safetensors");
    pollster::block_on(save_lora_checkpoint(
        &gpu.ctx,
        &adapter,
        &trainer.model.named_lora_params(),
        Some(&lora_meta()),
    ))
    .expect("save adapter");
    pollster::block_on(save_classifier_head(&gpu.ctx, &head_path, &trainer.head))
        .expect("save head");

    let fresh = build(&gpu, dir.path(), None);
    let ckpt = load_lora_checkpoint(&adapter).expect("load adapter");
    assert_eq!(ckpt.metadata, Some(lora_meta()));
    apply_checkpoint(&gpu.ctx, &ckpt, &fresh.named_lora_params()).expect("apply");
    let head = load_classifier_head(&gpu.ctx, &head_path, &TAPS, true).expect("load head");
    assert!(head.skip_first);
    assert_eq!(head.classes(), CLASSES);
    let again = predict_rows(&gpu.engine, &fresh, &head, &rows, 4, 0).expect("predict");
    let diff = max_abs_diff(&probs.concat(), &again.concat());
    assert!(
        diff < 1e-6,
        "probabilities moved by {diff} after save and load"
    );

    // A head saved for other taps does not load silently.
    assert!(load_classifier_head(&gpu.ctx, &head_path, &[1], true).is_err());
    assert!(load_classifier_head(&gpu.ctx, &head_path, &[1, 3], true).is_err());
}

#[test]
#[ignore = "needs a GPU"]
fn two_accumulated_batches_of_four_match_one_batch_of_eight() {
    let dir = TempDir::new("classify-accum");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let (rows, labels) = labelled_rows(3);
    let (rows, labels) = (refs(&rows[..8]), &labels[..8]);

    let mut accum = classifier(&gpu, dir.path(), 1e-2, 1e-2, 1).with_grad_accum(2);
    let first = accum.train_batch(&rows[..4], &labels[..4]).expect("train");
    assert!(first.update.is_none(), "the window is not complete yet");
    assert_eq!(accum.pending_batches(), 1);
    let second = accum.train_batch(&rows[4..], &labels[4..]).expect("train");
    let accum_update = second.update.expect("the window is complete");

    let mut whole = classifier(&gpu, dir.path(), 1e-2, 1e-2, 1);
    let stats = whole.train_batch(&rows, labels).expect("train");
    let whole_update = stats.update.expect("one batch per step");

    // Equal batch sizes: the mean of the two batch means is the mean over
    // all eight rows, for the loss and for every gradient.
    let mean = (first.loss + second.loss) / 2.0;
    assert!(
        relative_diff(mean, stats.loss) < 1e-5,
        "{mean} vs {}",
        stats.loss
    );
    assert!(
        relative_diff(accum_update.grad_norm, whole_update.grad_norm) < 1e-4,
        "gradient norms {} vs {}",
        accum_update.grad_norm,
        whole_update.grad_norm
    );

    // flush() averages an unfinished window over the batches it holds.
    let mut partial = classifier(&gpu, dir.path(), 1e-2, 1e-2, 1).with_grad_accum(3);
    let only = partial
        .train_batch(&rows[..4], &labels[..4])
        .expect("train");
    assert!(only.update.is_none());
    let flushed = partial.flush().expect("one batch pending");
    assert!(partial.flush().is_none(), "nothing left to flush");
    assert_eq!(partial.optimizer_steps(), 1);
    let mut single = classifier(&gpu, dir.path(), 1e-2, 1e-2, 1);
    let reference = single.train_batch(&rows[..4], &labels[..4]).expect("train");
    let reference = reference.update.expect("one batch per step");
    assert!(
        relative_diff(flushed.grad_norm, reference.grad_norm) < 1e-5,
        "{} vs {}",
        flushed.grad_norm,
        reference.grad_norm
    );
}

#[test]
#[ignore = "needs a GPU"]
fn each_learning_rate_drives_its_own_group() {
    let dir = TempDir::new("classify-lrs");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let (rows, labels) = labelled_rows(2);
    let rows = refs(&rows);

    for (lr, head_lr) in [(1e-2, 0.0), (0.0, 1e-2)] {
        let mut trainer = classifier(&gpu, dir.path(), lr, head_lr, 3);
        let lora_before = values(&gpu, trainer.lora_params());
        let head_before = values(&gpu, trainer.head_params());
        for _ in 0..3 {
            let update = trainer.train_batch(&rows, &labels).expect("train").update;
            let update = update.expect("one batch per step");
            assert_eq!((update.lr, update.head_lr), (lr, head_lr));
        }
        let lora_after = values(&gpu, trainer.lora_params());
        let head_after = values(&gpu, trainer.head_params());
        let lora_moved = lora_before != lora_after;
        let head_moved = head_before != head_after;
        assert_eq!(lora_moved, lr > 0.0, "lr {lr}: LoRA moved {lora_moved}");
        assert_eq!(
            head_moved,
            head_lr > 0.0,
            "head lr {head_lr}: head moved {head_moved}"
        );
    }

    // A schedule scales both rates by the same factor.
    let mut trainer = classifier(&gpu, dir.path(), 1e-2, 1e-1, 3)
        .with_lr_scheduler(Box::new(WarmupCosineScheduler::new(1.0, 0.1, 4, 8)));
    let update = trainer.train_batch(&rows, &labels).expect("train").update;
    let update = update.expect("one batch per step");
    assert!((update.lr - 1e-2 / 4.0).abs() < 1e-9, "{}", update.lr);
    assert!(
        (update.head_lr - 1e-1 / 4.0).abs() < 1e-8,
        "{}",
        update.head_lr
    );
}

#[test]
#[ignore = "needs a GPU"]
fn a_non_finite_gradient_skips_the_step() {
    let dir = TempDir::new("classify-nan");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let mut trainer = classifier(&gpu, dir.path(), 1e-2, 1e-2, 0);
    let (rows, labels) = labelled_rows(2);
    let nan: Vec<u8> = [f32::NAN; HIDDEN]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    trainer.head.norms[0].data.write_data(&gpu.ctx, &nan);
    let lora_before = values(&gpu, trainer.lora_params());

    let stats = trainer.train_batch(&refs(&rows), &labels).expect("train");
    let update = stats.update.expect("the window is complete");
    assert!(!update.applied);
    assert!(!update.grad_norm.is_finite());
    assert_eq!(trainer.skipped_steps(), 1);
    assert_eq!(trainer.optimizer_steps(), 0);
    assert_eq!(values(&gpu, trainer.lora_params()), lora_before);
    assert!(
        trainer
            .lora_params()
            .iter()
            .chain(trainer.head_params())
            .all(|p| p.grad.borrow().is_none()),
        "the dropped window's gradients are cleared"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn skip_first_pools_over_the_tokens_after_the_first() {
    let gpu = gpu();
    // Two rows of up to four positions; the second row has one token.
    let pool = host(&gpu, &mean_pool_matrix(&gpu.ctx, 2, 4, &[3, 1], true));
    let mut expected = vec![0.0f32; 16];
    expected[1] = 0.5;
    expected[2] = 0.5;
    assert_eq!(pool, expected, "a one-token row pools to zeros");
    let plain = host(&gpu, &mean_pool_matrix(&gpu.ctx, 2, 4, &[3, 1], false));
    let mut expected = vec![0.0f32; 16];
    expected[..3].fill(1.0 / 3.0);
    expected[12] = 1.0;
    assert_eq!(plain, expected);

    let dir = TempDir::new("classify-skip");
    write_tiny_llama(dir.path());
    let model = build(&gpu, dir.path(), None);
    let skip = ClassifierHead::new_seeded(gpu.ctx.clone(), HIDDEN, CLASSES, &TAPS, 7)
        .with_skip_first(true);
    let keep = ClassifierHead::new_seeded(gpu.ctx.clone(), HIDDEN, CLASSES, &TAPS, 7);
    assert!(!keep.skip_first, "off by default");
    let (rows, _) = labelled_rows(2);

    let mut alone = Vec::new();
    for row in &rows {
        alone.extend(host(
            &gpu,
            &skip.forward(&gpu.engine, &model, &[row.as_slice()], &[row.len()]),
        ));
    }
    let longest = rows.iter().map(Vec::len).max().unwrap_or(0);
    let lengths: Vec<usize> = rows.iter().map(Vec::len).collect();
    let padded: Vec<Vec<u32>> = rows
        .iter()
        .map(|r| {
            let mut r = r.clone();
            r.resize(longest, (VOCAB - 1) as u32);
            r
        })
        .collect();
    let batched = host(
        &gpu,
        &skip.forward(&gpu.engine, &model, &refs(&padded), &lengths),
    );
    let diff = max_abs_diff(&batched, &alone);
    assert!(diff < 1e-4, "padding moved the logits by {diff}");
    let kept = host(
        &gpu,
        &keep.forward(&gpu.engine, &model, &refs(&padded), &lengths),
    );
    assert!(
        max_abs_diff(&kept, &batched) > 1e-4,
        "leaving the first token out changes the logits"
    );

    let one = host(
        &gpu,
        &skip.forward(&gpu.engine, &model, &[&[5u32][..]], &[1]),
    );
    assert_eq!(one, vec![0.0; CLASSES], "nothing left to pool");
}

#[test]
#[ignore = "needs a GPU"]
fn a_model_cut_after_the_deepest_tap_gives_the_same_logits() {
    let dir = TempDir::new("classify-cut");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let full = build(&gpu, dir.path(), None);
    let cut = build(&gpu, dir.path(), Some(1));
    assert_eq!(
        2 * cut.named_lora_params().len(),
        full.named_lora_params().len()
    );

    let head = ClassifierHead::new_seeded(gpu.ctx.clone(), HIDDEN, CLASSES, &[1], 4);
    let (rows, _) = labelled_rows(1);
    let a = predict_rows(&gpu.engine, &full, &head, &rows, 8, 0).expect("predict");
    let b = predict_rows(&gpu.engine, &cut, &head, &rows, 8, 0).expect("predict");
    let diff = max_abs_diff(&a.concat(), &b.concat());
    assert!(
        diff < 1e-6,
        "cutting unused blocks moved the probabilities by {diff}"
    );
}
