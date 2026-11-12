//! Per-block activation checkpointing (`Lfm2Model::with_block_checkpointing`)
//! must give the same losses and gradients as the plain backward pass.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test block_checkpointing -- --ignored --test-threads=1
//! ```

mod common;

use common::*;
use ferrotensor_autograd::Tensor;
use ferrotensor_graph::lfm::Lfm2Model;
use ferrotensor_graph::model_builder::Lfm2ModelBuilder;
use ferrotensor_loader::{ModelLoader, WeightsContext};
use ferrotensor_trainer::{Batch, StepStats};
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

/// The tiny model in `dir`: LoRA on q/k/v/o, or (with `trainable_base`) no
/// LoRA and every base weight trainable.
fn build(gpu: &Gpu, dir: &Path, checkpointing: bool, trainable_base: bool) -> Lfm2Model {
    let config = ModelLoader::load_config(dir.join("config.json")).expect("config");
    let mut weights = WeightsContext::new();
    weights.load_model_dir(dir).expect("weights");
    let mut builder = Lfm2ModelBuilder::new(gpu.ctx.clone(), gpu.engine.clone(), &weights, config)
        .with_trainable_base(trainable_base);
    if !trainable_base {
        builder = builder.with_lora(lora_config());
    }
    builder
        .build()
        .expect("build")
        .with_block_checkpointing(checkpointing)
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

/// The parameters whose gradients are compared, in a fixed order.
fn compared_params(model: &Lfm2Model) -> Vec<(String, Tensor)> {
    let named = if model.lora_params().is_empty() {
        model
            .named_params()
            .into_iter()
            .filter(|(_, t)| t.requires_grad)
            .collect()
    } else {
        model.named_lora_params()
    };
    let mut named: Vec<(String, Tensor)> = named.into_iter().collect();
    named.sort_by(|a, b| a.0.cmp(&b.0));
    named
}

/// Logits for one row (`forward`) or several equally long rows
/// (`forward_batch`).
fn logits(gpu: &Gpu, model: &Lfm2Model, rows: &[Vec<u32>]) -> Tensor {
    if rows.len() == 1 {
        model.forward(&gpu.engine, &rows[0], 0, None, None)
    } else {
        let refs: Vec<&[u32]> = rows.iter().map(Vec::as_slice).collect();
        model.forward_batch(&gpu.engine, &refs)
    }
}

/// The loss tensor and its value for `rows` against `targets`.
fn loss(gpu: &Gpu, model: &Lfm2Model, rows: &[Vec<u32>], targets: &[Vec<u32>]) -> (Tensor, f32) {
    let logits = logits(gpu, model, rows);
    if rows.len() == 1 {
        let mask = vec![1u8; targets[0].len()];
        let loss =
            ferrotensor_trainer::masked_cross_entropy(&logits, &targets[0], &mask, &gpu.engine)
                .expect("loss");
        (loss.tensor, loss.value)
    } else {
        let flat: Vec<u32> = targets.iter().flatten().copied().collect();
        let mask = vec![1u8; flat.len()];
        let loss = ferrotensor_trainer::masked_cross_entropy_batch(
            &logits,
            &flat,
            &mask,
            rows.len(),
            &gpu.engine,
        )
        .expect("loss");
        (loss.tensor, loss.per_sequence.iter().map(|(v, _)| v).sum())
    }
}

type Grads = Vec<(String, Option<Vec<f32>>)>;

/// Loss value and the gradient of every compared parameter after one
/// backward pass.
fn loss_and_grads(
    gpu: &Gpu,
    model: &Lfm2Model,
    rows: &[Vec<u32>],
    targets: &[Vec<u32>],
) -> (f32, Grads) {
    let (tensor, value) = loss(gpu, model, rows, targets);
    tensor.backward(&gpu.engine);
    let grads = compared_params(model)
        .into_iter()
        .map(|(name, t)| {
            let g = t
                .grad
                .borrow()
                .as_ref()
                .map(|g| pollster::block_on(g.to_vec(&gpu.ctx)));
            (name, g)
        })
        .collect();
    (value, grads)
}

fn assert_same(on: &(f32, Grads), off: &(f32, Grads)) {
    let ((a, ga), (b, gb)) = (on, off);
    assert!(
        (a - b).abs() <= 1e-5 * b.abs().max(1.0),
        "loss {a} (checkpointed) vs {b}"
    );
    assert_eq!(ga.len(), gb.len(), "number of compared parameters");
    for ((name, x), (name_b, y)) in ga.iter().zip(gb) {
        assert_eq!(name, name_b);
        match (x, y) {
            (None, None) => {}
            (Some(x), Some(y)) => {
                assert_eq!(x.len(), y.len(), "{name}: gradient length");
                let (worst, at) = x
                    .iter()
                    .zip(y)
                    .map(|(x, y)| (x - y).abs() - 1e-4 * y.abs())
                    .zip(0..)
                    .fold((f32::MIN, 0), |m, (d, i)| if d > m.0 { (d, i) } else { m });
                assert!(
                    worst <= 1e-6,
                    "{name}[{at}]: {} (checkpointed) vs {}",
                    x[at],
                    y[at]
                );
            }
            _ => panic!(
                "{name}: gradient present with checkpointing: {}, without: {}",
                x.is_some(),
                y.is_some()
            ),
        }
    }
}

/// Batch-1 input and target from the pattern sequence.
fn one_row() -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let seq = pattern_sequence();
    (vec![seq[..23].to_vec()], vec![seq[1..].to_vec()])
}

/// Compare checkpointing on and off for `rows`; returns the checkpointed
/// result.
fn compare(
    gpu: &Gpu,
    dir: &Path,
    rows: &[Vec<u32>],
    targets: &[Vec<u32>],
    trainable_base: bool,
) -> (f32, Grads) {
    let off = build(gpu, dir, false, trainable_base);
    let on = build(gpu, dir, true, trainable_base);
    assert!(!off.block_checkpointing());
    assert!(on.block_checkpointing());
    set_lora_b(gpu, &off);
    set_lora_b(gpu, &on);
    let result_off = loss_and_grads(gpu, &off, rows, targets);
    let result_on = loss_and_grads(gpu, &on, rows, targets);
    assert_same(&result_on, &result_off);
    result_on
}

fn assert_lora_grads_present(grads: &Grads) {
    assert_eq!(grads.len(), LAYERS * 4 * 2, "q/k/v/o adapters, A and B");
    for (name, g) in grads {
        assert!(g.is_some(), "{name} received no gradient");
    }
    let layer0_a_nonzero = grads
        .iter()
        .filter(|(n, _)| n.starts_with("model.layers.0.") && n.ends_with("lora_A.weight"))
        .any(|(_, g)| g.as_ref().is_some_and(|g| g.iter().any(|&v| v != 0.0)));
    assert!(layer0_a_nonzero, "every layer-0 LoRA A gradient is zero");
}

#[test]
#[ignore = "needs a GPU"]
fn block_checkpointing_matches_plain_backward_batch_1() {
    let dir = TempDir::new("ckpt-b1");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let (rows, targets) = one_row();
    let (_, grads) = compare(&gpu, dir.path(), &rows, &targets, false);
    assert_lora_grads_present(&grads);
}

#[test]
#[ignore = "needs a GPU"]
fn block_checkpointing_matches_plain_backward_batch_3() {
    let dir = TempDir::new("ckpt-b3");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let seq = pattern_sequence();
    let rows = [
        seq[0..16].to_vec(),
        seq[3..19].to_vec(),
        seq[6..22].to_vec(),
    ];
    let targets = [
        seq[1..17].to_vec(),
        seq[4..20].to_vec(),
        seq[7..23].to_vec(),
    ];
    let (_, grads) = compare(&gpu, dir.path(), &rows, &targets, false);
    assert_lora_grads_present(&grads);
}

#[test]
#[ignore = "needs a GPU"]
fn block_checkpointing_matches_with_trainable_base() {
    let dir = TempDir::new("ckpt-base");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let (rows, targets) = one_row();
    let (_, grads) = compare(&gpu, dir.path(), &rows, &targets, true);
    // Every block weight (norms and projections) trains.
    let block_grads = grads
        .iter()
        .filter(|(n, _)| n.starts_with("model.layers."))
        .collect::<Vec<_>>();
    assert_eq!(block_grads.len(), LAYERS * 9, "7 projections and 2 norms");
    for (name, g) in block_grads {
        assert!(g.is_some(), "{name} received no gradient");
    }
}

/// Data pointers of every tensor reachable from `root` through the graph.
fn reachable(root: &Tensor) -> HashSet<*const ()> {
    let mut seen = HashSet::new();
    let mut stack = vec![root.clone()];
    while let Some(t) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&t.data) as *const ()) {
            continue;
        }
        if let Some(creator) = &t.creator {
            stack.extend(creator.inputs());
        }
    }
    seen
}

/// Every non-leaf tensor reachable from `root`.
fn reachable_intermediates(root: &Tensor) -> Vec<Tensor> {
    let mut seen = HashSet::new();
    let mut stack = vec![root.clone()];
    let mut out = Vec::new();
    while let Some(t) = stack.pop() {
        if !seen.insert(Arc::as_ptr(&t.data) as *const ()) {
            continue;
        }
        if let Some(creator) = &t.creator {
            stack.extend(creator.inputs());
            out.push(t.clone());
        }
    }
    out
}

fn strong_counts(params: &[Tensor]) -> Vec<usize> {
    params.iter().map(|t| Arc::strong_count(&t.data)).collect()
}

#[test]
#[ignore = "needs a GPU"]
fn block_checkpointing_hides_block_internals_from_the_graph() {
    let dir = TempDir::new("ckpt-graph");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let (rows, targets) = one_row();

    let off = build(&gpu, dir.path(), false, false);
    let (_, off_params) = sorted_lora_params(&off);
    let off_before = strong_counts(&off_params);
    let off_logits = logits(&gpu, &off, &rows);
    let off_seen = reachable(&off_logits);
    assert_eq!(off_params.len(), LAYERS * 4 * 2);
    for p in &off_params {
        assert!(off_seen.contains(&(Arc::as_ptr(&p.data) as *const ())));
    }
    // The recorded graph holds every adapter (this is what the check below
    // relies on to see a block's graph).
    for (after, before) in strong_counts(&off_params).iter().zip(&off_before) {
        assert!(after > before, "the plain graph does not hold the adapters");
    }

    let on = build(&gpu, dir.path(), true, false);
    assert!(on.block_checkpointing());
    set_lora_b(&gpu, &on);
    let (_, on_params) = sorted_lora_params(&on);
    let before = strong_counts(&on_params);
    let (loss_t, _) = loss(&gpu, &on, &rows, &targets);
    let on_seen = reachable(&loss_t);
    for p in &on_params {
        assert!(
            !on_seen.contains(&(Arc::as_ptr(&p.data) as *const ())),
            "a LoRA tensor is reachable through the checkpointed graph"
        );
    }
    assert!(
        on_seen.len() < off_seen.len(),
        "checkpointed graph has {} tensors, plain graph {}",
        on_seen.len(),
        off_seen.len()
    );
    // After the forward pass only the replay closure (one clone of each
    // block) holds the adapters: the first pass's graph is gone.
    let after_forward = strong_counts(&on_params);
    for (after, before) in after_forward.iter().zip(&before) {
        assert_eq!(
            *after,
            before + 1,
            "block internals outlived the forward pass"
        );
    }

    loss_t.backward(&gpu.engine);

    // Every adapter received its gradient through the replay...
    for p in &on_params {
        assert!(p.grad.borrow().is_some(), "a LoRA tensor got no gradient");
    }
    // ...the replayed graphs were released (nothing but the replay closures
    // still references the adapters)...
    assert_eq!(
        strong_counts(&on_params),
        after_forward,
        "a recomputed block's graph outlived backward"
    );
    // ...and no intermediate still holds a gradient.
    for t in reachable_intermediates(&loss_t) {
        assert!(
            t.grad.borrow().is_none(),
            "an intermediate of shape {:?} kept its gradient",
            t.data.shape
        );
    }
    drop(loss_t);
    assert_eq!(
        strong_counts(&on_params),
        before,
        "the graph leaked adapters"
    );
}

fn make_batch(seqs: &[(Vec<u32>, Vec<u32>)]) -> Batch {
    let max = seqs.iter().map(|s| s.0.len()).max().unwrap_or(0);
    let pad = |v: &Vec<u32>| {
        let mut p = v.clone();
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
        loss_masks: seqs.iter().map(|_| vec![1u8; max]).collect(),
        batch_size: seqs.len(),
        max_seq_len: max,
    }
}

/// Gradient-norm cap of the short runs, low enough to clip.
const CLIP: f32 = 0.01;

/// Losses and gradient norms of a short run: one-sequence steps (`forward`)
/// and batched steps (`forward_batch`), with gradient accumulation and
/// clipping.
fn short_run(gpu: &Gpu, dir: &Path, checkpointing: bool) -> Vec<StepStats> {
    let model = build(gpu, dir, checkpointing, false);
    let (_, params) = sorted_lora_params(&model);
    let mut trainer = trainer(gpu, model, 1e-2)
        .with_grad_accum_steps(2)
        .with_clip_grad_norm(CLIP);
    let seq = pattern_sequence();
    let (input, target) = (&seq[..seq.len() - 1], &seq[1..]);
    let batch = make_batch(&[
        (seq[0..9].to_vec(), seq[1..10].to_vec()),
        (seq[2..18].to_vec(), seq[3..19].to_vec()),
        (seq[5..16].to_vec(), seq[6..17].to_vec()),
        (seq[1..21].to_vec(), seq[2..22].to_vec()),
    ]);
    let mut stats = Vec::new();
    for _ in 0..3 {
        for _ in 0..2 {
            stats.push(trainer.train_step(input, target, &params).expect("step"));
        }
        stats.extend(trainer.train_batch(&batch, &params).expect("batch"));
    }
    stats
}

#[test]
#[ignore = "needs a GPU"]
fn trainer_steps_match_with_block_checkpointing() {
    let dir = TempDir::new("ckpt-trainer");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let off = short_run(&gpu, dir.path(), false);
    let on = short_run(&gpu, dir.path(), true);
    assert_eq!(on.len(), off.len());
    let mut clipped = false;
    for (i, (a, b)) in on.iter().zip(&off).enumerate() {
        assert_eq!(a.tokens, b.tokens, "step {i}: tokens");
        assert_eq!(
            a.optimizer_step, b.optimizer_step,
            "step {i}: optimizer step"
        );
        assert!(
            (a.loss - b.loss).abs() <= 1e-5 * b.loss.abs().max(1.0),
            "step {i}: loss {} (checkpointed) vs {}",
            a.loss,
            b.loss
        );
        match (a.grad_norm, b.grad_norm) {
            (Some(x), Some(y)) => {
                assert!((x - y).abs() <= 1e-4 * y, "step {i}: grad norm {x} vs {y}");
                clipped |= y > CLIP;
            }
            (None, None) => {}
            other => panic!("step {i}: grad norm {other:?}"),
        }
    }
    assert!(
        off.iter().filter(|s| s.optimizer_step).count() >= 4,
        "too few optimizer steps"
    );
    assert!(clipped, "clipping never applied");
}
