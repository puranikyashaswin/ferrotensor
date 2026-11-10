//! Sequence classification on a tiny Llama-style model: hidden-state taps,
//! the classifier head, its loss and its gradients.

mod common;

use common::*;
use ferrotensor_autograd::{AdamW, Tensor};
use ferrotensor_backend::UnifiedTensor;
use ferrotensor_graph::ClassifierHead;
use ferrotensor_trainer::classification_loss;

const CLASSES: usize = 3;

/// Right-pad `rows` with `pad` to the longest; returns the padded rows and
/// the original lengths.
fn pad_rows(rows: &[Vec<u32>], pad: u32) -> (Vec<Vec<u32>>, Vec<usize>) {
    let longest = rows.iter().map(Vec::len).max().unwrap_or(0);
    let lengths = rows.iter().map(Vec::len).collect();
    let padded = rows
        .iter()
        .map(|r| {
            let mut r = r.clone();
            r.resize(longest, pad);
            r
        })
        .collect();
    (padded, lengths)
}

fn refs(rows: &[Vec<u32>]) -> Vec<&[u32]> {
    rows.iter().map(Vec::as_slice).collect()
}

/// Six sequences of different lengths, two per class; class `c` draws its
/// tokens from its own third of the vocabulary.
fn labelled_rows() -> (Vec<Vec<u32>>, Vec<u32>) {
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for class in 0..CLASSES as u32 {
        for variant in 0..2u32 {
            let len = 6 + (class + 2 * variant) as usize;
            let row = (0..len as u32)
                .map(|i| class * 20 + (i * (3 + variant) + variant) % 20)
                .collect();
            rows.push(row);
            labels.push(class);
        }
    }
    (rows, labels)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "needs a GPU"]
fn hidden_state_taps_follow_the_requested_order_and_stop_early() {
    let dir = TempDir::new("taps");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    let (rows, _) = pad_rows(&labelled_rows().0, 0);
    let rows = refs(&rows);

    let states = model.hidden_states_batch(&gpu.engine, &rows, &[2, 1, 2, 0]);
    assert_eq!(states.len(), 4);
    for state in &states {
        assert_eq!(state.data.shape, vec![rows.len(), rows[0].len(), HIDDEN]);
    }
    let (second, first, again, embed) = (
        host(&gpu, &states[0]),
        host(&gpu, &states[1]),
        host(&gpu, &states[2]),
        host(&gpu, &states[3]),
    );
    assert_eq!(second, again, "the same tap twice is the same tensor");
    assert!(
        max_abs_diff(&first, &second) > 1e-3,
        "blocks 1 and 2 differ"
    );
    assert!(
        max_abs_diff(&embed, &first) > 1e-3,
        "block 1 changes the embeddings"
    );

    // Tapping only block 1 runs one block and gives the same values.
    let only_first = model.hidden_states_batch(&gpu.engine, &rows, &[1]);
    assert!(max_abs_diff(&host(&gpu, &only_first[0]), &first) < 1e-5);
}

#[test]
#[ignore = "needs a GPU"]
fn padding_does_not_change_the_logits() {
    let dir = TempDir::new("padding");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    let head = ClassifierHead::new_seeded(gpu.ctx.clone(), HIDDEN, CLASSES, &[1, 2], 7);
    let (rows, _) = labelled_rows();

    // Each row alone, without padding.
    let mut alone = Vec::new();
    for row in &rows {
        let logits = head.forward(&gpu.engine, &model, &[row.as_slice()], &[row.len()]);
        assert_eq!(logits.data.shape, vec![1, CLASSES]);
        alone.extend(host(&gpu, &logits));
    }
    // The same rows in one batch, padded with two different tokens.
    for pad in [0, (VOCAB - 1) as u32] {
        let (padded, lengths) = pad_rows(&rows, pad);
        let logits = head.forward(&gpu.engine, &model, &refs(&padded), &lengths);
        assert_eq!(logits.data.shape, vec![rows.len(), CLASSES]);
        let diff = max_abs_diff(&host(&gpu, &logits), &alone);
        assert!(diff < 1e-4, "pad token {pad}: logits moved by {diff}");
    }
}

#[test]
#[ignore = "needs a GPU"]
fn classifier_gradients_match_finite_differences() {
    let dir = TempDir::new("fd");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    let (rows, labels) = labelled_rows();
    let (padded, lengths) = pad_rows(&rows, 0);
    let rows = refs(&padded);
    let taps = vec![1, 2];
    let head = ClassifierHead::new_seeded(gpu.ctx.clone(), HIDDEN, CLASSES, &taps, 3);

    let logits = head.forward(&gpu.engine, &model, &rows, &lengths);
    let loss = classification_loss(&logits, &labels, &gpu.engine).expect("loss");
    assert_eq!(loss.tokens, labels.len());
    loss.tensor.backward(&gpu.engine);

    // The loss with one head tensor replaced by `values`.
    let loss_with = |which: usize, is_norm: bool, values: &[f32]| -> f32 {
        let leaf = |t: &Tensor| {
            Tensor::new(
                gpu.ctx.clone(),
                UnifiedTensor::new(&gpu.ctx, &host(&gpu, t), &t.data.shape),
            )
        };
        let mut norms: Vec<Tensor> = head.norms.iter().map(leaf).collect();
        let mut weights: Vec<Tensor> = head.weights.iter().map(leaf).collect();
        let slot = if is_norm {
            &mut norms[which]
        } else {
            &mut weights[which]
        };
        let shape = slot.data.shape.clone();
        *slot = Tensor::new(
            gpu.ctx.clone(),
            UnifiedTensor::new(&gpu.ctx, values, &shape),
        );
        let perturbed = ClassifierHead::from_parts(taps.clone(), norms, weights);
        let logits = perturbed.forward(&gpu.engine, &model, &rows, &lengths);
        classification_loss(&logits, &labels, &gpu.engine)
            .expect("loss")
            .value
    };

    let step = 2e-2;
    for (which, is_norm, index) in [(0, false, 5), (1, false, 40), (0, true, 3), (1, true, 17)] {
        let tensor = if is_norm {
            &head.norms[which]
        } else {
            &head.weights[which]
        };
        let grad = tensor.grad.borrow().clone().expect("head gradient");
        let analytic = pollster::block_on(grad.to_vec(&gpu.ctx))[index];
        let base = host(&gpu, tensor);
        let (mut up, mut down) = (base.clone(), base);
        up[index] += step;
        down[index] -= step;
        let numeric =
            (loss_with(which, is_norm, &up) - loss_with(which, is_norm, &down)) / (2.0 * step);
        assert!(
            (analytic - numeric).abs() < 2e-3 + 0.05 * numeric.abs(),
            "tap {which} norm {is_norm} index {index}: analytic {analytic}, numeric {numeric}"
        );
    }

    // The gradient also reaches the decoder's LoRA adapters.
    let (_, lora) = sorted_lora_params(&model);
    assert!(
        lora.iter().all(|t| t.grad.borrow().is_some()),
        "every LoRA tensor should get a gradient through the hidden states"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn classifier_with_lora_learns_the_labels() {
    let dir = TempDir::new("learn");
    write_tiny_llama(dir.path());
    let gpu = gpu();
    let model = build_model(&gpu, dir.path());
    let head = ClassifierHead::new_seeded(gpu.ctx.clone(), HIDDEN, CLASSES, &[1, 2], 0);
    let (rows, labels) = labelled_rows();
    let (padded, lengths) = pad_rows(&rows, 0);
    let rows = refs(&padded);

    let (_, mut params) = sorted_lora_params(&model);
    params.extend(head.params());
    let mut optimizer = AdamW::new(gpu.engine.clone(), 1e-2);

    let mut losses = Vec::new();
    for _ in 0..60 {
        let logits = head.forward(&gpu.engine, &model, &rows, &lengths);
        let loss = classification_loss(&logits, &labels, &gpu.engine).expect("loss");
        assert!(loss.value.is_finite());
        losses.push(loss.value);
        loss.tensor.backward(&gpu.engine);
        optimizer.step(&params);
        optimizer.zero_grad(&params);
    }
    assert!(
        losses[losses.len() - 1] < 0.25 * losses[0],
        "the loss should fall on six separable sequences: {losses:?}"
    );

    let logits = host(&gpu, &head.forward(&gpu.engine, &model, &rows, &lengths));
    for (row, &label) in labels.iter().enumerate() {
        let scores = &logits[row * CLASSES..(row + 1) * CLASSES];
        let predicted = (0..CLASSES)
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .expect("classes");
        assert_eq!(predicted as u32, label, "row {row}: scores {scores:?}");
    }
}

#[test]
#[ignore = "needs a GPU"]
fn classification_loss_rejects_bad_labels() {
    let gpu = gpu();
    let logits = Tensor::new(
        gpu.ctx.clone(),
        UnifiedTensor::new(&gpu.ctx, &[0.5, -0.5, 0.0, 1.0, 2.0, -1.0], &[2, CLASSES]),
    );
    assert!(classification_loss(&logits, &[0, 2], &gpu.engine).is_some());
    assert!(
        classification_loss(&logits, &[0, CLASSES as u32], &gpu.engine).is_none(),
        "a label outside the classes"
    );
    assert!(
        classification_loss(&logits, &[0], &gpu.engine).is_none(),
        "one label for two rows"
    );
}
