//! `lm_head` and the loss over chunks of rows must give the loss and the
//! gradients of the whole head, and hold the logits of one chunk at a time.
//!
//! The GPU tests are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-trainer --test chunked_head -- --ignored --test-threads=1
//! ```

#![allow(clippy::expect_used)]

mod common;

use common::*;
use ferrotensor_autograd::Tensor;
use ferrotensor_backend::UnifiedTensor;
use ferrotensor_graph::{Linear, LoRAAdapter};
use ferrotensor_trainer::{
    chunked_head_cross_entropy, chunked_head_loss, head_chunk_rows, masked_cross_entropy_batch,
    masked_cross_entropy_from_logits, max_whole_head_rows, HEAD_CHUNK_MIN_BYTES, HEAD_CHUNK_ROWS,
    HEAD_LOGIT_BYTES_PER_VALUE,
};

const WIDTH: usize = 16;
const CLASSES: usize = 50;
const RANK: usize = 4;
const ALPHA: f32 = 8.0;
const SEQUENCES: usize = 3;
const SEQ_LEN: usize = 13;
const ROWS: usize = SEQUENCES * SEQ_LEN;

fn leaf(gpu: &Gpu, data: &[f32], shape: &[usize], trainable: bool) -> Tensor {
    let data = UnifiedTensor::new(&gpu.ctx, data, shape);
    if trainable {
        Tensor::new(gpu.ctx.clone(), data)
    } else {
        Tensor::frozen(gpu.ctx.clone(), data)
    }
}

/// What the head has: a trainable weight, a bias and a LoRA adapter whose
/// two matrices are both non-zero, a trainable weight and bias, or a frozen
/// weight alone (the usual `lm_head` of a LoRA run).
#[derive(Clone, Copy)]
enum Head {
    Everything,
    FrozenWeight,
    /// A trainable weight and bias, no adapter: the head whose gradients
    /// have a short closed form (see [`host_gradients`]).
    WeightAndBias,
}

/// The head and the hidden states it projects: `[SEQUENCES, SEQ_LEN, WIDTH]`.
struct Case {
    head: Linear,
    hidden: Tensor,
}

impl Case {
    fn new(gpu: &Gpu, kind: Head) -> Self {
        let weight = pseudo_random(WIDTH * CLASSES, 2, 0.5);
        let hidden = leaf(
            gpu,
            &pseudo_random(ROWS * WIDTH, 1, 1.0),
            &[SEQUENCES, SEQ_LEN, WIDTH],
            true,
        );
        let head = match kind {
            Head::FrozenWeight => Linear::new(leaf(gpu, &weight, &[WIDTH, CLASSES], false)),
            Head::WeightAndBias => {
                let mut head = Linear::new(leaf(gpu, &weight, &[WIDTH, CLASSES], true));
                head.bias = Some(leaf(gpu, &pseudo_random(CLASSES, 3, 0.2), &[CLASSES], true));
                head
            }
            Head::Everything => {
                let mut head = Linear::new(leaf(gpu, &weight, &[WIDTH, CLASSES], true));
                head.bias = Some(leaf(gpu, &pseudo_random(CLASSES, 3, 0.2), &[CLASSES], true));
                head.lora = Some(LoRAAdapter {
                    a: leaf(
                        gpu,
                        &pseudo_random(WIDTH * RANK, 4, 0.5),
                        &[WIDTH, RANK],
                        true,
                    ),
                    b: leaf(
                        gpu,
                        &pseudo_random(RANK * CLASSES, 5, 0.5),
                        &[RANK, CLASSES],
                        true,
                    ),
                    alpha: ALPHA,
                    rank: RANK,
                });
                head
            }
        };
        Self { head, hidden }
    }

    /// The tensors whose gradients are compared, with their names.
    fn tensors(&self) -> Vec<(&'static str, Tensor)> {
        let mut all = vec![("hidden", self.hidden.clone())];
        all.push(("weight", self.head.weight.clone()));
        if let Some(bias) = &self.head.bias {
            all.push(("bias", bias.clone()));
        }
        if let Some(lora) = &self.head.lora {
            all.push(("lora a", lora.a.clone()));
            all.push(("lora b", lora.b.clone()));
        }
        all
    }
}

fn targets() -> Vec<u32> {
    (0..ROWS).map(|i| ((i * 7 + 3) % CLASSES) as u32).collect()
}

/// Sequence 0 is supervised from its fifth token on, sequence 1 only at its
/// last two (rows 24 and 25), sequence 2 at every other token. Rows 16 to 23
/// have no supervised row, so a chunk of 8 rows starting at row 16 has
/// none.
fn mixed_mask() -> Vec<u8> {
    (0..ROWS)
        .map(|i| match i / SEQ_LEN {
            0 => u8::from(i >= 4),
            1 => u8::from(i >= 2 * SEQ_LEN - 2),
            _ => u8::from(i % 2 == 0),
        })
        .collect()
}

/// As [`mixed_mask`], but sequence 1 has no supervised token at all.
fn one_empty_sequence_mask() -> Vec<u8> {
    let mut mask = mixed_mask();
    mask[SEQ_LEN..2 * SEQ_LEN].fill(0);
    mask
}

/// Loss and gradients of one run.
struct Outcome {
    per_sequence: Vec<(f32, usize)>,
    total: f32,
    grads: Vec<(&'static str, Vec<f32>)>,
}

/// The loss of `head(hidden)` computed on the whole logits, or over chunks
/// of `chunk` rows, and the gradient of every tensor of the case.
fn run(gpu: &Gpu, kind: Head, mask: &[u8], chunk: Option<usize>) -> Outcome {
    let case = Case::new(gpu, kind);
    let loss = match chunk {
        Some(rows) => chunked_head_cross_entropy(
            &case.head,
            &case.hidden,
            &targets(),
            mask,
            SEQUENCES,
            rows,
            &gpu.engine,
        )
        .expect("valid arguments"),
        None => {
            let logits = case.head.forward(&gpu.engine, &case.hidden);
            masked_cross_entropy_batch(&logits, &targets(), mask, SEQUENCES, &gpu.engine)
        }
    }
    .expect("a supervised position");
    loss.tensor.backward(&gpu.engine);
    let grads = case
        .tensors()
        .into_iter()
        .map(|(name, t)| {
            let grad = t.grad.borrow();
            let values = grad.as_ref().map_or_else(Vec::new, |g| {
                pollster::block_on(g.to_vec(&gpu.ctx)).expect("GPU readback")
            });
            (name, values)
        })
        .collect();
    Outcome {
        per_sequence: loss.per_sequence.clone(),
        total: host(gpu, &loss.tensor)[0],
        grads,
    }
}

fn assert_same(got: &Outcome, want: &Outcome, what: &str) {
    assert_eq!(got.per_sequence.len(), want.per_sequence.len(), "{what}");
    for (s, (g, w)) in got.per_sequence.iter().zip(&want.per_sequence).enumerate() {
        assert_eq!(g.1, w.1, "{what}: tokens of sequence {s}");
        assert!(
            (g.0 - w.0).abs() <= 1e-5,
            "{what}: loss of sequence {s}: {} vs {}",
            g.0,
            w.0
        );
    }
    assert!(
        (got.total - want.total).abs() <= 1e-5,
        "{what}: total loss {} vs {}",
        got.total,
        want.total
    );
    assert_eq!(got.grads.len(), want.grads.len(), "{what}: tensors");
    for ((name, g), (_, w)) in got.grads.iter().zip(&want.grads) {
        assert_eq!(g.len(), w.len(), "{what}: {name} gradient size");
        if w.is_empty() {
            // A frozen tensor gets no gradient from either path.
            continue;
        }
        let worst = g
            .iter()
            .zip(w)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= 1e-5, "{what}: {name} gradient differs by {worst}");
        let largest = w.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(largest > 1e-3, "{what}: {name} gradient is all zero");
    }
}

/// 39 rows in chunks of 8 are four chunks and one of 7. The third chunk
/// (rows 16 to 23) has no supervised row and is skipped.
#[test]
#[ignore = "needs a GPU"]
fn chunked_head_matches_the_whole_head() {
    let gpu = gpu();
    let mask = mixed_mask();
    for kind in [Head::Everything, Head::FrozenWeight] {
        let whole = run(&gpu, kind, &mask, None);
        let chunked = run(&gpu, kind, &mask, Some(8));
        assert_same(&chunked, &whole, "chunks of 8");
        // The skipped chunk gives its rows a zero gradient.
        let hidden = &chunked.grads[0].1;
        assert!(
            hidden[16 * WIDTH..24 * WIDTH].iter().all(|&g| g == 0.0),
            "rows of the skipped chunk"
        );
    }
}

#[test]
#[ignore = "needs a GPU"]
fn every_chunk_size_gives_the_same_loss_and_gradients() {
    let gpu = gpu();
    let mask = mixed_mask();
    let whole = run(&gpu, Head::Everything, &mask, None);
    // One row per chunk, chunks that divide the 13 rows of a sequence and
    // ones that cut across sequences, the whole head in one chunk and a
    // chunk larger than the head.
    for rows in [1, 5, 13, 26, 38, 39, 100] {
        let chunked = run(&gpu, Head::Everything, &mask, Some(rows));
        assert_same(&chunked, &whole, &format!("chunks of {rows}"));
    }
}

#[test]
#[ignore = "needs a GPU"]
fn a_sequence_without_supervised_tokens_has_no_loss_and_no_gradient() {
    let gpu = gpu();
    let mask = one_empty_sequence_mask();
    let whole = run(&gpu, Head::Everything, &mask, None);
    let chunked = run(&gpu, Head::Everything, &mask, Some(8));
    assert_same(&chunked, &whole, "an empty sequence");
    assert_eq!(chunked.per_sequence[1], (0.0, 0));
}

/// The mean loss of each sequence from the host values of the case, in f64:
/// `x @ W + b + (x @ A @ B) * alpha / rank`, then `logsumexp - logit[target]`
/// averaged over the supervised rows.
fn host_losses(mask: &[u8]) -> Vec<f64> {
    let x = pseudo_random(ROWS * WIDTH, 1, 1.0);
    let w = pseudo_random(WIDTH * CLASSES, 2, 0.5);
    let bias = pseudo_random(CLASSES, 3, 0.2);
    let a = pseudo_random(WIDTH * RANK, 4, 0.5);
    let b = pseudo_random(RANK * CLASSES, 5, 0.5);
    let scale = f64::from(ALPHA) / RANK as f64;
    let targets = targets();
    let mut losses = [0.0f64; SEQUENCES];
    let mut tokens = [0usize; SEQUENCES];
    for (row, &supervised) in mask.iter().enumerate().take(ROWS) {
        if supervised == 0 {
            continue;
        }
        let x = &x[row * WIDTH..(row + 1) * WIDTH];
        let low: Vec<f64> = (0..RANK)
            .map(|r| {
                (0..WIDTH)
                    .map(|k| f64::from(x[k]) * f64::from(a[k * RANK + r]))
                    .sum()
            })
            .collect();
        let logits: Vec<f64> = (0..CLASSES)
            .map(|c| {
                let base: f64 = (0..WIDTH)
                    .map(|k| f64::from(x[k]) * f64::from(w[k * CLASSES + c]))
                    .sum();
                let lora: f64 = (0..RANK)
                    .map(|r| low[r] * f64::from(b[r * CLASSES + c]))
                    .sum();
                base + f64::from(bias[c]) + lora * scale
            })
            .collect();
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = max + logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln();
        losses[row / SEQ_LEN] += lse - logits[targets[row] as usize];
        tokens[row / SEQ_LEN] += 1;
    }
    losses
        .iter()
        .zip(&tokens)
        .map(|(l, &t)| l / t.max(1) as f64)
        .collect()
}

/// The loss against an f64 computation on the host, independent of every
/// kernel the head uses.
#[test]
#[ignore = "needs a GPU"]
fn chunked_loss_matches_a_host_computation() {
    let gpu = gpu();
    let mask = mixed_mask();
    let want = host_losses(&mask);
    let got = run(&gpu, Head::Everything, &mask, Some(8));
    for (s, (g, w)) in got.per_sequence.iter().zip(&want).enumerate() {
        assert!(
            (f64::from(g.0) - w).abs() <= 1e-4,
            "sequence {s}: {} vs {w}",
            g.0
        );
    }
}

/// The gradients of the summed mean losses of the sequences with respect to
/// the hidden states, the head weight and the bias, for a head of a weight
/// and a bias alone, in f64 from the host values of the case. For a
/// supervised row `r` of sequence `s` with `n_s` supervised rows,
/// `dz = (softmax(z) - onehot(target)) / n_s`; the hidden gradient is
/// `dz W^T`, the weight gradient `sum x^T dz` and the bias gradient `sum dz`.
/// A row left out of the loss has a zero gradient.
fn host_gradients(mask: &[u8]) -> [Vec<f64>; 3] {
    let x = pseudo_random(ROWS * WIDTH, 1, 1.0);
    let w = pseudo_random(WIDTH * CLASSES, 2, 0.5);
    let bias = pseudo_random(CLASSES, 3, 0.2);
    let targets = targets();
    let mut tokens = [0usize; SEQUENCES];
    for (row, &supervised) in mask.iter().enumerate().take(ROWS) {
        tokens[row / SEQ_LEN] += usize::from(supervised != 0);
    }
    let mut d_hidden = vec![0.0f64; ROWS * WIDTH];
    let mut d_weight = vec![0.0f64; WIDTH * CLASSES];
    let mut d_bias = vec![0.0f64; CLASSES];
    for (row, &supervised) in mask.iter().enumerate().take(ROWS) {
        if supervised == 0 {
            continue;
        }
        let xr = &x[row * WIDTH..(row + 1) * WIDTH];
        let logits: Vec<f64> = (0..CLASSES)
            .map(|c| {
                let dot: f64 = (0..WIDTH)
                    .map(|k| f64::from(xr[k]) * f64::from(w[k * CLASSES + c]))
                    .sum();
                dot + f64::from(bias[c])
            })
            .collect();
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = logits.iter().map(|l| (l - max).exp()).sum();
        let n = tokens[row / SEQ_LEN] as f64;
        let dz: Vec<f64> = (0..CLASSES)
            .map(|c| {
                let p = (logits[c] - max).exp() / total;
                (p - f64::from(u8::from(c == targets[row] as usize))) / n
            })
            .collect();
        for (k, &xk) in xr.iter().enumerate() {
            d_hidden[row * WIDTH + k] = (0..CLASSES)
                .map(|c| dz[c] * f64::from(w[k * CLASSES + c]))
                .sum();
            for (c, &dzc) in dz.iter().enumerate() {
                d_weight[k * CLASSES + c] += f64::from(xk) * dzc;
            }
        }
        for (b, &dzc) in d_bias.iter_mut().zip(&dz) {
            *b += dzc;
        }
    }
    [d_hidden, d_weight, d_bias]
}

/// The gradients of the hidden states, the head weight and the bias against
/// an f64 computation on the host, independent of every kernel the head
/// uses. 39 rows in chunks of 8: a skipped chunk of masked rows, masked rows
/// inside chunks and a last chunk of 7.
#[test]
#[ignore = "needs a GPU"]
fn chunked_gradients_match_a_host_computation() {
    let gpu = gpu();
    let mask = mixed_mask();
    let want = host_gradients(&mask);
    let got = run(&gpu, Head::WeightAndBias, &mask, Some(8));
    assert_eq!(got.grads.len(), 3);
    for ((name, grad), want) in got.grads.iter().zip(&want) {
        assert_eq!(grad.len(), want.len(), "{name} gradient size");
        let worst = grad
            .iter()
            .zip(want)
            .map(|(g, w)| (f64::from(*g) - w).abs())
            .fold(0.0f64, f64::max);
        assert!(worst <= 1e-5, "{name} gradient differs by {worst}");
        let largest = want.iter().map(|w| w.abs()).fold(0.0f64, f64::max);
        assert!(largest > 1e-3, "{name} reference gradient is all zero");
    }
}

/// The loss `ferrotensor eval` reports: the mean over the supervised rows of
/// the whole logits, read to the host, against the same mean with the logits
/// formed over chunks (forward only). The chunk sizes cut the 39 rows at the
/// sequence borders, across them and into a short last chunk, and the mask
/// leaves a whole chunk out.
#[test]
#[ignore = "needs a GPU"]
fn chunked_eval_loss_equals_the_whole_one() {
    let gpu = gpu();
    for kind in [Head::Everything, Head::FrozenWeight] {
        let case = Case::new(&gpu, kind);
        let mask = mixed_mask();
        let logits = ferrotensor_autograd::no_grad(|| case.head.forward(&gpu.engine, &case.hidden));
        let host_logits = host(&gpu, &logits);
        let (want, want_rows) =
            masked_cross_entropy_from_logits(&host_logits, CLASSES, &targets(), &mask)
                .expect("supervised rows");
        for rows in [1, 8, 13, 38, 39, 100] {
            let (got, got_rows) = chunked_head_loss(
                &case.head,
                &case.hidden,
                &targets(),
                &mask,
                rows,
                &gpu.engine,
            )
            .expect("valid arguments")
            .expect("supervised rows");
            assert_eq!(got_rows, want_rows, "chunks of {rows}");
            assert!(
                (got - want).abs() <= 1e-5,
                "chunks of {rows}: loss {got} vs {want}"
            );
        }
        // No supervised row, no loss; no gradient was made on the way.
        let none = chunked_head_loss(
            &case.head,
            &case.hidden,
            &targets(),
            &[0; ROWS],
            8,
            &gpu.engine,
        );
        assert!(matches!(none, Ok(None)));
        for (name, t) in case.tensors() {
            assert!(t.grad.borrow().is_none(), "{name} has a gradient");
        }
    }
}

/// Bytes of the `[rows, classes]` buffers (the logits and their
/// gradient) among the largest buffers live at the peak of one run of the
/// head and the loss, with the whole logits or in chunks of `chunk` rows.
fn peak_logit_bytes(gpu: &Gpu, rows: usize, classes: usize, chunk: Option<usize>) -> u64 {
    let memory = gpu.ctx.memory();
    let head = Linear::new(leaf(
        gpu,
        &pseudo_random(WIDTH * classes, 11, 0.3),
        &[WIDTH, classes],
        false,
    ));
    let hidden = leaf(
        gpu,
        &pseudo_random(rows * WIDTH, 12, 1.0),
        &[rows, WIDTH],
        true,
    );
    let targets: Vec<u32> = (0..rows).map(|i| ((i * 7 + 3) % classes) as u32).collect();
    let mask = vec![1u8; rows];
    gpu.ctx.synchronize();
    memory.reset_peak();
    let loss = match chunk {
        Some(chunk_rows) => {
            chunked_head_cross_entropy(&head, &hidden, &targets, &mask, 1, chunk_rows, &gpu.engine)
                .expect("valid arguments")
        }
        None => {
            let logits = head.forward(&gpu.engine, &hidden);
            masked_cross_entropy_batch(&logits, &targets, &mask, 1, &gpu.engine)
        }
    }
    .expect("supervised positions");
    loss.tensor.backward(&gpu.engine);
    gpu.ctx.synchronize();
    let peak = memory.peak().expect("detailed tracking is on");
    peak.largest
        .iter()
        .filter(|b| b.shape.len() == 2 && b.shape[1] == classes && b.shape[0] != WIDTH)
        .map(|b| b.allocated_bytes)
        .sum()
}

/// 64 rows over 512 classes: the whole logits and their gradient are 256 KiB
/// and a chunk of 20 rows is 80 KiB.
#[test]
#[ignore = "needs a GPU"]
fn a_chunked_head_holds_one_chunk_of_logits() {
    const ROWS_BIG: usize = 64;
    const CLASSES_BIG: usize = 512;
    // Not the hidden width, which the buffer filter tells the head weight by.
    const CHUNK: usize = 20;
    let gpu = gpu();
    // Tracking has to be on before the buffers are made to list them.
    gpu.ctx.memory().set_detailed(true);
    let whole = peak_logit_bytes(&gpu, ROWS_BIG, CLASSES_BIG, None);
    let chunked = peak_logit_bytes(&gpu, ROWS_BIG, CLASSES_BIG, Some(CHUNK));
    let value = HEAD_LOGIT_BYTES_PER_VALUE;
    assert!(
        whole >= (ROWS_BIG * CLASSES_BIG) as u64 * value,
        "whole head: {whole} bytes"
    );
    let bound = (CHUNK * CLASSES_BIG) as u64 * value;
    assert!(
        chunked <= bound,
        "chunked head: {chunked} bytes, bound {bound}"
    );
    assert!(
        chunked >= bound / 2,
        "the logits of one chunk were not counted: {chunked} bytes"
    );
}

/// No GPU: the condition that turns chunking on.
#[test]
fn chunking_starts_above_one_gib_of_logits_and_gradient() {
    assert_eq!(HEAD_CHUNK_MIN_BYTES, 1 << 30);
    assert_eq!(HEAD_LOGIT_BYTES_PER_VALUE, 8);
    // SmolLM2 at the speed gate: 8 x 256 rows over 49,152 classes is
    // 0.75 GiB, and 1 x 256 rows is 96 MiB.
    assert_eq!(head_chunk_rows(8 * 256, 49_152), None);
    assert_eq!(head_chunk_rows(256, 49_152), None);
    // Exactly 1 GiB (2,048 x 65,536 x 8) does not chunk, one more class does.
    assert_eq!(head_chunk_rows(2048, 65_536), None);
    assert_eq!(head_chunk_rows(2048, 65_537), Some(HEAD_CHUNK_ROWS));
    // Qwen (151,936 classes), Llama 3 (128,256) and Gemma (262,208).
    assert_eq!(head_chunk_rows(2048, 151_936), Some(512));
    assert_eq!(head_chunk_rows(2048, 128_256), Some(512));
    assert_eq!(head_chunk_rows(1024, 262_208), Some(512));
    // 512 rows are one chunk already, however large the vocabulary.
    assert_eq!(head_chunk_rows(512, 1_000_000), None);
    assert_eq!(head_chunk_rows(513, 1_000_000), Some(512));
    // A product that overflows u64 still chunks.
    assert_eq!(head_chunk_rows(usize::MAX, usize::MAX), Some(512));
}

/// No GPU: the most rows the head can hold whole is the largest row count
/// `head_chunk_rows` leaves alone, so a planner can count on it.
#[test]
fn the_most_whole_rows_are_the_last_ones_that_do_not_chunk() {
    for vocab in [1, 49_152, 65_536, 128_256, 151_936, 262_208, 1_000_000] {
        let most = max_whole_head_rows(vocab);
        assert_eq!(head_chunk_rows(most, vocab), None, "{vocab} classes");
        // One more row chunks, unless the chunk is no smaller than the rows.
        assert_eq!(
            head_chunk_rows(most + 1, vocab).is_none(),
            most < HEAD_CHUNK_ROWS,
            "{vocab} classes"
        );
    }
    assert_eq!(max_whole_head_rows(151_936), 883);
    assert_eq!(max_whole_head_rows(262_208), HEAD_CHUNK_ROWS);
    assert_eq!(max_whole_head_rows(usize::MAX), HEAD_CHUNK_ROWS);
}

/// Arguments the chunked head cannot run are errors; only "nothing
/// supervised" is `Ok(None)`. The mismatch between a spec and the rows is
/// tested next to `run_chunked_head`, which the trainer's loss step uses.
#[test]
#[ignore = "needs a GPU"]
fn bad_arguments_are_errors_not_an_empty_step() {
    let gpu = gpu();
    let case = Case::new(&gpu, Head::Everything);
    let mask = mixed_mask();
    let loss = |mask: &[u8], sequences: usize, chunk_rows: usize| {
        chunked_head_cross_entropy(
            &case.head,
            &case.hidden,
            &targets(),
            mask,
            sequences,
            chunk_rows,
            &gpu.engine,
        )
        .map(|loss| loss.is_some())
    };
    // The error cases come first. A successful `loss` adds the head's
    // gradients while its chunks run, so these checks need a clean head.
    // 39 rows do not split into 2 sequences, nor into 0.
    assert!(loss(&mask, 2, 8).is_err());
    assert!(loss(&mask, 0, 8).is_err());
    // A chunk of no rows cannot run.
    assert!(loss(&mask, SEQUENCES, 0).is_err());
    let eval = |chunk_rows: usize| {
        chunked_head_loss(
            &case.head,
            &case.hidden,
            &targets(),
            &mask,
            chunk_rows,
            &gpu.engine,
        )
    };
    assert!(eval(0).is_err());
    // An error must leave no gradient behind.
    for (name, t) in case.tensors() {
        assert!(t.grad.borrow().is_none(), "{name} has a gradient");
    }
    // The evaluation runs forward only, so it adds none either.
    assert!(matches!(eval(8), Ok(Some(_))));
    for (name, t) in case.tensors() {
        assert!(t.grad.borrow().is_none(), "{name} has a gradient");
    }
    // The cases that run a step come last, because they add gradients.
    assert!(matches!(loss(&mask, SEQUENCES, 8), Ok(true)));
    assert!(matches!(loss(&[0; ROWS], SEQUENCES, 8), Ok(false)));
}
