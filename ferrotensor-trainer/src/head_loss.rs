//! `lm_head` and the masked cross-entropy over chunks of rows.
//!
//! The logits of a long sequence over a large vocabulary are the biggest
//! buffers of a training step: 2,048 rows over Qwen's 151,936 classes are
//! 1.2 GB as f32, and the kernel writes a gradient of the same size. This
//! module forms them for one chunk of rows at a time. For each chunk it
//! projects the chunk's hidden states, runs the loss kernel, runs backward
//! through the head (adding into the gradients of the head's weight, bias
//! and LoRA adapter, which are not chunk-sized), keeps the gradient of the
//! chunk's hidden states and drops the chunk's logits before the next chunk
//! starts. The result is one loss node on the full hidden states, which
//! carries the collected gradient into the rest of the model.
//!
//! The head's own gradients are therefore added when the chunks run, not
//! when `backward` is called on the loss.

use crate::{
    ce_kernel, ce_loss_tensor, masked_ce_rows, BatchLoss, CeRows, GpuCrossEntropy, TrainError,
    MASKED_CE_SKIP,
};
use ferrotensor_autograd::{no_grad, Tensor};
use ferrotensor_backend::{ComputeEngine, DType, MemoryCategory, UnifiedTensor};
use ferrotensor_graph::Linear;

/// Bytes the loss kernel holds per logit: the f32 logits and their f32
/// gradient.
pub const HEAD_LOGIT_BYTES_PER_VALUE: u64 = 8;

/// The logits and their gradient must take more than this for
/// [`head_chunk_rows`] to chunk the head: 1 GiB. SmolLM2 at 8 sequences of
/// 256 tokens (2,048 rows over 49,152 classes, 0.75 GiB) stays below it, so
/// the runs of the speed benchmarks keep the unchunked path, which makes one
/// matmul for the whole head instead of one per chunk. A model with a
/// 128,000-class vocabulary or more passes it from about 1,000 rows.
pub const HEAD_CHUNK_MIN_BYTES: u64 = 1 << 30;

/// Rows per chunk when the head is chunked. At 512 rows the logits and
/// their gradient take 0.6 GB for Qwen's vocabulary and 1.1 GB for Gemma's
/// (262,208), and each chunk is still a matmul with enough rows to fill the
/// GPU.
pub const HEAD_CHUNK_ROWS: usize = 512;

/// Rows per chunk for a head of `rows` rows over `vocab` classes, or `None`
/// when the logits and their gradient (`rows * vocab * 8` bytes) are no more
/// than [`HEAD_CHUNK_MIN_BYTES`] and are best formed whole.
pub fn head_chunk_rows(rows: usize, vocab: usize) -> Option<usize> {
    let bytes = (rows as u64)
        .saturating_mul(vocab as u64)
        .saturating_mul(HEAD_LOGIT_BYTES_PER_VALUE);
    (bytes > HEAD_CHUNK_MIN_BYTES && rows > HEAD_CHUNK_ROWS).then_some(HEAD_CHUNK_ROWS)
}

/// The most rows of a head over `vocab` classes that [`head_chunk_rows`]
/// leaves unchunked: whatever the number of rows, the logits of a head never
/// hold more rows than this at once. A planner that does not know how many
/// rows the trainer gathers (the supervised rows of a micro-batch, at most
/// all of its positions) counts this many at most.
pub fn max_whole_head_rows(vocab: usize) -> usize {
    let per_row = (vocab as u64)
        .saturating_mul(HEAD_LOGIT_BYTES_PER_VALUE)
        .max(1);
    let by_size = usize::try_from(HEAD_CHUNK_MIN_BYTES / per_row).unwrap_or(usize::MAX);
    by_size.max(HEAD_CHUNK_ROWS)
}

/// The width of `hidden`, the number of classes of `head` and the number of
/// rows of `hidden`, or the reason the head cannot run over chunks of
/// `chunk_rows` rows of `hidden`.
fn head_dims(
    head: &Linear,
    hidden: &Tensor,
    chunk_rows: usize,
) -> Result<(usize, usize, usize), TrainError> {
    let width = hidden.data.shape.last().copied().unwrap_or(0);
    let vocab = head.weight.data.shape.last().copied().unwrap_or(0);
    if chunk_rows == 0 {
        return Err(TrainError("a chunk needs at least one row".into()));
    }
    if width == 0 || head.weight.data.shape.first() != Some(&width) {
        return Err(TrainError(format!(
            "hidden states of width {width} do not fit a head of shape {:?}",
            head.weight.data.shape
        )));
    }
    if hidden.data.dtype != DType::FP32 {
        return Err(TrainError(
            "the chunked head needs f32 hidden states".into(),
        ));
    }
    let rows = hidden.data.shape.iter().product::<usize>() / width;
    Ok((width, vocab, rows))
}

/// Whether `rows` rows of a head over `vocab` classes make up `sequences`
/// equally long sequences. A caller that has passed this knows that a
/// `None` from [`masked_ce_rows`] means that no row is supervised.
pub(crate) fn check_row_split(
    rows: usize,
    vocab: usize,
    sequences: usize,
) -> Result<(), TrainError> {
    if vocab == 0 {
        return Err(TrainError("the head has no classes".into()));
    }
    if sequences == 0 || !rows.is_multiple_of(sequences) {
        return Err(TrainError(format!(
            "{rows} rows do not split into {sequences} equally long sequences"
        )));
    }
    Ok(())
}

/// Runs `head` and the weighted cross-entropy `spec` over chunks of
/// `chunk_rows` rows of `hidden` (`[.., width]`, f32, one row per entry of
/// `spec`). The returned `grad` is `d loss / d hidden`, shaped like `hidden`
/// (a one-element placeholder when `hidden` needs no gradient), and
/// `row_loss` holds every row's loss. The head's trainable tensors have had
/// the gradient of every chunk added.
///
/// A chunk whose rows are all left out of the loss is skipped: its gradient
/// is zero.
///
/// # Errors
/// If `spec` does not have one target and one weight for each row of
/// `hidden`.
pub(crate) fn run_chunked_head(
    head: &Linear,
    hidden: &Tensor,
    spec: CeRows,
    chunk_rows: usize,
    engine: &ComputeEngine,
) -> Result<GpuCrossEntropy, TrainError> {
    let ctx = &hidden.ctx;
    let width = hidden.data.shape.last().copied().unwrap_or(1).max(1);
    let rows = hidden.data.shape.iter().product::<usize>() / width;
    if spec.targets.len() != rows || spec.weights.len() != rows {
        return Err(TrainError(format!(
            "{} targets and {} weights for {rows} rows of hidden states",
            spec.targets.len(),
            spec.weights.len()
        )));
    }
    let chunk_rows = chunk_rows.max(1);
    // Everything but a chunk's logits and their gradient is scratch: the
    // memory report then shows the logits alone as loss.
    let _scratch = ctx.memory().scope(MemoryCategory::BackwardScratch);
    let table = hidden.data.reshaped(&[rows, width]);
    let grad = if hidden.requires_grad {
        UnifiedTensor::zeros_on_gpu(ctx, &hidden.data.shape)
    } else {
        UnifiedTensor::empty(ctx, &[1])
    };
    let row_loss = UnifiedTensor::zeros_on_gpu(ctx, &[rows]);
    let mut start = 0;
    while start < rows {
        let end = (start + chunk_rows).min(rows);
        if spec.targets[start..end]
            .iter()
            .any(|&t| t != MASKED_CE_SKIP)
        {
            let (chunk_grad, chunk_loss) = run_chunk(
                head,
                &table,
                hidden.requires_grad,
                start..end,
                (&spec.targets[start..end], &spec.weights[start..end]),
                engine,
            );
            let f32_bytes = std::mem::size_of::<f32>();
            if let Some(g) = chunk_grad {
                engine.copy_tensor(ctx, &g, &grad, start * width * f32_bytes);
            }
            engine.copy_tensor(ctx, &chunk_loss, &row_loss, start * f32_bytes);
        }
        start = end;
    }
    Ok(GpuCrossEntropy {
        grad,
        row_loss,
        ranges: spec.ranges,
        tokens: spec.tokens,
    })
}

/// One chunk: the logits of rows `rows` of `table` (`[all rows, width]`),
/// their loss and gradient for `(targets, weights)`, and backward through
/// `head`. Returns the gradient of the chunk's hidden states (when
/// `needs_grad`) and the loss of each row. The logits are dropped on return.
fn run_chunk(
    head: &Linear,
    table: &UnifiedTensor,
    needs_grad: bool,
    rows: std::ops::Range<usize>,
    (targets, weights): (&[u32], &[f32]),
    engine: &ComputeEngine,
) -> (Option<UnifiedTensor>, UnifiedTensor) {
    let ctx = &head.weight.ctx;
    let width = table.shape.last().copied().unwrap_or(1);
    let n = rows.len();
    let ids: Vec<u32> = (rows.start as u32..rows.end as u32).collect();
    let id_buf = UnifiedTensor::from_bytes(ctx, bytemuck::cast_slice(&ids), &[n]);
    let part = UnifiedTensor::empty(ctx, &[n, width]);
    engine.embedding_gather(ctx, table, &id_buf, &part);
    // A leaf of its own: backward through the head stops here, and the rest
    // of the model's graph waits for the loss node of the whole head.
    let input = if needs_grad {
        Tensor::new(ctx.clone(), part)
    } else {
        Tensor::frozen(ctx.clone(), part)
    };
    let logits = {
        let _forward = ctx.memory().scope(MemoryCategory::Activations);
        head.forward(engine, &input)
    };
    logits.data.set_memory_category(MemoryCategory::Loss);
    let (logits_grad, row_loss) = ce_kernel(&logits, targets, weights, engine);
    logits_grad.set_memory_category(MemoryCategory::Loss);
    // The weights already carry the seed, so the loss node returns its
    // gradient as it is (the value of the seed is not read).
    let value = UnifiedTensor::new(ctx, &[0.0], &[1]);
    let loss = ce_loss_tensor(&logits, logits_grad, value, Some(1.0));
    loss.backward(engine);
    let chunk_grad = input.grad.borrow_mut().take();
    (chunk_grad, row_loss)
}

/// Mean cross-entropy of `head` applied to `hidden` (`[.., width]`, f32)
/// against `target_ids` where `loss_mask` is 1, for `sequences` equally long
/// sequences, as [`masked_cross_entropy_batch`](crate::masked_cross_entropy_batch)
/// gives it for the logits `head.forward(hidden)`, but with the logits formed
/// `chunk_rows` rows at a time.
///
/// The loss and every gradient equal the unchunked ones to rounding. The
/// gradients of the head's weight, bias and LoRA adapter are added to their
/// `grad` slots by this call, whatever happens to the returned tensor; call
/// `backward` on it (with its default seed of 1) to send the gradient on to
/// `hidden`. Returns `Ok(None)` when no position is supervised.
///
/// # Errors
/// If `chunk_rows` is 0, `hidden` is not f32 or does not match the head's
/// width, or the rows do not split into `sequences`.
#[expect(
    clippy::expect_used,
    reason = "a readback fails only on device loss, and the caller has no error path"
)]
pub fn chunked_head_cross_entropy(
    head: &Linear,
    hidden: &Tensor,
    target_ids: &[u32],
    loss_mask: &[u8],
    sequences: usize,
    chunk_rows: usize,
    engine: &ComputeEngine,
) -> Result<Option<BatchLoss>, TrainError> {
    let (_, vocab, rows) = head_dims(head, hidden, chunk_rows)?;
    check_row_split(rows, vocab, sequences)?;
    let Some(spec) = masked_ce_rows(rows, vocab, target_ids, loss_mask, sequences, 1.0) else {
        return Ok(None);
    };
    let ce = run_chunked_head(head, hidden, spec, chunk_rows, engine)?;
    let ctx = &hidden.ctx;
    let losses =
        pollster::block_on(ce.row_loss.to_vec(ctx)).expect("GPU device lost during readback");
    let per_sequence = ce.per_sequence(&losses);
    let total: f32 = per_sequence.iter().map(|&(l, _)| l).sum();
    let data = UnifiedTensor::new(ctx, &[total], &[1]);
    let tensor = ce_loss_tensor(hidden, ce.grad, data, Some(1.0));
    Ok(Some(BatchLoss {
        tensor,
        per_sequence,
    }))
}

/// Average cross-entropy of `head` applied to `hidden` (`[.., width]`, f32)
/// against `target_ids` where `loss_mask` is not 0, with no gradient and
/// the logits formed `chunk_rows` rows at a time: the loss of
/// [`masked_cross_entropy_from_logits`](crate::masked_cross_entropy_from_logits)
/// for the logits `head.forward(hidden)`, without the whole logits on the
/// GPU or on the host. Returns the mean and the number of supervised rows,
/// or `Ok(None)` when no row is supervised.
///
/// # Errors
/// If `chunk_rows` is 0, or `hidden` is not f32 or does not match the
/// head's width.
///
/// A chunk whose rows are all left out is skipped. The kernel still writes
/// a gradient for the rows of a chunk, which is dropped with it.
#[expect(
    clippy::expect_used,
    reason = "a readback fails only on device loss, and the caller has no error path"
)]
pub fn chunked_head_loss(
    head: &Linear,
    hidden: &Tensor,
    target_ids: &[u32],
    loss_mask: &[u8],
    chunk_rows: usize,
    engine: &ComputeEngine,
) -> Result<Option<(f32, usize)>, TrainError> {
    let (width, vocab, rows) = head_dims(head, hidden, chunk_rows)?;
    check_row_split(rows, vocab, 1)?;
    let ctx = &hidden.ctx;
    let Some(spec) = masked_ce_rows(rows, vocab, target_ids, loss_mask, 1, 1.0) else {
        return Ok(None);
    };
    let table = hidden.data.reshaped(&[rows, width]);
    let row_loss = UnifiedTensor::zeros_on_gpu(ctx, &[rows]);
    let f32_bytes = std::mem::size_of::<f32>();
    let mut start = 0;
    while start < rows {
        let end = (start + chunk_rows).min(rows);
        if spec.targets[start..end]
            .iter()
            .any(|&t| t != MASKED_CE_SKIP)
        {
            let ids: Vec<u32> = (start as u32..end as u32).collect();
            let id_buf = UnifiedTensor::from_bytes(ctx, bytemuck::cast_slice(&ids), &[ids.len()]);
            let part = UnifiedTensor::empty(ctx, &[ids.len(), width]);
            engine.embedding_gather(ctx, &table, &id_buf, &part);
            let chunk_loss = no_grad(|| {
                let logits = head.forward(engine, &Tensor::frozen(ctx.clone(), part));
                let (_grad, loss) = ce_kernel(
                    &logits,
                    &spec.targets[start..end],
                    &spec.weights[start..end],
                    engine,
                );
                loss
            });
            engine.copy_tensor(ctx, &chunk_loss, &row_loss, start * f32_bytes);
        }
        start = end;
    }
    let losses = pollster::block_on(row_loss.to_vec(ctx)).expect("GPU device lost during readback");
    let tokens: usize = spec.tokens.iter().sum();
    let sum: f64 = losses.iter().map(|&l| f64::from(l)).sum();
    Ok(Some(((sum / tokens as f64) as f32, tokens)))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use ferrotensor_backend::WgpuContext;
    use std::sync::Arc;

    fn leaf(ctx: &Arc<WgpuContext>, data: &[f32], shape: &[usize], trainable: bool) -> Tensor {
        let data = UnifiedTensor::new(ctx, data, shape);
        if trainable {
            Tensor::new(ctx.clone(), data)
        } else {
            Tensor::frozen(ctx.clone(), data)
        }
    }

    /// A spec with a number of targets or weights other than the number of
    /// rows is an error, so that `loss_and_step` cannot take it for a step
    /// with no supervised token.
    #[test]
    #[ignore = "needs a GPU"]
    fn a_spec_that_does_not_fit_the_rows_is_an_error() {
        let ctx = Arc::new(pollster::block_on(WgpuContext::new()));
        let engine = ComputeEngine::new(&ctx);
        let head = Linear::new(leaf(&ctx, &[0.1; 6], &[2, 3], false));
        let hidden = leaf(&ctx, &[0.5; 8], &[4, 2], true);
        let spec = |targets: usize, weights: usize| CeRows {
            targets: vec![1; targets],
            weights: vec![0.25; weights],
            ranges: std::iter::once(0..4).collect(),
            tokens: vec![4],
        };
        for (targets, weights) in [(3, 4), (4, 3), (5, 4), (4, 5), (0, 0)] {
            let result = run_chunked_head(&head, &hidden, spec(targets, weights), 2, &engine);
            assert!(
                result.is_err(),
                "{targets} targets and {weights} weights for 4 rows"
            );
        }
        let ce = run_chunked_head(&head, &hidden, spec(4, 4), 2, &engine).expect("a fitting spec");
        assert_eq!(ce.tokens, [4]);
    }
}
