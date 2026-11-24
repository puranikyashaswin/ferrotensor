//! Losses, optimisers and gradient utilities.

use super::{grid, groups64, uniform, ComputeEngine};
use crate::{UnifiedTensor, WgpuContext};

/// Target id that excludes a row from [`ComputeEngine::masked_cross_entropy`].
pub const MASKED_CE_SKIP: u32 = u32::MAX;

/// Elements summed by one workgroup of
/// [`ComputeEngine::sum_squares_partial`].
pub const SUM_SQUARES_CHUNK: usize = 4096;

/// Most elements one flat buffer of a [`FlatLayout`] may hold: 64 MiB of
/// `f32`, under wgpu's default 128 MiB limit on one storage binding.
pub const FLAT_MAX_ELEMS: usize = 1 << 24;

/// Where each of a list of `f32` tensors starts in one flat buffer.
///
/// Every tensor starts on a [`SUM_SQUARES_CHUNK`] boundary and the gap after
/// it stays zero. A sum of squares over the flat buffer then fills the same
/// chunks, with the same elements in each, as one run per tensor, so its
/// partial sums are bit-equal to theirs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlatLayout {
    offsets: Vec<usize>,
    sizes: Vec<usize>,
    total: usize,
}

impl FlatLayout {
    /// The layout of tensors with `sizes` elements, or `None` when the flat
    /// buffer would hold more than [`FLAT_MAX_ELEMS`] elements.
    pub fn new(sizes: &[usize]) -> Option<Self> {
        let mut offsets = Vec::with_capacity(sizes.len());
        let mut end = 0usize;
        for &n in sizes {
            offsets.push(end);
            let padded = n
                .div_ceil(SUM_SQUARES_CHUNK)
                .checked_mul(SUM_SQUARES_CHUNK)?;
            end = end.checked_add(padded)?;
            if end > FLAT_MAX_ELEMS {
                return None;
            }
        }
        Some(Self {
            offsets,
            sizes: sizes.to_vec(),
            total: end,
        })
    }

    /// Element offset of tensor `i`, or 0 when there is no such tensor.
    pub fn offset(&self, i: usize) -> usize {
        self.offsets.get(i).copied().unwrap_or(0)
    }

    /// Elements of each tensor, as given to [`new`](Self::new).
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// Elements of the flat buffer: a multiple of [`SUM_SQUARES_CHUNK`].
    pub fn total(&self) -> usize {
        self.total
    }
}

/// Hyper-parameters for [`ComputeEngine::adamw_update`]; laid out to match
/// the WGSL uniform.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdamParams {
    /// Learning rate.
    pub lr: f32,
    /// First-moment decay.
    pub beta1: f32,
    /// Second-moment decay.
    pub beta2: f32,
    /// Denominator epsilon.
    pub eps: f32,
    /// Decoupled weight decay.
    pub weight_decay: f32,
    /// 1-based step number, used for bias correction.
    pub t: f32,
    /// Padding to a 16-byte multiple; set to zero.
    pub _pad: [f32; 2],
}

impl ComputeEngine {
    /// In-place SGD step: `weight -= lr * grad`.
    pub fn sgd_update(
        &self,
        ctx: &WgpuContext,
        weight: &UnifiedTensor,
        grad: &UnifiedTensor,
        lr: f32,
    ) {
        let lr_buf = uniform(ctx, bytemuck::cast_slice(&[lr]));
        self.run(
            ctx,
            &self.sgd,
            &[(0, weight.buffer()), (1, grad.buffer()), (2, &lr_buf)],
            groups64(weight.shape.iter().product()),
        );
    }

    /// In-place AdamW step, updating `weight` and the moment buffers `m`
    /// and `v`.
    pub fn adamw_update(
        &self,
        ctx: &WgpuContext,
        weight: &UnifiedTensor,
        grad: &UnifiedTensor,
        m: &UnifiedTensor,
        v: &UnifiedTensor,
        params: &AdamParams,
    ) {
        let params_buf = uniform(ctx, bytemuck::bytes_of(params));
        self.run(
            ctx,
            &self.adamw,
            &[
                (0, weight.buffer()),
                (1, grad.buffer()),
                (3, m.buffer()),
                (4, v.buffer()),
                (5, &params_buf),
            ],
            groups64(weight.shape.iter().product()),
        );
    }

    /// Copies all of `src` into the flat buffer `flat` at element `offset`.
    pub fn copy_into_flat(
        &self,
        ctx: &WgpuContext,
        src: &UnifiedTensor,
        flat: &UnifiedTensor,
        offset: usize,
    ) {
        self.copy_tensor(ctx, src, flat, offset * 4);
    }

    /// Fills all of `dst` from the flat buffer `flat`, starting at element
    /// `offset`.
    pub fn copy_from_flat(
        &self,
        ctx: &WgpuContext,
        flat: &UnifiedTensor,
        offset: usize,
        dst: &UnifiedTensor,
    ) {
        ctx.record(|encoder| {
            encoder.copy_buffer_to_buffer(
                flat.buffer(),
                (offset * 4) as u64,
                dst.buffer(),
                0,
                dst.size_in_bytes(),
            );
        });
    }

    /// Per-element squared error `out = 0.5 * (pred - target)^2` (not reduced).
    pub fn mse_loss(
        &self,
        ctx: &WgpuContext,
        pred: &UnifiedTensor,
        target: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.binary(ctx, &self.mse_loss, pred, target, out);
    }

    /// Gradient of [`mse_loss`](Self::mse_loss): `grad_in = pred - target`.
    pub fn mse_loss_backward(
        &self,
        ctx: &WgpuContext,
        pred: &UnifiedTensor,
        target: &UnifiedTensor,
        grad_in: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.mse_loss_bw,
            &[
                (0, pred.buffer()),
                (1, target.buffer()),
                (3, grad_in.buffer()),
            ],
            groups64(pred.shape.iter().product()),
        );
    }

    /// Cross-entropy loss of one logits vector: `loss_out[0] = -log(softmax(logits)[target])`.
    pub fn cross_entropy_loss(
        &self,
        ctx: &WgpuContext,
        logits: &UnifiedTensor,
        target: u32,
        loss_out: &UnifiedTensor,
    ) {
        let vocab_size = logits.shape.iter().product::<usize>() as u32;
        let params = uniform(ctx, bytemuck::cast_slice(&[target, vocab_size, 0, 0]));
        self.run(
            ctx,
            &self.ce_loss,
            &[(0, logits.buffer()), (1, loss_out.buffer()), (2, &params)],
            (1, 1, 1),
        );
    }

    /// Cross-entropy loss backward: gradient = softmax(logits) - one_hot(target)
    pub fn cross_entropy_backward(
        &self,
        ctx: &WgpuContext,
        logits: &UnifiedTensor,
        target: u32,
        grad_out: &UnifiedTensor,
    ) {
        let vocab_size = logits.shape.iter().product::<usize>() as u32;
        let params = uniform(ctx, bytemuck::cast_slice(&[target, vocab_size, 0, 0]));
        self.run(
            ctx,
            &self.ce_loss_bw,
            &[(0, logits.buffer()), (1, grad_out.buffer()), (2, &params)],
            groups64(vocab_size as usize),
        );
    }

    /// Compute squared elements (for gradient norm computation)
    pub fn gradient_norm_squared(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.grad_norm,
            &[(0, input.buffer()), (1, out.buffer())],
            groups64(input.shape.iter().product()),
        );
    }

    /// Clip gradients by scaling: grad *= min(1.0, max_norm / actual_norm)
    pub fn clip_grad_by_norm(&self, ctx: &WgpuContext, grad: &UnifiedTensor, scale: f32) {
        let params = uniform(ctx, bytemuck::cast_slice(&[scale]));
        self.run(
            ctx,
            &self.grad_clip,
            &[(0, grad.buffer()), (1, &params)],
            groups64(grad.shape.iter().product()),
        );
    }

    /// Gradient accumulation: accum_grad += new_grad
    /// Essential for micro-batching on memory-constrained devices
    pub fn gradient_accumulate(
        &self,
        ctx: &WgpuContext,
        accum_grad: &UnifiedTensor,
        new_grad: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.grad_accum,
            &[(0, accum_grad.buffer()), (1, new_grad.buffer())],
            groups64(accum_grad.shape.iter().product()),
        );
    }

    /// Gradient scaling: grad *= scale_factor
    /// Used to average gradients after accumulation
    pub fn gradient_scale(&self, ctx: &WgpuContext, grad: &UnifiedTensor, scale: f32) {
        let params = uniform(ctx, bytemuck::cast_slice(&[scale]));
        self.run(
            ctx,
            &self.grad_scale,
            &[(0, grad.buffer()), (1, &params)],
            groups64(grad.shape.iter().product()),
        );
    }

    /// Masked cross-entropy of `logits` (`[rows, vocab]`, any leading
    /// dimensions) against one target id per row, without leaving the GPU.
    ///
    /// `targets` holds `rows` `u32` ids; [`MASKED_CE_SKIP`] leaves a row out.
    /// Writes each row's loss (log-sum-exp minus the target logit, 0 for a
    /// skipped row) to `row_loss` (`[rows]`) and `(softmax - onehot) * inv`
    /// (zero for a skipped row) to `grad` (the shape of `logits`). With
    /// `inv = 1 / supervised rows` the gradient is that of the mean loss.
    pub fn masked_cross_entropy(
        &self,
        ctx: &WgpuContext,
        logits: &UnifiedTensor,
        targets: &UnifiedTensor,
        inv: f32,
        grad: &UnifiedTensor,
        row_loss: &UnifiedTensor,
    ) {
        let vocab = *logits.shape.last().unwrap_or(&1);
        let rows = logits.shape.iter().product::<usize>() / vocab.max(1);
        let weights = UnifiedTensor::new(ctx, &vec![inv; rows.max(1)], &[rows.max(1)]);
        self.masked_cross_entropy_weighted(ctx, logits, targets, &weights, grad, row_loss);
    }

    /// [`masked_cross_entropy`](Self::masked_cross_entropy) with a weight per
    /// row instead of one `inv`: row `r`'s gradient is
    /// `(softmax - onehot) * weights[r]`. Giving each row `1 / supervised
    /// rows of its own sequence` makes a batch of sequences produce the sum
    /// of their per-sequence mean-loss gradients.
    pub fn masked_cross_entropy_weighted(
        &self,
        ctx: &WgpuContext,
        logits: &UnifiedTensor,
        targets: &UnifiedTensor,
        weights: &UnifiedTensor,
        grad: &UnifiedTensor,
        row_loss: &UnifiedTensor,
    ) {
        let vocab = *logits.shape.last().unwrap_or(&1);
        let rows = logits.shape.iter().product::<usize>() / vocab.max(1);
        if rows == 0 || vocab == 0 {
            return;
        }
        let params = uniform(
            ctx,
            bytemuck::cast_slice(&[vocab as u32, rows as u32, 0, 0]),
        );
        self.run(
            ctx,
            &self.masked_ce,
            &[
                (0, logits.buffer()),
                (1, targets.buffer()),
                (2, grad.buffer()),
                (3, row_loss.buffer()),
                (4, &params),
                (5, weights.buffer()),
            ],
            grid(rows as u32),
        );
    }

    /// Sums the squares of `input` in chunks of [`SUM_SQUARES_CHUNK`]
    /// elements, writing chunk `b`'s sum to `out[offset + b]`. Returns the
    /// number of chunks written; adding them up gives the squared L2 norm.
    pub fn sum_squares_partial(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        out: &UnifiedTensor,
        offset: usize,
    ) -> usize {
        let n = input.shape.iter().product::<usize>();
        let chunks = n.div_ceil(SUM_SQUARES_CHUNK);
        if chunks == 0 {
            return 0;
        }
        let params = uniform(ctx, bytemuck::cast_slice(&[n as u32, offset as u32, 0, 0]));
        self.run(
            ctx,
            &self.sum_squares,
            &[(0, input.buffer()), (1, out.buffer()), (2, &params)],
            grid(chunks as u32),
        );
        chunks
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::{FlatLayout, FLAT_MAX_ELEMS, SUM_SQUARES_CHUNK};

    #[test]
    fn flat_layout_starts_every_tensor_on_a_chunk_boundary() {
        let layout = FlatLayout::new(&[5000, 7, 4096, 0, 4097]).expect("fits");
        assert_eq!(layout.offset(0), 0);
        assert_eq!(layout.offset(1), 8192);
        assert_eq!(layout.offset(2), 12288);
        assert_eq!(layout.offset(3), 16384);
        assert_eq!(layout.offset(4), 16384);
        assert_eq!(layout.total(), 16384 + 8192);
        assert_eq!(layout.sizes(), &[5000, 7, 4096, 0, 4097]);
        assert_eq!(layout.total() % SUM_SQUARES_CHUNK, 0);
        // An index past the end has no tensor.
        assert_eq!(layout.offset(5), 0);
    }

    #[test]
    fn flat_layout_refuses_more_than_the_element_cap() {
        let half = FLAT_MAX_ELEMS / 2;
        assert!(FlatLayout::new(&[half, half]).is_some());
        assert!(FlatLayout::new(&[half, half + 1]).is_none());
        assert!(FlatLayout::new(&[usize::MAX]).is_none());
    }
}
