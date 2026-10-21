//! Losses, optimisers and gradient utilities.

use super::{groups64, uniform, ComputeEngine};
use crate::{UnifiedTensor, WgpuContext};

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
}
