//! Normalisation and softmax kernels.

use super::{grid, groups64, uniform, ComputeEngine};
use crate::{UnifiedTensor, WgpuContext};

/// Whether a norm row can run in the vec4 kernels, which keep the row in the
/// registers of the 256 threads of one workgroup: two vec4 per thread.
fn row_fits_registers(size: usize) -> bool {
    size.is_multiple_of(4) && size <= 2048
}

impl ComputeEngine {
    /// RMSNorm over the last dimension with eps = 1e-5:
    /// `out = x / sqrt(mean(x^2) + eps) * weight`.
    pub fn rms_norm(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.rms_norm_eps(ctx, input, weight, out, 1e-5);
    }

    /// RMSNorm over the last dimension with a caller-chosen `eps`.
    pub fn rms_norm_eps(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        out: &UnifiedTensor,
        eps: f32,
    ) {
        let size = *input.shape.last().unwrap_or(&1);
        let rows = input.shape.iter().product::<usize>() / size;
        let params = uniform(
            ctx,
            bytemuck::cast_slice(&[size as u32, rows as u32, eps.to_bits(), 0]),
        );
        let kernel = if row_fits_registers(size) {
            &self.rms_norm_v4
        } else {
            &self.rms_norm
        };
        self.run(
            ctx,
            kernel,
            &[
                (0, input.buffer()),
                (1, weight.buffer()),
                (2, out.buffer()),
                (3, &params),
            ],
            grid(rows as u32),
        );
    }

    /// Gradients of [`rms_norm_eps`](Self::rms_norm_eps) with respect to
    /// `input` and `weight`, given the output gradient `grad_out`.
    ///
    /// `grad_input` has `input`'s shape; `grad_weight` has `weight`'s and is
    /// summed over all rows, and `None` skips it (a frozen weight).
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        grad_out: &UnifiedTensor,
        grad_input: &UnifiedTensor,
        grad_weight: Option<&UnifiedTensor>,
        eps: f32,
    ) {
        self.rms_norm_backward_beta(
            ctx,
            input,
            weight,
            grad_out,
            (grad_input, 0.0),
            grad_weight,
            eps,
        );
    }

    /// [`rms_norm_backward`](Self::rms_norm_backward) that adds into
    /// `grad_input`, given as `(tensor, beta)`: the tensor ends as
    /// `norm_gradient + beta * tensor`.
    ///
    /// With a residual stream the tensor holds the gradient the skip
    /// connection already delivered, so `beta = 1` replaces the separate
    /// `add` dispatch that sums the two. With `beta = 0` the kernel never
    /// reads the tensor, which may then be uninitialised. `grad_weight` is
    /// always overwritten, never accumulated.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward_beta(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        grad_out: &UnifiedTensor,
        grad_input: (&UnifiedTensor, f32),
        grad_weight: Option<&UnifiedTensor>,
        eps: f32,
    ) {
        let (grad_input, beta) = grad_input;
        let size = *input.shape.last().unwrap_or(&1);
        let rows = input.shape.iter().product::<usize>() / size.max(1);
        if rows == 0 || size == 0 {
            return;
        }
        let inv_rms = UnifiedTensor::empty(ctx, &[rows]);
        let params = uniform(
            ctx,
            bytemuck::cast_slice(&[size as u32, rows as u32, eps.to_bits(), beta.to_bits()]),
        );
        let kernel = if row_fits_registers(size) {
            &self.rms_norm_bw_v4
        } else {
            &self.rms_norm_bw
        };
        self.run(
            ctx,
            kernel,
            &[
                (0, input.buffer()),
                (1, grad_out.buffer()),
                (2, weight.buffer()),
                (3, grad_input.buffer()),
                (4, inv_rms.buffer()),
                (5, &params),
            ],
            grid(rows as u32),
        );
        let Some(grad_weight) = grad_weight else {
            return;
        };
        let params = uniform(ctx, bytemuck::cast_slice(&[size as u32, rows as u32, 0, 0]));
        self.run(
            ctx,
            &self.rms_norm_bw_weight,
            &[
                (0, input.buffer()),
                (1, grad_out.buffer()),
                (2, inv_rms.buffer()),
                (3, grad_weight.buffer()),
                (4, &params),
            ],
            groups64(size),
        );
    }

    /// Fused RMSNorm + Residual: out = RMSNorm(x + residual) * weight
    pub fn rms_norm_residual(
        &self,
        ctx: &WgpuContext,
        x: &UnifiedTensor,
        residual: &UnifiedTensor,
        weight: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let size = x.shape.iter().product::<usize>() as u32;
        let params = uniform(ctx, bytemuck::cast_slice(&[size]));
        self.run(
            ctx,
            &self.rms_residual,
            &[
                (0, x.buffer()),
                (1, residual.buffer()),
                (2, weight.buffer()),
                (3, out.buffer()),
                (4, &params),
            ],
            (1, 1, 1),
        );
    }

    /// Softmax over the last dimension of `input`.
    pub fn softmax(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        let size = *input.shape.last().unwrap_or(&1);
        let rows = input.shape.iter().product::<usize>() / size;
        let params = uniform(ctx, bytemuck::cast_slice(&[size as u32, rows as u32]));
        self.run(
            ctx,
            &self.softmax,
            &[(0, input.buffer()), (1, out.buffer()), (2, &params)],
            grid(rows as u32),
        );
    }

    /// Gradient of softmax over the last dimension, given the forward
    /// output `final_out`: `grad_in = s * (grad_out - sum(grad_out * s))`.
    pub fn softmax_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        final_out: &UnifiedTensor,
        grad_in: &UnifiedTensor,
    ) {
        let size = *grad_out.shape.last().unwrap_or(&1);
        let rows = grad_out.shape.iter().product::<usize>() / size;
        let params = uniform(ctx, bytemuck::cast_slice(&[size as u32, rows as u32]));
        self.run(
            ctx,
            &self.softmax_bw,
            &[
                (2, &params),
                (3, grad_out.buffer()),
                (4, final_out.buffer()),
                (5, grad_in.buffer()),
            ],
            grid(rows as u32),
        );
    }
}
