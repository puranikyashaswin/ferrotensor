//! Normalisation and softmax kernels.

use super::{grid, groups64, uniform, ComputeEngine};
use crate::{UnifiedTensor, WgpuContext};

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
        self.run(
            ctx,
            &self.rms_norm,
            &[
                (0, input.buffer()),
                (1, weight.buffer()),
                (2, out.buffer()),
                (3, &params),
            ],
            groups64(rows),
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
