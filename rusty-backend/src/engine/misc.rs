//! Top-k selection and depthwise 1-D convolution.

use super::{groups64, uniform, ComputeEngine};
use crate::{UnifiedTensor, WgpuContext};

impl ComputeEngine {
    pub fn top_k(&self, ctx: &WgpuContext, input: &UnifiedTensor, out_idx: &UnifiedTensor, k: u32) {
        let params = uniform(ctx, bytemuck::cast_slice(&[k]));
        self.run(
            ctx,
            &self.top_k,
            &[(0, input.buffer()), (1, out_idx.buffer()), (2, &params)],
            (1, 1, 1),
        );
    }

    pub fn conv1d(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        bias: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let dim = weight.shape[0];
        let seq_len = input.shape[input.shape.len() - 1];
        let params = uniform(ctx, bytemuck::cast_slice(&[dim as u32, seq_len as u32]));
        self.run(
            ctx,
            &self.conv1d,
            &[
                (0, input.buffer()),
                (1, weight.buffer()),
                (2, out.buffer()),
                (3, bias.buffer()),
                (4, &params),
            ],
            groups64(dim * seq_len),
        );
    }
}
