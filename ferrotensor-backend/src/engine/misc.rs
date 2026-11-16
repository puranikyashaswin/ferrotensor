//! Top-k selection and depthwise 1-D convolution.

use super::{groups64, ComputeEngine};
use crate::{UnifiedTensor, WgpuContext};

impl ComputeEngine {
    /// Writes the indices of the `k` largest values of `input`, largest first,
    /// as `u32`s into `out_idx`. Runs on a single GPU thread; meant for sampling
    /// from one logits vector.
    pub fn top_k(&self, ctx: &WgpuContext, input: &UnifiedTensor, out_idx: &UnifiedTensor, k: u32) {
        let params = ctx.uniform_buffer(bytemuck::cast_slice(&[k]));
        self.run(
            ctx,
            &self.top_k,
            &[
                (0, input.shared_buffer()),
                (1, out_idx.shared_buffer()),
                (2, &params),
            ],
            (1, 1, 1),
        );
    }

    /// Depthwise 1-D convolution with kernel size 3 and padding 1:
    /// `out[d, s] = bias[d] + sum_k weight[d, k] * input[d, s + k - 1]`, with
    /// `input`/`out` `[dim, seq_len]`, `weight` `[dim, 3]` and `bias` `[dim]`.
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
        let params = ctx.uniform_buffer(bytemuck::cast_slice(&[dim as u32, seq_len as u32]));
        self.run(
            ctx,
            &self.conv1d,
            &[
                (0, input.shared_buffer()),
                (1, weight.shared_buffer()),
                (2, out.shared_buffer()),
                (3, bias.shared_buffer()),
                (4, &params),
            ],
            groups64(dim * seq_len),
        );
    }

    /// Gradients of [`conv1d`](Self::conv1d) with respect to `input`,
    /// `weight` and `bias`, given the output gradient `grad_out`. Each
    /// gradient has the shape of the tensor it belongs to. Writes only the
    /// first `dim * seq_len` elements of `grad_input`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_backward(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        grad_out: &UnifiedTensor,
        grad_input: &UnifiedTensor,
        grad_weight: &UnifiedTensor,
        grad_bias: &UnifiedTensor,
    ) {
        let dim = weight.shape[0];
        let seq_len = input.shape[input.shape.len() - 1];
        let params = ctx.uniform_buffer(bytemuck::cast_slice(&[dim as u32, seq_len as u32]));
        self.run(
            ctx,
            &self.conv1d_bw,
            &[
                (0, input.shared_buffer()),
                (1, weight.shared_buffer()),
                (2, grad_out.shared_buffer()),
                (3, grad_input.shared_buffer()),
                (4, grad_weight.shared_buffer()),
                (5, grad_bias.shared_buffer()),
                (6, &params),
            ],
            groups64(dim),
        );
    }
}
