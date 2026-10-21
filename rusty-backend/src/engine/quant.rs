//! Dequantisation kernels.

use super::{groups64, ComputeEngine};
use crate::{UnifiedTensor, WgpuContext};

impl ComputeEngine {
    /// Unpacks bytes to floats: element `i` of `out` is byte `i` of `input`
    /// (read as packed `u32`s) converted to `f32`, without scaling.
    pub fn dequantize(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        let words = input.shape.iter().product::<usize>().div_ceil(4);
        self.run(
            ctx,
            &self.dequant,
            &[(0, input.buffer()), (1, out.buffer())],
            groups64(words),
        );
    }

    /// 8-bit affine dequantisation with groups of 64 (MLX layout):
    /// `out[i] = byte(weight, i) * scales[i / 64] + biases[i / 64]`.
    pub fn affine_dequantize(
        &self,
        ctx: &WgpuContext,
        weight: &wgpu::Buffer,
        scales: &wgpu::Buffer,
        biases: &wgpu::Buffer,
        out: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.affine_dequant,
            &[(0, weight), (1, scales), (2, biases), (3, out.buffer())],
            groups64(out.shape.iter().product()),
        );
    }
}
