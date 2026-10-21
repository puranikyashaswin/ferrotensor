//! Dequantisation kernels.

use super::{groups64, ComputeEngine};
use crate::{UnifiedTensor, WgpuContext};

impl ComputeEngine {
    pub fn dequantize(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        let words = input.shape.iter().product::<usize>().div_ceil(4);
        self.run(
            ctx,
            &self.dequant,
            &[(0, input.buffer()), (1, out.buffer())],
            groups64(words),
        );
    }

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
