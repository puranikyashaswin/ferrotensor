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
            (groups64(words), 1, 1),
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
        let workgroups = groups64(out.shape.iter().product());
        // Metal limit is 65535 per dimension, use 2D dispatch for large tensors
        let max_wg = 65535u32;
        let groups = if workgroups <= max_wg {
            (workgroups, 1, 1)
        } else {
            (max_wg, workgroups.div_ceil(max_wg).min(max_wg), 1)
        };
        self.run(
            ctx,
            &self.affine_dequant,
            &[(0, weight), (1, scales), (2, biases), (3, out.buffer())],
            groups,
        );
    }
}
