//! Dequantisation kernels.

use super::{groups64, record_dispatch, ComputeEngine};
use crate::kernel_timing::KernelLabel;
use crate::{UnifiedTensor, WgpuContext};
use std::sync::Arc;

impl ComputeEngine {
    /// Unpacks bytes to floats: element `i` of `out` is byte `i` of `input`
    /// (read as packed `u32`s) converted to `f32`, without scaling.
    pub fn dequantize(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        let words = input.shape.iter().product::<usize>().div_ceil(4);
        self.run(
            ctx,
            &self.dequant,
            &[(0, input.shared_buffer()), (1, out.shared_buffer())],
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
        // Raw buffers have no pool identity to cache a bind group by; this
        // runs once per weight at load time.
        let bind_group = ctx.create_bind_group(
            &self.affine_dequant.layout,
            &[(0, weight), (1, scales), (2, biases), (3, out.buffer())],
        );
        let label = KernelLabel {
            entry: self.affine_dequant.entry_point,
            layout: "",
            size_class: "",
        };
        record_dispatch(
            ctx,
            &self.affine_dequant,
            Arc::new(bind_group),
            label,
            groups64(out.shape.iter().product()),
        );
    }
}
