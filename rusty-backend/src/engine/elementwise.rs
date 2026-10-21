//! Element-wise kernels: arithmetic, activations and their backward passes.

use super::{groups64, ComputeEngine, Kernel};
use crate::{UnifiedTensor, WgpuContext};

impl ComputeEngine {
    pub fn add(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.binary(ctx, &self.add, a, b, out);
    }

    pub fn mul(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.binary(ctx, &self.mul, a, b, out);
    }

    pub fn sub(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.binary(ctx, &self.sub, a, b, out);
    }

    pub(super) fn binary(
        &self,
        ctx: &WgpuContext,
        kernel: &Kernel,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            kernel,
            &[(0, a.buffer()), (1, b.buffer()), (2, out.buffer())],
            (groups64(a.shape.iter().product()), 1, 1),
        );
    }

    pub fn add_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        grad_a: &UnifiedTensor,
        grad_b: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.add_bw,
            &[
                (3, grad_out.buffer()),
                (4, grad_a.buffer()),
                (5, grad_b.buffer()),
            ],
            (groups64(grad_out.shape.iter().product()), 1, 1),
        );
    }

    pub fn sub_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        grad_a: &UnifiedTensor,
        grad_b: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.sub_bw,
            &[
                (3, grad_out.buffer()),
                (4, grad_a.buffer()),
                (5, grad_b.buffer()),
            ],
            (groups64(grad_out.shape.iter().product()), 1, 1),
        );
    }

    pub fn mul_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        grad_a: &UnifiedTensor,
        grad_b: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.mul_bw,
            &[
                (3, grad_out.buffer()),
                (4, a.buffer()),
                (5, b.buffer()),
                (6, grad_a.buffer()),
                (7, grad_b.buffer()),
            ],
            (groups64(grad_out.shape.iter().product()), 1, 1),
        );
    }

    pub fn silu(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        self.run(
            ctx,
            &self.silu,
            &[(0, input.buffer()), (1, out.buffer())],
            (groups64(input.shape.iter().product()), 1, 1),
        );
    }

    pub fn silu_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        input: &UnifiedTensor,
        grad_in: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.silu_bw,
            &[
                (3, grad_out.buffer()),
                (4, input.buffer()),
                (5, grad_in.buffer()),
            ],
            (groups64(grad_out.shape.iter().product()), 1, 1),
        );
    }

    pub fn relu(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        self.run(
            ctx,
            &self.relu,
            &[(0, input.buffer()), (1, out.buffer())],
            (groups64(input.shape.iter().product()), 1, 1),
        );
    }

    pub fn relu_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        input: &UnifiedTensor,
        grad_in: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.relu_bw,
            &[
                (3, grad_out.buffer()),
                (4, input.buffer()),
                (5, grad_in.buffer()),
            ],
            (groups64(grad_out.shape.iter().product()), 1, 1),
        );
    }

    /// Fused SiLU-Gate: out = SiLU(up) * gate
    /// Reduces memory bandwidth by 2x compared to separate ops
    pub fn silu_gate_fused(
        &self,
        ctx: &WgpuContext,
        up: &UnifiedTensor,
        gate: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.binary(ctx, &self.silu_gate, up, gate, out);
    }

    /// Fused SiLU-Gate backward: computes gradients for up and gate
    pub fn silu_gate_backward(
        &self,
        ctx: &WgpuContext,
        up: &UnifiedTensor,
        gate: &UnifiedTensor,
        grad_out: &UnifiedTensor,
        grad_up: &UnifiedTensor,
        grad_gate: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.silu_gate_bw,
            &[
                (0, up.buffer()),
                (1, gate.buffer()),
                (3, grad_out.buffer()),
                (4, grad_up.buffer()),
                (5, grad_gate.buffer()),
            ],
            (groups64(up.shape.iter().product()), 1, 1),
        );
    }
}
