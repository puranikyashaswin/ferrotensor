//! Element-wise kernels: arithmetic, activations and their backward passes.

use super::{groups64, ComputeEngine, Kernel};
use crate::{UnifiedTensor, WgpuContext};

impl ComputeEngine {
    /// `out = a + b`, element-wise; `a`, `b` and `out` have the same length.
    pub fn add(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.binary(ctx, &self.add, a, b, out);
    }

    /// `out = a * b`, element-wise.
    pub fn mul(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.binary(ctx, &self.mul, a, b, out);
    }

    /// `out = a - b`, element-wise.
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
            groups64(a.shape.iter().product()),
        );
    }

    /// Gradients of `a + b`: writes `grad_out` to both `grad_a` and `grad_b`.
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
            groups64(grad_out.shape.iter().product()),
        );
    }

    /// Gradients of `a - b`: `grad_a = grad_out`, `grad_b = -grad_out`.
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
            groups64(grad_out.shape.iter().product()),
        );
    }

    /// Gradients of `a * b`: `grad_a = grad_out * b`, `grad_b = grad_out * a`.
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
            groups64(grad_out.shape.iter().product()),
        );
    }

    /// `out = x * sigmoid(x)`, element-wise.
    pub fn silu(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        self.run(
            ctx,
            &self.silu,
            &[(0, input.buffer()), (1, out.buffer())],
            groups64(input.shape.iter().product()),
        );
    }

    /// Gradient of SiLU: `grad_in = grad_out * s * (1 + x * (1 - s))` with
    /// `s = sigmoid(input)`.
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
            groups64(grad_out.shape.iter().product()),
        );
    }

    /// `out = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`, element-wise:
    /// GELU in the tanh form of GPT-2 (`gelu_new`), BLOOM and Gemma.
    pub fn gelu_tanh(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        self.run(
            ctx,
            &self.gelu_tanh,
            &[(0, input.buffer()), (1, out.buffer())],
            groups64(input.shape.iter().product()),
        );
    }

    /// Gradient of [`gelu_tanh`](Self::gelu_tanh): `grad_out` times the
    /// derivative of the activation at `input`.
    pub fn gelu_tanh_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        input: &UnifiedTensor,
        grad_in: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.gelu_tanh_bw,
            &[
                (3, grad_out.buffer()),
                (4, input.buffer()),
                (5, grad_in.buffer()),
            ],
            groups64(grad_out.shape.iter().product()),
        );
    }

    /// `out = x * Phi(x)` with `Phi` the standard normal CDF, element-wise:
    /// GELU in the exact (erf) form of GPT-NeoX.
    ///
    /// The kernel has no `erf` to call; it uses the approximation of
    /// Abramowitz and Stegun 7.1.26, with an absolute error of at most 1.5e-7.
    pub fn gelu_erf(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        self.run(
            ctx,
            &self.gelu_erf,
            &[(0, input.buffer()), (1, out.buffer())],
            groups64(input.shape.iter().product()),
        );
    }

    /// Gradient of [`gelu_erf`](Self::gelu_erf): `grad_out * (Phi(x) + x * phi(x))`
    /// with `phi` the standard normal density.
    pub fn gelu_erf_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        input: &UnifiedTensor,
        grad_in: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.gelu_erf_bw,
            &[
                (3, grad_out.buffer()),
                (4, input.buffer()),
                (5, grad_in.buffer()),
            ],
            groups64(grad_out.shape.iter().product()),
        );
    }

    /// `out = max(x, 0)`, element-wise.
    pub fn relu(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        self.run(
            ctx,
            &self.relu,
            &[(0, input.buffer()), (1, out.buffer())],
            groups64(input.shape.iter().product()),
        );
    }

    /// Gradient of ReLU: `grad_out` where `input > 0`, else 0.
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
            groups64(grad_out.shape.iter().product()),
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
            groups64(up.shape.iter().product()),
        );
    }
}
