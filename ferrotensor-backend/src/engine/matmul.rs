//! Matrix multiplication kernels.

use super::{uniform, ComputeEngine};
use crate::{UnifiedTensor, WgpuContext};

/// Operand layout understood by the `matmul_blocked` kernel.
#[derive(Clone, Copy)]
#[repr(u32)]
enum Layout {
    /// `a @ b` with `a: [M, K]`, `b: [K, N]`.
    Plain = 0,
    /// `a @ b^T` with `a: [M, K]`, `b: [N, K]`.
    TransposedB = 1,
    /// `a^T @ b` with `a: [K, M]`, `b: [K, N]`.
    TransposedA = 2,
}

impl ComputeEngine {
    /// `out[M, N] = a[M, K] @ b[K, N]`.
    pub fn matmul(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let dims = (a.shape[0] as u32, a.shape[1] as u32, b.shape[1] as u32);
        self.matmul_dispatch(ctx, Layout::Plain, dims, a, b, out);
    }

    /// `out[M, N] = a[M, K] @ b[N, K]^T` (e.g. the input gradient of a
    /// linear layer).
    pub fn matmul_bt(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let dims = (a.shape[0] as u32, a.shape[1] as u32, b.shape[0] as u32);
        self.matmul_dispatch(ctx, Layout::TransposedB, dims, a, b, out);
    }

    /// `out[M, N] = a[K, M]^T @ b[K, N]` (e.g. the weight gradient of a
    /// linear layer).
    pub fn matmul_at(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let dims = (a.shape[1] as u32, a.shape[0] as u32, b.shape[1] as u32);
        self.matmul_dispatch(ctx, Layout::TransposedA, dims, a, b, out);
    }

    fn matmul_dispatch(
        &self,
        ctx: &WgpuContext,
        layout: Layout,
        (m, k, n): (u32, u32, u32),
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let params = uniform(ctx, bytemuck::cast_slice(&[m, k, n, layout as u32]));
        // One 16x16 workgroup per 64x64 output tile.
        self.run(
            ctx,
            &self.matmul,
            &[
                (0, a.buffer()),
                (1, b.buffer()),
                (2, out.buffer()),
                (3, &params),
            ],
            (n.div_ceil(64), m.div_ceil(64), 1),
        );
    }
}
