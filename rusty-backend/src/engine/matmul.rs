//! Matrix multiplication kernels.

use super::{uniform, ComputeEngine, Kernel};
use crate::{UnifiedTensor, WgpuContext};

impl ComputeEngine {
    pub fn matmul(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let dims = (a.shape[0] as u32, a.shape[1] as u32, b.shape[1] as u32);
        self.matmul_dispatch(ctx, &self.matmul, dims, a, b, out);
    }

    pub fn matmul_bt(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let dims = (a.shape[0] as u32, a.shape[1] as u32, b.shape[0] as u32);
        self.matmul_dispatch(ctx, &self.matmul_bt, dims, a, b, out);
    }

    pub fn matmul_at(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let dims = (a.shape[1] as u32, a.shape[0] as u32, b.shape[1] as u32);
        self.matmul_dispatch(ctx, &self.matmul_at, dims, a, b, out);
    }

    fn matmul_dispatch(
        &self,
        ctx: &WgpuContext,
        kernel: &Kernel,
        (m, k, n): (u32, u32, u32),
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let params = uniform(ctx, bytemuck::cast_slice(&[m, k, n, 0u32]));
        self.run(
            ctx,
            kernel,
            &[
                (0, a.buffer()),
                (1, b.buffer()),
                (2, out.buffer()),
                (3, &params),
            ],
            (n.div_ceil(16), m.div_ceil(16), 1),
        );
    }
}
