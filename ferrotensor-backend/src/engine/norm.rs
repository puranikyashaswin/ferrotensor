//! Normalisation and softmax kernels.

use super::{grid, groups64, uniform, ComputeEngine};
use crate::{UnifiedTensor, WgpuContext};

/// A LayerNorm weight or bias whose length is not the input's last dimension,
/// or a weight and a bias of different lengths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerNormError {
    /// The last dimension of the input: the length both parameters need.
    pub size: usize,
    /// Elements in the weight.
    pub weight_len: usize,
    /// Elements in the bias.
    pub bias_len: usize,
}

impl std::fmt::Display for LayerNormError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "layer_norm needs a weight and a bias of {} elements, the input's last dimension; \
             the weight has {} and the bias has {}",
            self.size, self.weight_len, self.bias_len
        )
    }
}

impl std::error::Error for LayerNormError {}

impl ComputeEngine {
    /// RMSNorm over the last dimension with eps = 1e-5:
    /// `out = x / sqrt(mean(x^2) + eps) * weight`.
    pub fn rms_norm(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.rms_norm_eps(ctx, input, weight, out, 1e-5);
    }

    /// RMSNorm over the last dimension with a caller-chosen `eps`.
    pub fn rms_norm_eps(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        out: &UnifiedTensor,
        eps: f32,
    ) {
        let size = *input.shape.last().unwrap_or(&1);
        let rows = input.shape.iter().product::<usize>() / size;
        let params = uniform(
            ctx,
            bytemuck::cast_slice(&[size as u32, rows as u32, eps.to_bits(), 0]),
        );
        self.run(
            ctx,
            &self.rms_norm,
            &[
                (0, input.buffer()),
                (1, weight.buffer()),
                (2, out.buffer()),
                (3, &params),
            ],
            grid(rows as u32),
        );
    }

    /// Gradients of [`rms_norm_eps`](Self::rms_norm_eps) with respect to
    /// `input` and `weight`, given the output gradient `grad_out`.
    ///
    /// `grad_input` has `input`'s shape; `grad_weight` has `weight`'s and is
    /// summed over all rows, and `None` skips it (a frozen weight).
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        grad_out: &UnifiedTensor,
        grad_input: &UnifiedTensor,
        grad_weight: Option<&UnifiedTensor>,
        eps: f32,
    ) {
        self.rms_norm_backward_beta(
            ctx,
            input,
            weight,
            grad_out,
            (grad_input, 0.0),
            grad_weight,
            eps,
        );
    }

    /// [`rms_norm_backward`](Self::rms_norm_backward) that adds into
    /// `grad_input`, given as `(tensor, beta)`: the tensor ends as
    /// `norm_gradient + beta * tensor`.
    ///
    /// With a residual stream the tensor holds the gradient the skip
    /// connection already delivered, so `beta = 1` replaces the separate
    /// `add` dispatch that sums the two. With `beta = 0` the kernel never
    /// reads the tensor, which may then be uninitialised. `grad_weight` is
    /// always overwritten, never accumulated.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward_beta(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        grad_out: &UnifiedTensor,
        grad_input: (&UnifiedTensor, f32),
        grad_weight: Option<&UnifiedTensor>,
        eps: f32,
    ) {
        let (grad_input, beta) = grad_input;
        let size = *input.shape.last().unwrap_or(&1);
        let rows = input.shape.iter().product::<usize>() / size.max(1);
        if rows == 0 || size == 0 {
            return;
        }
        let inv_rms = UnifiedTensor::empty(ctx, &[rows]);
        let params = uniform(
            ctx,
            bytemuck::cast_slice(&[size as u32, rows as u32, eps.to_bits(), beta.to_bits()]),
        );
        self.run(
            ctx,
            &self.rms_norm_bw,
            &[
                (0, input.buffer()),
                (1, grad_out.buffer()),
                (2, weight.buffer()),
                (3, grad_input.buffer()),
                (4, inv_rms.buffer()),
                (5, &params),
            ],
            grid(rows as u32),
        );
        let Some(grad_weight) = grad_weight else {
            return;
        };
        let params = uniform(ctx, bytemuck::cast_slice(&[size as u32, rows as u32, 0, 0]));
        self.run(
            ctx,
            &self.rms_norm_bw_weight,
            &[
                (0, input.buffer()),
                (1, grad_out.buffer()),
                (2, inv_rms.buffer()),
                (3, grad_weight.buffer()),
                (4, &params),
            ],
            groups64(size),
        );
    }

    /// LayerNorm over the last dimension:
    /// `out = (x - mean) / sqrt(var + eps) * weight + bias`, with the
    /// variance taken about the row's mean in a second pass.
    ///
    /// `stats` is `[rows, 2]`: the kernel writes each row's mean and
    /// reciprocal standard deviation there for
    /// [`layer_norm_backward`](Self::layer_norm_backward).
    ///
    /// # Errors
    ///
    /// Returns a [`LayerNormError`] when `weight` or `bias` does not have as
    /// many elements as the last dimension of `input`. Nothing runs then.
    #[expect(
        clippy::too_many_arguments,
        reason = "the input, both parameters, the output and the saved statistics are separate buffers"
    )]
    pub fn layer_norm(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        bias: &UnifiedTensor,
        out: &UnifiedTensor,
        stats: &UnifiedTensor,
        eps: f32,
    ) -> Result<(), LayerNormError> {
        let size = *input.shape.last().unwrap_or(&1);
        let weight_len = weight.shape.iter().product::<usize>();
        let bias_len = bias.shape.iter().product::<usize>();
        if weight_len != size || bias_len != size {
            return Err(LayerNormError {
                size,
                weight_len,
                bias_len,
            });
        }
        let rows = input.shape.iter().product::<usize>() / size.max(1);
        if rows == 0 || size == 0 {
            return Ok(());
        }
        let params = uniform(
            ctx,
            bytemuck::cast_slice(&[size as u32, rows as u32, eps.to_bits(), 0]),
        );
        self.run(
            ctx,
            &self.layer_norm,
            &[
                (0, input.buffer()),
                (1, weight.buffer()),
                (2, bias.buffer()),
                (3, out.buffer()),
                (4, stats.buffer()),
                (5, &params),
            ],
            grid(rows as u32),
        );
        Ok(())
    }

    /// Gradients of [`layer_norm`](Self::layer_norm) with respect to `input`,
    /// `weight` and `bias`, given the output gradient `grad_out` and the
    /// `stats` the forward pass wrote.
    ///
    /// `grad_input` has `input`'s shape; `grad_weight` and `grad_bias` have
    /// the shape of the parameters and are summed over all rows. `None`
    /// skips that gradient (a frozen parameter); both `None` skips the
    /// kernel that sums over rows.
    #[expect(
        clippy::too_many_arguments,
        reason = "the saved tensors and the three gradients are separate buffers, as in rms_norm_backward"
    )]
    pub fn layer_norm_backward(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        grad_out: &UnifiedTensor,
        stats: &UnifiedTensor,
        grad_input: &UnifiedTensor,
        grad_weight: Option<&UnifiedTensor>,
        grad_bias: Option<&UnifiedTensor>,
    ) {
        let size = *input.shape.last().unwrap_or(&1);
        let rows = input.shape.iter().product::<usize>() / size.max(1);
        if rows == 0 || size == 0 {
            return;
        }
        let params = uniform(ctx, bytemuck::cast_slice(&[size as u32, rows as u32, 0, 0]));
        self.run(
            ctx,
            &self.layer_norm_bw,
            &[
                (0, input.buffer()),
                (1, grad_out.buffer()),
                (2, weight.buffer()),
                (3, stats.buffer()),
                (4, grad_input.buffer()),
                (5, &params),
            ],
            grid(rows as u32),
        );
        if grad_weight.is_none() && grad_bias.is_none() {
            return;
        }
        // The kernel binds both outputs; one that is not wanted gets a
        // one-element buffer it never writes.
        let unused = UnifiedTensor::empty(ctx, &[1]);
        let flags = u32::from(grad_weight.is_some()) | (u32::from(grad_bias.is_some()) << 1);
        let params = uniform(
            ctx,
            bytemuck::cast_slice(&[size as u32, rows as u32, flags, 0]),
        );
        self.run(
            ctx,
            &self.layer_norm_bw_affine,
            &[
                (0, input.buffer()),
                (1, grad_out.buffer()),
                (2, stats.buffer()),
                (3, grad_weight.unwrap_or(&unused).buffer()),
                (4, grad_bias.unwrap_or(&unused).buffer()),
                (5, &params),
            ],
            groups64(size),
        );
    }

    /// Fused RMSNorm + Residual: out = RMSNorm(x + residual) * weight
    pub fn rms_norm_residual(
        &self,
        ctx: &WgpuContext,
        x: &UnifiedTensor,
        residual: &UnifiedTensor,
        weight: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let size = x.shape.iter().product::<usize>() as u32;
        let params = uniform(ctx, bytemuck::cast_slice(&[size]));
        self.run(
            ctx,
            &self.rms_residual,
            &[
                (0, x.buffer()),
                (1, residual.buffer()),
                (2, weight.buffer()),
                (3, out.buffer()),
                (4, &params),
            ],
            (1, 1, 1),
        );
    }

    /// Softmax over the last dimension of `input`.
    pub fn softmax(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        let size = *input.shape.last().unwrap_or(&1);
        let rows = input.shape.iter().product::<usize>() / size;
        let params = uniform(ctx, bytemuck::cast_slice(&[size as u32, rows as u32]));
        self.run(
            ctx,
            &self.softmax,
            &[(0, input.buffer()), (1, out.buffer()), (2, &params)],
            grid(rows as u32),
        );
    }

    /// Gradient of softmax over the last dimension, given the forward
    /// output `final_out`: `grad_in = s * (grad_out - sum(grad_out * s))`.
    pub fn softmax_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        final_out: &UnifiedTensor,
        grad_in: &UnifiedTensor,
    ) {
        let size = *grad_out.shape.last().unwrap_or(&1);
        let rows = grad_out.shape.iter().product::<usize>() / size;
        let params = uniform(ctx, bytemuck::cast_slice(&[size as u32, rows as u32]));
        self.run(
            ctx,
            &self.softmax_bw,
            &[
                (2, &params),
                (3, grad_out.buffer()),
                (4, final_out.buffer()),
                (5, grad_in.buffer()),
            ],
            grid(rows as u32),
        );
    }
}
