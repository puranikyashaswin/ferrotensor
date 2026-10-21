//! Rotary embeddings and fused (flash) attention.

use super::{dummy_storage, groups64, uniform, ComputeEngine};
use crate::{UnifiedTensor, WgpuContext};
use wgpu::Buffer;

impl ComputeEngine {
    pub fn rope(&self, ctx: &WgpuContext, tensor: &UnifiedTensor, head_dim: usize, pos: usize) {
        // [batch, seq, ...] tensors: every head vector of token s gets
        // position pos + s. 1-D/2-D tensors are a single token.
        let seq_len = if tensor.shape.len() >= 3 {
            tensor.shape[1]
        } else {
            1
        };
        let total = tensor.shape.iter().product::<usize>();
        let tokens = if tensor.shape.len() >= 3 {
            tensor.shape[0] * tensor.shape[1]
        } else {
            1
        };
        let heads = total / (tokens * head_dim).max(1);
        let params = uniform(
            ctx,
            bytemuck::cast_slice(&[head_dim as u32, seq_len as u32, pos as u32, heads as u32]),
        );
        let pairs = tensor.shape.iter().product::<usize>() / 2;
        self.run(
            ctx,
            &self.rope,
            &[(0, tensor.buffer()), (1, &params)],
            groups64(pairs),
        );
    }

    /// Single-head attention: `q`, `k`, `v`, `out` are `[seq_len, head_dim]`.
    ///
    /// Runs the v2 kernel on `[1, seq_len, 1, head_dim]` views of the inputs.
    pub fn flash_attention(
        &self,
        ctx: &WgpuContext,
        q: &UnifiedTensor,
        k: &UnifiedTensor,
        v: &UnifiedTensor,
        out: &UnifiedTensor,
        causal: bool,
    ) {
        let [q4, k4, v4, o4] = [q, k, v, out].map(single_head_view);
        self.flash_attention_v2(ctx, &q4, &k4, &v4, &o4, causal, 0.0, 0, None);
    }

    /// Fused attention forward pass with grouped-query attention, causal
    /// masking, an optional additive mask and attention dropout.
    ///
    /// * `q`, `out` - `[batch, seq_q, num_heads, head_dim]`
    /// * `k`, `v` - `[batch, seq_k, num_kv_heads, head_dim]`; `num_heads`
    ///   must be a multiple of `num_kv_heads`
    /// * `causal` - key `j` is visible to query `i` only when `j <= i`
    /// * `dropout_prob`, `seed` - attention dropout; pass `0.0` to disable
    /// * `mask` - additive mask `[batch, heads, seq_q, seq_k]`, where size-1
    ///   dimensions broadcast
    ///
    /// `head_dim` must be at most 128. See
    /// [`flash_attention_v2_with`](Self::flash_attention_v2_with) for
    /// KV-cache offsets and sliding windows.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_v2(
        &self,
        ctx: &WgpuContext,
        q: &UnifiedTensor,
        k: &UnifiedTensor,
        v: &UnifiedTensor,
        out: &UnifiedTensor,
        causal: bool,
        dropout_prob: f32,
        seed: u64,
        mask: Option<&UnifiedTensor>,
    ) {
        let opts = AttentionOptions {
            causal,
            dropout_prob,
            seed,
            ..AttentionOptions::default()
        };
        self.flash_attention_v2_with(ctx, q, k, v, out, &opts, mask);
    }

    /// [`flash_attention_v2`](Self::flash_attention_v2) with every option.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_v2_with(
        &self,
        ctx: &WgpuContext,
        q: &UnifiedTensor,
        k: &UnifiedTensor,
        v: &UnifiedTensor,
        out: &UnifiedTensor,
        opts: &AttentionOptions,
        mask: Option<&UnifiedTensor>,
    ) {
        let dims = AttentionDims::new(q, k);
        let params = dims.params(ctx, opts, mask);
        let dummy;
        let mask_buf = match mask {
            Some(m) => m.buffer(),
            None => {
                dummy = dummy_storage(ctx);
                &dummy
            }
        };
        self.run(
            ctx,
            &self.flash_attn_v2,
            &[
                (0, q.buffer()),
                (1, k.buffer()),
                (2, v.buffer()),
                (3, out.buffer()),
                (4, &params),
                (5, mask_buf),
            ],
            (1, dims.seq_q.div_ceil(64), dims.batch * dims.heads),
        );
    }

    /// Single-head attention backward pass for `[seq_len, head_dim]` tensors.
    /// Writes (does not accumulate into) `grad_q`, `grad_k` and `grad_v`.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_backward(
        &self,
        ctx: &WgpuContext,
        q: &UnifiedTensor,
        k: &UnifiedTensor,
        v: &UnifiedTensor,
        out: &UnifiedTensor,
        grad_out: &UnifiedTensor,
        grad_q: &UnifiedTensor,
        grad_k: &UnifiedTensor,
        grad_v: &UnifiedTensor,
        causal: bool,
    ) {
        let views = [q, k, v, out, grad_out, grad_q, grad_k, grad_v].map(single_head_view);
        let [q4, k4, v4, o4, go4, gq4, gk4, gv4] = &views;
        self.flash_attention_v2_backward(ctx, q4, k4, v4, o4, go4, gq4, gk4, gv4, causal);
    }

    /// Backward pass of [`flash_attention_v2`](Self::flash_attention_v2)
    /// without mask or dropout. See
    /// [`flash_attention_v2_backward_with`](Self::flash_attention_v2_backward_with).
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_v2_backward(
        &self,
        ctx: &WgpuContext,
        q: &UnifiedTensor,
        k: &UnifiedTensor,
        v: &UnifiedTensor,
        out: &UnifiedTensor,
        grad_out: &UnifiedTensor,
        grad_q: &UnifiedTensor,
        grad_k: &UnifiedTensor,
        grad_v: &UnifiedTensor,
        causal: bool,
    ) {
        let opts = AttentionOptions {
            causal,
            ..AttentionOptions::default()
        };
        self.flash_attention_v2_backward_with(
            ctx, q, k, v, out, grad_out, grad_q, grad_k, grad_v, &opts, None,
        );
    }

    /// Backward pass of [`flash_attention_v2`](Self::flash_attention_v2).
    ///
    /// `out` is the forward output for the same inputs; `opts` and `mask`
    /// must match the forward call (dropout decisions are regenerated from
    /// `opts.seed`). Writes `grad_q`, `grad_k` and
    /// `grad_v` (same shapes as `q`, `k`, `v`); no gradient is produced for
    /// the mask.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_v2_backward_with(
        &self,
        ctx: &WgpuContext,
        q: &UnifiedTensor,
        k: &UnifiedTensor,
        v: &UnifiedTensor,
        out: &UnifiedTensor,
        grad_out: &UnifiedTensor,
        grad_q: &UnifiedTensor,
        grad_k: &UnifiedTensor,
        grad_v: &UnifiedTensor,
        opts: &AttentionOptions,
        mask: Option<&UnifiedTensor>,
    ) {
        let dims = AttentionDims::new(q, k);
        let params = dims.params_bwd(ctx, opts, mask);
        let dummy;
        let mask_buf = match mask {
            Some(m) => m.buffer(),
            None => {
                dummy = dummy_storage(ctx);
                &dummy
            }
        };
        // Per (batch, head, query) row: [Delta, LSE].
        let stats = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Flash Bwd Stats"),
            size: (dims.batch * dims.heads * dims.seq_q).max(1) as u64 * 8,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let (b, h, hkv) = (dims.batch, dims.heads, dims.kv_heads);
        self.run(
            ctx,
            &self.flash_attn_v2_bw_preprocess,
            &[
                (0, q.buffer()),
                (1, k.buffer()),
                (3, out.buffer()),
                (4, grad_out.buffer()),
                (8, &params),
                (9, mask_buf),
                (10, &stats),
            ],
            (dims.seq_q.div_ceil(64), h, b),
        );
        self.run(
            ctx,
            &self.flash_attn_v2_bw_dq,
            &[
                (0, q.buffer()),
                (1, k.buffer()),
                (2, v.buffer()),
                (4, grad_out.buffer()),
                (5, grad_q.buffer()),
                (8, &params),
                (9, mask_buf),
                (10, &stats),
            ],
            (dims.seq_q.div_ceil(64), h, b),
        );
        self.run(
            ctx,
            &self.flash_attn_v2_bw_dkdv,
            &[
                (0, q.buffer()),
                (1, k.buffer()),
                (2, v.buffer()),
                (4, grad_out.buffer()),
                (6, grad_k.buffer()),
                (7, grad_v.buffer()),
                (8, &params),
                (9, mask_buf),
                (10, &stats),
            ],
            (dims.seq_k.div_ceil(64), hkv, b),
        );
    }
}

/// `[seq, dim]` tensor viewed as `[1, seq, 1, dim]`.
fn single_head_view(t: &UnifiedTensor) -> UnifiedTensor {
    assert_eq!(t.shape.len(), 2, "expected a [seq_len, head_dim] tensor");
    t.reshaped(&[1, t.shape[0], 1, t.shape[1]])
}

/// Shapes of a `[batch, seq, heads, head_dim]` attention call.
struct AttentionDims {
    batch: u32,
    seq_q: u32,
    seq_k: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
}

impl AttentionDims {
    fn new(q: &UnifiedTensor, k: &UnifiedTensor) -> Self {
        assert_eq!(q.shape.len(), 4, "q must be [batch, seq, heads, head_dim]");
        assert_eq!(
            k.shape.len(),
            4,
            "k must be [batch, seq, kv_heads, head_dim]"
        );
        let dims = Self {
            batch: q.shape[0] as u32,
            seq_q: q.shape[1] as u32,
            heads: q.shape[2] as u32,
            head_dim: q.shape[3] as u32,
            seq_k: k.shape[1] as u32,
            kv_heads: k.shape[2] as u32,
        };
        assert!(
            dims.head_dim <= 128,
            "flash attention supports head_dim <= 128, got {}",
            dims.head_dim
        );
        assert!(
            dims.kv_heads > 0 && dims.heads % dims.kv_heads == 0,
            "num_heads ({}) must be a multiple of num_kv_heads ({})",
            dims.heads,
            dims.kv_heads
        );
        dims
    }

    fn scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Uniform for the forward kernel (`FlashV2Params`).
    fn params(
        &self,
        ctx: &WgpuContext,
        opts: &AttentionOptions,
        mask: Option<&UnifiedTensor>,
    ) -> Buffer {
        let (has_mask, s) = mask_strides(mask);
        let params: [u32; 20] = [
            self.batch,
            self.heads,
            self.seq_q,
            self.seq_k,
            self.head_dim,
            opts.causal as u32,
            self.scale().to_bits(),
            self.kv_heads,
            opts.dropout_prob.to_bits(),
            opts.seed as u32,
            has_mask,
            opts.causal_offset,
            s[0],
            s[1],
            s[2],
            s[3],
            opts.window,
            0,
            0,
            0,
        ];
        uniform(ctx, bytemuck::cast_slice(&params))
    }

    /// Uniform for the backward kernels (`FlashBwdParams`).
    fn params_bwd(
        &self,
        ctx: &WgpuContext,
        opts: &AttentionOptions,
        mask: Option<&UnifiedTensor>,
    ) -> Buffer {
        let (has_mask, s) = mask_strides(mask);
        let params: [u32; 20] = [
            self.batch,
            self.heads,
            self.seq_q,
            self.seq_k,
            self.head_dim,
            self.kv_heads,
            self.scale().to_bits(),
            opts.dropout_prob.to_bits(),
            opts.seed as u32,
            opts.causal as u32,
            has_mask,
            opts.causal_offset,
            s[0],
            s[1],
            s[2],
            s[3],
            opts.window,
            0,
            0,
            0,
        ];
        uniform(ctx, bytemuck::cast_slice(&params))
    }
}

/// Options for [`ComputeEngine::flash_attention_v2_with`] and
/// [`ComputeEngine::flash_attention_v2_backward_with`].
///
/// Query `i` sits at absolute position `i + causal_offset`; key `j` is
/// visible when `j <= i + causal_offset` (if `causal`) and
/// `|i + causal_offset - j| < window` (if `window > 0`).
#[derive(Debug, Clone, Copy, Default)]
pub struct AttentionOptions {
    /// Hide keys after each query's absolute position.
    pub causal: bool,
    /// Absolute position of the first query, e.g. the number of tokens
    /// already in a KV cache when decoding.
    pub causal_offset: u32,
    /// Sliding-window size in tokens; 0 means unlimited.
    pub window: u32,
    /// Attention dropout probability; 0.0 disables dropout.
    pub dropout_prob: f32,
    /// Seed for the dropout pattern (only the low 32 bits are used).
    pub seed: u64,
}

/// Broadcast strides `[b, h, q, k]` for an additive attention mask of shape
/// `[batch, heads, seq_q, seq_k]`; size-1 dimensions get stride 0.
fn mask_strides(mask: Option<&UnifiedTensor>) -> (u32, [u32; 4]) {
    let Some(m) = mask else {
        return (0, [0; 4]);
    };
    let shape = &m.shape;
    if shape.len() != 4 {
        return (1, [0; 4]);
    }
    let dim_b = shape[0] as u32;
    let dim_h = shape[1] as u32;
    let dim_q = shape[2] as u32;
    let dim_k = shape[3] as u32;
    let s = |dim: u32, stride: u32| if dim > 1 { stride } else { 0 };
    (
        1,
        [
            s(dim_b, dim_k * dim_q * dim_h),
            s(dim_h, dim_k * dim_q),
            s(dim_q, dim_k),
            s(dim_k, 1),
        ],
    )
}
