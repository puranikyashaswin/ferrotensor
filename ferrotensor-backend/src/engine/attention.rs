//! Rotary embeddings and fused (flash) attention.

use super::flash_coop::Shape;
use super::{dummy_storage, grid, groups64, uniform, Arg, ComputeEngine, Params, Scratch};
use crate::{DType, UnifiedTensor, WgpuContext};

/// Query or key rows per workgroup of the attention kernels (`FA_TILE_Q` and `FAB_ROWS` in WGSL).
const ATTENTION_ROWS: u32 = 16;

impl ComputeEngine {
    /// Rotate-half RoPE (theta 10000) applied in place.
    ///
    /// `tensor` is `[batch, seq, ...]`, `[seq, hidden]` or a single token
    /// `[hidden]`; every head vector of token `s` gets position `pos + s`.
    pub fn rope(&self, ctx: &WgpuContext, tensor: &UnifiedTensor, head_dim: usize, pos: usize) {
        self.rope_theta(ctx, tensor, head_dim, pos, 10000.0);
    }

    /// [`rope`](Self::rope) with base frequency `theta` (the `rope_theta` of
    /// a Hugging Face config, e.g. 100000 for SmolLM2).
    pub fn rope_theta(
        &self,
        ctx: &WgpuContext,
        tensor: &UnifiedTensor,
        head_dim: usize,
        pos: usize,
        theta: f32,
    ) {
        self.rotate(ctx, (tensor, None), head_dim, pos, theta, false);
    }

    /// The inverse of [`rope_theta`](Self::rope_theta), in place: rotates by
    /// the negated angles. RoPE is a rotation, so this is also its backward
    /// pass (the gradient of a rotation is the transposed rotation).
    pub fn rope_theta_inverse(
        &self,
        ctx: &WgpuContext,
        tensor: &UnifiedTensor,
        head_dim: usize,
        pos: usize,
        theta: f32,
    ) {
        self.rotate(ctx, (tensor, None), head_dim, pos, theta, true);
    }

    /// [`rope_theta`](Self::rope_theta) of `input` written to `out` (same
    /// shape), leaving `input` as it is.
    pub fn rope_theta_to(
        &self,
        ctx: &WgpuContext,
        (input, out): (&UnifiedTensor, &UnifiedTensor),
        head_dim: usize,
        pos: usize,
        theta: f32,
    ) {
        self.rotate(ctx, (input, Some(out)), head_dim, pos, theta, false);
    }

    /// [`rope_theta_inverse`](Self::rope_theta_inverse) of `input` written
    /// to `out` (same shape), leaving `input` as it is.
    pub fn rope_theta_inverse_to(
        &self,
        ctx: &WgpuContext,
        (input, out): (&UnifiedTensor, &UnifiedTensor),
        head_dim: usize,
        pos: usize,
        theta: f32,
    ) {
        self.rotate(ctx, (input, Some(out)), head_dim, pos, theta, true);
    }

    /// Shared body of [`rope_theta`](Self::rope_theta) and its inverse: in
    /// place, or into `out` when it is given.
    fn rotate(
        &self,
        ctx: &WgpuContext,
        (tensor, out): (&UnifiedTensor, Option<&UnifiedTensor>),
        head_dim: usize,
        pos: usize,
        theta: f32,
        inverse: bool,
    ) {
        let (seq_len, tokens) = match tensor.shape.len() {
            0 | 1 => (1, 1),
            2 => (tensor.shape[0], tensor.shape[0]),
            _ => (tensor.shape[1], tensor.shape[0] * tensor.shape[1]),
        };
        let total = tensor.shape.iter().product::<usize>();
        let heads = total / (tokens * head_dim).max(1);
        let params = uniform(
            ctx,
            bytemuck::cast_slice(&[
                head_dim as u32,
                seq_len as u32,
                pos as u32,
                heads as u32,
                theta.to_bits(),
                inverse as u32,
                0,
                0,
            ]),
        );
        let pairs = tensor.shape.iter().product::<usize>() / 2;
        match out {
            None => self.run(
                ctx,
                &self.rope,
                &[(0, tensor.buffer()), (1, &params)],
                groups64(pairs),
            ),
            Some(out) => self.run(
                ctx,
                &self.rope_out,
                &[(0, tensor.buffer()), (1, out.buffer()), (2, &params)],
                groups64(pairs),
            ),
        }
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
        self.attention_forward(ctx, q, k, v, out, None, opts, mask);
    }

    /// [`flash_attention_v2_with`](Self::flash_attention_v2_with) that also
    /// saves each query row's log-sum-exp for
    /// [`flash_attention_v2_backward_with_stats`](Self::flash_attention_v2_backward_with_stats).
    ///
    /// `stats` is an `f32` tensor of `batch * num_heads * seq_q * 2`
    /// elements (e.g. shaped `[batch, num_heads, seq_q, 2]`). Element
    /// `2 * row + 1`, with `row = (b * num_heads + h) * seq_q + i`, receives
    /// the log-sum-exp of that row's scores; the backward pass uses element
    /// `2 * row` as scratch.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors flash_attention_v2_with plus the stats output"
    )]
    pub fn flash_attention_v2_with_stats(
        &self,
        ctx: &WgpuContext,
        q: &UnifiedTensor,
        k: &UnifiedTensor,
        v: &UnifiedTensor,
        out: &UnifiedTensor,
        stats: &UnifiedTensor,
        opts: &AttentionOptions,
        mask: Option<&UnifiedTensor>,
    ) {
        self.attention_forward(ctx, q, k, v, out, Some(stats), opts, mask);
    }

    /// Shared body of the forward entry points; `stats`, when given,
    /// receives the per-row log-sum-exp.
    #[expect(
        clippy::too_many_arguments,
        reason = "one argument per kernel binding and option"
    )]
    fn attention_forward(
        &self,
        ctx: &WgpuContext,
        q: &UnifiedTensor,
        k: &UnifiedTensor,
        v: &UnifiedTensor,
        out: &UnifiedTensor,
        stats: Option<&UnifiedTensor>,
        opts: &AttentionOptions,
        mask: Option<&UnifiedTensor>,
    ) {
        let dims = AttentionDims::new(q, k);
        if let Some(s) = stats {
            dims.check_stats(s);
        }
        let params = dims.params(ctx, opts, mask, stats.is_some());
        let dummy;
        let mask_buf: &dyn Arg = match mask {
            Some(m) => m.buffer(),
            None => {
                dummy = dummy_storage(ctx);
                &dummy
            }
        };
        // One query (decoding) gets a kernel whose threads split the keys;
        // the other gives each query 8 of its 128 threads and leaves 120 idle.
        // Only the many-query kernel writes stats, so a one-query training
        // call takes it and decoding is unchanged.
        if dims.seq_q == 1 && stats.is_none() {
            self.run(
                ctx,
                &self.flash_attn_decode,
                &[
                    (0, q.buffer()),
                    (1, k.buffer()),
                    (2, v.buffer()),
                    (3, out.buffer()),
                    (4, &params),
                    (5, mask_buf),
                ],
                (1, 1, dims.batch * dims.heads),
            );
            return;
        }
        let stats_dummy;
        let stats_buf: &dyn Arg = match stats {
            Some(s) => s.buffer(),
            None => {
                stats_dummy = dummy_storage(ctx);
                &stats_dummy
            }
        };
        let shape = dims.shape();
        if self.flash_coop_routes(shape, opts, mask.is_some()) {
            self.flash_coop_forward(ctx, shape, [q, k, v, out], stats_buf, &params);
            return;
        }
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
                (6, stats_buf),
            ],
            (
                1,
                dims.seq_q.div_ceil(ATTENTION_ROWS),
                dims.batch * dims.heads,
            ),
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
        let grads = [grad_q, grad_k, grad_v];
        self.attention_backward(ctx, q, k, v, out, grad_out, grads, None, opts, mask);
    }

    /// [`flash_attention_v2_backward_with`](Self::flash_attention_v2_backward_with)
    /// that reads the log-sum-exp saved by
    /// [`flash_attention_v2_with_stats`](Self::flash_attention_v2_with_stats)
    /// instead of recomputing it from the scores.
    ///
    /// `stats` is the tensor that forward call filled; its scratch half is
    /// overwritten.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors flash_attention_v2_backward_with plus the saved stats"
    )]
    pub fn flash_attention_v2_backward_with_stats(
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
        stats: &UnifiedTensor,
        opts: &AttentionOptions,
        mask: Option<&UnifiedTensor>,
    ) {
        let grads = [grad_q, grad_k, grad_v];
        self.attention_backward(ctx, q, k, v, out, grad_out, grads, Some(stats), opts, mask);
    }

    /// Shared body of the backward entry points; without `saved` stats the
    /// kernel recomputes each row's log-sum-exp.
    #[expect(
        clippy::too_many_arguments,
        reason = "one argument per kernel binding and option"
    )]
    fn attention_backward(
        &self,
        ctx: &WgpuContext,
        q: &UnifiedTensor,
        k: &UnifiedTensor,
        v: &UnifiedTensor,
        out: &UnifiedTensor,
        grad_out: &UnifiedTensor,
        [grad_q, grad_k, grad_v]: [&UnifiedTensor; 3],
        saved: Option<&UnifiedTensor>,
        opts: &AttentionOptions,
        mask: Option<&UnifiedTensor>,
    ) {
        let dims = AttentionDims::new(q, k);
        if let Some(s) = saved {
            dims.check_stats(s);
        }
        // With grouped-query attention each query head's dK and dV go to a
        // partial that flash_group_sum adds up; with one query head per KV
        // head they go straight to the gradients.
        let group = dims.heads / dims.kv_heads;
        let params = dims.params_bwd(ctx, opts, mask, (saved.is_some(), group == 1));
        let dummy;
        let mask_buf: &dyn Arg = match mask {
            Some(m) => m.buffer(),
            None => {
                dummy = dummy_storage(ctx);
                &dummy
            }
        };
        // Per (batch, head, query) row: [Delta, LSE].
        let scratch;
        let stats: &dyn Arg = match saved {
            Some(s) => s.buffer(),
            None => {
                let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Flash Bwd Stats"),
                    size: (dims.batch * dims.heads * dims.seq_q).max(1) as u64 * 8,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                // Dropped at the end of this function, before the dispatches run.
                ctx.memory().count_dispatch_buffer(buffer.size());
                scratch = Scratch(buffer);
                &scratch
            }
        };
        let (b, h) = (dims.batch, dims.heads);
        // The cooperative kernels need the saved log-sum-exp.
        let shape = dims.shape();
        let coop = saved.is_some() && self.flash_coop_routes(shape, opts, mask.is_some());
        if coop {
            self.flash_coop_dq(
                ctx,
                shape,
                [q, k, v, out, grad_out, grad_q],
                (stats, &params),
            );
        } else {
            self.run(
                ctx,
                &self.flash_attn_v2_bw_dq,
                &[
                    (0, q.buffer()),
                    (1, k.buffer()),
                    (2, v.buffer()),
                    (3, out.buffer()),
                    (4, grad_out.buffer()),
                    (5, grad_q.buffer()),
                    (8, &params),
                    (9, mask_buf),
                    (10, stats),
                ],
                (1, dims.seq_q.div_ceil(ATTENTION_ROWS), b * h),
            );
        }
        let dkdv_grid = (1, dims.seq_k.div_ceil(ATTENTION_ROWS), b * h);
        // dK and dV go to `grads`: the gradients themselves, or the
        // per-query-head partials.
        let dkdv = |grads: (&dyn Arg, &dyn Arg)| {
            if coop {
                self.flash_coop_dkdv(ctx, shape, [q, k, v, grad_out], grads, (stats, &params));
                return;
            }
            self.run(
                ctx,
                &self.flash_attn_v2_bw_dkdv,
                &[
                    (0, q.buffer()),
                    (1, k.buffer()),
                    (2, v.buffer()),
                    (4, grad_out.buffer()),
                    (6, grads.0),
                    (7, grads.1),
                    (8, &params),
                    (9, mask_buf),
                    (10, stats),
                ],
                dkdv_grid,
            );
        };
        if group == 1 {
            dkdv((grad_k.buffer(), grad_v.buffer()));
            return;
        }
        // Per-query-head partials, [batch, seq_k, num_heads, head_dim], made for
        // this call like the stats scratch above.
        let partial_size = (b * dims.seq_k * h * dims.head_dim).max(1) as u64 * 4;
        let partial = |label: &'static str| {
            let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: partial_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            ctx.memory().count_dispatch_buffer(buffer.size());
            Scratch(buffer)
        };
        let (dk_part, dv_part) = (
            partial("Flash Bwd dK Partial"),
            partial("Flash Bwd dV Partial"),
        );
        dkdv((&dk_part, &dv_part));
        let elements = b * dims.seq_k * dims.kv_heads * dims.head_dim;
        self.run(
            ctx,
            &self.flash_group_sum,
            &[
                (6, grad_k.buffer()),
                (7, grad_v.buffer()),
                (8, &params),
                (11, &dk_part),
                (12, &dv_part),
            ],
            grid(elements.div_ceil(256)),
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
            dims.kv_heads > 0 && dims.heads.is_multiple_of(dims.kv_heads),
            "num_heads ({}) must be a multiple of num_kv_heads ({})",
            dims.heads,
            dims.kv_heads
        );
        dims
    }

    /// Panics unless `stats` holds two `f32` values per (batch, head, query) row.
    fn check_stats(&self, stats: &UnifiedTensor) {
        let rows = (self.batch * self.heads * self.seq_q) as usize;
        assert!(
            stats.dtype == DType::FP32 && stats.shape.iter().product::<usize>() == rows * 2,
            "attention stats must be {} f32 values (2 per batch, head and query), got {:?} {:?}",
            rows * 2,
            stats.dtype,
            stats.shape
        );
    }

    fn shape(&self) -> Shape {
        Shape {
            batch: self.batch,
            heads: self.heads,
            seq_q: self.seq_q,
            seq_k: self.seq_k,
            head_dim: self.head_dim,
        }
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
        write_stats: bool,
    ) -> Params {
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
            (opts.seed >> 32) as u32,
            write_stats as u32,
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
        (lse_saved, dkv_direct): (bool, bool),
    ) -> Params {
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
            (opts.seed >> 32) as u32,
            lse_saved as u32,
            dkv_direct as u32,
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
    /// Seed for the dropout pattern (all 64 bits are used).
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
