//! Fused projections for training: the q, k and v projections of an attention
//! layer with RoPE and flash attention behind them, and the gate and up
//! projections of a SwiGLU feed-forward with the activation.
//!
//! Each is one autograd node with one output, so no op has to hand out
//! several tensors:
//!
//! - [`Tensor::fused_attention`]: `x -> softmax(rope(x Wq) rope(x Wk)^T) (x Wv)`,
//!   where the three base products and the three LoRA down products run as one
//!   matmul against `[Wq | Wk | Wv | Aq | Ak | Av]`, the LoRA up products run
//!   as one grouped dispatch, and RoPE is applied while flash attention loads
//!   its tiles. The backward pass mirrors it: one flash backward that gives
//!   the gradients of the un-rotated q, k and v, one grouped dispatch for the
//!   LoRA gradients and one matmul `[dQ | dK | dV | dU] @ W^T` for the gradient
//!   of `x`.
//! - [`Tensor::fused_gate_up_silu`]: `x -> silu(x Wg) * (x Wu)` the same way.
//!
//! The base weights are frozen f16 matrices; [`FusedProjections`] keeps them
//! concatenated, with room for the LoRA `A` matrices, whose f16 mirror
//! [`FusedProjections::refresh_mirror`] rewrites. Run it after every optimizer
//! step, adapter load and resume: a mirror that is not refreshed multiplies by
//! the `A` of an earlier step, and nothing warns about it. The concatenated
//! buffer is the only copy a caller needs to keep: it can drop the separate
//! weights and ask for one back with [`FusedProjections::projection`].

use super::{AutogradNode, Tensor, GRAD_ENABLED};
use ferrotensor_backend::{
    AttentionOptions, ComputeEngine, DType, FusedA, FusedOut, FusedSpec, LoraGrad, LoraMirror,
    LoraProj, UnifiedTensor, WgpuContext, LORA_GROUP_MAX, LORA_GROUP_MAX_RANK,
};
use std::cell::Cell;
use std::sync::Arc;

/// Unwraps the result of a kernel call whose shapes were checked when the
/// node was built.
#[expect(
    clippy::panic,
    reason = "the shapes were checked when the node was built, so an error is a bug and the op has no error path"
)]
fn must(result: Result<(), String>) {
    if let Err(e) = result {
        panic!("fused projection: {e}");
    }
}

/// Half `i` of an f16 tensor read back as 32-bit words: `to_vec` hands the
/// words back as f32 bit patterns, two halves each, low half first.
fn half_at(words: &[f32], i: usize) -> u16 {
    let word = words[i / 2].to_bits();
    (if i.is_multiple_of(2) {
        word & 0xffff
    } else {
        word >> 16
    }) as u16
}

/// The bytes of `halves` as an f16 tensor buffer, padded to whole words.
fn half_bytes(halves: &[u16]) -> Vec<u8> {
    let mut bytes: Vec<u8> = halves.iter().flat_map(|h| h.to_le_bytes()).collect();
    bytes.resize(bytes.len().next_multiple_of(4).max(4), 0);
    bytes
}

/// The LoRA adapters of a group of fused projections.
#[derive(Clone)]
pub struct FusedLora {
    /// `A` of each projection, f32 `[in, rank]`.
    pub a: Vec<Tensor>,
    /// `B` of each projection, f32 `[rank, out]`.
    pub b: Vec<Tensor>,
    /// `alpha / rank` of each projection.
    pub scales: Vec<f32>,
    /// The common rank.
    pub rank: usize,
}

/// Frozen f16 projections that share an input, concatenated along their
/// output columns, with the room and the plumbing for their LoRA adapters.
pub struct FusedProjections {
    /// `[in, sum of widths + count * rank]` f16: the base weights, then the
    /// mirror of every `A`.
    weight: UnifiedTensor,
    widths: Vec<usize>,
    lora: Option<FusedLora>,
}

impl FusedProjections {
    /// Most projections one group takes.
    pub const MAX_PROJECTIONS: usize = LORA_GROUP_MAX;

    /// Concatenates the frozen f16 `weights` (each `[in, out_i]`) and, with
    /// `lora`, the mirror of the adapters' `A` matrices after them, and fills
    /// the mirror.
    ///
    /// # Errors
    ///
    /// When a weight is not a frozen f16 matrix, the shapes of the weights or
    /// the adapters do not agree, an adapter does not require a gradient, the
    /// rank is odd or above the grouped kernels' limit, the row of the
    /// concatenated matrix has an odd length, or a readback fails.
    pub fn new(
        ctx: &WgpuContext,
        engine: &ComputeEngine,
        weights: &[&Tensor],
        lora: Option<FusedLora>,
    ) -> Result<Self, String> {
        let count = weights.len();
        if !(2..=Self::MAX_PROJECTIONS).contains(&count) {
            return Err(format!("{count} projections, want 2 to 3"));
        }
        let mut widths = Vec::with_capacity(count);
        let mut rows = 0;
        for w in weights {
            let (r, c) = match (w.data.dtype, w.data.shape.as_slice(), w.requires_grad) {
                (DType::FP16, &[r, c], false) if r > 0 && c > 0 => (r, c),
                _ => {
                    return Err(format!(
                        "a weight must be a frozen f16 matrix, got {:?} {:?}",
                        w.data.dtype, w.data.shape
                    ))
                }
            };
            if rows != 0 && r != rows {
                return Err("the weights have different input sizes".into());
            }
            rows = r;
            widths.push(c);
        }
        let base: usize = widths.iter().sum();
        let mut extra = 0;
        if let Some(l) = &lora {
            let ok = l.a.len() == count
                && l.b.len() == count
                && l.scales.len() == count
                && l.rank > 0
                && l.rank.is_multiple_of(2)
                && l.rank <= LORA_GROUP_MAX_RANK;
            if !ok {
                return Err("the adapters do not match the projections".into());
            }
            for (g, (a, b)) in l.a.iter().zip(&l.b).enumerate() {
                let a_ok = a.data.dtype == DType::FP32 && a.data.shape == [rows, l.rank];
                let b_ok = b.data.dtype == DType::FP32 && b.data.shape == [l.rank, widths[g]];
                if !a_ok || !b_ok || !a.requires_grad || !b.requires_grad {
                    return Err(format!("adapter {g} does not fit its projection"));
                }
            }
            extra = count * l.rank;
        }
        let total = base + extra;
        if !total.is_multiple_of(2) {
            return Err(format!("the concatenated row of {total} is odd"));
        }
        let mut halves = vec![0u16; rows * total];
        let mut col = 0;
        for (w, &width) in weights.iter().zip(&widths) {
            let words = pollster::block_on(w.data.to_vec(ctx))
                .map_err(|e| format!("reading a weight failed: {e:?}"))?;
            for r in 0..rows {
                for c in 0..width {
                    halves[r * total + col + c] = half_at(&words, r * width + c);
                }
            }
            col += width;
        }
        let weight = UnifiedTensor::from_bytes_with_dtype(
            ctx,
            &half_bytes(&halves),
            &[rows, total],
            DType::FP16,
        );
        let fused = Self {
            weight,
            widths,
            lora,
        };
        fused.refresh_mirror(engine, ctx)?;
        Ok(fused)
    }

    /// The common input size.
    pub fn input_size(&self) -> usize {
        self.weight.shape[0]
    }

    /// The output widths of the projections.
    pub fn widths(&self) -> &[usize] {
        &self.widths
    }

    /// Whether the group has LoRA adapters.
    pub fn has_lora(&self) -> bool {
        self.lora.is_some()
    }

    /// Whether every adapter still requires a gradient. A group is only used
    /// while this holds: the backward pass writes the adapters' gradients.
    pub fn adapters_train(&self) -> bool {
        self.lora
            .as_ref()
            .is_none_or(|l| l.a.iter().chain(&l.b).all(|t| t.requires_grad))
    }

    /// Copies every `A` into its columns of the f16 weight.
    ///
    /// # Errors
    ///
    /// When the kernel refuses the call.
    pub fn refresh_mirror(&self, engine: &ComputeEngine, ctx: &WgpuContext) -> Result<(), String> {
        let Some(l) = &self.lora else {
            return Ok(());
        };
        let base: usize = self.widths.iter().sum();
        let srcs: Vec<LoraMirror<'_>> =
            l.a.iter()
                .enumerate()
                .map(|(g, a)| LoraMirror {
                    a: &a.data,
                    col0: base + g * l.rank,
                })
                .collect();
        engine.lora_mirror_f16(ctx, &self.weight, &srcs)
    }

    /// The mirror of each `A` as read back from the f16 weight: `[in, rank]`
    /// row-major raw halves, one vector per projection. For tests.
    ///
    /// # Errors
    ///
    /// When the readback fails.
    pub fn mirror_halves(&self, ctx: &WgpuContext) -> Result<Vec<Vec<u16>>, String> {
        let Some(l) = &self.lora else {
            return Ok(Vec::new());
        };
        let (rows, total) = (self.weight.shape[0], self.weight.shape[1]);
        let words = pollster::block_on(self.weight.to_vec(ctx))
            .map_err(|e| format!("reading the weight failed: {e:?}"))?;
        let base: usize = self.widths.iter().sum();
        Ok((0..l.a.len())
            .map(|g| {
                let col0 = base + g * l.rank;
                (0..rows)
                    .flat_map(|r| (0..l.rank).map(move |j| (r, j)))
                    .map(|(r, j)| half_at(&words, r * total + col0 + j))
                    .collect()
            })
            .collect())
    }

    /// The concatenated weight as read back: `[in, sum of widths + count *
    /// rank]` row-major raw halves, the base weights then the mirrors. For
    /// tests.
    ///
    /// # Errors
    ///
    /// When the readback fails.
    pub fn weight_halves(&self, ctx: &WgpuContext) -> Result<Vec<u16>, String> {
        let words = pollster::block_on(self.weight.to_vec(ctx))
            .map_err(|e| format!("reading the weight failed: {e:?}"))?;
        let numel = self.weight.shape.iter().product::<usize>();
        Ok((0..numel).map(|i| half_at(&words, i)).collect())
    }

    /// Projection `index` as a frozen f16 `[in, width]` tensor, copied out of
    /// the concatenated buffer: what a caller that dropped its own copy needs
    /// for a pass the group cannot serve.
    ///
    /// # Errors
    ///
    /// When `index` is out of range or the readback fails.
    pub fn projection(&self, ctx: &Arc<WgpuContext>, index: usize) -> Result<Tensor, String> {
        let Some(&width) = self.widths.get(index) else {
            return Err(format!("projection {index} of {}", self.widths.len()));
        };
        let col0: usize = self.widths[..index].iter().sum();
        let (rows, total) = (self.weight.shape[0], self.weight.shape[1]);
        let words = pollster::block_on(self.weight.to_vec(ctx))
            .map_err(|e| format!("reading the weight failed: {e:?}"))?;
        let halves: Vec<u16> = (0..rows)
            .flat_map(|r| (0..width).map(move |c| (r, c)))
            .map(|(r, c)| half_at(&words, r * total + col0 + c))
            .collect();
        let data = UnifiedTensor::from_bytes_with_dtype(
            ctx,
            &half_bytes(&halves),
            &[rows, width],
            DType::FP16,
        );
        Ok(Tensor::frozen(ctx.clone(), data))
    }

    /// Writes the products of `x` (`[m, in]`) with every projection into
    /// `outs`, and with LoRA the products with the `A` matrices into `u`
    /// (`[m, count * rank]`), then adds each projection's adapter output.
    fn forward(
        &self,
        engine: &ComputeEngine,
        ctx: &WgpuContext,
        x: &UnifiedTensor,
        outs: &[&UnifiedTensor],
        u: Option<&UnifiedTensor>,
    ) {
        let mut targets: Vec<&UnifiedTensor> = outs.to_vec();
        targets.extend(u);
        must(engine.matmul_fused_into(
            ctx,
            &FusedSpec {
                a: FusedA::Plain(x),
                b: &self.weight,
                transposed_b: false,
                out: FusedOut::Split {
                    outs: &targets,
                    bias: None,
                    alpha: 1.0,
                },
            },
        ));
        if let (Some(l), Some(u)) = (&self.lora, u) {
            let projs: Vec<LoraProj<'_>> =
                l.b.iter()
                    .zip(outs)
                    .zip(&l.scales)
                    .map(|((b, &act), &scale)| LoraProj {
                        b: &b.data,
                        act,
                        scale,
                    })
                    .collect();
            must(engine.lora_ub_grouped(ctx, u, &projs));
        }
    }

    /// The backward pass of [`forward`](Self::forward) given the gradients
    /// `dys` of the outputs. The gradients of the adapters go to `targets`
    /// (`A` gradients then `B` gradients) as `beta * target + product`; the
    /// gradient of `x` is handed to `x_grad` as a function of an output
    /// buffer and an epilogue when `x_needs_grad`.
    #[expect(
        clippy::too_many_arguments,
        reason = "one argument per buffer the backward pass reads or writes"
    )]
    fn backward(
        &self,
        engine: &ComputeEngine,
        ctx: &WgpuContext,
        x: &UnifiedTensor,
        u: Option<&UnifiedTensor>,
        dys: &[&UnifiedTensor],
        targets: Option<(&[UnifiedTensor], f32)>,
        x_grad: Option<&Tensor>,
    ) {
        let m = x.shape[0];
        let mut du = None;
        if let (Some(l), Some(u)) = (&self.lora, u) {
            let d = UnifiedTensor::empty(ctx, &[m, l.rank * l.a.len()]);
            let projs: Vec<LoraProj<'_>> =
                l.b.iter()
                    .zip(dys)
                    .zip(&l.scales)
                    .map(|((b, &act), &scale)| LoraProj {
                        b: &b.data,
                        act,
                        scale,
                    })
                    .collect();
            must(engine.lora_du_grouped(ctx, &d, &projs));
            if let Some((grads, beta)) = targets {
                let count = l.a.len();
                let projs: Vec<LoraGrad<'_>> = (0..count)
                    .map(|g| LoraGrad {
                        dy: dys[g],
                        grad_a: &grads[g],
                        grad_b: &grads[count + g],
                        scale: l.scales[g],
                    })
                    .collect();
                must(engine.lora_grads_grouped(ctx, [x, u, &d], &projs, beta));
            }
            du = Some(d);
        }
        if let Some(x_tensor) = x_grad {
            let mut sources: Vec<&UnifiedTensor> = dys.to_vec();
            sources.extend(du.as_ref());
            x_tensor.accumulate_grad_with(engine, |out, epilogue| {
                must(engine.matmul_fused_into(
                    ctx,
                    &FusedSpec {
                        a: FusedA::Sources(&sources),
                        b: &self.weight,
                        transposed_b: true,
                        out: FusedOut::AlphaBeta { out, epilogue },
                    },
                ));
            });
        }
    }

    /// The tensors the group trains: every `A`, then every `B`.
    fn adapter_tensors(&self) -> Vec<Tensor> {
        self.lora
            .iter()
            .flat_map(|l| l.a.iter().chain(&l.b).cloned())
            .collect()
    }

    /// Runs the backward pass and adds the adapters' gradients into the
    /// slots of their tensors: in place when every slot holds a gradient
    /// nothing else shares, else through fresh buffers that are added to
    /// what the slot held.
    fn backward_into_slots(
        &self,
        engine: &ComputeEngine,
        ctx: &WgpuContext,
        x: &Tensor,
        u: Option<&UnifiedTensor>,
        dys: &[&UnifiedTensor],
    ) {
        let x2 = x.data.reshaped(&[dys[0].shape[0], self.input_size()]);
        let params = self.adapter_tensors();
        let size = |t: &UnifiedTensor| t.shape.iter().product::<usize>();
        let in_place = !params.is_empty()
            && params.iter().all(|p| {
                p.grad
                    .borrow()
                    .as_ref()
                    .is_some_and(|g| g.is_unique() && size(g) == size(&p.data))
            });
        let targets: Vec<UnifiedTensor> = params
            .iter()
            .map(|p| match p.grad.borrow().as_ref() {
                Some(g) if in_place => g.clone(),
                _ => UnifiedTensor::empty(ctx, &p.data.shape),
            })
            .collect();
        let beta = if in_place { 1.0 } else { 0.0 };
        let x_grad = x.requires_grad.then_some(x);
        let lora_targets = (!params.is_empty()).then_some((targets.as_slice(), beta));
        self.backward(engine, ctx, &x2, u, dys, lora_targets, x_grad);
        if !in_place {
            for (p, target) in params.iter().zip(targets) {
                let mut slot = p.grad.borrow_mut();
                *slot = Some(match slot.take() {
                    Some(existing) => {
                        let sum = UnifiedTensor::empty(ctx, &p.data.shape);
                        engine.add(ctx, &existing, &target, &sum);
                        sum
                    }
                    None => target,
                });
            }
        }
    }

    /// The gradients of the adapters as fresh buffers, `A`s then `B`s, and of
    /// `x` when it needs one, for [`AutogradNode::backward`].
    fn backward_fresh(
        &self,
        engine: &ComputeEngine,
        ctx: &WgpuContext,
        x: &Tensor,
        u: Option<&UnifiedTensor>,
        dys: &[&UnifiedTensor],
    ) -> (UnifiedTensor, Vec<UnifiedTensor>) {
        let x2 = x.data.reshaped(&[dys[0].shape[0], self.input_size()]);
        let params = self.adapter_tensors();
        let targets: Vec<UnifiedTensor> = params
            .iter()
            .map(|p| UnifiedTensor::empty(ctx, &p.data.shape))
            .collect();
        // The gradient of x goes through a scratch tensor with an empty slot.
        let scratch = Tensor {
            data: x.data.clone(),
            grad: Arc::new(std::cell::RefCell::new(None)),
            ctx: x.ctx.clone(),
            creator: None,
            requires_grad: x.requires_grad,
        };
        let x_grad = x.requires_grad.then_some(&scratch);
        let lora_targets = (!params.is_empty()).then_some((targets.as_slice(), 0.0));
        self.backward(engine, ctx, &x2, u, dys, lora_targets, x_grad);
        let grad_x = scratch
            .grad
            .borrow_mut()
            .take()
            .unwrap_or_else(|| UnifiedTensor::empty(ctx, &[1]));
        (grad_x, targets)
    }
}

/// The geometry and options of [`Tensor::fused_attention`].
#[derive(Clone, Copy, Debug)]
pub struct AttentionGeometry {
    /// Query heads.
    pub n_heads: usize,
    /// Key and value heads; divides `n_heads`.
    pub n_kv_heads: usize,
    /// Width of a head.
    pub head_dim: usize,
    /// RoPE base frequency.
    pub theta: f32,
    /// Position of the first token.
    pub pos: u32,
    /// Attention dropout probability.
    pub dropout_prob: f32,
    /// Dropout seed.
    pub seed: u64,
}

impl AttentionGeometry {
    fn options(&self) -> AttentionOptions {
        AttentionOptions {
            causal: true,
            dropout_prob: self.dropout_prob,
            seed: self.seed,
            ..Default::default()
        }
    }
}

/// Backward node of [`Tensor::fused_attention`].
pub struct FusedAttentionNode {
    x: Tensor,
    proj: Arc<FusedProjections>,
    geom: AttentionGeometry,
    shape: [usize; 3],
    q: UnifiedTensor,
    k: UnifiedTensor,
    v: UnifiedTensor,
    u: Option<UnifiedTensor>,
    out: UnifiedTensor,
    stats: UnifiedTensor,
}

impl FusedAttentionNode {
    fn view4(&self, t: &UnifiedTensor, heads: usize) -> UnifiedTensor {
        t.reshaped(&[self.shape[0], self.shape[1], heads, self.geom.head_dim])
    }

    /// The gradients of q, k and v before RoPE, as `[m, width]` matrices.
    fn flash_backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> [UnifiedTensor; 3] {
        let ctx = &self.x.ctx;
        let m = self.shape[0] * self.shape[1];
        let (qw, kw) = (self.q.shape[1], self.k.shape[1]);
        let dq = UnifiedTensor::empty(ctx, &[m, qw]);
        let dk = UnifiedTensor::empty(ctx, &[m, kw]);
        let dv = UnifiedTensor::empty(ctx, &[m, kw]);
        let (heads, kv) = (self.geom.n_heads, self.geom.n_kv_heads);
        let views = [
            self.view4(&dq, heads),
            self.view4(&dk, kv),
            self.view4(&dv, kv),
        ];
        engine.flash_attention_v2_rope_backward_with_stats(
            ctx,
            &self.view4(&self.q, heads),
            &self.view4(&self.k, kv),
            &self.view4(&self.v, kv),
            &self.view4(&self.out, heads),
            &self.view4(grad, heads),
            [&views[0], &views[1], &views[2]],
            &self.stats,
            &self.geom.options(),
            (self.geom.theta, self.geom.pos),
        );
        [dq, dk, dv]
    }
}

impl AutogradNode for FusedAttentionNode {
    fn inputs(&self) -> Vec<Tensor> {
        let mut inputs = vec![self.x.clone()];
        inputs.extend(self.proj.adapter_tensors());
        inputs
    }

    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let ctx = &self.x.ctx;
        let [dq, dk, dv] = self.flash_backward(grad, engine);
        let (grad_x, adapter) =
            self.proj
                .backward_fresh(engine, ctx, &self.x, self.u.as_ref(), &[&dq, &dk, &dv]);
        let mut grads = vec![grad_x];
        grads.extend(adapter);
        grads
    }

    fn backward_accumulate(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> bool {
        let ctx = &self.x.ctx;
        let [dq, dk, dv] = self.flash_backward(grad, engine);
        self.proj
            .backward_into_slots(engine, ctx, &self.x, self.u.as_ref(), &[&dq, &dk, &dv]);
        true
    }
}

/// Backward node of [`Tensor::fused_gate_up_silu`].
pub struct FusedGateUpNode {
    x: Tensor,
    proj: Arc<FusedProjections>,
    gate: UnifiedTensor,
    up: UnifiedTensor,
    u: Option<UnifiedTensor>,
}

impl FusedGateUpNode {
    fn silu_backward(
        &self,
        grad: &UnifiedTensor,
        engine: &ComputeEngine,
    ) -> (UnifiedTensor, UnifiedTensor) {
        let ctx = &self.x.ctx;
        let d_gate = UnifiedTensor::empty(ctx, &self.gate.shape);
        let d_up = UnifiedTensor::empty(ctx, &self.up.shape);
        let g = grad.reshaped(&self.gate.shape);
        engine.silu_gate_backward(ctx, &self.gate, &self.up, &g, &d_gate, &d_up);
        (d_gate, d_up)
    }
}

impl AutogradNode for FusedGateUpNode {
    fn inputs(&self) -> Vec<Tensor> {
        let mut inputs = vec![self.x.clone()];
        inputs.extend(self.proj.adapter_tensors());
        inputs
    }

    fn backward(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> Vec<UnifiedTensor> {
        let ctx = &self.x.ctx;
        let (d_gate, d_up) = self.silu_backward(grad, engine);
        let (grad_x, adapter) =
            self.proj
                .backward_fresh(engine, ctx, &self.x, self.u.as_ref(), &[&d_gate, &d_up]);
        let mut grads = vec![grad_x];
        grads.extend(adapter);
        grads
    }

    fn backward_accumulate(&self, grad: &UnifiedTensor, engine: &ComputeEngine) -> bool {
        let ctx = &self.x.ctx;
        let (d_gate, d_up) = self.silu_backward(grad, engine);
        self.proj
            .backward_into_slots(engine, ctx, &self.x, self.u.as_ref(), &[&d_gate, &d_up]);
        true
    }
}

/// Whether recording is on, as [`Tensor::flash_attention`] decides it.
fn recording() -> bool {
    GRAD_ENABLED.with(Cell::get)
}

impl Tensor {
    /// Self-attention over `x` (`[batch, seq, in]`) with the projections of
    /// `proj` (three: q, k and v), RoPE inside flash attention and causal
    /// masking. Returns `[batch, seq, n_heads * head_dim]`, the input of the
    /// output projection.
    ///
    /// The mirror of the adapters' `A` matrices must be current (see
    /// [`FusedProjections::refresh_mirror`]).
    ///
    /// # Panics
    ///
    /// When `x` is not 3-D, `proj` does not hold three projections of the
    /// widths `geom` implies, or the input size differs.
    #[expect(
        clippy::panic,
        reason = "shape precondition of an op that returns Tensor"
    )]
    pub fn fused_attention(
        x: &Tensor,
        proj: &Arc<FusedProjections>,
        geom: &AttentionGeometry,
        engine: &ComputeEngine,
    ) -> Tensor {
        let ctx = &x.ctx;
        let [batch, seq, hidden] = x.data.shape[..] else {
            panic!(
                "fused_attention expects [batch, seq, in], got {:?}",
                x.data.shape
            );
        };
        let (qw, kw) = (
            geom.n_heads * geom.head_dim,
            geom.n_kv_heads * geom.head_dim,
        );
        if proj.widths() != [qw, kw, kw] || proj.input_size() != hidden {
            panic!(
                "fused_attention: projections {:?} over {} inputs do not fit {geom:?} over {hidden}",
                proj.widths(),
                proj.input_size()
            );
        }
        let m = batch * seq;
        let x2 = x.data.reshaped(&[m, hidden]);
        let q = UnifiedTensor::empty(ctx, &[m, qw]);
        let k = UnifiedTensor::empty(ctx, &[m, kw]);
        let v = UnifiedTensor::empty(ctx, &[m, kw]);
        let u = proj
            .lora
            .as_ref()
            .map(|l| UnifiedTensor::empty(ctx, &[m, 3 * l.rank]));
        proj.forward(engine, ctx, &x2, &[&q, &k, &v], u.as_ref());

        let out = UnifiedTensor::empty(ctx, &[batch, seq, qw]);
        let stats = UnifiedTensor::empty(ctx, &[batch, geom.n_heads, seq, 2]);
        let view =
            |t: &UnifiedTensor, heads: usize| t.reshaped(&[batch, seq, heads, geom.head_dim]);
        let record = recording();
        engine.flash_attention_v2_rope_with_stats(
            ctx,
            &view(&q, geom.n_heads),
            &view(&k, geom.n_kv_heads),
            &view(&v, geom.n_kv_heads),
            &view(&out, geom.n_heads),
            record.then_some(&stats),
            &geom.options(),
            (geom.theta, geom.pos),
        );
        if !record {
            return Tensor::frozen(ctx.clone(), out);
        }
        let node = Arc::new(FusedAttentionNode {
            x: x.clone(),
            proj: proj.clone(),
            geom: *geom,
            shape: [batch, seq, hidden],
            q,
            k,
            v,
            u,
            out: out.clone(),
            stats,
        });
        Tensor::with_creator(ctx.clone(), out, node)
    }

    /// `silu(x Wg) * (x Wu)` for `x` of shape `[.., in]` and the two
    /// projections of `proj` (gate then up), of shape `[.., inter]`.
    ///
    /// The mirror of the adapters' `A` matrices must be current (see
    /// [`FusedProjections::refresh_mirror`]).
    ///
    /// # Panics
    ///
    /// When `x` is not 2-D or 3-D, `proj` does not hold two projections of
    /// one width, or the input size differs.
    #[expect(
        clippy::panic,
        reason = "shape precondition of an op that returns Tensor"
    )]
    pub fn fused_gate_up_silu(
        x: &Tensor,
        proj: &Arc<FusedProjections>,
        engine: &ComputeEngine,
    ) -> Tensor {
        let ctx = &x.ctx;
        let shape = x.data.shape.clone();
        let hidden = shape.last().copied().unwrap_or(0);
        let &[inter, inter_up] = proj.widths() else {
            panic!("fused_gate_up_silu expects two projections");
        };
        if !(2..=3).contains(&shape.len()) || inter != inter_up || proj.input_size() != hidden {
            panic!(
                "fused_gate_up_silu: projections {:?} over {} inputs do not fit {shape:?}",
                proj.widths(),
                proj.input_size()
            );
        }
        let m = shape.iter().product::<usize>() / hidden;
        let x2 = x.data.reshaped(&[m, hidden]);
        let gate = UnifiedTensor::empty(ctx, &[m, inter]);
        let up = UnifiedTensor::empty(ctx, &[m, inter]);
        let u = proj
            .lora
            .as_ref()
            .map(|l| UnifiedTensor::empty(ctx, &[m, 2 * l.rank]));
        proj.forward(engine, ctx, &x2, &[&gate, &up], u.as_ref());
        let mut out_shape = shape;
        if let Some(last) = out_shape.last_mut() {
            *last = inter;
        }
        let out = UnifiedTensor::empty(ctx, &out_shape);
        engine.silu_gate_fused(ctx, &gate, &up, &out.reshaped(&[m, inter]));
        if !recording() {
            return Tensor::frozen(ctx.clone(), out);
        }
        let node = Arc::new(FusedGateUpNode {
            x: x.clone(),
            proj: proj.clone(),
            gate,
            up,
            u,
        });
        Tensor::with_creator(ctx.clone(), out, node)
    }
}
