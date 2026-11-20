//! Fused products: one matmul whose operand loader and epilogue do the work
//! of the neighbouring kernels.
//!
//! [`FusedSpec`] names the operands and what the product feeds:
//!
//! - the fused QKV projection `x @ [W_q | W_k | W_v | A_q | A_k | A_v]` writes
//!   its columns to the Q, K and V buffers and to the LoRA `U` matrix with
//!   [`FusedOut::Split`], with the bias of the Qwen2 projections added on the
//!   way;
//! - the backward `[dQ | dK | dV | dU] @ W^T` reads its four matrices in place
//!   with [`FusedA::Sources`];
//! - the down projection reads the gate/up output with [`FusedA::SwiGlu`], and
//!   its backward writes the gradient of that output with
//!   [`FusedOut::SwiGluBackward`];
//! - [`FusedOut::Residual`] adds a residual, or a scaled LoRA product, to the
//!   result while storing it, and [`FusedOut::AlphaBeta`] scales and
//!   accumulates in place.
//!
//! The LoRA `A` fold needs no kind of its own: `A` is more columns of the f16
//! weight `b` (and, in the backward, `dU` is a fourth source), so the split
//! sends those columns to the `U` buffer.
//!
//! [`ComputeEngine::matmul_fused_into`] runs the cooperative-matrix kernel
//! when the device has it and
//! [`EXPERIMENTAL_ENV`](crate::adapters::EXPERIMENTAL_ENV) is `1`.
//! [`ComputeEngine::matmul_fused_reference_into`] runs the same product on any
//! device through a plain kernel with one thread per output element that
//! shares the loaders' and epilogues' element code and rounds `a` to f16 like
//! the staging does: slow, and the semantics the fast kernel is tested
//! against. Neither splits K, and neither counts f16 overflows beyond what
//! the cooperative kernel always does.

use super::coop::{self, EpilogueKind, LoaderKind, MAX_SEGMENTS, MAX_SOURCES, SOURCE_BINDING};
use super::{grid, immediate_params, uniform, Arg, ComputeEngine, Epilogue, Kernel, Params};
use crate::{DType, UnifiedTensor, WgpuContext};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

/// The first operand `a` of a fused product, in f32.
#[derive(Clone, Copy)]
pub enum FusedA<'a> {
    /// One `[M, K]` matrix.
    Plain(&'a UnifiedTensor),
    /// Two to four matrices `[M, w]` side by side along K: the product reads
    /// them as the `[M, w_0 + w_1 + ...]` matrix they would make, with no
    /// copy. Each width must be at least 1.
    Sources(&'a [&'a UnifiedTensor]),
    /// The gate/up output `[M, 2K]`: the product reads `silu(g) * u` from its
    /// first half `g` and second half `u`, as the `[M, K]` matrix `a`.
    SwiGlu(&'a UnifiedTensor),
}

/// Where a fused product goes, and what it combines with on the way.
#[derive(Clone, Copy)]
pub enum FusedOut<'a> {
    /// `out = alpha * product + beta * out`, as [`Epilogue`].
    AlphaBeta {
        /// The `[M, N]` result.
        out: &'a UnifiedTensor,
        /// Scale and accumulate factors.
        epilogue: Epilogue,
    },
    /// `out = alpha * product + residual`: the residual add, or a LoRA
    /// product with its scale `alpha` added to the base product. `out` is
    /// only written, so it may be uninitialised, and it is not `residual`.
    Residual {
        /// The `[M, N]` result.
        out: &'a UnifiedTensor,
        /// The `[M, N]` matrix added.
        residual: &'a UnifiedTensor,
        /// Factor on the product.
        alpha: f32,
    },
    /// The N columns go to one to four `[M, w]` outputs in order (their
    /// widths add up to N), each element `alpha * product`, plus `bias[col]`
    /// (a row of N values) when there is a bias.
    Split {
        /// The outputs, only written.
        outs: &'a [&'a UnifiedTensor],
        /// The bias over all N columns.
        bias: Option<&'a UnifiedTensor>,
        /// Factor on the product.
        alpha: f32,
    },
    /// The backward of `silu(g) * u`: the product is the gradient `dH` of
    /// that, `[M, N]` with N the width of `g`, and the output `d_gu` is
    /// `[M, 2N]` with `dH * u * silu'(g)` in its first half and
    /// `dH * silu(g)` in its second, from the forward's `[M, 2N]` output `gu`.
    SwiGluBackward {
        /// The `[M, 2N]` gradient of `gu`, only written.
        d_gu: &'a UnifiedTensor,
        /// The forward's gate/up output.
        gu: &'a UnifiedTensor,
    },
}

/// One fused product `a @ op(b)` with its loader and its epilogue.
#[derive(Clone, Copy)]
pub struct FusedSpec<'a> {
    /// The first operand.
    pub a: FusedA<'a>,
    /// The f16 weight, `[K, N]` or, with `transposed_b`, `[N, K]`.
    pub b: &'a UnifiedTensor,
    /// Whether `b` is stored `[N, K]`.
    pub transposed_b: bool,
    /// The epilogue and its output.
    pub out: FusedOut<'a>,
}

/// A [`FusedSpec`] after its shapes were checked, in the terms the kernels
/// take.
pub(super) struct Resolved<'a> {
    /// `(M, K, N)`.
    pub(super) dims: (u32, u32, u32),
    pub(super) transposed_b: bool,
    pub(super) loader: LoaderKind,
    pub(super) epilogue: EpilogueKind,
    /// Whether every width of `a` is a multiple of 4, so that it can be read
    /// as `vec4<f32>`.
    pub(super) a_vec4: bool,
    /// `(alpha, beta)`.
    pub(super) scalars: (f32, f32),
    /// `params[2]` then `params[3]`: the column bounds of the loader and the
    /// segment starts of the epilogue.
    pub(super) tail: [u32; 8],
    /// The first source of `a`, binding 0.
    a0: &'a UnifiedTensor,
    /// The other sources, from binding 5.
    a_more: Vec<&'a UnifiedTensor>,
    b: &'a UnifiedTensor,
    /// The output, binding 2.
    out: &'a UnifiedTensor,
    /// The epilogue's other buffers and their bindings.
    extras: Vec<(u32, &'a UnifiedTensor)>,
}

/// A fused kernel without cooperative matrices: its loader, epilogue and
/// whether `b` is stored `[N, K]`.
pub(super) type ReferenceKey = (LoaderKind, EpilogueKind, bool);

impl<'a> Resolved<'a> {
    /// The kernel's bindings: the tensors, `params`, and the overflow
    /// counter when there is one (binding 4).
    pub(super) fn bind<'s>(
        &'s self,
        params: &'s Params,
        overflows: Option<&'s wgpu::Buffer>,
    ) -> Vec<(u32, &'s dyn Arg)> {
        let mut v: Vec<(u32, &dyn Arg)> = vec![
            (0, self.a0.buffer() as &dyn Arg),
            (1, self.b.buffer() as &dyn Arg),
            (2, self.out.buffer() as &dyn Arg),
            (3, params as &dyn Arg),
        ];
        if let Some(buffer) = overflows {
            v.push((4, buffer as &dyn Arg));
        }
        for (s, t) in (SOURCE_BINDING..).zip(&self.a_more) {
            v.push((s, t.buffer() as &dyn Arg));
        }
        for &(binding, t) in &self.extras {
            v.push((binding, t.buffer() as &dyn Arg));
        }
        v
    }

    /// The words of `params`: sizes, scalars, then the bounds, `vec4s` of
    /// them in all. `kps` is K per slice, which is all of K rounded up to a
    /// step.
    pub(super) fn params_words(&self, kps: u32, vec4s: u32) -> Vec<u32> {
        let (m, k, n) = self.dims;
        let mut w = vec![
            m,
            k,
            n,
            kps,
            self.scalars.0.to_bits(),
            self.scalars.1.to_bits(),
            0,
            0,
        ];
        w.extend(self.tail.iter().take(4 * vec4s.saturating_sub(2) as usize));
        w
    }
}

/// `t`'s two dimensions.
fn dims2(t: &UnifiedTensor, what: &str) -> Result<(usize, usize), String> {
    match t.shape.as_slice() {
        [r, c] => Ok((*r, *c)),
        s => Err(format!("matmul_fused: {what} must be 2-D, got {s:?}")),
    }
}

fn to_u32(v: usize, what: &str) -> Result<u32, String> {
    u32::try_from(v).map_err(|_| format!("matmul_fused: {what} {v} does not fit 32 bits"))
}

fn numel(t: &UnifiedTensor) -> usize {
    t.shape.iter().product()
}

/// The running sums of `widths` after each of its first three entries: the
/// column bounds, in the first words of a `params` slot.
fn bounds(widths: &[usize]) -> Result<[u32; 4], String> {
    let mut out = [0u32; 4];
    let mut sum = 0usize;
    for (slot, w) in out.iter_mut().zip(widths.iter().take(3)) {
        sum += w;
        *slot = to_u32(sum, "a column bound")?;
    }
    Ok(out)
}

/// Checks `spec`'s shapes and aliasing, and turns it into kernel terms.
fn resolve<'a>(spec: &FusedSpec<'a>) -> Result<Resolved<'a>, String> {
    let b = spec.b;
    if b.dtype != DType::FP16 {
        return Err("matmul_fused: b must be f16".into());
    }
    let (b0, b1) = dims2(b, "b")?;
    let mut tail = [0u32; 8];
    // `a`: its loader, shape and tensors.
    let (loader, m, k, a_vec4, a0, a_more) = match spec.a {
        FusedA::Plain(a) => {
            let (m, k) = dims2(a, "a")?;
            (LoaderKind::Plain, m, k, k.is_multiple_of(4), a, Vec::new())
        }
        FusedA::Sources(list) => {
            let count = list.len();
            let (Some(&first), true) = (list.first(), (2..=MAX_SOURCES).contains(&count)) else {
                return Err(format!(
                    "matmul_fused: Sources takes 2 to {MAX_SOURCES} matrices, got {count}"
                ));
            };
            let mut widths = Vec::with_capacity(count);
            let m = dims2(first, "a source")?.0;
            for t in list {
                let (rows, w) = dims2(t, "a source")?;
                if rows != m || w == 0 {
                    return Err(format!(
                        "matmul_fused: every source needs {m} rows and a width, got {:?}",
                        t.shape
                    ));
                }
                widths.push(w);
            }
            let b4 = bounds(&widths)?;
            tail[..4].copy_from_slice(&b4);
            let vec4 = widths.iter().all(|w| w.is_multiple_of(4));
            let loader = LoaderKind::Sources { count: count as u8 };
            (
                loader,
                m,
                widths.iter().sum::<usize>(),
                vec4,
                first,
                list.iter().skip(1).copied().collect(),
            )
        }
        FusedA::SwiGlu(gu) => {
            let (m, two_k) = dims2(gu, "gate/up output")?;
            if two_k == 0 || !two_k.is_multiple_of(2) {
                return Err(format!(
                    "matmul_fused: the gate/up output needs an even, non-zero width, got {two_k}"
                ));
            }
            let k = two_k / 2;
            (
                LoaderKind::SwiGlu,
                m,
                k,
                k.is_multiple_of(4),
                gu,
                Vec::new(),
            )
        }
    };
    let (b_k, n) = if spec.transposed_b {
        (b1, b0)
    } else {
        (b0, b1)
    };
    if b_k != k {
        return Err(format!(
            "matmul_fused: a has K = {k} but b {:?} (transposed_b {}) has {b_k}",
            b.shape, spec.transposed_b
        ));
    }
    if m == 0 || k == 0 || n == 0 {
        return Err("matmul_fused: empty product".into());
    }
    let mut scalars = (1.0, 0.0);
    let mut extras = Vec::new();
    // The buffers the kernel reads and the ones it writes, for the aliasing check.
    let mut reads: Vec<&UnifiedTensor> = std::iter::once(a0)
        .chain(a_more.iter().copied())
        .chain(std::iter::once(b))
        .collect();
    let mut writes = Vec::new();
    let size = |t: &UnifiedTensor, want: usize, what: &str| -> Result<(), String> {
        if numel(t) == want {
            Ok(())
        } else {
            Err(format!(
                "matmul_fused: {what} {:?} needs {want} elements",
                t.shape
            ))
        }
    };
    let (epilogue, out) = match spec.out {
        FusedOut::AlphaBeta { out, epilogue } => {
            size(out, m * n, "out")?;
            scalars = (epilogue.alpha, epilogue.beta);
            writes.push(out);
            (EpilogueKind::AlphaBeta, out)
        }
        FusedOut::Residual {
            out,
            residual,
            alpha,
        } => {
            size(out, m * n, "out")?;
            size(residual, m * n, "residual")?;
            scalars = (alpha, 0.0);
            extras.push((coop::FIRST_BINDING, residual));
            reads.push(residual);
            writes.push(out);
            (EpilogueKind::Residual, out)
        }
        FusedOut::Split { outs, bias, alpha } => {
            let count = outs.len();
            let (Some(&first), true) = (outs.first(), (1..=MAX_SEGMENTS).contains(&count)) else {
                return Err(format!(
                    "matmul_fused: Split takes 1 to {MAX_SEGMENTS} outputs, got {count}"
                ));
            };
            let mut widths = Vec::with_capacity(count);
            for t in outs {
                let (rows, w) = dims2(t, "a split output")?;
                if rows != m || w == 0 {
                    return Err(format!(
                        "matmul_fused: every split output needs {m} rows and a width, got {:?}",
                        t.shape
                    ));
                }
                widths.push(w);
            }
            if widths.iter().sum::<usize>() != n {
                return Err(format!(
                    "matmul_fused: split widths {widths:?} do not add up to N = {n}"
                ));
            }
            tail[4..].copy_from_slice(&bounds(&widths)?);
            scalars = (alpha, 0.0);
            for (binding, &t) in (coop::FIRST_BINDING..).zip(outs.iter().skip(1)) {
                extras.push((binding, t));
            }
            if let Some(bias) = bias {
                size(bias, n, "bias")?;
                extras.push((coop::BIAS_BINDING, bias));
                reads.push(bias);
            }
            writes.extend(outs.iter().copied());
            let kind = EpilogueKind::ColumnSplit {
                segments: count as u8,
                bias: bias.is_some(),
            };
            (kind, first)
        }
        FusedOut::SwiGluBackward { d_gu, gu } => {
            size(d_gu, m * 2 * n, "d_gu")?;
            size(gu, m * 2 * n, "gu")?;
            extras.push((coop::FIRST_BINDING, gu));
            reads.push(gu);
            writes.push(d_gu);
            (EpilogueKind::SwiGluBackward, d_gu)
        }
    };
    // A buffer bound read-only and writable in one dispatch is a wgpu
    // validation error, and two outputs on one buffer would overwrite each
    // other.
    for (i, w) in writes.iter().enumerate() {
        let shared = reads.iter().any(|r| r.buffer() == w.buffer())
            || writes.iter().skip(i + 1).any(|o| o.buffer() == w.buffer());
        if shared {
            return Err(
                "matmul_fused: an output shares its buffer with an input or another output".into(),
            );
        }
    }
    Ok(Resolved {
        dims: (to_u32(m, "M")?, to_u32(k, "K")?, to_u32(n, "N")?),
        transposed_b: spec.transposed_b,
        loader,
        epilogue,
        a_vec4,
        scalars,
        tail,
        a0,
        a_more,
        b,
        out,
        extras,
    })
}

impl ComputeEngine {
    /// Runs `spec`: on the cooperative-matrix kernel when
    /// [`coop_matmul_active`](Self::coop_matmul_active) and a tile fits, else
    /// through [`matmul_fused_reference_into`](Self::matmul_fused_reference_into).
    /// Any M is accepted, K is not split, and `a` values f16 cannot hold are
    /// clamped and counted as in [`matmul_coop_into`](Self::matmul_coop_into).
    ///
    /// # Errors
    ///
    /// When the shapes of `spec` do not match, an output shares a buffer with
    /// an input or another output, or `b` is not f16.
    pub fn matmul_fused_into(&self, ctx: &WgpuContext, spec: &FusedSpec<'_>) -> Result<(), String> {
        let resolved = resolve(spec)?;
        if self.coop_matmul_active() && self.matmul_fused_coop(ctx, &resolved).is_ok() {
            return Ok(());
        }
        self.fused_reference(ctx, &resolved);
        Ok(())
    }

    /// Runs `spec` on the cooperative-matrix kernel, whether or not
    /// [`coop_matmul_active`](Self::coop_matmul_active).
    ///
    /// # Errors
    ///
    /// As [`matmul_fused_into`](Self::matmul_fused_into), and when the device
    /// has no cooperative matrices or no tile fits the product.
    #[doc(hidden)]
    pub fn matmul_fused_coop_into(
        &self,
        ctx: &WgpuContext,
        spec: &FusedSpec<'_>,
    ) -> Result<(), String> {
        self.matmul_fused_coop(ctx, &resolve(spec)?)
    }

    /// Runs `spec` on the plain reference kernel, on any device: one thread
    /// per output element, `a` rounded to f16, f32 sums in order of K.
    ///
    /// # Errors
    ///
    /// When the shapes of `spec` do not match, an output shares a buffer with
    /// an input or another output, or `b` is not f16.
    pub fn matmul_fused_reference_into(
        &self,
        ctx: &WgpuContext,
        spec: &FusedSpec<'_>,
    ) -> Result<(), String> {
        self.fused_reference(ctx, &resolve(spec)?);
        Ok(())
    }

    /// The reference kernel for `key`, compiled the first time it runs.
    fn fused_reference_kernel(&self, ctx: &WgpuContext, key: ReferenceKey) -> Arc<Kernel> {
        let mut cache = self
            .fused_ref
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        Arc::clone(cache.entry(key).or_insert_with(|| {
            let mut src = coop::reference_source(key.0, key.1, key.2);
            if self.immediates {
                src = immediate_params(&src);
            }
            let module = ctx
                .shader_checks()
                .module(&ctx.device, "matmul_fused_ref", src);
            Arc::new(Kernel::new(ctx, &module, "fused_ref"))
        }))
    }

    /// Records the reference kernel for `r`.
    fn fused_reference(&self, ctx: &WgpuContext, r: &Resolved<'_>) {
        let kernel = self.fused_reference_kernel(ctx, (r.loader, r.epilogue, r.transposed_b));
        let vec4s = coop::kind_params_vec4s(r.loader, r.epilogue);
        let params = uniform(ctx, bytemuck::cast_slice(&r.params_words(0, vec4s)));
        let (m, _, n) = r.dims;
        let groups = grid((m * n).div_ceil(64));
        self.run(ctx, &kernel, &r.bind(&params, None), groups);
    }
}

/// The reference kernels of an engine, by loader, epilogue and layout of `b`.
pub(super) type ReferenceKernels = Mutex<HashMap<ReferenceKey, Arc<Kernel>>>;
