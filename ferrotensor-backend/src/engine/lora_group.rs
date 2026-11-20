//! Grouped LoRA kernels: one dispatch serves up to three projections that
//! share an input (q, k and v), each with its own buffers, and the kernel
//! that keeps an f16 copy of the LoRA `A` matrices.
//!
//! A LoRA projection `Y = x @ W + s * (x @ A) @ B` costs four small
//! products in the forward and backward passes, and each ran as its own
//! dispatch per projection. The products are tiny (rank 8), so a dispatch
//! costs about as much as its work. Here the products of q, k and v run in
//! one dispatch each:
//!
//! - [`ComputeEngine::lora_ub_grouped`]: `Y_g += s_g * U_g @ B_g`, with the
//!   `U_g = x @ A_g` of the projections side by side in one `[M, ld]` matrix.
//! - [`ComputeEngine::lora_du_grouped`]: `dU_g = s_g * dY_g @ B_g^T`, written
//!   to the same side-by-side layout.
//! - [`ComputeEngine::lora_grads_grouped`]: `grad_A_g += x^T @ dU_g` and
//!   `grad_B_g += s_g * U_g^T @ dY_g`.
//! - [`ComputeEngine::lora_mirror_f16`]: copies each f32 `A_g` into its
//!   columns of an f16 weight matrix, to run again after every optimizer
//!   step, adapter load and resume, or the copy goes stale.
//!
//! Projection `g` owns columns `g * rank .. (g + 1) * rank` of `U` and `dU`.
//! The kernels come from the template `shaders/lora_group.wgsl`, one module
//! per kernel, group size and rank, compiled the first time they run.

use super::{immediate_params, uniform, Arg, ComputeEngine, Kernel};
use crate::{DType, UnifiedTensor, WgpuContext};
use std::sync::{Arc, PoisonError};
use wgpu::Buffer;

/// Most projections one grouped dispatch serves.
pub const LORA_GROUP_MAX: usize = 3;

/// Largest LoRA rank the grouped kernels take; a larger rank needs more
/// workgroup memory than every device offers.
pub const LORA_GROUP_MAX_RANK: usize = 16;

/// Threads per workgroup of the reducing kernels.
const THREADS: u32 = 256;

/// Largest workgroup count per grid dimension.
const MAX_GROUPS: u32 = 65_535;

/// A projection of [`ComputeEngine::lora_ub_grouped`] and
/// [`ComputeEngine::lora_du_grouped`].
#[derive(Clone, Copy)]
pub struct LoraProj<'a> {
    /// `B`, f32 `[rank, n]`.
    pub b: &'a UnifiedTensor,
    /// The projection's output `Y` (`m * n` f32, updated) for
    /// `lora_ub_grouped`, its output gradient `dY` (read) for
    /// `lora_du_grouped`.
    pub act: &'a UnifiedTensor,
    /// The LoRA scale `s`.
    pub scale: f32,
}

/// A projection of [`ComputeEngine::lora_grads_grouped`].
#[derive(Clone, Copy)]
pub struct LoraGrad<'a> {
    /// The output gradient `dY`, `m * n` f32.
    pub dy: &'a UnifiedTensor,
    /// `grad_A`, f32 with `k * rank` elements, laid out `[k, rank]`.
    pub grad_a: &'a UnifiedTensor,
    /// `grad_B`, f32 `[rank, n]`.
    pub grad_b: &'a UnifiedTensor,
    /// The LoRA scale `s`.
    pub scale: f32,
}

/// A projection of [`ComputeEngine::lora_mirror_f16`].
#[derive(Clone, Copy)]
pub struct LoraMirror<'a> {
    /// `A`, f32 `[rows, rank]`.
    pub a: &'a UnifiedTensor,
    /// The first column of the projection's copy in the f16 matrix, even.
    pub col0: usize,
}

/// Which kernel of the template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
    Ub,
    Du,
    Grads,
    Mirror,
}

impl Kind {
    /// The name of its section in the template.
    fn name(self) -> &'static str {
        match self {
            Self::Ub => "ub",
            Self::Du => "du",
            Self::Grads => "grads",
            Self::Mirror => "mirror",
        }
    }

    fn entry(self) -> &'static str {
        match self {
            Self::Ub => "lora_ub",
            Self::Du => "lora_du",
            Self::Grads => "lora_grads",
            Self::Mirror => "lora_mirror",
        }
    }
}

/// A compiled grouped kernel: the kernel, its group size and the rank it
/// was generated for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct Key {
    kind: Kind,
    count: u32,
    rank: u32,
}

/// A storage binding of a generated kernel.
struct Slot {
    name: String,
    access: &'static str,
    ty: &'static str,
}

fn slot(name: String, access: &'static str, ty: &'static str) -> Slot {
    Slot { name, access, ty }
}

/// The storage bindings of the kernel for `key` in binding order, which is
/// also the order the host binds buffers in; the parameters follow them.
fn slots(key: Key) -> Vec<Slot> {
    let per = |prefix: &str, access, ty| -> Vec<Slot> {
        (0..key.count)
            .map(|g| slot(format!("{prefix}{g}"), access, ty))
            .collect()
    };
    let mut s = Vec::new();
    match key.kind {
        Kind::Ub => {
            s.push(slot("u".into(), "read", "f32"));
            s.extend(per("y", "read_write", "f32"));
            s.extend(per("b", "read", "f32"));
        }
        Kind::Du => {
            s.push(slot("du".into(), "read_write", "f32"));
            s.extend(per("dy", "read", "f32"));
            s.extend(per("b", "read", "f32"));
        }
        Kind::Grads => {
            s.push(slot("x".into(), "read", "f32"));
            s.push(slot("u".into(), "read", "f32"));
            s.push(slot("du".into(), "read", "f32"));
            s.extend(per("dy", "read", "f32"));
            s.extend(per("ga", "read_write", "f32"));
            s.extend(per("gb", "read_write", "f32"));
        }
        Kind::Mirror => {
            s.push(slot("dst".into(), "read_write", "u32"));
            s.extend(per("a", "read", "f32"));
        }
    }
    s
}

/// Expands every `//#repeat` .. `//#end` block of `body` once per projection
/// below `count`, replacing `$g` with the projection, `$t0` with twice it and
/// `$t1` with twice it plus one.
fn expand(body: &str, count: u32) -> String {
    let mut out = String::new();
    let mut block: Option<Vec<&str>> = None;
    for line in body.lines() {
        let marker = line.trim();
        if marker == "//#repeat" && block.is_none() {
            block = Some(Vec::new());
        } else if marker == "//#end" && block.is_some() {
            if let Some(lines) = block.take() {
                for g in 0..count {
                    for l in &lines {
                        out += &l
                            .replace("$t0", &(2 * g).to_string())
                            .replace("$t1", &(2 * g + 1).to_string())
                            .replace("$g", &g.to_string());
                        out.push('\n');
                    }
                }
            }
        } else if let Some(lines) = block.as_mut() {
            lines.push(line);
        } else {
            out += line;
            out.push('\n');
        }
    }
    out
}

/// The WGSL of the kernel for `key`, entry point [`Kind::entry`].
///
/// # Errors
///
/// Returns an error when the template has no section for the kernel, or for
/// a mirror kernel with an odd rank, which cannot copy whole words.
pub(super) fn source(key: Key) -> Result<String, String> {
    if key.kind == Kind::Mirror && !key.rank.is_multiple_of(2) {
        return Err(format!("lora_mirror needs an even rank, got {}", key.rank));
    }
    // A Windows checkout with core.autocrlf has CRLF line ends.
    let template = include_str!("../shaders/lora_group.wgsl").replace("\r\n", "\n");
    let body = template
        .split("//#kernel ")
        .filter_map(|section| section.split_once('\n'))
        .find(|(name, _)| *name == key.kind.name())
        .map(|(_, body)| body)
        .ok_or_else(|| format!("lora_group.wgsl has no section {:?}", key.kind.name()))?;
    let slots = slots(key);
    let mut decls = String::new();
    for (binding, s) in slots.iter().enumerate() {
        decls += &format!(
            "@group(0) @binding({binding}) var<storage, {access}> {name}: array<{ty}>;\n",
            access = s.access,
            name = s.name,
            ty = s.ty,
        );
    }
    let params_len = if key.kind == Kind::Mirror { 2 } else { 3 };
    decls += &format!(
        "@group(0) @binding({}) var<uniform> params: array<vec4<u32>, {params_len}>;\n",
        slots.len()
    );
    Ok(format!(
        "const RANK: u32 = {rank}u;\nconst RED_LEN: u32 = {red}u;\n{decls}\n{body}",
        rank = key.rank,
        red = key.rank * THREADS,
        body = expand(body, key.count),
    ))
}

/// A matrix tensor's `(rows, columns)`.
fn matrix(what: &str, name: &str, t: &UnifiedTensor) -> Result<(usize, usize), String> {
    match (t.dtype, t.shape.as_slice()) {
        (DType::FP32, &[r, c]) if r > 0 && c > 0 => Ok((r, c)),
        _ => Err(format!(
            "{what}: {name} must be a non-empty 2-D f32 tensor, got {:?} {:?}",
            t.dtype, t.shape
        )),
    }
}

fn to_u32(what: &str, name: &str, v: usize) -> Result<u32, String> {
    u32::try_from(v).map_err(|_| format!("{what}: {name} = {v} does not fit 32 bits"))
}

/// An f32 tensor of `want` elements.
fn sized(what: &str, name: &str, t: &UnifiedTensor, want: usize) -> Result<(), String> {
    let have: usize = t.shape.iter().product();
    if t.dtype != DType::FP32 || have != want {
        return Err(format!(
            "{what}: {name} must be f32 with {want} elements, got {:?} {:?}",
            t.dtype, t.shape
        ));
    }
    Ok(())
}

/// What the projections of one call agree on.
struct Dims {
    /// Rows of `U` and `dU`.
    m: u32,
    /// Row length of `U` and `dU`.
    ld: u32,
    rank: u32,
    /// Columns of each projection, zero past the group.
    ns: [u32; LORA_GROUP_MAX],
}

impl Dims {
    /// Reads and checks the shapes of `side` (`U` or `dU`, `[m, ld]`) and of
    /// each projection's `B` (`[rank, n]`) and activation (`m * n` f32).
    fn new(
        what: &str,
        side: &UnifiedTensor,
        projs: &[(&UnifiedTensor, &UnifiedTensor)],
    ) -> Result<Self, String> {
        let count = projs.len();
        if count == 0 || count > LORA_GROUP_MAX {
            return Err(format!(
                "{what}: {count} projections, want 1 to {LORA_GROUP_MAX}"
            ));
        }
        let (m, ld) = matrix(what, "the U side matrix", side)?;
        let mut rank = 0;
        let mut ns = [0u32; LORA_GROUP_MAX];
        for (g, (b, act)) in projs.iter().enumerate() {
            let (r, n) = matrix(what, "B", b)?;
            if g == 0 {
                rank = r;
            } else if r != rank {
                return Err(format!("{what}: ranks {rank} and {r} in one group"));
            }
            sized(what, "the activation", act, m * n)?;
            ns[g] = to_u32(what, "n", n)?;
            to_u32(what, "m * n", m * n)?;
        }
        if rank > LORA_GROUP_MAX_RANK || ld < count * rank {
            return Err(format!(
                "{what}: rank {rank} with {count} projections over rows of {ld}: \
                 the rank is at most {LORA_GROUP_MAX_RANK} and the rows hold {count} ranks"
            ));
        }
        to_u32(what, "m * ld", m * ld)?;
        Ok(Self {
            m: to_u32(what, "m", m)?,
            ld: to_u32(what, "ld", ld)?,
            rank: to_u32(what, "rank", rank)?,
            ns,
        })
    }

    fn max_n(&self) -> u32 {
        self.ns.into_iter().max().unwrap_or(0)
    }
}

/// `[s0 bits, s1 bits, s2 bits, 0]`.
fn scale_words(scales: impl Iterator<Item = f32>) -> [u32; 4] {
    let mut words = [0u32; 4];
    for (word, s) in words.iter_mut().zip(scales) {
        *word = s.to_bits();
    }
    words
}

fn grid_check(what: &str, (x, y, z): (u32, u32, u32)) -> Result<(), String> {
    if x > MAX_GROUPS || y > MAX_GROUPS || z > MAX_GROUPS {
        return Err(format!("{what}: a grid of {x} x {y} x {z} is too large"));
    }
    Ok(())
}

impl ComputeEngine {
    fn lora_group_kernel(&self, ctx: &WgpuContext, key: Key) -> Result<Arc<Kernel>, String> {
        let mut cache = self
            .lora_group
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(kernel) = cache.get(&key) {
            return Ok(Arc::clone(kernel));
        }
        let mut src = source(key)?;
        if self.immediates {
            src = immediate_params(&src);
        }
        let entry = key.kind.entry();
        let module = ctx.shader_checks().module(&ctx.device, entry, src);
        let kernel = Arc::new(Kernel::new(ctx, &module, entry));
        cache.insert(key, Arc::clone(&kernel));
        Ok(kernel)
    }

    /// Records the kernel for `key` over `buffers` (in [`slots`] order) with
    /// the parameter words `params`.
    fn lora_group_run(
        &self,
        ctx: &WgpuContext,
        key: Key,
        buffers: &[&Buffer],
        params: &[u32],
        groups: (u32, u32, u32),
    ) -> Result<(), String> {
        let what = key.kind.entry();
        grid_check(what, groups)?;
        let limits = ctx.device.limits();
        if buffers.len() as u32 > limits.max_storage_buffers_per_shader_stage {
            return Err(format!(
                "{what}: needs {} storage buffers per stage, this device allows {}",
                buffers.len(),
                limits.max_storage_buffers_per_shader_stage
            ));
        }
        let reduces = key.kind == Kind::Du || key.kind == Kind::Grads;
        if reduces
            && (THREADS > self.small_m_max_threads
                || key.rank * THREADS * 4 > self.small_m_max_bytes)
        {
            return Err(format!(
                "{what}: rank {} needs {} bytes of workgroup memory and {THREADS} threads, \
                 this device has {} and {}",
                key.rank,
                key.rank * THREADS * 4,
                self.small_m_max_bytes,
                self.small_m_max_threads
            ));
        }
        let kernel = self.lora_group_kernel(ctx, key)?;
        let params = uniform(ctx, bytemuck::cast_slice(params));
        let mut entries: Vec<(u32, &dyn Arg)> = buffers
            .iter()
            .enumerate()
            .map(|(binding, &buffer)| (binding as u32, buffer as &dyn Arg))
            .collect();
        entries.push((buffers.len() as u32, &params));
        self.run(ctx, &kernel, &entries, groups);
        Ok(())
    }

    /// Adds `s_g * U_g @ B_g` to each projection's output `Y_g` in one
    /// dispatch: `Y_g[i, c] += s_g * sum_j u[i, g * rank + j] * B_g[j, c]`.
    /// `u` is f32 `[m, ld]` with `ld >= projs.len() * rank`.
    ///
    /// # Errors
    ///
    /// Returns an error when `projs` is empty or has more than
    /// [`LORA_GROUP_MAX`] projections, the shapes or ranks do not agree, the
    /// rank is above [`LORA_GROUP_MAX_RANK`], or the device cannot run the
    /// kernel.
    pub fn lora_ub_grouped(
        &self,
        ctx: &WgpuContext,
        u: &UnifiedTensor,
        projs: &[LoraProj<'_>],
    ) -> Result<(), String> {
        let what = "lora_ub_grouped";
        let pairs: Vec<_> = projs.iter().map(|p| (p.b, p.act)).collect();
        let d = Dims::new(what, u, &pairs)?;
        let key = Key {
            kind: Kind::Ub,
            count: projs.len() as u32,
            rank: d.rank,
        };
        let mut buffers: Vec<&Buffer> = vec![u.buffer()];
        buffers.extend(projs.iter().map(|p| p.act.buffer()));
        buffers.extend(projs.iter().map(|p| p.b.buffer()));
        let scales = scale_words(projs.iter().map(|p| p.scale));
        let params = [[d.m, d.ld, 0, 0], [d.ns[0], d.ns[1], d.ns[2], 0], scales].concat();
        let groups = (d.max_n().div_ceil(64), d.m.div_ceil(4), key.count);
        self.lora_group_run(ctx, key, &buffers, &params, groups)
    }

    /// Writes `dU_g = s_g * dY_g @ B_g^T` for each projection into `du`, in
    /// one dispatch: `du[i, g * rank + j] = s_g * sum_c dY_g[i, c] * B_g[j, c]`.
    /// `du` is f32 `[m, ld]` with `ld >= projs.len() * rank`; its other
    /// columns are left alone. `projs[g].act` is `dY_g`.
    ///
    /// # Errors
    ///
    /// As [`lora_ub_grouped`](Self::lora_ub_grouped).
    pub fn lora_du_grouped(
        &self,
        ctx: &WgpuContext,
        du: &UnifiedTensor,
        projs: &[LoraProj<'_>],
    ) -> Result<(), String> {
        let what = "lora_du_grouped";
        let pairs: Vec<_> = projs.iter().map(|p| (p.b, p.act)).collect();
        let d = Dims::new(what, du, &pairs)?;
        let key = Key {
            kind: Kind::Du,
            count: projs.len() as u32,
            rank: d.rank,
        };
        let mut buffers: Vec<&Buffer> = vec![du.buffer()];
        buffers.extend(projs.iter().map(|p| p.act.buffer()));
        buffers.extend(projs.iter().map(|p| p.b.buffer()));
        let scales = scale_words(projs.iter().map(|p| p.scale));
        let params = [[d.m, d.ld, 0, 0], [d.ns[0], d.ns[1], d.ns[2], 0], scales].concat();
        let groups = (1, d.m.div_ceil(4), key.count);
        self.lora_group_run(ctx, key, &buffers, &params, groups)
    }

    /// Accumulates both gradients of each projection in one dispatch:
    /// `grad_A_g = beta * grad_A_g + x^T @ dU_g` and
    /// `grad_B_g = beta * grad_B_g + s_g * U_g^T @ dY_g`, where `U_g` and
    /// `dU_g` are columns `g * rank ..` of `u` and `du`. A `beta` of 0
    /// overwrites the gradients without reading them, 1 accumulates.
    /// `x` is f32 `[m, k]`, `u` and `du` f32 `[m, ld]`.
    ///
    /// # Errors
    ///
    /// Returns an error as [`lora_ub_grouped`](Self::lora_ub_grouped) does,
    /// or when `x`, `u`, `du` or a gradient has a size that does not match.
    pub fn lora_grads_grouped(
        &self,
        ctx: &WgpuContext,
        [x, u, du]: [&UnifiedTensor; 3],
        projs: &[LoraGrad<'_>],
        beta: f32,
    ) -> Result<(), String> {
        let what = "lora_grads_grouped";
        // The `b` of Dims is only read for its shape: grad_B is [rank, n].
        let pairs: Vec<_> = projs.iter().map(|p| (p.grad_b, p.dy)).collect();
        let d = Dims::new(what, u, &pairs)?;
        let (xm, k) = matrix(what, "x", x)?;
        if xm != d.m as usize {
            return Err(format!("{what}: x has {xm} rows, u has {}", d.m));
        }
        sized(what, "dU", du, (d.m * d.ld) as usize)?;
        let k32 = to_u32(what, "k", k)?;
        to_u32(what, "m * k", xm * k)?;
        for p in projs {
            sized(what, "grad_A", p.grad_a, k * d.rank as usize)?;
        }
        let key = Key {
            kind: Kind::Grads,
            count: projs.len() as u32,
            rank: d.rank,
        };
        let mut buffers: Vec<&Buffer> = vec![x.buffer(), u.buffer(), du.buffer()];
        buffers.extend(projs.iter().map(|p| p.dy.buffer()));
        buffers.extend(projs.iter().map(|p| p.grad_a.buffer()));
        buffers.extend(projs.iter().map(|p| p.grad_b.buffer()));
        let scales = scale_words(projs.iter().map(|p| p.scale));
        let params = [
            [d.m, k32, d.ld, beta.to_bits()],
            [d.ns[0], d.ns[1], d.ns[2], 0],
            scales,
        ]
        .concat();
        let groups = (k32.max(d.max_n()).div_ceil(32), 1, 2 * key.count);
        self.lora_group_run(ctx, key, &buffers, &params, groups)
    }

    /// Copies each f32 `A_g` (`[rows, rank]`) into columns `col0 ..
    /// col0 + rank` of the f16 matrix `dst` (`[rows, ld]`, the layout of
    /// [`UnifiedTensor::from_f32_as_f16`]) in one dispatch, rounding to
    /// nearest even like that function. Other columns are left alone.
    /// Values past the f16 range are not meaningful, as for any f16 copy.
    ///
    /// Run it after every optimizer step that changes an `A`, after an
    /// adapter load and after a resume; a copy that is not refreshed is
    /// silently stale.
    ///
    /// # Errors
    ///
    /// Returns an error when `srcs` is empty or has more than
    /// [`LORA_GROUP_MAX`] entries, `dst` is not a 2-D f16 tensor, the ranks
    /// or row counts differ, the rank is odd or above
    /// [`LORA_GROUP_MAX_RANK`], `ld` or a `col0` is odd (a thread owns whole
    /// 32-bit words), a copy runs past `ld` or two copies overlap, or the
    /// device cannot run the kernel.
    pub fn lora_mirror_f16(
        &self,
        ctx: &WgpuContext,
        dst: &UnifiedTensor,
        srcs: &[LoraMirror<'_>],
    ) -> Result<(), String> {
        let what = "lora_mirror_f16";
        let count = srcs.len();
        if count == 0 || count > LORA_GROUP_MAX {
            return Err(format!(
                "{what}: {count} sources, want 1 to {LORA_GROUP_MAX}"
            ));
        }
        let (rows, ld) = match (dst.dtype, dst.shape.as_slice()) {
            (DType::FP16, &[r, c]) if r > 0 && c > 0 => (r, c),
            _ => {
                return Err(format!(
                    "{what}: dst must be a non-empty 2-D f16 tensor, got {:?} {:?}",
                    dst.dtype, dst.shape
                ))
            }
        };
        if dst.size_in_bytes() < (rows * ld * 2) as u64 {
            return Err(format!("{what}: dst holds fewer than {rows} x {ld} halves"));
        }
        let (_, rank) = matrix(what, "A", srcs[0].a)?;
        if rank > LORA_GROUP_MAX_RANK || !rank.is_multiple_of(2) || !ld.is_multiple_of(2) {
            return Err(format!(
                "{what}: rank {rank} and row length {ld} must be even, the rank at most {LORA_GROUP_MAX_RANK}"
            ));
        }
        let mut col0s = [0u32; 4];
        let mut spans = Vec::new();
        for (g, s) in srcs.iter().enumerate() {
            let (r, c) = matrix(what, "A", s.a)?;
            if (r, c) != (rows, rank) {
                return Err(format!(
                    "{what}: A is {r} x {c}, want {rows} x {rank} like the first"
                ));
            }
            if !s.col0.is_multiple_of(2) || s.col0 + rank > ld {
                return Err(format!(
                    "{what}: columns {} .. {} must start even and fit the {ld} columns of dst",
                    s.col0,
                    s.col0 + rank
                ));
            }
            col0s[g] = to_u32(what, "col0", s.col0)?;
            spans.push((s.col0, s.col0 + rank));
        }
        spans.sort_unstable();
        if spans.windows(2).any(|w| w[0].1 > w[1].0) {
            return Err(format!("{what}: two copies overlap"));
        }
        let key = Key {
            kind: Kind::Mirror,
            count: count as u32,
            rank: to_u32(what, "rank", rank)?,
        };
        let words = to_u32(what, "rows * rank / 2", rows * rank / 2)?;
        to_u32(what, "rows * ld", rows * ld)?;
        let blocks = words.div_ceil(64);
        let groups = (
            blocks.min(MAX_GROUPS),
            blocks.div_ceil(MAX_GROUPS),
            key.count,
        );
        let mut buffers: Vec<&Buffer> = vec![dst.buffer()];
        buffers.extend(srcs.iter().map(|s| s.a.buffer()));
        let params = [
            [
                to_u32(what, "rows", rows)?,
                to_u32(what, "ld", ld)?,
                key.rank,
                0,
            ],
            col0s,
        ]
        .concat();
        self.lora_group_run(ctx, key, &buffers, &params, groups)
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::{expand, slots, source, Key, Kind, LORA_GROUP_MAX, LORA_GROUP_MAX_RANK};
    use crate::engine::{immediate_params, IMMEDIATE_BYTES};

    fn keys() -> Vec<Key> {
        let mut keys = Vec::new();
        for kind in [Kind::Ub, Kind::Du, Kind::Grads, Kind::Mirror] {
            for count in 1..=LORA_GROUP_MAX as u32 {
                for rank in [1, 2, 3, 4, 8, 12, LORA_GROUP_MAX_RANK as u32] {
                    let key = Key { kind, count, rank };
                    // The mirror copies whole words: an odd rank has no kernel.
                    if kind == Kind::Mirror && !rank.is_multiple_of(2) {
                        assert!(source(key).is_err(), "{key:?}");
                    } else {
                        keys.push(key);
                    }
                }
            }
        }
        keys
    }

    fn validate(key: Key, src: &str, caps: naga::valid::Capabilities) {
        let module = match naga::front::wgsl::parse_str(src) {
            Ok(m) => m,
            Err(e) => panic!("{key:?}: WGSL parse error:\n{}", e.emit_to_string(src)),
        };
        let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps);
        if let Err(e) = validator.validate(&module) {
            panic!("{key:?}: WGSL validation error:\n{}", e.emit_to_string(src));
        }
    }

    #[test]
    fn every_kernel_validates() {
        for key in keys() {
            let src = match source(key) {
                Ok(src) => src,
                Err(e) => panic!("{key:?}: {e}"),
            };
            assert!(src.contains(key.kind.entry()), "{key:?}");
            assert!(!src.contains("$g") && !src.contains("$t"), "{key:?}");
            validate(
                key,
                &src,
                naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
            );
        }
    }

    #[test]
    fn every_kernel_validates_with_immediates() {
        for key in keys() {
            let src = match source(key) {
                Ok(src) => immediate_params(&src),
                Err(e) => panic!("{key:?}: {e}"),
            };
            assert!(!src.contains("var<uniform>"), "{key:?}");
            let module = match naga::front::wgsl::parse_str(&src) {
                Ok(m) => m,
                Err(e) => panic!("{key:?}: WGSL parse error:\n{}", e.emit_to_string(&src)),
            };
            assert_eq!(
                naga::valid::ImmediateSlots::size_for_module(&module),
                IMMEDIATE_BYTES,
                "{key:?}"
            );
            validate(
                key,
                &src,
                naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32
                    | naga::valid::Capabilities::IMMEDIATES,
            );
        }
    }

    /// Each kernel binds what the host passes, in this order, and stays
    /// within the 8 storage buffers every device offers where it can.
    #[test]
    fn binding_counts_follow_the_group_size() {
        for count in 1..=LORA_GROUP_MAX as u32 {
            let n = |kind| {
                slots(Key {
                    kind,
                    count,
                    rank: 8,
                })
                .len() as u32
            };
            assert_eq!(n(Kind::Ub), 1 + 2 * count);
            assert_eq!(n(Kind::Du), 1 + 2 * count);
            assert_eq!(n(Kind::Grads), 3 + 3 * count);
            assert_eq!(n(Kind::Mirror), 1 + count);
        }
        let names: Vec<_> = slots(Key {
            kind: Kind::Grads,
            count: 2,
            rank: 8,
        })
        .into_iter()
        .map(|s| s.name)
        .collect();
        assert_eq!(
            names,
            ["x", "u", "du", "dy0", "dy1", "ga0", "ga1", "gb0", "gb1"]
        );
    }

    #[test]
    fn expand_repeats_blocks_per_projection() {
        let body =
            "head $g\n//#repeat\nif (t == $t0) { a$g; }\nif (t == $t1) { b$g; }\n//#end\ntail\n";
        assert_eq!(
            expand(body, 2),
            "head $g\nif (t == 0) { a0; }\nif (t == 1) { b0; }\nif (t == 2) { a1; }\nif (t == 3) { b1; }\ntail\n"
        );
        assert_eq!(expand(body, 0), "head $g\ntail\n");
    }
}
