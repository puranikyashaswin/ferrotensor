//! The compute engine: compiles the WGSL kernels and dispatches them.

use crate::context::Dispatch;
use crate::kernel_timing::KernelLabel;
use crate::{UnifiedTensor, WgpuContext};
use std::cell::OnceCell;
use std::collections::HashMap;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};
use wgpu::{util::DeviceExt, Buffer};

mod attention;
mod coop;
mod elementwise;
mod flash_coop;
mod lora_group;
mod matmul;
mod matmul_fused;
mod misc;
mod norm;
mod quant;
mod skinny;
mod small_m;
mod training;

pub use attention::{AttentionOptions, RopeFreqs, RopeTableError};
pub(crate) use coop::autotune_requested;
pub use coop::{AutotuneReport, CoopConfig, CoopTile, AUTOTUNE_ENV};
pub use flash_coop::FLASH_COOP_ENV;
pub use lora_group::{LoraGrad, LoraMirror, LoraProj, LORA_GROUP_MAX, LORA_GROUP_MAX_RANK};
pub use matmul::{Epilogue, MatmulPath};
pub use matmul_fused::{FusedA, FusedOut, FusedSpec};
pub use skinny::SkinnyTile;
pub use small_m::{SmallMLaunch, SmallMTile};
pub use training::{AdamParams, FlatLayout, FLAT_MAX_ELEMS, MASKED_CE_SKIP, SUM_SQUARES_CHUNK};

/// The full WGSL source compiled into the engine's shader module.
fn shader_source() -> String {
    let kernels_src = include_str!("../shaders/kernels.wgsl");
    // A Windows checkout with core.autocrlf has CRLF line ends, which the
    // multi-line patterns of half_b_matmul and q4_b_matmul would not match.
    let matmul_src = &include_str!("../shaders/matmul.wgsl").replace("\r\n", "\n");
    let flash_v2_src = include_str!("../shaders/flash_attention_v2.wgsl");
    let flash_v2_bw_src = include_str!("../shaders/flash_attention_v2_backward.wgsl");
    let matmul_h_src = half_b_matmul(matmul_src);
    let matmul_q4_src = q4_b_matmul(matmul_src);
    let matvec_src = &include_str!("../shaders/matvec.wgsl").replace("\r\n", "\n");
    let matvec_h_src = half_w_matvec(matvec_src);
    let matvec_q4_src = include_str!("../shaders/matvec_q4.wgsl");
    let thin_src = &include_str!("../shaders/matmul_thin.wgsl").replace("\r\n", "\n");
    let thin_h_src = half_w_thin_matmul(thin_src);
    let split_reduce_src = include_str!("../shaders/matmul_split_reduce.wgsl");
    format!(
        "{kernels_src}\n{matmul_src}\n{matmul_h_src}\n{matmul_q4_src}\n{matvec_src}\n{matvec_h_src}\n{matvec_q4_src}\n{thin_src}\n{thin_h_src}\n{split_reduce_src}\n{flash_v2_src}\n{flash_v2_bw_src}"
    )
}

/// Bytes of immediate data every module declares when kernel parameters go
/// through immediates: the largest parameters, the 20 words of
/// `FlashV2Params` and `FlashBwdParams`.
pub(crate) const IMMEDIATE_BYTES: u32 = 80;

/// `src` with every `var<uniform>` kernel parameter turned into a
/// `var<immediate>`. naga sizes a module's immediate data by its first
/// immediate variable, so an unused one of [`IMMEDIATE_BYTES`] comes first;
/// each entry point still reads only its own parameters, from offset 0.
fn immediate_params(src: &str) -> String {
    const UNIFORM: &str = "var<uniform>";
    let mut out = String::with_capacity(src.len() + 64);
    out.push_str("var<immediate> ft_immediate_size: array<vec4<u32>, 5>;\n");
    for line in src.lines() {
        // Every parameter is declared on one line as
        // `@group(0) @binding(N) var<uniform> name: type;`.
        match line.find(UNIFORM) {
            Some(at) => {
                out.push_str("var<immediate>");
                out.push_str(&line[at + UNIFORM.len()..]);
            }
            None => out.push_str(line),
        }
        out.push('\n');
    }
    out
}

/// `matvec_h`: the matrix-vector kernel with `w` stored as f16 pairs packed
/// into `u32` words, generated from matvec.wgsl like [`half_b_matmul`]. With
/// an even column count a column pair is one word, read once.
fn half_w_matvec(src: &str) -> String {
    const W_DECL: &str = "var<storage, read> mv_w: array<f32>;";
    const W_LOAD: &str = "    if (n + 1u < cols) { return vec2<f32>(mv_w[i], mv_w[i + 1u]); }\n    return vec2<f32>(mv_w[i], 0.0);";
    assert!(
        src.contains(W_DECL) && src.contains(W_LOAD),
        "matvec.wgsl changed"
    );
    src.replace(
        W_LOAD,
        "    if ((cols & 1u) == 0u) { return unpack2x16float(mv_w[i >> 1u]); }\n    let lo = unpack2x16float(mv_w[i >> 1u])[i & 1u];\n    if (n + 1u >= cols) { return vec2<f32>(lo, 0.0); }\n    return vec2<f32>(lo, unpack2x16float(mv_w[(i + 1u) >> 1u])[(i + 1u) & 1u]);",
    )
    .replace(W_DECL, "var<storage, read> mv_w: array<u32>;")
    .replace("fn matvec(", "fn matvec_h(")
    .replace("mv_", "mvh_")
    .replace("MV_", "MVH_")
}

/// `matmul_thin_h`: the thin matmul with its wide operand `w` stored as f16
/// pairs packed into `u32` words, generated from matmul_thin.wgsl like
/// [`half_b_matmul`]. A pair that starts on an even element is one word,
/// read once.
fn half_w_thin_matmul(src: &str) -> String {
    const W_DECL: &str = "var<storage, read> mt_w: array<f32>;";
    const W_LOAD: &str = "    if (both) { return vec2<f32>(mt_w[i], mt_w[i + 1u]); }\n    return vec2<f32>(mt_w[i], 0.0);";
    assert!(
        src.contains(W_DECL) && src.contains(W_LOAD),
        "matmul_thin.wgsl changed"
    );
    src.replace(
        W_LOAD,
        "    let word = unpack2x16float(mt_w[i >> 1u]);\n    if ((i & 1u) == 0u) { return select(vec2<f32>(word.x, 0.0), word, both); }\n    if (!both) { return vec2<f32>(word.y, 0.0); }\n    return vec2<f32>(word.y, unpack2x16float(mt_w[(i >> 1u) + 1u]).x);",
    )
    .replace(W_DECL, "var<storage, read> mt_w: array<u32>;")
    .replace("fn matmul_thin(", "fn matmul_thin_h(")
    .replace("mt_", "mth_")
    .replace("MT_", "MTH_")
}

/// `matmul_blocked_h`: the matmul kernel with `b` stored as f16 pairs packed
/// into `u32` words (element `i` in the low half of word `i / 2` when `i` is
/// even, the high half when odd). Generated from matmul.wgsl so the two
/// kernels cannot drift apart; only the `b` binding and its loader differ.
fn half_b_matmul(src: &str) -> String {
    const B_DECL: &str = "var<storage, read> mmb_b: array<f32>;";
    const B_LOAD: &str =
        "    if (mode == 1u) { return mmb_b[gn * k + gk]; }\n    return mmb_b[gk * n + gn];";
    assert!(
        src.contains(B_DECL) && src.contains(B_LOAD),
        "matmul.wgsl changed"
    );
    src.replace(
        B_LOAD,
        "    var i = gk * n + gn;\n    if (mode == 1u) { i = gn * k + gk; }\n    return unpack2x16float(mmb_b[i >> 1u])[i & 1u];",
    )
    .replace(B_DECL, "var<storage, read> mmb_b: array<u32>;")
    .replace("matmul_blocked", "matmul_blocked_h")
    .replace("mmb_", "mmh_")
    .replace("MMB_", "MMH_")
}

/// `matmul_blocked_q4`: the matmul kernel with `b` in the 4-bit format of
/// [`crate::q4`] (codes then per-group scale/min words in one `u32`
/// buffer; logical `b[K, N]` stored in `[N, K]` order). Generated from
/// matmul.wgsl like [`half_b_matmul`]; only the `b` binding, its loader and
/// the mode whose `b` runs along k differ. The storage index is
/// `gn * k + gk` in mode 0 and `gk * n + gn` in mode 1 (the reverse of the
/// f16 kernel, because of the `[N, K]` order), so mode 0 is the one staged
/// along k.
/// The params' last word carries the layout in its low 8 bits and log2 of
/// the group size `G` above them; the scale words start after the codes, at
/// `ceil(k * n / G) * G / 8`.
fn q4_b_matmul(src: &str) -> String {
    const B_DECL: &str = "var<storage, read> mmb_b: array<f32>;";
    const B_LOAD: &str =
        "    if (mode == 1u) { return mmb_b[gn * k + gk]; }\n    return mmb_b[gk * n + gn];";
    const MODE: &str = "    let mode = mmb_params[0].w;";
    const K_INNER: &str = "fn mmb_b_k_inner(mode: u32) -> bool { return mode == 1u; }";
    assert!(
        src.contains(B_DECL) && src.contains(B_LOAD) && src.contains(MODE) && src.contains(K_INNER),
        "matmul.wgsl changed"
    );
    src.replace(
        B_LOAD,
        "    var j = gn * k + gk;\n    if (mode == 1u) { j = gk * n + gn; }\n    let lg = mmb_params[0].w >> 8u;\n    let q = (mmb_b[j >> 3u] >> ((j & 7u) * 4u)) & 15u;\n    let groups = (k * n + (1u << lg) - 1u) >> lg;\n    let sm = unpack2x16float(mmb_b[(groups << (lg - 3u)) + (j >> lg)]);\n    return f32(q) * sm.x + sm.y;",
    )
    .replace(MODE, "    let mode = mmb_params[0].w & 255u;")
    .replace(
        K_INNER,
        "fn mmb_b_k_inner(mode: u32) -> bool { return mode == 0u; }",
    )
    .replace(B_DECL, "var<storage, read> mmb_b: array<u32>;")
    .replace("matmul_blocked", "matmul_blocked_q4")
    .replace("mmb_", "mmq_")
    .replace("MMB_", "MMQ_")
}

/// Entry points of kernels.wgsl that call `arrayLength`, so they must be
/// bound to exactly their tensors' bytes (see [`needs_exact_size`]). A test
/// checks this list against the shader source.
const EXACT_SIZE_ENTRIES: [&str; 29] = [
    "rope",
    "rope_out",
    "silu",
    "add",
    "mul",
    "dequantize",
    "top_k",
    "add_backward",
    "mul_backward",
    "silu_backward_real",
    "sgd_update",
    "adamw_update",
    "sub",
    "sub_backward",
    "mse_loss",
    "mse_loss_backward",
    "relu",
    "relu_backward",
    "affine_dequantize",
    "gradient_norm_squared",
    "clip_grad_by_norm",
    "silu_gate_fused",
    "silu_gate_backward",
    "gradient_accumulate",
    "gradient_scale",
    "fp32_to_fp16",
    "fp16_to_fp32",
    "embedding_gather",
    "embedding_scatter",
];

/// Whether the kernel at `entry_point` reads the length of a buffer with
/// `arrayLength`. Only kernels.wgsl does, so every other kernel takes its
/// sizes from its parameters and may be bound to the whole of a pooled
/// buffer.
fn needs_exact_size(entry_point: &str) -> bool {
    EXACT_SIZE_ENTRIES.contains(&entry_point)
}

/// A compiled compute pipeline together with its bind group layout.
///
/// The layout is derived from the shader (`layout: None`), so it always
/// matches the bindings the entry point actually uses.
struct Kernel {
    /// Shared with the dispatches waiting to be encoded.
    pipeline: Arc<wgpu::ComputePipeline>,
    layout: wgpu::BindGroupLayout,
    entry_point: &'static str,
}

impl Kernel {
    /// Builds the pipeline for `entry_point` of `module`, zeroing workgroup
    /// memory as `ctx`'s shader checks ask.
    fn new(ctx: &WgpuContext, module: &wgpu::ShaderModule, entry_point: &'static str) -> Self {
        let pipeline = Arc::new(ctx.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: None,
                module,
                entry_point: Some(entry_point),
                compilation_options: ctx.shader_checks().compilation_options(),
                cache: None,
            },
        ));
        let layout = pipeline.get_bind_group_layout(0);
        Self {
            pipeline,
            layout,
            entry_point,
        }
    }
}

/// Workgroup grid for `groups` 1-D workgroups. Grids wider than 65535 (the
/// per-dimension limit) spill into y; the kernels rebuild a linear index.
fn grid(groups: u32) -> (u32, u32, u32) {
    const MAX: u32 = 65535;
    if groups <= MAX {
        (groups, 1, 1)
    } else {
        (MAX, groups.div_ceil(MAX), 1)
    }
}

/// Grid of 64-wide workgroups covering `n` elements.
fn groups64(n: usize) -> (u32, u32, u32) {
    grid((n as u32).div_ceil(64))
}

/// A kernel's parameters: the bytes of its `var<uniform>` (a `var<immediate>`
/// when the context uses immediates), and the uniform buffer holding them
/// when it does not.
struct Params {
    bytes: Vec<u8>,
    buffer: OnceCell<Buffer>,
}

impl Params {
    /// The uniform buffer holding the parameters, made on first use.
    fn buffer(&self, ctx: &WgpuContext) -> &Buffer {
        let bytes = &self.bytes;
        self.buffer.get_or_init(|| uniform_buffer(ctx, bytes))
    }
}

/// The parameters `bytes` for one or more dispatches on `ctx`. Without
/// immediates they are put in a uniform buffer at once.
fn uniform(ctx: &WgpuContext, bytes: &[u8]) -> Params {
    let params = Params {
        bytes: bytes.to_vec(),
        buffer: OnceCell::new(),
    };
    if !ctx.uses_immediates() {
        params.buffer(ctx);
    }
    params
}

/// What a dispatch binds: a tensor's buffer, a buffer made for this call
/// alone, or the kernel's parameters.
enum ArgRef<'a> {
    Buffer(&'a Buffer),
    Scratch(&'a Buffer),
    Params(&'a Params),
}

/// A buffer made for one call and dropped after it. A dispatch that binds
/// one does not cache its bind group, which could never be used again.
struct Scratch(Buffer);

impl Arg for Scratch {
    fn arg(&self) -> ArgRef<'_> {
        ArgRef::Scratch(&self.0)
    }
}

/// A value [`ComputeEngine::run`] can bind, so a buffer and the kernel's
/// parameters share one entry list.
trait Arg {
    fn arg(&self) -> ArgRef<'_>;
}

impl Arg for Buffer {
    fn arg(&self) -> ArgRef<'_> {
        ArgRef::Buffer(self)
    }
}

impl Arg for Params {
    fn arg(&self) -> ArgRef<'_> {
        ArgRef::Params(self)
    }
}

/// Creates a uniform buffer holding `bytes`. Callers drop it once the
/// dispatch is recorded, so it is counted as awaiting GPU from the start.
fn uniform_buffer(ctx: &WgpuContext, bytes: &[u8]) -> Buffer {
    ctx.memory().count_dispatch_buffer(bytes.len() as u64);
    ctx.counters.uniform_buffers.fetch_add(1, Relaxed);
    ctx.counters
        .uniform_bytes
        .fetch_add(bytes.len() as u64, Relaxed);
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform"),
            contents: bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

/// Compiled WGSL kernels and the methods that dispatch them.
///
/// Every method records its work into the context's pending command encoder
/// (see [`WgpuContext`]) and returns without waiting; reading a result (e.g.
/// with [`UnifiedTensor::to_vec`](crate::UnifiedTensor::to_vec)) submits it
/// and waits for the GPU.
/// Tensors are dense, row-major `f32` unless stated otherwise.
pub struct ComputeEngine {
    matmul: Kernel,
    matmul_h: Kernel,
    matmul_q4: Kernel,
    matmul_thin: Kernel,
    matmul_thin_h: Kernel,
    matvec: Kernel,
    matvec_h: Kernel,
    matvec_q4: Kernel,
    rope: Kernel,
    rope_out: Kernel,
    emb_gather: Kernel,
    emb_scatter: Kernel,
    rms_norm: Kernel,
    rms_norm_bw: Kernel,
    rms_norm_bw_weight: Kernel,
    silu: Kernel,
    softmax: Kernel,
    add: Kernel,
    mul: Kernel,
    sub: Kernel,
    add_bw: Kernel,
    sub_bw: Kernel,
    mul_bw: Kernel,
    silu_bw: Kernel,
    sgd: Kernel,
    adamw: Kernel,
    mse_loss: Kernel,
    mse_loss_bw: Kernel,
    relu: Kernel,
    relu_bw: Kernel,
    softmax_bw: Kernel,
    dequant: Kernel,
    top_k: Kernel,
    argmax_rows: Kernel,
    affine_dequant: Kernel,
    conv1d: Kernel,
    conv1d_bw: Kernel,
    ce_loss: Kernel,
    ce_loss_bw: Kernel,
    grad_norm: Kernel,
    grad_clip: Kernel,
    silu_gate: Kernel,
    silu_gate_bw: Kernel,
    rms_residual: Kernel,
    grad_accum: Kernel,
    grad_scale: Kernel,
    masked_ce: Kernel,
    sum_squares: Kernel,
    flash_attn_v2: Kernel,
    flash_attn_decode: Kernel,
    flash_attn_v2_bw_dq: Kernel,
    flash_attn_v2_bw_dkdv: Kernel,
    flash_group_sum: Kernel,
    split_reduce: Kernel,
    /// Small-M matmul kernels, compiled the first time a tile and layout run.
    small_m: Mutex<HashMap<small_m::SmallMKey, Arc<Kernel>>>,
    /// The device's workgroup memory in bytes, which bounds the small-M tiles.
    small_m_max_bytes: u32,
    /// The most threads a workgroup of this device can have.
    small_m_max_threads: u32,
    /// Workgroups a small-M matmul's K split aims for on this device.
    split_target: u32,
    /// Cooperative-matrix matmul kernels, when the device has them.
    coop: Option<coop::Coop>,
    /// Skinny matmul kernels, compiled the first time a tile and layout run.
    skinny: Mutex<HashMap<skinny::SkinnyKey, Arc<Kernel>>>,
    /// The reference kernels of the fused products, compiled the first time they run.
    fused_ref: matmul_fused::ReferenceKernels,
    /// Grouped LoRA kernels, compiled the first time a kernel, group size and
    /// rank run.
    lora_group: Mutex<HashMap<lora_group::Key, Arc<Kernel>>>,
    /// Whether the kernels read their parameters from immediates.
    immediates: bool,
    /// The buffer bound to the RoPE table slot of a rotation without a table,
    /// created the first time one runs.
    rope_stub: std::sync::OnceLock<Buffer>,
}

impl ComputeEngine {
    /// The 4-byte buffer for the table binding of an unscaled rotation. It
    /// is one buffer for the engine's life, so those dispatches stay
    /// cacheable.
    fn rope_stub(&self, ctx: &WgpuContext) -> &Buffer {
        self.rope_stub.get_or_init(|| {
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("RoPE Table Stub"),
                    contents: &[0u8; 4],
                    usage: wgpu::BufferUsages::STORAGE,
                })
        })
    }

    /// Compiles every kernel for `ctx`'s device, reading its parameters from
    /// immediates when [`WgpuContext::uses_immediates`] says so.
    pub fn new(ctx: &WgpuContext) -> Self {
        let immediates = ctx.uses_immediates();
        let combined_src = if immediates {
            immediate_params(&shader_source())
        } else {
            shader_source()
        };
        let module = ctx
            .shader_checks()
            .module(&ctx.device, "Kernels", combined_src);
        let k = |entry: &'static str| Kernel::new(ctx, &module, entry);
        let limits = ctx.device.limits();

        Self {
            matmul: k("matmul_blocked"),
            matmul_h: k("matmul_blocked_h"),
            matmul_q4: k("matmul_blocked_q4"),
            matmul_thin: k("matmul_thin"),
            matmul_thin_h: k("matmul_thin_h"),
            matvec: k("matvec"),
            matvec_h: k("matvec_h"),
            matvec_q4: k("matvec_q4"),
            rope: k("rope"),
            rope_out: k("rope_out"),
            emb_gather: k("embedding_gather"),
            emb_scatter: k("embedding_scatter"),
            rms_norm: k("rms_norm"),
            rms_norm_bw: k("rms_norm_backward"),
            rms_norm_bw_weight: k("rms_norm_backward_weight"),
            silu: k("silu"),
            softmax: k("softmax"),
            add: k("add"),
            mul: k("mul"),
            sub: k("sub"),
            add_bw: k("add_backward"),
            sub_bw: k("sub_backward"),
            mul_bw: k("mul_backward"),
            silu_bw: k("silu_backward_real"),
            sgd: k("sgd_update"),
            adamw: k("adamw_update"),
            mse_loss: k("mse_loss"),
            mse_loss_bw: k("mse_loss_backward"),
            relu: k("relu"),
            relu_bw: k("relu_backward"),
            softmax_bw: k("softmax_backward"),
            dequant: k("dequantize"),
            top_k: k("top_k"),
            argmax_rows: k("argmax_rows"),
            affine_dequant: k("affine_dequantize"),
            conv1d: k("conv1d"),
            conv1d_bw: k("conv1d_backward"),
            ce_loss: k("cross_entropy_loss"),
            ce_loss_bw: k("cross_entropy_backward"),
            grad_norm: k("gradient_norm_squared"),
            grad_clip: k("clip_grad_by_norm"),
            silu_gate: k("silu_gate_fused"),
            silu_gate_bw: k("silu_gate_backward"),
            rms_residual: k("rms_norm_residual"),
            grad_accum: k("gradient_accumulate"),
            grad_scale: k("gradient_scale"),
            masked_ce: k("masked_cross_entropy"),
            sum_squares: k("sum_squares_partial"),
            flash_attn_v2: k("flash_attention_v2_metal"),
            flash_attn_decode: k("flash_attention_decode"),
            flash_attn_v2_bw_dq: k("flash_bwd_dq"),
            flash_attn_v2_bw_dkdv: k("flash_bwd_dk_dv"),
            flash_group_sum: k("flash_group_sum"),
            split_reduce: k("matmul_split_reduce"),
            small_m: Mutex::new(HashMap::new()),
            small_m_max_bytes: limits.max_compute_workgroup_storage_size,
            small_m_max_threads: limits
                .max_compute_invocations_per_workgroup
                .min(limits.max_compute_workgroup_size_x),
            split_target: small_m::split_target(ctx.adapter.get_info().device_type),
            coop: coop::Coop::new(
                ctx,
                small_m::split_target(ctx.adapter.get_info().device_type),
            ),
            skinny: Mutex::new(HashMap::new()),
            fused_ref: Mutex::new(HashMap::new()),
            lora_group: Mutex::new(HashMap::new()),
            immediates,
            rope_stub: std::sync::OnceLock::new(),
        }
    }

    /// Whether the kernels read their parameters from immediates rather than
    /// a uniform buffer per dispatch.
    pub fn uses_immediates(&self) -> bool {
        self.immediates
    }

    /// Binds `entries` (binding index, buffer or parameters) to `kernel` and
    /// records a dispatch of `groups` workgroups into `ctx`'s pending work.
    fn run(
        &self,
        ctx: &WgpuContext,
        kernel: &Kernel,
        entries: &[(u32, &dyn Arg)],
        groups: (u32, u32, u32),
    ) {
        let label = KernelLabel {
            entry: kernel.entry_point,
            layout: "",
            size_class: "",
        };
        self.run_labeled(ctx, kernel, label, entries, groups);
    }

    /// [`run`](Self::run), with the `label` kernel timing reports the
    /// dispatch under.
    fn run_labeled(
        &self,
        ctx: &WgpuContext,
        kernel: &Kernel,
        label: KernelLabel,
        entries: &[(u32, &dyn Arg)],
        groups: (u32, u32, u32),
    ) {
        self.run_tagged(ctx, kernel, label, entries, groups, None);
    }

    /// [`run_labeled`](Self::run_labeled), for a dispatch that is part of
    /// the autotuner's sample `tune`, whose GPU time is read back for it.
    fn run_tagged(
        &self,
        ctx: &WgpuContext,
        kernel: &Kernel,
        label: KernelLabel,
        entries: &[(u32, &dyn Arg)],
        groups: (u32, u32, u32),
        tune: Option<u64>,
    ) {
        let mut immediates = None;
        // A group that binds a buffer made for this dispatch alone is not
        // worth caching.
        let mut cacheable = true;
        let exact_sizes = needs_exact_size(kernel.entry_point);
        let mut bound = Vec::with_capacity(entries.len());
        for &(binding, arg) in entries {
            // A kernel that bounds its loops by `arrayLength` sees a pooled
            // buffer larger than its tensor as the tensor's bytes only; any
            // other kernel takes its sizes from its parameters, so it binds
            // the whole buffer (size 0), and one group serves every tensor
            // size the buffer is reused for.
            let (buffer, size) = match arg.arg() {
                ArgRef::Buffer(buffer) if exact_sizes => {
                    (buffer, ctx.pool.bound_size(buffer).unwrap_or(0))
                }
                ArgRef::Buffer(buffer) => (buffer, 0),
                ArgRef::Params(params) if self.immediates => {
                    immediates = Some(params.bytes.clone());
                    continue;
                }
                ArgRef::Params(params) => {
                    cacheable = false;
                    (params.buffer(ctx), 0)
                }
                ArgRef::Scratch(buffer) => {
                    cacheable = false;
                    (buffer, 0)
                }
            };
            bound.push((binding, buffer, size));
        }
        let create = || {
            let entries: Vec<wgpu::BindGroupEntry<'_>> = bound
                .iter()
                .map(|&(binding, buffer, size)| wgpu::BindGroupEntry {
                    binding,
                    resource: match wgpu::BufferSize::new(size) {
                        Some(size) => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: 0,
                            size: Some(size),
                        }),
                        None => buffer.as_entire_binding(),
                    },
                })
                .collect();
            ctx.counters.bind_groups.fetch_add(1, Relaxed);
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &kernel.layout,
                entries: &entries,
            })
        };
        let bind_group = if cacheable {
            let (group, hit) =
                ctx.pool
                    .bind_groups()
                    .get_or_create(&kernel.pipeline, &bound, create);
            let counter = if hit {
                &ctx.counters.bind_group_hits
            } else {
                &ctx.counters.bind_group_misses
            };
            counter.fetch_add(1, Relaxed);
            group
        } else {
            create()
        };
        ctx.counters.dispatches.fetch_add(1, Relaxed);
        ctx.record_dispatch(Dispatch {
            pipeline: Arc::clone(&kernel.pipeline),
            bind_group,
            groups,
            label,
            tune,
            immediates,
        });
    }

    /// Copies all of `src` into `dst`, starting at byte offset `dst_offset`.
    pub fn copy_tensor(
        &self,
        ctx: &WgpuContext,
        src: &UnifiedTensor,
        dst: &UnifiedTensor,
        dst_offset: usize,
    ) {
        ctx.record(|encoder| {
            encoder.copy_buffer_to_buffer(
                src.buffer(),
                0,
                dst.buffer(),
                dst_offset as u64,
                src.size_in_bytes(),
            );
        });
    }
}

/// A 4-byte storage buffer bound where a kernel needs a buffer that is unused.
/// Counted like [`uniform_buffer`].
fn dummy_storage(ctx: &WgpuContext) -> Scratch {
    ctx.memory().count_dispatch_buffer(4);
    let buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dummy Storage"),
            contents: &[0u8; 4],
            usage: wgpu::BufferUsages::STORAGE,
        });
    Scratch(buffer)
}

#[cfg(test)]
#[allow(clippy::panic)]
mod shader_tests {
    /// Parses and validates the combined WGSL with naga, so shader errors
    /// fail `cargo test` without needing a GPU.
    #[test]
    fn combined_wgsl_validates() {
        let src = super::shader_source();
        let module = match naga::front::wgsl::parse_str(&src) {
            Ok(m) => m,
            Err(e) => panic!("WGSL parse error:\n{}", e.emit_to_string(&src)),
        };
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            // The f16 kernels unpack halves with `unpack2x16float`, which naga
            // 30 gates behind this capability; wgpu grants it from the adapter's
            // `SHADER_F16_IN_F32` downlevel flag.
            naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        );
        if let Err(e) = validator.validate(&module) {
            panic!("WGSL validation error:\n{}", e.emit_to_string(&src));
        }
    }

    /// The immediates form of the kernels parses and validates, keeps no
    /// uniform, and gives the module the immediate size the device is asked
    /// for.
    #[test]
    fn immediate_params_validate() {
        let Some(tile) = super::SmallMTile::for_rows(32, 16 * 1024) else {
            panic!("no small-M tile for 32 rows");
        };
        let sources = [
            super::immediate_params(&super::shader_source()),
            super::immediate_params(&super::small_m::source(tile, true, true)),
        ];
        for src in &sources {
            assert!(!src.contains("var<uniform>"));
            let module = match naga::front::wgsl::parse_str(src) {
                Ok(m) => m,
                Err(e) => panic!("WGSL parse error:\n{}", e.emit_to_string(src)),
            };
            assert_eq!(
                naga::valid::ImmediateSlots::size_for_module(&module),
                super::IMMEDIATE_BYTES
            );
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32
                    | naga::valid::Capabilities::IMMEDIATES,
            );
            if let Err(e) = validator.validate(&module) {
                panic!("WGSL validation error:\n{}", e.emit_to_string(src));
            }
        }
    }

    /// The functions of `src` that call `arrayLength`, each a chunk that
    /// starts at a line beginning `fn `.
    fn array_length_functions(src: &str) -> Vec<String> {
        let needle = ["array", "Length("].concat();
        src.split("\nfn ")
            .skip(1)
            .filter(|chunk| chunk.contains(&needle))
            .map(|chunk| chunk.split('(').next().unwrap_or_default().to_string())
            .collect()
    }

    /// The kernels bound to exact sizes are the ones that call `arrayLength`:
    /// a kernel missing from the list would read the slack of a pooled
    /// buffer as part of its tensor.
    #[test]
    fn exact_size_entries_are_the_kernels_that_call_array_length() {
        let mut found = array_length_functions(&super::shader_source());
        found.sort_unstable();
        let mut listed: Vec<String> = super::EXACT_SIZE_ENTRIES
            .iter()
            .map(|&entry| entry.to_string())
            .collect();
        listed.sort_unstable();
        assert_eq!(found, listed);
        assert!(super::needs_exact_size("add"));
        assert!(!super::needs_exact_size("rms_norm"));
        assert!(!super::needs_exact_size("matmul_blocked"));
    }

    /// The kernels generated in Rust source never call `arrayLength`, so the
    /// check above covers every kernel.
    #[test]
    fn generated_kernels_do_not_call_array_length() {
        let needle = ["array", "Length("].concat();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/engine");
        let mut dirs = vec![root];
        while let Some(dir) = dirs.pop() {
            for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{dir:?}: {e}")) {
                let path = entry.unwrap_or_else(|e| panic!("{dir:?}: {e}")).path();
                if path.is_dir() {
                    dirs.push(path);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let text =
                        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
                    assert!(!text.contains(&needle), "{path:?} calls arrayLength");
                }
            }
        }
    }

    /// Only the parameter declarations change, keeping their names and types.
    #[test]
    fn immediate_params_rewrites_declarations() {
        let src = "@group(0) @binding(3) var<uniform> mv_params: vec4<u32>; // K, N\nfn f() {}";
        assert_eq!(
            super::immediate_params(src),
            "var<immediate> ft_immediate_size: array<vec4<u32>, 5>;\n\
             var<immediate> mv_params: vec4<u32>; // K, N\nfn f() {}\n"
        );
    }
}
