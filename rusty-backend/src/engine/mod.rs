//! The compute engine: compiles the WGSL kernels and dispatches them.

use crate::{UnifiedTensor, WgpuContext};
use std::borrow::Cow;
use wgpu::{util::DeviceExt, Buffer, Device};

mod attention;
mod elementwise;
mod matmul;
mod misc;
mod norm;
mod quant;
mod training;

pub use attention::AttentionOptions;
pub use training::AdamParams;

/// The full WGSL source compiled into the engine's shader module.
fn shader_source() -> String {
    let kernels_src = include_str!("../shaders/kernels.wgsl");
    let flash_v2_src = include_str!("../shaders/flash_attention_v2.wgsl");
    let flash_v2_bw_src = include_str!("../shaders/flash_attention_v2_backward.wgsl");
    format!("{}\n{}\n{}", kernels_src, flash_v2_src, flash_v2_bw_src)
}

/// A compiled compute pipeline together with its bind group layout.
///
/// The layout is derived from the shader (`layout: None`), so it always
/// matches the bindings the entry point actually uses.
struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl Kernel {
    fn new(device: &Device, module: &wgpu::ShaderModule, entry_point: &str) -> Self {
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: None,
            module,
            entry_point,
        });
        let layout = pipeline.get_bind_group_layout(0);
        Self { pipeline, layout }
    }
}

/// Number of 64-wide workgroups needed to cover `n` elements.
fn groups64(n: usize) -> u32 {
    (n as u32).div_ceil(64)
}

/// Creates a uniform buffer holding `bytes`.
fn uniform(ctx: &WgpuContext, bytes: &[u8]) -> Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform"),
            contents: bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

pub struct ComputeEngine {
    matmul: Kernel,
    matmul_bt: Kernel,
    matmul_at: Kernel,
    rope: Kernel,
    rms_norm: Kernel,
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
    affine_dequant: Kernel,
    conv1d: Kernel,
    ce_loss: Kernel,
    ce_loss_bw: Kernel,
    grad_norm: Kernel,
    grad_clip: Kernel,
    silu_gate: Kernel,
    silu_gate_bw: Kernel,
    rms_residual: Kernel,
    grad_accum: Kernel,
    grad_scale: Kernel,
    flash_attn_v2: Kernel,
    flash_attn_v2_bw_preprocess: Kernel,
    flash_attn_v2_bw_dq: Kernel,
    flash_attn_v2_bw_dkdv: Kernel,
}

impl ComputeEngine {
    pub fn new(ctx: &WgpuContext) -> Self {
        let combined_src = shader_source();
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Kernels"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(combined_src)),
            });
        let k = |entry: &str| Kernel::new(&ctx.device, &module, entry);

        Self {
            matmul: k("matmul_tiled"),
            matmul_bt: k("matmul_bt"),
            matmul_at: k("matmul_at"),
            rope: k("rope"),
            rms_norm: k("rms_norm"),
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
            affine_dequant: k("affine_dequantize"),
            conv1d: k("conv1d"),
            ce_loss: k("cross_entropy_loss"),
            ce_loss_bw: k("cross_entropy_backward"),
            grad_norm: k("gradient_norm_squared"),
            grad_clip: k("clip_grad_by_norm"),
            silu_gate: k("silu_gate_fused"),
            silu_gate_bw: k("silu_gate_backward"),
            rms_residual: k("rms_norm_residual"),
            grad_accum: k("gradient_accumulate"),
            grad_scale: k("gradient_scale"),
            flash_attn_v2: k("flash_attention_v2_metal"),
            flash_attn_v2_bw_preprocess: k("flash_bwd_preprocess"),
            flash_attn_v2_bw_dq: k("flash_bwd_dq"),
            flash_attn_v2_bw_dkdv: k("flash_bwd_dk_dv"),
        }
    }

    /// Binds `entries` (binding index, buffer) to `kernel` and dispatches
    /// `groups` workgroups in its own command buffer.
    fn run(
        &self,
        ctx: &WgpuContext,
        kernel: &Kernel,
        entries: &[(u32, &Buffer)],
        groups: (u32, u32, u32),
    ) {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = entries
            .iter()
            .map(|(binding, buf)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: buf.as_entire_binding(),
            })
            .collect();
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &kernel.layout,
            entries: &entries,
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn copy_tensor(
        &self,
        ctx: &WgpuContext,
        src: &UnifiedTensor,
        dst: &UnifiedTensor,
        dst_offset: usize,
    ) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(
            src.buffer(),
            0,
            dst.buffer(),
            dst_offset as u64,
            src.size_in_bytes(),
        );
        ctx.queue.submit(Some(encoder.finish()));
    }
}

/// A 4-byte storage buffer bound where a kernel needs a buffer that is unused.
fn dummy_storage(ctx: &WgpuContext) -> Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dummy Storage"),
            contents: &[0u8; 4],
            usage: wgpu::BufferUsages::STORAGE,
        })
}

#[cfg(test)]
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
            naga::valid::Capabilities::empty(),
        );
        if let Err(e) = validator.validate(&module) {
            panic!("WGSL validation error:\n{}", e.emit_to_string(&src));
        }
    }
}
