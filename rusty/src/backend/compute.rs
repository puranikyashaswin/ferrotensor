//! Compute Engine - GPU Kernel Dispatch
//!
//! This module handles compilation and dispatch of GPU compute kernels.

use std::borrow::Cow;
use std::collections::HashMap;
use wgpu::{BindGroup, BindGroupLayout, Buffer, ComputePipeline, Device, ShaderModule};

use super::KERNELS_WGSL;

/// Parameters for AdamW optimizer (must match WGSL struct layout)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdamParams {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub t: f32,
    pub _pad: [f32; 2],
}

impl Default for AdamParams {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            t: 1.0,
            _pad: [0.0; 2],
        }
    }
}

/// GPU Compute Engine for kernel dispatch
pub struct ComputeEngine {
    shader_module: ShaderModule,
    pipelines: HashMap<String, ComputePipeline>,
    bind_group_layouts: HashMap<String, BindGroupLayout>,
}

impl ComputeEngine {
    /// Create a new compute engine with all kernels compiled.
    pub fn new(device: &Device) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Rusty Kernels"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(KERNELS_WGSL)),
        });

        let mut pipelines = HashMap::new();
        let bind_group_layouts = HashMap::new();

        // Compile all kernels
        let kernel_names = [
            // Binary ops
            "add",
            "sub",
            "mul",
            "div_kernel",
            // Scalar ops
            "add_scalar",
            "mul_scalar",
            // MatMul
            "matmul",
            "matmul_bt",
            "matmul_at",
            // Activations
            "relu",
            "silu",
            "gelu",
            "sigmoid",
            "tanh_act",
            // Normalization
            "rms_norm",
            "layer_norm",
            // Softmax & Loss
            "softmax",
            "log_softmax",
            "cross_entropy",
            // Positional
            "rope",
            // Embedding
            "embedding_lookup",
            // Backward ops
            "add_backward",
            "sub_backward",
            "mul_backward",
            "relu_backward",
            "silu_backward",
            "gelu_backward",
            "softmax_backward",
            // Optimizer
            "sgd_update",
            "adamw_update",
            // Loss
            "mse_loss",
            "mse_loss_backward",
            // Quantization
            "affine_dequantize",
            // Utilities
            "copy",
            "fill",
        ];

        for name in kernel_names {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &shader_module,
                entry_point: name,
            });
            pipelines.insert(name.to_string(), pipeline);
        }

        Self {
            shader_module,
            pipelines,
            bind_group_layouts,
        }
    }

    /// Get a compute pipeline by name.
    pub fn get_pipeline(&self, name: &str) -> Option<&ComputePipeline> {
        self.pipelines.get(name)
    }

    /// Create a bind group for binary operations (a, b, out).
    pub fn create_binary_bind_group(
        &self,
        device: &Device,
        pipeline: &ComputePipeline,
        a: &Buffer,
        b: &Buffer,
        out: &Buffer,
    ) -> BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Binary Op Bind Group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
            ],
        })
    }

    /// Create a bind group for unary operations (in, out).
    pub fn create_unary_bind_group(
        &self,
        device: &Device,
        pipeline: &ComputePipeline,
        input: &Buffer,
        output: &Buffer,
    ) -> BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Unary Op Bind Group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        })
    }

    /// Create a bind group for matrix multiplication.
    pub fn create_matmul_bind_group(
        &self,
        device: &Device,
        pipeline: &ComputePipeline,
        a: &Buffer,
        b: &Buffer,
        out: &Buffer,
        params: &Buffer,
    ) -> BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Bind Group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch a compute shader.
    pub fn dispatch(
        &self,
        device: &Device,
        queue: &wgpu::Queue,
        pipeline_name: &str,
        bind_group: &BindGroup,
        workgroups: (u32, u32, u32),
    ) {
        let pipeline = self
            .pipelines
            .get(pipeline_name)
            .unwrap_or_else(|| panic!("Pipeline '{}' not found", pipeline_name));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(pipeline_name),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Calculate workgroup size for 1D dispatch.
    pub fn workgroups_1d(size: u32) -> u32 {
        (size + 63) / 64
    }

    /// Calculate workgroup size for 2D dispatch (16x16 tiles).
    pub fn workgroups_2d(rows: u32, cols: u32) -> (u32, u32) {
        ((cols + 15) / 16, (rows + 15) / 16)
    }
}

impl std::fmt::Debug for ComputeEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputeEngine")
            .field("num_pipelines", &self.pipelines.len())
            .finish()
    }
}
