use std::borrow::Cow;
use std::sync::Arc;
use wgpu::{util::DeviceExt, Adapter, Buffer, Device, Instance, Queue};

pub use wgpu; // Re-export wgpu for downstream crates

pub mod pool;
use pool::BufferPool;

#[derive(Debug)]
pub struct WgpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub pool: Arc<BufferPool>,
}

impl WgpuContext {
    pub async fn new() -> Self {
        Self::new_with_options(wgpu::PowerPreference::HighPerformance, false).await
    }

    pub async fn new_with_options(
        power_preference: wgpu::PowerPreference,
        force_fallback_adapter: bool,
    ) -> Self {
        let instance = Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                force_fallback_adapter,
                compatible_surface: None,
            })
            .await
            .expect("Failed to find adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("Failed to create device");
        let pool = Arc::new(BufferPool::new(
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        ));
        Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            pool,
        }
    }

    pub fn adapter_info(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?})", info.name, info.backend)
    }
}

/// Data type for tensors - supports mixed precision training
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    FP32, // 32-bit float (default)
    FP16, // 16-bit float (half precision for memory savings)
}

impl DType {
    /// Size of a single element in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::FP32 => 4,
            DType::FP16 => 2,
        }
    }
}

impl Default for DType {
    fn default() -> Self {
        DType::FP32
    }
}

#[derive(Debug, Clone)]
pub struct UnifiedTensor {
    data: Arc<TensorData>,
    pub shape: Vec<usize>,
    pub dtype: DType, // Track tensor data type for mixed precision
}

#[derive(Debug)]
struct TensorData {
    buffer: Arc<Buffer>,
    size_in_bytes: u64,
    pool: Option<Arc<BufferPool>>,
}

impl Drop for TensorData {
    fn drop(&mut self) {
        if let Some(pool) = &self.pool {
            pool.return_buffer(self.size_in_bytes, self.buffer.clone());
        }
    }
}

impl UnifiedTensor {
    pub fn buffer(&self) -> &Buffer {
        &self.data.buffer
    }

    pub fn size_in_bytes(&self) -> u64 {
        self.data.size_in_bytes
    }

    pub fn write_data(&self, context: &WgpuContext, data: &[u8]) {
        context.queue.write_buffer(self.buffer(), 0, data);
    }
}

impl UnifiedTensor {
    pub fn new(context: &WgpuContext, data: &[f32], shape: &[usize]) -> Self {
        let size_in_bytes = (data.len() * 4) as u64;
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tensor"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        Self {
            data: Arc::new(TensorData {
                buffer: Arc::new(buffer),
                size_in_bytes,
                pool: None,
            }),
            shape: shape.to_vec(),
            dtype: DType::FP32,
        }
    }

    pub fn empty(context: &WgpuContext, shape: &[usize]) -> Self {
        Self::empty_with_dtype(context, shape, DType::FP32)
    }

    pub fn empty_with_dtype(context: &WgpuContext, shape: &[usize], dtype: DType) -> Self {
        let size_in_bytes = (shape.iter().product::<usize>() * dtype.size_bytes()) as u64;
        let buffer = context.pool.get(&context.device, size_in_bytes);
        Self {
            data: Arc::new(TensorData {
                buffer,
                size_in_bytes,
                pool: Some(context.pool.clone()),
            }),
            shape: shape.to_vec(),
            dtype,
        }
    }

    pub fn zeros(context: &WgpuContext, shape: &[usize]) -> Self {
        let size = shape.iter().product::<usize>();
        let data = vec![0.0f32; size];
        Self::new(context, &data, shape)
    }

    pub fn from_bytes(context: &WgpuContext, data: &[u8], shape: &[usize]) -> Self {
        Self::from_bytes_with_dtype(context, data, shape, DType::FP32)
    }

    pub fn from_bytes_with_dtype(
        context: &WgpuContext,
        data: &[u8],
        shape: &[usize],
        dtype: DType,
    ) -> Self {
        let size_in_bytes = data.len() as u64;
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Byte Tensor"),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        Self {
            data: Arc::new(TensorData {
                buffer: Arc::new(buffer),
                size_in_bytes,
                pool: None,
            }),
            shape: shape.to_vec(),
            dtype,
        }
    }

    /// Create an FP16 tensor (half precision for memory savings)
    pub fn empty_fp16(context: &WgpuContext, shape: &[usize]) -> Self {
        Self::empty_with_dtype(context, shape, DType::FP16)
    }

    /// Create a new tensor view with different shape but sharing the same underlying data
    pub fn reshaped(&self, new_shape: &[usize]) -> Self {
        // Validation: sizes must match
        let current_elements: usize = self.shape.iter().product();
        let new_elements: usize = new_shape.iter().product();
        assert_eq!(
            current_elements, new_elements,
            "Cannot reshape tensor of size {} to {:?}",
            current_elements, new_shape
        );

        Self {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            dtype: self.dtype,
        }
    }

    pub async fn to_vec(&self, context: &WgpuContext) -> Vec<f32> {
        let size = self.size_in_bytes();
        let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&self.data.buffer, 0, &staging, 0, size);
        context.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |_| tx.send(()).unwrap());
        context.device.poll(wgpu::Maintain::Wait);
        rx.receive().await;

        let data = slice.get_mapped_range();
        let res = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        res
    }
}

pub struct ComputeEngine {
    matmul_pipeline: wgpu::ComputePipeline,
    matmul_bg_layout: wgpu::BindGroupLayout,
    rope_pipeline: wgpu::ComputePipeline,
    rope_bg_layout: wgpu::BindGroupLayout,
    rms_pipeline: wgpu::ComputePipeline,
    rms_bg_layout: wgpu::BindGroupLayout,
    silu_pipeline: wgpu::ComputePipeline,
    silu_bg_layout: wgpu::BindGroupLayout,
    softmax_pipeline: wgpu::ComputePipeline,
    softmax_bg_layout: wgpu::BindGroupLayout,
    add_pipeline: wgpu::ComputePipeline,
    add_bg_layout: wgpu::BindGroupLayout,
    mul_pipeline: wgpu::ComputePipeline,
    mul_bg_layout: wgpu::BindGroupLayout,
    sub_pipeline: wgpu::ComputePipeline,
    sub_bg_layout: wgpu::BindGroupLayout,

    // --- Backward Pipelines ---
    add_bw_pipeline: wgpu::ComputePipeline,
    add_bw_bg_layout: wgpu::BindGroupLayout,
    sub_bw_pipeline: wgpu::ComputePipeline,
    sub_bw_bg_layout: wgpu::BindGroupLayout,
    mul_bw_pipeline: wgpu::ComputePipeline,
    mul_bw_bg_layout: wgpu::BindGroupLayout,
    silu_bw_pipeline: wgpu::ComputePipeline,
    silu_bw_bg_layout: wgpu::BindGroupLayout,
    matmul_at_pipeline: wgpu::ComputePipeline,
    matmul_bt_pipeline: wgpu::ComputePipeline,
    matmul_bk_bg_layout: wgpu::BindGroupLayout,
    sgd_pipeline: wgpu::ComputePipeline,
    sgd_bg_layout: wgpu::BindGroupLayout,
    adamw_pipeline: wgpu::ComputePipeline,
    adamw_bg_layout: wgpu::BindGroupLayout,
    mse_loss_pipeline: wgpu::ComputePipeline,
    mse_loss_bg_layout: wgpu::BindGroupLayout,
    mse_loss_bw_pipeline: wgpu::ComputePipeline,
    mse_loss_bw_bg_layout: wgpu::BindGroupLayout,
    relu_pipeline: wgpu::ComputePipeline,
    relu_bg_layout: wgpu::BindGroupLayout,
    relu_bw_pipeline: wgpu::ComputePipeline,
    relu_bw_bg_layout: wgpu::BindGroupLayout,
    softmax_bw_pipeline: wgpu::ComputePipeline,
    softmax_bw_bg_layout: wgpu::BindGroupLayout,
    dequant_pipeline: wgpu::ComputePipeline,
    dequant_bg_layout: wgpu::BindGroupLayout,
    top_k_pipeline: wgpu::ComputePipeline,
    top_k_bg_layout: wgpu::BindGroupLayout,
    affine_dequant_pipeline: wgpu::ComputePipeline,
    affine_dequant_bg_layout: wgpu::BindGroupLayout,
    conv1d_pipeline: wgpu::ComputePipeline,
    conv1d_bg_layout: wgpu::BindGroupLayout,
    // --- Training Pipelines ---
    ce_loss_pipeline: wgpu::ComputePipeline,
    ce_loss_bg_layout: wgpu::BindGroupLayout,
    ce_loss_bw_pipeline: wgpu::ComputePipeline,
    ce_loss_bw_bg_layout: wgpu::BindGroupLayout,
    grad_norm_pipeline: wgpu::ComputePipeline,
    grad_norm_bg_layout: wgpu::BindGroupLayout,
    grad_clip_pipeline: wgpu::ComputePipeline,
    grad_clip_bg_layout: wgpu::BindGroupLayout,
    // --- Fused Kernels ---
    silu_gate_pipeline: wgpu::ComputePipeline,
    silu_gate_bg_layout: wgpu::BindGroupLayout,
    silu_gate_bw_pipeline: wgpu::ComputePipeline,
    silu_gate_bw_bg_layout: wgpu::BindGroupLayout,
    rms_residual_pipeline: wgpu::ComputePipeline,
    rms_residual_bg_layout: wgpu::BindGroupLayout,
    grad_accum_pipeline: wgpu::ComputePipeline,
    grad_accum_bg_layout: wgpu::BindGroupLayout,
    grad_scale_pipeline: wgpu::ComputePipeline,
    grad_scale_bg_layout: wgpu::BindGroupLayout,
    // --- Flash Attention ---
    flash_attn_pipeline: wgpu::ComputePipeline,
    flash_attn_bg_layout: wgpu::BindGroupLayout,
    flash_attn_bw_pipeline: wgpu::ComputePipeline,
    flash_attn_bw_bg_layout: wgpu::BindGroupLayout,
    // --- Flash Attention V2 (Native GQA) ---
    flash_attn_v2_pipeline: wgpu::ComputePipeline,
    flash_attn_v2_bg_layout: wgpu::BindGroupLayout,
    // --- Flash Attention V2 Backward (Optimized) ---
    flash_attn_v2_bw_preprocess_pipeline: wgpu::ComputePipeline,
    flash_attn_v2_bw_dq_pipeline: wgpu::ComputePipeline,
    flash_attn_v2_bw_dkdv_pipeline: wgpu::ComputePipeline,
    flash_attn_v2_bw_bg_layout: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdamParams {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub t: f32,
    pub _pad: [f32; 2],
}

impl ComputeEngine {
    pub fn new(ctx: &WgpuContext) -> Self {
        // Concatenate kernel files
        let kernels_src = include_str!("kernels.wgsl");
        let flash_v2_src = include_str!("flash_attention_v2.wgsl");
        let flash_v2_bw_src = include_str!("flash_attention_v2_backward_optimized.wgsl");
        let combined_src = format!("{}\n{}\n{}", kernels_src, flash_v2_src, flash_v2_bw_src);

        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Kernels"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(combined_src)),
            });

        let add_bg_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Add Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let mul_bg_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Mul Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let add_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Add Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&add_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "add",
                });

        let mul_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Mul Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&mul_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "mul",
                });

        let sub_bg_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Sub Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let sub_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Sub Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&sub_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "sub",
                });

        // --- Backward Pipelines ---
        let add_bw_bg_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Add BW Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let add_bw_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Add BW Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&add_bw_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "add_backward",
                });

        let sub_bw_bg_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Sub BW Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let sub_bw_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Sub BW Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&sub_bw_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "sub_backward",
                });

        let mul_bw_bg_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Mul BW Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let mul_bw_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Mul BW Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&mul_bw_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "mul_backward",
                });

        let silu_bw_bg_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SiLU BW Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let silu_bw_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("SiLU BW Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&silu_bw_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "silu_backward_real",
                });

        let matmul_bk_bg_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("MatMul BK Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let matmul_at_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MatMul AT Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&matmul_bk_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "matmul_at",
                });
        let matmul_bt_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MatMul BT Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&matmul_bk_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "matmul_bt",
                });

        let sgd_bg_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SGD Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let sgd_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("SGD Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&sgd_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "sgd_update",
                });

        let adamw_bg_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("AdamW Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let adamw_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("AdamW Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&adamw_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "adamw_update",
                });

        let mse_loss_bg_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("MSE Loss Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let mse_loss_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MSE Loss Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&mse_loss_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "mse_loss",
                });

        let mse_loss_bw_bg_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("MSE Loss BW Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let mse_loss_bw_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MSE Loss BW Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&mse_loss_bw_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "mse_loss_backward",
                });

        let matmul_bg_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("MatMul Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let matmul_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MatMul Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&matmul_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "matmul_tiled",
                });

        let rope_bg_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("RoPE Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let rope_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("RoPE Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&rope_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "rope",
                });

        let rms_bg_layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RMS Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let rms_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("RMS Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&rms_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "rms_norm",
                });

        let silu_bg_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SiLU Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let silu_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("SiLU Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&silu_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "silu",
                });

        let softmax_bg_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Softmax Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let softmax_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Softmax Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&softmax_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "softmax",
                });

        let dequant_bg_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Dequant Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let dequant_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Dequant Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&dequant_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "dequantize",
                });

        let top_k_bg_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Top-K Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        let top_k_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Top-K Pipeline"),
                    layout: Some(&ctx.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&top_k_bg_layout],
                            push_constant_ranges: &[],
                        },
                    )),
                    module: &module,
                    entry_point: "top_k",
                });

        Self {
            matmul_pipeline,
            matmul_bg_layout,
            rope_pipeline,
            rope_bg_layout,
            rms_pipeline,
            rms_bg_layout,
            silu_pipeline,
            silu_bg_layout,
            softmax_pipeline,
            softmax_bg_layout,
            add_pipeline,
            add_bg_layout,
            mul_pipeline,
            mul_bg_layout,
            sub_pipeline,
            sub_bg_layout,
            add_bw_pipeline,
            add_bw_bg_layout,
            sub_bw_pipeline,
            sub_bw_bg_layout,
            mul_bw_pipeline,
            mul_bw_bg_layout,
            silu_bw_pipeline,
            silu_bw_bg_layout,
            matmul_at_pipeline,
            matmul_bt_pipeline,
            matmul_bk_bg_layout,
            sgd_pipeline,
            sgd_bg_layout,
            adamw_pipeline,
            adamw_bg_layout,
            mse_loss_pipeline,
            mse_loss_bg_layout,
            mse_loss_bw_pipeline,
            mse_loss_bw_bg_layout,
            dequant_pipeline,
            dequant_bg_layout,
            top_k_pipeline,
            top_k_bg_layout,
            affine_dequant_pipeline: {
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("Affine Dequant Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 2,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 3,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                let pipeline =
                    ctx.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("Affine Dequant Pipeline"),
                            layout: Some(&ctx.device.create_pipeline_layout(
                                &wgpu::PipelineLayoutDescriptor {
                                    label: None,
                                    bind_group_layouts: &[&layout],
                                    push_constant_ranges: &[],
                                },
                            )),
                            module: &module,
                            entry_point: "affine_dequantize",
                        });
                (pipeline, layout)
            }
            .0,
            affine_dequant_bg_layout: {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Affine Dequant Layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            },
            conv1d_pipeline: {
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("Conv1D Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 2,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 3,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 4,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                let pipeline =
                    ctx.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("Conv1D Pipeline"),
                            layout: Some(&ctx.device.create_pipeline_layout(
                                &wgpu::PipelineLayoutDescriptor {
                                    label: None,
                                    bind_group_layouts: &[&layout],
                                    push_constant_ranges: &[],
                                },
                            )),
                            module: &module,
                            entry_point: "conv1d",
                        });
                (pipeline, layout)
            }
            .0,
            conv1d_bg_layout: {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Conv1D Layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            },
            relu_pipeline: {
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("Relu Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                let pipeline =
                    ctx.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some("Relu Pipeline"),
                            layout: Some(&ctx.device.create_pipeline_layout(
                                &wgpu::PipelineLayoutDescriptor {
                                    label: None,
                                    bind_group_layouts: &[&layout],
                                    push_constant_ranges: &[],
                                },
                            )),
                            module: &module,
                            entry_point: "relu",
                        });
                (pipeline, layout)
            }
            .0,
            relu_bg_layout: {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Relu Layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            },
            relu_bw_pipeline: {
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("Relu BW Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 3,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 4,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 5,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Relu BW Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "relu_backward",
                    })
            },
            relu_bw_bg_layout: ctx.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("Relu BW Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                },
            ),
            softmax_bw_pipeline: {
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("Softmax BW Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 3,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 4,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 5,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 2,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Softmax BW Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "softmax_backward",
                    })
            },
            softmax_bw_bg_layout: ctx.device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    label: Some("Softmax BW Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                },
            ),
            // --- Training: Cross-Entropy Loss ---
            ce_loss_bg_layout: {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("CE Loss Layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            },
            ce_loss_pipeline: {
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("CE Loss Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 2,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("CE Loss Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "cross_entropy_loss",
                    })
            },
            ce_loss_bw_bg_layout: {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("CE Loss BW Layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            },
            ce_loss_bw_pipeline: {
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("CE Loss BW Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 2,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("CE Loss BW Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "cross_entropy_backward",
                    })
            },
            // --- Training: Gradient Norm/Clipping ---
            grad_norm_bg_layout: {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Grad Norm Layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            },
            grad_norm_pipeline: {
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("Grad Norm Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Grad Norm Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "gradient_norm_squared",
                    })
            },
            grad_clip_bg_layout: {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Grad Clip Layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            },
            grad_clip_pipeline: {
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("Grad Clip Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Grad Clip Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "clip_grad_by_norm",
                    })
            },
            // --- Fused Kernels ---
            silu_gate_bg_layout: {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("SiLU Gate Layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            },
            silu_gate_pipeline: {
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("SiLU Gate Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 2,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("SiLU Gate Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "silu_gate_fused",
                    })
            },
            silu_gate_bw_bg_layout: {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("SiLU Gate BW Layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 5,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            },
            silu_gate_bw_pipeline: {
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("SiLU Gate BW Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 3,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 4,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 5,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("SiLU Gate BW Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "silu_gate_backward",
                    })
            },
            rms_residual_bg_layout: {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("RMS Residual Layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            },
            rms_residual_pipeline: {
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("RMS Residual Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 2,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 3,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 4,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("RMS Residual Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "rms_norm_residual",
                    })
            },
            grad_accum_bg_layout: {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Grad Accum Layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            },
            grad_accum_pipeline: {
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("Grad Accum Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Grad Accum Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "gradient_accumulate",
                    })
            },
            grad_scale_bg_layout: {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Grad Scale Layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            },
            grad_scale_pipeline: {
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("Grad Scale Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Grad Scale Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "gradient_scale",
                    })
            },
            // --- Flash Attention Pipeline ---
            flash_attn_bg_layout: {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Flash Attn Layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            },
            flash_attn_pipeline: {
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("Flash Attn Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 2,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 3,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 4,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Flash Attn Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "flash_attention_simple",
                    })
            },
            // --- Flash Attention V2 Pipeline (Native GQA) ---
            flash_attn_v2_bg_layout: {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Flash Attn V2 Layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 5,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            },
            flash_attn_v2_pipeline: {
                // Reuse layout definition logic as we can't easily reference the struct field here
                // We create a temp layout for pipeline creation
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("Flash Attn V2 Layout Temp"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 2,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 3,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 4,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });

                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Flash Attn V2 Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "flash_attention_v2_metal",
                    })
            },

            // --- Flash Attention V2 Backward Pipeline (Naive) ---
            flash_attn_v2_bw_bg_layout: {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Flash Attn V2 BW Layout"),
                        entries: &[
                            // 0-2: Unused
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // 3-5: Q, K, V
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 5,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // 6-7: Out, GradOut
                            wgpu::BindGroupLayoutEntry {
                                binding: 6,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 7,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // 8-10: dQ, dK, dV
                            wgpu::BindGroupLayoutEntry {
                                binding: 8,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 9,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 10,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // 11: Params
                            wgpu::BindGroupLayoutEntry {
                                binding: 11,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 11,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // 12: Mask
                            wgpu::BindGroupLayoutEntry {
                                binding: 12,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // 13: Mask Params
                            wgpu::BindGroupLayoutEntry {
                                binding: 13,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // 14: Delta
                            wgpu::BindGroupLayoutEntry {
                                binding: 14,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // 15: LSE
                            wgpu::BindGroupLayoutEntry {
                                binding: 15,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            },

            flash_attn_v2_bw_preprocess_pipeline: {
                let layout_desc = wgpu::BindGroupLayoutDescriptor {
                    label: Some("FA V2 BW Preprocess Layout"),
                    entries: &[
                        // Minimal bindings for Preprocess (or full, we can reuse full layout)
                        // It must match the shader group(0) definition.
                        // Our shader binds EVERYTHING. So we must use the Full layout.
                        // But wait, the pipeline creation takes a Layout object, not a descriptor.
                        // And we just created `flash_attn_v2_bw_bg_layout`. We can reuse it!
                        // But we can't access `flash_attn_v2_bw_bg_layout` variable easily because it's inside struct init.
                        // So we recreate the layout OR move layout creation up?
                        // The pattern used here creates layout effectively.
                        // Let's copy-paste the layout entries again for temporary layout creation used by pipeline.
                        // Same pattern as `flash_attn_v2_bw_pipeline`.
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 8,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 9,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 10,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 11,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 12,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 13,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 14,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 15,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                };
                let layout = ctx.device.create_bind_group_layout(&layout_desc);
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Flash Attn V2 BW Preprocess Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "flash_bwd_preprocess",
                    })
            },

            flash_attn_v2_bw_dq_pipeline: {
                // Reuse logic (copy-paste implies duplicated layout creation but fine for now)
                // Ideally refactor this verbose construction but in Rust struct init it's hard.
                let layout_desc = wgpu::BindGroupLayoutDescriptor {
                    label: Some("FA V2 BW DQ Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 8,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 9,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 10,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 11,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 12,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 13,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 14,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 15,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                };
                let layout = ctx.device.create_bind_group_layout(&layout_desc);
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Flash Attn V2 BW DQ Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "flash_bwd_dq",
                    })
            },

            flash_attn_v2_bw_dkdv_pipeline: {
                let layout_desc = wgpu::BindGroupLayoutDescriptor {
                    label: Some("FA V2 BW DKDV Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 8,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 9,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 10,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 11,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 12,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 13,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 14,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 15,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                };
                let layout = ctx.device.create_bind_group_layout(&layout_desc);
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Flash Attn V2 BW DKDV Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "flash_bwd_dk_dv",
                    })
            },
            flash_attn_bw_bg_layout: {
                ctx.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Flash Attn BW Layout"),
                        entries: &[
                            // Q, K, V, Out, GradOut (read)
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // GradQ, GradK, GradV (write)
                            wgpu::BindGroupLayoutEntry {
                                binding: 5,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 6,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 7,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 8,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            },
            flash_attn_bw_pipeline: {
                let layout =
                    ctx.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("Flash Attn BW Layout"),
                            entries: &[
                                wgpu::BindGroupLayoutEntry {
                                    binding: 0,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 1,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 2,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 3,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 4,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 5,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 6,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 7,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 8,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                        });
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Flash Attn BW Pipeline"),
                        layout: Some(&ctx.device.create_pipeline_layout(
                            &wgpu::PipelineLayoutDescriptor {
                                label: None,
                                bind_group_layouts: &[&layout],
                                push_constant_ranges: &[],
                            },
                        )),
                        module: &module,
                        entry_point: "flash_attention_backward",
                    })
            },
        }
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

    pub fn add(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.add_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer().as_entire_binding(),
                },
            ],
        });
        self.dispatch_binary(
            ctx,
            &self.add_pipeline,
            &bg,
            a.shape.iter().product::<usize>() as u32,
        );
    }

    pub fn mul(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.mul_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer().as_entire_binding(),
                },
            ],
        });
        self.dispatch_binary(
            ctx,
            &self.mul_pipeline,
            &bg,
            a.shape.iter().product::<usize>() as u32,
        );
    }

    pub fn sub(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.sub_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer().as_entire_binding(),
                },
            ],
        });
        self.dispatch_binary(
            ctx,
            &self.sub_pipeline,
            &bg,
            a.shape.iter().product::<usize>() as u32,
        );
    }

    fn dispatch_binary(
        &self,
        ctx: &WgpuContext,
        pipeline: &wgpu::ComputePipeline,
        bg: &wgpu::BindGroup,
        size: u32,
    ) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups((size + 63) / 64, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn add_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        grad_a: &UnifiedTensor,
        grad_b: &UnifiedTensor,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.add_bw_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grad_a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grad_b.buffer().as_entire_binding(),
                },
            ],
        });
        self.dispatch_binary(
            ctx,
            &self.add_bw_pipeline,
            &bg,
            grad_out.shape.iter().product::<usize>() as u32,
        );
    }

    pub fn sub_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        grad_a: &UnifiedTensor,
        grad_b: &UnifiedTensor,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.sub_bw_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grad_a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grad_b.buffer().as_entire_binding(),
                },
            ],
        });
        self.dispatch_binary(
            ctx,
            &self.sub_bw_pipeline,
            &bg,
            grad_out.shape.iter().product::<usize>() as u32,
        );
    }

    pub fn mul_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        grad_a: &UnifiedTensor,
        grad_b: &UnifiedTensor,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.mul_bw_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: grad_a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: grad_b.buffer().as_entire_binding(),
                },
            ],
        });
        self.dispatch_binary(
            ctx,
            &self.mul_bw_pipeline,
            &bg,
            grad_out.shape.iter().product::<usize>() as u32,
        );
    }

    pub fn silu_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        input: &UnifiedTensor,
        grad_in: &UnifiedTensor,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.silu_bw_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grad_in.buffer().as_entire_binding(),
                },
            ],
        });
        self.dispatch_binary(
            ctx,
            &self.silu_bw_pipeline,
            &bg,
            grad_out.shape.iter().product::<usize>() as u32,
        );
    }

    pub fn matmul_bt(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let m = a.shape[0] as u32;
        let k = a.shape[1] as u32;
        let n = b.shape[0] as u32;
        let params = [m, k, n, 0];
        let uniform_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MatMul BT Uniform"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.matmul_bk_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.matmul_bt_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n + 15) / 16, (m + 15) / 16, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn matmul_at(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let m = a.shape[1] as u32;
        let k = a.shape[0] as u32;
        let n = b.shape[1] as u32;
        let params = [m, k, n, 0];
        let uniform_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MatMul AT Uniform"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.matmul_bk_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.matmul_at_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n + 15) / 16, (m + 15) / 16, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn matmul(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let m = a.shape[0] as u32;
        let k = a.shape[1] as u32;
        let n = b.shape[1] as u32;
        let params = [m, k, n, 0];
        let uniform_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MatMul Uniform"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.matmul_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.matmul_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n + 15) / 16, (m + 15) / 16, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn rope(&self, ctx: &WgpuContext, tensor: &UnifiedTensor, head_dim: usize, pos: usize) {
        let seq_len = if tensor.shape.len() >= 3 {
            tensor.shape[1]
        } else {
            1
        };
        let params = [head_dim as u32, seq_len as u32, pos as u32, 0];
        let uniform_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RoPE Uniform"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.rope_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tensor.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.rope_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let total_elements = tensor.shape.iter().product::<usize>();
            let pairs = total_elements / 2;
            pass.dispatch_workgroups((pairs as u32 + 63) / 64, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn rms_norm(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let size = input.shape.last().unwrap_or(&1);
        let rows = input.shape.iter().product::<usize>() / *size;
        let params = [*size as u32, rows as u32];
        let uniform_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RMS Uniform"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.rms_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.rms_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(rows as u32, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn silu(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.silu_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.buffer().as_entire_binding(),
                },
            ],
        });
        self.dispatch_binary(
            ctx,
            &self.silu_pipeline,
            &bg,
            input.shape.iter().product::<usize>() as u32,
        );
    }

    pub fn softmax(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        let size = input.shape.last().unwrap();
        let rows = input.shape.iter().product::<usize>() / *size;
        let params = [*size as u32, rows as u32];
        let uniform_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Softmax Uniform"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.softmax_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.softmax_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(rows as u32, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn sgd_update(
        &self,
        ctx: &WgpuContext,
        weight: &UnifiedTensor,
        grad: &UnifiedTensor,
        lr: f32,
    ) {
        let lr_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LR Uniform"),
                contents: bytemuck::cast_slice(&[lr]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.sgd_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lr_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.sgd_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let total_elements = weight.shape.iter().product::<usize>();
            pass.dispatch_workgroups((total_elements as u32 + 63) / 64, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn adamw_update(
        &self,
        ctx: &WgpuContext,
        weight: &UnifiedTensor,
        grad: &UnifiedTensor,
        m: &UnifiedTensor,
        v: &UnifiedTensor,
        params: &AdamParams,
    ) {
        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Adam Params"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        // We use dummy placeholders for empty bindings to satisfy layout if needed,
        // though here we have a dedicated layout.
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.adamw_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                }, // LR placeholder or shared
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: m.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: v.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.adamw_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let total_elements = weight.shape.iter().product::<usize>();
            pass.dispatch_workgroups((total_elements as u32 + 63) / 64, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn mse_loss(
        &self,
        ctx: &WgpuContext,
        pred: &UnifiedTensor,
        target: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.mse_loss_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pred.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer().as_entire_binding(),
                },
            ],
        });
        self.dispatch_binary(
            ctx,
            &self.mse_loss_pipeline,
            &bg,
            pred.shape.iter().product::<usize>() as u32,
        );
    }

    pub fn mse_loss_backward(
        &self,
        ctx: &WgpuContext,
        pred: &UnifiedTensor,
        target: &UnifiedTensor,
        grad_in: &UnifiedTensor,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.mse_loss_bw_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pred.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: target.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad_in.buffer().as_entire_binding(),
                },
            ],
        });
        self.dispatch_binary(
            ctx,
            &self.mse_loss_bw_pipeline,
            &bg,
            pred.shape.iter().product::<usize>() as u32,
        );
    }

    pub fn relu(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.relu_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.buffer().as_entire_binding(),
                },
            ],
        });
        self.dispatch_binary(
            ctx,
            &self.relu_pipeline,
            &bg,
            input.shape.iter().product::<usize>() as u32,
        );
    }

    pub fn relu_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        input: &UnifiedTensor,
        grad_in: &UnifiedTensor,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.relu_bw_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grad_in.buffer().as_entire_binding(),
                },
            ],
        });
        self.dispatch_binary(
            ctx,
            &self.relu_bw_pipeline,
            &bg,
            grad_out.shape.iter().product::<usize>() as u32,
        );
    }

    pub fn softmax_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        final_out: &UnifiedTensor,
        grad_in: &UnifiedTensor,
    ) {
        let size_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Softmax size"),
                contents: bytemuck::cast_slice(&[
                    *grad_out.shape.last().unwrap_or(&0) as u32,
                    (grad_out.shape.iter().product::<usize>()
                        / *grad_out.shape.last().unwrap_or(&1)) as u32,
                ]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.softmax_bw_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: final_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grad_in.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: size_buf.as_entire_binding(),
                },
            ],
        });
        self.dispatch_binary(
            ctx,
            &self.softmax_bw_pipeline,
            &bg,
            (grad_out.shape.iter().product::<usize>() / *grad_out.shape.last().unwrap_or(&1))
                as u32,
        );
    }

    pub fn dequantize(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.dequant_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.buffer().as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.dequant_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let total_elements = input.shape.iter().product::<usize>();
            let u32_elements = (total_elements + 3) / 4;
            pass.dispatch_workgroups((u32_elements as u32 + 63) / 64, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn top_k(&self, ctx: &WgpuContext, input: &UnifiedTensor, out_idx: &UnifiedTensor, k: u32) {
        let uniform_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Top-K Uniform"),
                contents: bytemuck::cast_slice(&[k]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.top_k_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_idx.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.top_k_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1); // Single-threaded simple implementation
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn affine_dequantize(
        &self,
        ctx: &WgpuContext,
        weight: &wgpu::Buffer,
        scales: &wgpu::Buffer,
        biases: &wgpu::Buffer,
        out: &UnifiedTensor,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.affine_dequant_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weight.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scales.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: biases.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.buffer().as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.affine_dequant_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let total_elements = out.shape.iter().product::<usize>();
            let workgroups = (total_elements as u32 + 63) / 64;
            // Metal limit is 65535 per dimension, use 2D dispatch for large tensors
            let max_wg = 65535u32;
            if workgroups <= max_wg {
                pass.dispatch_workgroups(workgroups, 1, 1);
            } else {
                // Split into 2D grid
                let wg_x = max_wg;
                let wg_y = (workgroups + max_wg - 1) / max_wg;
                pass.dispatch_workgroups(wg_x, wg_y.min(max_wg), 1);
            }
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    pub fn conv1d(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        bias: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let dim = weight.shape[0];
        let seq_len = input.shape[input.shape.len() - 1];

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ConvParams {
            dim: u32,
            seq_len: u32,
        }

        let params = ConvParams {
            dim: dim as u32,
            seq_len: seq_len as u32,
        };
        let param_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Conv Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.conv1d_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bias.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.conv1d_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let total = dim * seq_len;
            pass.dispatch_workgroups((total as u32 + 63) / 64, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    // --- Training Methods ---

    /// Cross-entropy loss forward: computes -log(softmax(logits)[target])
    pub fn cross_entropy_loss(
        &self,
        ctx: &WgpuContext,
        logits: &UnifiedTensor,
        target: u32,
        loss_out: &UnifiedTensor,
    ) {
        let vocab_size = logits.shape.iter().product::<usize>() as u32;
        let params: [u32; 4] = [target, vocab_size, 0, 0];
        let uniform_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CE Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.ce_loss_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: logits.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: loss_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.ce_loss_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    /// Cross-entropy loss backward: gradient = softmax(logits) - one_hot(target)
    pub fn cross_entropy_backward(
        &self,
        ctx: &WgpuContext,
        logits: &UnifiedTensor,
        target: u32,
        grad_out: &UnifiedTensor,
    ) {
        let vocab_size = logits.shape.iter().product::<usize>() as u32;
        let params: [u32; 4] = [target, vocab_size, 0, 0];
        let uniform_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CE BW Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.ce_loss_bw_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: logits.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.ce_loss_bw_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((vocab_size + 63) / 64, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    /// Compute squared elements (for gradient norm computation)
    pub fn gradient_norm_squared(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.grad_norm_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.buffer().as_entire_binding(),
                },
            ],
        });
        let size = input.shape.iter().product::<usize>() as u32;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.grad_norm_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((size + 63) / 64, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    /// Clip gradients by scaling: grad *= min(1.0, max_norm / actual_norm)
    pub fn clip_grad_by_norm(&self, ctx: &WgpuContext, grad: &UnifiedTensor, scale: f32) {
        let uniform_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Clip Scale"),
                contents: bytemuck::cast_slice(&[scale]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.grad_clip_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let size = grad.shape.iter().product::<usize>() as u32;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.grad_clip_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((size + 63) / 64, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    // --- Fused Kernels for Training Efficiency ---

    /// Fused SiLU-Gate: out = SiLU(up) * gate
    /// Reduces memory bandwidth by 2x compared to separate ops
    pub fn silu_gate_fused(
        &self,
        ctx: &WgpuContext,
        up: &UnifiedTensor,
        gate: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.silu_gate_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: up.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gate.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer().as_entire_binding(),
                },
            ],
        });
        let size = up.shape.iter().product::<usize>() as u32;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.silu_gate_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((size + 63) / 64, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    /// Fused SiLU-Gate backward: computes gradients for up and gate
    pub fn silu_gate_backward(
        &self,
        ctx: &WgpuContext,
        up: &UnifiedTensor,
        gate: &UnifiedTensor,
        grad_out: &UnifiedTensor,
        grad_up: &UnifiedTensor,
        grad_gate: &UnifiedTensor,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.silu_gate_bw_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: up.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gate.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grad_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grad_up.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grad_gate.buffer().as_entire_binding(),
                },
            ],
        });
        let size = up.shape.iter().product::<usize>() as u32;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.silu_gate_bw_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((size + 63) / 64, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
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
        let uniform_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RMS Size"),
                contents: bytemuck::cast_slice(&[size]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.rms_residual_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: residual.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.rms_residual_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    /// Gradient accumulation: accum_grad += new_grad
    /// Essential for micro-batching on memory-constrained devices
    pub fn gradient_accumulate(
        &self,
        ctx: &WgpuContext,
        accum_grad: &UnifiedTensor,
        new_grad: &UnifiedTensor,
    ) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.grad_accum_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: accum_grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: new_grad.buffer().as_entire_binding(),
                },
            ],
        });
        let size = accum_grad.shape.iter().product::<usize>() as u32;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.grad_accum_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((size + 63) / 64, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    /// Gradient scaling: grad *= scale_factor
    /// Used to average gradients after accumulation
    pub fn gradient_scale(&self, ctx: &WgpuContext, grad: &UnifiedTensor, scale: f32) {
        let uniform_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Scale Factor"),
                contents: bytemuck::cast_slice(&[scale]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.grad_scale_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        let size = grad.shape.iter().product::<usize>() as u32;
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.grad_scale_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((size + 63) / 64, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    /// Flash Attention forward pass with online softmax
    /// Memory efficient: O(N) instead of O(N^2) for standard attention
    ///
    /// # Arguments
    /// * `q` - Query tensor [seq_len, head_dim]
    /// * `k` - Key tensor [seq_len, head_dim]
    /// * `v` - Value tensor [seq_len, head_dim]
    /// * `out` - Output tensor [seq_len, head_dim]
    /// * `causal` - Whether to apply causal masking
    pub fn flash_attention(
        &self,
        ctx: &WgpuContext,
        q: &UnifiedTensor,
        k: &UnifiedTensor,
        v: &UnifiedTensor,
        out: &UnifiedTensor,
        causal: bool,
    ) {
        let seq_len = q.shape[0] as u32;
        let head_dim = q.shape[1] as u32;
        let causal_flag = if causal { 1u32 } else { 0u32 };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Flash Attn Params"),
                contents: bytemuck::cast_slice(&[seq_len, head_dim, causal_flag, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.flash_attn_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.flash_attn_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            // Dispatch one workgroup per query position
            pass.dispatch_workgroups((seq_len + 63) / 64, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    /// Flash Attention V2 forward pass with Native GQA Support
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, seq_q, num_heads, head_dim]
    /// * `k` - Key tensor [batch, seq_k, num_kv_heads, head_dim]
    /// * `v` - Value tensor [batch, seq_k, num_kv_heads, head_dim]
    /// * `out` - Output tensor [batch, seq_q, num_heads, head_dim]
    /// * `causal` - Whether to apply causal masking
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
        // Assume 4D shapes: [batch, seq, heads, dim]
        let batch_size = q.shape[0] as u32;
        let seq_q = q.shape[1] as u32;
        let num_heads = q.shape[2] as u32;
        let head_dim = q.shape[3] as u32;

        // K/V shapes
        let seq_k = k.shape[1] as u32;
        let num_kv_heads = k.shape[2] as u32;

        let causal_flag = if causal { 1u32 } else { 0u32 };
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Mask Params
        let mut has_mask = 0u32;
        let mut mask_stride_b = 0u32;
        let mut mask_stride_h = 0u32;
        let mut mask_stride_q = 0u32;
        let mut mask_stride_k = 0u32;

        let dummy_mask;
        let mask_buf = if let Some(m) = mask {
            has_mask = 1;
            // Calculate strides for broadcasting
            // Assume mask is 4D: [batch, heads, seq_q, seq_k]
            // If dimensions are 1, stride is 0 (broadcast)
            let shape = &m.shape;
            if shape.len() == 4 {
                let _s3 = 1;
                let s2 = if shape[3] > 1 { shape[3] as u32 } else { 0 }; // K dim
                let s1 = if shape[2] > 1 {
                    shape[2] as u32 * s2
                } else {
                    0
                }; // Q dim
                let _s0 = if shape[1] > 1 {
                    shape[1] as u32 * s1
                } else {
                    0
                }; // H dim

                // Actually, strides logic depends on memory layout (contiguous)
                // Stride[i] is how much index increases when dim[i] increases by 1
                // If shape[i] == 1, accessing index x is always 0, so stride effectively 0 for that dim input

                // But we are mapping (b, h, q, k) to mask index.
                // if mask_dim[i] == 1: mask_idx += 0
                // else: mask_idx += input_idx * stride[i]
                // stride[i] is the products of lower dimensions

                let dim_k = shape[3] as u32;
                let dim_q = shape[2] as u32;
                let dim_h = shape[1] as u32;
                let dim_b = shape[0] as u32;

                // Strides of the MASK tensor
                let real_stride_k = 1u32;
                let real_stride_q = dim_k;
                let real_stride_h = dim_k * dim_q;
                let real_stride_b = dim_k * dim_q * dim_h;

                mask_stride_k = if dim_k > 1 { real_stride_k } else { 0 };
                mask_stride_q = if dim_q > 1 { real_stride_q } else { 0 };
                mask_stride_h = if dim_h > 1 { real_stride_h } else { 0 };
                mask_stride_b = if dim_b > 1 { real_stride_b } else { 0 };
            }

            m.buffer()
        } else {
            // Create dummy buffer
            dummy_mask = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Dummy Mask"),
                    contents: &[0u8; 4],
                    usage: wgpu::BufferUsages::STORAGE,
                });
            &dummy_mask
        };

        // flash_v2_params struct alignment:
        // vec4: batch_size, num_heads, seq_q, seq_k
        // vec4: head_dim, causal, scale, num_kv_heads
        // vec4: dropout_prob, seed_val, has_mask, pad
        // vec4: mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k
        let params: [u32; 16] = [
            batch_size,
            num_heads,
            seq_q,
            seq_k,
            head_dim,
            causal_flag,
            scale.to_bits(), // bit-cast f32 to u32
            num_kv_heads,
            dropout_prob.to_bits(),
            (seed & 0xFFFFFFFF) as u32,
            has_mask,
            0,
            mask_stride_b,
            mask_stride_h,
            mask_stride_q,
            mask_stride_k,
        ];

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Flash Attn V2 Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.flash_attn_v2_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: mask_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.flash_attn_v2_pipeline);
            pass.set_bind_group(0, &bg, &[]);

            // Dispatch size: (1, seq_q / TILE_Q, batch * num_heads)
            // TILE_Q = 32 in metal kernel
            let tile_q = 32u32;
            let x = 1u32;
            let y = (seq_q + tile_q - 1) / tile_q;
            let z = batch_size * num_heads;

            pass.dispatch_workgroups(x, y, z);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    /// Flash Attention backward pass - computes dQ, dK, dV gradients
    ///
    /// # Arguments
    /// * `q`, `k`, `v` - Original inputs
    /// * `out` - Forward output
    /// * `grad_out` - Gradient of loss w.r.t. output
    /// * `grad_q`, `grad_k`, `grad_v` - Output gradient tensors
    /// * `causal` - Whether causal mask was used
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
        let seq_len = q.shape[0] as u32;
        let head_dim = q.shape[1] as u32;
        let causal_flag = if causal { 1u32 } else { 0u32 };

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Flash Attn BW Params"),
                contents: bytemuck::cast_slice(&[seq_len, head_dim, causal_flag, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.flash_attn_bw_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grad_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grad_q.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: grad_k.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: grad_v.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.flash_attn_bw_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((seq_len + 63) / 64, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    /// Flash Attention V2 Backward (Optimized Tiled 2-Pass)
    /// Computes dQ, dK, dV without atomic adds.
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
        // Note: We are using the V2 kernel which is 4D-aware
        // Assuming q, k, v are [batch, seq, heads, dim]

        let batch_size = if q.shape.len() >= 4 {
            q.shape[0] as u32
        } else {
            1
        };
        let seq_q = if q.shape.len() >= 4 {
            q.shape[1] as u32
        } else {
            q.shape[0] as u32
        };
        let num_heads = if q.shape.len() >= 4 {
            q.shape[2] as u32
        } else {
            1
        };
        let head_dim = q.shape.last().copied().unwrap_or(64) as u32;

        let seq_k = if k.shape.len() >= 4 {
            k.shape[1] as u32
        } else {
            k.shape[0] as u32
        };
        let num_kv_heads = if k.shape.len() >= 4 {
            k.shape[2] as u32
        } else {
            1
        };

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Params struct
        let params: [u32; 10] = [
            batch_size,
            num_heads,
            seq_q,
            seq_k,
            head_dim,
            num_kv_heads,
            scale.to_bits(),
            0,
            0, // No dropout/seed passed in this API yet
            if causal { 1 } else { 0 },
        ];

        // Mask buffer (dummy for now)
        let dummy_mask = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dummy Mask"),
                contents: &[0u8; 4],
                usage: wgpu::BufferUsages::STORAGE,
            });
        let mask_p: [u32; 5] = [0, 0, 0, 0, 0];

        let params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FA V2 BW Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let mask_p_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FA V2 BW Mask Params"),
                contents: bytemuck::cast_slice(&mask_p),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Intermediate Buffers: Delta and LSE
        // Shape: [Batch, Heads, Seq_Q]
        let inter_size = (batch_size * num_heads * seq_q) as u64 * 4;
        let delta_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Delta Buffer"),
            size: inter_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let lse_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LSE Buffer"),
            size: inter_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.flash_attn_v2_bw_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_q.buffer().as_entire_binding(),
                }, // Ignored by shader
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_k.buffer().as_entire_binding(),
                }, // Ignored
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad_v.buffer().as_entire_binding(),
                }, // Ignored
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: q.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: k.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: v.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: grad_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: grad_q.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: grad_k.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: grad_v.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: dummy_mask.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: mask_p_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: delta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: lse_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());

            // 1. Preprocess: Compute Delta and LSE
            pass.set_pipeline(&self.flash_attn_v2_bw_preprocess_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((seq_q + 63) / 64, num_heads, batch_size);

            // 2. Compute dQ
            pass.set_pipeline(&self.flash_attn_v2_bw_dq_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(seq_q, num_heads, batch_size);

            // 3. Compute dK, dV
            pass.set_pipeline(&self.flash_attn_v2_bw_dkdv_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(seq_k, num_kv_heads, batch_size);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }
}
