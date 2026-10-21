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
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Rusty Device"),
                    required_features: wgpu::Features::empty(),
                    // Use what the hardware supports rather than the WebGPU
                    // defaults (e.g. 8 storage buffers per stage, 128 MiB
                    // bindings), which are too small for model weights.
                    required_limits: adapter.limits(),
                },
                None,
            )
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

/// The full WGSL source compiled into the engine's shader module.
fn shader_source() -> String {
    let kernels_src = include_str!("kernels.wgsl");
    let flash_v2_src = include_str!("flash_attention_v2.wgsl");
    let flash_v2_bw_src = include_str!("flash_attention_v2_backward_optimized.wgsl");
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

    pub fn add(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.binary(ctx, &self.add, a, b, out);
    }

    pub fn mul(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.binary(ctx, &self.mul, a, b, out);
    }

    pub fn sub(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.binary(ctx, &self.sub, a, b, out);
    }

    fn binary(
        &self,
        ctx: &WgpuContext,
        kernel: &Kernel,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            kernel,
            &[(0, a.buffer()), (1, b.buffer()), (2, out.buffer())],
            (groups64(a.shape.iter().product()), 1, 1),
        );
    }

    pub fn add_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        grad_a: &UnifiedTensor,
        grad_b: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.add_bw,
            &[
                (3, grad_out.buffer()),
                (4, grad_a.buffer()),
                (5, grad_b.buffer()),
            ],
            (groups64(grad_out.shape.iter().product()), 1, 1),
        );
    }

    pub fn sub_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        grad_a: &UnifiedTensor,
        grad_b: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.sub_bw,
            &[
                (3, grad_out.buffer()),
                (4, grad_a.buffer()),
                (5, grad_b.buffer()),
            ],
            (groups64(grad_out.shape.iter().product()), 1, 1),
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
        self.run(
            ctx,
            &self.mul_bw,
            &[
                (3, grad_out.buffer()),
                (4, a.buffer()),
                (5, b.buffer()),
                (6, grad_a.buffer()),
                (7, grad_b.buffer()),
            ],
            (groups64(grad_out.shape.iter().product()), 1, 1),
        );
    }

    pub fn silu_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        input: &UnifiedTensor,
        grad_in: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.silu_bw,
            &[
                (3, grad_out.buffer()),
                (4, input.buffer()),
                (5, grad_in.buffer()),
            ],
            (groups64(grad_out.shape.iter().product()), 1, 1),
        );
    }

    fn matmul_dispatch(
        &self,
        ctx: &WgpuContext,
        kernel: &Kernel,
        (m, k, n): (u32, u32, u32),
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let params = uniform(ctx, bytemuck::cast_slice(&[m, k, n, 0u32]));
        self.run(
            ctx,
            kernel,
            &[
                (0, a.buffer()),
                (1, b.buffer()),
                (2, out.buffer()),
                (3, &params),
            ],
            (n.div_ceil(16), m.div_ceil(16), 1),
        );
    }

    pub fn matmul_bt(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let dims = (a.shape[0] as u32, a.shape[1] as u32, b.shape[0] as u32);
        self.matmul_dispatch(ctx, &self.matmul_bt, dims, a, b, out);
    }

    pub fn matmul_at(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let dims = (a.shape[1] as u32, a.shape[0] as u32, b.shape[1] as u32);
        self.matmul_dispatch(ctx, &self.matmul_at, dims, a, b, out);
    }

    pub fn matmul(
        &self,
        ctx: &WgpuContext,
        a: &UnifiedTensor,
        b: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let dims = (a.shape[0] as u32, a.shape[1] as u32, b.shape[1] as u32);
        self.matmul_dispatch(ctx, &self.matmul, dims, a, b, out);
    }

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
            (groups64(pairs), 1, 1),
        );
    }

    pub fn rms_norm(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        weight: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        let size = *input.shape.last().unwrap_or(&1);
        let rows = input.shape.iter().product::<usize>() / size;
        let params = uniform(ctx, bytemuck::cast_slice(&[size as u32, rows as u32]));
        self.run(
            ctx,
            &self.rms_norm,
            &[
                (0, input.buffer()),
                (1, weight.buffer()),
                (2, out.buffer()),
                (3, &params),
            ],
            (rows as u32, 1, 1),
        );
    }

    pub fn silu(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        self.run(
            ctx,
            &self.silu,
            &[(0, input.buffer()), (1, out.buffer())],
            (groups64(input.shape.iter().product()), 1, 1),
        );
    }

    pub fn softmax(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        let size = *input.shape.last().unwrap_or(&1);
        let rows = input.shape.iter().product::<usize>() / size;
        let params = uniform(ctx, bytemuck::cast_slice(&[size as u32, rows as u32]));
        self.run(
            ctx,
            &self.softmax,
            &[(0, input.buffer()), (1, out.buffer()), (2, &params)],
            (rows as u32, 1, 1),
        );
    }

    pub fn sgd_update(
        &self,
        ctx: &WgpuContext,
        weight: &UnifiedTensor,
        grad: &UnifiedTensor,
        lr: f32,
    ) {
        let lr_buf = uniform(ctx, bytemuck::cast_slice(&[lr]));
        self.run(
            ctx,
            &self.sgd,
            &[(0, weight.buffer()), (1, grad.buffer()), (2, &lr_buf)],
            (groups64(weight.shape.iter().product()), 1, 1),
        );
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
        let params_buf = uniform(ctx, bytemuck::bytes_of(params));
        self.run(
            ctx,
            &self.adamw,
            &[
                (0, weight.buffer()),
                (1, grad.buffer()),
                (3, m.buffer()),
                (4, v.buffer()),
                (5, &params_buf),
            ],
            (groups64(weight.shape.iter().product()), 1, 1),
        );
    }

    pub fn mse_loss(
        &self,
        ctx: &WgpuContext,
        pred: &UnifiedTensor,
        target: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.binary(ctx, &self.mse_loss, pred, target, out);
    }

    pub fn mse_loss_backward(
        &self,
        ctx: &WgpuContext,
        pred: &UnifiedTensor,
        target: &UnifiedTensor,
        grad_in: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.mse_loss_bw,
            &[
                (0, pred.buffer()),
                (1, target.buffer()),
                (3, grad_in.buffer()),
            ],
            (groups64(pred.shape.iter().product()), 1, 1),
        );
    }

    pub fn relu(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        self.run(
            ctx,
            &self.relu,
            &[(0, input.buffer()), (1, out.buffer())],
            (groups64(input.shape.iter().product()), 1, 1),
        );
    }

    pub fn relu_backward(
        &self,
        ctx: &WgpuContext,
        grad_out: &UnifiedTensor,
        input: &UnifiedTensor,
        grad_in: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.relu_bw,
            &[
                (3, grad_out.buffer()),
                (4, input.buffer()),
                (5, grad_in.buffer()),
            ],
            (groups64(grad_out.shape.iter().product()), 1, 1),
        );
    }

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
            (rows as u32, 1, 1),
        );
    }

    pub fn dequantize(&self, ctx: &WgpuContext, input: &UnifiedTensor, out: &UnifiedTensor) {
        let words = input.shape.iter().product::<usize>().div_ceil(4);
        self.run(
            ctx,
            &self.dequant,
            &[(0, input.buffer()), (1, out.buffer())],
            (groups64(words), 1, 1),
        );
    }

    pub fn top_k(&self, ctx: &WgpuContext, input: &UnifiedTensor, out_idx: &UnifiedTensor, k: u32) {
        let params = uniform(ctx, bytemuck::cast_slice(&[k]));
        self.run(
            ctx,
            &self.top_k,
            &[(0, input.buffer()), (1, out_idx.buffer()), (2, &params)],
            (1, 1, 1),
        );
    }

    pub fn affine_dequantize(
        &self,
        ctx: &WgpuContext,
        weight: &wgpu::Buffer,
        scales: &wgpu::Buffer,
        biases: &wgpu::Buffer,
        out: &UnifiedTensor,
    ) {
        let workgroups = groups64(out.shape.iter().product());
        // Metal limit is 65535 per dimension, use 2D dispatch for large tensors
        let max_wg = 65535u32;
        let groups = if workgroups <= max_wg {
            (workgroups, 1, 1)
        } else {
            (max_wg, workgroups.div_ceil(max_wg).min(max_wg), 1)
        };
        self.run(
            ctx,
            &self.affine_dequant,
            &[(0, weight), (1, scales), (2, biases), (3, out.buffer())],
            groups,
        );
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
        let params = uniform(ctx, bytemuck::cast_slice(&[dim as u32, seq_len as u32]));
        self.run(
            ctx,
            &self.conv1d,
            &[
                (0, input.buffer()),
                (1, weight.buffer()),
                (2, out.buffer()),
                (3, bias.buffer()),
                (4, &params),
            ],
            (groups64(dim * seq_len), 1, 1),
        );
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
        let params = uniform(ctx, bytemuck::cast_slice(&[target, vocab_size, 0, 0]));
        self.run(
            ctx,
            &self.ce_loss,
            &[(0, logits.buffer()), (1, loss_out.buffer()), (2, &params)],
            (1, 1, 1),
        );
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
        let params = uniform(ctx, bytemuck::cast_slice(&[target, vocab_size, 0, 0]));
        self.run(
            ctx,
            &self.ce_loss_bw,
            &[(0, logits.buffer()), (1, grad_out.buffer()), (2, &params)],
            (vocab_size.div_ceil(64), 1, 1),
        );
    }

    /// Compute squared elements (for gradient norm computation)
    pub fn gradient_norm_squared(
        &self,
        ctx: &WgpuContext,
        input: &UnifiedTensor,
        out: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.grad_norm,
            &[(0, input.buffer()), (1, out.buffer())],
            (groups64(input.shape.iter().product()), 1, 1),
        );
    }

    /// Clip gradients by scaling: grad *= min(1.0, max_norm / actual_norm)
    pub fn clip_grad_by_norm(&self, ctx: &WgpuContext, grad: &UnifiedTensor, scale: f32) {
        let params = uniform(ctx, bytemuck::cast_slice(&[scale]));
        self.run(
            ctx,
            &self.grad_clip,
            &[(0, grad.buffer()), (1, &params)],
            (groups64(grad.shape.iter().product()), 1, 1),
        );
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
        self.binary(ctx, &self.silu_gate, up, gate, out);
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
        self.run(
            ctx,
            &self.silu_gate_bw,
            &[
                (0, up.buffer()),
                (1, gate.buffer()),
                (3, grad_out.buffer()),
                (4, grad_up.buffer()),
                (5, grad_gate.buffer()),
            ],
            (groups64(up.shape.iter().product()), 1, 1),
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

    /// Gradient accumulation: accum_grad += new_grad
    /// Essential for micro-batching on memory-constrained devices
    pub fn gradient_accumulate(
        &self,
        ctx: &WgpuContext,
        accum_grad: &UnifiedTensor,
        new_grad: &UnifiedTensor,
    ) {
        self.run(
            ctx,
            &self.grad_accum,
            &[(0, accum_grad.buffer()), (1, new_grad.buffer())],
            (groups64(accum_grad.shape.iter().product()), 1, 1),
        );
    }

    /// Gradient scaling: grad *= scale_factor
    /// Used to average gradients after accumulation
    pub fn gradient_scale(&self, ctx: &WgpuContext, grad: &UnifiedTensor, scale: f32) {
        let params = uniform(ctx, bytemuck::cast_slice(&[scale]));
        self.run(
            ctx,
            &self.grad_scale,
            &[(0, grad.buffer()), (1, &params)],
            (groups64(grad.shape.iter().product()), 1, 1),
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
