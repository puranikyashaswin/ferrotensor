//! GPU-resident tensors.

use crate::pool::BufferPool;
use crate::WgpuContext;
use std::sync::Arc;
use wgpu::{util::DeviceExt, Buffer};

/// Data type for tensors - supports mixed precision training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DType {
    /// 32-bit float (the default).
    #[default]
    FP32,
    /// 16-bit float (half precision, for memory savings).
    FP16,
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

/// A dense, row-major tensor stored in a GPU buffer.
///
/// Cloning is cheap: clones and [`reshaped`](Self::reshaped) views share
/// the buffer.
#[derive(Debug, Clone)]
pub struct UnifiedTensor {
    data: Arc<TensorData>,
    /// Dimensions, outermost first.
    pub shape: Vec<usize>,
    /// Element type of the buffer.
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
    /// The underlying GPU buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.data.buffer
    }

    /// Size of the tensor's data in bytes.
    pub fn size_in_bytes(&self) -> u64 {
        self.data.size_in_bytes
    }

    /// Uploads `data` to the start of the buffer.
    pub fn write_data(&self, context: &WgpuContext, data: &[u8]) {
        context.queue.write_buffer(self.buffer(), 0, data);
    }
}

impl UnifiedTensor {
    /// Uploads `data` as an `f32` tensor of the given shape.
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

    /// An uninitialised `f32` tensor drawn from the context's buffer pool.
    pub fn empty(context: &WgpuContext, shape: &[usize]) -> Self {
        Self::empty_with_dtype(context, shape, DType::FP32)
    }

    /// An uninitialised tensor of `dtype` drawn from the buffer pool.
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

    /// A zero-filled `f32` tensor.
    pub fn zeros(context: &WgpuContext, shape: &[usize]) -> Self {
        let size = shape.iter().product::<usize>();
        let data = vec![0.0f32; size];
        Self::new(context, &data, shape)
    }

    /// Uploads raw bytes as an `f32`-typed tensor (e.g. packed quantised data).
    pub fn from_bytes(context: &WgpuContext, data: &[u8], shape: &[usize]) -> Self {
        Self::from_bytes_with_dtype(context, data, shape, DType::FP32)
    }

    /// Uploads raw bytes as a tensor of `dtype`.
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

    /// Copies the tensor back to the CPU, waiting for queued GPU work.
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
