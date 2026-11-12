//! GPU-resident tensors.

use crate::memory::{Allocation, MemoryCategory};
use crate::pool::BufferPool;
use crate::WgpuContext;
use std::sync::atomic::{AtomicBool, Ordering};
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
    /// 4-bit affine codes with per-group f16 scale and minimum (see
    /// [`crate::q4`]); built only with [`UnifiedTensor::from_f32_as_q4`].
    Q4,
}

impl DType {
    /// Size of a single element in bytes.
    ///
    /// # Panics
    ///
    /// For [`DType::Q4`], which has no per-element size (codes and group
    /// scales share one buffer, see [`crate::q4::packed_words`]).
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::FP32 => 4,
            DType::FP16 => 2,
            DType::Q4 => panic!(
                "DType::Q4 has no per-element size; build Q4 tensors with UnifiedTensor::from_f32_as_q4"
            ),
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
    /// Counted in the context's memory tracker; `None` only while dropping.
    memory: Option<Allocation>,
}

impl TensorData {
    fn new(
        context: &WgpuContext,
        buffer: Arc<Buffer>,
        size_in_bytes: u64,
        pool: Option<Arc<BufferPool>>,
        shape: &[usize],
        dtype: DType,
    ) -> Self {
        let numel = shape.iter().product::<usize>() as u64;
        let logical = match dtype {
            // Codes and group scales: the packed buffer is what the shape needs.
            DType::Q4 => size_in_bytes,
            _ => (numel * dtype.size_bytes() as u64).min(size_in_bytes),
        };
        // A buffer without a pool is dropped with the tensor; a pooled one
        // is kept or dropped by the pool, which counts it either way.
        let memory =
            context
                .memory()
                .allocate(dtype, shape, logical, size_in_bytes, pool.is_none());
        Self {
            buffer,
            size_in_bytes,
            pool,
            memory: Some(memory),
        }
    }
}

impl Drop for TensorData {
    fn drop(&mut self) {
        // Stop counting the bytes as live before the pool counts them idle
        // (or, past its cap, as awaiting GPU).
        drop(self.memory.take());
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

    /// What the buffer is counted as in the context's
    /// [`MemoryTracker`](crate::MemoryTracker).
    pub fn memory_category(&self) -> MemoryCategory {
        self.data
            .memory
            .as_ref()
            .map_or(MemoryCategory::Other, Allocation::category)
    }

    /// Count the buffer (shared by clones and reshaped views) as `category`
    /// from now on. Changes no data.
    pub fn set_memory_category(&self, category: MemoryCategory) {
        if let Some(memory) = &self.data.memory {
            memory.set_category(category);
        }
    }

    /// Uploads `data` to the start of the buffer, after all recorded work.
    pub fn write_data(&self, context: &WgpuContext, data: &[u8]) {
        // A queue write runs before commands not yet submitted, so submit
        // them first: ops recorded earlier must see the old contents.
        context.flush();
        context.queue.write_buffer(self.buffer(), 0, data);
        context.note_upload(data.len() as u64);
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
        context.note_upload(size_in_bytes);
        Self {
            data: Arc::new(TensorData::new(
                context,
                Arc::new(buffer),
                size_in_bytes,
                None,
                shape,
                DType::FP32,
            )),
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
        assert!(
            dtype != DType::Q4,
            "empty_with_dtype: Q4 tensors are built with from_f32_as_q4"
        );
        let size_in_bytes = (shape.iter().product::<usize>() * dtype.size_bytes()) as u64;
        let buffer = context.pool.get(&context.device, size_in_bytes);
        Self {
            data: Arc::new(TensorData::new(
                context,
                buffer,
                size_in_bytes,
                Some(context.pool.clone()),
                shape,
                dtype,
            )),
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
        context.note_upload(size_in_bytes);
        Self {
            data: Arc::new(TensorData::new(
                context,
                Arc::new(buffer),
                size_in_bytes,
                None,
                shape,
                dtype,
            )),
            shape: shape.to_vec(),
            dtype,
        }
    }

    /// Uploads `data` rounded to f16, two values packed per 32-bit word
    /// (the first in the low half), as an [`DType::FP16`] tensor of `shape`.
    /// Half the memory of [`new`](Self::new); read it with the `*_f16b`
    /// matmuls, not [`to_vec`](Self::to_vec), which reads f32.
    pub fn from_f32_as_f16(context: &WgpuContext, data: &[f32], shape: &[usize]) -> Self {
        let mut bytes: Vec<u8> = data
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        // Storage bindings are whole 32-bit words.
        bytes.resize(bytes.len().next_multiple_of(4).max(4), 0);
        Self::from_bytes_with_dtype(context, &bytes, shape, DType::FP16)
    }

    /// Quantises row-major `data` of 2-D `shape` `[k, n]` to the 4-bit
    /// format of [`crate::q4`] as a [`DType::Q4`] tensor of logical shape
    /// `shape`: about 4.5 bits per weight. Read it with the `*_q4b`
    /// matmuls, not [`to_vec`](Self::to_vec). Fails, uploading nothing, on
    /// NaN or infinite values and on values beyond the f16 range.
    pub fn from_f32_as_q4(
        context: &WgpuContext,
        data: &[f32],
        shape: &[usize],
    ) -> Result<Self, crate::q4::Q4Error> {
        let [k, n] = shape[..] else {
            panic!("from_f32_as_q4 expects a 2-D shape, got {shape:?}");
        };
        let words = crate::q4::quantize(data, k, n)?;
        Ok(Self::from_bytes_with_dtype(
            context,
            bytemuck::cast_slice(&words),
            shape,
            DType::Q4,
        ))
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
        // The 4-bit storage order and grouping depend on the 2-D shape.
        assert!(
            self.dtype != DType::Q4 || new_shape == self.shape.as_slice(),
            "Cannot reshape a Q4 tensor of shape {:?} to {:?}",
            self.shape,
            new_shape
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
        // Counted as staging while the copy runs, then as awaiting GPU until
        // the next wait polls the device and wgpu destroys it.
        let _counted = context.memory().allocate_raw(MemoryCategory::Staging, size);
        context.record(|encoder| {
            encoder.copy_buffer_to_buffer(&self.data.buffer, 0, &staging, 0, size);
        });
        context.flush();

        let slice = staging.slice(..);
        let mapped = Arc::new(AtomicBool::new(false));
        let flag = mapped.clone();
        slice.map_async(wgpu::MapMode::Read, move |_| {
            flag.store(true, Ordering::Release)
        });
        context.poll_until(|| mapped.load(Ordering::Acquire));

        let data = slice.get_mapped_range();
        let res = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        res
    }
}
