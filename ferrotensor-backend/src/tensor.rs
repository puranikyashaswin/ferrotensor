//! GPU-resident tensors.

use crate::bind_cache::BindGroupCache;
use crate::memory::{Allocation, MemoryCategory};
use crate::pool::BufferPool;
use crate::profile::GpuCounters;
use crate::q4::{Q4Error, Q4Group};
use crate::WgpuContext;
use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;
use wgpu::{util::DeviceExt, Buffer};

/// Why a readback ([`UnifiedTensor::to_vec`], [`UnifiedTensor::read_many`])
/// returned no values: wgpu could not map the staging buffer, which happens
/// when the GPU device was lost.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ReadbackError {
    /// The staging buffer's `map_async` reported a failure.
    Map(wgpu::BufferAsyncError),
    /// The mapped range of the staging buffer could not be read.
    Range(wgpu::MapRangeError),
}

impl fmt::Display for ReadbackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Map(e) => write!(f, "GPU readback failed to map its staging buffer: {e}"),
            Self::Range(e) => write!(f, "GPU readback failed to read its staging buffer: {e}"),
        }
    }
}

impl std::error::Error for ReadbackError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Map(e) => Some(e),
            Self::Range(e) => Some(e),
        }
    }
}

/// Element type of a tensor buffer.
///
/// Arithmetic is f32. [`DType::FP16`] is only a storage format for frozen
/// weights (see [`UnifiedTensor::from_f32_as_f16`]); there is no f16
/// activation, gradient or mixed-precision training path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DType {
    /// 32-bit float (the default).
    #[default]
    FP32,
    /// 16-bit float storage, two values per 32-bit word. Only the `*_f16b`
    /// matmuls read it, and they accumulate in f32.
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
    #[expect(
        clippy::panic,
        reason = "documented panic for DType::Q4, which has no per-element size"
    )]
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
    pub dtype: DType,
    q4_group: Option<Q4Group>,
}

#[derive(Debug)]
struct TensorData {
    buffer: Arc<Buffer>,
    size_in_bytes: u64,
    pool: Option<Arc<BufferPool>>,
    /// Counted in the context's memory tracker; `None` only while dropping.
    memory: Option<Allocation>,
    /// Evicts the buffer when the tensor owns it and drops it.
    bind_groups: Arc<BindGroupCache>,
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
            bind_groups: Arc::clone(context.pool.bind_groups()),
        }
    }
}

impl Drop for TensorData {
    fn drop(&mut self) {
        // Stop counting the bytes as live before the pool counts them idle
        // (or, past its cap, as awaiting GPU).
        drop(self.memory.take());
        match &self.pool {
            Some(pool) => pool.return_buffer(self.size_in_bytes, self.buffer.clone()),
            None => self.bind_groups.evict(&self.buffer),
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
            q4_group: None,
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
            q4_group: None,
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
            q4_group: None,
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
    /// `shape`, `group` weights per scale (4 + 32 / `group` bits per weight,
    /// 4.5 with the default 64). Read it with the `*_q4b` matmuls, not
    /// [`to_vec`](Self::to_vec). Fails, uploading nothing, on NaN or
    /// infinite values, on values beyond the f16 range and when `shape` is
    /// not 2-D.
    pub fn from_f32_as_q4(
        context: &WgpuContext,
        data: &[f32],
        shape: &[usize],
        group: Q4Group,
    ) -> Result<Self, Q4Error> {
        let [k, n] = shape[..] else {
            return Err(Q4Error::NotMatrix { rank: shape.len() });
        };
        let words = crate::q4::quantize(data, k, n, group)?;
        let tensor =
            Self::from_bytes_with_dtype(context, bytemuck::cast_slice(&words), shape, DType::Q4);
        Ok(Self {
            q4_group: Some(group),
            ..tensor
        })
    }

    /// Weights per scale of a [`DType::Q4`] tensor; `None` for other types.
    pub fn q4_group(&self) -> Option<Q4Group> {
        self.q4_group
    }

    /// Create an empty FP16 tensor. No code in this workspace calls it.
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
            q4_group: self.q4_group,
        }
    }

    /// Whether no other tensor, view or clone shares this tensor's storage, so
    /// writing into it in place changes nothing anyone else reads.
    #[must_use]
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.data) == 1
    }

    /// Copies the tensor back to the CPU, waiting for queued GPU work.
    ///
    /// # Errors
    /// [`ReadbackError`] when wgpu cannot map the staging buffer, which
    /// happens when the GPU device was lost.
    pub async fn to_vec(&self, context: &WgpuContext) -> Result<Vec<f32>, ReadbackError> {
        let start = Instant::now();
        let size = self.size_in_bytes();
        context.counters.readbacks.fetch_add(1, Ordering::Relaxed);
        context
            .counters
            .readback_bytes
            .fetch_add(size, Ordering::Relaxed);
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
        let data = map_for_read(context, &slice)?;
        let res = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        GpuCounters::add_elapsed(&context.counters.readback_nanos, start);
        Ok(res)
    }

    /// Reads the f32 values of every tensor in `tensors` back to the host
    /// with one copy into one staging buffer, so the queue drains once
    /// however many tensors there are.
    ///
    /// # Errors
    /// As [`to_vec`](Self::to_vec).
    pub async fn read_many(
        context: &WgpuContext,
        tensors: &[&UnifiedTensor],
    ) -> Result<Vec<Vec<f32>>, ReadbackError> {
        let start = Instant::now();
        let sizes: Vec<u64> = tensors.iter().map(|t| t.size_in_bytes()).collect();
        let size: u64 = sizes.iter().sum();
        if size == 0 {
            return Ok(tensors.iter().map(|_| Vec::new()).collect());
        }
        context.counters.readbacks.fetch_add(1, Ordering::Relaxed);
        context
            .counters
            .readback_bytes
            .fetch_add(size, Ordering::Relaxed);
        let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let _counted = context.memory().allocate_raw(MemoryCategory::Staging, size);
        context.record(|encoder| {
            let mut offset = 0;
            for (t, &bytes) in tensors.iter().zip(&sizes) {
                if bytes > 0 {
                    encoder.copy_buffer_to_buffer(&t.data.buffer, 0, &staging, offset, bytes);
                }
                offset += bytes;
            }
        });
        context.flush();

        let slice = staging.slice(..);
        let data = map_for_read(context, &slice)?;
        let all: &[f32] = bytemuck::cast_slice(&data);
        let mut res = Vec::with_capacity(tensors.len());
        let mut at = 0;
        for &bytes in &sizes {
            let len = (bytes / 4) as usize;
            res.push(all[at..at + len].to_vec());
            at += len;
        }
        drop(data);
        staging.unmap();
        GpuCounters::add_elapsed(&context.counters.readback_nanos, start);
        Ok(res)
    }
}

impl UnifiedTensor {
    /// Uploads the transpose of the row-major `[rows, cols]` matrix `data`
    /// as a `[cols, rows]` tensor of `dtype`, converting each value while it
    /// is written into the upload bytes, so no transposed f32 copy is made on
    /// the host.
    ///
    /// The buffer holds the same bytes as transposing `data` and passing the
    /// result to [`new`](Self::new) ([`DType::FP32`]) or
    /// [`from_f32_as_f16`](Self::from_f32_as_f16) ([`DType::FP16`]).
    ///
    /// # Panics
    /// If `data.len() != rows * cols`, or if `dtype` is [`DType::Q4`] (build
    /// 4-bit tensors with [`from_f32_as_q4`](Self::from_f32_as_q4)).
    pub fn from_f32_transposed(
        context: &WgpuContext,
        data: &[f32],
        rows: usize,
        cols: usize,
        dtype: DType,
    ) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "from_f32_transposed: {} values for a [{rows}, {cols}] matrix",
            data.len()
        );
        assert!(
            dtype != DType::Q4,
            "from_f32_transposed: DType::Q4 is not supported; use UnifiedTensor::from_f32_as_q4"
        );
        let len = upload_len(data.len(), dtype);
        Self::from_fill(context, len, &[cols, rows], dtype, |out| {
            write_transposed(data, rows, cols, dtype, out)
        })
    }

    /// A tensor of `dtype` whose first `size_in_bytes` bytes `fill` writes
    /// (into zeroed memory). The
    /// buffer is laid out as by `create_buffer_init`, which
    /// [`from_bytes_with_dtype`](Self::from_bytes_with_dtype) uses: its
    /// size is rounded up to whole 32-bit words, at least one, with zeros
    /// after the data, and a zero-byte tensor gets an empty, unmapped buffer.
    fn from_fill(
        context: &WgpuContext,
        size_in_bytes: usize,
        shape: &[usize],
        dtype: DType,
        fill: impl FnOnce(&mut [u8]),
    ) -> Self {
        // wgpu hands out mapped memory write-only, so `fill` writes into host
        // memory that `create_buffer_init` then copies in.
        let mut host = vec![0u8; size_in_bytes];
        fill(&mut host);
        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Byte Tensor"),
                contents: &host,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        context.note_upload(size_in_bytes as u64);
        Self {
            data: Arc::new(TensorData::new(
                context,
                Arc::new(buffer),
                size_in_bytes as u64,
                None,
                shape,
                dtype,
            )),
            shape: shape.to_vec(),
            dtype,
            q4_group: None,
        }
    }
}

/// Bytes [`UnifiedTensor::from_f32_transposed`] uploads for `n` values:
/// four per value for [`DType::FP32`]; for [`DType::FP16`] two per value,
/// rounded up to whole 32-bit words and at least one word, as in
/// [`UnifiedTensor::from_f32_as_f16`].
fn upload_len(n: usize, dtype: DType) -> usize {
    match dtype {
        DType::FP32 => n * 4,
        DType::FP16 => (n * 2).next_multiple_of(4).max(4),
        DType::Q4 => unreachable!("from_f32_transposed rejects DType::Q4"),
    }
}

/// Writes the `[cols, rows]` transpose of the row-major `[rows, cols]`
/// matrix `src` into `out` as `dtype` values: f32 in native byte order (as
/// `bytemuck` casts them for [`UnifiedTensor::new`]) or little-endian f16
/// rounded by `half::f16::from_f32` (as in
/// [`UnifiedTensor::from_f32_as_f16`]). Bytes past the values are left as
/// they are. `out` is filled front to back.
#[expect(
    clippy::expect_used,
    reason = "out holds one slot per value, see the length contract above"
)]
fn write_transposed(src: &[f32], rows: usize, cols: usize, dtype: DType, out: &mut [u8]) {
    debug_assert_eq!(src.len(), rows * cols);
    debug_assert!(out.len() >= src.len() * dtype.size_bytes());
    match dtype {
        DType::FP32 => {
            let mut chunks = out.as_chunks_mut::<4>().0.iter_mut();
            for col in 0..cols {
                for row in 0..rows {
                    *chunks.next().expect("output holds every value") =
                        src[row * cols + col].to_ne_bytes();
                }
            }
        }
        DType::FP16 => {
            let mut chunks = out.as_chunks_mut::<2>().0.iter_mut();
            for col in 0..cols {
                for row in 0..rows {
                    *chunks.next().expect("output holds every value") =
                        half::f16::from_f32(src[row * cols + col]).to_le_bytes();
                }
            }
        }
        DType::Q4 => unreachable!("from_f32_transposed rejects DType::Q4"),
    }
}

/// Maps `slice` of a `MAP_READ` staging buffer and returns its bytes,
/// polling the device until wgpu reports the mapping.
///
/// # Errors
/// [`ReadbackError`] when the mapping or the mapped range fails.
fn map_for_read(
    context: &WgpuContext,
    slice: &wgpu::BufferSlice<'_>,
) -> Result<wgpu::BufferView, ReadbackError> {
    // 0 while the mapping is pending, then MAPPED or FAILED.
    const MAPPED: u8 = 1;
    const FAILED: u8 = 2;
    let state = Arc::new(AtomicU8::new(0));
    let flag = state.clone();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        flag.store(
            if result.is_ok() { MAPPED } else { FAILED },
            Ordering::Release,
        )
    });
    context.poll_until(|| state.load(Ordering::Acquire) != 0);
    if state.load(Ordering::Acquire) == FAILED {
        return Err(ReadbackError::Map(wgpu::BufferAsyncError));
    }
    slice.get_mapped_range().map_err(ReadbackError::Range)
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::{upload_len, write_transposed, DType};

    /// The bytes the model builder uploaded for a transposed linear weight
    /// before `from_f32_transposed`: a transposed f32 copy, then either its
    /// bytes as `UnifiedTensor::new` sees them or the f16 bytes
    /// `UnifiedTensor::from_f32_as_f16` builds. Kept as the reference.
    fn old_transposed_bytes(src: &[f32], rows: usize, cols: usize, dtype: DType) -> Vec<u8> {
        let mut transposed = vec![0.0; src.len()];
        for row in 0..rows {
            for col in 0..cols {
                transposed[col * rows + row] = src[row * cols + col];
            }
        }
        match dtype {
            DType::FP32 => bytemuck::cast_slice::<f32, u8>(&transposed).to_vec(),
            DType::FP16 => {
                let mut bytes: Vec<u8> = transposed
                    .iter()
                    .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                    .collect();
                bytes.resize(bytes.len().next_multiple_of(4).max(4), 0);
                bytes
            }
            DType::Q4 => unreachable!("the reference covers f32 and f16 only"),
        }
    }

    /// What `from_fill` hands to `write_transposed`: a zeroed buffer.
    fn new_transposed_bytes(src: &[f32], rows: usize, cols: usize, dtype: DType) -> Vec<u8> {
        let mut out = vec![0u8; upload_len(src.len(), dtype)];
        write_transposed(src, rows, cols, dtype, &mut out);
        out
    }

    fn pseudo_random_bits(n: usize, mut seed: u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                f32::from_bits(seed as u32)
            })
            .collect()
    }

    /// Values at and around every special case of the f32 to f16 rounding.
    fn edge_values() -> Vec<f32> {
        let mut v = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
            f32::from_bits(0x7f80_0001), // signalling NaN, low payload
            f32::from_bits(0x7fc1_2345), // quiet NaN with payload
            f32::from_bits(0xffbf_ffff), // negative NaN, all payload bits
            f32::from_bits(1),           // smallest f32 subnormal
            f32::from_bits(0x007f_ffff), // largest f32 subnormal
            -f32::from_bits(0x0040_0000),
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
            65504.0, // f16 max
            65519.0, // rounds down to f16 max
            65520.0, // overflows f16 to infinity
            -65520.0,
            1e10,
            2f32.powi(-14),                               // f16 smallest normal
            2f32.powi(-24),                               // f16 smallest subnormal
            2f32.powi(-25), // half of it: a tie, rounds to even (zero)
            f32::from_bits(2f32.powi(-25).to_bits() + 1), // rounds up
            1.0e-9,         // underflows f16 to zero
            -1.0e-9,
            0.1,
            1.0 + 2f32.powi(-11), // tie between two f16 values
            1.0 + 3.0 * 2f32.powi(-11),
            2049.0,
        ];
        // Every f16 value widened to f32, plus its f32 neighbours.
        for bits in (0u16..=u16::MAX).step_by(7) {
            let x = half::f16::from_bits(bits).to_f32();
            v.push(x);
            v.push(f32::from_bits(x.to_bits().wrapping_add(1)));
            v.push(f32::from_bits(x.to_bits().wrapping_sub(1)));
        }
        v
    }

    fn assert_same(src: &[f32], rows: usize, cols: usize) {
        for dtype in [DType::FP32, DType::FP16] {
            let old = old_transposed_bytes(src, rows, cols, dtype);
            let new = new_transposed_bytes(src, rows, cols, dtype);
            assert_eq!(
                old.len(),
                new.len(),
                "{dtype:?} [{rows}, {cols}]: byte length"
            );
            if let Some(i) = old.iter().zip(&new).position(|(a, b)| a != b) {
                panic!("{dtype:?} [{rows}, {cols}]: first differing byte at {i}");
            }
        }
    }

    #[test]
    fn transposed_bytes_match_the_old_three_step_upload() {
        for (rows, cols) in [
            (1, 1),
            (1, 7),
            (7, 1),
            (2, 3),
            (3, 2),
            (5, 5),
            (17, 33),
            (64, 48),
        ] {
            let src = pseudo_random_bits(rows * cols, (rows * 131 + cols) as u64);
            assert_same(&src, rows, cols);
        }
    }

    #[test]
    fn transposed_bytes_match_on_edge_values() {
        let mut values = edge_values();
        // A whole number of rows of 13 (odd, so f16 rows end mid-word).
        while !values.len().is_multiple_of(13) {
            values.push(-0.0);
        }
        let rows = values.len() / 13;
        assert_same(&values, rows, 13);
        assert_same(&values, 13, rows);
        assert_same(&values, 1, values.len());
    }

    #[test]
    fn empty_and_odd_sized_uploads_are_padded_like_from_f32_as_f16() {
        assert_eq!(upload_len(0, DType::FP16), 4);
        assert_eq!(upload_len(1, DType::FP16), 4);
        assert_eq!(upload_len(3, DType::FP16), 8);
        assert_eq!(upload_len(4, DType::FP16), 8);
        assert_eq!(upload_len(0, DType::FP32), 0);
        assert_eq!(upload_len(3, DType::FP32), 12);
        assert_same(&[], 0, 0);
        assert_same(&[], 0, 5);
        assert_same(&[1.5], 1, 1);
        assert_same(&[1.5, -2.0, 3.25], 1, 3);
    }

    #[test]
    fn transpose_puts_columns_first() {
        let src = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = new_transposed_bytes(&src, 2, 3, DType::FP32);
        let values: Vec<f32> = out
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_ne_bytes(*c))
            .collect();
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
