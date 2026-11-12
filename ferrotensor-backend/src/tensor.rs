//! GPU-resident tensors.

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
        context.note_upload(size_in_bytes);
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

impl UnifiedTensor {
    /// Uploads the transpose of the row-major `[rows, cols]` matrix `data`
    /// as a `[cols, rows]` tensor of `dtype`, converting each value while it
    /// is written into the buffer's mapped memory, so no transposed or
    /// converted copy is made on the host.
    ///
    /// The buffer holds the same bytes as transposing `data` and passing the
    /// result to [`new`](Self::new) ([`DType::FP32`]) or
    /// [`from_f32_as_f16`](Self::from_f32_as_f16) ([`DType::FP16`]).
    ///
    /// # Panics
    /// If `data.len() != rows * cols`.
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
        let len = upload_len(data.len(), dtype);
        Self::from_fill(context, len, &[cols, rows], dtype, |out| {
            write_transposed(data, rows, cols, dtype, out)
        })
    }

    /// A tensor of `dtype` whose first `size_in_bytes` bytes `fill` writes
    /// straight into the buffer's mapped memory (zeroed beforehand). The
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
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let buffer = if size_in_bytes == 0 {
            let empty: &mut [u8] = &mut [];
            fill(empty);
            context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Byte Tensor"),
                size: 0,
                usage,
                mapped_at_creation: false,
            })
        } else {
            let align_mask = wgpu::COPY_BUFFER_ALIGNMENT - 1;
            let padded = ((size_in_bytes as u64 + align_mask) & !align_mask)
                .max(wgpu::COPY_BUFFER_ALIGNMENT);
            let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Byte Tensor"),
                size: padded,
                usage,
                mapped_at_creation: true,
            });
            {
                let mut mapped = buffer.slice(..).get_mapped_range_mut();
                fill(&mut mapped[..size_in_bytes]);
            }
            buffer.unmap();
            buffer
        };
        context.note_upload(size_in_bytes as u64);
        Self {
            data: Arc::new(TensorData {
                buffer: Arc::new(buffer),
                size_in_bytes: size_in_bytes as u64,
                pool: None,
            }),
            shape: shape.to_vec(),
            dtype,
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
    }
}

/// Writes the `[cols, rows]` transpose of the row-major `[rows, cols]`
/// matrix `src` into `out` as `dtype` values: f32 in native byte order (as
/// `bytemuck` casts them for [`UnifiedTensor::new`]) or little-endian f16
/// rounded by `half::f16::from_f32` (as in
/// [`UnifiedTensor::from_f32_as_f16`]). Bytes past the values are left as
/// they are. `out` is filled front to back, so writes to mapped GPU memory
/// stay sequential.
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
    }
}

#[cfg(test)]
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
