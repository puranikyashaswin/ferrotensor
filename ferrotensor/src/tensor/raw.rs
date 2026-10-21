//! Device dispatch for the primitive kernels.
//!
//! Every `*_raw` method computes a fresh tensor with no autograd node: on a
//! GPU device it dispatches the WGSL kernel, on the CPU it runs the matching
//! reference kernel in [`super::cpu`]. The public ops in `ops.rs` build on
//! these and attach backward nodes.

use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferUsages};

use super::core::{alloc, Storage};
use super::cpu;
use super::shape::{broadcast_shapes, broadcast_strides, contiguous_strides, split_at_dim};
use super::Tensor;
use crate::backend::{reduce_op, ComputeEngine};
use crate::device::GpuContext;

fn uniform(ctx: &GpuContext, words: [u32; 4]) -> Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params"),
            contents: bytemuck::cast_slice(&words),
            usage: BufferUsages::UNIFORM,
        })
}

/// A read-only `u32` storage buffer (at least one element).
fn u32_buffer(ctx: &GpuContext, data: &[u32]) -> Buffer {
    let padded: &[u32] = if data.is_empty() { &[0] } else { data };
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Meta"),
            contents: bytemuck::cast_slice(padded),
            usage: BufferUsages::STORAGE,
        })
}

fn as_u32(values: &[usize]) -> Vec<u32> {
    values
        .iter()
        .map(|&v| u32::try_from(v).expect("tensor dimension exceeds u32"))
        .collect()
}

impl Tensor {
    /// The GPU context and buffer, when this tensor lives on a GPU.
    fn gpu(&self) -> Option<(&GpuContext, &Buffer)> {
        match (self.device.gpu_ctx(), &*self.storage) {
            (Some(ctx), Storage::Gpu(buffer)) => Some((ctx, buffer)),
            _ => None,
        }
    }

    fn gpu_output(&self, buffer: Buffer, shape: Vec<usize>) -> Tensor {
        Tensor::from_storage(Storage::Gpu(buffer), shape, &self.device)
    }

    /// A new tensor on this tensor's device.
    pub(crate) fn host_output(&self, data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::from_vec(data, shape, &self.device)
    }

    pub(crate) fn assert_same_device(&self, other: &Tensor) {
        assert!(
            self.device.same_as(&other.device),
            "tensors are on different devices ({} and {})",
            self.device.name(),
            other.device.name()
        );
    }

    /// A copy with its own storage (no autograd).
    pub fn deep_clone(&self) -> Tensor {
        match self.gpu() {
            Some((ctx, buffer)) => {
                let out = alloc(ctx, self.numel());
                let mut encoder =
                    ctx.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("deep_clone"),
                        });
                encoder.copy_buffer_to_buffer(buffer, 0, &out, 0, (self.numel().max(1) * 4) as u64);
                ctx.queue.submit(std::iter::once(encoder.finish()));
                self.gpu_output(out, self.shape.clone())
            }
            None => self.host_output(self.to_vec(), &self.shape),
        }
    }

    /// Element-wise op `op` (see `backend::unary_op`) with scalars `a`, `b`.
    pub(crate) fn unary_raw(&self, op: u32, a: f32, b: f32) -> Tensor {
        let n = self.numel();
        match self.gpu() {
            Some((ctx, input)) if n > 0 => {
                let out = alloc(ctx, n);
                let params = uniform(ctx, [op, n as u32, a.to_bits(), b.to_bits()]);
                ctx.engine.run(
                    &ctx.device,
                    &ctx.queue,
                    "unary",
                    &[input, &out, &params],
                    ComputeEngine::workgroups_1d(n),
                );
                self.gpu_output(out, self.shape.clone())
            }
            _ => self.host_output(cpu::unary(op, &self.to_vec(), a, b), &self.shape),
        }
    }

    /// Broadcasting binary op `op` (see `backend::binary_op`).
    ///
    /// # Panics
    /// Panics when the shapes cannot be broadcast together.
    pub(crate) fn binary_raw(&self, other: &Tensor, op: u32) -> Tensor {
        self.assert_same_device(other);
        let out_shape = broadcast_shapes(&self.shape, &other.shape).unwrap_or_else(|| {
            panic!(
                "shapes {:?} and {:?} cannot be broadcast together",
                self.shape, other.shape
            )
        });
        let sa = broadcast_strides(&self.shape, &out_shape);
        let sb = broadcast_strides(&other.shape, &out_shape);
        let n: usize = out_shape.iter().product();
        match (self.gpu(), other.gpu()) {
            (Some((ctx, a)), Some((_, b))) if n > 0 => {
                let out = alloc(ctx, n);
                let params = uniform(ctx, [op, n as u32, out_shape.len() as u32, 0]);
                let meta = u32_buffer(ctx, &as_u32(&[&out_shape[..], &sa, &sb].concat()));
                ctx.engine.run(
                    &ctx.device,
                    &ctx.queue,
                    "binary",
                    &[a, b, &out, &params, &meta],
                    ComputeEngine::workgroups_1d(n),
                );
                self.gpu_output(out, out_shape)
            }
            _ => {
                let data = cpu::binary(op, &self.to_vec(), &sa, &other.to_vec(), &sb, &out_shape);
                self.host_output(data, &out_shape)
            }
        }
    }

    /// Reduce over `dim` (removed from the shape) with `op` (see
    /// `backend::reduce_op`).
    pub(crate) fn reduce_raw(&self, dim: usize, op: u32) -> Tensor {
        let (outer, size, inner) = split_at_dim(&self.shape, dim);
        let mut out_shape = self.shape.clone();
        out_shape.remove(dim);
        if size == 0 {
            assert_eq!(op, reduce_op::SUM, "max/min of an empty dimension");
            return Tensor::zeros(&out_shape, &self.device);
        }
        let n = outer * inner;
        match self.gpu() {
            Some((ctx, input)) if n > 0 => {
                let out = alloc(ctx, n);
                let params = uniform(ctx, [op, outer as u32, size as u32, inner as u32]);
                ctx.engine.run(
                    &ctx.device,
                    &ctx.queue,
                    "reduce",
                    &[input, &out, &params],
                    ComputeEngine::workgroups_1d(n),
                );
                self.gpu_output(out, out_shape)
            }
            _ => {
                let data = cpu::reduce(op, &self.to_vec(), outer, size, inner);
                self.host_output(data, &out_shape)
            }
        }
    }

    /// Softmax (or log-softmax) over `dim`.
    pub(crate) fn softmax_raw(&self, dim: usize, log: bool) -> Tensor {
        let (outer, size, inner) = split_at_dim(&self.shape, dim);
        let n = outer * inner;
        match self.gpu() {
            Some((ctx, input)) if n > 0 && size > 0 => {
                let out = alloc(ctx, self.numel());
                let params = uniform(ctx, [log as u32, outer as u32, size as u32, inner as u32]);
                ctx.engine.run(
                    &ctx.device,
                    &ctx.queue,
                    "softmax",
                    &[input, &out, &params],
                    ComputeEngine::workgroups_1d(n),
                );
                self.gpu_output(out, self.shape.clone())
            }
            _ => {
                let data = cpu::softmax(&self.to_vec(), outer, size, inner, log);
                self.host_output(data, &self.shape)
            }
        }
    }

    /// Matrix product over the last two dims, with leading batch dims that
    /// are either equal or present on one side only.
    pub(crate) fn matmul_raw(&self, other: &Tensor) -> Tensor {
        self.assert_same_device(other);
        assert!(
            self.ndim() >= 2 && other.ndim() >= 2,
            "matmul needs tensors with at least 2 dims, got {:?} and {:?}",
            self.shape,
            other.shape
        );
        let (m, k) = (self.shape[self.ndim() - 2], self.shape[self.ndim() - 1]);
        let (k2, n) = (other.shape[other.ndim() - 2], other.shape[other.ndim() - 1]);
        assert_eq!(
            k, k2,
            "matmul inner dimensions differ: {:?} @ {:?}",
            self.shape, other.shape
        );
        let a_batch = &self.shape[..self.ndim() - 2];
        let b_batch = &other.shape[..other.ndim() - 2];
        let (batch_shape, a_batched, b_batched) = if a_batch == b_batch {
            (a_batch.to_vec(), true, true)
        } else if b_batch.is_empty() {
            (a_batch.to_vec(), true, false)
        } else if a_batch.is_empty() {
            (b_batch.to_vec(), false, true)
        } else {
            panic!(
                "matmul batch dims must match or be absent on one side: {:?} @ {:?}",
                self.shape, other.shape
            );
        };
        let batch: usize = batch_shape.iter().product();
        let out_shape = [&batch_shape[..], &[m, n]].concat();

        // A batched left side against a shared right matrix is one big matmul.
        let (rows, batches, a_flag, b_flag) = if a_batched && !b_batched {
            (batch * m, 1, false, false)
        } else {
            (m, batch, a_batched, b_batched)
        };

        let total = batch * m * n;
        match (self.gpu(), other.gpu()) {
            (Some((ctx, a)), Some((_, b))) if total > 0 && k > 0 => {
                let out = alloc(ctx, total);
                let flags = a_flag as u32 | ((b_flag as u32) << 1);
                let params = uniform(ctx, [rows as u32, k as u32, n as u32, flags]);
                ctx.engine.run(
                    &ctx.device,
                    &ctx.queue,
                    "matmul",
                    &[a, b, &out, &params],
                    ComputeEngine::workgroups_matmul(rows, n, batches),
                );
                self.gpu_output(out, out_shape)
            }
            _ => {
                let data = cpu::matmul(
                    &self.to_vec(),
                    &other.to_vec(),
                    rows,
                    k,
                    n,
                    batches,
                    a_flag,
                    b_flag,
                );
                self.host_output(data, &out_shape)
            }
        }
    }

    /// Copy into `out_shape`, reading output dim `d` with source stride
    /// `src_strides[d]`, starting at element `offset`.
    pub(crate) fn strided_copy_raw(
        &self,
        out_shape: &[usize],
        src_strides: &[usize],
        offset: usize,
    ) -> Tensor {
        let n: usize = out_shape.iter().product();
        match self.gpu() {
            Some((ctx, input)) if n > 0 => {
                let out = alloc(ctx, n);
                let params = uniform(ctx, [n as u32, out_shape.len() as u32, offset as u32, 0]);
                let meta = u32_buffer(ctx, &as_u32(&[out_shape, src_strides].concat()));
                ctx.engine.run(
                    &ctx.device,
                    &ctx.queue,
                    "permute",
                    &[input, &out, &params, &meta],
                    ComputeEngine::workgroups_1d(n),
                );
                self.gpu_output(out, out_shape.to_vec())
            }
            _ => {
                let data = cpu::permute(&self.to_vec(), out_shape, src_strides, offset);
                self.host_output(data, out_shape)
            }
        }
    }

    /// Reorder dimensions: output dim `d` is input dim `perm[d]`.
    pub(crate) fn permute_raw(&self, perm: &[usize]) -> Tensor {
        let strides = contiguous_strides(&self.shape);
        let out_shape: Vec<usize> = perm.iter().map(|&p| self.shape[p]).collect();
        let src_strides: Vec<usize> = perm.iter().map(|&p| strides[p]).collect();
        self.strided_copy_raw(&out_shape, &src_strides, 0)
    }

    /// Broadcast to `shape` (materialised).
    pub(crate) fn expand_raw(&self, shape: &[usize]) -> Tensor {
        if self.shape == shape {
            return self.clone();
        }
        let strides = broadcast_strides(&self.shape, shape);
        self.strided_copy_raw(shape, &strides, 0)
    }

    /// Rows of the first dimension picked by `idx`.
    pub(crate) fn gather_rows_raw(&self, idx: &[usize]) -> Tensor {
        let rows = self.shape[0];
        let row: usize = self.shape[1..].iter().product();
        for &i in idx {
            assert!(
                i < rows,
                "index {i} out of range for dimension of size {rows}"
            );
        }
        let out_shape = [&[idx.len()][..], &self.shape[1..]].concat();
        let idx32 = as_u32(idx);
        let n = idx.len() * row;
        match self.gpu() {
            Some((ctx, table)) if n > 0 => {
                let out = alloc(ctx, n);
                let indices = u32_buffer(ctx, &idx32);
                let params = uniform(ctx, [idx.len() as u32, row as u32, 0, 0]);
                ctx.engine.run(
                    &ctx.device,
                    &ctx.queue,
                    "gather_rows",
                    &[table, &indices, &out, &params],
                    ComputeEngine::workgroups_1d(n),
                );
                self.gpu_output(out, out_shape)
            }
            _ => {
                let data = cpu::gather_rows(&self.to_vec(), &idx32, row);
                self.host_output(data, &out_shape)
            }
        }
    }
}
