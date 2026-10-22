//! Core Tensor Implementation
//!
//! The [`Tensor`] struct, its storage, constructors and data access.

use std::sync::{Arc, RwLock};
use wgpu::util::DeviceExt;
use wgpu::{Buffer, BufferUsages};

use super::autograd::Node;
use crate::device::GpuContext;
use crate::Device;

/// Where a tensor's values live.
pub(crate) enum Storage {
    /// Host memory, row-major.
    Cpu(RwLock<Vec<f32>>),
    /// A GPU storage buffer of `f32`, row-major.
    Gpu(Buffer),
}

/// Shared slot holding a tensor's accumulated gradient.
pub(crate) type GradSlot = Arc<RwLock<Option<Tensor>>>;

/// A multi-dimensional `f32` array on a [`Device`], with reverse-mode
/// automatic differentiation.
///
/// Tensors are cheap to clone: clones share storage and gradient, so a
/// parameter held by both a layer and an optimizer is the same variable.
///
/// # Example
/// ```rust
/// use ferrotensor::prelude::*;
///
/// let device = Device::cpu();
/// let w = Tensor::from_data(&[1.0, 2.0, 3.0], &[3], &device).requires_grad_();
/// let loss = w.mul(&w).sum(); // sum(w^2)
/// loss.backward();
/// assert_eq!(w.grad().unwrap().to_vec(), vec![2.0, 4.0, 6.0]);
/// ```
#[derive(Clone)]
pub struct Tensor {
    pub(crate) storage: Arc<Storage>,
    /// Shape of the tensor (row-major, contiguous).
    pub shape: Vec<usize>,
    pub(crate) device: Device,
    pub(crate) grad: GradSlot,
    /// Whether gradients flow to (and through) this tensor.
    pub requires_grad: bool,
    pub(crate) node: Option<Arc<Node>>,
}

impl Tensor {
    // ========================================================================
    // CREATION
    // ========================================================================

    /// Create a tensor from row-major data.
    ///
    /// # Panics
    /// Panics when `data.len()` differs from the product of `shape`.
    pub fn from_data(data: &[f32], shape: &[usize], device: &Device) -> Self {
        let size: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            size,
            "data length {} does not match shape {:?}",
            data.len(),
            shape
        );
        let storage = match device.gpu_ctx() {
            Some(ctx) => Storage::Gpu(upload(ctx, data)),
            None => Storage::Cpu(RwLock::new(data.to_vec())),
        };
        Self::from_storage(storage, shape.to_vec(), device)
    }

    /// Create a tensor that takes ownership of `data`.
    pub fn from_vec(data: Vec<f32>, shape: &[usize], device: &Device) -> Self {
        if device.is_cpu() {
            let size: usize = shape.iter().product();
            assert_eq!(
                data.len(),
                size,
                "data length {} does not match shape {:?}",
                data.len(),
                shape
            );
            Self::from_storage(Storage::Cpu(RwLock::new(data)), shape.to_vec(), device)
        } else {
            Self::from_data(&data, shape, device)
        }
    }

    /// A 0-dimensional tensor holding `value`.
    pub fn scalar(value: f32, device: &Device) -> Self {
        Self::from_data(&[value], &[], device)
    }

    pub(crate) fn from_storage(storage: Storage, shape: Vec<usize>, device: &Device) -> Self {
        Self {
            storage: Arc::new(storage),
            shape,
            device: device.clone(),
            grad: Arc::new(RwLock::new(None)),
            requires_grad: false,
            node: None,
        }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &[usize], device: &Device) -> Self {
        Self::full(shape, 0.0, device)
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: &[usize], device: &Device) -> Self {
        Self::full(shape, 1.0, device)
    }

    /// Create a tensor filled with a constant value.
    pub fn full(shape: &[usize], value: f32, device: &Device) -> Self {
        let size: usize = shape.iter().product();
        Self::from_vec(vec![value; size], shape, device)
    }

    /// Zeros with the shape and device of `self`.
    pub fn zeros_like(&self) -> Self {
        Self::zeros(&self.shape, &self.device)
    }

    /// Ones with the shape and device of `self`.
    pub fn ones_like(&self) -> Self {
        Self::ones(&self.shape, &self.device)
    }

    /// A constant with the shape and device of `self`.
    pub fn full_like(&self, value: f32) -> Self {
        Self::full(&self.shape, value, &self.device)
    }

    /// Create a tensor with random normal values (mean 0, std 1).
    pub fn randn(shape: &[usize], device: &Device) -> Self {
        Self::randn_with(shape, &mut rand::thread_rng(), device)
    }

    /// Like [`Tensor::randn`], drawing from `rng` (for reproducible runs).
    pub fn randn_with(shape: &[usize], rng: &mut impl rand::Rng, device: &Device) -> Self {
        let size: usize = shape.iter().product();
        // Box-Muller; 1 - u keeps the log argument in (0, 1].
        let data: Vec<f32> = (0..size)
            .map(|_| {
                let u1: f32 = 1.0 - rng.gen::<f32>();
                let u2: f32 = rng.gen();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();
        Self::from_vec(data, shape, device)
    }

    /// Create a tensor with random uniform values in [0, 1).
    pub fn rand(shape: &[usize], device: &Device) -> Self {
        Self::uniform(shape, 0.0, 1.0, device)
    }

    /// Create a tensor with random uniform values in [low, high).
    pub fn uniform(shape: &[usize], low: f32, high: f32, device: &Device) -> Self {
        Self::uniform_with(shape, low, high, &mut rand::thread_rng(), device)
    }

    /// Like [`Tensor::uniform`], drawing from `rng`.
    pub fn uniform_with(
        shape: &[usize],
        low: f32,
        high: f32,
        rng: &mut impl rand::Rng,
        device: &Device,
    ) -> Self {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size)
            .map(|_| rng.gen::<f32>() * (high - low) + low)
            .collect();
        Self::from_vec(data, shape, device)
    }

    /// Create an `n x n` identity matrix.
    pub fn eye(n: usize, device: &Device) -> Self {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::from_vec(data, &[n, n], device)
    }

    /// Create a 1-D tensor with values from `start` (inclusive) to `end`
    /// (exclusive) spaced by `step`.
    ///
    /// # Panics
    /// Panics when `step` is not positive.
    pub fn arange(start: f32, end: f32, step: f32, device: &Device) -> Self {
        assert!(step > 0.0, "arange step must be positive");
        let len = ((end - start) / step).ceil().max(0.0) as usize;
        let data: Vec<f32> = (0..len).map(|i| start + i as f32 * step).collect();
        Self::from_vec(data, &[len], device)
    }

    /// Create a 1-D tensor of `steps` values evenly spaced over `[start, end]`.
    pub fn linspace(start: f32, end: f32, steps: usize, device: &Device) -> Self {
        let data: Vec<f32> = match steps {
            0 => vec![],
            1 => vec![start],
            _ => (0..steps)
                .map(|i| start + (end - start) * i as f32 / (steps - 1) as f32)
                .collect(),
        };
        Self::from_vec(data, &[steps], device)
    }

    // ========================================================================
    // PROPERTIES
    // ========================================================================

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Shape as a slice.
    pub fn size(&self) -> &[usize] {
        &self.shape
    }

    /// The device this tensor lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// True when the tensor holds exactly one element.
    pub fn is_scalar(&self) -> bool {
        self.numel() == 1
    }

    /// True for a tensor not produced by a tracked op (a graph leaf).
    pub fn is_leaf(&self) -> bool {
        self.node.is_none()
    }

    /// Set the `requires_grad` flag.
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// Return `self` with `requires_grad` set, as a fresh leaf.
    pub fn requires_grad_(mut self) -> Self {
        self.requires_grad = true;
        self.node = None;
        self
    }

    /// A tensor sharing this one's values but cut from the autograd graph.
    pub fn detach(&self) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            shape: self.shape.clone(),
            device: self.device.clone(),
            grad: Arc::new(RwLock::new(None)),
            requires_grad: false,
            node: None,
        }
    }

    /// Same values with a new shape, sharing storage (no autograd).
    pub(crate) fn view_raw(&self, shape: &[usize]) -> Self {
        let mut t = self.detach();
        t.shape = shape.to_vec();
        t
    }

    /// Identity of the autograd variable (shared by clones).
    pub(crate) fn id(&self) -> usize {
        Arc::as_ptr(&self.grad) as *const () as usize
    }

    // ========================================================================
    // DATA ACCESS
    // ========================================================================

    /// Copy the values to a `Vec<f32>` (row-major).
    pub fn to_vec(&self) -> Vec<f32> {
        match &*self.storage {
            Storage::Cpu(data) => data.read().expect("tensor lock poisoned").clone(),
            Storage::Gpu(buffer) => {
                let ctx = self.device.gpu_ctx().expect("GPU storage on a CPU device");
                pollster::block_on(download(ctx, buffer, self.numel()))
            }
        }
    }

    /// Async version of [`Tensor::to_vec`].
    pub async fn to_vec_async(&self) -> Vec<f32> {
        match &*self.storage {
            Storage::Cpu(data) => data.read().expect("tensor lock poisoned").clone(),
            Storage::Gpu(buffer) => {
                let ctx = self.device.gpu_ctx().expect("GPU storage on a CPU device");
                download(ctx, buffer, self.numel()).await
            }
        }
    }

    /// The value of a one-element tensor.
    ///
    /// # Panics
    /// Panics when the tensor has more than one element.
    pub fn item(&self) -> f32 {
        assert!(
            self.is_scalar(),
            "item() needs a one-element tensor, got shape {:?}",
            self.shape
        );
        self.to_vec()[0]
    }

    /// Overwrite the values in place (no autograd).
    ///
    /// Every tensor sharing this storage (clones, reshaped views) sees the
    /// new values.
    pub fn copy_from_slice(&self, data: &[f32]) {
        assert_eq!(
            data.len(),
            self.numel(),
            "data length must match tensor size"
        );
        match &*self.storage {
            Storage::Cpu(values) => values
                .write()
                .expect("tensor lock poisoned")
                .copy_from_slice(data),
            Storage::Gpu(buffer) => {
                if let Some(ctx) = self.device.gpu_ctx() {
                    ctx.write_buffer(buffer, bytemuck::cast_slice(data));
                }
            }
        }
    }

    /// Overwrite the values in place with those of `src` (no autograd).
    ///
    /// Used by optimizers to update parameters without breaking the
    /// parameter's identity.
    pub fn assign(&self, src: &Tensor) {
        assert_eq!(self.shape, src.shape, "assign needs matching shapes");
        let src = src.to_device(&self.device);
        match (&*self.storage, &*src.storage) {
            (Storage::Gpu(dst), Storage::Gpu(from)) => {
                let ctx = self.device.gpu_ctx().expect("GPU storage on a CPU device");
                ctx.copy(from, dst, (self.numel() * 4) as u64);
            }
            _ => self.copy_from_slice(&src.to_vec()),
        }
    }

    /// Copy to `device` (differentiable: gradients flow back to the source).
    pub fn to_device(&self, device: &Device) -> Tensor {
        if self.device.same_as(device) {
            return self.clone();
        }
        let out = Tensor::from_data(&self.to_vec(), &self.shape, device);
        let source = self.device.clone();
        out.with_node("to_device", vec![self.clone()], move |g| {
            vec![Some(g.to_device(&source))]
        })
    }

    // ========================================================================
    // GRADIENTS
    // ========================================================================

    /// The accumulated gradient, if any.
    pub fn grad(&self) -> Option<Tensor> {
        self.grad.read().expect("grad lock poisoned").clone()
    }

    /// Replace the accumulated gradient.
    pub fn set_grad(&self, grad: Option<Tensor>) {
        if let Some(g) = &grad {
            assert_eq!(g.shape, self.shape, "gradient shape must match tensor");
        }
        *self.grad.write().expect("grad lock poisoned") = grad;
    }

    /// Clear the accumulated gradient.
    pub fn zero_grad(&self) {
        self.set_grad(None);
    }
}

/// Upload `data` into a new storage buffer (at least 4 bytes: wgpu rejects
/// empty bindings).
pub(crate) fn upload(ctx: &GpuContext, data: &[f32]) -> Buffer {
    let contents: &[u8] = if data.is_empty() {
        &[0u8; 4]
    } else {
        bytemuck::cast_slice(data)
    };
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tensor"),
            contents,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        })
}

/// An uninitialised storage buffer for `n` floats.
pub(crate) fn alloc(ctx: &GpuContext, n: usize) -> Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Tensor"),
        size: (n.max(1) * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

async fn download(ctx: &GpuContext, buffer: &Buffer, n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let size = (n * 4) as u64;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging"),
        size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.copy(buffer, &staging, size);
    ctx.flush();

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    ctx.wait_for(&rx).expect("failed to map GPU buffer");

    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("requires_grad", &self.requires_grad)
            .field("device", &self.device.name())
            .finish()
    }
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.to_vec();
        let fmt_row = |row: &[f32]| {
            row.iter()
                .map(|x| format!("{x:.4}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        match self.ndim() {
            0 => write!(f, "tensor({:.4})", data[0]),
            1 => write!(f, "tensor([{}])", fmt_row(&data)),
            2 => {
                writeln!(f, "tensor([")?;
                for row in data.chunks(self.shape[1].max(1)) {
                    writeln!(f, "  [{}],", fmt_row(row))?;
                }
                write!(f, "])")
            }
            _ => write!(f, "tensor(shape={:?})", self.shape),
        }
    }
}
