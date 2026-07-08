//! Core Tensor Implementation
//!
//! The Tensor struct is the fundamental data type for all operations.

use std::sync::Arc;
use wgpu::{Buffer, BufferUsages};

use super::autograd::GradCell;
use crate::Device;

/// A multi-dimensional array stored on GPU with automatic differentiation support.
///
/// # Example
/// ```rust,no_run
/// use rusty::prelude::*;
///
/// let device = Device::new();
/// let a = Tensor::randn(&[2, 3], &device);
/// let b = Tensor::ones(&[2, 3], &device);
/// let c = a.add(&b);
/// ```
#[derive(Clone)]
pub struct Tensor {
    /// GPU buffer containing tensor data (f32)
    pub(crate) buffer: Arc<Buffer>,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Device this tensor lives on
    pub(crate) device: Device,
    /// Gradient storage for autograd
    pub(crate) grad: Arc<GradCell>,
    /// Whether this tensor requires gradient computation
    pub requires_grad: bool,
    /// Creator node for backward pass
    pub(crate) creator: Option<Arc<dyn super::autograd::AutogradNode>>,
}

impl Tensor {
    // ========================================================================
    // CREATION METHODS
    // ========================================================================

    /// Create a tensor from raw data.
    pub fn from_data(data: &[f32], shape: &[usize], device: &Device) -> Self {
        let size: usize = shape.iter().product();
        assert_eq!(data.len(), size, "Data length must match shape");

        let buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tensor Buffer"),
                contents: bytemuck::cast_slice(data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            });

        Self {
            buffer: Arc::new(buffer),
            shape: shape.to_vec(),
            device: device.clone(),
            grad: Arc::new(GradCell::new()),
            requires_grad: false,
            creator: None,
        }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &[usize], device: &Device) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![0.0f32; size];
        Self::from_data(&data, shape, device)
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: &[usize], device: &Device) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![1.0f32; size];
        Self::from_data(&data, shape, device)
    }

    /// Create a tensor filled with a constant value.
    pub fn full(shape: &[usize], value: f32, device: &Device) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![value; size];
        Self::from_data(&data, shape, device)
    }

    /// Create a tensor with random normal values (mean=0, std=1).
    pub fn randn(shape: &[usize], device: &Device) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let size: usize = shape.iter().product();

        // Box-Muller transform for normal distribution
        let data: Vec<f32> = (0..size)
            .map(|_| {
                let u1: f32 = rng.gen();
                let u2: f32 = rng.gen();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();

        Self::from_data(&data, shape, device)
    }

    /// Create a tensor with random uniform values in [0, 1).
    pub fn rand(shape: &[usize], device: &Device) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size).map(|_| rng.gen()).collect();
        Self::from_data(&data, shape, device)
    }

    /// Create a tensor with random uniform values in [low, high).
    pub fn uniform(shape: &[usize], low: f32, high: f32, device: &Device) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (0..size)
            .map(|_| rng.gen::<f32>() * (high - low) + low)
            .collect();
        Self::from_data(&data, shape, device)
    }

    /// Create an identity matrix.
    pub fn eye(n: usize, device: &Device) -> Self {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::from_data(&data, &[n, n], device)
    }

    /// Create a 1D tensor with evenly spaced values.
    pub fn arange(start: f32, end: f32, step: f32, device: &Device) -> Self {
        let mut data = Vec::new();
        let mut val = start;
        while val < end {
            data.push(val);
            val += step;
        }
        let len = data.len();
        Self::from_data(&data, &[len], device)
    }

    // ========================================================================
    // PROPERTIES
    // ========================================================================

    /// Return the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Return the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Return the shape as a slice.
    pub fn size(&self) -> &[usize] {
        &self.shape
    }

    /// Check if this tensor is a scalar (0-dimensional or single element).
    pub fn is_scalar(&self) -> bool {
        self.numel() == 1
    }

    /// Set requires_grad flag.
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// Create a new tensor with requires_grad set to true.
    pub fn requires_grad_(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    /// Detach tensor from computation graph (no gradient tracking).
    pub fn detach(&self) -> Self {
        Self {
            buffer: Arc::clone(&self.buffer),
            shape: self.shape.clone(),
            device: self.device.clone(),
            grad: Arc::new(GradCell::new()),
            requires_grad: false,
            creator: None,
        }
    }

    // ========================================================================
    // DATA ACCESS
    // ========================================================================

    /// Copy tensor data to CPU as a Vec<f32>.
    pub fn to_vec(&self) -> Vec<f32> {
        pollster::block_on(self.to_vec_async())
    }

    /// Async version of to_vec.
    pub async fn to_vec_async(&self) -> Vec<f32> {
        let size = self.numel() * std::mem::size_of::<f32>();

        // Create staging buffer
        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy to staging
        let mut encoder =
            self.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Encoder"),
                });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, size as u64);
        self.device.queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        result
    }

    /// Get a single scalar value (for scalar tensors).
    pub fn item(&self) -> f32 {
        assert!(self.is_scalar(), "item() only works on scalar tensors");
        self.to_vec()[0]
    }

    /// Write data to this tensor's buffer.
    pub fn copy_from_slice(&self, data: &[f32]) {
        assert_eq!(
            data.len(),
            self.numel(),
            "Data length must match tensor size"
        );
        self.device
            .queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }

    // ========================================================================
    // GRADIENT METHODS
    // ========================================================================

    /// Get the gradient if it exists.
    pub fn grad(&self) -> Option<Tensor> {
        self.grad.get().map(|buffer| Tensor {
            buffer: buffer.clone(),
            shape: self.shape.clone(),
            device: self.device.clone(),
            grad: Arc::new(GradCell::new()),
            requires_grad: false,
            creator: None,
        })
    }

    /// Zero out the gradient.
    pub fn zero_grad(&self) {
        self.grad.clear();
    }

    /// Backward pass - compute gradients for all tensors in the graph.
    pub fn backward(&self) {
        if self.creator.is_none() {
            return;
        }

        // Initialize gradient to ones for the output
        let ones = vec![1.0f32; self.numel()];
        self.grad.set(&self.device, &ones, &self.shape);

        // Topological sort
        let mut sorted = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.build_topo(&mut sorted, &mut visited);

        // Backward pass
        for tensor in sorted.into_iter().rev() {
            if let Some(ref creator) = tensor.creator {
                if let Some(grad_buffer) = tensor.grad.get() {
                    let grad_tensor = Tensor {
                        buffer: grad_buffer.clone(),
                        shape: tensor.shape.clone(),
                        device: tensor.device.clone(),
                        grad: Arc::new(GradCell::new()),
                        requires_grad: false,
                        creator: None,
                    };
                    creator.backward(&grad_tensor, &tensor.device);
                }
            }
        }
    }

    fn build_topo(&self, sorted: &mut Vec<Tensor>, visited: &mut std::collections::HashSet<usize>) {
        let ptr = Arc::as_ptr(&self.buffer) as usize;
        if visited.contains(&ptr) {
            return;
        }
        visited.insert(ptr);

        if let Some(ref creator) = self.creator {
            for input in creator.inputs() {
                input.build_topo(sorted, visited);
            }
        }
        sorted.push(self.clone());
    }
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
        if self.ndim() == 1 {
            write!(
                f,
                "tensor([{:.4}])",
                data.iter()
                    .map(|x| format!("{:.4}", x))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else if self.ndim() == 2 {
            writeln!(f, "tensor([")?;
            for i in 0..self.shape[0] {
                let start = i * self.shape[1];
                let end = start + self.shape[1];
                let row: String = data[start..end]
                    .iter()
                    .map(|x| format!("{:8.4}", x))
                    .collect::<Vec<_>>()
                    .join(", ");
                writeln!(f, "  [{}]", row)?;
            }
            write!(f, "])")
        } else {
            write!(f, "tensor(shape={:?})", self.shape)
        }
    }
}

use wgpu::util::DeviceExt;
