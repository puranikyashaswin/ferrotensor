//! Device abstraction for GPU/CPU computation.
//!
//! A [`Device`] is either the CPU or a wgpu GPU (Metal, Vulkan, DX12). Every
//! tensor op runs on both: the GPU path dispatches WGSL kernels, the CPU path
//! runs the reference implementation in `tensor::cpu`. Tensors on the CPU
//! device need no GPU at all, which is what the test suite and CI use.

use std::sync::Arc;
use wgpu::{Adapter, Device as WgpuDevice, Queue};

use crate::backend::ComputeEngine;

/// GPU resources shared by every tensor on one GPU device.
pub(crate) struct GpuContext {
    pub(crate) adapter: Adapter,
    pub(crate) device: WgpuDevice,
    pub(crate) queue: Queue,
    pub(crate) engine: ComputeEngine,
}

#[derive(Clone)]
enum Kind {
    Cpu,
    Gpu(Arc<GpuContext>),
}

/// A compute device: the CPU or a GPU.
///
/// Cloning is cheap; clones refer to the same GPU context.
///
/// # Example
/// ```rust,no_run
/// use ferrotensor::Device;
///
/// // A GPU if one is available, else the CPU.
/// let device = Device::best();
/// println!("Using: {} ({})", device.name(), device.backend());
///
/// // Always available, no GPU needed.
/// let cpu = Device::cpu();
/// assert!(cpu.is_cpu());
/// ```
#[derive(Clone)]
pub struct Device {
    kind: Kind,
}

impl Device {
    /// Create a GPU device with the best available backend.
    ///
    /// On macOS this is Metal; on Windows/Linux, Vulkan or DX12.
    ///
    /// # Panics
    /// Panics when no GPU adapter is available. Use [`Device::gpu`] to get an
    /// `Option`, or [`Device::best`] to fall back to the CPU.
    pub fn new() -> Self {
        Self::gpu().expect("no GPU adapter available; use Device::best() or Device::cpu()")
    }

    /// The CPU device.
    pub fn cpu() -> Self {
        Self { kind: Kind::Cpu }
    }

    /// A GPU device, or `None` when no adapter or device can be created.
    pub fn gpu() -> Option<Self> {
        pollster::block_on(Self::gpu_async())
    }

    /// Async version of [`Device::gpu`].
    pub async fn gpu_async() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Ferrotensor Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                },
                None,
            )
            .await
            .ok()?;
        let engine = ComputeEngine::new(&device);
        Some(Self {
            kind: Kind::Gpu(Arc::new(GpuContext {
                adapter,
                device,
                queue,
                engine,
            })),
        })
    }

    /// Async version of [`Device::new`].
    ///
    /// # Panics
    /// Panics when no GPU is available.
    pub async fn new_async() -> Self {
        Self::gpu_async()
            .await
            .expect("no GPU adapter available; use Device::best() or Device::cpu()")
    }

    /// A GPU if one is available, otherwise the CPU.
    pub fn best() -> Self {
        Self::gpu().unwrap_or_else(Self::cpu)
    }

    /// True for the CPU device.
    pub fn is_cpu(&self) -> bool {
        matches!(self.kind, Kind::Cpu)
    }

    /// True for a GPU device.
    pub fn is_gpu(&self) -> bool {
        !self.is_cpu()
    }

    /// Name of the device ("CPU" or the GPU adapter name).
    pub fn name(&self) -> String {
        match &self.kind {
            Kind::Cpu => "CPU".to_string(),
            Kind::Gpu(ctx) => ctx.adapter.get_info().name,
        }
    }

    /// Backend name: "Cpu", or the wgpu backend (Metal, Vulkan, Dx12, ...).
    pub fn backend(&self) -> String {
        match &self.kind {
            Kind::Cpu => "Cpu".to_string(),
            Kind::Gpu(ctx) => format!("{:?}", ctx.adapter.get_info().backend),
        }
    }

    /// Whether the device can run training (every device can).
    pub fn supports_training(&self) -> bool {
        true
    }

    /// True when both handles refer to the same device.
    pub fn same_as(&self, other: &Device) -> bool {
        match (&self.kind, &other.kind) {
            (Kind::Cpu, Kind::Cpu) => true,
            (Kind::Gpu(a), Kind::Gpu(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// The GPU context, or `None` on the CPU.
    pub(crate) fn gpu_ctx(&self) -> Option<&GpuContext> {
        match &self.kind {
            Kind::Cpu => None,
            Kind::Gpu(ctx) => Some(ctx),
        }
    }
}

impl Default for Device {
    /// Same as [`Device::best`].
    fn default() -> Self {
        Self::best()
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("name", &self.name())
            .field("backend", &self.backend())
            .finish()
    }
}
