//! Device abstraction for GPU/CPU computation.
//!
//! The `Device` struct encapsulates all GPU resources (adapter, device, queue)
//! and provides a clean interface for creating tensors on specific hardware.

use std::sync::Arc;
use wgpu::{Adapter, Device as WgpuDevice, Instance, Queue};

use crate::backend::ComputeEngine;

/// Represents a compute device (GPU or CPU).
///
/// # Example
/// ```rust,no_run
/// use rusty::Device;
///
/// // Create device with default backend (Metal on macOS)
/// let device = Device::new();
/// println!("Using: {}", device.name());
/// ```
#[derive(Clone)]
pub struct Device {
    pub(crate) instance: Arc<Instance>,
    pub(crate) adapter: Arc<Adapter>,
    pub(crate) device: Arc<WgpuDevice>,
    pub(crate) queue: Arc<Queue>,
    pub(crate) engine: Arc<ComputeEngine>,
}

impl Device {
    /// Create a new Device with the best available GPU backend.
    ///
    /// On macOS, this will use Metal. On Windows/Linux, this will use Vulkan.
    /// Falls back to CPU if no GPU is available.
    pub fn new() -> Self {
        pollster::block_on(Self::new_async())
    }

    /// Async version of `new()` for use in async contexts.
    pub async fn new_async() -> Self {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find a suitable GPU adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Rusty Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create GPU device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let engine = Arc::new(ComputeEngine::new(&device));

        Self {
            instance: Arc::new(instance),
            adapter: Arc::new(adapter),
            device,
            queue,
            engine,
        }
    }

    /// Get the name of the GPU adapter.
    pub fn name(&self) -> String {
        self.adapter.get_info().name
    }

    /// Get the backend type (Metal, Vulkan, DX12, etc.)
    pub fn backend(&self) -> String {
        format!("{:?}", self.adapter.get_info().backend)
    }

    /// Check if this device supports the required features for training.
    pub fn supports_training(&self) -> bool {
        // All our supported backends support training
        true
    }

    /// Get a reference to the compute engine for kernel dispatch.
    pub(crate) fn engine(&self) -> &ComputeEngine {
        &self.engine
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::new()
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
