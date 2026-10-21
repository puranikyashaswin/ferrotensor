//! GPU device setup.

use crate::pool::BufferPool;
use std::sync::Arc;
use wgpu::{Adapter, Device, Instance, Queue};

/// A GPU device, its queue and a buffer pool shared by tensors.
#[derive(Debug)]
pub struct WgpuContext {
    /// The wgpu instance the adapter came from.
    pub instance: Instance,
    /// The physical GPU.
    pub adapter: Adapter,
    /// Logical device; requested with the adapter's full limits.
    pub device: Arc<Device>,
    /// Queue that every kernel dispatch is submitted to.
    pub queue: Arc<Queue>,
    /// Pool that [`UnifiedTensor::empty`](crate::UnifiedTensor::empty) draws from.
    pub pool: Arc<BufferPool>,
}

impl WgpuContext {
    /// Opens the high-performance adapter.
    ///
    /// # Panics
    /// Panics if no adapter or device is available.
    pub async fn new() -> Self {
        Self::new_with_options(wgpu::PowerPreference::HighPerformance, false).await
    }

    /// Opens an adapter with the given power preference, optionally forcing
    /// the software fallback adapter.
    ///
    /// # Panics
    /// Panics if no adapter or device is available.
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
                    label: Some("Ferrotensor Device"),
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

    /// Adapter name and backend, e.g. `"Apple M2 (Metal)"`.
    pub fn adapter_info(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?})", info.name, info.backend)
    }
}
