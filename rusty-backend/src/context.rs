//! GPU device setup.

use crate::pool::BufferPool;
use std::sync::Arc;
use wgpu::{Adapter, Device, Instance, Queue};

#[derive(Debug)]
pub struct WgpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub pool: Arc<BufferPool>,
}

impl WgpuContext {
    pub async fn new() -> Self {
        Self::new_with_options(wgpu::PowerPreference::HighPerformance, false).await
    }

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
                    label: Some("Rusty Device"),
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

    pub fn adapter_info(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?})", info.name, info.backend)
    }
}
