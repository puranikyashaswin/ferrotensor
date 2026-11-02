//! GPU device setup.

use crate::pool::BufferPool;
use std::sync::{Arc, Mutex};
use wgpu::{Adapter, CommandEncoder, Device, Instance, Queue};

/// Recorded GPU work is submitted once this many ops are pending, so the GPU
/// starts on long sequences without waiting for a sync point.
const MAX_PENDING_OPS: u32 = 128;

/// A GPU device, its queue and a buffer pool shared by tensors.
///
/// Kernel dispatches and copies from [`ComputeEngine`](crate::ComputeEngine)
/// are recorded into one pending command encoder and submitted together:
/// every 128 ops, and whenever a result is read
/// ([`UnifiedTensor::to_vec`](crate::UnifiedTensor::to_vec)) or a buffer is
/// written ([`UnifiedTensor::write_data`](crate::UnifiedTensor::write_data)).
/// Call [`flush`](Self::flush) before submitting work of your own on
/// `queue` or waiting on `device`, so it runs after the recorded ops.
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
    pending: Mutex<Pending>,
}

/// Recorded work not yet submitted.
#[derive(Default)]
struct Pending {
    encoder: Option<CommandEncoder>,
    ops: u32,
}

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending").field("ops", &self.ops).finish()
    }
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
    /// the software fallback adapter. The `WGPU_BACKEND` environment variable
    /// (e.g. `vulkan`) restricts which backends are tried.
    ///
    /// # Panics
    /// Panics if no adapter or device is available.
    pub async fn new_with_options(
        power_preference: wgpu::PowerPreference,
        force_fallback_adapter: bool,
    ) -> Self {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: crate::adapters::backends(),
            ..Default::default()
        });
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
            pending: Mutex::new(Pending::default()),
        }
    }

    /// Records GPU work into the pending command encoder; it runs after
    /// everything recorded before it. Submits once enough ops are pending.
    pub(crate) fn record(&self, f: impl FnOnce(&mut CommandEncoder)) {
        let mut pending = self.pending.lock().unwrap();
        let encoder = pending.encoder.get_or_insert_with(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
        });
        f(encoder);
        pending.ops += 1;
        if pending.ops >= MAX_PENDING_OPS {
            self.submit(&mut pending);
        }
    }

    /// Submits all recorded work to the queue. Reads and writes through
    /// [`UnifiedTensor`](crate::UnifiedTensor) do this themselves.
    pub fn flush(&self) {
        let mut pending = self.pending.lock().unwrap();
        self.submit(&mut pending);
    }

    fn submit(&self, pending: &mut Pending) {
        pending.ops = 0;
        if let Some(encoder) = pending.encoder.take() {
            self.queue.submit(Some(encoder.finish()));
        }
    }

    /// Adapter name and backend, e.g. `"Apple M2 (Metal)"`.
    pub fn adapter_info(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?})", info.name, info.backend)
    }
}
