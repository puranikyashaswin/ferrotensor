//! GPU device setup.

use crate::pool::BufferPool;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use wgpu::{Adapter, CommandEncoder, Device, Instance, Queue};

/// Recorded GPU work is submitted once this many ops are pending, so the GPU
/// starts on long sequences without waiting for a sync point.
const MAX_PENDING_OPS: u32 = 128;

/// Uploaded bytes after which [`WgpuContext::note_upload`] waits for the
/// staging copies to finish so wgpu can free them.
pub const UPLOAD_RELEASE_BYTES: u64 = 256 << 20;

/// A GPU device, its queue and a buffer pool shared by tensors.
///
/// Kernel dispatches and copies from [`ComputeEngine`](crate::ComputeEngine)
/// are recorded into one pending command encoder and submitted together:
/// every 128 ops, and whenever a result is read
/// ([`UnifiedTensor::to_vec`](crate::UnifiedTensor::to_vec)) or a buffer is
/// written ([`UnifiedTensor::write_data`](crate::UnifiedTensor::write_data)).
/// Call [`flush`](Self::flush) before submitting work of your own on
/// `queue`, so it runs after the recorded ops, and
/// [`synchronize`](Self::synchronize) to wait for the GPU.
///
/// Uploads (creating a tensor from host data, or writing one) go through a
/// host-side staging buffer that wgpu frees only after a later submission
/// completes. The context counts uploaded bytes and, every
/// [`UPLOAD_RELEASE_BYTES`], submits and waits so those buffers are freed;
/// without that, loading a model would hold a staging copy of every weight.
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
    /// Bytes uploaded since staging buffers were last released.
    staged: AtomicU64,
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
            staged: AtomicU64::new(0),
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

    /// Submits all recorded work and pending uploads and blocks until the
    /// GPU has finished them. Staging buffers of finished uploads are freed.
    pub fn synchronize(&self) {
        self.staged.store(0, Ordering::Relaxed);
        self.flush();
        // An empty submission carries wgpu's pending buffer writes.
        self.queue.submit(std::iter::empty());
        let done = Arc::new(AtomicBool::new(false));
        let flag = done.clone();
        self.queue
            .on_submitted_work_done(move || flag.store(true, Ordering::Release));
        self.poll_until(|| done.load(Ordering::Acquire));
    }

    /// Records that `bytes` were uploaded through a staging buffer (by
    /// `create_buffer_init` or `queue.write_buffer`). Once
    /// [`UPLOAD_RELEASE_BYTES`] have accumulated it
    /// [`synchronize`](Self::synchronize)s, so host memory held by staging
    /// stays bounded. [`UnifiedTensor`](crate::UnifiedTensor) calls this for
    /// its uploads; call it after uploading through `queue` yourself.
    pub fn note_upload(&self, bytes: u64) {
        let total = self.staged.fetch_add(bytes, Ordering::Relaxed) + bytes;
        if total >= UPLOAD_RELEASE_BYTES {
            self.synchronize();
        }
    }

    /// Polls the device until `done` returns true.
    ///
    /// Never blocks in `Maintain::Wait`: wgpu-core 0.19 gives up waiting
    /// there after 5 s and then frees the still-running submission's
    /// resources as if it had finished, which crashes drivers that are still
    /// reading them (seen with long batches on lavapipe).
    pub(crate) fn poll_until(&self, done: impl Fn() -> bool) {
        let mut polls = 0u32;
        loop {
            self.device.poll(wgpu::Maintain::Poll);
            if done() {
                // Callers wait for the last submitted work, so that poll let
                // wgpu destroy every buffer dropped before it.
                self.memory().gpu_reclaimed();
                return;
            }
            if polls < 100 {
                std::thread::yield_now();
            } else {
                std::thread::sleep(Duration::from_micros(50));
            }
            polls += 1;
        }
    }

    fn submit(&self, pending: &mut Pending) {
        pending.ops = 0;
        if let Some(encoder) = pending.encoder.take() {
            self.queue.submit(Some(encoder.finish()));
        }
    }

    /// Byte counters of this context's tensors: live, peak, idle in the
    /// pool, and by [`MemoryCategory`](crate::MemoryCategory).
    pub fn memory(&self) -> &Arc<crate::MemoryTracker> {
        self.pool.tracker()
    }

    /// Adapter name and backend, e.g. `"Apple M2 (Metal)"`.
    pub fn adapter_info(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?})", info.name, info.backend)
    }
}
