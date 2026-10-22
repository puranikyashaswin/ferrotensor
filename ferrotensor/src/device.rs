//! Device abstraction for GPU/CPU computation.
//!
//! A [`Device`] is either the CPU or a wgpu GPU (Metal, Vulkan, DX12). Every
//! tensor op runs on both: the GPU path dispatches WGSL kernels, the CPU path
//! runs the reference implementation in `tensor::cpu`. Tensors on the CPU
//! device need no GPU at all, which is what the test suite and CI use.

use std::num::NonZeroU64;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use wgpu::{
    Adapter, BindingResource, Buffer, BufferBinding, BufferUsages, CommandEncoder,
    Device as WgpuDevice, Queue,
};

use crate::backend::{ComputeEngine, Dispatch};

/// Ops recorded before the pending commands are submitted anyway, so the GPU
/// starts on long op sequences that never read back to the host.
const MAX_PENDING_OPS: usize = 256;

/// Size of the buffer holding the small per-op arguments ([`Arg::Params`],
/// [`Arg::Meta`]) of one batch.
const ARGS_BYTES: usize = 256 * 1024;

/// How long a wait polls the GPU before blocking. wgpu's blocking wait on
/// Metal sleeps in 1 ms steps, which adds up to a millisecond to every read
/// back of a short batch.
const SPIN_WAIT: Duration = Duration::from_millis(10);

/// One binding of a kernel dispatch.
pub(crate) enum Arg<'a> {
    /// A storage buffer, bound whole.
    Buffer(&'a Buffer),
    /// A 16-byte uniform.
    Params([u32; 4]),
    /// A read-only `u32` storage array.
    Meta(&'a [u32]),
}

/// GPU resources shared by every tensor on one GPU device.
///
/// Ops are recorded and submitted together at the next sync point (a read
/// back, a host write, or every [`MAX_PENDING_OPS`] ops) instead of one
/// submission per op. Consecutive dispatches share one compute pass, and
/// their uniforms and shape metadata share one buffer that is written once
/// per batch. The queue runs everything in recording order, so results are
/// the same as submitting each op on its own.
pub(crate) struct GpuContext {
    pub(crate) adapter: Adapter,
    pub(crate) device: WgpuDevice,
    pub(crate) queue: Queue,
    pub(crate) engine: ComputeEngine,
    /// Holds [`Pending::args`] while the batch runs.
    args: Buffer,
    /// Offset alignment of uniform and storage bindings.
    align: usize,
    pending: Mutex<Pending>,
}

/// Work recorded since the last submission.
#[derive(Default)]
struct Pending {
    encoder: Option<CommandEncoder>,
    /// Dispatches not yet encoded; they go into one compute pass.
    dispatches: Vec<Dispatch>,
    /// Contents of [`GpuContext::args`] for this batch.
    args: Vec<u8>,
    ops: usize,
}

impl GpuContext {
    fn new(adapter: Adapter, device: WgpuDevice, queue: Queue) -> Self {
        let engine = ComputeEngine::new(&device);
        let limits = device.limits();
        let align = limits
            .min_uniform_buffer_offset_alignment
            .max(limits.min_storage_buffer_offset_alignment)
            .max(4) as usize;
        let args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Op arguments"),
            size: ARGS_BYTES as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            adapter,
            device,
            queue,
            engine,
            args,
            align,
            pending: Mutex::default(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Pending> {
        self.pending.lock().expect("GPU command lock poisoned")
    }

    /// Record a dispatch of `kernel` with `args[i]` bound to `@binding(i)`.
    pub(crate) fn dispatch(
        &self,
        kernel: &'static str,
        args: &[Arg<'_>],
        workgroups: (u32, u32, u32),
    ) {
        let padded = |bytes: usize| bytes.div_ceil(self.align) * self.align;
        let needed: usize = args
            .iter()
            .map(|arg| match arg {
                Arg::Buffer(_) => 0,
                Arg::Params(_) => padded(16),
                Arg::Meta(words) => padded(words.len().max(1) * 4),
            })
            .sum();
        assert!(
            needed <= ARGS_BYTES,
            "kernel arguments exceed {ARGS_BYTES} bytes"
        );

        let mut pending = self.lock();
        if pending.args.len() + needed > ARGS_BYTES {
            self.submit(&mut pending);
        }
        let staged = &mut pending.args;
        let resources = args
            .iter()
            .map(|arg| {
                let bytes: &[u8] = match arg {
                    Arg::Buffer(buffer) => return buffer.as_entire_binding(),
                    Arg::Params(words) => bytemuck::cast_slice(words),
                    Arg::Meta([]) => &[0; 4],
                    Arg::Meta(words) => bytemuck::cast_slice(words),
                };
                let offset = staged.len();
                staged.extend_from_slice(bytes);
                staged.resize(padded(staged.len()), 0);
                BindingResource::Buffer(BufferBinding {
                    buffer: &self.args,
                    offset: offset as u64,
                    size: NonZeroU64::new(bytes.len() as u64),
                })
            })
            .collect();
        let dispatch = self
            .engine
            .dispatch(&self.device, kernel, resources, workgroups);
        pending.dispatches.push(dispatch);
        self.count_op(&mut pending);
    }

    /// Record a copy of the first `bytes` bytes of `src` into `dst`.
    pub(crate) fn copy(&self, src: &Buffer, dst: &Buffer, bytes: u64) {
        let mut pending = self.lock();
        self.encoder(&mut pending)
            .copy_buffer_to_buffer(src, 0, dst, 0, bytes);
        self.count_op(&mut pending);
    }

    /// Submit every recorded op.
    pub(crate) fn flush(&self) {
        self.submit(&mut self.lock());
    }

    /// Overwrite the start of `buffer` with `data`.
    ///
    /// The queue applies writes before the next submission's commands, so
    /// the ops recorded so far are submitted first.
    pub(crate) fn write_buffer(&self, buffer: &Buffer, data: &[u8]) {
        let mut pending = self.lock();
        self.submit(&mut pending);
        self.queue.write_buffer(buffer, 0, data);
    }

    /// Poll the device until a GPU callback sends on `rx`: busy-polling for
    /// up to [`SPIN_WAIT`], then blocking.
    pub(crate) fn wait_for<T>(&self, rx: &Receiver<T>) -> T {
        let start = Instant::now();
        loop {
            let blocking = start.elapsed() >= SPIN_WAIT;
            self.device.poll(if blocking {
                wgpu::Maintain::Wait
            } else {
                wgpu::Maintain::Poll
            });
            match rx.try_recv() {
                Ok(value) => return value,
                Err(TryRecvError::Empty) if !blocking => std::thread::yield_now(),
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => panic!("GPU callback dropped"),
            }
        }
    }

    fn count_op(&self, pending: &mut Pending) {
        pending.ops += 1;
        if pending.ops >= MAX_PENDING_OPS {
            self.submit(pending);
        }
    }

    /// The batch's encoder, with the pending dispatches encoded into it.
    fn encoder<'p>(&self, pending: &'p mut Pending) -> &'p mut CommandEncoder {
        let encoder = pending.encoder.get_or_insert_with(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Ferrotensor ops"),
                })
        });
        if !pending.dispatches.is_empty() {
            self.engine.encode(encoder, &pending.dispatches);
            pending.dispatches.clear();
        }
        encoder
    }

    fn submit(&self, pending: &mut Pending) {
        if pending.ops == 0 {
            return;
        }
        self.encoder(pending);
        if !pending.args.is_empty() {
            self.queue.write_buffer(&self.args, 0, &pending.args);
            pending.args.clear();
        }
        if let Some(encoder) = pending.encoder.take() {
            self.queue.submit(std::iter::once(encoder.finish()));
        }
        pending.ops = 0;
    }
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
        Some(Self {
            kind: Kind::Gpu(Arc::new(GpuContext::new(adapter, device, queue))),
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

    /// Wait until every op issued on this device has finished.
    ///
    /// GPU ops are queued and run asynchronously; reading a tensor's values
    /// waits for the ops it depends on, so this is only needed for timing.
    /// Does nothing on the CPU.
    pub fn synchronize(&self) {
        if let Some(ctx) = self.gpu_ctx() {
            ctx.flush();
            let (tx, rx) = std::sync::mpsc::channel();
            ctx.queue.on_submitted_work_done(move || {
                let _ = tx.send(());
            });
            ctx.wait_for(&rx);
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
