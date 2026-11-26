//! Device abstraction for GPU/CPU computation.
//!
//! A [`Device`] is either the CPU or a wgpu GPU (Metal, Vulkan, DX12). Every
//! tensor op runs on both: the GPU path dispatches WGSL kernels, the CPU path
//! runs the reference implementation in `tensor::cpu`. Tensors on the CPU
//! device need no GPU at all, which is what the test suite and CI use.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use wgpu::{
    Adapter, BindingResource, Buffer, BufferBinding, BufferUsages, CommandEncoder,
    Device as WgpuDevice, Queue,
};

use crate::backend::{ComputeEngine, Dispatch, FUSED_STORAGE_BINDINGS};
use crate::tensor::Storage;

/// Ops recorded before the pending commands are submitted anyway, so the GPU
/// starts on long op sequences that never read back to the host.
const MAX_PENDING_OPS: usize = 256;

/// Size of the buffer holding the small per-op arguments ([`Arg::Params`],
/// [`Arg::Meta`]) of one batch.
const ARGS_BYTES: usize = 256 * 1024;

/// How long a wait busy-polls the GPU before sleeping between polls.
/// Sleeping costs up to its length on every read back of a short batch.
const SPIN_WAIT: Duration = Duration::from_millis(10);
/// Sleep between polls once [`SPIN_WAIT`] has passed.
const POLL_SLEEP: Duration = Duration::from_micros(50);

/// Largest total size of freed tensor buffers kept for reuse.
const POOL_BYTES: u64 = 128 << 20;

/// Tensor buffers one `fused` dispatch binds (its last binding is the program).
const FUSED_SLOTS: usize = (FUSED_STORAGE_BINDINGS - 1) as usize;

/// Registers of the `fused` kernel (its `regs` array). A register holds an
/// input of the group or a result some tensor still holds, and each of those
/// has a binding slot, so there are as many registers as slots.
const FUSED_REGS: usize = FUSED_SLOTS;

/// Kind of a `fused` instruction that loads an input into a register (0 and 1
/// are the unary and binary steps).
const FUSED_LOAD: u32 = 2;

/// Last word of a `fused` instruction whose result is not stored.
const NO_STORE: u32 = u32::MAX;

/// One element-wise op for [`GpuContext::elementwise`].
pub(crate) struct ElementwiseStep {
    /// 0 for a `unary_op`, 1 for a `binary_op`.
    pub(crate) kind: u32,
    pub(crate) op: u32,
    /// Scalars of a unary op.
    pub(crate) a: f32,
    pub(crate) b: f32,
    /// One input for a unary op, two for a binary op, each with `n` elements.
    pub(crate) inputs: Vec<Arc<Storage>>,
    /// A new buffer of `n` elements.
    pub(crate) output: Arc<Storage>,
    pub(crate) n: usize,
}

/// The instructions of one `fused` dispatch.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct GroupSummary {
    /// Unary and binary steps.
    pub(crate) steps: usize,
    /// Inputs loaded from their buffers.
    pub(crate) loads: usize,
    /// Results written to their buffers.
    pub(crate) stores: usize,
}

#[cfg(test)]
impl GroupSummary {
    fn of(program: &[u32]) -> Self {
        let words = || program.chunks(8);
        Self {
            steps: words().filter(|w| w[0] != FUSED_LOAD).count(),
            loads: words().filter(|w| w[0] == FUSED_LOAD).count(),
            stores: words().filter(|w| w[7] != NO_STORE).count(),
        }
    }
}

/// What a register of an open fusion group holds.
enum Value {
    /// The group input bound at this slot.
    Input(usize),
    /// The result of the instruction that starts at this program word.
    Result { storage: Arc<Storage>, word: usize },
}

/// Element-wise steps over the same element count, recorded as one `fused`
/// dispatch when the group closes.
///
/// Values stay in the kernel's registers from step to step. A result is
/// written to its buffer only when a tensor still holds it at the close: a
/// result no tensor holds any more (an intermediate in code without
/// autograd) is freed at the next step or at the close, never written, and
/// its buffer goes back to the pool. Holding the storage of every input
/// keeps its buffer out of the pool until the dispatch is recorded.
struct FusionGroup {
    n: usize,
    /// Storage the group reads, by binding slot. The results that are written
    /// take the slots after them.
    inputs: Vec<Arc<Storage>>,
    /// What each register holds; `None` is free.
    regs: [Option<Value>; FUSED_REGS],
    /// Eight words per instruction; see the `fused` kernel.
    program: Vec<u32>,
}

impl FusionGroup {
    fn new(n: usize) -> Self {
        Self {
            n,
            inputs: Vec::new(),
            regs: Default::default(),
            program: Vec::new(),
        }
    }

    fn holds(&self, value: &Value, storage: &Arc<Storage>) -> bool {
        match value {
            Value::Input(slot) => Arc::ptr_eq(&self.inputs[*slot], storage),
            Value::Result { storage: s, .. } => Arc::ptr_eq(s, storage),
        }
    }

    /// The register holding `storage`, if any.
    fn register(&self, storage: &Arc<Storage>) -> Option<usize> {
        self.regs
            .iter()
            .position(|r| r.as_ref().is_some_and(|v| self.holds(v, storage)))
    }

    /// Free the registers of values no tensor holds any more: no later op
    /// can read them. A freed result is never stored, and dropping it here
    /// returns its buffer to the pool. An input keeps its slot, since an
    /// earlier instruction loads it.
    fn release_dead(&mut self) {
        for reg in &mut self.regs {
            let dead = match reg {
                Some(Value::Input(slot)) => Arc::strong_count(&self.inputs[*slot]) == 1,
                Some(Value::Result { storage, .. }) => Arc::strong_count(storage) == 1,
                None => false,
            };
            if dead {
                *reg = None;
            }
        }
    }

    /// Whether `step` fits: a slot for every input and for every result
    /// that may still be stored, and a register for every value. Call
    /// [`FusionGroup::release_dead`] first.
    fn fits(&self, step: &ElementwiseStep) -> bool {
        let mut loads = 0;
        let mut new_slots = 0;
        for (k, s) in step.inputs.iter().enumerate() {
            if self.register(s).is_some() || step.inputs[..k].iter().any(|t| Arc::ptr_eq(t, s)) {
                continue;
            }
            loads += 1;
            if !self.inputs.iter().any(|t| Arc::ptr_eq(t, s)) {
                new_slots += 1;
            }
        }
        let results = self
            .regs
            .iter()
            .filter(|r| matches!(r, Some(Value::Result { .. })))
            .count();
        let used = self.regs.iter().filter(|r| r.is_some()).count();
        self.inputs.len() + new_slots + results < FUSED_SLOTS && used + loads < FUSED_REGS
    }

    /// A free register. [`FusionGroup::fits`] leaves one for each new value.
    #[expect(
        clippy::expect_used,
        reason = "fits() is checked before a step is pushed"
    )]
    fn free_register(&self) -> usize {
        self.regs
            .iter()
            .position(Option::is_none)
            .expect("no free register in a fusion group")
    }

    /// Add `step` when it has the group's element count and fits.
    fn join(&mut self, step: &ElementwiseStep) -> bool {
        self.release_dead();
        let fits = self.n == step.n && self.fits(step);
        if fits {
            self.push(step);
        }
        fits
    }

    fn push(&mut self, step: &ElementwiseStep) {
        let mut src = [0u32; 2];
        for (k, s) in step.inputs.iter().enumerate() {
            let reg = match self.register(s) {
                Some(reg) => reg,
                None => {
                    let slot = match self.inputs.iter().position(|t| Arc::ptr_eq(t, s)) {
                        Some(slot) => slot,
                        None => {
                            self.inputs.push(Arc::clone(s));
                            self.inputs.len() - 1
                        }
                    };
                    let reg = self.free_register();
                    self.regs[reg] = Some(Value::Input(slot));
                    self.program.extend_from_slice(&[
                        FUSED_LOAD,
                        0,
                        slot as u32,
                        0,
                        reg as u32,
                        0,
                        0,
                        NO_STORE,
                    ]);
                    reg
                }
            };
            src[k] = reg as u32;
        }
        let dst = self.free_register();
        self.regs[dst] = Some(Value::Result {
            storage: Arc::clone(&step.output),
            word: self.program.len(),
        });
        self.program.extend_from_slice(&[
            step.kind,
            step.op,
            src[0],
            src[1],
            dst as u32,
            step.a.to_bits(),
            step.b.to_bits(),
            NO_STORE,
        ]);
    }

    /// The storage to bind, by slot, and the program with a store for each
    /// result a tensor still holds. `None` when no tensor holds any result:
    /// the dispatch would then change nothing.
    fn finish(mut self) -> Option<(Vec<Arc<Storage>>, Vec<u32>)> {
        self.release_dead();
        let mut slots = self.inputs;
        let read = slots.len();
        for value in self.regs.into_iter().flatten() {
            if let Value::Result { storage, word } = value {
                self.program[word + 7] = slots.len() as u32;
                slots.push(storage);
            }
        }
        (slots.len() > read).then_some((slots, self.program))
    }
}

/// Freed tensor buffers by size, reused by later ops.
///
/// Reuse needs no synchronisation: the queue runs the new owner's ops after
/// every op of the old one that was recorded before it.
#[derive(Default)]
pub(crate) struct BufferPool {
    free: HashMap<u64, Vec<Buffer>>,
    bytes: u64,
}

/// A tensor's storage buffer; it goes back to its device's pool when dropped.
pub(crate) struct GpuBuffer {
    buffer: Option<Buffer>,
    pool: Arc<Mutex<BufferPool>>,
}

impl std::ops::Deref for GpuBuffer {
    type Target = Buffer;

    #[expect(clippy::expect_used, reason = "the buffer is taken out only in drop")]
    fn deref(&self) -> &Buffer {
        self.buffer.as_ref().expect("buffer used after drop")
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        let Ok(mut pool) = self.pool.lock() else {
            return;
        };
        let size = buffer.size();
        if pool.bytes + size <= POOL_BYTES {
            pool.bytes += size;
            pool.free.entry(size).or_default().push(buffer);
        }
    }
}

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
    pool: Arc<Mutex<BufferPool>>,
    /// Whether element-wise ops are fused (`FERROTENSOR_FUSION=1`).
    fusion: AtomicBool,
    /// Placeholders for the unused slots of a `fused` dispatch: one each, so
    /// no buffer is bound twice as read-write.
    fused_dummies: Vec<Buffer>,
    /// The last `fused` dispatch recorded.
    #[cfg(test)]
    last_fused: Mutex<Option<GroupSummary>>,
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
    /// Element-wise steps not yet recorded as a `fused` dispatch.
    group: Option<FusionGroup>,
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
        let fused_dummies = (0..FUSED_SLOTS)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Fused slot placeholder"),
                    size: 4,
                    usage: BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let fusion =
            engine.has("fused") && std::env::var("FERROTENSOR_FUSION").as_deref() == Ok("1");
        Self {
            adapter,
            device,
            queue,
            engine,
            args,
            align,
            pending: Mutex::default(),
            pool: Arc::default(),
            fusion: AtomicBool::new(fusion),
            fused_dummies,
            #[cfg(test)]
            last_fused: Mutex::default(),
        }
    }

    /// Whether element-wise ops go through [`GpuContext::elementwise`].
    pub(crate) fn fusion(&self) -> bool {
        self.fusion.load(Ordering::Relaxed)
    }

    /// Turn fusion on or off (on only where the `fused` kernel compiled).
    /// Tests use it to compare fused and unfused runs on one device.
    #[cfg(test)]
    pub(crate) fn set_fusion(&self, on: bool) {
        self.fusion
            .store(on && self.engine.has("fused"), Ordering::Relaxed);
    }

    /// Record an element-wise op, joining the open fusion group when it has
    /// the same element count and room for the op's buffers.
    pub(crate) fn elementwise(&self, step: ElementwiseStep) {
        let mut pending = self.lock();
        if !pending.group.as_mut().is_some_and(|g| g.join(&step)) {
            self.close_group(&mut pending);
            let mut group = FusionGroup::new(step.n);
            group.push(&step);
            pending.group = Some(group);
        }
        self.count_op(&mut pending);
    }

    /// Record the open fusion group, if any, as one `fused` dispatch. A
    /// group whose results no tensor holds records nothing.
    fn close_group(&self, pending: &mut Pending) {
        let Some(group) = pending.group.take() else {
            return;
        };
        let n = group.n;
        let Some((slots, program)) = group.finish() else {
            return;
        };
        #[cfg(test)]
        if let Ok(mut last) = self.last_fused.lock() {
            *last = Some(GroupSummary::of(&program));
        }
        let buffers: Vec<&Buffer> = slots
            .iter()
            .map(|s| match &**s {
                Storage::Gpu(buffer) => &**buffer,
                Storage::Cpu(_) => unreachable!("fused step on a CPU tensor"),
            })
            .collect();
        let mut args: Vec<Arg<'_>> = (0..FUSED_SLOTS)
            .map(|i| Arg::Buffer(buffers.get(i).copied().unwrap_or(&self.fused_dummies[i])))
            .collect();
        args.push(Arg::Meta(&program));
        args.push(Arg::Params([n as u32, (program.len() / 8) as u32, 0, 0]));
        self.stage(pending, "fused", &args, ComputeEngine::workgroups_1d(n));
    }

    /// The last `fused` dispatch recorded since the previous call.
    #[cfg(test)]
    pub(crate) fn last_fused(&self) -> Option<GroupSummary> {
        self.last_fused.lock().ok().and_then(|mut last| last.take())
    }

    /// Drop every pooled buffer.
    #[cfg(test)]
    pub(crate) fn clear_pool(&self) {
        if let Ok(mut pool) = self.pool.lock() {
            *pool = BufferPool::default();
        }
    }

    /// Free buffers of `bytes` bytes in the pool.
    #[cfg(test)]
    pub(crate) fn pooled(&self, bytes: u64) -> usize {
        self.pool
            .lock()
            .map_or(0, |pool| pool.free.get(&bytes).map_or(0, Vec::len))
    }

    /// A tensor buffer of `bytes` bytes: a freed one when available, else a
    /// new one made by `create`.
    #[expect(
        clippy::expect_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub(crate) fn buffer(&self, bytes: u64, create: impl FnOnce() -> Buffer) -> GpuBuffer {
        let reused = {
            let mut pool = self.pool.lock().expect("buffer pool lock poisoned");
            let buffer = pool.free.get_mut(&bytes).and_then(Vec::pop);
            if buffer.is_some() {
                pool.bytes -= bytes;
            }
            buffer
        };
        self.adopt(reused.unwrap_or_else(create))
    }

    /// Wrap a new tensor buffer so it goes to the pool when dropped.
    pub(crate) fn adopt(&self, buffer: Buffer) -> GpuBuffer {
        GpuBuffer {
            buffer: Some(buffer),
            pool: Arc::clone(&self.pool),
        }
    }

    #[expect(
        clippy::expect_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
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
        let mut pending = self.lock();
        self.close_group(&mut pending);
        self.stage(&mut pending, kernel, args, workgroups);
        self.count_op(&mut pending);
    }

    /// Stage `args` in the batch's argument buffer and queue the dispatch.
    fn stage(
        &self,
        pending: &mut Pending,
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

        if pending.args.len() + needed > ARGS_BYTES {
            self.submit(pending);
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
    /// up to [`SPIN_WAIT`], then sleeping [`POLL_SLEEP`] between polls.
    ///
    /// Never blocks in a waiting poll: wgpu-core 0.19 gave up waiting
    /// there after 5 s and then freed the still-running submission's
    /// resources as if it had finished, which crashed drivers that were still
    /// reading them.
    #[expect(
        clippy::panic,
        reason = "the callback sender lives until wgpu fires it, so a dropped one means the device is lost"
    )]
    pub(crate) fn wait_for<T>(&self, rx: &Receiver<T>) -> T {
        let start = Instant::now();
        loop {
            // `PollType::Poll` neither waits nor names a submission, so
            // neither `PollError` (a wait timeout, a wrong submission index)
            // can occur; were one returned, `rx` stays empty and the loop
            // polls again.
            if let Err(e) = self.device.poll(wgpu::PollType::Poll) {
                eprintln!("ferrotensor: non-blocking device poll failed: {e}");
            }
            match rx.try_recv() {
                Ok(value) => return value,
                Err(TryRecvError::Empty) if start.elapsed() < SPIN_WAIT => std::thread::yield_now(),
                Err(TryRecvError::Empty) => std::thread::sleep(POLL_SLEEP),
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

    /// The batch's encoder, with the pending dispatches (and the open
    /// fusion group) encoded into it.
    fn encoder<'p>(&self, pending: &'p mut Pending) -> &'p mut CommandEncoder {
        self.close_group(pending);
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
    /// On macOS this is Metal; on Windows/Linux, Vulkan or DX12. Set the
    /// `WGPU_BACKEND` environment variable (e.g. `vulkan`) to restrict the
    /// choice.
    ///
    /// # Panics
    /// Panics when no GPU adapter is available. Use [`Device::gpu`] to get an
    /// `Option`, or [`Device::best`] to fall back to the CPU.
    #[expect(
        clippy::expect_used,
        reason = "documented panic; Device::gpu and Device::best are the fallible constructors"
    )]
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
        // WGPU_BACKEND (e.g. "vulkan", "metal", "dx12", "gl") restricts the backends tried.
        // WGPU_DX12_COMPILER ("dxc" or "fxc") picks the DX12 shader compiler.
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all());
        desc.backend_options = wgpu::BackendOptions::from_env_or_default();
        let instance = wgpu::Instance::new(desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                apply_limit_buckets: false,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Ferrotensor Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
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
    #[expect(
        clippy::expect_used,
        reason = "documented panic; Device::gpu_async is the fallible constructor"
    )]
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

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

    use super::*;
    use std::sync::RwLock;

    fn storage() -> Arc<Storage> {
        Arc::new(Storage::Cpu(RwLock::new(Vec::new())))
    }

    fn step(inputs: &[&Arc<Storage>], output: &Arc<Storage>) -> ElementwiseStep {
        ElementwiseStep {
            kind: inputs.len() as u32 - 1,
            op: 0,
            a: 0.0,
            b: 0.0,
            inputs: inputs.iter().map(|s| Arc::clone(s)).collect(),
            output: Arc::clone(output),
            n: 4,
        }
    }

    /// The store word of each unary or binary instruction, in order.
    fn stores(program: &[u32]) -> Vec<u32> {
        program
            .chunks(8)
            .filter(|w| w[0] != FUSED_LOAD)
            .map(|w| w[7])
            .collect()
    }

    #[test]
    fn a_chain_whose_intermediates_are_dropped_stores_only_its_result() {
        let x = storage();
        let mut group = FusionGroup::new(4);
        let mut y = Arc::clone(&x);
        for _ in 0..100 {
            let next = storage();
            assert!(group.join(&step(&[&y], &next)));
            y = next;
        }
        let (slots, program) = group.finish().expect("y holds a result");
        assert_eq!(slots.len(), 2);
        assert!(Arc::ptr_eq(&slots[0], &x) && Arc::ptr_eq(&slots[1], &y));
        assert_eq!(program.len(), 101 * 8, "one load and 100 steps");
        let mut expected = vec![NO_STORE; 100];
        expected[99] = 1;
        assert_eq!(stores(&program), expected);
    }

    #[test]
    fn held_results_take_slots_and_are_all_stored() {
        let x = storage();
        let mut group = FusionGroup::new(4);
        let mut kept = vec![Arc::clone(&x)];
        for _ in 0..FUSED_SLOTS {
            let next = storage();
            if !group.join(&step(&[kept.last().unwrap()], &next)) {
                break;
            }
            kept.push(next);
        }
        assert_eq!(kept.len(), FUSED_SLOTS, "x and six held results");

        // Dropping a held result frees its slot for one more step, and that
        // result is not stored.
        drop(kept.remove(3));
        let next = storage();
        assert!(group.join(&step(&[kept.last().unwrap()], &next)));
        kept.push(next);

        let (slots, program) = group.finish().expect("results are held");
        assert_eq!(slots.len(), FUSED_SLOTS);
        let stores = stores(&program);
        assert_eq!(stores.len(), 7);
        assert_eq!(stores[2], NO_STORE, "the dropped result");
        let written: Vec<&Arc<Storage>> = stores
            .iter()
            .filter(|&&s| s != NO_STORE)
            .map(|&s| &slots[s as usize])
            .collect();
        assert_eq!(written.len(), 6);
        for (slot, held) in written.iter().zip(&kept[1..]) {
            assert!(Arc::ptr_eq(slot, held));
        }
    }

    #[test]
    fn a_dropped_input_keeps_its_slot() {
        let mut group = FusionGroup::new(4);
        let mut y = storage();
        let mut steps = 0;
        for _ in 0..FUSED_SLOTS {
            let w = storage();
            let next = storage();
            if !group.join(&step(&[&y, &w], &next)) {
                break;
            }
            y = next;
            steps += 1;
        }
        // Slots: the first y and four w, all dropped, and the last result.
        assert_eq!(steps, 4);
        let (slots, program) = group.finish().expect("y holds a result");
        assert_eq!(slots.len(), 6);
        assert_eq!(stores(&program), [NO_STORE, NO_STORE, NO_STORE, 5]);
    }

    #[test]
    fn an_input_on_both_sides_is_loaded_once() {
        let x = storage();
        let out = storage();
        let mut group = FusionGroup::new(4);
        assert!(group.join(&step(&[&x, &x], &out)));
        let (slots, program) = group.finish().expect("out is held");
        assert_eq!(slots.len(), 2);
        assert_eq!(program.len(), 2 * 8);
        let (load, add) = program.split_at(8);
        assert_eq!(load[0], FUSED_LOAD);
        assert_eq!((add[2], add[3]), (load[4], load[4]));
    }

    #[test]
    fn a_group_whose_results_are_all_dropped_records_nothing() {
        let x = storage();
        let mut group = FusionGroup::new(4);
        assert!(group.join(&step(&[&x], &storage())));
        assert!(group.finish().is_none());
    }

    #[test]
    fn a_step_of_another_length_does_not_join() {
        let x = storage();
        let out = storage();
        let mut group = FusionGroup::new(4);
        let mut other = step(&[&x], &out);
        other.n = 5;
        assert!(!group.join(&other));
    }
}
