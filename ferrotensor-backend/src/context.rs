//! GPU device setup.

use crate::kernel_timing::{
    KernelLabel, KernelTimer, KernelTimes, KernelTiming, KernelTimingMode, PendingTimes,
};
use crate::pool::BufferPool;
use crate::profile::{GpuCounterSnapshot, GpuCounters};
use crate::shader_checks::ShaderChecks;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use wgpu::{Adapter, BindGroup, CommandEncoder, ComputePipeline, Device, Instance, Queue};

/// Recorded GPU work is submitted once this many ops are pending, so the GPU
/// starts on long sequences without waiting for a sync point.
const MAX_PENDING_OPS: u32 = 128;

/// Queued dispatches are encoded into a pass once this many are waiting. On
/// an A10G (20 Nov 2026) a step of SmolLM2-135M ran faster the longer the
/// pass, up to this many, which is the pending-ops cap: each pass costs host
/// time in the driver, and a longer pass gave 1.7B no GPU slowdown.
const MAX_PASS_DISPATCHES: usize = 128;

/// Environment variable that replaces `MAX_PENDING_OPS` when the context is
/// created (a positive integer), to tune submission on a given GPU.
pub const PENDING_OPS_ENV: &str = "FERROTENSOR_PENDING_OPS";

/// Environment variable that replaces `MAX_PASS_DISPATCHES` when the context
/// is created (a positive integer).
pub const PASS_DISPATCHES_ENV: &str = "FERROTENSOR_PASS_DISPATCHES";

/// Environment variable that, set to `1` when the context is created, passes
/// kernel parameters in uniform buffers even on a device with immediates (see
/// [`WgpuContext::uses_immediates`]).
pub const UNIFORM_PARAMS_ENV: &str = "FERROTENSOR_UNIFORM_PARAMS";

/// A positive integer limit from an environment variable's `value`, or
/// `default` when it is unset, zero or not a number.
fn limit_from_env_value<T>(value: Option<&str>, default: T) -> T
where
    T: std::str::FromStr + PartialEq + From<u8>,
{
    match value.map(|v| v.trim().parse::<T>()) {
        Some(Ok(n)) if n != T::from(0) => n,
        _ => default,
    }
}

/// Uploaded bytes after which [`WgpuContext::note_upload`] waits for the
/// staging copies to finish so wgpu can free them.
pub const UPLOAD_RELEASE_BYTES: u64 = 256 << 20;

/// A GPU device, its queue and a buffer pool shared by tensors.
///
/// Kernel dispatches and copies from [`ComputeEngine`](crate::ComputeEngine)
/// are recorded into one pending command encoder and submitted together:
/// up to 8 consecutive dispatches share one compute pass, which a copy or a
/// submission also ends. They are submitted every 128 ops, and whenever a result is read
/// ([`UnifiedTensor::to_vec`](crate::UnifiedTensor::to_vec)) or a buffer is
/// written ([`UnifiedTensor::write_data`](crate::UnifiedTensor::write_data)).
/// Call [`flush`](Self::flush) before submitting work of your own on
/// `queue`, so it runs after the recorded ops, and
/// [`synchronize`](Self::synchronize) to wait for the GPU.
///
/// After each submission, and whenever it waits for the GPU, the context
/// drops pooled buffers above the pool's cap that the GPU has finished with
/// and polls the device without blocking, so wgpu destroys them at once (see
/// [`BufferPool`] on overflow).
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
    pub(crate) counters: GpuCounters,
    /// Whether kernel parameters go through immediates.
    immediates: bool,
    /// Timestamp queries, with [`KernelTiming::On`] on an adapter that has them.
    timer: Option<KernelTimer>,
    /// Ops recorded before a submit, from [`PENDING_OPS_ENV`].
    max_pending_ops: u32,
    /// Dispatches per compute pass, from [`PASS_DISPATCHES_ENV`].
    max_pass_dispatches: usize,
    /// Runtime checks for the built-in kernels, from
    /// [`SHADER_CHECKS_ENV`](crate::SHADER_CHECKS_ENV).
    shader_checks: ShaderChecks,
}

/// Recorded work not yet submitted: the encoder, and the dispatches not yet
/// encoded into it.
#[derive(Default)]
struct Pending {
    encoder: Option<CommandEncoder>,
    dispatches: Vec<Dispatch>,
    ops: u32,
    times: PendingTimes,
}

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending")
            .field("ops", &self.ops)
            .field("dispatches", &self.dispatches.len())
            .finish()
    }
}

/// One kernel dispatch waiting to be encoded.
pub(crate) struct Dispatch {
    pub(crate) pipeline: Arc<ComputePipeline>,
    pub(crate) bind_group: BindGroup,
    pub(crate) groups: (u32, u32, u32),
    pub(crate) label: KernelLabel,
    /// The autotuner's sample this dispatch is part of: its GPU time is
    /// read back for the tuner (see `engine::coop::autotune`).
    pub(crate) tune: Option<u64>,
    /// The kernel's parameters, when they go through immediates.
    pub(crate) immediates: Option<Vec<u8>>,
}

impl Dispatch {
    /// Binds this dispatch's buffers and parameters in `pass`, whose pipeline
    /// is already set, and dispatches it.
    pub(crate) fn encode(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_bind_group(0, &self.bind_group, &[]);
        if let Some(bytes) = &self.immediates {
            pass.set_immediates(0, bytes);
        }
        let (x, y, z) = self.groups;
        pass.dispatch_workgroups(x, y, z);
    }
}

/// Why [`WgpuContext::try_new_with_options`] could not open a GPU.
#[derive(Debug)]
#[non_exhaustive]
pub enum ContextError {
    /// wgpu found no adapter on the backends it was allowed to try.
    NoAdapter {
        /// The backends that were tried (see [`crate::adapters::backends`]).
        backends: wgpu::Backends,
        /// Whether only the software fallback adapter was asked for.
        force_fallback_adapter: bool,
    },
    /// The adapter was found but refused to create a device.
    DeviceRefused {
        /// Adapter name reported by the driver.
        adapter: String,
        /// The error wgpu returned.
        source: wgpu::RequestDeviceError,
    },
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter {
                backends,
                force_fallback_adapter,
            } => {
                let names: Vec<String> = backends
                    .iter_names()
                    .map(|(name, _)| name.to_lowercase())
                    .collect();
                let kind = if *force_fallback_adapter {
                    "software fallback"
                } else {
                    "GPU"
                };
                write!(
                    f,
                    "no {kind} adapter found for backends {}; check the GPU drivers, \
                     or set WGPU_BACKEND to one of vulkan, metal, dx12, gl",
                    names.join(", ")
                )
            }
            Self::DeviceRefused { adapter, source } => {
                write!(
                    f,
                    "GPU adapter {adapter} refused to create a device: {source}"
                )
            }
        }
    }
}

impl std::error::Error for ContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoAdapter { .. } => None,
            Self::DeviceRefused { source, .. } => Some(source),
        }
    }
}

impl WgpuContext {
    /// Opens the high-performance adapter.
    ///
    /// # Panics
    /// Panics with the [`ContextError`] message if no adapter or device is
    /// available; [`try_new`](Self::try_new) returns it instead.
    pub async fn new() -> Self {
        Self::new_with_options(wgpu::PowerPreference::HighPerformance, false).await
    }

    /// Opens an adapter like [`try_new_with_options`](Self::try_new_with_options).
    ///
    /// # Panics
    /// Panics with the [`ContextError`] message if no adapter or device is
    /// available.
    #[expect(
        clippy::panic,
        reason = "documented panic; try_new_with_options is the fallible constructor"
    )]
    pub async fn new_with_options(
        power_preference: wgpu::PowerPreference,
        force_fallback_adapter: bool,
    ) -> Self {
        Self::try_new_with_options(power_preference, force_fallback_adapter)
            .await
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Opens the high-performance adapter.
    ///
    /// # Errors
    /// See [`try_new_with_options`](Self::try_new_with_options).
    pub async fn try_new() -> Result<Self, ContextError> {
        Self::try_new_with_options(wgpu::PowerPreference::HighPerformance, false).await
    }

    /// Opens an adapter with the given power preference, optionally forcing
    /// the software fallback adapter. The `WGPU_BACKEND` environment variable
    /// (e.g. `vulkan`) restricts which backends are tried, and
    /// `WGPU_DX12_COMPILER` (`dxc` or `fxc`) picks the DX12 shader compiler.
    ///
    /// # Errors
    /// [`ContextError::NoAdapter`] if wgpu finds no adapter, and
    /// [`ContextError::DeviceRefused`] if the adapter will not create a device.
    pub async fn try_new_with_options(
        power_preference: wgpu::PowerPreference,
        force_fallback_adapter: bool,
    ) -> Result<Self, ContextError> {
        Self::try_new_with_timing(power_preference, force_fallback_adapter, KernelTiming::Off).await
    }

    /// Opens an adapter like [`try_new_with_options`](Self::try_new_with_options);
    /// with [`KernelTiming::On`] it also asks for timestamp queries, when the
    /// adapter has them, and times every kernel (see
    /// [`kernel_times`](Self::kernel_times)).
    ///
    /// # Errors
    /// As [`try_new_with_options`](Self::try_new_with_options).
    pub async fn try_new_with_timing(
        power_preference: wgpu::PowerPreference,
        force_fallback_adapter: bool,
        timing: KernelTiming,
    ) -> Result<Self, ContextError> {
        let backends = crate::adapters::backends();
        let instance = crate::adapters::instance(backends);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                force_fallback_adapter,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|_| ContextError::NoAdapter {
                backends,
                force_fallback_adapter,
            })?;

        // The autotuner times the dispatches it tags, so it needs timestamps
        // even when kernel timing is off.
        let tuning = crate::engine::autotune_requested();
        let timing_mode = match timing {
            KernelTiming::Off if !tuning => None,
            KernelTiming::Off | KernelTiming::On => {
                KernelTimingMode::for_features(adapter.features())
            }
        };
        let required_features = timing_mode
            .map_or(wgpu::Features::empty(), KernelTimingMode::features)
            | crate::adapters::optional_features(&adapter);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Ferrotensor Device"),
                required_features,
                // Use what the hardware supports rather than the WebGPU
                // defaults (e.g. 8 storage buffers per stage, 128 MiB
                // bindings), which are too small for model weights.
                required_limits: adapter.limits(),
                experimental_features: crate::adapters::experimental_features(required_features),
                ..Default::default()
            })
            .await
            .map_err(|source| ContextError::DeviceRefused {
                adapter: adapter.get_info().name,
                source,
            })?;
        let pool = Arc::new(BufferPool::new(
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        ));
        pool.set_overflow(crate::pool::pool_overflow_from_env_value(
            std::env::var(crate::pool::POOL_OVERFLOW_ENV)
                .ok()
                .as_deref(),
        ));
        let env = |name: &str| std::env::var(name).ok();
        let max_pending_ops =
            limit_from_env_value(env(PENDING_OPS_ENV).as_deref(), MAX_PENDING_OPS);
        let max_pass_dispatches =
            limit_from_env_value(env(PASS_DISPATCHES_ENV).as_deref(), MAX_PASS_DISPATCHES);
        let timer = timing_mode.map(|mode| {
            KernelTimer::new(mode, &queue, max_pending_ops, timing == KernelTiming::On)
        });
        let immediates = device.features().contains(wgpu::Features::IMMEDIATES)
            && device.limits().max_immediate_size >= crate::engine::IMMEDIATE_BYTES
            && !env(UNIFORM_PARAMS_ENV).is_some_and(|v| v.trim() == "1");
        Ok(Self {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            pool,
            pending: Mutex::new(Pending::default()),
            staged: AtomicU64::new(0),
            counters: GpuCounters::default(),
            timer,
            max_pending_ops,
            max_pass_dispatches,
            shader_checks: ShaderChecks::from_env(),
            immediates,
        })
    }

    /// Whether [`ComputeEngine`](crate::ComputeEngine) passes kernel
    /// parameters as immediates, set with each dispatch, rather than in a
    /// uniform buffer made for it. True when the device has
    /// `wgpu::Features::IMMEDIATES` for the largest parameters and
    /// [`UNIFORM_PARAMS_ENV`] was not `1`.
    pub fn uses_immediates(&self) -> bool {
        self.immediates
    }

    /// This context with kernel parameters in uniform buffers, as
    /// [`UNIFORM_PARAMS_ENV`] asks; build engines for it after this call.
    #[must_use]
    pub fn with_uniform_params(mut self) -> Self {
        self.immediates = false;
        self
    }

    /// Records GPU work into the pending command encoder; it runs after
    /// everything recorded before it. Submits once enough ops are pending.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub(crate) fn record(&self, f: impl FnOnce(&mut CommandEncoder)) {
        let mut pending = self.pending.lock().unwrap();
        f(self.encoder(&mut pending));
        self.count_op(&mut pending);
    }

    /// Records a kernel dispatch into the open compute pass; it runs after
    /// everything recorded before it. Submits once enough ops are pending.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub(crate) fn record_dispatch(&self, dispatch: Dispatch) {
        let mut pending = self.pending.lock().unwrap();
        pending.dispatches.push(dispatch);
        if pending.dispatches.len() >= self.max_pass_dispatches {
            self.encoder(&mut pending);
        }
        self.count_op(&mut pending);
    }

    fn count_op(&self, pending: &mut Pending) {
        pending.ops += 1;
        if pending.ops >= self.max_pending_ops {
            self.submit(pending);
        }
    }

    /// The pending encoder, with the pending dispatches encoded into it as
    /// one compute pass (or one pass each, timed at its start and end, with
    /// [`KernelTimingMode::PassBoundaries`]).
    ///
    /// wgpu-core 0.19 builds usage trackers sized by the highest live buffer
    /// id for every pass and drops them at its end, so one pass per dispatch
    /// cost more host time than the dispatch. Within a pass wgpu still checks
    /// usage per dispatch and puts a barrier between a write and a later
    /// access, so each dispatch sees the results of those before it.
    fn encoder<'p>(&self, pending: &'p mut Pending) -> &'p mut CommandEncoder {
        let Pending {
            encoder,
            dispatches,
            times,
            ..
        } = pending;
        let encoder = encoder.get_or_insert_with(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
        });
        if dispatches.is_empty() {
            return encoder;
        }
        let timer = self.timer.as_ref().filter(|t| t.times_any(dispatches));
        if let Some(timer) = timer {
            let passes = timer.encode(&self.device, encoder, dispatches, times);
            self.counters
                .compute_passes
                .fetch_add(passes, Ordering::Relaxed);
            dispatches.clear();
        } else {
            self.counters.compute_passes.fetch_add(1, Ordering::Relaxed);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let mut bound: Option<&Arc<ComputePipeline>> = None;
            for dispatch in dispatches.iter() {
                if !bound.is_some_and(|pipeline| Arc::ptr_eq(pipeline, &dispatch.pipeline)) {
                    pass.set_pipeline(&dispatch.pipeline);
                    bound = Some(&dispatch.pipeline);
                }
                dispatch.encode(&mut pass);
            }
            drop(pass);
            dispatches.clear();
        }
        encoder
    }

    /// Submits all recorded work to the queue. Reads and writes through
    /// [`UnifiedTensor`](crate::UnifiedTensor) do this themselves.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub fn flush(&self) {
        let mut pending = self.pending.lock().unwrap();
        self.submit(&mut pending);
    }

    /// Submits all recorded work and pending uploads and blocks until the
    /// GPU has finished them. Staging buffers of finished uploads are freed.
    pub fn synchronize(&self) {
        self.counters.synchronizes.fetch_add(1, Ordering::Relaxed);
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
        self.counters
            .upload_bytes
            .fetch_add(bytes, Ordering::Relaxed);
        let total = self.staged.fetch_add(bytes, Ordering::Relaxed) + bytes;
        if total >= UPLOAD_RELEASE_BYTES {
            self.counters
                .forced_synchronizes
                .fetch_add(1, Ordering::Relaxed);
            self.synchronize();
        }
    }

    /// Polls the device until `done` returns true.
    ///
    /// Never blocks in a waiting poll: wgpu-core 0.19 gave up waiting
    /// there after 5 s and then frees the still-running submission's
    /// resources as if it had finished, which crashed drivers that were still
    /// reading them (seen with long batches on lavapipe).
    pub(crate) fn poll_until(&self, done: impl Fn() -> bool) {
        self.poll_until_done(done);
        // Callers wait for the last submitted work, so that poll let wgpu
        // destroy every buffer dropped before it.
        self.memory().gpu_reclaimed();
        self.release_idle_buffers();
        if let Some(timer) = &self.timer {
            timer.collect();
        }
    }

    /// Polls the device until `done` returns true, counting the time as GPU
    /// wait; see [`poll_until`](Self::poll_until) for why it never blocks.
    fn poll_until_done(&self, done: impl Fn() -> bool) {
        let start = Instant::now();
        let mut polls = 0u32;
        loop {
            self.poll();
            if done() {
                GpuCounters::add_elapsed(&self.counters.gpu_wait_nanos, start);
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

    /// Submits the recorded work and returns the number of the last
    /// submission, to pass to [`wait_for_submission`](Self::wait_for_submission).
    pub fn submission_mark(&self) -> u64 {
        self.flush();
        self.memory().recording_serial().saturating_sub(1)
    }

    /// Submits the recorded work, then polls until submission `mark` (from
    /// [`submission_mark`](Self::submission_mark)) has finished on the GPU
    /// and drops the pool's idle buffers above its cap that only that work
    /// used.
    ///
    /// Work submitted after `mark` keeps running. A trainer that never reads
    /// the GPU inside a window calls this once per micro-batch so the host
    /// stays at most one micro-batch ahead; otherwise, on a GPU-bound model,
    /// the pool cannot trim and holds the idle buffers of several
    /// micro-batches at once.
    pub fn wait_for_submission(&self, mark: u64) {
        self.flush();
        let memory = self.memory();
        self.poll_until_done(|| memory.completed_serial() >= mark);
        self.release_idle_buffers();
    }

    fn submit(&self, pending: &mut Pending) {
        pending.ops = 0;
        if !pending.dispatches.is_empty() {
            self.encoder(pending);
        }
        if let Some(mut encoder) = pending.encoder.take() {
            if let Some(timer) = &self.timer {
                timer.resolve(&mut encoder, &pending.times);
            }
            let memory = self.memory();
            let serial = memory.begin_submission();
            self.counters.submits.fetch_add(1, Ordering::Relaxed);
            // wgpu maintains the device here: it destroys buffers whose work
            // has finished and runs the callbacks of finished submissions.
            self.queue.submit(Some(encoder.finish()));
            if let Some(timer) = &self.timer {
                timer.submitted(&mut pending.times);
            }
            let tracker = memory.clone();
            self.queue
                .on_submitted_work_done(move || tracker.gpu_completed(serial));
            self.release_idle_buffers();
        }
    }

    /// Drops pooled buffers above the cap that the GPU has finished with and,
    /// if any, polls once without blocking so wgpu destroys them now rather
    /// than at the next submission.
    fn release_idle_buffers(&self) {
        if self.pool.trim() > 0 {
            self.poll();
        }
    }

    /// Polls the device without blocking. Each poll runs wgpu's device
    /// maintenance, whose cost grows with the number of live buffers.
    fn poll(&self) {
        self.counters.device_polls.fetch_add(1, Ordering::Relaxed);
        // `PollType::Poll` neither waits nor names a submission, so neither
        // `PollError` (a wait timeout, a wrong submission index) can occur.
        if let Err(e) = self.device.poll(wgpu::PollType::Poll) {
            log::warn!("non-blocking device poll failed: {e}");
        }
    }

    /// Byte counters of this context's tensors: live, peak, idle in the
    /// pool, and by [`MemoryCategory`](crate::MemoryCategory).
    pub fn memory(&self) -> &Arc<crate::MemoryTracker> {
        self.pool.tracker()
    }

    /// Totals of the GPU work recorded and waited for on this context.
    pub fn gpu_counters(&self) -> GpuCounterSnapshot {
        self.counters.snapshot()
    }

    /// The runtime checks the built-in kernels are compiled with, read from
    /// [`SHADER_CHECKS_ENV`](crate::SHADER_CHECKS_ENV) when the context was created.
    pub fn shader_checks(&self) -> ShaderChecks {
        self.shader_checks
    }

    /// Where this context writes kernel timestamps; `None` when it was
    /// opened without [`KernelTiming::On`] or the adapter has no timestamp
    /// queries.
    pub fn kernel_timing_mode(&self) -> Option<KernelTimingMode> {
        self.timer
            .as_ref()
            .filter(|timer| timer.all)
            .map(|timer| timer.mode)
    }

    /// Whether the context can time the dispatches the autotuner tags.
    pub(crate) fn has_timestamps(&self) -> bool {
        self.timer.is_some()
    }

    /// The GPU times of the autotuner's tagged dispatches that the GPU has
    /// finished, each as its sample id and duration (`None`: unusable
    /// timestamps); never waits.
    pub(crate) fn take_tune_samples(&self) -> Vec<(u64, Option<Duration>)> {
        self.timer.as_ref().map_or_else(Vec::new, |timer| {
            timer.collect();
            timer.take_samples()
        })
    }

    /// GPU time by kernel, of the work the GPU finished before the last
    /// wait for it; empty without kernel timing. Never waits.
    pub fn kernel_times(&self) -> KernelTimes {
        self.timer
            .as_ref()
            .filter(|timer| timer.all)
            .map_or_else(KernelTimes::default, |timer| {
                timer.collect();
                timer.totals()
            })
    }

    /// The cooperative-matrix configurations this device can run; empty when
    /// it was opened without cooperative matrices.
    pub fn cooperative_matrix(&self) -> Vec<crate::adapters::CoopMatrixConfig> {
        if self
            .device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        {
            crate::adapters::cooperative_matrix_configs(&self.adapter)
        } else {
            Vec::new()
        }
    }

    /// Adapter name and backend, e.g. `"Apple M2 (Metal)"`.
    pub fn adapter_info(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} ({:?})", info.name, info.backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_limits_take_a_positive_integer_from_the_environment() {
        assert_eq!(limit_from_env_value(None, 128u32), 128);
        assert_eq!(limit_from_env_value(Some(" 64 "), 128u32), 64);
        assert_eq!(limit_from_env_value(Some("1"), 8usize), 1);
        for bad in ["0", "-3", "eight", ""] {
            assert_eq!(limit_from_env_value(Some(bad), 8usize), 8, "{bad:?}");
        }
    }

    #[test]
    fn no_adapter_message_names_backends_and_what_to_do() {
        let err = ContextError::NoAdapter {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            force_fallback_adapter: false,
        };
        assert_eq!(
            err.to_string(),
            "no GPU adapter found for backends vulkan, metal; check the GPU drivers, \
             or set WGPU_BACKEND to one of vulkan, metal, dx12, gl"
        );
    }

    #[test]
    fn no_adapter_message_says_fallback_was_forced() {
        let err = ContextError::NoAdapter {
            backends: wgpu::Backends::VULKAN,
            force_fallback_adapter: true,
        };
        assert_eq!(
            err.to_string(),
            "no software fallback adapter found for backends vulkan; check the GPU drivers, \
             or set WGPU_BACKEND to one of vulkan, metal, dx12, gl"
        );
    }
}
