//! Counts of the GPU work a [`WgpuContext`](crate::WgpuContext) does:
//! kernel dispatches, the compute passes, bind groups and uniform buffers
//! made for them, submissions, device polls, waits, uploads and readbacks.
//!
//! The counters are relaxed atomics and always on; recording an op adds to
//! a few of them. Take a [`GpuCounterSnapshot`] with
//! [`WgpuContext::gpu_counters`](crate::WgpuContext::gpu_counters) before and
//! after some work and subtract with [`GpuCounterSnapshot::since`].

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

/// A stretch of host time that `--profile` reports on its own. The spans
/// never overlap: a span's time leaves out any span nested inside it, so
/// their sum is at most the wall time and the rest is "other".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostSpan {
    /// Building a dispatch: reading its arguments, the bind group cache
    /// lookup, creating a bind group or a uniform buffer.
    Record,
    /// Encoding pending dispatches into compute passes and creating the
    /// command encoder.
    Encode,
    /// `CommandEncoder::finish`, with the kernel timing resolve.
    Finish,
    /// `Queue::submit` and registering the completion callback.
    Submit,
    /// Dropping idle pooled buffers over the cap and the poll that destroys
    /// them.
    Release,
    /// Polling for the GPU to finish: the fence wait, with its yields and
    /// sleeps.
    Wait,
    /// Freeing the memory the finished work held and collecting kernel
    /// timestamps, after a wait.
    Reclaim,
    /// A readback outside its flush and its wait: the staging buffer, the
    /// map, the copy to the host and the unmap.
    MapCopy,
    /// Preparing a batch on the host: loading it and padding it.
    Data,
    /// The optimizer step on the host: packing gradients, clipping and
    /// encoding the update, outside its readbacks and waits.
    Optimizer,
    /// `Device::create_bind_group`, on a bind group cache miss.
    BindMake,
}

impl HostSpan {
    /// How many spans there are.
    pub const COUNT: usize = 11;

    /// Every span, in report order.
    pub const ALL: [Self; Self::COUNT] = [
        Self::Record,
        Self::Encode,
        Self::Finish,
        Self::Submit,
        Self::Release,
        Self::Wait,
        Self::Reclaim,
        Self::MapCopy,
        Self::Data,
        Self::Optimizer,
        Self::BindMake,
    ];

    /// The span's snake-case name in the report.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Record => "record",
            Self::Encode => "encode",
            Self::Finish => "finish",
            Self::Submit => "submit",
            Self::Release => "release",
            Self::Wait => "wait",
            Self::Reclaim => "reclaim",
            Self::MapCopy => "map_copy",
            Self::Data => "data",
            Self::Optimizer => "optimizer",
            Self::BindMake => "bind_make",
        }
    }

    const fn index(self) -> usize {
        self as usize
    }
}

/// Times a [`HostSpan`] from its creation to its drop, then adds the time
/// to the span, less the time of spans that ran inside it.
#[derive(Debug)]
#[must_use = "the span is timed until this value is dropped"]
pub struct SpanGuard<'a> {
    counters: &'a GpuCounters,
    span: HostSpan,
    start: Instant,
    accounted_at_start: u64,
}

impl Drop for SpanGuard<'_> {
    fn drop(&mut self) {
        let wall = u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let inner = self
            .counters
            .accounted_nanos
            .load(Relaxed)
            .saturating_sub(self.accounted_at_start);
        let own = wall.saturating_sub(inner);
        self.counters.spans[self.span.index()].fetch_add(own, Relaxed);
        self.counters.accounted_nanos.fetch_add(own, Relaxed);
        if self.counters.in_readback.load(Relaxed) {
            self.counters.readback_spans[self.span.index()].fetch_add(own, Relaxed);
        }
    }
}

/// Marks the host time spent until it drops as inside a readback.
#[derive(Debug)]
pub(crate) struct ReadbackScope<'a>(&'a AtomicBool);

impl Drop for ReadbackScope<'_> {
    fn drop(&mut self) {
        self.0.store(false, Relaxed);
    }
}

/// Running totals of one context's GPU work.
#[derive(Debug, Default)]
pub(crate) struct GpuCounters {
    pub(crate) dispatches: AtomicU64,
    pub(crate) compute_passes: AtomicU64,
    pub(crate) bind_groups: AtomicU64,
    pub(crate) bind_group_hits: AtomicU64,
    pub(crate) bind_group_misses: AtomicU64,
    pub(crate) uniform_buffers: AtomicU64,
    pub(crate) uniform_bytes: AtomicU64,
    pub(crate) submits: AtomicU64,
    pub(crate) device_polls: AtomicU64,
    pub(crate) synchronizes: AtomicU64,
    pub(crate) forced_synchronizes: AtomicU64,
    pub(crate) upload_bytes: AtomicU64,
    pub(crate) readbacks: AtomicU64,
    pub(crate) readback_bytes: AtomicU64,
    pub(crate) readback_nanos: AtomicU64,
    /// Own time of each [`HostSpan`].
    spans: [AtomicU64; HostSpan::COUNT],
    /// The part of `spans` that ran inside a readback.
    readback_spans: [AtomicU64; HostSpan::COUNT],
    /// The sum of `spans`, read by a running span to leave out the spans
    /// nested in it.
    accounted_nanos: AtomicU64,
    in_readback: AtomicBool,
}

impl GpuCounters {
    /// Adds the wall time since `start` to `nanos`.
    pub(crate) fn add_elapsed(nanos: &AtomicU64, start: Instant) {
        let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        nanos.fetch_add(elapsed, Relaxed);
    }

    /// Starts timing `span`.
    pub(crate) fn span(&self, span: HostSpan) -> SpanGuard<'_> {
        SpanGuard {
            counters: self,
            span,
            start: Instant::now(),
            accounted_at_start: self.accounted_nanos.load(Relaxed),
        }
    }

    /// Counts the spans that end before the returned scope drops as inside
    /// a readback.
    pub(crate) fn readback_scope(&self) -> ReadbackScope<'_> {
        self.in_readback.store(true, Relaxed);
        ReadbackScope(&self.in_readback)
    }

    pub(crate) fn snapshot(&self) -> GpuCounterSnapshot {
        GpuCounterSnapshot {
            dispatches: self.dispatches.load(Relaxed),
            compute_passes: self.compute_passes.load(Relaxed),
            bind_groups: self.bind_groups.load(Relaxed),
            bind_group_hits: self.bind_group_hits.load(Relaxed),
            bind_group_misses: self.bind_group_misses.load(Relaxed),
            uniform_buffers: self.uniform_buffers.load(Relaxed),
            uniform_bytes: self.uniform_bytes.load(Relaxed),
            submits: self.submits.load(Relaxed),
            device_polls: self.device_polls.load(Relaxed),
            synchronizes: self.synchronizes.load(Relaxed),
            forced_synchronizes: self.forced_synchronizes.load(Relaxed),
            upload_bytes: self.upload_bytes.load(Relaxed),
            readbacks: self.readbacks.load(Relaxed),
            readback_bytes: self.readback_bytes.load(Relaxed),
            gpu_wait: Duration::from_nanos(self.spans[HostSpan::Wait.index()].load(Relaxed)),
            readback_time: Duration::from_nanos(self.readback_nanos.load(Relaxed)),
            spans: std::array::from_fn(|i| Duration::from_nanos(self.spans[i].load(Relaxed))),
            readback_spans: std::array::from_fn(|i| {
                Duration::from_nanos(self.readback_spans[i].load(Relaxed))
            }),
        }
    }
}

/// The GPU work counters of a context at one moment, or the difference
/// between two such moments.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuCounterSnapshot {
    /// Compute kernel dispatches recorded (buffer copies are not counted).
    pub dispatches: u64,
    /// Compute passes the dispatches were encoded in: one per run of at most
    /// 8 dispatches between copies and submissions.
    pub compute_passes: u64,
    /// Bind groups created: one per dispatch the bind group cache did not
    /// serve.
    pub bind_groups: u64,
    /// Dispatches that reused a cached bind group: the same kernel bound to
    /// the same buffers as an earlier dispatch.
    pub bind_group_hits: u64,
    /// Dispatches that looked in the bind group cache, missed and cached the
    /// group they created. A dispatch that binds a buffer made for it alone,
    /// such as a uniform buffer of parameters, does not look and counts as
    /// neither.
    pub bind_group_misses: u64,
    /// Uniform buffers created for kernel parameters.
    pub uniform_buffers: u64,
    /// Bytes of those uniform buffers, as requested.
    pub uniform_bytes: u64,
    /// Command buffers submitted to the queue (the empty submission inside
    /// each synchronize is not counted).
    pub submits: u64,
    /// Device polls while waiting for the GPU or freeing pooled buffers.
    /// Each runs wgpu's device maintenance, as each submit does too.
    pub device_polls: u64,
    /// Calls to [`WgpuContext::synchronize`](crate::WgpuContext::synchronize).
    pub synchronizes: u64,
    /// The part of `synchronizes` that
    /// [`WgpuContext::note_upload`](crate::WgpuContext::note_upload) forced
    /// to free staging memory.
    pub forced_synchronizes: u64,
    /// Bytes uploaded from the host.
    pub upload_bytes: u64,
    /// Reads of a tensor back to the host.
    pub readbacks: u64,
    /// Bytes read back to the host.
    pub readback_bytes: u64,
    /// Wall time spent polling for the GPU to finish, in synchronize and in
    /// readbacks.
    pub gpu_wait: Duration,
    /// Wall time spent in readbacks, including their wait for the GPU.
    pub readback_time: Duration,
    /// Host time by [`HostSpan`], indexed as [`HostSpan::ALL`].
    pub spans: [Duration; HostSpan::COUNT],
    /// The part of `spans` that ran inside readbacks, indexed the same.
    pub readback_spans: [Duration; HostSpan::COUNT],
}

impl GpuCounterSnapshot {
    /// The work done between `earlier` and `self`.
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            dispatches: self.dispatches.saturating_sub(earlier.dispatches),
            compute_passes: self.compute_passes.saturating_sub(earlier.compute_passes),
            bind_groups: self.bind_groups.saturating_sub(earlier.bind_groups),
            bind_group_hits: self.bind_group_hits.saturating_sub(earlier.bind_group_hits),
            bind_group_misses: self
                .bind_group_misses
                .saturating_sub(earlier.bind_group_misses),
            uniform_buffers: self.uniform_buffers.saturating_sub(earlier.uniform_buffers),
            uniform_bytes: self.uniform_bytes.saturating_sub(earlier.uniform_bytes),
            submits: self.submits.saturating_sub(earlier.submits),
            device_polls: self.device_polls.saturating_sub(earlier.device_polls),
            synchronizes: self.synchronizes.saturating_sub(earlier.synchronizes),
            forced_synchronizes: self
                .forced_synchronizes
                .saturating_sub(earlier.forced_synchronizes),
            upload_bytes: self.upload_bytes.saturating_sub(earlier.upload_bytes),
            readbacks: self.readbacks.saturating_sub(earlier.readbacks),
            readback_bytes: self.readback_bytes.saturating_sub(earlier.readback_bytes),
            gpu_wait: self.gpu_wait.saturating_sub(earlier.gpu_wait),
            readback_time: self.readback_time.saturating_sub(earlier.readback_time),
            spans: std::array::from_fn(|i| self.spans[i].saturating_sub(earlier.spans[i])),
            readback_spans: std::array::from_fn(|i| {
                self.readback_spans[i].saturating_sub(earlier.readback_spans[i])
            }),
        }
    }

    /// The host time in `span`.
    #[must_use]
    pub fn span(&self, span: HostSpan) -> Duration {
        self.spans[span.index()]
    }

    /// The part of the host time in `span` that ran inside readbacks.
    #[must_use]
    pub fn readback_span(&self, span: HostSpan) -> Duration {
        self.readback_spans[span.index()]
    }

    /// Every count (not the times) with its snake-case name, in report order.
    pub fn counts(&self) -> [(&'static str, u64); 14] {
        [
            ("dispatches", self.dispatches),
            ("compute_passes", self.compute_passes),
            ("bind_groups", self.bind_groups),
            ("bind_group_hits", self.bind_group_hits),
            ("bind_group_misses", self.bind_group_misses),
            ("uniform_buffers", self.uniform_buffers),
            ("uniform_bytes", self.uniform_bytes),
            ("submits", self.submits),
            ("device_polls", self.device_polls),
            ("synchronizes", self.synchronizes),
            ("forced_synchronizes", self.forced_synchronizes),
            ("upload_bytes", self.upload_bytes),
            ("readbacks", self.readbacks),
            ("readback_bytes", self.readback_bytes),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn since_subtracts_every_field() {
        let counters = GpuCounters::default();
        counters.dispatches.fetch_add(3, Relaxed);
        counters.uniform_bytes.fetch_add(16, Relaxed);
        counters.spans[HostSpan::Wait.index()].fetch_add(1_000, Relaxed);
        let earlier = counters.snapshot();

        counters.dispatches.fetch_add(5, Relaxed);
        counters.compute_passes.fetch_add(2, Relaxed);
        counters.bind_groups.fetch_add(5, Relaxed);
        counters.bind_group_hits.fetch_add(3, Relaxed);
        counters.bind_group_misses.fetch_add(5, Relaxed);
        counters.uniform_buffers.fetch_add(2, Relaxed);
        counters.uniform_bytes.fetch_add(32, Relaxed);
        counters.submits.fetch_add(1, Relaxed);
        counters.device_polls.fetch_add(7, Relaxed);
        counters.synchronizes.fetch_add(2, Relaxed);
        counters.forced_synchronizes.fetch_add(1, Relaxed);
        counters.upload_bytes.fetch_add(4096, Relaxed);
        counters.readbacks.fetch_add(1, Relaxed);
        counters.readback_bytes.fetch_add(64, Relaxed);
        counters.spans[HostSpan::Wait.index()].fetch_add(2_500, Relaxed);
        counters.spans[HostSpan::Submit.index()].fetch_add(40, Relaxed);
        counters.readback_spans[HostSpan::Submit.index()].fetch_add(30, Relaxed);
        counters.readback_nanos.fetch_add(700, Relaxed);
        let later = counters.snapshot();

        assert_eq!(
            later.since(&earlier),
            GpuCounterSnapshot {
                dispatches: 5,
                compute_passes: 2,
                bind_groups: 5,
                bind_group_hits: 3,
                bind_group_misses: 5,
                uniform_buffers: 2,
                uniform_bytes: 32,
                submits: 1,
                device_polls: 7,
                synchronizes: 2,
                forced_synchronizes: 1,
                upload_bytes: 4096,
                readbacks: 1,
                readback_bytes: 64,
                gpu_wait: Duration::from_nanos(2_500),
                readback_time: Duration::from_nanos(700),
                spans: {
                    let mut spans = [Duration::ZERO; HostSpan::COUNT];
                    spans[HostSpan::Wait.index()] = Duration::from_nanos(2_500);
                    spans[HostSpan::Submit.index()] = Duration::from_nanos(40);
                    spans
                },
                readback_spans: {
                    let mut spans = [Duration::ZERO; HostSpan::COUNT];
                    spans[HostSpan::Submit.index()] = Duration::from_nanos(30);
                    spans
                },
            }
        );
        assert_eq!(later.since(&later), GpuCounterSnapshot::default());
        assert_eq!(
            later.since(&earlier).counts()[0],
            ("dispatches", 5),
            "counts are named in report order"
        );
    }

    #[test]
    fn a_span_leaves_out_the_spans_nested_in_it() {
        let counters = GpuCounters::default();
        {
            let _outer = counters.span(HostSpan::Optimizer);
            std::thread::sleep(Duration::from_millis(10));
            {
                let _inner = counters.span(HostSpan::Submit);
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        let spans = counters.snapshot().spans;
        let outer = spans[HostSpan::Optimizer.index()];
        let inner = spans[HostSpan::Submit.index()];
        assert!(inner >= Duration::from_millis(100), "inner {inner:?}");
        assert!(
            outer >= Duration::from_millis(10) && outer < Duration::from_millis(100),
            "outer {outer:?} must not contain the inner 100 ms"
        );
        assert_eq!(
            counters.accounted_nanos.load(Relaxed),
            (outer + inner).as_nanos() as u64,
            "the running total is the sum of the spans"
        );
    }

    #[test]
    fn spans_inside_a_readback_scope_are_counted_apart() {
        let counters = GpuCounters::default();
        drop(counters.span(HostSpan::Encode));
        {
            let _scope = counters.readback_scope();
            let _span = counters.span(HostSpan::Encode);
            std::thread::sleep(Duration::from_millis(5));
        }
        drop(counters.span(HostSpan::Encode));
        let snapshot = counters.snapshot();
        let in_readback = snapshot.readback_span(HostSpan::Encode);
        assert!(in_readback >= Duration::from_millis(5));
        assert!(
            snapshot.span(HostSpan::Encode) >= in_readback,
            "the readback part is part of the total"
        );
        assert!(
            snapshot.span(HostSpan::Encode) < in_readback + Duration::from_millis(20),
            "spans outside the scope are not in the readback part"
        );
        assert_eq!(snapshot.readback_span(HostSpan::Submit), Duration::ZERO);
    }

    #[test]
    fn span_names_are_unique_and_follow_the_order() {
        let names: Vec<&str> = HostSpan::ALL.iter().map(|s| s.name()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), HostSpan::COUNT);
        for (i, span) in HostSpan::ALL.iter().enumerate() {
            assert_eq!(span.index(), i);
        }
        assert_eq!(names[5], "wait");
    }
}
