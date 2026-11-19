//! Counts of the GPU work a [`WgpuContext`](crate::WgpuContext) does:
//! kernel dispatches, the compute passes, bind groups and uniform buffers
//! made or reused for them, submissions, device polls, waits, uploads and
//! readbacks.
//!
//! The counters are relaxed atomics and always on; recording an op adds to
//! a few of them. Take a [`GpuCounterSnapshot`] with
//! [`WgpuContext::gpu_counters`](crate::WgpuContext::gpu_counters) before and
//! after some work and subtract with [`GpuCounterSnapshot::since`].

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

/// Running totals of one context's GPU work.
#[derive(Debug, Default)]
pub(crate) struct GpuCounters {
    pub(crate) dispatches: AtomicU64,
    pub(crate) compute_passes: AtomicU64,
    pub(crate) bind_groups: AtomicU64,
    pub(crate) bind_groups_reused: AtomicU64,
    pub(crate) uniform_buffers: AtomicU64,
    pub(crate) uniform_buffers_reused: AtomicU64,
    pub(crate) uniform_bytes: AtomicU64,
    pub(crate) submits: AtomicU64,
    pub(crate) device_polls: AtomicU64,
    pub(crate) synchronizes: AtomicU64,
    pub(crate) forced_synchronizes: AtomicU64,
    pub(crate) upload_bytes: AtomicU64,
    pub(crate) readbacks: AtomicU64,
    pub(crate) readback_bytes: AtomicU64,
    pub(crate) gpu_wait_nanos: AtomicU64,
    pub(crate) readback_nanos: AtomicU64,
}

impl GpuCounters {
    /// Adds the wall time since `start` to `nanos`.
    pub(crate) fn add_elapsed(nanos: &AtomicU64, start: Instant) {
        let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        nanos.fetch_add(elapsed, Relaxed);
    }

    pub(crate) fn snapshot(&self) -> GpuCounterSnapshot {
        GpuCounterSnapshot {
            dispatches: self.dispatches.load(Relaxed),
            compute_passes: self.compute_passes.load(Relaxed),
            bind_groups: self.bind_groups.load(Relaxed),
            bind_groups_reused: self.bind_groups_reused.load(Relaxed),
            uniform_buffers: self.uniform_buffers.load(Relaxed),
            uniform_buffers_reused: self.uniform_buffers_reused.load(Relaxed),
            uniform_bytes: self.uniform_bytes.load(Relaxed),
            submits: self.submits.load(Relaxed),
            device_polls: self.device_polls.load(Relaxed),
            synchronizes: self.synchronizes.load(Relaxed),
            forced_synchronizes: self.forced_synchronizes.load(Relaxed),
            upload_bytes: self.upload_bytes.load(Relaxed),
            readbacks: self.readbacks.load(Relaxed),
            readback_bytes: self.readback_bytes.load(Relaxed),
            gpu_wait: Duration::from_nanos(self.gpu_wait_nanos.load(Relaxed)),
            readback_time: Duration::from_nanos(self.readback_nanos.load(Relaxed)),
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
    /// Bind groups created for dispatches.
    pub bind_groups: u64,
    /// Dispatches that reused a cached bind group instead.
    pub bind_groups_reused: u64,
    /// Uniform buffers created for kernel parameters.
    pub uniform_buffers: u64,
    /// Dispatches that reused a cached uniform buffer instead.
    pub uniform_buffers_reused: u64,
    /// Bytes of the uniform buffers created, as requested.
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
}

impl GpuCounterSnapshot {
    /// The work done between `earlier` and `self`.
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            dispatches: self.dispatches.saturating_sub(earlier.dispatches),
            compute_passes: self.compute_passes.saturating_sub(earlier.compute_passes),
            bind_groups: self.bind_groups.saturating_sub(earlier.bind_groups),
            bind_groups_reused: self
                .bind_groups_reused
                .saturating_sub(earlier.bind_groups_reused),
            uniform_buffers: self.uniform_buffers.saturating_sub(earlier.uniform_buffers),
            uniform_buffers_reused: self
                .uniform_buffers_reused
                .saturating_sub(earlier.uniform_buffers_reused),
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
        }
    }

    /// Every count (not the times) with its snake-case name, in report order.
    pub fn counts(&self) -> [(&'static str, u64); 14] {
        [
            ("dispatches", self.dispatches),
            ("compute_passes", self.compute_passes),
            ("bind_groups", self.bind_groups),
            ("bind_groups_reused", self.bind_groups_reused),
            ("uniform_buffers", self.uniform_buffers),
            ("uniform_buffers_reused", self.uniform_buffers_reused),
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
        counters.gpu_wait_nanos.fetch_add(1_000, Relaxed);
        let earlier = counters.snapshot();

        counters.dispatches.fetch_add(5, Relaxed);
        counters.compute_passes.fetch_add(2, Relaxed);
        counters.bind_groups.fetch_add(5, Relaxed);
        counters.bind_groups_reused.fetch_add(7, Relaxed);
        counters.uniform_buffers.fetch_add(2, Relaxed);
        counters.uniform_buffers_reused.fetch_add(3, Relaxed);
        counters.uniform_bytes.fetch_add(32, Relaxed);
        counters.submits.fetch_add(1, Relaxed);
        counters.device_polls.fetch_add(7, Relaxed);
        counters.synchronizes.fetch_add(2, Relaxed);
        counters.forced_synchronizes.fetch_add(1, Relaxed);
        counters.upload_bytes.fetch_add(4096, Relaxed);
        counters.readbacks.fetch_add(1, Relaxed);
        counters.readback_bytes.fetch_add(64, Relaxed);
        counters.gpu_wait_nanos.fetch_add(2_500, Relaxed);
        counters.readback_nanos.fetch_add(700, Relaxed);
        let later = counters.snapshot();

        assert_eq!(
            later.since(&earlier),
            GpuCounterSnapshot {
                dispatches: 5,
                compute_passes: 2,
                bind_groups: 5,
                bind_groups_reused: 7,
                uniform_buffers: 2,
                uniform_buffers_reused: 3,
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
            }
        );
        assert_eq!(later.since(&later), GpuCounterSnapshot::default());
        assert_eq!(
            later.since(&earlier).counts()[0],
            ("dispatches", 5),
            "counts are named in report order"
        );
    }
}
