//! GPU time per kernel from timestamp queries, for a context opened with
//! [`KernelTiming::On`] on an adapter that has them.
//!
//! Every dispatch gets a pair of timestamps in a query set that holds one
//! submission's dispatches. At each submission the pairs are resolved and
//! copied into a buffer, which is mapped once the GPU has run them; the next
//! wait for the GPU reads the mapped buffers and adds each pair's difference
//! to its kernel. Timing never adds a wait of its own.
//!
//! A timestamp records when the GPU reached that point of its command
//! stream, so a kernel's time includes the barrier before it, and the sum
//! over kernels leaves out time the GPU spent idle between them. The table
//! attributes GPU time to kernels; it is not an exact budget.

use crate::context::Dispatch;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use wgpu::{Buffer, CommandEncoder, Device, QuerySet};

/// Whether a context measures the GPU time of each kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelTiming {
    /// No timestamps; the context records and submits work unchanged.
    Off,
    /// Timestamps around every dispatch, when the adapter supports them.
    On,
}

/// Where a timing context writes its timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelTimingMode {
    /// Before and after each dispatch, inside compute passes that up to 8
    /// dispatches share, as without timing.
    InsidePasses,
    /// At the start and end of compute passes of one dispatch each, on
    /// adapters that cannot write timestamps inside a pass.
    PassBoundaries,
}

impl KernelTimingMode {
    /// The mode an adapter with `features` supports; `None` without
    /// timestamp queries.
    pub fn for_features(features: wgpu::Features) -> Option<Self> {
        if features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES) {
            Some(Self::InsidePasses)
        } else if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            Some(Self::PassBoundaries)
        } else {
            None
        }
    }

    /// Snake-case name, as reports write it.
    pub fn name(self) -> &'static str {
        match self {
            Self::InsidePasses => "inside_passes",
            Self::PassBoundaries => "pass_boundaries",
        }
    }

    pub(crate) fn features(self) -> wgpu::Features {
        match self {
            Self::InsidePasses => {
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
            }
            Self::PassBoundaries => wgpu::Features::TIMESTAMP_QUERY,
        }
    }
}

/// What a dispatch ran: the kernel's entry point and, for matmul, the
/// operand layout and a size class (empty for other kernels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct KernelLabel {
    pub(crate) entry: &'static str,
    pub(crate) layout: &'static str,
    pub(crate) size_class: &'static str,
}

impl KernelLabel {
    fn name(self) -> String {
        if self.layout.is_empty() {
            self.entry.to_owned()
        } else {
            format!("{} ({}, {})", self.entry, self.layout, self.size_class)
        }
    }
}

/// How often a kernel ran and its summed GPU time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KernelTime {
    /// Dispatches timed.
    pub dispatches: u64,
    /// Sum of their GPU times.
    pub gpu_time: Duration,
}

/// GPU time by kernel name since a context opened, or between two moments.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KernelTimes(BTreeMap<String, KernelTime>);

impl FromIterator<(String, KernelTime)> for KernelTimes {
    fn from_iter<I: IntoIterator<Item = (String, KernelTime)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl KernelTimes {
    /// The kernels timed between `earlier` and `self`.
    pub fn since(&self, earlier: &Self) -> Self {
        self.0
            .iter()
            .filter_map(|(name, now)| {
                let before = earlier.0.get(name).copied().unwrap_or_default();
                let dispatches = now.dispatches.saturating_sub(before.dispatches);
                (dispatches > 0).then(|| {
                    let gpu_time = now.gpu_time.saturating_sub(before.gpu_time);
                    (
                        name.clone(),
                        KernelTime {
                            dispatches,
                            gpu_time,
                        },
                    )
                })
            })
            .collect()
    }

    /// Every kernel, longest GPU time first.
    pub fn by_gpu_time(&self) -> Vec<(&str, KernelTime)> {
        let mut rows: Vec<_> = self.0.iter().map(|(n, t)| (n.as_str(), *t)).collect();
        rows.sort_by(|a, b| b.1.gpu_time.cmp(&a.1.gpu_time).then(a.0.cmp(b.0)));
        rows
    }

    /// Sum of the GPU times of all kernels.
    pub fn total(&self) -> Duration {
        self.0.values().map(|t| t.gpu_time).sum()
    }
}

/// Adds the time between each pair of `ticks` to the kernel of the same
/// index in `labels`. A pair that runs backwards (a counter that wrapped)
/// adds a dispatch with no time.
fn add_timestamp_pairs(
    totals: &mut HashMap<KernelLabel, KernelTime>,
    labels: &[KernelLabel],
    ticks: &[u64],
    period_ns: f32,
) {
    for (label, pair) in labels.iter().zip(ticks.as_chunks::<2>().0) {
        let total = totals.entry(*label).or_default();
        total.dispatches += 1;
        let nanos = pair[1].saturating_sub(pair[0]) as f64 * f64::from(period_ns);
        total.gpu_time += Duration::from_nanos(nanos as u64);
    }
}

const MAPPING: u8 = 0;
const MAPPED: u8 = 1;
const MAP_FAILED: u8 = 2;

/// A query set for one submission's timestamps and the buffers they are
/// resolved and read back through.
#[derive(Debug)]
pub(crate) struct TimingSlot {
    queries: QuerySet,
    resolved: Buffer,
    readback: Buffer,
}

/// A submitted slot whose readback buffer is being mapped.
#[derive(Debug)]
struct InFlight {
    slot: TimingSlot,
    labels: Vec<KernelLabel>,
    state: Arc<AtomicU8>,
}

/// Dispatches encoded into the pending command encoder, and the slot their
/// timestamps go to.
#[derive(Debug, Default)]
pub(crate) struct PendingTimes {
    slot: Option<TimingSlot>,
    labels: Vec<KernelLabel>,
}

#[derive(Debug, Default)]
struct TimerState {
    free: Vec<TimingSlot>,
    in_flight: Vec<InFlight>,
    totals: HashMap<KernelLabel, KernelTime>,
}

/// Timestamp queries of one context.
#[derive(Debug)]
pub(crate) struct KernelTimer {
    pub(crate) mode: KernelTimingMode,
    /// Nanoseconds per timestamp tick.
    period_ns: f32,
    /// Timestamps one submission can hold: two per dispatch.
    queries_per_submit: u32,
    state: Mutex<TimerState>,
}

impl KernelTimer {
    /// A timer for submissions of at most `dispatches_per_submit` dispatches.
    pub(crate) fn new(
        mode: KernelTimingMode,
        queue: &wgpu::Queue,
        dispatches_per_submit: u32,
    ) -> Self {
        Self {
            mode,
            period_ns: queue.get_timestamp_period(),
            queries_per_submit: 2 * dispatches_per_submit,
            state: Mutex::default(),
        }
    }

    fn slot(&self, device: &Device) -> TimingSlot {
        if let Some(slot) = self.lock().free.pop() {
            return slot;
        }
        let bytes = u64::from(self.queries_per_submit) * u64::from(wgpu::QUERY_SIZE);
        let buffer = |usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Kernel timestamps"),
                size: bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        TimingSlot {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Kernel timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: self.queries_per_submit,
            }),
            resolved: buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
            readback: buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
        }
    }

    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    fn lock(&self) -> std::sync::MutexGuard<'_, TimerState> {
        self.state.lock().unwrap()
    }

    /// Encodes `dispatches` with a timestamp pair each, and returns how many
    /// compute passes that took.
    pub(crate) fn encode(
        &self,
        device: &Device,
        encoder: &mut CommandEncoder,
        dispatches: &[Dispatch],
        times: &mut PendingTimes,
    ) -> u64 {
        let PendingTimes { slot, labels } = times;
        let queries = &slot.get_or_insert_with(|| self.slot(device)).queries;
        debug_assert!(
            2 * (labels.len() + dispatches.len()) <= self.queries_per_submit as usize,
            "a submission holds at most queries_per_submit / 2 dispatches"
        );
        let mut next_pair = |dispatch: &Dispatch| {
            let first = 2 * labels.len() as u32;
            labels.push(dispatch.label);
            first
        };
        match self.mode {
            KernelTimingMode::InsidePasses => {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                for dispatch in dispatches {
                    let first = next_pair(dispatch);
                    pass.set_pipeline(&dispatch.pipeline);
                    pass.set_bind_group(0, &dispatch.bind_group, &[]);
                    pass.write_timestamp(queries, first);
                    let (x, y, z) = dispatch.groups;
                    pass.dispatch_workgroups(x, y, z);
                    pass.write_timestamp(queries, first + 1);
                }
                1
            }
            KernelTimingMode::PassBoundaries => {
                for dispatch in dispatches {
                    let first = next_pair(dispatch);
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: queries,
                            beginning_of_pass_write_index: Some(first),
                            end_of_pass_write_index: Some(first + 1),
                        }),
                    });
                    pass.set_pipeline(&dispatch.pipeline);
                    pass.set_bind_group(0, &dispatch.bind_group, &[]);
                    let (x, y, z) = dispatch.groups;
                    pass.dispatch_workgroups(x, y, z);
                }
                dispatches.len() as u64
            }
        }
    }

    /// Records the copy of this submission's timestamps into its readback
    /// buffer; call it before the encoder is finished.
    pub(crate) fn resolve(&self, encoder: &mut CommandEncoder, times: &PendingTimes) {
        if let Some(slot) = &times.slot {
            let count = 2 * times.labels.len() as u32;
            encoder.resolve_query_set(&slot.queries, 0..count, &slot.resolved, 0);
            let bytes = u64::from(count) * u64::from(wgpu::QUERY_SIZE);
            encoder.copy_buffer_to_buffer(&slot.resolved, 0, &slot.readback, 0, bytes);
        }
    }

    /// Starts mapping the readback buffer of the submission just made.
    pub(crate) fn submitted(&self, times: &mut PendingTimes) {
        let Some(slot) = times.slot.take() else {
            return;
        };
        let state = Arc::new(AtomicU8::new(MAPPING));
        let flag = Arc::clone(&state);
        slot.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let done = if result.is_ok() { MAPPED } else { MAP_FAILED };
                flag.store(done, Ordering::Release);
            });
        self.lock().in_flight.push(InFlight {
            slot,
            labels: std::mem::take(&mut times.labels),
            state,
        });
    }

    /// Adds the timestamps of every submission whose readback is mapped to
    /// the totals, and frees their slots for reuse.
    pub(crate) fn collect(&self) {
        let mut state = self.lock();
        let TimerState {
            free,
            in_flight,
            totals,
        } = &mut *state;
        let mut i = 0;
        while i < in_flight.len() {
            match in_flight[i].state.load(Ordering::Acquire) {
                MAPPING => i += 1,
                MAPPED => {
                    let done = in_flight.swap_remove(i);
                    let ticks: Vec<u64> = done
                        .slot
                        .readback
                        .slice(..)
                        .get_mapped_range()
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|bytes| u64::from_le_bytes(*bytes))
                        .collect();
                    add_timestamp_pairs(totals, &done.labels, &ticks, self.period_ns);
                    done.slot.readback.unmap();
                    free.push(done.slot);
                }
                // Mapping fails only when the device is lost; the slot and
                // its timestamps are dropped.
                _ => {
                    in_flight.swap_remove(i);
                }
            }
        }
    }

    /// Totals of every timestamp collected so far.
    pub(crate) fn totals(&self) -> KernelTimes {
        self.lock()
            .totals
            .iter()
            .map(|(label, time)| (label.name(), *time))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MATMUL_LARGE: KernelLabel = KernelLabel {
        entry: "matmul_blocked",
        layout: "a@b",
        size_class: "out>=1M",
    };
    const ROPE: KernelLabel = KernelLabel {
        entry: "rope",
        layout: "",
        size_class: "",
    };

    #[test]
    fn pairs_add_to_their_kernel_scaled_by_the_period() {
        let mut totals = HashMap::new();
        let labels = [MATMUL_LARGE, ROPE, MATMUL_LARGE];
        // Ticks of 2 ns: 100, 7 and 50 ticks.
        let ticks = [1_000, 1_100, 1_100, 1_107, 2_000, 2_050];
        add_timestamp_pairs(&mut totals, &labels, &ticks, 2.0);
        // A wrapped counter: the end tick is below the start.
        add_timestamp_pairs(&mut totals, &[ROPE], &[500, 10], 2.0);

        assert_eq!(
            totals[&MATMUL_LARGE],
            KernelTime {
                dispatches: 2,
                gpu_time: Duration::from_nanos(300),
            }
        );
        assert_eq!(
            totals[&ROPE],
            KernelTime {
                dispatches: 2,
                gpu_time: Duration::from_nanos(14),
            }
        );
    }

    #[test]
    fn labels_name_matmul_variants() {
        assert_eq!(MATMUL_LARGE.name(), "matmul_blocked (a@b, out>=1M)");
        assert_eq!(ROPE.name(), "rope");
    }

    #[test]
    fn since_subtracts_and_drops_kernels_that_did_not_run() {
        let time = |dispatches, nanos| KernelTime {
            dispatches,
            gpu_time: Duration::from_nanos(nanos),
        };
        let earlier: KernelTimes = [
            ("rope".to_owned(), time(4, 400)),
            ("silu".to_owned(), time(2, 50)),
        ]
        .into_iter()
        .collect();
        let later: KernelTimes = [
            ("rope".to_owned(), time(6, 1_000)),
            ("silu".to_owned(), time(2, 50)),
            ("adamw_update".to_owned(), time(1, 2_000)),
        ]
        .into_iter()
        .collect();

        let step = later.since(&earlier);
        assert_eq!(
            step.by_gpu_time(),
            vec![("adamw_update", time(1, 2_000)), ("rope", time(2, 600))]
        );
        assert_eq!(step.total(), Duration::from_nanos(2_600));
        assert_eq!(later.since(&later), KernelTimes::default());
    }

    #[test]
    fn mode_follows_the_adapter_features() {
        use wgpu::Features as F;
        assert_eq!(KernelTimingMode::for_features(F::empty()), None);
        assert_eq!(
            KernelTimingMode::for_features(F::TIMESTAMP_QUERY),
            Some(KernelTimingMode::PassBoundaries)
        );
        assert_eq!(
            KernelTimingMode::for_features(F::TIMESTAMP_QUERY | F::TIMESTAMP_QUERY_INSIDE_PASSES),
            Some(KernelTimingMode::InsidePasses)
        );
    }
}
