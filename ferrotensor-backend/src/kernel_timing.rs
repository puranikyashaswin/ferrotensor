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

/// One matmul product `out[m, n] = op(a)[m, k] @ op(b)[k, n]`: the kernel
/// that ran it and its exact dimensions.
///
/// Matmul kernels are 2-D, so there is no batch dimension: a caller with
/// several sequences folds them into `m` (rows of `x` are sequences times
/// padded length), and each dispatch is one product. The thin kernel may
/// compute the transposed product internally; the shape is always the
/// logical one of the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MatmulShape {
    /// The kernel's entry point.
    pub entry: &'static str,
    /// Operand layout: `a@b`, `a@bT` or `aT@b`.
    pub layout: &'static str,
    /// Rows of the output.
    pub m: u32,
    /// Columns of the output.
    pub n: u32,
    /// Summed dimension.
    pub k: u32,
}

/// What one timed dispatch was: its kernel label and, for a matmul, its
/// shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Timed {
    pub(crate) label: KernelLabel,
    pub(crate) shape: Option<MatmulShape>,
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

/// GPU time by matmul shape since a context opened, or between two moments.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MatmulShapes(BTreeMap<MatmulShape, KernelTime>);

impl FromIterator<(MatmulShape, KernelTime)> for MatmulShapes {
    fn from_iter<I: IntoIterator<Item = (MatmulShape, KernelTime)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl MatmulShapes {
    /// The shapes timed between `earlier` and `self`.
    pub fn since(&self, earlier: &Self) -> Self {
        self.0
            .iter()
            .filter_map(|(shape, now)| {
                let before = earlier.0.get(shape).copied().unwrap_or_default();
                let dispatches = now.dispatches.saturating_sub(before.dispatches);
                (dispatches > 0).then(|| {
                    let gpu_time = now.gpu_time.saturating_sub(before.gpu_time);
                    (
                        *shape,
                        KernelTime {
                            dispatches,
                            gpu_time,
                        },
                    )
                })
            })
            .collect()
    }

    /// Every shape, longest GPU time first.
    pub fn by_gpu_time(&self) -> Vec<(MatmulShape, KernelTime)> {
        let mut rows: Vec<_> = self.0.iter().map(|(s, t)| (*s, *t)).collect();
        rows.sort_by(|a, b| b.1.gpu_time.cmp(&a.1.gpu_time).then(a.0.cmp(&b.0)));
        rows
    }
}

/// Adds the time between each pair of `ticks` to the kernel of the same
/// index in `timed`, and to its matmul shape when it has one. A pair that
/// runs backwards (a counter that wrapped) adds a dispatch with no time.
fn add_timestamp_pairs(
    totals: &mut HashMap<KernelLabel, KernelTime>,
    shapes: &mut HashMap<MatmulShape, KernelTime>,
    timed: &[Timed],
    ticks: &[u64],
    period_ns: f32,
) {
    for (dispatch, pair) in timed.iter().zip(ticks.as_chunks::<2>().0) {
        let nanos = pair[1].saturating_sub(pair[0]) as f64 * f64::from(period_ns);
        let gpu_time = Duration::from_nanos(nanos as u64);
        let total = totals.entry(dispatch.label).or_default();
        total.dispatches += 1;
        total.gpu_time += gpu_time;
        if let Some(shape) = dispatch.shape {
            let total = shapes.entry(shape).or_default();
            total.dispatches += 1;
            total.gpu_time += gpu_time;
        }
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
    timed: Vec<Timed>,
    state: Arc<AtomicU8>,
}

/// Dispatches encoded into the pending command encoder, and the slot their
/// timestamps go to.
#[derive(Debug, Default)]
pub(crate) struct PendingTimes {
    slot: Option<TimingSlot>,
    timed: Vec<Timed>,
}

#[derive(Debug, Default)]
struct TimerState {
    free: Vec<TimingSlot>,
    in_flight: Vec<InFlight>,
    totals: HashMap<KernelLabel, KernelTime>,
    shapes: HashMap<MatmulShape, KernelTime>,
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
        let PendingTimes { slot, timed } = times;
        let queries = &slot.get_or_insert_with(|| self.slot(device)).queries;
        debug_assert!(
            2 * (timed.len() + dispatches.len()) <= self.queries_per_submit as usize,
            "a submission holds at most queries_per_submit / 2 dispatches"
        );
        let mut next_pair = |dispatch: &Dispatch| {
            let first = 2 * timed.len() as u32;
            timed.push(Timed {
                label: dispatch.label,
                shape: dispatch.shape,
            });
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
            let count = 2 * times.timed.len() as u32;
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
            timed: std::mem::take(&mut times.timed),
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
            shapes,
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
                    add_timestamp_pairs(totals, shapes, &done.timed, &ticks, self.period_ns);
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

    /// Totals by matmul shape of every timestamp collected so far.
    pub(crate) fn shape_totals(&self) -> MatmulShapes {
        self.lock()
            .shapes
            .iter()
            .map(|(shape, time)| (*shape, *time))
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

    const SHAPE: MatmulShape = MatmulShape {
        entry: "matmul_blocked",
        layout: "a@b",
        m: 66,
        n: 2048,
        k: 2048,
    };

    fn plain(label: KernelLabel) -> Timed {
        Timed { label, shape: None }
    }

    #[test]
    fn pairs_add_to_their_kernel_scaled_by_the_period() {
        let mut totals = HashMap::new();
        let mut shapes = HashMap::new();
        let timed = [plain(MATMUL_LARGE), plain(ROPE), plain(MATMUL_LARGE)];
        // Ticks of 2 ns: 100, 7 and 50 ticks.
        let ticks = [1_000, 1_100, 1_100, 1_107, 2_000, 2_050];
        add_timestamp_pairs(&mut totals, &mut shapes, &timed, &ticks, 2.0);
        // A wrapped counter: the end tick is below the start.
        add_timestamp_pairs(&mut totals, &mut shapes, &[plain(ROPE)], &[500, 10], 2.0);

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
        assert!(shapes.is_empty(), "dispatches without a shape add none");
    }

    #[test]
    fn shapes_add_up_per_distinct_shape_and_leave_kernel_totals_alone() {
        let other = MatmulShape { n: 8192, ..SHAPE };
        let with = |shape| Timed {
            label: MATMUL_LARGE,
            shape: Some(shape),
        };
        let mut totals = HashMap::new();
        let mut shapes = HashMap::new();
        let timed = [with(SHAPE), with(other), with(SHAPE), plain(ROPE)];
        // Ticks of 1 ns: 100, 450, 300 and 5 ticks.
        let ticks = [0, 100, 100, 550, 550, 850, 850, 855];
        add_timestamp_pairs(&mut totals, &mut shapes, &timed, &ticks, 1.0);

        assert_eq!(
            totals[&MATMUL_LARGE],
            KernelTime {
                dispatches: 3,
                gpu_time: Duration::from_nanos(850),
            }
        );
        let shapes: MatmulShapes = shapes.into_iter().collect();
        assert_eq!(
            shapes.by_gpu_time(),
            vec![
                (
                    other,
                    KernelTime {
                        dispatches: 1,
                        gpu_time: Duration::from_nanos(450),
                    }
                ),
                (
                    SHAPE,
                    KernelTime {
                        dispatches: 2,
                        gpu_time: Duration::from_nanos(400),
                    }
                ),
            ]
        );
    }

    #[test]
    fn shapes_since_subtracts_and_drops_shapes_that_did_not_run() {
        let time = |dispatches, nanos| KernelTime {
            dispatches,
            gpu_time: Duration::from_nanos(nanos),
        };
        let other = MatmulShape { k: 64, ..SHAPE };
        let earlier: MatmulShapes = [(SHAPE, time(2, 100)), (other, time(1, 10))]
            .into_iter()
            .collect();
        let later: MatmulShapes = [(SHAPE, time(5, 400)), (other, time(1, 10))]
            .into_iter()
            .collect();
        let step = later.since(&earlier);
        assert_eq!(step.by_gpu_time(), vec![(SHAPE, time(3, 300))]);
        assert_eq!(later.since(&later), MatmulShapes::default());
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
