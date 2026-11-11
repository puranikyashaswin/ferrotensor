//! Accounting of the GPU memory held by tensors, split by what it is for.
//!
//! Every [tensor](crate::UnifiedTensor) buffer is counted while it
//! is alive, under a [`MemoryCategory`]: the category current on the
//! context's [`MemoryTracker`] when the tensor was created (see
//! [`MemoryTracker::scope`]), or the one set later with
//! [`set_memory_category`](crate::UnifiedTensor::set_memory_category).
//! Buffers returned to the [`BufferPool`](crate::pool::BufferPool) and kept
//! for reuse are counted separately as idle pool bytes.
//!
//! The counters are relaxed atomics and always on. The breakdown at the
//! peak and the list of the largest buffers need a lock per allocation, so
//! they are only kept after [`MemoryTracker::set_detailed`]`(true)`.
//!
//! Sizes are what Ferrotensor asked wgpu for, after wgpu rounds every buffer
//! up to a multiple of 4 bytes (see [`memory::wgpu_buffer_size`](crate::memory::wgpu_buffer_size)).
//! The driver and its allocator add their own alignment and block overhead,
//! and pipelines, shaders, uniform buffers and upload staging copies are not
//! tensors, so the totals are lower than what `nvidia-smi` or Activity
//! Monitor show.

use crate::DType;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

/// What a GPU buffer is used for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemoryCategory {
    /// Frozen model weights (f16 or f32), uploaded when the model is built.
    BaseWeights = 0,
    /// Parameters the optimizer updates: LoRA adapters or trainable base weights.
    Trainable = 1,
    /// Optimizer state, e.g. the AdamW moments.
    OptimizerState = 2,
    /// Forward-pass outputs and intermediates (including those the autograd
    /// graph keeps for backward).
    Activations = 3,
    /// Accumulated gradients of the trainable parameters.
    Gradients = 4,
    /// Other buffers made during backward: gradients of activations and
    /// kernel temporaries.
    BackwardScratch = 5,
    /// Vocabulary-sized loss buffers: logits, `d loss / d logits`, per-row losses.
    Loss = 6,
    /// Readback buffers that copy results to the host.
    Staging = 7,
    /// Anything created outside a tagged scope.
    Other = 8,
}

impl MemoryCategory {
    /// Number of categories.
    pub const COUNT: usize = 9;

    /// Every category, in report order (the index of each is `c as usize`).
    pub const ALL: [MemoryCategory; Self::COUNT] = [
        Self::BaseWeights,
        Self::Trainable,
        Self::OptimizerState,
        Self::Activations,
        Self::Gradients,
        Self::BackwardScratch,
        Self::Loss,
        Self::Staging,
        Self::Other,
    ];

    /// Snake-case name, used as a JSON key.
    pub fn name(self) -> &'static str {
        match self {
            Self::BaseWeights => "base_weights",
            Self::Trainable => "trainable",
            Self::OptimizerState => "optimizer_state",
            Self::Activations => "activations",
            Self::Gradients => "gradients",
            Self::BackwardScratch => "backward_scratch",
            Self::Loss => "loss",
            Self::Staging => "staging",
            Self::Other => "other",
        }
    }

    /// Human-readable label for tables.
    pub fn label(self) -> &'static str {
        match self {
            Self::BaseWeights => "base weights (frozen)",
            Self::Trainable => "trainable parameters",
            Self::OptimizerState => "optimizer state",
            Self::Activations => "activations",
            Self::Gradients => "parameter gradients",
            Self::BackwardScratch => "backward scratch",
            Self::Loss => "loss (vocab-sized)",
            Self::Staging => "readback staging",
            Self::Other => "other",
        }
    }

    fn from_index(i: u8) -> Self {
        Self::ALL[i as usize]
    }
}

/// Bytes wgpu allocates for a buffer of `requested` bytes: rounded up to a
/// multiple of 4 (`COPY_BUFFER_ALIGNMENT`), and 4 for an empty buffer.
pub fn wgpu_buffer_size(requested: u64) -> u64 {
    requested.max(1).next_multiple_of(4)
}

/// Bytes held by one category.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CategoryBytes {
    /// Bytes allocated (after wgpu's rounding).
    pub allocated_bytes: u64,
    /// Bytes the tensors' shapes and dtypes need.
    pub logical_bytes: u64,
    /// Part of `allocated_bytes` in f16 tensors.
    pub f16_bytes: u64,
    /// Number of live buffers.
    pub buffers: u64,
}

/// One live buffer, as listed in [`MemorySnapshot::largest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferRecord {
    /// Its category.
    pub category: MemoryCategory,
    /// Element type of the tensor.
    pub dtype: DType,
    /// Shape of the tensor when it was created.
    pub shape: Vec<usize>,
    /// Bytes the shape and dtype need.
    pub logical_bytes: u64,
    /// Bytes allocated.
    pub allocated_bytes: u64,
    id: u64,
}

/// Bytes held at one moment.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemorySnapshot {
    /// Bytes allocated to live tensors (the sum over `categories`).
    pub live_bytes: u64,
    /// Bytes the live tensors' shapes and dtypes need.
    pub logical_bytes: u64,
    /// Bytes of freed buffers the pool keeps for reuse.
    pub pool_idle_bytes: u64,
    /// Per category, indexed by `MemoryCategory as usize`.
    pub categories: [CategoryBytes; MemoryCategory::COUNT],
    /// The largest live buffers, largest first (at most
    /// [`LARGEST_BUFFERS`]); empty unless detailed tracking is on.
    pub largest: Vec<BufferRecord>,
}

impl MemorySnapshot {
    /// Bytes of `category`.
    pub fn category(&self, category: MemoryCategory) -> &CategoryBytes {
        &self.categories[category as usize]
    }
}

/// How many buffers [`MemorySnapshot::largest`] lists.
pub const LARGEST_BUFFERS: usize = 10;

#[derive(Default)]
struct Counters {
    allocated: AtomicU64,
    logical: AtomicU64,
    f16: AtomicU64,
    buffers: AtomicU64,
}

#[derive(Default)]
struct Registry {
    next_id: u64,
    live: HashMap<u64, BufferRecord>,
    /// The detailed snapshot at the peak, and the last buffer id issued
    /// when it was taken: a buffer alive now with an id up to that (or with
    /// no id) was alive then.
    peak: Option<(u64, MemorySnapshot)>,
}

/// Live, peak and pooled byte counters of one context's tensors.
///
/// Reach it through [`WgpuContext::memory`](crate::WgpuContext::memory).
pub struct MemoryTracker {
    current: AtomicU8,
    categories: [Counters; MemoryCategory::COUNT],
    live: AtomicU64,
    peak_live: AtomicU64,
    pool_idle: AtomicU64,
    peak_held: AtomicU64,
    detailed: AtomicBool,
    registry: Mutex<Registry>,
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self {
            current: AtomicU8::new(MemoryCategory::Other as u8),
            categories: Default::default(),
            live: AtomicU64::new(0),
            peak_live: AtomicU64::new(0),
            pool_idle: AtomicU64::new(0),
            peak_held: AtomicU64::new(0),
            detailed: AtomicBool::new(false),
            registry: Mutex::new(Registry::default()),
        }
    }
}

impl std::fmt::Debug for MemoryTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryTracker")
            .field("live_bytes", &self.live_bytes())
            .field("peak_live_bytes", &self.peak_live_bytes())
            .field("pool_idle_bytes", &self.pool_idle_bytes())
            .finish()
    }
}

/// Restores the previous category when dropped; see [`MemoryTracker::scope`].
#[must_use = "the category applies only while the guard is alive"]
pub struct MemoryScope<'a> {
    tracker: &'a MemoryTracker,
    previous: u8,
}

impl Drop for MemoryScope<'_> {
    fn drop(&mut self) {
        self.tracker.current.store(self.previous, Relaxed);
    }
}

/// A counted buffer; releases its bytes when dropped.
pub(crate) struct Allocation {
    tracker: Arc<MemoryTracker>,
    id: u64,
    category: AtomicU8,
    f16: bool,
    logical: u64,
    allocated: u64,
}

impl std::fmt::Debug for Allocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Allocation")
            .field("category", &self.category())
            .field("allocated", &self.allocated)
            .finish()
    }
}

impl Allocation {
    pub(crate) fn category(&self) -> MemoryCategory {
        MemoryCategory::from_index(self.category.load(Relaxed))
    }

    pub(crate) fn set_category(&self, category: MemoryCategory) {
        self.tracker.retag(self, category);
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        self.tracker.release(self);
    }
}

impl MemoryTracker {
    /// A tracker with all counters at zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Tag tensors created until the guard drops with `category`.
    pub fn scope(&self, category: MemoryCategory) -> MemoryScope<'_> {
        let previous = self.current.swap(category as u8, Relaxed);
        MemoryScope {
            tracker: self,
            previous,
        }
    }

    /// The category new tensors get.
    pub fn category(&self) -> MemoryCategory {
        MemoryCategory::from_index(self.current.load(Relaxed))
    }

    /// Keep the per-category breakdown at the peak and the largest live
    /// buffers (costs a lock per allocation). Buffers created while this
    /// was off are counted but not listed in
    /// [`largest`](MemorySnapshot::largest).
    pub fn set_detailed(&self, on: bool) {
        self.detailed.store(on, Relaxed);
        if on {
            self.reset_peak();
        }
    }

    /// Whether detailed tracking is on.
    pub fn is_detailed(&self) -> bool {
        self.detailed.load(Relaxed)
    }

    /// Bytes allocated to live tensors.
    pub fn live_bytes(&self) -> u64 {
        self.live.load(Relaxed)
    }

    /// Highest [`live_bytes`](Self::live_bytes) since the last
    /// [`reset_peak`](Self::reset_peak).
    pub fn peak_live_bytes(&self) -> u64 {
        self.peak_live.load(Relaxed)
    }

    /// Bytes of freed buffers the pool keeps for reuse.
    pub fn pool_idle_bytes(&self) -> u64 {
        self.pool_idle.load(Relaxed)
    }

    /// Highest live plus idle pool bytes since the last
    /// [`reset_peak`](Self::reset_peak).
    pub fn peak_held_bytes(&self) -> u64 {
        self.peak_held.load(Relaxed)
    }

    /// Start a new peak from the current state.
    pub fn reset_peak(&self) {
        let live = self.live_bytes();
        self.peak_live.store(live, Relaxed);
        self.peak_held.store(live + self.pool_idle_bytes(), Relaxed);
        if self.is_detailed() {
            let mut reg = self.registry.lock().unwrap();
            let snap = self.snapshot(&reg);
            reg.peak = Some((reg.next_id, snap));
        }
    }

    /// The bytes held now (with the largest buffers when detailed).
    pub fn current(&self) -> MemorySnapshot {
        if self.is_detailed() {
            let reg = self.registry.lock().unwrap();
            self.snapshot(&reg)
        } else {
            self.snapshot(&Registry::default())
        }
    }

    /// The bytes held when [`peak_live_bytes`](Self::peak_live_bytes) was
    /// reached, if detailed tracking was on since the last reset.
    pub fn peak(&self) -> Option<MemorySnapshot> {
        let reg = self.registry.lock().unwrap();
        reg.peak.as_ref().map(|(_, s)| s.clone())
    }

    fn snapshot(&self, reg: &Registry) -> MemorySnapshot {
        let mut snap = MemorySnapshot {
            pool_idle_bytes: self.pool_idle_bytes(),
            ..MemorySnapshot::default()
        };
        for (out, c) in snap.categories.iter_mut().zip(&self.categories) {
            *out = CategoryBytes {
                allocated_bytes: c.allocated.load(Relaxed),
                logical_bytes: c.logical.load(Relaxed),
                f16_bytes: c.f16.load(Relaxed),
                buffers: c.buffers.load(Relaxed),
            };
            snap.live_bytes += out.allocated_bytes;
            snap.logical_bytes += out.logical_bytes;
        }
        let mut largest: Vec<&BufferRecord> = Vec::with_capacity(LARGEST_BUFFERS + 1);
        for rec in reg.live.values() {
            let pos = largest.partition_point(|r| {
                (r.allocated_bytes, std::cmp::Reverse(r.id))
                    >= (rec.allocated_bytes, std::cmp::Reverse(rec.id))
            });
            if pos < LARGEST_BUFFERS {
                largest.insert(pos, rec);
                largest.truncate(LARGEST_BUFFERS);
            }
        }
        snap.largest = largest.into_iter().cloned().collect();
        snap
    }

    fn adjust(&self, category: u8, f16: bool, logical: u64, allocated: u64, add: bool) {
        let c = &self.categories[category as usize];
        let op = |a: &AtomicU64, v: u64| {
            if add {
                a.fetch_add(v, Relaxed);
            } else {
                a.fetch_sub(v, Relaxed);
            }
        };
        op(&c.allocated, allocated);
        op(&c.logical, logical);
        if f16 {
            op(&c.f16, allocated);
        }
        op(&c.buffers, 1);
    }

    /// Count a new buffer of `requested` bytes (before wgpu's rounding)
    /// holding a tensor of `shape` and `dtype` that needs `logical` bytes.
    pub(crate) fn allocate(
        self: &Arc<Self>,
        dtype: DType,
        shape: &[usize],
        logical: u64,
        requested: u64,
    ) -> Allocation {
        let allocated = wgpu_buffer_size(requested);
        let category = self.current.load(Relaxed);
        let f16 = dtype == DType::FP16;
        self.adjust(category, f16, logical, allocated, true);
        let live = self.live.fetch_add(allocated, Relaxed) + allocated;
        self.peak_held
            .fetch_max(live + self.pool_idle_bytes(), Relaxed);
        let previous_peak = self.peak_live.fetch_max(live, Relaxed);
        let mut id = 0;
        if self.is_detailed() {
            let mut reg = self.registry.lock().unwrap();
            reg.next_id += 1;
            id = reg.next_id;
            reg.live.insert(
                id,
                BufferRecord {
                    category: MemoryCategory::from_index(category),
                    dtype,
                    shape: shape.to_vec(),
                    logical_bytes: logical,
                    allocated_bytes: allocated,
                    id,
                },
            );
            if live > previous_peak || reg.peak.is_none() {
                let snap = self.snapshot(&reg);
                reg.peak = Some((id, snap));
            }
        }
        Allocation {
            tracker: self.clone(),
            id,
            category: AtomicU8::new(category),
            f16,
            logical,
            allocated,
        }
    }

    /// Count a buffer of `requested` bytes that is not a tensor (e.g. a
    /// readback copy) until the returned value drops, under `category` or
    /// the current one.
    pub(crate) fn allocate_raw(
        self: &Arc<Self>,
        category: Option<MemoryCategory>,
        requested: u64,
    ) -> Allocation {
        let _scope = self.scope(category.unwrap_or_else(|| self.category()));
        self.allocate(DType::FP32, &[requested as usize], requested, requested)
    }

    fn release(&self, a: &Allocation) {
        self.adjust(
            a.category.load(Relaxed),
            a.f16,
            a.logical,
            a.allocated,
            false,
        );
        self.live.fetch_sub(a.allocated, Relaxed);
        if a.id != 0 {
            self.registry.lock().unwrap().live.remove(&a.id);
        }
    }

    fn retag(&self, a: &Allocation, category: MemoryCategory) {
        let old = a.category.swap(category as u8, Relaxed);
        if old == category as u8 {
            return;
        }
        self.adjust(old, a.f16, a.logical, a.allocated, false);
        self.adjust(category as u8, a.f16, a.logical, a.allocated, true);
        if a.id == 0 && !self.is_detailed() {
            return;
        }
        let mut reg = self.registry.lock().unwrap();
        if let Some(rec) = reg.live.get_mut(&a.id) {
            rec.category = category;
        }
        // The buffer is alive now; if it already existed when the peak was
        // taken, it was alive then too, so move it in the peak breakdown.
        if let Some((last_id, snap)) = reg.peak.as_mut() {
            if a.id <= *last_id {
                let from = &mut snap.categories[old as usize];
                from.allocated_bytes -= a.allocated;
                from.logical_bytes -= a.logical;
                from.buffers -= 1;
                if a.f16 {
                    from.f16_bytes -= a.allocated;
                }
                let to = &mut snap.categories[category as usize];
                to.allocated_bytes += a.allocated;
                to.logical_bytes += a.logical;
                to.buffers += 1;
                if a.f16 {
                    to.f16_bytes += a.allocated;
                }
                if a.id != 0 {
                    for rec in snap.largest.iter_mut().filter(|r| r.id == a.id) {
                        rec.category = category;
                    }
                }
            }
        }
    }

    /// A pooled buffer of `requested` bytes was handed out again.
    pub(crate) fn pool_take(&self, requested: u64) {
        self.pool_idle
            .fetch_sub(wgpu_buffer_size(requested), Relaxed);
    }

    /// A buffer of `requested` bytes was kept in the pool for reuse.
    pub(crate) fn pool_put(&self, requested: u64) {
        let idle = self
            .pool_idle
            .fetch_add(wgpu_buffer_size(requested), Relaxed)
            + wgpu_buffer_size(requested);
        self.peak_held.fetch_max(self.live_bytes() + idle, Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker() -> Arc<MemoryTracker> {
        Arc::new(MemoryTracker::new())
    }

    fn sum(s: &MemorySnapshot) -> u64 {
        s.categories.iter().map(|c| c.allocated_bytes).sum()
    }

    #[test]
    fn wgpu_rounds_to_four_bytes() {
        assert_eq!(wgpu_buffer_size(0), 4);
        assert_eq!(wgpu_buffer_size(1), 4);
        assert_eq!(wgpu_buffer_size(4), 4);
        assert_eq!(wgpu_buffer_size(6), 8);
        assert_eq!(wgpu_buffer_size(4096), 4096);
    }

    #[test]
    fn allocate_and_release_update_live_and_peak() {
        let t = tracker();
        let a = t.allocate(DType::FP32, &[4], 16, 16);
        let b = t.allocate(DType::FP16, &[3], 6, 6);
        assert_eq!(t.live_bytes(), 16 + 8);
        assert_eq!(t.peak_live_bytes(), 24);
        drop(a);
        assert_eq!(t.live_bytes(), 8);
        assert_eq!(t.peak_live_bytes(), 24);
        assert!(t.peak_live_bytes() >= t.live_bytes());
        let snap = t.current();
        assert_eq!(snap.logical_bytes, 6);
        assert_eq!(snap.category(MemoryCategory::Other).f16_bytes, 8);
        drop(b);
        assert_eq!(t.live_bytes(), 0);
        assert_eq!(t.current(), MemorySnapshot::default());
    }

    #[test]
    fn pool_reuse_moves_bytes_between_live_and_idle() {
        let t = tracker();
        let a = t.allocate(DType::FP32, &[256], 1024, 1024);
        // A tensor drop releases the allocation, then the pool keeps it.
        drop(a);
        t.pool_put(1024);
        assert_eq!((t.live_bytes(), t.pool_idle_bytes()), (0, 1024));
        assert_eq!(t.peak_held_bytes(), 1024);
        // Reuse: taken from the pool, counted live again.
        t.pool_take(1024);
        let b = t.allocate(DType::FP32, &[256], 1024, 1024);
        assert_eq!((t.live_bytes(), t.pool_idle_bytes()), (1024, 0));
        assert_eq!(t.peak_held_bytes(), 1024, "reuse holds no extra bytes");
        drop(b);
    }

    #[test]
    fn scopes_tag_and_categories_sum_to_live() {
        let t = tracker();
        t.set_detailed(true);
        let w = {
            let _s = t.scope(MemoryCategory::BaseWeights);
            let w = t.allocate(DType::FP16, &[10], 20, 20);
            {
                let _inner = t.scope(MemoryCategory::Trainable);
                assert_eq!(t.category(), MemoryCategory::Trainable);
            }
            assert_eq!(t.category(), MemoryCategory::BaseWeights);
            w
        };
        assert_eq!(t.category(), MemoryCategory::Other);
        let act = {
            let _s = t.scope(MemoryCategory::Activations);
            t.allocate(DType::FP32, &[100], 400, 400)
        };
        let snap = t.current();
        assert_eq!(
            snap.category(MemoryCategory::BaseWeights).allocated_bytes,
            20
        );
        assert_eq!(snap.category(MemoryCategory::BaseWeights).f16_bytes, 20);
        assert_eq!(
            snap.category(MemoryCategory::Activations).allocated_bytes,
            400
        );
        assert_eq!(sum(&snap), snap.live_bytes);
        assert_eq!(snap.live_bytes, t.live_bytes());
        assert_eq!(snap.largest.len(), 2);
        assert_eq!(snap.largest[0].allocated_bytes, 400);
        drop((w, act));
    }

    #[test]
    fn peak_snapshot_is_kept_and_follows_retags() {
        let t = tracker();
        t.set_detailed(true);
        let logits = t.allocate(DType::FP32, &[2, 8], 64, 64);
        let tmp = t.allocate(DType::FP32, &[32], 128, 128);
        drop(tmp);
        let peak = t.peak().expect("peak");
        assert_eq!(peak.live_bytes, 192);
        assert_eq!(sum(&peak), 192);
        assert!(t.peak_live_bytes() >= t.live_bytes());

        logits.set_category(MemoryCategory::Loss);
        let peak = t.peak().expect("peak");
        assert_eq!(peak.category(MemoryCategory::Loss).allocated_bytes, 64);
        assert_eq!(peak.category(MemoryCategory::Other).allocated_bytes, 128);
        assert_eq!(sum(&peak), peak.live_bytes);
        assert_eq!(
            peak.largest
                .iter()
                .find(|r| r.allocated_bytes == 64)
                .map(|r| r.category),
            Some(MemoryCategory::Loss)
        );

        // A new peak window starts from what is live now.
        t.reset_peak();
        assert_eq!(t.peak_live_bytes(), 64);
        assert_eq!(t.peak().expect("peak").live_bytes, 64);
        drop(logits);
        assert_eq!(t.live_bytes(), 0);
    }

    #[test]
    fn largest_buffers_are_capped_and_ordered() {
        let t = tracker();
        t.set_detailed(true);
        let held: Vec<_> = (1..=15u64)
            .map(|i| t.allocate(DType::FP32, &[i as usize], i * 4, i * 4))
            .collect();
        let snap = t.current();
        assert_eq!(snap.largest.len(), LARGEST_BUFFERS);
        assert_eq!(snap.largest[0].allocated_bytes, 60);
        assert!(snap
            .largest
            .windows(2)
            .all(|w| w[0].allocated_bytes >= w[1].allocated_bytes));
        drop(held);
        assert!(t.current().largest.is_empty());
    }
}
