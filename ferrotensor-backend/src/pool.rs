use crate::bind_cache::BindGroupCache;
use crate::memory::{wgpu_buffer_size, MemoryTracker};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

/// Default cap on the bytes a pool keeps for reuse (512 MiB).
pub const DEFAULT_MAX_POOLED_BYTES: u64 = 512 * 1024 * 1024;

/// Environment variable read when a [`WgpuContext`](crate::WgpuContext) is
/// created: `0`, `false`, `off` or `no` turn pool overflow off (see
/// [`BufferPool::set_overflow`]); unset or any other value leaves it on.
pub const POOL_OVERFLOW_ENV: &str = "FERROTENSOR_POOL_OVERFLOW";

/// Whether pool overflow is on for a [`POOL_OVERFLOW_ENV`] value
/// (`None` when the variable is unset).
pub fn pool_overflow_from_env_value(value: Option<&str>) -> bool {
    let value = value.map(|v| v.trim().to_ascii_lowercase());
    !matches!(value.as_deref(), Some("0" | "false" | "off" | "no"))
}

/// Smallest request, in bytes, that is rounded up to a size class; smaller
/// ones keep their exact size, which is at most 64 distinct values.
const MIN_CLASSED_BYTES: u64 = 256;

/// log2 of the classes per power of two: 8, so a class is at most 12.5%
/// above the sizes it serves.
const CLASS_STEPS_LOG2: u32 = 3;

/// The most a reused idle buffer may be larger than the class of the request
/// it serves, as a ratio: 2 means an idle buffer of up to twice the class is
/// taken when none of the class itself is idle.
const MAX_REUSE_RATIO: u64 = 2;

/// The size in bytes of the buffer a request of `requested` bytes is served
/// from.
///
/// The request is rounded as wgpu does, then up to the next of 8 evenly
/// spaced classes between two powers of two, so a class is at most 12.5%
/// above the requests it serves. Below 256 bytes it stays exact. A sequence
/// length that changes from one micro-batch to the next then lands on a few
/// classes, not on a new size every time.
pub fn size_class(requested: u64) -> u64 {
    let size = wgpu_buffer_size(requested);
    if size < MIN_CLASSED_BYTES {
        return size;
    }
    let power = 1u64 << (63 - size.leading_zeros());
    size.next_multiple_of(power >> CLASS_STEPS_LOG2)
}

/// Free lists of GPU buffers keyed by their size in bytes.
///
/// A tensor's buffer is returned when its last reference drops. A request is
/// rounded up to its [`size_class`] and served by an idle buffer of exactly
/// that size, else by the smallest idle buffer of up to twice that size, so a
/// buffer can be up to twice as large as the tensor in it.
/// Kernels still see the tensor's own size: the engine binds only the tensor's
/// bytes of such a buffer, not the whole of it, since kernels bound their
/// loops by `arrayLength`.
///
/// Reusing a buffer that recorded or submitted GPU work still reads is safe:
/// every op is recorded into one queue in program order, so the work of the
/// new owner runs after the old owner's, and wgpu inserts the barriers
/// between them.
///
/// # Cap and overflow
///
/// The pool keeps at most `max_pooled_bytes` of idle buffers once the GPU has
/// finished with them. Dropping a buffer frees nothing until the GPU has
/// finished the work recorded before the drop: wgpu destroys it only then.
/// So with *overflow* on (the default), buffers returned beyond the cap are
/// kept for reuse until that work is known to be finished, and only then
/// dropped ([`WgpuContext`](crate::WgpuContext) checks after each submission
/// and whenever it waits for the GPU, and polls the device so wgpu destroys
/// them at once). In a training step, where the GPU runs behind the host,
/// a later op of the same size takes such a buffer instead of allocating a
/// new one while the old one still awaits destruction.
///
/// With overflow off, buffers beyond the cap are dropped as they are
/// returned (the behaviour before overflow existed).
pub struct BufferPool {
    buffers: Mutex<PoolState>,
    usage: BufferUsages,
    tracker: Arc<MemoryTracker>,
    /// Bind groups of the context's dispatches; the pool evicts the buffers
    /// it lets go.
    bind_groups: Arc<BindGroupCache>,
    created: AtomicU64,
    reused: AtomicU64,
    /// Buffers in use that are larger than the tensor they hold, with the
    /// tensor's bytes after wgpu's rounding.
    bound: Mutex<HashMap<Buffer, u64>>,
}

struct PoolState {
    lists: FreeLists<Arc<Buffer>>,
    max_pooled_bytes: u64,
    overflow: bool,
}

/// Idle buffers by size; each list oldest first, every entry tagged with
/// the submission that was being recorded when it was returned.
struct FreeLists<B> {
    free: BTreeMap<u64, VecDeque<(u64, B)>>,
    pooled_bytes: u64,
}

impl<B> Default for FreeLists<B> {
    fn default() -> Self {
        Self {
            free: BTreeMap::new(),
            pooled_bytes: 0,
        }
    }
}

impl<B> FreeLists<B> {
    /// The most recently returned buffer of exactly `size` bytes.
    fn take(&mut self, size: u64) -> Option<B> {
        let list = self.free.get_mut(&size)?;
        let (_, buffer) = list.pop_back()?;
        if list.is_empty() {
            self.free.remove(&size);
        }
        self.pooled_bytes -= size;
        Some(buffer)
    }

    /// The most recently returned buffer of the smallest size from `size` to
    /// `max` bytes that has one, with that size.
    fn take_up_to(&mut self, size: u64, max: u64) -> Option<(u64, B)> {
        if size > max {
            return None;
        }
        let found = *self.free.range(size..=max).next()?.0;
        self.take(found).map(|buffer| (found, buffer))
    }

    fn put(&mut self, size: u64, serial: u64, buffer: B) {
        self.free
            .entry(size)
            .or_default()
            .push_back((serial, buffer));
        self.pooled_bytes += size;
    }

    /// Removes the oldest buffers returned while a submission up to `done`
    /// was being recorded, until at most `max` bytes remain. Buffers
    /// returned later are kept even above `max`.
    #[expect(
        clippy::expect_used,
        reason = "the size was just read from the free map, so its list exists and is non-empty"
    )]
    fn trim(&mut self, max: u64, done: u64) -> Vec<(u64, B)> {
        let mut out = Vec::new();
        while self.pooled_bytes > max {
            let oldest = self
                .free
                .iter()
                .filter_map(|(&size, list)| list.front().map(|&(serial, _)| (serial, size)))
                .filter(|&(serial, _)| serial <= done)
                .min();
            let Some((_, size)) = oldest else { break };
            let list = self.free.get_mut(&size).expect("listed size");
            let (_, buffer) = list.pop_front().expect("non-empty list");
            if list.is_empty() {
                self.free.remove(&size);
            }
            self.pooled_bytes -= size;
            out.push((size, buffer));
        }
        out
    }

    fn drain(&mut self) -> Vec<(u64, B)> {
        self.pooled_bytes = 0;
        std::mem::take(&mut self.free)
            .into_iter()
            .flat_map(|(size, list)| list.into_iter().map(move |(_, b)| (size, b)))
            .collect()
    }
}

impl std::fmt::Debug for BufferPool {
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.buffers.lock().unwrap();
        let counts: HashMap<u64, usize> = state
            .lists
            .free
            .iter()
            .map(|(sz, list)| (*sz, list.len()))
            .collect();
        f.debug_struct("BufferPool")
            .field("usage", &self.usage)
            .field("buffer_counts", &counts)
            .field("pooled_bytes", &state.lists.pooled_bytes)
            .field("overflow", &state.overflow)
            .finish()
    }
}

impl BufferPool {
    /// A pool whose buffers are created with `usage`, with overflow on.
    pub fn new(usage: BufferUsages) -> Self {
        Self {
            buffers: Mutex::new(PoolState {
                lists: FreeLists::default(),
                max_pooled_bytes: DEFAULT_MAX_POOLED_BYTES,
                overflow: true,
            }),
            usage,
            tracker: Arc::new(MemoryTracker::new()),
            bind_groups: Arc::new(BindGroupCache::default()),
            created: AtomicU64::new(0),
            reused: AtomicU64::new(0),
            bound: Mutex::new(HashMap::new()),
        }
    }

    /// Counters of the tensors drawn from (and returned to) this pool, and
    /// of every other tensor of the context that owns it.
    pub fn tracker(&self) -> &Arc<MemoryTracker> {
        &self.tracker
    }

    /// The bind group cache of the context that owns this pool.
    pub(crate) fn bind_groups(&self) -> &Arc<BindGroupCache> {
        &self.bind_groups
    }

    fn release_all(&self, state: &mut PoolState) {
        for (size, buffer) in state.lists.drain() {
            self.bind_groups.evict(&buffer);
            self.tracker.pool_take(size);
            self.tracker.gpu_dropped(size);
        }
    }

    /// Change how many bytes of freed buffers the pool keeps (0 disables
    /// reuse once the GPU has finished with them). Buffers above the new
    /// cap are released immediately.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub fn set_max_pooled_bytes(&self, max: u64) {
        let mut state = self.buffers.lock().unwrap();
        state.max_pooled_bytes = max;
        if state.lists.pooled_bytes > max {
            self.release_all(&mut state);
        }
    }

    /// The most bytes of freed buffers the pool keeps
    /// ([`DEFAULT_MAX_POOLED_BYTES`] unless changed).
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub fn max_pooled_bytes(&self) -> u64 {
        self.buffers.lock().unwrap().max_pooled_bytes
    }

    /// Keep buffers returned beyond the cap for reuse until the GPU has
    /// finished the work recorded before they were returned (`true`, the
    /// default), or drop them at once (`false`). See the
    /// [type docs](Self). Changes no results, only which buffers tensors
    /// get. The context sets it from [`POOL_OVERFLOW_ENV`] when created.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub fn set_overflow(&self, on: bool) {
        self.buffers.lock().unwrap().overflow = on;
    }

    /// Whether overflow is on (see [`set_overflow`](Self::set_overflow)).
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub fn overflow(&self) -> bool {
        self.buffers.lock().unwrap().overflow
    }

    /// Bytes currently held for reuse.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub fn pooled_bytes(&self) -> u64 {
        self.buffers.lock().unwrap().lists.pooled_bytes
    }

    /// Buffers this pool has created (requests no idle buffer could serve).
    pub fn buffers_created(&self) -> u64 {
        self.created.load(Relaxed)
    }

    /// Requests served by an idle buffer.
    pub fn buffers_reused(&self) -> u64 {
        self.reused.load(Relaxed)
    }

    /// A free buffer for a tensor of `size` bytes: an idle one of the
    /// [`size_class`] of `size`, else the smallest idle one of up to twice
    /// that, else a new one of that size.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub fn get(&self, device: &Device, size: u64) -> Arc<Buffer> {
        let exact = wgpu_buffer_size(size);
        let class = size_class(size);
        let mut state = self.buffers.lock().unwrap();
        // The exact size is the second try: a class the device cannot
        // allocate (see below) is created, and returned, at the exact size.
        let mut idle = state.lists.take(class).map(|buffer| (class, buffer));
        if idle.is_none() && class != exact {
            idle = state.lists.take(exact).map(|buffer| (exact, buffer));
        }
        // Lengths that change on every micro-batch land on more classes than
        // the cap holds idle, so each class would be trimmed before its
        // length came back. A larger idle buffer, up to `MAX_REUSE_RATIO`
        // times the class, serves the request instead of a new one.
        if idle.is_none() {
            idle = state
                .lists
                .take_up_to(class, class.saturating_mul(MAX_REUSE_RATIO));
        }
        if let Some((taken, buffer)) = idle {
            drop(state);
            self.tracker.pool_take(taken);
            self.reused.fetch_add(1, Relaxed);
            self.note_bound(&buffer, exact);
            return buffer;
        }
        drop(state);

        self.created.fetch_add(1, Relaxed);
        let created = if class <= device.limits().max_buffer_size {
            class
        } else {
            exact
        };
        let buffer = Arc::new(device.create_buffer(&BufferDescriptor {
            label: Some("Pooled Buffer"),
            size: created,
            usage: self.usage,
            mapped_at_creation: false,
        }));
        self.note_bound(&buffer, exact);
        buffer
    }

    /// Records that `buffer` holds a tensor of `size` bytes, when it is
    /// larger than that.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    fn note_bound(&self, buffer: &Buffer, size: u64) {
        if buffer.size() != size {
            self.bound.lock().unwrap().insert(buffer.clone(), size);
        }
    }

    /// The bytes of `buffer` a kernel must see when the buffer is larger than
    /// the tensor it holds; `None` for any other buffer, which is bound whole.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub(crate) fn bound_size(&self, buffer: &Buffer) -> Option<u64> {
        self.bound.lock().unwrap().get(buffer).copied()
    }

    /// Forgets the tensor size of `buffer`, which is no longer in use.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    fn forget_bound(&self, buffer: &Buffer) {
        self.bound.lock().unwrap().remove(buffer);
    }

    /// Returns `buffer` (of `size` bytes, its own size) to the pool for reuse.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub fn return_buffer(&self, size: u64, buffer: Arc<Buffer>) {
        self.forget_bound(&buffer);
        let mut state = self.buffers.lock().unwrap();
        if !state.overflow && state.lists.pooled_bytes + size > state.max_pooled_bytes {
            // Dropping the last reference releases the GPU memory once the
            // work using it has finished and the device is polled.
            self.bind_groups.evict(&buffer);
            self.tracker.gpu_dropped(size);
            return;
        }
        state
            .lists
            .put(size, self.tracker.recording_serial(), buffer);
        self.tracker.pool_put(size);
    }

    /// Drops the oldest idle buffers above the cap whose work the GPU is
    /// known to have finished; returns how many were dropped. wgpu destroys
    /// them at the next device poll, so the caller polls right after and
    /// they are not counted as awaiting GPU.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub(crate) fn trim(&self) -> usize {
        let dropped = {
            let mut state = self.buffers.lock().unwrap();
            let max = state.max_pooled_bytes;
            state.lists.trim(max, self.tracker.completed_serial())
        };
        for (size, buffer) in &dropped {
            self.bind_groups.evict(buffer);
            self.tracker.pool_take(*size);
        }
        dropped.len()
    }

    /// Drops every pooled buffer.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub fn clear(&self) {
        let mut state = self.buffers.lock().unwrap();
        self.release_all(&mut state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_matches_exact_size_newest_first() {
        let mut l = FreeLists::default();
        l.put(64, 1, "a");
        l.put(64, 2, "b");
        l.put(128, 2, "c");
        assert_eq!(l.pooled_bytes, 256);
        assert_eq!(l.take(32), None, "no size classes");
        assert_eq!(l.take(64), Some("b"));
        assert_eq!(l.take(64), Some("a"));
        assert_eq!(l.take(64), None);
        assert_eq!(l.pooled_bytes, 128);
    }

    #[test]
    fn take_up_to_gives_the_smallest_idle_size_in_range() {
        let mut l = FreeLists::default();
        l.put(300, 1, "a");
        l.put(400, 1, "b");
        l.put(400, 2, "c");
        l.put(900, 2, "d");
        assert_eq!(l.take_up_to(301, 600), Some((400, "c")), "newest of 400");
        assert_eq!(l.take_up_to(301, 600), Some((400, "b")));
        assert_eq!(l.take_up_to(301, 600), None, "900 is out of range");
        assert_eq!(l.take_up_to(300, 600), Some((300, "a")), "the bounds count");
        assert_eq!(l.take_up_to(500, 400), None, "empty range");
        assert_eq!(l.take_up_to(500, 900), Some((900, "d")));
        assert_eq!(l.pooled_bytes, 0);
        assert!(l.free.is_empty());
    }

    #[test]
    fn trim_keeps_buffers_the_gpu_may_still_use() {
        let mut l = FreeLists::default();
        l.put(100, 1, "old");
        l.put(100, 3, "new");
        l.put(50, 2, "mid");
        // Cap 0, but only submission 1 is known to be done.
        let out = l.trim(0, 1);
        assert_eq!(out, vec![(100, "old")]);
        assert_eq!(l.pooled_bytes, 150);
        // Oldest first; stops once under the cap.
        let out = l.trim(100, 3);
        assert_eq!(out, vec![(50, "mid")]);
        assert_eq!(l.pooled_bytes, 100);
        assert_eq!(l.trim(100, 3), vec![]);
        assert_eq!(l.trim(0, 3), vec![(100, "new")]);
        assert_eq!(l.pooled_bytes, 0);
        assert!(l.free.is_empty());
    }

    #[test]
    fn freed_buffer_of_a_requested_size_is_reused() {
        // A step frees and requests the same sizes: beyond the cap, overflow
        // serves the second request from the first buffer.
        let mut l = FreeLists::default();
        let mut created = 0;
        let mut get = |l: &mut FreeLists<u32>, size: u64| {
            l.take(size).unwrap_or_else(|| {
                created += 1;
                created
            })
        };
        let a = get(&mut l, 4096);
        l.put(4096, 1, a);
        let b = get(&mut l, 4096);
        assert_eq!(b, a, "reused");
        let _ = get(&mut l, 8192);
        assert_eq!(created, 2);
    }

    #[test]
    fn size_classes_round_up_by_at_most_an_eighth() {
        // Small requests keep their exact (4-byte aligned) size.
        assert_eq!(size_class(0), 4);
        assert_eq!(size_class(1), 4);
        assert_eq!(size_class(6), 8);
        assert_eq!(size_class(252), 252);
        // 8 classes per power of two from 256 bytes up.
        assert_eq!(size_class(256), 256);
        assert_eq!(size_class(257), 288);
        assert_eq!(size_class(288), 288);
        assert_eq!(size_class(300), 320);
        assert_eq!(size_class(4000), 4096);
        assert_eq!(size_class(4096), 4096);
        assert_eq!(size_class(4097), 4608);
        assert_eq!(size_class(5000), 5120);
        assert_eq!(size_class(196_608_000), 201_326_592);
        for requested in
            (1..70_000u64).chain([1 << 20, (1 << 20) + 1, 3 << 28, u64::from(u32::MAX)])
        {
            let class = size_class(requested);
            let exact = wgpu_buffer_size(requested);
            assert!(class >= exact, "{requested}: class {class} below {exact}");
            assert_eq!(class % 4, 0, "{requested}");
            assert!(
                class - exact <= exact / 8,
                "{requested}: class {class} is more than an eighth above {exact}"
            );
            assert_eq!(size_class(class), class, "{requested}: classes are stable");
        }
    }

    #[test]
    fn lengths_that_differ_every_micro_batch_share_few_classes() {
        // A [len, 64] f32 activation for every length from 64 to 255, as
        // exact padding produces them.
        let mut classes: Vec<u64> = (64..256u64).map(|len| size_class(len * 64 * 4)).collect();
        classes.dedup();
        classes.sort_unstable();
        classes.dedup();
        assert!(classes.len() <= 17, "{} classes", classes.len());
        let exact = (64..256u64).count();
        assert!(classes.len() * 10 < exact, "{} of {exact}", classes.len());
    }

    #[test]
    fn env_value_turns_overflow_off() {
        assert!(pool_overflow_from_env_value(None));
        assert!(pool_overflow_from_env_value(Some("1")));
        assert!(pool_overflow_from_env_value(Some("on")));
        for off in ["0", "false", "OFF", " no "] {
            assert!(!pool_overflow_from_env_value(Some(off)), "{off}");
        }
    }
}
