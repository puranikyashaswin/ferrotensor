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

/// Default number of full GPU waits a returned buffer is kept for, even above
/// the idle cap (see [`BufferPool::set_keep_waits`]). A training step waits
/// for the GPU about twice (the loss readback and the gradient norm), so 8
/// covers the last four steps.
pub const DEFAULT_KEEP_WAITS: u64 = 8;

/// Environment variable read when a [`WgpuContext`](crate::WgpuContext) is
/// created: the number of full GPU waits an idle buffer is kept for (see
/// [`BufferPool::set_keep_waits`]); `0` gives the strict cap of earlier
/// versions. Unset, or anything that is not a whole number, leaves
/// [`DEFAULT_KEEP_WAITS`].
pub const POOL_KEEP_WAITS_ENV: &str = "FERROTENSOR_POOL_KEEP_WAITS";

/// The keep-waits count for a [`POOL_KEEP_WAITS_ENV`] value (`None` when the
/// variable is unset).
pub fn pool_keep_waits_from_env_value(value: Option<&str>) -> u64 {
    value
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_KEEP_WAITS)
}

/// Smallest request, in bytes, that is rounded up to a size class; smaller
/// ones keep their exact size, which is at most 64 distinct values.
const MIN_CLASSED_BYTES: u64 = 256;

/// log2 of the classes per power of two: 8, so a class is at most 12.5%
/// above the sizes it serves.
const CLASS_STEPS_LOG2: u32 = 3;

/// The most a reused idle buffer may be larger than the class of the request
/// it serves, as a ratio: 4 means an idle buffer of up to four times the class
/// is taken when none of the class itself is idle.
const MAX_REUSE_RATIO: u64 = 4;

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
/// that size, else by the smallest idle buffer of up to four times that size,
/// so a buffer can be up to four times as large as the tensor in it.
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
///
/// # Keeping the working set
///
/// A step that needs more idle bytes than the cap holds would destroy buffers
/// each time the GPU goes idle and create them again in the next step, and
/// destroying a buffer takes host time while the GPU has nothing queued. So
/// the cap applies only to buffers that have been idle for at least
/// `keep_waits` full GPU waits ([`DEFAULT_KEEP_WAITS`]): a buffer returned more
/// recently is kept even above the cap. A step's working set comes back every
/// step and is never trimmed; a size class that stops recurring is released
/// after `keep_waits` waits. [`set_max_pooled_bytes`](Self::set_max_pooled_bytes)
/// makes the cap strict again (`keep_waits` 0).
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
    keep_waits: u64,
    overflow: bool,
}

/// An idle buffer's tags: the submission that was being recorded when it was
/// returned, and the number of full GPU waits made by then.
#[derive(Clone, Copy)]
struct Returned {
    serial: u64,
    waits: u64,
}

/// Idle buffers by size; each list oldest first, every entry tagged with
/// when it was returned.
struct FreeLists<B> {
    free: BTreeMap<u64, VecDeque<(Returned, B)>>,
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

    fn put(&mut self, size: u64, serial: u64, waits: u64, buffer: B) {
        self.free
            .entry(size)
            .or_default()
            .push_back((Returned { serial, waits }, buffer));
        self.pooled_bytes += size;
    }

    /// Removes the oldest buffers until at most `max` bytes remain. Only a
    /// buffer returned while a submission up to `done` was being recorded, and
    /// at least `keep` full waits before the `waits` made so far, is removed;
    /// the others are kept even above `max`.
    #[expect(
        clippy::expect_used,
        reason = "the size was just read from the free map, so its list exists and is non-empty"
    )]
    fn trim(&mut self, max: u64, done: u64, waits: u64, keep: u64) -> Vec<(u64, B)> {
        let mut out = Vec::new();
        while self.pooled_bytes > max {
            let oldest = self
                .free
                .iter()
                .filter_map(|(&size, list)| list.front().map(|&(tag, _)| (tag, size)))
                .filter(|&(tag, _)| tag.serial <= done && tag.waits.saturating_add(keep) <= waits)
                .map(|(tag, size)| (tag.serial, size))
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
                keep_waits: DEFAULT_KEEP_WAITS,
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
    /// cap are released immediately. The cap becomes strict: this sets
    /// [`keep_waits`](Self::keep_waits) to 0, so no buffer is kept above it
    /// because it was returned recently; call
    /// [`set_keep_waits`](Self::set_keep_waits) after this to change that.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub fn set_max_pooled_bytes(&self, max: u64) {
        let mut state = self.buffers.lock().unwrap();
        state.max_pooled_bytes = max;
        state.keep_waits = 0;
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

    /// Keep a returned buffer even above the cap until `waits` full GPU waits
    /// have passed without a request taking it (see the
    /// [type docs](Self#keeping-the-working-set)); 0 trims strictly to the
    /// cap. Changes no results. The context sets it from
    /// [`POOL_KEEP_WAITS_ENV`] when created.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub fn set_keep_waits(&self, waits: u64) {
        self.buffers.lock().unwrap().keep_waits = waits;
    }

    /// The full GPU waits a returned buffer is kept for above the cap
    /// ([`DEFAULT_KEEP_WAITS`] unless changed).
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub fn keep_waits(&self) -> u64 {
        self.buffers.lock().unwrap().keep_waits
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
    /// [`size_class`] of `size`, else the smallest idle one of up to four times
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
        state.lists.put(
            size,
            self.tracker.recording_serial(),
            self.tracker.full_waits(),
            buffer,
        );
        self.tracker.pool_put(size);
    }

    /// Drops the oldest idle buffers above the cap whose work the GPU is
    /// known to have finished and that were returned at least `keep_waits`
    /// full waits ago; returns how many were dropped. wgpu destroys
    /// them at the next device poll, so the caller polls right after and
    /// they are not counted as awaiting GPU.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub(crate) fn trim(&self) -> usize {
        let dropped = {
            let mut state = self.buffers.lock().unwrap();
            let (max, keep) = (state.max_pooled_bytes, state.keep_waits);
            state.lists.trim(
                max,
                self.tracker.completed_serial(),
                self.tracker.full_waits(),
                keep,
            )
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
        l.put(64, 1, 0, "a");
        l.put(64, 2, 0, "b");
        l.put(128, 2, 0, "c");
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
        l.put(300, 1, 0, "a");
        l.put(400, 1, 0, "b");
        l.put(400, 2, 0, "c");
        l.put(900, 2, 0, "d");
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
        l.put(100, 1, 0, "old");
        l.put(100, 3, 0, "new");
        l.put(50, 2, 0, "mid");
        // Cap 0, but only submission 1 is known to be done.
        let out = l.trim(0, 1, 0, 0);
        assert_eq!(out, vec![(100, "old")]);
        assert_eq!(l.pooled_bytes, 150);
        // Oldest first; stops once under the cap.
        let out = l.trim(100, 3, 0, 0);
        assert_eq!(out, vec![(50, "mid")]);
        assert_eq!(l.pooled_bytes, 100);
        assert_eq!(l.trim(100, 3, 0, 0), vec![]);
        assert_eq!(l.trim(0, 3, 0, 0), vec![(100, "new")]);
        assert_eq!(l.pooled_bytes, 0);
        assert!(l.free.is_empty());
    }

    #[test]
    fn trim_keeps_buffers_returned_within_the_keep_waits() {
        let mut l = FreeLists::default();
        l.put(100, 1, 0, "a");
        l.put(100, 2, 5, "b");
        l.put(50, 3, 6, "c");
        // Six waits made, keep 4: only "a" (returned at wait 0) is old enough.
        assert_eq!(l.trim(0, 9, 6, 4), vec![(100, "a")]);
        assert_eq!(l.pooled_bytes, 150);
        // "b" is 4 waits old at wait 9, "c" only 3.
        assert_eq!(l.trim(0, 9, 9, 4), vec![(100, "b")]);
        assert_eq!(l.pooled_bytes, 50);
        assert_eq!(l.trim(0, 9, 10, 4), vec![(50, "c")]);
        assert!(l.free.is_empty());
    }

    #[test]
    fn old_buffers_are_still_trimmed_to_the_cap_only() {
        let mut l = FreeLists::default();
        l.put(100, 1, 0, "a");
        l.put(100, 2, 0, "b");
        l.put(100, 3, 0, "c");
        assert_eq!(l.trim(150, 9, 8, 8), vec![(100, "a"), (100, "b")]);
        assert_eq!(l.pooled_bytes, 100);
        // A buffer the GPU may still use is kept however old it is.
        assert_eq!(l.trim(0, 2, 99, 8), vec![]);
        assert_eq!(l.pooled_bytes, 100);
    }

    const MIB: u64 = 1 << 20;

    /// Steps that request `sizes(step)`, return every buffer at the end and
    /// wait for the GPU twice, trimming to a 512 MiB cap after each step:
    /// the buffers created in each step and the bytes idle at the end.
    fn simulate(keep: u64, steps: usize, sizes: impl Fn(usize) -> Vec<u64>) -> (Vec<u32>, u64) {
        let mut l = FreeLists::default();
        let (mut next, mut serial, mut waits) = (0u32, 0u64, 0u64);
        let mut created = Vec::new();
        for step in 0..steps {
            let before = next;
            let held: Vec<(u64, u32)> = sizes(step)
                .into_iter()
                .map(|size| {
                    let buffer = l.take(size).unwrap_or_else(|| {
                        next += 1;
                        next
                    });
                    (size, buffer)
                })
                .collect();
            created.push(next - before);
            serial += 1;
            for (size, buffer) in held {
                l.put(size, serial, waits, buffer);
            }
            waits += 2;
            l.trim(512 * MIB, serial, waits, keep);
        }
        (created, l.pooled_bytes)
    }

    /// Twenty distinct sizes from 96 to 115 MiB: 2,110 MiB, over the cap.
    fn working_set() -> Vec<u64> {
        (96..116).map(|mib| mib * MIB).collect()
    }

    #[test]
    fn a_working_set_over_the_cap_is_created_once_not_every_step() {
        // Strict cap: the four largest buffers (454 MiB) survive each step's
        // trim, so each later step creates the other 16.
        let (created, idle) = simulate(0, 6, |_| working_set());
        assert_eq!(created, vec![20, 16, 16, 16, 16, 16]);
        assert_eq!(idle, 454 * MIB);
        // Kept for 8 waits (four steps): nothing is created after step 0
        // and all 2,110 MiB stay idle.
        let (created, idle) = simulate(8, 6, |_| working_set());
        assert_eq!(created, vec![20, 0, 0, 0, 0, 0]);
        assert_eq!(idle, 2110 * MIB);
    }

    #[test]
    fn a_size_that_recurs_every_third_step_is_kept_only_if_the_wait_count_covers_it() {
        // 300 MiB is requested in steps 0, 3 and 6: six waits apart.
        let sizes = |step: usize| {
            let mut sizes = working_set();
            if step % 3 == 0 {
                sizes.push(300 * MIB);
            }
            sizes
        };
        let (created, _) = simulate(8, 8, sizes);
        assert_eq!(created, vec![21, 0, 0, 0, 0, 0, 0, 0]);
        // Kept for 4 waits it is released before it comes back.
        let (created, _) = simulate(4, 8, sizes);
        assert_eq!(created, vec![21, 0, 0, 1, 0, 0, 1, 0]);
    }

    #[test]
    fn a_size_that_stops_recurring_is_released_after_the_keep_waits() {
        // Step 0 uses a 300 MiB buffer, later steps never ask for it again.
        let sizes = |step: usize| {
            let mut sizes = working_set();
            if step == 0 {
                sizes.push(300 * MIB);
            }
            sizes
        };
        // Returned at wait 0, old enough at wait 8, which the trim at the end
        // of step 3 reaches: 2,110 MiB of the working set stay.
        let (_, idle) = simulate(8, 3, sizes);
        assert_eq!(idle, 2410 * MIB, "still kept after three steps");
        let (_, idle) = simulate(8, 4, sizes);
        assert_eq!(idle, 2110 * MIB, "released after four");
    }

    #[test]
    fn env_value_sets_the_keep_waits() {
        assert_eq!(pool_keep_waits_from_env_value(None), DEFAULT_KEEP_WAITS);
        assert_eq!(pool_keep_waits_from_env_value(Some("0")), 0);
        assert_eq!(pool_keep_waits_from_env_value(Some(" 16 ")), 16);
        for bad in ["", "-1", "many", "1.5"] {
            assert_eq!(
                pool_keep_waits_from_env_value(Some(bad)),
                DEFAULT_KEEP_WAITS,
                "{bad:?}"
            );
        }
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
        l.put(4096, 1, 0, a);
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
