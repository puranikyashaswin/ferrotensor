use crate::memory::MemoryTracker;
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

/// Free lists of GPU buffers keyed by exact byte size.
///
/// A tensor's buffer is returned when its last reference drops, and handed
/// out again only to a tensor of exactly the same byte size: kernels bind
/// whole buffers, so a larger one would change what they see.
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
    created: AtomicU64,
    reused: AtomicU64,
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
            created: AtomicU64::new(0),
            reused: AtomicU64::new(0),
        }
    }

    /// Counters of the tensors drawn from (and returned to) this pool, and
    /// of every other tensor of the context that owns it.
    pub fn tracker(&self) -> &Arc<MemoryTracker> {
        &self.tracker
    }

    fn release_all(&self, state: &mut PoolState) {
        for (size, _) in state.lists.drain() {
            self.tracker.pool_take(size);
            self.tracker.gpu_dropped(size);
        }
    }

    /// Change how many bytes of freed buffers the pool keeps (0 disables
    /// reuse once the GPU has finished with them). Buffers above the new
    /// cap are released immediately.
    pub fn set_max_pooled_bytes(&self, max: u64) {
        let mut state = self.buffers.lock().unwrap();
        state.max_pooled_bytes = max;
        if state.lists.pooled_bytes > max {
            self.release_all(&mut state);
        }
    }

    /// The most bytes of freed buffers the pool keeps
    /// ([`DEFAULT_MAX_POOLED_BYTES`] unless changed).
    pub fn max_pooled_bytes(&self) -> u64 {
        self.buffers.lock().unwrap().max_pooled_bytes
    }

    /// Keep buffers returned beyond the cap for reuse until the GPU has
    /// finished the work recorded before they were returned (`true`, the
    /// default), or drop them at once (`false`). See the
    /// [type docs](Self). Changes no results, only which buffers tensors
    /// get. The context sets it from [`POOL_OVERFLOW_ENV`] when created.
    pub fn set_overflow(&self, on: bool) {
        self.buffers.lock().unwrap().overflow = on;
    }

    /// Whether overflow is on (see [`set_overflow`](Self::set_overflow)).
    pub fn overflow(&self) -> bool {
        self.buffers.lock().unwrap().overflow
    }

    /// Bytes currently held for reuse.
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

    /// A free buffer of exactly `size` bytes, or a new one.
    pub fn get(&self, device: &Device, size: u64) -> Arc<Buffer> {
        let mut state = self.buffers.lock().unwrap();
        if let Some(buffer) = state.lists.take(size) {
            self.tracker.pool_take(size);
            self.reused.fetch_add(1, Relaxed);
            return buffer;
        }
        drop(state);

        self.created.fetch_add(1, Relaxed);
        Arc::new(device.create_buffer(&BufferDescriptor {
            label: Some("Pooled Buffer"),
            size,
            usage: self.usage,
            mapped_at_creation: false,
        }))
    }

    /// Returns `buffer` (of `size` bytes) to the pool for reuse.
    pub fn return_buffer(&self, size: u64, buffer: Arc<Buffer>) {
        let mut state = self.buffers.lock().unwrap();
        if !state.overflow && state.lists.pooled_bytes + size > state.max_pooled_bytes {
            // Dropping the last reference releases the GPU memory once the
            // work using it has finished and the device is polled.
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
    pub(crate) fn trim(&self) -> usize {
        let dropped = {
            let mut state = self.buffers.lock().unwrap();
            let max = state.max_pooled_bytes;
            state.lists.trim(max, self.tracker.completed_serial())
        };
        for (size, _) in &dropped {
            self.tracker.pool_take(*size);
        }
        dropped.len()
    }

    /// Drops every pooled buffer.
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
    fn env_value_turns_overflow_off() {
        assert!(pool_overflow_from_env_value(None));
        assert!(pool_overflow_from_env_value(Some("1")));
        assert!(pool_overflow_from_env_value(Some("on")));
        for off in ["0", "false", "OFF", " no "] {
            assert!(!pool_overflow_from_env_value(Some(off)), "{off}");
        }
    }
}
