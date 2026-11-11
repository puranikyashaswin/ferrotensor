use crate::memory::MemoryTracker;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

/// Default cap on the bytes a pool keeps for reuse (512 MiB).
pub const DEFAULT_MAX_POOLED_BYTES: u64 = 512 * 1024 * 1024;

/// Free lists of GPU buffers keyed by exact byte size.
///
/// Returned buffers are kept for reuse until the pool holds
/// `max_pooled_bytes`; beyond that they are released. Without the cap a
/// workload with many distinct tensor sizes (e.g. training on sequences of
/// varying length) would keep every buffer it ever freed.
pub struct BufferPool {
    buffers: Mutex<PoolState>,
    usage: BufferUsages,
    tracker: Arc<MemoryTracker>,
}

#[derive(Default)]
struct PoolState {
    free: HashMap<u64, Vec<Arc<Buffer>>>,
    pooled_bytes: u64,
    max_pooled_bytes: u64,
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.buffers.lock().unwrap();
        let counts: HashMap<u64, usize> = state
            .free
            .iter()
            .map(|(sz, list)| (*sz, list.len()))
            .collect();
        f.debug_struct("BufferPool")
            .field("usage", &self.usage)
            .field("buffer_counts", &counts)
            .field("pooled_bytes", &state.pooled_bytes)
            .finish()
    }
}

impl BufferPool {
    /// A pool whose buffers are created with `usage`.
    pub fn new(usage: BufferUsages) -> Self {
        Self {
            buffers: Mutex::new(PoolState {
                max_pooled_bytes: DEFAULT_MAX_POOLED_BYTES,
                ..PoolState::default()
            }),
            usage,
            tracker: Arc::new(MemoryTracker::new()),
        }
    }

    /// Counters of the tensors drawn from (and returned to) this pool, and
    /// of every other tensor of the context that owns it.
    pub fn tracker(&self) -> &Arc<MemoryTracker> {
        &self.tracker
    }

    fn release_all(&self, state: &mut PoolState) {
        for (size, list) in &state.free {
            for _ in list {
                self.tracker.pool_take(*size);
                self.tracker.gpu_dropped(*size);
            }
        }
        state.free.clear();
        state.pooled_bytes = 0;
    }

    /// Change how many bytes of freed buffers the pool keeps (0 disables
    /// reuse). Buffers above the new cap are released immediately.
    pub fn set_max_pooled_bytes(&self, max: u64) {
        let mut state = self.buffers.lock().unwrap();
        state.max_pooled_bytes = max;
        if state.pooled_bytes > max {
            self.release_all(&mut state);
        }
    }

    /// The most bytes of freed buffers the pool keeps
    /// ([`DEFAULT_MAX_POOLED_BYTES`] unless changed).
    pub fn max_pooled_bytes(&self) -> u64 {
        self.buffers.lock().unwrap().max_pooled_bytes
    }

    /// Bytes currently held for reuse.
    pub fn pooled_bytes(&self) -> u64 {
        self.buffers.lock().unwrap().pooled_bytes
    }

    /// A free buffer of exactly `size` bytes, or a new one.
    pub fn get(&self, device: &Device, size: u64) -> Arc<Buffer> {
        let mut state = self.buffers.lock().unwrap();
        if let Some(buffer) = state.free.get_mut(&size).and_then(Vec::pop) {
            state.pooled_bytes -= size;
            self.tracker.pool_take(size);
            return buffer;
        }

        // Create new if none available
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
        if state.pooled_bytes + size > state.max_pooled_bytes {
            // Dropping the last reference releases the GPU memory once the
            // work using it has finished and the device is polled.
            self.tracker.gpu_dropped(size);
            return;
        }
        state.pooled_bytes += size;
        state.free.entry(size).or_default().push(buffer);
        self.tracker.pool_put(size);
    }

    /// Drops every pooled buffer.
    pub fn clear(&self) {
        let mut state = self.buffers.lock().unwrap();
        self.release_all(&mut state);
    }
}
