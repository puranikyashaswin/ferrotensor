use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};

/// Free lists of GPU buffers keyed by exact byte size.
pub struct BufferPool {
    buffers: Mutex<HashMap<u64, Vec<Arc<Buffer>>>>,
    usage: BufferUsages,
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let buffers = self.buffers.lock().unwrap();
        let counts: HashMap<u64, usize> =
            buffers.iter().map(|(sz, list)| (*sz, list.len())).collect();
        f.debug_struct("BufferPool")
            .field("usage", &self.usage)
            .field("buffer_counts", &counts)
            .finish()
    }
}

impl BufferPool {
    /// A pool whose buffers are created with `usage`.
    pub fn new(usage: BufferUsages) -> Self {
        Self {
            buffers: Mutex::new(HashMap::new()),
            usage,
        }
    }

    /// A free buffer of exactly `size` bytes, or a new one.
    pub fn get(&self, device: &Device, size: u64) -> Arc<Buffer> {
        let mut buffers = self.buffers.lock().unwrap();
        if let Some(list) = buffers.get_mut(&size) {
            if let Some(buffer) = list.pop() {
                return buffer;
            }
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
        let mut buffers = self.buffers.lock().unwrap();
        buffers.entry(size).or_default().push(buffer);
    }

    /// Drops every pooled buffer.
    pub fn clear(&self) {
        let mut buffers = self.buffers.lock().unwrap();
        buffers.clear();
    }
}
