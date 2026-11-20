//! A cache of the bind groups kernel dispatches use, so a dispatch that binds
//! the same buffers to the same kernel as an earlier one reuses its group.
//!
//! A bind group refers to buffers, not to their contents, so a group stays
//! right for as long as its buffers exist: a pooled buffer handed to a new
//! tensor is the same buffer, and the cached group reads the new tensor's
//! data. Each key holds clones of its pipeline and buffers, so none of them
//! is freed, and its identity taken by a new object, while the entry exists.
//! The pool and the tensors evict a buffer's entries when they let it go,
//! so the cache does not keep freed memory alive.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use wgpu::{BindGroup, Buffer, ComputePipeline};

/// Groups kept before the cache is emptied, which bounds it when buffers are
/// never let go.
const MAX_GROUPS: usize = 4096;

/// What a cached group binds: a pipeline, and the buffer at each binding with
/// the bytes of it that are bound (0 for the whole buffer).
#[derive(Clone)]
pub(crate) struct BindKey {
    pipeline: Arc<ComputePipeline>,
    buffers: Vec<(u32, Buffer, u64)>,
}

impl BindKey {
    /// The key for binding `buffers` (binding index, buffer, bound bytes) to
    /// `pipeline`.
    pub(crate) fn new(pipeline: &Arc<ComputePipeline>, buffers: Vec<(u32, Buffer, u64)>) -> Self {
        Self {
            pipeline: Arc::clone(pipeline),
            buffers,
        }
    }

    fn binds(&self, buffer: &Buffer) -> bool {
        self.buffers.iter().any(|(_, b, _)| b == buffer)
    }
}

impl PartialEq for BindKey {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.pipeline, &other.pipeline) && self.buffers == other.buffers
    }
}

impl Eq for BindKey {}

impl Hash for BindKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.pipeline).hash(state);
        self.buffers.hash(state);
    }
}

/// The hash of `value`. The maps are keyed by it rather than by keys that
/// hold wgpu handles, and compare the handles within a bucket, since two
/// values may share a hash.
fn hash_of<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[derive(Default)]
struct CacheState {
    /// Groups by the hash of their key.
    groups: HashMap<u64, Vec<(BindKey, BindGroup)>>,
    /// The keys that bind each buffer, by the hash of the buffer, to evict
    /// them when it is let go.
    by_buffer: HashMap<u64, Vec<BindKey>>,
    len: usize,
}

impl CacheState {
    /// Removes `key`'s group and its listing under every buffer but `skip`.
    fn remove(&mut self, key: &BindKey, skip: &Buffer) {
        let hash = hash_of(key);
        if let Some(bucket) = self.groups.get_mut(&hash) {
            let before = bucket.len();
            bucket.retain(|(k, _)| k != key);
            self.len -= before - bucket.len();
            if bucket.is_empty() {
                self.groups.remove(&hash);
            }
        }
        for (_, buffer, _) in &key.buffers {
            if buffer == skip {
                continue;
            }
            let hash = hash_of(buffer);
            let emptied = self.by_buffer.get_mut(&hash).is_some_and(|keys| {
                keys.retain(|k| k != key);
                keys.is_empty()
            });
            if emptied {
                self.by_buffer.remove(&hash);
            }
        }
    }
}

/// Bind groups by pipeline and buffers. Shared by a context's pool and its
/// tensors, which evict the buffers they let go.
#[derive(Default)]
pub(crate) struct BindGroupCache {
    state: Mutex<CacheState>,
}

impl std::fmt::Debug for BindGroupCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BindGroupCache")
            .field("groups", &self.lock().len)
            .finish()
    }
}

impl BindGroupCache {
    /// The state. A panic while it was held cannot leave it half changed in
    /// a way that returns a wrong group, so a poisoned lock is used as is.
    fn lock(&self) -> MutexGuard<'_, CacheState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The group cached for `key`, or the one `make` creates, which is then
    /// cached; and whether it was cached.
    pub(crate) fn get_or_create(
        &self,
        key: BindKey,
        make: impl FnOnce() -> BindGroup,
    ) -> (BindGroup, bool) {
        let hash = hash_of(&key);
        let mut state = self.lock();
        let cached = state
            .groups
            .get(&hash)
            .and_then(|bucket| bucket.iter().find(|(k, _)| k == &key));
        if let Some((_, group)) = cached {
            return (group.clone(), true);
        }
        if state.len >= MAX_GROUPS {
            *state = CacheState::default();
        }
        let group = make();
        for (_, buffer, _) in &key.buffers {
            let keys = state.by_buffer.entry(hash_of(buffer)).or_default();
            if !keys.contains(&key) {
                keys.push(key.clone());
            }
        }
        state
            .groups
            .entry(hash)
            .or_default()
            .push((key, group.clone()));
        state.len += 1;
        (group, false)
    }

    /// Drops every group that binds `buffer`, so the cache no longer keeps
    /// it alive.
    pub(crate) fn evict(&self, buffer: &Buffer) {
        let hash = hash_of(buffer);
        let mut state = self.lock();
        let Some(keys) = state.by_buffer.remove(&hash) else {
            return;
        };
        // Keys of other buffers with the same hash stay listed.
        let (evicted, kept): (Vec<_>, Vec<_>) = keys.into_iter().partition(|k| k.binds(buffer));
        if !kept.is_empty() {
            state.by_buffer.insert(hash, kept);
        }
        for key in &evicted {
            state.remove(key, buffer);
        }
    }
}
