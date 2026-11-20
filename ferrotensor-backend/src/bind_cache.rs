//! A cache of the bind groups kernel dispatches use, so a dispatch that binds
//! the same buffers to the same kernel as an earlier one reuses its group.
//!
//! A bind group refers to buffers, not to their contents, so a group stays
//! right for as long as its buffers exist: a pooled buffer handed to a new
//! tensor is the same buffer, and the cached group reads the new tensor's
//! data. Each entry holds clones of its pipeline and buffers, so none of them
//! is freed, and its identity taken by a new object, while the entry exists.
//! The pool and the tensors evict a buffer's entries when they let it go,
//! so the cache does not keep freed memory alive. When the cache is full the
//! oldest entries go first.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use wgpu::{BindGroup, Buffer, ComputePipeline};

/// Groups kept before the oldest are dropped, which bounds the cache when
/// buffers are never let go. A training step at SmolLM2-135M makes about
/// 7,400 groups, so this holds a few steps' worth.
const MAX_GROUPS: usize = 32_768;

/// One buffer of a dispatch: its binding index, the buffer and the bytes of
/// it that are bound (0 for the whole buffer).
pub(crate) type Binding<'a> = (u32, &'a Buffer, u64);

/// The hash of a dispatch's pipeline and bindings.
fn key_hash(pipeline: &Arc<ComputePipeline>, bindings: &[Binding<'_>]) -> u64 {
    let mut hasher = DefaultHasher::new();
    Arc::as_ptr(pipeline).hash(&mut hasher);
    bindings.hash(&mut hasher);
    hasher.finish()
}

/// The hash of `value`. The maps are keyed by it rather than by wgpu
/// handles, and compare the handles within a bucket, since two values may
/// share a hash.
fn hash_of<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// A cached group with what it was made for.
struct Entry {
    /// Unique among the entries ever made, so a hash bucket tells apart two
    /// entries and a listing of a gone entry never names a newer one.
    seq: u64,
    pipeline: Arc<ComputePipeline>,
    buffers: Vec<(u32, Buffer, u64)>,
    group: BindGroup,
}

impl Entry {
    fn matches(&self, pipeline: &Arc<ComputePipeline>, bindings: &[Binding<'_>]) -> bool {
        Arc::ptr_eq(&self.pipeline, pipeline)
            && self.buffers.len() == bindings.len()
            && self.buffers.iter().zip(bindings).all(
                |((at, buffer, size), &(at2, buffer2, size2))| {
                    *at == at2 && buffer == buffer2 && *size == size2
                },
            )
    }

    fn binds(&self, buffer: &Buffer) -> bool {
        self.buffers.iter().any(|(_, b, _)| b == buffer)
    }
}

/// Where an entry is: the hash of its key, and its `seq`.
type EntryRef = (u64, u64);

#[derive(Default)]
struct CacheState {
    /// Entries by the hash of their key.
    groups: HashMap<u64, Vec<Entry>>,
    /// The entries that bind each buffer, by the hash of the buffer, to evict
    /// them when it is let go. A list may still name an entry that is gone;
    /// it is skipped when read and dropped when the list is full.
    by_buffer: HashMap<u64, Vec<EntryRef>>,
    /// Every entry made, oldest first, with the same allowance for gone ones.
    order: VecDeque<EntryRef>,
    next_seq: u64,
    len: usize,
}

impl CacheState {
    fn is_live(groups: &HashMap<u64, Vec<Entry>>, (hash, seq): EntryRef) -> bool {
        groups
            .get(&hash)
            .is_some_and(|bucket| bucket.iter().any(|e| e.seq == seq))
    }

    /// Drops the entry `(hash, seq)` when it is still cached.
    fn remove(&mut self, (hash, seq): EntryRef) {
        let Some(bucket) = self.groups.get_mut(&hash) else {
            return;
        };
        let Some(at) = bucket.iter().position(|e| e.seq == seq) else {
            return;
        };
        bucket.swap_remove(at);
        if bucket.is_empty() {
            self.groups.remove(&hash);
        }
        self.len -= 1;
    }

    /// Drops the oldest entries until there is room for one more, and the
    /// names of gone entries once they are most of `order`.
    fn make_room(&mut self) {
        while self.len >= MAX_GROUPS {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.remove(oldest);
        }
        if self.order.len() > 2 * self.len + MAX_GROUPS / 8 {
            let groups = &self.groups;
            self.order.retain(|&r| Self::is_live(groups, r));
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

    /// The group cached for `pipeline` and `bindings`, or the one `make`
    /// creates, which is then cached; and whether it was cached.
    pub(crate) fn get_or_create(
        &self,
        pipeline: &Arc<ComputePipeline>,
        bindings: &[Binding<'_>],
        make: impl FnOnce() -> BindGroup,
    ) -> (BindGroup, bool) {
        let hash = key_hash(pipeline, bindings);
        let mut guard = self.lock();
        let state = &mut *guard;
        let cached = state
            .groups
            .get(&hash)
            .and_then(|bucket| bucket.iter().find(|e| e.matches(pipeline, bindings)));
        if let Some(entry) = cached {
            return (entry.group.clone(), true);
        }
        state.make_room();
        let group = make();
        let seq = state.next_seq;
        state.next_seq += 1;
        let entry_ref = (hash, seq);
        for &(_, buffer, _) in bindings {
            let list = state.by_buffer.entry(hash_of(buffer)).or_default();
            if list.last() == Some(&entry_ref) {
                continue;
            }
            // A list that is about to grow first drops the entries gone from
            // it, so a buffer that lives long (a weight) keeps a list as long
            // as its live entries, not as long as its history.
            if list.len() == list.capacity() {
                let groups = &state.groups;
                list.retain(|&r| CacheState::is_live(groups, r));
            }
            list.push(entry_ref);
        }
        state.order.push_back(entry_ref);
        state.groups.entry(hash).or_default().push(Entry {
            seq,
            pipeline: Arc::clone(pipeline),
            buffers: bindings
                .iter()
                .map(|&(at, buffer, size)| (at, buffer.clone(), size))
                .collect(),
            group: group.clone(),
        });
        state.len += 1;
        (group, false)
    }

    /// Drops every group that binds `buffer`, so the cache no longer keeps
    /// it alive.
    pub(crate) fn evict(&self, buffer: &Buffer) {
        let hash = hash_of(buffer);
        let mut guard = self.lock();
        let state = &mut *guard;
        let Some(listed) = state.by_buffer.remove(&hash) else {
            return;
        };
        // Entries of other buffers with the same hash stay listed.
        let mut kept = Vec::new();
        for entry_ref @ (key, seq) in listed {
            let binds = state
                .groups
                .get(&key)
                .and_then(|bucket| bucket.iter().find(|e| e.seq == seq))
                .map(|entry| entry.binds(buffer));
            match binds {
                Some(true) => state.remove(entry_ref),
                Some(false) => kept.push(entry_ref),
                None => {}
            }
        }
        if !kept.is_empty() {
            state.by_buffer.insert(hash, kept);
        }
    }
}
