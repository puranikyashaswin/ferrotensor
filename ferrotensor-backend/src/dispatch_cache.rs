//! Reuse of the uniform buffers and bind groups that kernel dispatches need.
//!
//! The dispatches of one training step repeat in the next: the same kernels
//! on the same pooled buffers with the same parameters. So each distinct
//! parameter block is uploaded once into a uniform buffer that is never
//! written again, and each bind group is created once per kernel layout and
//! set of bound buffers. Bind groups are keyed by the buffers' wgpu ids,
//! which wgpu never gives to another buffer, so a cached bind group cannot
//! bind a buffer other than the one it was made for.
//!
//! A cached bind group keeps the buffers it binds alive inside wgpu. After
//! every submission and wait the context drops the bind groups that bind a
//! buffer Ferrotensor has dropped, so wgpu can free it. A cache that reaches
//! its cap drops the half of its entries used longest ago.

use crate::WgpuContext;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex, Weak};
use wgpu::{util::DeviceExt, BindGroup, BindGroupLayout, Buffer, Id};

/// Most distinct parameter blocks kept.
const MAX_UNIFORMS: usize = 4096;

/// Most bind groups kept.
const MAX_BIND_GROUPS: usize = 16384;

/// A kernel's bind group layout and its (binding, buffer) pairs.
type BindKey = (Id<BindGroupLayout>, Vec<(u32, Id<Buffer>)>);

struct CachedBindGroup {
    bind_group: Arc<BindGroup>,
    /// The bound buffers; a dead one means the bind group can be dropped.
    buffers: Vec<Weak<Buffer>>,
}

/// A map of at most `cap` entries, each stamped with its last use.
struct RecentlyUsed<K, V> {
    entries: HashMap<K, (u64, V)>,
    clock: u64,
    cap: usize,
}

impl<K: Hash + Eq, V> RecentlyUsed<K, V> {
    fn new(cap: usize) -> Self {
        Self {
            entries: HashMap::new(),
            clock: 0,
            cap,
        }
    }

    fn get<Q: Hash + Eq + ?Sized>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        self.clock += 1;
        let entry = self.entries.get_mut(key)?;
        entry.0 = self.clock;
        Some(&entry.1)
    }

    /// Inserts `value`; when full, first drops the half of the entries used
    /// longest ago. Entries used every training step are never among them,
    /// while those of values that change every step (an optimizer step
    /// number) are.
    fn insert(&mut self, key: K, value: V) {
        if self.entries.len() >= self.cap {
            let mut stamps: Vec<u64> = self.entries.values().map(|(stamp, _)| *stamp).collect();
            let middle = stamps.len() / 2;
            let (_, &mut cutoff, _) = stamps.select_nth_unstable(middle);
            self.entries.retain(|_, (stamp, _)| *stamp >= cutoff);
        }
        self.clock += 1;
        self.entries.insert(key, (self.clock, value));
    }
}

/// The uniform buffers and bind groups of a [`WgpuContext`].
pub(crate) struct DispatchCache {
    uniforms: Mutex<RecentlyUsed<Vec<u8>, Arc<Buffer>>>,
    bind_groups: Mutex<RecentlyUsed<BindKey, CachedBindGroup>>,
}

impl Default for DispatchCache {
    fn default() -> Self {
        Self {
            uniforms: Mutex::new(RecentlyUsed::new(MAX_UNIFORMS)),
            bind_groups: Mutex::new(RecentlyUsed::new(MAX_BIND_GROUPS)),
        }
    }
}

impl std::fmt::Debug for DispatchCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DispatchCache").finish_non_exhaustive()
    }
}

impl WgpuContext {
    /// A uniform buffer holding `bytes`, created on first use and shared by
    /// every later dispatch with the same parameters.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub(crate) fn uniform_buffer(&self, bytes: &[u8]) -> Arc<Buffer> {
        let mut uniforms = self.dispatch_cache.uniforms.lock().unwrap();
        if let Some(buffer) = uniforms.get(bytes) {
            self.counters.uniform_buffers_reused.fetch_add(1, Relaxed);
            return buffer.clone();
        }
        // Counted as awaiting GPU until the next wait, as it was when every
        // dispatch dropped its own; kept by the cache, it is then not
        // counted, like pipelines and bind groups.
        self.memory().count_dispatch_buffer(bytes.len() as u64);
        self.counters.uniform_buffers.fetch_add(1, Relaxed);
        self.counters
            .uniform_bytes
            .fetch_add(bytes.len() as u64, Relaxed);
        let buffer = Arc::new(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Uniform"),
                    contents: bytes,
                    usage: wgpu::BufferUsages::UNIFORM,
                }),
        );
        uniforms.insert(bytes.to_vec(), buffer.clone());
        buffer
    }

    /// A bind group of `layout` binding each buffer of `entries` whole at
    /// its binding index, created on first use.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub(crate) fn bind_group(
        &self,
        layout: &BindGroupLayout,
        entries: &[(u32, &Arc<Buffer>)],
    ) -> Arc<BindGroup> {
        let key = (
            layout.global_id(),
            entries.iter().map(|(i, b)| (*i, b.global_id())).collect(),
        );
        let mut cache = self.dispatch_cache.bind_groups.lock().unwrap();
        if let Some(cached) = cache.get(&key) {
            self.counters.bind_groups_reused.fetch_add(1, Relaxed);
            return cached.bind_group.clone();
        }
        let plain: Vec<(u32, &Buffer)> = entries.iter().map(|&(i, b)| (i, &**b)).collect();
        let bind_group = Arc::new(self.create_bind_group(layout, &plain));
        let buffers = entries.iter().map(|(_, b)| Arc::downgrade(b)).collect();
        cache.insert(
            key,
            CachedBindGroup {
                bind_group: bind_group.clone(),
                buffers,
            },
        );
        bind_group
    }

    /// A new bind group of `layout` binding each buffer of `entries` whole
    /// at its binding index, not cached.
    pub(crate) fn create_bind_group(
        &self,
        layout: &BindGroupLayout,
        entries: &[(u32, &Buffer)],
    ) -> BindGroup {
        let wgpu_entries: Vec<wgpu::BindGroupEntry<'_>> = entries
            .iter()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        self.counters.bind_groups.fetch_add(1, Relaxed);
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &wgpu_entries,
        })
    }

    /// Drops the cached bind groups that bind a buffer which has been
    /// dropped, so wgpu can free it; returns how many were dropped.
    #[expect(
        clippy::unwrap_used,
        reason = "lock poisoning means another thread panicked while holding the lock"
    )]
    pub(crate) fn evict_dead_bind_groups(&self) -> usize {
        let mut cache = self.dispatch_cache.bind_groups.lock().unwrap();
        let before = cache.entries.len();
        cache
            .entries
            .retain(|_, (_, cached)| cached.buffers.iter().all(|b| b.strong_count() > 0));
        before - cache.entries.len()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{RecentlyUsed, WgpuContext, MAX_UNIFORMS};
    use crate::{ComputeEngine, UnifiedTensor};

    const N: usize = 256;

    fn setup() -> (WgpuContext, ComputeEngine) {
        let ctx = pollster::block_on(WgpuContext::new());
        let engine = ComputeEngine::new(&ctx);
        (ctx, engine)
    }

    /// Cached (bind groups, uniform buffers).
    fn cached(ctx: &WgpuContext) -> (usize, usize) {
        let cache = &ctx.dispatch_cache;
        (
            cache.bind_groups.lock().unwrap().entries.len(),
            cache.uniforms.lock().unwrap().entries.len(),
        )
    }

    #[test]
    fn a_full_cache_keeps_the_recently_used_half() {
        let mut cache = RecentlyUsed::new(4);
        for key in 0..4 {
            cache.insert(key, key * 10);
        }
        assert_eq!(cache.get(&0), Some(&0));
        assert_eq!(cache.get(&2), Some(&20));
        cache.insert(4, 40);
        let mut kept: Vec<i32> = cache.entries.keys().copied().collect();
        kept.sort_unstable();
        assert_eq!(kept, vec![0, 2, 4], "1 and 3 were used longest ago");
        assert_eq!(cache.get(&1), None);
    }

    fn read(ctx: &WgpuContext, t: &UnifiedTensor) -> Vec<f32> {
        pollster::block_on(t.to_vec(ctx))
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn a_dropped_buffer_is_never_bound_again() {
        let (ctx, engine) = setup();
        // Tensors from host data own their buffer: dropping one destroys it,
        // and wgpu may give its slot to the next buffer.
        let first = UnifiedTensor::new(&ctx, &[1.0; N], &[N]);
        engine.gradient_scale(&ctx, &first, 2.0);
        assert_eq!(read(&ctx, &first), vec![2.0; N]);
        drop(first);

        let before = ctx.gpu_counters();
        let second = UnifiedTensor::new(&ctx, &[3.0; N], &[N]);
        engine.gradient_scale(&ctx, &second, 2.0);
        assert_eq!(read(&ctx, &second), vec![6.0; N]);
        let work = ctx.gpu_counters().since(&before);
        assert_eq!(
            (work.bind_groups, work.bind_groups_reused),
            (1, 0),
            "a new bind group for the new buffer"
        );
        assert_eq!(
            (work.uniform_buffers, work.uniform_buffers_reused),
            (0, 1),
            "the same scale reuses its uniform buffer"
        );

        ctx.synchronize();
        assert_eq!(
            cached(&ctx),
            (1, 1),
            "the first buffer's bind group is gone"
        );
        drop(second);
        ctx.synchronize();
        assert_eq!(cached(&ctx), (0, 1));
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn a_pooled_buffer_keeps_its_bind_group_for_the_next_tensor() {
        let (ctx, engine) = setup();
        let first = UnifiedTensor::empty(&ctx, &[N]);
        first.write_data(&ctx, bytemuck::cast_slice(&[1.0f32; N]));
        engine.gradient_scale(&ctx, &first, 2.0);
        assert_eq!(read(&ctx, &first), vec![2.0; N]);
        drop(first);

        let before = ctx.gpu_counters();
        // The pool hands the same buffer to the next tensor of its size.
        let second = UnifiedTensor::empty(&ctx, &[N]);
        second.write_data(&ctx, bytemuck::cast_slice(&[5.0f32; N]));
        engine.gradient_scale(&ctx, &second, 2.0);
        assert_eq!(read(&ctx, &second), vec![10.0; N]);
        let work = ctx.gpu_counters().since(&before);
        assert_eq!((work.bind_groups, work.bind_groups_reused), (0, 1));
        ctx.synchronize();
        assert_eq!(cached(&ctx), (1, 1), "kept while the pool holds the buffer");
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn caches_stay_bounded() {
        let (ctx, engine) = setup();
        let t = UnifiedTensor::new(&ctx, &[1.0; N], &[N]);
        // Distinct scales just above 1, one more than the uniform cache
        // holds: each needs its own uniform buffer and bind group.
        let scales: Vec<f32> = (0..=MAX_UNIFORMS as u32)
            .map(|i| f32::from_bits(1.0f32.to_bits() + i))
            .collect();
        for &scale in &scales {
            engine.gradient_scale(&ctx, &t, scale);
        }
        let expected = scales.iter().fold(1.0f32, |x, s| x * s);
        assert_eq!(read(&ctx, &t), vec![expected; N]);
        assert_eq!(ctx.gpu_counters().uniform_buffers, scales.len() as u64);
        // The uniform cache dropped its older half at the cap, and the bind
        // groups of the dropped uniform buffers went with them.
        ctx.synchronize();
        let kept = MAX_UNIFORMS / 2 + 1;
        assert_eq!(cached(&ctx), (kept, kept));
    }
}
