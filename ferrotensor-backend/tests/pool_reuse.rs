//! Buffer pool overflow and the awaiting-GPU counter on a real device.
//!
//! Needs a GPU, so `#[ignore]`d by default:
//!
//! ```text
//! cargo test -p ferrotensor-backend --test pool_reuse -- --ignored --test-threads=1
//! ```

use ferrotensor_backend::memory::wgpu_buffer_size;
use ferrotensor_backend::{wgpu, ComputeEngine, UnifiedTensor, WgpuContext};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const N: usize = 1024;

fn setup() -> (WgpuContext, ComputeEngine, UnifiedTensor) {
    let ctx = pollster::block_on(WgpuContext::new());
    let engine = ComputeEngine::new(&ctx);
    // Nothing is kept once the GPU is done with it, so any reuse below is
    // reuse beyond the cap.
    ctx.pool.set_max_pooled_bytes(0);
    let src = UnifiedTensor::new(&ctx, &[1.0; N], &[N]);
    ctx.synchronize();
    (ctx, engine, src)
}

/// Waits for all submitted work by polling, without `synchronize`, so only
/// the bookkeeping the context does on submissions and polls runs.
fn wait_for_gpu(ctx: &WgpuContext) {
    let done = Arc::new(AtomicBool::new(false));
    let flag = done.clone();
    ctx.queue
        .on_submitted_work_done(move || flag.store(true, Ordering::Release));
    while !done.load(Ordering::Acquire) {
        ctx.device.poll(wgpu::Maintain::Poll);
        std::thread::yield_now();
    }
}

#[test]
#[ignore = "needs a GPU"]
fn freed_buffer_is_reused_beyond_the_cap_with_overflow() {
    let (ctx, engine, src) = setup();
    let mem = ctx.memory();
    let bytes = wgpu_buffer_size(N as u64 * 4);

    ctx.pool.set_overflow(true);
    let created = ctx.pool.buffers_created();
    let reused = ctx.pool.buffers_reused();
    let a = UnifiedTensor::empty(&ctx, &[N]);
    engine.copy_tensor(&ctx, &src, &a, 0);
    drop(a);
    assert_eq!(ctx.pool.pooled_bytes(), bytes, "kept beyond the cap");
    assert_eq!(mem.awaiting_gpu_bytes(), 0);
    // The copy into `a` is still recorded, not run; `b` gets its buffer and
    // is written after it.
    let b = UnifiedTensor::empty(&ctx, &[N]);
    assert_eq!(
        ctx.pool.buffers_created(),
        created + 1,
        "one buffer for both"
    );
    assert_eq!(ctx.pool.buffers_reused(), reused + 1);
    assert_eq!(ctx.pool.pooled_bytes(), 0);
    engine.gradient_scale(&ctx, &src, 2.0);
    engine.copy_tensor(&ctx, &src, &b, 0);
    assert_eq!(pollster::block_on(b.to_vec(&ctx)), vec![2.0; N]);
    drop(b);
    assert_eq!(ctx.pool.pooled_bytes(), bytes);
    // Once the GPU has finished, the next wait trims the pool to its cap.
    ctx.synchronize();
    assert_eq!(ctx.pool.pooled_bytes(), 0, "trimmed to the cap");
    assert_eq!(mem.pool_idle_bytes(), 0);
    assert_eq!(mem.awaiting_gpu_bytes(), 0, "trimmed buffers are destroyed");

    // Without overflow the same sequence allocates twice.
    ctx.pool.set_overflow(false);
    let created = ctx.pool.buffers_created();
    let c = UnifiedTensor::empty(&ctx, &[N]);
    engine.copy_tensor(&ctx, &src, &c, 0);
    drop(c);
    assert_eq!(ctx.pool.pooled_bytes(), 0);
    assert_eq!(mem.awaiting_gpu_bytes(), bytes, "dropped past the cap");
    let d = UnifiedTensor::empty(&ctx, &[N]);
    assert_eq!(ctx.pool.buffers_created(), created + 2);
    drop(d);
    ctx.synchronize();
    assert_eq!(mem.awaiting_gpu_bytes(), 0);
}

#[test]
#[ignore = "needs a GPU"]
fn awaiting_gpu_counts_only_unfinished_work() {
    let (ctx, engine, src) = setup();
    let mem = ctx.memory();
    let bytes = wgpu_buffer_size(N as u64 * 4);
    ctx.pool.set_overflow(false);
    assert_eq!(mem.awaiting_gpu_bytes(), 0);

    let a = UnifiedTensor::empty(&ctx, &[N]);
    engine.copy_tensor(&ctx, &src, &a, 0);
    drop(a);
    ctx.flush();
    // Recorded in the next batch, which is not submitted.
    let b = UnifiedTensor::empty(&ctx, &[N]);
    engine.copy_tensor(&ctx, &src, &b, 0);
    drop(b);
    assert_eq!(mem.awaiting_gpu_bytes(), 2 * bytes);

    // The first batch finishes; no synchronize, no read back.
    wait_for_gpu(&ctx);
    assert_eq!(
        mem.awaiting_gpu_bytes(),
        bytes,
        "only the unsubmitted batch's buffer is still counted"
    );
    assert_eq!(mem.peak_awaiting_gpu_bytes(), 2 * bytes);
    ctx.synchronize();
    assert_eq!(mem.awaiting_gpu_bytes(), 0);
}
