//! Buffer pool overflow and the awaiting-GPU counter on a real device.
//!
//! Needs a GPU, so `#[ignore]`d by default:
//!
//! ```text
//! cargo test -p ferrotensor-backend --test pool_reuse -- --ignored --test-threads=1
//! ```
#![allow(clippy::expect_used)]

use ferrotensor_backend::memory::wgpu_buffer_size;
use ferrotensor_backend::{wgpu, ComputeEngine, UnifiedTensor, WgpuContext};
use std::io::Write as _;
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
        assert!(
            ctx.device.poll(wgpu::PollType::Poll).is_ok(),
            "the device was lost"
        );
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
    assert_eq!(
        pollster::block_on(b.to_vec(&ctx)).expect("GPU readback"),
        vec![2.0; N]
    );
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

#[test]
#[ignore = "needs a GPU"]
fn waiting_for_a_submission_trims_the_idle_buffers_it_used() {
    let (ctx, engine, src) = setup();
    let bytes = wgpu_buffer_size(N as u64 * 4);
    ctx.pool.set_overflow(true);
    let synchronizes = ctx.gpu_counters().synchronizes;

    let a = UnifiedTensor::empty(&ctx, &[N]);
    engine.copy_tensor(&ctx, &src, &a, 0);
    drop(a);
    let mark = ctx.submission_mark();
    // Submitted, but not known to be finished: kept beyond the cap.
    assert_eq!(ctx.pool.pooled_bytes(), bytes);

    ctx.wait_for_submission(mark);
    assert_eq!(ctx.pool.pooled_bytes(), 0, "trimmed to the cap");
    assert_eq!(ctx.memory().pool_idle_bytes(), 0);
    assert_eq!(ctx.gpu_counters().synchronizes, synchronizes);

    // A mark with nothing new recorded returns at once.
    let mark = ctx.submission_mark();
    ctx.wait_for_submission(mark);
    assert_eq!(ctx.pool.pooled_bytes(), 0);
}

/// Copies `n` ones into a new pooled tensor of `n` elements.
fn ones(ctx: &WgpuContext, engine: &ComputeEngine, n: usize) -> UnifiedTensor {
    let src = UnifiedTensor::new(ctx, &vec![1.0; n], &[n]);
    let out = UnifiedTensor::empty(ctx, &[n]);
    engine.copy_tensor(ctx, &src, &out, 0);
    out
}

#[test]
#[ignore = "needs a GPU"]
fn a_buffer_larger_than_its_tensor_is_bound_to_the_tensors_bytes() {
    let ctx = pollster::block_on(WgpuContext::new());
    let engine = ComputeEngine::new(&ctx);

    // 1,024 and 1,000 elements are 4,096 and 4,000 bytes, one size class.
    drop(ones(&ctx, &engine, 1024));
    let created = ctx.pool.buffers_created();
    let small = ones(&ctx, &engine, 1000);
    assert_eq!(ctx.pool.buffers_created(), created, "the class's buffer");
    assert_eq!(small.size_in_bytes(), 4000);

    // The kernel runs 1,024 threads (16 groups of 64) and stops at
    // `arrayLength`: with the whole buffer bound it would scale the 24
    // elements past the tensor too.
    engine.gradient_scale(&ctx, &small, 2.0);
    drop(small);
    let whole = UnifiedTensor::empty(&ctx, &[1024]);
    assert_eq!(ctx.pool.buffers_created(), created, "the same buffer again");
    let values = pollster::block_on(whole.to_vec(&ctx)).expect("GPU readback");
    assert_eq!(values.len(), 1024);
    assert!(values[..1000].iter().all(|&v| v == 2.0), "scaled");
    assert!(values[1000..].iter().all(|&v| v == 1.0), "past the tensor");
}

#[test]
#[ignore = "needs a GPU"]
fn buffer_creations_stay_flat_when_the_length_changes_every_step() {
    let ctx = pollster::block_on(WgpuContext::new());
    let engine = ComputeEngine::new(&ctx);

    // One step holds a few activations of `len` rows at once, as a forward
    // pass does, then frees them.
    let step = |len: usize| {
        let live = [
            ones(&ctx, &engine, len * 64),
            ones(&ctx, &engine, len * 64),
            ones(&ctx, &engine, len * 256),
            ones(&ctx, &engine, len * len),
        ];
        for tensor in &live {
            engine.gradient_scale(&ctx, tensor, 1.0);
        }
    };
    let created_over = |lengths: &[usize]| {
        let before = ctx.pool.buffers_created();
        for &len in lengths {
            step(len);
        }
        ctx.pool.buffers_created() - before
    };

    // Warm up on every second length (a class spans about 4% of a length or
    // more at these sizes), then run every length from 100 to 200 in an
    // order that changes the size on every step.
    let warm: Vec<usize> = (100..=200).step_by(2).collect();
    let created_warm = created_over(&warm);
    let all: Vec<usize> = (0..=100).map(|i| 100 + (i * 37) % 101).collect();
    assert_eq!(all.len(), 101);
    let created_all = created_over(&all);
    let created_again = created_over(&all);
    let _ = writeln!(
        std::io::stderr(),
        "buffers created: {created_warm} over {} warm-up steps, {created_all} over {} steps \
         of new lengths, {created_again} over the same steps again",
        warm.len(),
        all.len(),
    );
    assert_eq!(created_all, 0, "every new length is in a class seen before");
    assert_eq!(created_again, 0);
}
