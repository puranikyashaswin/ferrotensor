//! GPU work counters on a real device.
//!
//! Needs a GPU, so `#[ignore]`d by default:
//!
//! ```text
//! cargo test -p ferrotensor-backend --test gpu_counters -- --ignored --test-threads=1
//! ```
//!
//! The kernel timing tests skip, with the reason printed, on an adapter
//! without timestamp queries.

#![allow(clippy::unwrap_used)]

use ferrotensor_backend::{
    wgpu, ComputeEngine, KernelTiming, KernelTimingMode, UnifiedTensor, WgpuContext,
};
use std::collections::BTreeMap;
use std::time::Instant;

const N: usize = 1024;

#[test]
#[ignore = "needs a GPU"]
fn counters_match_a_known_sequence_of_ops() {
    let ctx = pollster::block_on(WgpuContext::new());
    let engine = ComputeEngine::new(&ctx);

    let start = ctx.gpu_counters();
    let a = UnifiedTensor::new(&ctx, &[1.0; N], &[N]);
    let b = UnifiedTensor::new(&ctx, &[2.0; N], &[N]);
    let c = UnifiedTensor::empty(&ctx, &[N]);
    let d = UnifiedTensor::empty(&ctx, &[N]);
    ctx.synchronize();
    let setup = ctx.gpu_counters().since(&start);
    assert_eq!(setup.upload_bytes, 2 * N as u64 * 4);
    assert_eq!(setup.synchronizes, 1);
    assert_eq!(setup.forced_synchronizes, 0);
    assert_eq!(setup.dispatches, 0);
    assert_eq!(setup.compute_passes, 0);
    assert_eq!(setup.submits, 0, "nothing was recorded");

    let before = ctx.gpu_counters();
    engine.add(&ctx, &a, &b, &c);
    engine.gradient_scale(&ctx, &c, 2.0);
    engine.copy_tensor(&ctx, &c, &d, 0);
    engine.gradient_scale(&ctx, &d, 0.5);
    let values = pollster::block_on(d.to_vec(&ctx));
    let ops = ctx.gpu_counters().since(&before);

    // (1 + 2) * 2 * 0.5
    assert_eq!(values, vec![3.0; N]);
    assert_eq!(ops.dispatches, 3, "add and two scales; the copy is not one");
    assert_eq!(
        ops.compute_passes, 2,
        "add and the first scale share a pass; the copy ends it"
    );
    assert!(ops.device_polls >= 1, "the readback polls: {ops:?}");
    assert_eq!(ops.bind_groups, 3);
    assert_eq!(ops.uniform_buffers, 2, "one f32 scale per gradient_scale");
    assert_eq!(ops.uniform_bytes, 8);
    assert_eq!(ops.submits, 1, "the readback submits all recorded work");
    assert_eq!(ops.synchronizes, 0);
    assert_eq!(ops.upload_bytes, 0);
    assert_eq!(ops.readbacks, 1);
    assert_eq!(ops.readback_bytes, N as u64 * 4);
    assert!(
        ops.gpu_wait <= ops.readback_time,
        "the only wait was inside the readback: {ops:?}"
    );
}

#[test]
#[ignore = "needs a GPU"]
fn dispatches_sharing_a_pass_see_each_others_writes() {
    let ctx = pollster::block_on(WgpuContext::new());
    let engine = ComputeEngine::new(&ctx);
    let x = UnifiedTensor::new(&ctx, &[1.0; N], &[N]);
    let y = UnifiedTensor::new(&ctx, &[3.0; N], &[N]);
    let sum = UnifiedTensor::empty(&ctx, &[N]);
    ctx.synchronize();

    let before = ctx.gpu_counters();
    for _ in 0..10 {
        engine.gradient_scale(&ctx, &x, 2.0);
    }
    // Reads x as written by the last scale, then sum is scaled in place.
    engine.add(&ctx, &x, &y, &sum);
    engine.gradient_scale(&ctx, &sum, 0.5);
    let values = pollster::block_on(sum.to_vec(&ctx));
    let ops = ctx.gpu_counters().since(&before);

    // (1 * 2^10 + 3) * 0.5
    assert_eq!(values, vec![513.5; N]);
    assert_eq!(ops.dispatches, 12);
    assert_eq!(ops.compute_passes, 2, "12 dispatches, at most 8 per pass");
    assert_eq!(ops.submits, 1);
}

/// A context that times kernels, or `None`, with the reason printed, on an
/// adapter without timestamp queries.
fn timing_context() -> Option<WgpuContext> {
    let ctx = pollster::block_on(WgpuContext::try_new_with_timing(
        wgpu::PowerPreference::HighPerformance,
        false,
        KernelTiming::On,
    ))
    .unwrap();
    match ctx.kernel_timing_mode() {
        Some(mode) => {
            eprintln!("kernel timing on {}: {}", ctx.adapter_info(), mode.name());
            Some(ctx)
        }
        None => {
            eprintln!("skipped: {} has no timestamp queries", ctx.adapter_info());
            None
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn kernel_times_count_each_dispatch_under_its_kernel() {
    let Some(ctx) = timing_context() else {
        return;
    };
    let engine = ComputeEngine::new(&ctx);
    let a = UnifiedTensor::new(&ctx, &[1.0; N], &[N]);
    let b = UnifiedTensor::new(&ctx, &[2.0; N], &[N]);
    let c = UnifiedTensor::empty(&ctx, &[N]);
    let d = UnifiedTensor::empty(&ctx, &[N]);
    let x = UnifiedTensor::new(&ctx, &[0.5; 128 * 128], &[128, 128]);
    let y = UnifiedTensor::new(&ctx, &[2.0; 128 * 128], &[128, 128]);
    let xy = UnifiedTensor::empty(&ctx, &[128, 128]);
    ctx.synchronize();

    let before = ctx.gpu_counters();
    let kernels_before = ctx.kernel_times();
    let wall = Instant::now();
    engine.add(&ctx, &a, &b, &c);
    engine.gradient_scale(&ctx, &c, 2.0);
    engine.matmul(&ctx, &x, &y, &xy);
    engine.copy_tensor(&ctx, &c, &d, 0);
    engine.gradient_scale(&ctx, &d, 0.5);
    let values = pollster::block_on(d.to_vec(&ctx));
    let products = pollster::block_on(xy.to_vec(&ctx));
    let wall = wall.elapsed();
    let ops = ctx.gpu_counters().since(&before);
    let kernels = ctx.kernel_times().since(&kernels_before);

    // Timestamps change neither the results nor what runs.
    assert_eq!(values, vec![3.0; N]);
    assert_eq!(products, vec![128.0; 128 * 128], "128 * 0.5 * 2");
    assert_eq!(ops.dispatches, 4);
    let mode = ctx.kernel_timing_mode().unwrap();
    let passes = match mode {
        KernelTimingMode::InsidePasses => 2,
        KernelTimingMode::PassBoundaries => 4,
    };
    assert_eq!(ops.compute_passes, passes, "{mode:?}");

    let dispatches: BTreeMap<&str, u64> = kernels
        .by_gpu_time()
        .into_iter()
        .map(|(name, time)| (name, time.dispatches))
        .collect();
    assert_eq!(
        dispatches,
        BTreeMap::from([
            ("add", 1),
            ("gradient_scale", 2),
            ("matmul_blocked (a@b, out<1M)", 1),
        ])
    );
    assert!(
        kernels.total() <= wall,
        "GPU time {:?} fits in the wall time {wall:?}",
        kernels.total()
    );
}

#[test]
#[ignore = "needs a GPU"]
fn kernel_times_span_several_submissions() {
    let Some(ctx) = timing_context() else {
        return;
    };
    let engine = ComputeEngine::new(&ctx);
    let x = UnifiedTensor::new(&ctx, &[1.0; N], &[N]);
    ctx.synchronize();

    let before = ctx.gpu_counters();
    // 300 dispatches are three submissions of at most 128, each with its
    // own query set.
    for _ in 0..300 {
        engine.gradient_scale(&ctx, &x, 1.0);
    }
    let values = pollster::block_on(x.to_vec(&ctx));
    let ops = ctx.gpu_counters().since(&before);

    assert_eq!(values, vec![1.0; N]);
    assert_eq!(ops.submits, 3);
    let times = ctx.kernel_times().by_gpu_time();
    assert_eq!(times.len(), 1, "{times:?}");
    assert_eq!(times[0].0, "gradient_scale");
    assert_eq!(times[0].1.dispatches, 300);
}
