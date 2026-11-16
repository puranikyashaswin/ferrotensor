//! GPU work counters on a real device.
//!
//! Needs a GPU, so `#[ignore]`d by default:
//!
//! ```text
//! cargo test -p ferrotensor-backend --test gpu_counters -- --ignored --test-threads=1
//! ```

use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

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
fn dispatches_in_one_pass_see_each_others_writes() {
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
