//! Host memory while uploading many tensors, as when loading a model.
//!
//! `UnifiedTensor::new` copies its data into a host-side staging buffer that
//! wgpu frees only after a later queue submission completes. Loading a model
//! submits nothing, so without care every weight's staging copy stays alive
//! until training starts. This test uploads 768 MiB and checks that the peak
//! resident memory grows by well under twice that.
//!
//! Linux only (reads `/proc/self/status`), and only meaningful where device
//! memory is host memory too, such as Mesa lavapipe in CI: there the device
//! copy and the staging copy both show up as resident memory. Needs a GPU
//! adapter, so it is `#[ignore]`d; run with `--ignored`. It lives in its own
//! test binary so no other test shares the process's peak.

#![cfg(target_os = "linux")]

use ferrotensor_backend::{UnifiedTensor, WgpuContext};

/// Peak resident set size of this process (`VmHWM`), in bytes.
fn peak_rss() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    let line = status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .expect("VmHWM line");
    let kib: u64 = line
        .split_whitespace()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .expect("VmHWM value");
    kib * 1024
}

#[test]
#[ignore = "needs a GPU; run with --ignored"]
fn uploads_do_not_keep_every_staging_copy() {
    let ctx = pollster::block_on(WgpuContext::new());
    let chunk = vec![1.0f32; (16 << 20) / 4]; // 16 MiB
    let count = 48; // 768 MiB in all
    let total = (count * chunk.len() * 4) as u64;

    let before = peak_rss();
    let tensors: Vec<UnifiedTensor> = (0..count)
        .map(|_| UnifiedTensor::new(&ctx, &chunk, &[chunk.len()]))
        .collect();
    // Submit the pending staging copies and wait for them, so every device
    // buffer has been written.
    ctx.queue.submit(std::iter::empty());
    ctx.synchronize();
    let growth = peak_rss().saturating_sub(before);

    assert_eq!(
        pollster::block_on(tensors[count - 1].to_vec(&ctx))[0],
        1.0,
        "uploaded data did not arrive"
    );
    let mib = |b: u64| b >> 20;
    // Device copies alone are `total`; keeping every staging copy as well
    // would be about twice that.
    assert!(
        growth > total * 9 / 10,
        "peak grew by {} MiB for {} MiB of uploads: device memory is not counted here, \
         so this test cannot measure staging",
        mib(growth),
        mib(total)
    );
    assert!(
        growth < total * 8 / 5,
        "peak grew by {} MiB for {} MiB of uploads: staging copies were kept",
        mib(growth),
        mib(total)
    );
}
