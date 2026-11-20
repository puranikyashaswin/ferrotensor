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
    wgpu, ComputeEngine, GpuCounterSnapshot, KernelTiming, KernelTimingMode, UnifiedTensor,
    WgpuContext,
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
    let values = pollster::block_on(d.to_vec(&ctx)).unwrap();
    let ops = ctx.gpu_counters().since(&before);

    // (1 + 2) * 2 * 0.5
    assert_eq!(values, vec![3.0; N]);
    assert_eq!(ops.dispatches, 3, "add and two scales; the copy is not one");
    assert_eq!(
        ops.compute_passes, 2,
        "add and the first scale share a pass; the copy ends it"
    );
    assert!(ops.device_polls >= 1, "the readback polls: {ops:?}");
    assert_eq!(ops.bind_groups, 3, "each dispatch binds new buffers");
    assert_eq!(ops.bind_group_hits, 0);
    // One f32 scale per gradient_scale, set with the dispatch as immediates
    // or else in a uniform buffer, whose bind group is not cached.
    let (uniform_buffers, uniform_bytes, misses) = if ctx.uses_immediates() {
        (0, 0, 3)
    } else {
        (2, 8, 1)
    };
    assert_eq!(ops.bind_group_misses, misses);
    assert_eq!(ops.uniform_buffers, uniform_buffers);
    assert_eq!(ops.uniform_bytes, uniform_bytes);
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
    let values = pollster::block_on(sum.to_vec(&ctx)).unwrap();
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
    let values = pollster::block_on(d.to_vec(&ctx)).unwrap();
    let products = pollster::block_on(xy.to_vec(&ctx)).unwrap();
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
    let values = pollster::block_on(x.to_vec(&ctx)).unwrap();
    let ops = ctx.gpu_counters().since(&before);

    assert_eq!(values, vec![1.0; N]);
    assert_eq!(ops.submits, 3);
    let kernels = ctx.kernel_times();
    let times = kernels.by_gpu_time();
    assert_eq!(times.len(), 1, "{times:?}");
    assert_eq!(times[0].0, "gradient_scale");
    assert_eq!(times[0].1.dispatches, 300);
}

/// Runs gradient_scale, rms_norm_eps, softmax, matmul and causal
/// flash_attention_v2 on fixed inputs and reads every result back.
fn param_kernels(ctx: &WgpuContext, engine: &ComputeEngine) -> Vec<(&'static str, Vec<f32>)> {
    let t = |data: &[f32], shape: &[usize]| UnifiedTensor::new(ctx, data, shape);
    let read = |x: &UnifiedTensor| pollster::block_on(x.to_vec(ctx)).unwrap();

    let scaled = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]);
    engine.gradient_scale(ctx, &scaled, 0.5);

    let x = t(&wave(8, 1), &[2, 4]);
    let weight = t(&[1.0, 0.5, 2.0, -1.0], &[4]);
    let normed = UnifiedTensor::empty(ctx, &[2, 4]);
    engine.rms_norm_eps(ctx, &x, &weight, &normed, 1e-5);

    let logits = t(&wave(10, 2), &[2, 5]);
    let soft = UnifiedTensor::empty(ctx, &[2, 5]);
    engine.softmax(ctx, &logits, &soft);

    let (a, b) = (t(&wave(12, 3), &[3, 4]), t(&wave(20, 4), &[4, 5]));
    let product = UnifiedTensor::empty(ctx, &[3, 5]);
    engine.matmul(ctx, &a, &b, &product);

    // [batch 1, seq 4, heads 2, head_dim 8]
    let shape = [1, 4, 2, 8];
    let q = t(&wave(64, 5), &shape);
    let k = t(&wave(64, 6), &shape);
    let v = t(&wave(64, 7), &shape);
    let attended = UnifiedTensor::empty(ctx, &shape);
    engine.flash_attention_v2(ctx, &q, &k, &v, &attended, true, 0.0, 0, None);

    vec![
        ("gradient_scale", read(&scaled)),
        ("rms_norm_eps", read(&normed)),
        ("softmax", read(&soft)),
        ("matmul", read(&product)),
        ("flash_attention_v2", read(&attended)),
    ]
}

/// `n` values in [-1, 1] that differ for each `seed`.
fn wave(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + seed * 13) as f32 * 0.37).sin())
        .collect()
}

/// The results of [`param_kernels`], computed on the host.
fn param_kernels_on_host() -> Vec<Vec<f32>> {
    let scaled = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];

    let weight = [1.0, 0.5, 2.0, -1.0];
    let mut normed = Vec::new();
    for row in wave(8, 1).chunks(4) {
        let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / 4.0 + 1e-5).sqrt();
        normed.extend(row.iter().zip(weight).map(|(v, w)| v * inv * w));
    }

    let mut soft = Vec::new();
    for row in wave(10, 2).chunks(5) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
        soft.extend(row.iter().map(|v| (v - max).exp() / sum));
    }

    let (a, b) = (wave(12, 3), wave(20, 4));
    let product: Vec<f32> = (0..15)
        .map(|o| (0..4).map(|p| a[(o / 5) * 4 + p] * b[p * 5 + o % 5]).sum())
        .collect();

    let (q, k, v) = (wave(64, 5), wave(64, 6), wave(64, 7));
    let at = |s: usize, h: usize| (s * 2 + h) * 8;
    let mut attended = vec![0.0; 64];
    for h in 0..2 {
        for i in 0..4 {
            // Causal: query i sees keys 0..=i.
            let scores: Vec<f32> = (0..=i)
                .map(|j| {
                    let dot: f32 = (0..8).map(|d| q[at(i, h) + d] * k[at(j, h) + d]).sum();
                    dot / 8f32.sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|s| (s - max).exp()).sum();
            for (j, s) in scores.iter().enumerate() {
                let p = (s - max).exp() / sum;
                for d in 0..8 {
                    attended[at(i, h) + d] += p * v[at(j, h) + d];
                }
            }
        }
    }
    vec![scaled, normed, soft, product, attended]
}

#[test]
#[ignore = "needs a GPU"]
fn immediates_and_uniform_params_give_the_same_results() {
    let ctx = pollster::block_on(WgpuContext::new());
    let fallback = pollster::block_on(WgpuContext::new()).with_uniform_params();
    let engine = ComputeEngine::new(&ctx);
    let uniform_engine = ComputeEngine::new(&fallback);
    assert!(!fallback.uses_immediates() && !uniform_engine.uses_immediates());
    assert_eq!(engine.uses_immediates(), ctx.uses_immediates());

    let before = fallback.gpu_counters();
    let uniform = param_kernels(&fallback, &uniform_engine);
    let made = fallback.gpu_counters().since(&before).uniform_buffers;
    assert!(made >= 5, "at least one uniform buffer per kernel: {made}");
    for ((name, got), want) in uniform.iter().zip(param_kernels_on_host()) {
        assert_eq!(got.len(), want.len(), "{name}: length");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 1e-4, "{name}[{i}]: got {g}, want {w}");
        }
    }

    if !ctx.uses_immediates() {
        eprintln!(
            "skipped the immediates half: {} has no immediates",
            ctx.adapter_info()
        );
        return;
    }
    let before = ctx.gpu_counters();
    let immediate = param_kernels(&ctx, &engine);
    assert_eq!(ctx.gpu_counters().since(&before).uniform_buffers, 0);
    for ((name, got), (_, want)) in immediate.iter().zip(&uniform) {
        assert_eq!(got, want, "{name}: immediates and uniform buffers differ");
    }
}

fn read(ctx: &WgpuContext, x: &UnifiedTensor) -> Vec<f32> {
    pollster::block_on(x.to_vec(ctx)).unwrap()
}

/// (bind groups created, cache hits, cache misses) since `before`.
fn bind_counts(ctx: &WgpuContext, before: &GpuCounterSnapshot) -> (u64, u64, u64) {
    let ops = ctx.gpu_counters().since(before);
    (ops.bind_groups, ops.bind_group_hits, ops.bind_group_misses)
}

/// A dispatch that binds the same buffers to the same kernel reuses the
/// cached bind group. When the pool hands a dropped tensor's buffer to a new
/// tensor, the group reads the new tensor's data; when the buffer is freed,
/// a new buffer gets a new group.
#[test]
#[ignore = "needs a GPU"]
fn bind_group_cache_follows_reused_and_freed_buffers() {
    let ctx = pollster::block_on(WgpuContext::new());
    let engine = ComputeEngine::new(&ctx);
    let a = UnifiedTensor::new(&ctx, &[1.0, 2.0, 3.0, 4.0], &[4]);
    let b = UnifiedTensor::new(&ctx, &[10.0, 20.0, 30.0, 40.0], &[4]);
    let x = UnifiedTensor::empty(&ctx, &[4]);
    let sum = UnifiedTensor::empty(&ctx, &[4]);

    let before = ctx.gpu_counters();
    engine.add(&ctx, &a, &b, &x);
    engine.add(&ctx, &x, &b, &sum);
    assert_eq!(read(&ctx, &sum), vec![21.0, 42.0, 63.0, 84.0]);
    assert_eq!(bind_counts(&ctx, &before), (2, 0, 2));

    // The same kernel on the same buffers: the cached group.
    let before = ctx.gpu_counters();
    engine.add(&ctx, &a, &b, &x);
    assert_eq!(read(&ctx, &x), vec![11.0, 22.0, 33.0, 44.0]);
    assert_eq!(bind_counts(&ctx, &before), (0, 1, 0));

    // The pool hands x's buffer to y, which holds other data.
    let x_buffer = x.buffer().clone();
    drop(x);
    let y = UnifiedTensor::empty(&ctx, &[4]);
    assert!(*y.buffer() == x_buffer, "the pool reused x's buffer");
    drop(x_buffer);
    engine.add(&ctx, &a, &a, &y);
    let before = ctx.gpu_counters();
    engine.add(&ctx, &y, &b, &sum);
    assert_eq!(bind_counts(&ctx, &before), (0, 1, 0), "x's group, cached");
    assert_eq!(read(&ctx, &sum), vec![12.0, 24.0, 36.0, 48.0], "y's data");

    // Freed, y's buffer cannot come back: z gets a new buffer and group.
    drop(y);
    ctx.synchronize();
    ctx.pool.clear();
    let created = ctx.pool.buffers_created();
    let z = UnifiedTensor::empty(&ctx, &[4]);
    assert_eq!(ctx.pool.buffers_created(), created + 1, "a new buffer");
    engine.add(&ctx, &b, &b, &z);
    let before = ctx.gpu_counters();
    engine.add(&ctx, &z, &b, &sum);
    assert_eq!(bind_counts(&ctx, &before), (1, 0, 1));
    assert_eq!(read(&ctx, &sum), vec![30.0, 60.0, 90.0, 120.0], "z's data");
}

/// 100 and 101 rows of 10 floats are 4000 and 4040 bytes, which are the same
/// size class, so the pool hands a buffer of one to the other.
const ROWS_A: usize = 100;
const ROWS_B: usize = 101;
const COLS: usize = 10;

/// A kernel that takes its sizes from its parameters binds a whole pooled
/// buffer, so a tensor of another length in the same size class reuses the
/// group cached for the buffer.
#[test]
#[ignore = "needs a GPU"]
fn whole_buffer_groups_serve_tensors_of_other_lengths() {
    let ctx = pollster::block_on(WgpuContext::new());
    let engine = ComputeEngine::new(&ctx);
    let weight = UnifiedTensor::new(&ctx, &[1.0; COLS], &[COLS]);
    let src_a = UnifiedTensor::new(&ctx, &[2.0; ROWS_A * COLS], &[ROWS_A * COLS]);
    let src_b = UnifiedTensor::new(&ctx, &[2.0; ROWS_B * COLS], &[ROWS_B * COLS]);

    let x = UnifiedTensor::empty(&ctx, &[ROWS_A, COLS]);
    let y = UnifiedTensor::empty(&ctx, &[ROWS_A, COLS]);
    engine.copy_tensor(&ctx, &src_a, &x, 0);
    let before = ctx.gpu_counters();
    engine.rms_norm(&ctx, &x, &weight, &y);
    assert_eq!(bind_counts(&ctx, &before), (1, 0, 1));
    let (x_buffer, y_buffer) = (x.buffer().clone(), y.buffer().clone());
    // The pool gives back the most recently returned buffer first.
    drop(y);
    drop(x);

    let x = UnifiedTensor::empty(&ctx, &[ROWS_B, COLS]);
    let y = UnifiedTensor::empty(&ctx, &[ROWS_B, COLS]);
    assert!(*x.buffer() == x_buffer && *y.buffer() == y_buffer);
    engine.copy_tensor(&ctx, &src_b, &x, 0);
    let before = ctx.gpu_counters();
    engine.rms_norm(&ctx, &x, &weight, &y);
    assert_eq!(bind_counts(&ctx, &before), (0, 1, 0), "the cached group");
    let out = read(&ctx, &y);
    // 2 / sqrt(mean(2 * 2) + 1e-5), for each of the 101 rows.
    let want = 2.0 / (4.0f32 + 1e-5).sqrt();
    for (i, v) in out.iter().take(ROWS_B * COLS).enumerate() {
        assert!((v - want).abs() < 1e-5, "element {i}: {v} against {want}");
    }
}

/// A kernel that bounds its loops by `arrayLength` is bound to its tensors'
/// own bytes, so the same buffers at another length need a new group, and
/// the same buffers at the same length reuse theirs.
#[test]
#[ignore = "needs a GPU"]
fn exact_size_groups_follow_the_tensor_length() {
    let ctx = pollster::block_on(WgpuContext::new());
    let engine = ComputeEngine::new(&ctx);
    let [a, b, out] = [(); 3].map(|()| UnifiedTensor::empty(&ctx, &[ROWS_A, COLS]));
    let before = ctx.gpu_counters();
    engine.add(&ctx, &a, &b, &out);
    assert_eq!(bind_counts(&ctx, &before), (1, 0, 1));
    let buffers = [a.buffer().clone(), b.buffer().clone(), out.buffer().clone()];
    drop(out);
    drop(b);
    drop(a);

    let [a, b, out] = [(); 3].map(|()| UnifiedTensor::empty(&ctx, &[ROWS_B, COLS]));
    assert!(*a.buffer() == buffers[0] && *b.buffer() == buffers[1] && *out.buffer() == buffers[2]);
    let before = ctx.gpu_counters();
    engine.add(&ctx, &a, &b, &out);
    assert_eq!(bind_counts(&ctx, &before), (1, 0, 1), "a new length");
    let before = ctx.gpu_counters();
    engine.add(&ctx, &a, &b, &out);
    assert_eq!(bind_counts(&ctx, &before), (0, 1, 0), "the same length");
}
