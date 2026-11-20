//! RMSNorm backward that adds into the input's gradient slot: the values of a
//! CPU reference and of the separate kernel plus `add`, with a trained and a
//! frozen norm weight, an empty slot, a slot of its own and a shared one.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-autograd --test rms_norm_accumulate -- --ignored --test-threads=1
//! ```

#![allow(clippy::expect_used, clippy::arc_with_non_send_sync)]

use std::sync::Arc;

use ferrotensor_autograd::{rms_norm_backward_cpu, AutogradNode, RMSNormNode, Tensor};
use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

const ROWS: usize = 5;
const HIDDEN: usize = 37;
const EPS: f32 = 1e-5;

fn setup() -> (Arc<WgpuContext>, ComputeEngine) {
    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));
    let engine = ComputeEngine::new(&ctx);
    (ctx, engine)
}

fn host(ctx: &WgpuContext, t: &UnifiedTensor) -> Vec<f32> {
    pollster::block_on(t.to_vec(ctx)).expect("GPU readback")
}

fn grad_of(ctx: &WgpuContext, t: &Tensor) -> Vec<f32> {
    host(ctx, t.grad.borrow().as_ref().expect("a gradient"))
}

/// The address of the GPU buffer behind a gradient: clones share it.
fn grad_buffer(t: &Tensor) -> *const () {
    std::ptr::from_ref(t.grad.borrow().as_ref().expect("a gradient").buffer()).cast()
}

/// Deterministic pseudo-random values in [-1, 1).
fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[track_caller]
fn assert_close(name: &str, got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = atol + rtol * w.abs();
        assert!(
            (g - w).abs() <= tol,
            "{name}[{i}]: got {g}, want {w} (tol {tol})"
        );
    }
}

fn leaf(ctx: &Arc<WgpuContext>, data: &[f32], shape: &[usize]) -> Tensor {
    Tensor::new(ctx.clone(), UnifiedTensor::new(ctx, data, shape))
}

fn node(x: &Tensor, w: &Tensor) -> RMSNormNode {
    RMSNormNode {
        input: x.clone(),
        weight: w.clone(),
        eps: EPS,
    }
}

/// What the node's backward gives, then the slot's old contents added:
/// the two-kernel path the fused one replaces.
fn unfused(
    ctx: &WgpuContext,
    engine: &ComputeEngine,
    n: &RMSNormNode,
    g: &UnifiedTensor,
    old: &UnifiedTensor,
) -> Vec<f32> {
    let grads = n.backward(g, engine);
    let sum = UnifiedTensor::empty(ctx, &old.shape);
    engine.add(ctx, old, &grads[0], &sum);
    host(ctx, &sum)
}

#[test]
#[ignore = "needs a GPU"]
fn accumulating_backward_adds_into_its_own_slot_in_place() {
    let (ctx, engine) = setup();
    let x = rand_vec(ROWS * HIDDEN, 1);
    let w = rand_vec(HIDDEN, 2);
    let g = rand_vec(ROWS * HIDDEN, 3);
    let old = rand_vec(ROWS * HIDDEN, 4);
    let grad = UnifiedTensor::new(&ctx, &g, &[ROWS, HIDDEN]);
    let (want_x, want_w) = rms_norm_backward_cpu(&x, &g, &w, EPS, HIDDEN);
    let want: Vec<f32> = want_x.iter().zip(&old).map(|(a, b)| a + b).collect();

    for train_w in [true, false] {
        let input = leaf(&ctx, &x, &[ROWS, HIDDEN]);
        let data = UnifiedTensor::new(&ctx, &w, &[HIDDEN]);
        let weight = if train_w {
            Tensor::new(ctx.clone(), data)
        } else {
            Tensor::frozen(ctx.clone(), data)
        };
        let old_slot = UnifiedTensor::new(&ctx, &old, &[ROWS, HIDDEN]);
        let reference = unfused(&ctx, &engine, &node(&input, &weight), &grad, &old_slot);
        *input.grad.borrow_mut() = Some(old_slot);
        let before = grad_buffer(&input);

        assert!(node(&input, &weight).backward_accumulate(&grad, &engine));

        let label = format!("trained weight {train_w}");
        assert_eq!(before, grad_buffer(&input), "{label}: the slot is reused");
        let got = grad_of(&ctx, &input);
        assert_close(&format!("{label} grad_x"), &got, &want, 1e-4, 1e-3);
        assert_close(&format!("{label} vs unfused"), &got, &reference, 1e-6, 1e-6);
        if train_w {
            assert_close("grad_w", &grad_of(&ctx, &weight), &want_w, 1e-4, 1e-3);
        } else {
            assert!(weight.grad.borrow().is_none(), "a frozen weight gets none");
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn accumulating_backward_leaves_a_shared_slot_alone() {
    let (ctx, engine) = setup();
    let x = rand_vec(ROWS * HIDDEN, 5);
    let w = rand_vec(HIDDEN, 6);
    let g = rand_vec(ROWS * HIDDEN, 7);
    let old = rand_vec(ROWS * HIDDEN, 8);
    let grad = UnifiedTensor::new(&ctx, &g, &[ROWS, HIDDEN]);
    let (want_x, _) = rms_norm_backward_cpu(&x, &g, &w, EPS, HIDDEN);
    let want: Vec<f32> = want_x.iter().zip(&old).map(|(a, b)| a + b).collect();

    let input = leaf(&ctx, &x, &[ROWS, HIDDEN]);
    let weight = leaf(&ctx, &w, &[HIDDEN]);
    let shared = UnifiedTensor::new(&ctx, &old, &[ROWS, HIDDEN]);
    // Another holder of the buffer (a sibling's gradient) must not change.
    *input.grad.borrow_mut() = Some(shared.clone());

    assert!(node(&input, &weight).backward_accumulate(&grad, &engine));

    assert_close("sum", &grad_of(&ctx, &input), &want, 1e-4, 1e-3);
    assert_eq!(host(&ctx, &shared), old, "the shared buffer is untouched");
}

#[test]
#[ignore = "needs a GPU"]
fn accumulating_backward_with_an_empty_slot_stores() {
    let (ctx, engine) = setup();
    let x = rand_vec(ROWS * HIDDEN, 9);
    let w = rand_vec(HIDDEN, 10);
    let g = rand_vec(ROWS * HIDDEN, 11);
    let grad = UnifiedTensor::new(&ctx, &g, &[ROWS, HIDDEN]);
    let (want_x, want_w) = rms_norm_backward_cpu(&x, &g, &w, EPS, HIDDEN);

    let input = leaf(&ctx, &x, &[ROWS, HIDDEN]);
    let weight = leaf(&ctx, &w, &[HIDDEN]);
    assert!(node(&input, &weight).backward_accumulate(&grad, &engine));

    assert_close("grad_x", &grad_of(&ctx, &input), &want_x, 1e-4, 1e-3);
    assert_close("grad_w", &grad_of(&ctx, &weight), &want_w, 1e-4, 1e-3);
}

#[test]
#[ignore = "needs a GPU"]
fn a_frozen_input_keeps_the_default_backward() {
    let (ctx, engine) = setup();
    let x = rand_vec(ROWS * HIDDEN, 12);
    let w = rand_vec(HIDDEN, 13);
    let g = rand_vec(ROWS * HIDDEN, 14);
    let grad = UnifiedTensor::new(&ctx, &g, &[ROWS, HIDDEN]);
    let (_, want_w) = rms_norm_backward_cpu(&x, &g, &w, EPS, HIDDEN);

    let input = Tensor::frozen(ctx.clone(), UnifiedTensor::new(&ctx, &x, &[ROWS, HIDDEN]));
    let weight = leaf(&ctx, &w, &[HIDDEN]);
    let n = node(&input, &weight);
    assert!(!n.backward_accumulate(&grad, &engine));
    let grads = n.backward(&grad, &engine);
    assert_close("grad_w", &host(&ctx, &grads[1]), &want_w, 1e-4, 1e-3);
}

/// `out = xs + 2 * rms_norm(xs)` on a residual stream `xs`: the gradient the
/// skip connection delivers is in `xs`'s slot when the norm runs, through a
/// full `backward`, with two micro-batches adding into the leaves.
#[test]
#[ignore = "needs a GPU"]
fn a_residual_block_matches_the_cpu_reference() {
    let (ctx, engine) = setup();
    let x = rand_vec(ROWS * HIDDEN, 15);
    let w = rand_vec(HIDDEN, 16);
    let g = rand_vec(ROWS * HIDDEN, 17);
    let seed = UnifiedTensor::new(&ctx, &g, &[ROWS, HIDDEN]);
    let two_g: Vec<f32> = g.iter().map(|v| 2.0 * v).collect();
    let (norm_x, norm_w) = rms_norm_backward_cpu(&x, &two_g, &w, EPS, HIDDEN);

    for train_w in [true, false] {
        let input = leaf(&ctx, &x, &[ROWS, HIDDEN]);
        let data = UnifiedTensor::new(&ctx, &w, &[HIDDEN]);
        let weight = if train_w {
            Tensor::new(ctx.clone(), data)
        } else {
            Tensor::frozen(ctx.clone(), data)
        };
        for _ in 0..2 {
            let stream = Tensor::scale(&input, 1.0, &engine);
            let normed = Tensor::rms_norm(&stream, &weight, EPS, &engine);
            let mixed = Tensor::scale(&normed, 2.0, &engine);
            let out = Tensor::add(&stream, &mixed, &engine);
            *out.grad.borrow_mut() = Some(seed.clone());
            out.backward(&engine);
        }
        // Two passes double every gradient: skip path g plus the norm path.
        let want_x: Vec<f32> = g.iter().zip(&norm_x).map(|(a, b)| 2.0 * (a + b)).collect();
        let label = format!("trained weight {train_w}");
        assert_close(
            &format!("{label} grad_x"),
            &grad_of(&ctx, &input),
            &want_x,
            1e-4,
            1e-3,
        );
        if train_w {
            let want_w: Vec<f32> = norm_w.iter().map(|v| 2.0 * v).collect();
            assert_close(
                &format!("{label} grad_w"),
                &grad_of(&ctx, &weight),
                &want_w,
                1e-3,
                1e-3,
            );
        } else {
            assert!(weight.grad.borrow().is_none(), "{label}: no weight grad");
        }
    }
}

/// One row of two values with `eps = 0`, worked by hand: `x = [3, 4]`,
/// `w = [1, 1]`, `g = [1, 0]`. Then `inv = 1 / sqrt(12.5) = 0.282843`,
/// `dot = g . w . x = 3`, `grad_x = g w inv - x inv^3 dot / 2 =
/// [0.181019, -0.135764]` and `grad_w = g x inv = [0.848528, 0]`.
#[test]
#[ignore = "needs a GPU"]
fn a_worked_row_adds_onto_a_prior_gradient() {
    let (ctx, engine) = setup();
    let input = leaf(&ctx, &[3.0, 4.0], &[1, 2]);
    let weight = leaf(&ctx, &[1.0, 1.0], &[2]);
    *input.grad.borrow_mut() = Some(UnifiedTensor::new(&ctx, &[10.0, 20.0], &[1, 2]));
    let grad = UnifiedTensor::new(&ctx, &[1.0, 0.0], &[1, 2]);
    let n = RMSNormNode {
        input: input.clone(),
        weight: weight.clone(),
        eps: 0.0,
    };
    assert!(n.backward_accumulate(&grad, &engine));
    assert_close(
        "grad_x",
        &grad_of(&ctx, &input),
        &[10.181019, 19.864235],
        1e-5,
        1e-5,
    );
    assert_close(
        "grad_w",
        &grad_of(&ctx, &weight),
        &[0.848528, 0.0],
        1e-5,
        1e-5,
    );
}

/// Into an existing gradient the norm backward is one dispatch for a frozen
/// weight and two for a trained one; with the sum made by `add` it is one more.
#[test]
#[ignore = "needs a GPU"]
fn accumulating_removes_the_add_dispatch() {
    let (ctx, engine) = setup();
    let x = rand_vec(ROWS * HIDDEN, 18);
    let w = rand_vec(HIDDEN, 19);
    let g = UnifiedTensor::new(&ctx, &rand_vec(ROWS * HIDDEN, 20), &[ROWS, HIDDEN]);
    let old = rand_vec(ROWS * HIDDEN, 21);
    for train_w in [true, false] {
        let input = leaf(&ctx, &x, &[ROWS, HIDDEN]);
        let data = UnifiedTensor::new(&ctx, &w, &[HIDDEN]);
        let weight = if train_w {
            Tensor::new(ctx.clone(), data)
        } else {
            Tensor::frozen(ctx.clone(), data)
        };
        let n = node(&input, &weight);
        let old_slot = UnifiedTensor::new(&ctx, &old, &[ROWS, HIDDEN]);

        // The default path, as `Tensor::backward` runs it without the hook.
        let before = ctx.gpu_counters();
        let grads = n.backward(&g, &engine);
        let sum = UnifiedTensor::empty(&ctx, &old_slot.shape);
        engine.add(&ctx, &old_slot, &grads[0], &sum);
        let separate = ctx.gpu_counters().since(&before).dispatches;

        *input.grad.borrow_mut() = Some(old_slot);
        let before = ctx.gpu_counters();
        assert!(n.backward_accumulate(&g, &engine));
        let fused = ctx.gpu_counters().since(&before).dispatches;

        let kernels = if train_w { 2 } else { 1 };
        assert_eq!(fused, kernels, "trained weight {train_w}: fused");
        assert_eq!(separate, kernels + 1, "trained weight {train_w}: separate");
    }
}
