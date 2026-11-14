//! Checkpointed computations that capture their weights.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-autograd --test checkpoint_params -- --ignored --test-threads=1
//! ```

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Weak};

use ferrotensor_autograd::{checkpoint, checkpoint_with_params, Tensor};
use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

fn setup() -> (Arc<WgpuContext>, ComputeEngine) {
    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));
    let engine = ComputeEngine::new(&ctx);
    (ctx, engine)
}

fn leaf(ctx: &Arc<WgpuContext>, values: &[f32], shape: &[usize], trainable: bool) -> Tensor {
    let data = UnifiedTensor::new(ctx, values, shape);
    if trainable {
        Tensor::new(ctx.clone(), data)
    } else {
        Tensor::frozen(ctx.clone(), data)
    }
}

fn grad_of(ctx: &WgpuContext, t: &Tensor) -> Vec<f32> {
    let g = t.grad.borrow().clone().expect("tensor has a gradient");
    pollster::block_on(g.to_vec(ctx))
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 1e-6, "{what}[{i}]: {g} vs {w}");
    }
}

const X: [f32; 6] = [0.5, -1.0, 2.0, 1.5, 0.25, -0.75];
const W: [f32; 6] = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6];
const W2: [f32; 4] = [0.7, -0.3, 0.2, 0.9];

#[test]
#[ignore = "needs a GPU"]
fn checkpoint_with_params_trains_captured_weights() {
    let (ctx, engine) = setup();
    let x = leaf(&ctx, &X, &[2, 3], false);
    let w = leaf(&ctx, &W, &[3, 2], true);

    // Reference: the same computation without checkpointing.
    let y = Tensor::silu(&Tensor::matmul(&x, &w, &engine), &engine);
    y.backward(&engine);
    let g_ref = grad_of(&ctx, &w);
    assert!(
        g_ref.iter().any(|&v| v != 0.0),
        "reference gradient is zero"
    );
    w.zero_grad();

    // Plain `checkpoint` cannot see the captured weight: its output needs no
    // gradient because its only input is frozen.
    let wc = w.clone();
    let out = checkpoint(
        vec![x.clone()],
        move |i, e| vec![Tensor::silu(&Tensor::matmul(&i[0], &wc, e), e)],
        &engine,
    )
    .pop()
    .expect("one output");
    assert!(!out.requires_grad);
    drop(out);

    let forward = |w: Tensor| {
        move |i: &[Tensor], e: &ComputeEngine| vec![Tensor::silu(&Tensor::matmul(&i[0], &w, e), e)]
    };
    let out = checkpoint_with_params(
        vec![x.clone()],
        std::slice::from_ref(&w),
        forward(w.clone()),
        &engine,
    )
    .pop()
    .expect("one output");
    assert!(out.requires_grad);
    out.backward(&engine);
    assert_close(&grad_of(&ctx, &w), &g_ref, "w.grad");
    assert!(x.grad.borrow().is_none(), "the frozen input got a gradient");

    // A second micro-step without clearing adds a second copy, as without
    // checkpointing (gradient accumulation).
    let out = checkpoint_with_params(
        vec![x.clone()],
        std::slice::from_ref(&w),
        forward(w.clone()),
        &engine,
    )
    .pop()
    .expect("one output");
    out.backward(&engine);
    let doubled: Vec<f32> = g_ref.iter().map(|v| 2.0 * v).collect();
    assert_close(
        &grad_of(&ctx, &w),
        &doubled,
        "w.grad after two backward passes",
    );
}

/// Weak handles to the data and gradient slot of every tensor a checkpointed
/// computation created, so the test can see whether any of them outlives
/// backward.
type Created = Rc<RefCell<Vec<(Weak<UnifiedTensor>, Weak<RefCell<Option<UnifiedTensor>>>)>>>;

fn record(created: &Created, t: &Tensor) {
    created
        .borrow_mut()
        .push((Arc::downgrade(&t.data), Arc::downgrade(&t.grad)));
}

#[test]
#[ignore = "needs a GPU"]
fn checkpointed_replay_releases_its_intermediates() {
    let (ctx, engine) = setup();
    let x = leaf(&ctx, &X, &[2, 3], false);
    let w1 = leaf(&ctx, &W, &[3, 2], true);
    let w2 = leaf(&ctx, &W2, &[2, 2], true);

    // Reference: two "blocks" without checkpointing.
    let h = Tensor::silu(&Tensor::matmul(&x, &w1, &engine), &engine);
    let y = Tensor::silu(&Tensor::matmul(&h, &w2, &engine), &engine);
    y.backward(&engine);
    let (g1, g2) = (grad_of(&ctx, &w1), grad_of(&ctx, &w2));
    assert!(g1.iter().any(|&v| v != 0.0) && g2.iter().any(|&v| v != 0.0));
    w1.zero_grad();
    w2.zero_grad();
    drop((h, y));

    let created: Created = Rc::new(RefCell::new(Vec::new()));
    let block = |w: Tensor, created: Created| {
        move |i: &[Tensor], e: &ComputeEngine| {
            let m = Tensor::matmul(&i[0], &w, e);
            let s = Tensor::silu(&m, e);
            record(&created, &m);
            record(&created, &s);
            vec![s]
        }
    };
    let h = checkpoint_with_params(
        vec![x.clone()],
        std::slice::from_ref(&w1),
        block(w1.clone(), created.clone()),
        &engine,
    )
    .pop()
    .expect("one output");
    let y = checkpoint_with_params(
        vec![h.clone()],
        std::slice::from_ref(&w2),
        block(w2.clone(), created.clone()),
        &engine,
    )
    .pop()
    .expect("one output");
    // After the forward pass only the block outputs remain.
    assert_eq!(created.borrow().len(), 4, "two blocks, two tensors each");
    assert!(
        created
            .borrow()
            .iter()
            .all(|(d, g)| d.upgrade().is_none() && g.upgrade().is_none()),
        "a block's internals outlived its forward pass"
    );

    y.backward(&engine);

    // Each block was replayed exactly once, and nothing the replays created
    // (data or gradient slot) is still alive.
    assert_eq!(created.borrow().len(), 8, "each block replayed once");
    for (i, (data, grad)) in created.borrow().iter().enumerate() {
        assert!(data.upgrade().is_none(), "replayed tensor {i} still alive");
        assert!(
            grad.upgrade().is_none(),
            "replayed gradient slot {i} still alive"
        );
    }
    // Only leaves hold gradients: the block outputs' were released.
    assert!(y.grad.borrow().is_none(), "output kept its gradient");
    assert!(h.grad.borrow().is_none(), "block output kept its gradient");
    assert!(x.grad.borrow().is_none(), "the frozen input got a gradient");
    // The captured weights received their gradients exactly once.
    assert_close(&grad_of(&ctx, &w1), &g1, "w1.grad");
    assert_close(&grad_of(&ctx, &w2), &g2, "w2.grad");
}
