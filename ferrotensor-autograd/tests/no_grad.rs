//! `no_grad`: ops inside it record no graph, the state nests and is
//! restored afterwards, and backward still works once it has returned.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-autograd --test no_grad -- --ignored --test-threads=1
//! ```

#![allow(clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use ferrotensor_autograd::{no_grad, Tensor};
use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

fn setup() -> (Arc<WgpuContext>, ComputeEngine) {
    let ctx = Arc::new(pollster::block_on(WgpuContext::new()));
    let engine = ComputeEngine::new(&ctx);
    (ctx, engine)
}

fn host(ctx: &WgpuContext, t: &UnifiedTensor) -> Vec<f32> {
    pollster::block_on(t.to_vec(ctx))
}

#[test]
#[ignore = "needs a GPU"]
fn ops_inside_no_grad_record_nothing_and_release_their_inputs() {
    let (ctx, engine) = setup();
    let x = Tensor::new(ctx.clone(), UnifiedTensor::new(&ctx, &[1.0, -2.0], &[1, 2]));

    let h = Tensor::add(&x, &x, &engine);
    let h_data = Arc::downgrade(&h.data);
    let y = no_grad(|| Tensor::scale(&h, 3.0, &engine));
    drop(h);
    assert!(y.creator.is_none());
    assert!(!y.requires_grad);
    assert!(
        h_data.upgrade().is_none(),
        "the input of a no_grad op is released"
    );
    assert_eq!(host(&ctx, &y.data), [6.0, -12.0]);

    // Without no_grad the node keeps its input alive.
    let h = Tensor::add(&x, &x, &engine);
    let h_data = Arc::downgrade(&h.data);
    let y = Tensor::scale(&h, 3.0, &engine);
    drop(h);
    assert!(y.creator.is_some());
    assert!(y.requires_grad);
    assert!(h_data.upgrade().is_some());
}

#[test]
#[ignore = "needs a GPU"]
fn no_grad_nests_and_restores_recording() {
    let (ctx, engine) = setup();
    let x = Tensor::new(ctx.clone(), UnifiedTensor::new(&ctx, &[1.0, -2.0], &[1, 2]));

    let (inner, after_inner) = no_grad(|| {
        let inner = no_grad(|| Tensor::scale(&x, 2.0, &engine));
        (inner, Tensor::scale(&x, 2.0, &engine))
    });
    assert!(inner.creator.is_none());
    assert!(
        after_inner.creator.is_none(),
        "still inside the outer no_grad"
    );

    let unwound = std::panic::catch_unwind(|| no_grad(|| panic!("inside no_grad")));
    assert!(unwound.is_err());

    // Recording is back on: backward reaches x, d(sum 2x)/dx = 2.
    let y = Tensor::scale(&x, 2.0, &engine);
    assert!(y.creator.is_some());
    assert!(y.requires_grad);
    y.backward(&engine);
    let grad = x.grad.borrow().clone().expect("x has a gradient");
    assert_eq!(host(&ctx, &grad), [2.0, 2.0]);
}
