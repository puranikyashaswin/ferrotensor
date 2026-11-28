//! LayerNorm's autograd op with a frozen weight or bias: the frozen one gets
//! no gradient, and every other gradient has the bits it has when all three
//! inputs are trained.
//!
//! These need a real GPU, so they are `#[ignore]`d by default. Run them with:
//!
//! ```text
//! cargo test -p ferrotensor-autograd --test layer_norm_frozen -- --ignored --test-threads=1
//! ```

#![allow(clippy::expect_used, clippy::panic, clippy::arc_with_non_send_sync)]

use std::sync::{Arc, OnceLock};

use ferrotensor_autograd::Tensor;
use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

const ROWS: usize = 5;
const HIDDEN: usize = 37;

/// One device for every run: creating several is unreliable on some drivers.
fn gpu() -> &'static (Arc<WgpuContext>, ComputeEngine) {
    static GPU: OnceLock<(Arc<WgpuContext>, ComputeEngine)> = OnceLock::new();
    GPU.get_or_init(|| {
        let ctx = Arc::new(pollster::block_on(WgpuContext::new()));
        let engine = ComputeEngine::new(&ctx);
        (ctx, engine)
    })
}

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

/// Gradients `(x, weight, bias)` of `sum(seed * layer_norm(x, w, b))`, each
/// `None` when that input was frozen or got no gradient.
fn run(freeze_weight: bool, freeze_bias: bool) -> [Option<Vec<f32>>; 3] {
    let (ctx, engine) = gpu();
    let leaf = |data: &[f32], shape: &[usize], frozen: bool| {
        let t = UnifiedTensor::new(ctx, data, shape);
        if frozen {
            Tensor::frozen(ctx.clone(), t)
        } else {
            Tensor::new(ctx.clone(), t)
        }
    };
    let x = leaf(&rand_vec(ROWS * HIDDEN, 1), &[ROWS, HIDDEN], false);
    let w = leaf(&rand_vec(HIDDEN, 2), &[HIDDEN], freeze_weight);
    let b = leaf(&rand_vec(HIDDEN, 3), &[HIDDEN], freeze_bias);
    let out = Tensor::layer_norm(&x, &w, &b, 1e-5, engine).expect("layer_norm");
    let seed = UnifiedTensor::new(ctx, &rand_vec(ROWS * HIDDEN, 4), &[ROWS, HIDDEN]);
    out.accumulate_grad(engine, &seed);
    out.backward(engine);
    let host = |t: &Tensor| {
        t.grad
            .borrow()
            .as_ref()
            .map(|g| pollster::block_on(g.to_vec(ctx)).expect("GPU readback"))
    };
    [host(&x), host(&w), host(&b)]
}

#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn a_frozen_weight_or_bias_gets_no_gradient() {
    let [x, w, b] = run(false, false);
    let (x, w, b) = (
        x.expect("input gradient"),
        w.expect("weight gradient"),
        b.expect("bias gradient"),
    );
    assert_eq!((x.len(), w.len(), b.len()), (ROWS * HIDDEN, HIDDEN, HIDDEN));
    // The bias gradient is the seed summed over rows.
    let seed = rand_vec(ROWS * HIDDEN, 4);
    for (i, got) in b.iter().enumerate() {
        let want: f32 = (0..ROWS).map(|r| seed[r * HIDDEN + i]).sum();
        assert!(
            (got - want).abs() < 1e-5,
            "bias gradient {i}: {got} vs {want}"
        );
    }

    let [fx, fw, fb] = run(true, false);
    assert_eq!(fx.as_ref(), Some(&x), "input gradient with a frozen weight");
    assert!(fw.is_none(), "a frozen weight got a gradient");
    assert_eq!(fb.as_ref(), Some(&b), "bias gradient with a frozen weight");

    let [fx, fw, fb] = run(false, true);
    assert_eq!(fx.as_ref(), Some(&x), "input gradient with a frozen bias");
    assert_eq!(fw.as_ref(), Some(&w), "weight gradient with a frozen bias");
    assert!(fb.is_none(), "a frozen bias got a gradient");

    let [fx, fw, fb] = run(true, true);
    assert_eq!(fx.as_ref(), Some(&x), "input gradient with both frozen");
    assert!(fw.is_none() && fb.is_none());
}

/// A weight or bias of the wrong length is an error, not a read past the
/// parameter into whatever the buffer pool holds.
#[test]
#[ignore = "needs a GPU; run with --ignored --test-threads=1"]
fn a_parameter_of_the_wrong_length_is_an_error() {
    let (ctx, engine) = gpu();
    let leaf = |len: usize| {
        Tensor::new(
            ctx.clone(),
            UnifiedTensor::new(ctx, &rand_vec(len, 7), &[len]),
        )
    };
    let x = Tensor::new(
        ctx.clone(),
        UnifiedTensor::new(ctx, &rand_vec(ROWS * HIDDEN, 1), &[ROWS, HIDDEN]),
    );
    let layer_norm = |w: &Tensor, b: &Tensor| Tensor::layer_norm(&x, w, b, 1e-5, engine);

    let (good_w, good_b) = (leaf(HIDDEN), leaf(HIDDEN));
    assert!(layer_norm(&good_w, &good_b).is_ok());

    let short = leaf(HIDDEN - 1);
    let long = leaf(HIDDEN + 1);
    for (name, result) in [
        ("short weight", layer_norm(&short, &good_b)),
        ("long weight", layer_norm(&long, &good_b)),
        ("short bias", layer_norm(&good_w, &short)),
        ("long bias", layer_norm(&good_w, &long)),
        (
            "weight and bias that match each other",
            layer_norm(&short, &short),
        ),
    ] {
        let err = result
            .err()
            .unwrap_or_else(|| panic!("{name} was accepted"));
        assert_eq!(err.size, HIDDEN, "{name}");
        assert!(err.to_string().contains(&HIDDEN.to_string()), "{name}");
    }
}
