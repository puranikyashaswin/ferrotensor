//! Backward through the LFM2 short convolution.
//!
//! Needs a GPU, so it is `#[ignore]`d by default. Run it with:
//!
//! ```text
//! cargo test --release -p ferrotensor-graph --test lfm_conv -- --ignored
//! ```

mod common;

use common::{gpu, pseudo_random};
use ferrotensor_autograd::Tensor;
use ferrotensor_backend::UnifiedTensor;
use ferrotensor_graph::lfm::Lfm2Conv;
use ferrotensor_graph::Linear;

#[test]
#[ignore = "needs a GPU"]
fn conv_block_backward_reaches_conv_and_input_projection() {
    // The conv kernel reads the projection as `[dim, seq_len]` with
    // `seq_len` its last dimension, so a square input keeps it in range.
    const DIM: usize = 4;
    let gpu = gpu();
    let tensor = |shape: &[usize], seed| {
        let values = pseudo_random(shape.iter().product(), seed, 1.0);
        UnifiedTensor::new(&gpu.ctx, &values, shape)
    };
    let param = |shape: &[usize], seed| Tensor::new(gpu.ctx.clone(), tensor(shape, seed));
    let conv = Lfm2Conv {
        in_proj: Linear::new(param(&[DIM, DIM], 1)),
        conv_weight: param(&[DIM, 3], 2),
        conv_bias: param(&[DIM], 3),
        out_proj: Linear::new(param(&[DIM, DIM], 4)),
    };
    let x = Tensor::frozen(gpu.ctx.clone(), tensor(&[DIM, DIM], 5));

    conv.forward(&gpu.engine, &x).backward(&gpu.engine);

    for (name, t) in [
        ("in_proj weight", &conv.in_proj.weight),
        ("conv weight", &conv.conv_weight),
        ("conv bias", &conv.conv_bias),
    ] {
        let grad = t.grad.borrow();
        let grad = grad
            .as_ref()
            .unwrap_or_else(|| panic!("{name} received no gradient"));
        let values = pollster::block_on(grad.to_vec(&gpu.ctx));
        assert!(
            values.iter().any(|g| *g != 0.0),
            "{name} gradient is all zeros"
        );
    }
}
