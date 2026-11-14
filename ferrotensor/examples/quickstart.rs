//! Quickstart: a tensor, a matmul, a gradient.
//!
//! Run with: cargo run --release -p ferrotensor --example quickstart

use ferrotensor::prelude::*;

/// Panics unless `got` matches `want` element by element.
fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "got {got:?}, expected {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-4, "got {got:?}, expected {want:?}");
    }
}

fn main() {
    // A GPU (Metal/Vulkan/DX12) when one is available, else the CPU.
    let device = Device::best();
    println!("device: {} ({})", device.name(), device.backend());

    // y = (x @ w).sum(), so dy/dw = x^T @ ones.
    let x = Tensor::from_data(&[1., 2., 3., 4., 5., 6.], &[2, 3], &device);
    let w = Tensor::from_data(&[0.5, -1., 2., 0., 1., 1.], &[3, 2], &device).requires_grad_();

    let y = x.matmul(&w);
    let product = y.to_vec();
    println!("x @ w = {:?} (shape {:?})", product, y.shape);
    assert_close(&product, &[7.5, 2., 18., 2.]);

    y.sum().backward();
    let grad = w.grad().expect("w requires grad").to_vec();
    println!("dL/dw = {:?}", grad);
    // Row k of dL/dw is the sum of column k of x, repeated for both outputs.
    assert_close(&grad, &[5., 5., 7., 7., 9., 9.]);
}
