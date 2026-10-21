//! Quickstart: a tensor, a matmul, a gradient.
//!
//! Run with: cargo run --release -p ferrotensor --example quickstart

use ferrotensor::prelude::*;

fn main() {
    // A GPU (Metal/Vulkan/DX12) when one is available, else the CPU.
    let device = Device::best();
    println!("device: {} ({})", device.name(), device.backend());

    // y = (x @ w).sum(), so dy/dw = x^T @ ones.
    let x = Tensor::from_data(&[1., 2., 3., 4., 5., 6.], &[2, 3], &device);
    let w = Tensor::from_data(&[0.5, -1., 2., 0., 1., 1.], &[3, 2], &device).requires_grad_();

    let y = x.matmul(&w);
    println!("x @ w = {:?} (shape {:?})", y.to_vec(), y.shape);

    y.sum().backward();
    println!("dL/dw = {:?}", w.grad().expect("w requires grad").to_vec());
}
