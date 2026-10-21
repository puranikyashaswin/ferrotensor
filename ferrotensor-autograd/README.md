# ferrotensor-autograd

Reverse-mode automatic differentiation over [`ferrotensor-backend`](../ferrotensor-backend)
GPU tensors, used by `ferrotensor-graph` and `ferrotensor-trainer`. A `Tensor` wraps a
`UnifiedTensor` with a gradient slot and the node that produced it; ops such as
`Tensor::matmul` run a GPU kernel and record a node, and `Tensor::backward`
walks the graph in reverse. The crate also has `AdamW`, learning-rate
schedulers and gradient `checkpoint`ing.

```rust
use std::sync::Arc;
use ferrotensor_autograd::Tensor;
use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

pollster::block_on(async {
    let ctx = Arc::new(WgpuContext::new().await);
    let engine = ComputeEngine::new(&ctx);
    let a = Tensor::new(ctx.clone(), UnifiedTensor::new(&ctx, &[1., 2., 3., 4.], &[2, 2]));
    let b = Tensor::new(ctx.clone(), UnifiedTensor::new(&ctx, &[5., 6., 7., 8.], &[2, 2]));
    let loss = Tensor::mse_loss(&Tensor::matmul(&a, &b, &engine), &b, &engine);
    loss.backward(&engine);
    let grad = a.grad.borrow().as_ref().unwrap().to_vec(&ctx).await;
    println!("{grad:?}");
});
```

> **TODO:** this snippet is written against the current API but could not be
> run: it depends on `ComputeEngine::new`, which currently fails with a WGSL
> parse error (see the [`ferrotensor-backend`](../ferrotensor-backend/README.md) note).
> Re-verify once that is fixed.

For the PyTorch-style tensor API with its own autograd on CPU and GPU, see the
[`ferrotensor`](../ferrotensor) crate.
