# ferrotensor-backend

The GPU compute layer of the LLM stack in [Ferrotensor](../README.md): a wgpu context
(`WgpuContext`), `UnifiedTensor` buffers, and the hand-written WGSL kernels in
`src/*.wgsl` behind `ComputeEngine` (tiled matmul, RoPE, RMSNorm, SiLU,
softmax, AdamW and SGD updates, losses, int8/affine dequantisation, flash
attention v2 forward and backward), plus a buffer pool. Runs on Metal and
Vulkan.

```rust
use ferrotensor_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

pollster::block_on(async {
    let ctx = WgpuContext::new().await;
    let engine = ComputeEngine::new(&ctx);
    let a = UnifiedTensor::new(&ctx, &[1., 2., 3., 4.], &[4]);
    let b = UnifiedTensor::new(&ctx, &[5., 6., 7., 8.], &[4]);
    let out = UnifiedTensor::empty(&ctx, &[4]);
    engine.add(&ctx, &a, &b, &out);
    println!("{:?}", out.to_vec(&ctx).await); // [6.0, 8.0, 10.0, 12.0]
});
```

Runnable versions are in [`examples/`](examples): `init_wgpu`, `tensor_creation`,
`fused_op`, `ops_test` and `basic_tensor` (each has its real output in its doc
comment), for example
`cargo run -p ferrotensor-backend --example ops_test`.

The separate `ferrotensor` crate has its own small kernel set; this crate is the
backend of `ferrotensor-autograd`, `ferrotensor-graph` and `ferrotensor-trainer`.
