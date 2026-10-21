# rusty-backend

The GPU compute layer of the LLM stack in [Rusty](../README.md): a wgpu context
(`WgpuContext`), `UnifiedTensor` buffers, and the hand-written WGSL kernels in
`src/*.wgsl` behind `ComputeEngine` (tiled matmul, RoPE, RMSNorm, SiLU,
softmax, AdamW and SGD updates, losses, int8/affine dequantisation, flash
attention v2 forward and backward), plus a buffer pool. Runs on Metal and
Vulkan.

```rust
use rusty_backend::{ComputeEngine, UnifiedTensor, WgpuContext};

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

> **Known issue (TODO):** at the time of writing, `ComputeEngine::new` fails
> with a WGSL parse error ("Invalid sub-access into type") on the current
> `main`, so this snippet and `examples/ops_test.rs` panic. A fix is in
> progress; re-verify this example once it lands.

The separate `rusty` crate has its own small kernel set; this crate is the
backend of `rusty-autograd`, `rusty-graph` and `rusty-trainer`.
