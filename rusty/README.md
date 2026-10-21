# rusty

The PyTorch-style tensor API of the [Rusty](../README.md) workspace: `Tensor`
with reverse-mode autograd, `nn` layers, `optim` optimizers and LR schedules,
safetensors `serialize`, and a `DataLoader`. Tensors live on a `Device`: a GPU
through wgpu (Metal, Vulkan, DX12) with hand-written WGSL kernels, or the CPU,
which needs no GPU and is what the tests use. `f32` only.

```rust
use rusty::prelude::*;

let device = Device::best(); // GPU if present, else Device::cpu()
let x = Tensor::from_data(&[1., 2., 3., 4., 5., 6.], &[2, 3], &device);
let w = Tensor::from_data(&[0.5, -1., 2., 0., 1., 1.], &[3, 2], &device).requires_grad_();

let y = x.matmul(&w);
y.sum().backward();
println!("{:?} {:?}", y.to_vec(), w.grad().unwrap().to_vec());
```

This is [`examples/quickstart.rs`](examples/quickstart.rs):

```bash
cargo run --release -p rusty --example quickstart
```

Other examples: `mlp_two_moons`, `char_transformer`, `gpu_matmul_attention`,
`similarity_search` (outputs in [docs/use-cases.md](../docs/use-cases.md)).
To see how an op reaches the GPU, read [docs/architecture.md](../docs/architecture.md);
to add one, [docs/writing-a-kernel.md](../docs/writing-a-kernel.md).

```bash
cargo test -p rusty                                # CPU only, no GPU needed
RUSTY_RUN_GPU_TESTS=1 cargo test -p rusty --lib    # also GPU-vs-CPU parity
```
