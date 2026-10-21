# Architecture

This page follows one tensor op, `a.matmul(&b)`, from the Rust API to a WGSL
kernel and back, then explains how the crates fit together. File references
are to the `ferrotensor` crate unless noted.

## Two stacks in one workspace

The workspace holds two layers of code that do not depend on each other at the
source level:

| Stack | Crates | What it is |
|-------|--------|------------|
| **Tensor API** | `ferrotensor` | A PyTorch-style `Tensor` with autograd, `nn`, `optim`, `serialize` and a CPU device. Has its own small set of generic WGSL kernels (`ferrotensor/src/backend/kernels.rs`). |
| **LLM fine-tuning stack** | `ferrotensor-backend`, `ferrotensor-autograd`, `ferrotensor-graph`, `ferrotensor-loader`, `ferrotensor-trainer`, `ferrotensor-cli`, `ferrotensor-monitor`, `ferrotensor-hub` | Hand-written kernels for Llama/LFM2-style models (RoPE, RMSNorm, flash attention, AdamW, int8 dequantisation), LoRA, safetensors loading, the training CLI. |

`ferrotensor/Cargo.toml` lists the LLM crates as dependencies, but `ferrotensor/src` does
not call into them today (see [good-first-issues](good-first-issues.md)). The
rest of this page is about the `ferrotensor` tensor API; the LLM stack is described
in each crate's README.

```mermaid
flowchart TD
    cli[ferrotensor-cli] --> trainer[ferrotensor-trainer]
    cli --> monitor[ferrotensor-monitor]
    hub[ferrotensor-hub] --> graph
    trainer --> graph[ferrotensor-graph]
    trainer --> autograd[ferrotensor-autograd]
    trainer --> loader[ferrotensor-loader]
    graph --> autograd
    graph --> loader
    autograd --> backend[ferrotensor-backend]
    loader --> backend
    graph --> backend
    backend --> wgpu[wgpu: Metal / Vulkan / DX12]

    ferrotensor["ferrotensor (tensor API, autograd, nn, optim)"] --> kernels["own WGSL kernels + CPU reference"]
    kernels --> wgpu
```

## Following `a.matmul(&b)`

### 1. The public op (`tensor/ops.rs`)

`Tensor::matmul` is a thin wrapper. It calls the *raw* kernel, then attaches a
backward closure:

```rust
pub fn matmul(&self, other: &Tensor) -> Tensor {
    let out = self.matmul_raw(other);          // forward, no autograd
    let (a, b) = (self.detach(), other.detach());
    let need = (self.requires_grad, other.requires_grad);
    out.with_node("matmul", vec![self.clone(), other.clone()], move |g| {
        vec![
            need.0.then(|| sum_to_shape(&g.matmul_raw(&b.transpose()), &a.shape)),
            need.1.then(|| sum_to_shape(&a.transpose().matmul_raw(g), &b.shape)),
        ]
    })
}
```

`with_node` only records the node when gradient tracking is on and an input
requires grad (`tensor/autograd.rs`). Everything inside `no_grad(|| ...)`
skips it.

### 2. Device dispatch (`tensor/raw.rs`)

Every `*_raw` method checks where the tensor lives:

```rust
match self.gpu() {
    Some((ctx, buffer)) => { /* build uniform + dispatch WGSL kernel */ }
    None                => { /* run the reference kernel in tensor/cpu.rs */ }
}
```

A `Tensor` holds `Arc<Storage>`, where `Storage` is either
`Cpu(RwLock<Vec<f32>>)` or `Gpu(wgpu::Buffer)` (`tensor/core.rs`). Clones share
storage and the gradient slot, so a parameter held by a layer and by an
optimizer is the same variable.

On the GPU path the method:

1. allocates the output buffer (`alloc`),
2. packs the shape and strides into small uniform / storage buffers,
3. calls `ComputeEngine::run(device, queue, "matmul", &[a, b, out, params, ...], workgroups)`.

### 3. The compute engine (`backend/compute.rs`)

`ComputeEngine::new` compiles one WGSL module (`KERNELS_WGSL`, defined in
`backend/kernels.rs`) and builds one compute pipeline per entry point in
`KERNEL_NAMES`: `unary`, `binary`, `reduce`, `softmax`, `matmul`, `permute`,
`gather_rows`. `run` binds `buffers[i]` to `@binding(i)` of bind group 0,
records a compute pass, and submits it. `workgroups_1d` and
`workgroups_matmul` compute the dispatch grid.

The kernels are *generic*: `unary` takes an op code in a uniform and switches
on it, so most element-wise ops share one pipeline. That is why adding such an
op is a handful of lines (see [writing-a-kernel.md](writing-a-kernel.md)).

### 4. The WGSL kernel (`backend/kernels.rs`)

`matmul` is a 16x16 tiled kernel: each workgroup loads tiles of `A` and `B`
into workgroup memory, synchronises with `workgroupBarrier()`, and accumulates
one output element per thread. The batch index comes from the z dimension of
the grid.

### 5. Reading the result back

Nothing waits for the GPU until you ask for data. `Tensor::to_vec()` (and
`item()`) copy the buffer into a staging buffer, map it and block with
`pollster::block_on`. Chains of ops stay on the device; a `to_vec()` is the
synchronisation point. This is also why timing code must read a value back
(see `examples/gpu_matmul_attention.rs`).

## The backward pass

`loss.backward()` (`tensor/autograd.rs`) sorts the recorded graph
topologically and walks it in reverse, calling each node's closure with the
incoming gradient and summing gradients that reach the same input. The
closures are written with ordinary tensor ops, run with tracking disabled, so
the backward pass runs on the same device as the forward pass: GPU kernels for
a GPU tensor, the CPU reference for a CPU tensor.

## Why there is a CPU device

`Device::cpu()` runs the same ops through `tensor/cpu.rs`. It exists for three
reasons:

- **Correctness oracle.** The test suite checks every op's gradient against
  central finite differences on the CPU (`tensor/tests.rs`), then, with
  `FERROTENSOR_RUN_GPU_TESTS=1`, checks that the GPU forward values and gradients
  match the CPU.
- **CI without a GPU.** GitHub runners have no GPU, so the default
  `cargo test` never touches wgpu.
- **Small models.** For tiny workloads the CPU is faster (see
  [use-cases.md](use-cases.md)).

## Layers above the tensor

- `nn` (`nn/`): the `Module` trait (`forward`, `parameters`,
  `named_parameters`) and layers: `Linear`, `Embedding`, `LayerNorm`,
  `RMSNorm`, `Dropout`, `MultiHeadAttention`, activations, `Sequential`.
  Layers are compositions of tensor ops, so they are differentiable for free.
- `optim` (`optim/`): `SGD`, `AdamW`, gradient clipping, LR schedulers. The
  updates are tensor ops run under `no_grad`, so they execute on the
  parameter's device.
- `serialize`: safetensors save/load and module state dicts keyed by
  `named_parameters`.
- `data`: `Dataset`, `TensorDataset`, `DataLoader` (batching, seeded shuffle).

## The LLM stack in one paragraph

`ferrotensor-loader` reads safetensors and tokenizers into `ferrotensor-backend` buffers.
`ferrotensor-graph` builds a Llama/LFM2-style model from backend kernels (RMSNorm,
RoPE, flash attention, SwiGLU), with optional LoRA on the projections.
`ferrotensor-autograd` supplies the gradient tape and `ferrotensor-trainer` the loops
(SFT with response-only loss, pretraining, mixed-precision loss scaling,
checkpoint save/resume). `ferrotensor-cli` is the front end. This stack is
experimental; see the status table in the [README](../README.md).
