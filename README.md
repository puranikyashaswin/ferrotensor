<h1 align="center">Rusty</h1>

<p align="center"><strong>A pure-Rust GPU ML framework: no CUDA, no Python, no C++ toolchain.</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/rust-1.75+-orange" alt="Rust 1.75+">
  <img src="https://img.shields.io/badge/GPU-Metal%20%7C%20Vulkan%20%7C%20DX12-purple" alt="GPU backends">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license">
  <img src="https://img.shields.io/badge/status-pre--1.0-yellow" alt="pre-1.0">
</p>

Rusty runs tensors, autograd and neural networks on whatever GPU you have
(Apple Silicon through Metal, or Vulkan/DX12 on Windows and Linux) using
[wgpu](https://wgpu.rs) and hand-written WGSL compute kernels. There is no
CUDA, cuDNN or libtorch to install and no Python at runtime: `cargo build` is
the whole toolchain. The same tensor API also runs on a CPU device, which is
what the test suite and CI use, so every op is checked against finite
differences on the CPU and against the CPU on the GPU. It is built for the
people CUDA-first frameworks leave out (laptops, consumer and integrated
GPUs) and for anyone who wants to read, and change, every layer from the API
down to the shader. It is young: pre-1.0, f32 only, and small-scale by design.

- [Quickstart](#quickstart) · [Features](#feature-matrix) · [Architecture](#architecture) · [Compared with candle, burn and tch](#compared-with-candle-burn-and-tch) · [Roadmap](#roadmap)
- Docs: [architecture](docs/architecture.md) · [writing a kernel](docs/writing-a-kernel.md) · [use cases](docs/use-cases.md) · [LoRA SFT walkthrough](docs/examples/lora-sft.md) *(link placeholder: written by another branch)* · [benchmarks](benchmarks/RESULTS.md)

## Quickstart

You need Rust 1.75+ (`rustup` is enough). On macOS no other setup is needed.

```bash
git clone https://github.com/puranikyashaswin/rusty.git
cd rusty
cargo run --release -p rusty --example quickstart
```

Output on an Apple M2 (the device line changes with your hardware):

```
device: Apple M2 (Metal)
x @ w = [7.5, 2.0, 18.0, 2.0] (shape [2, 2])
dL/dw = [5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
```

That is [`rusty/examples/quickstart.rs`](rusty/examples/quickstart.rs): a
matmul and its gradient on the best available device. No GPU? Use
`Device::cpu()`; the same code runs.

In your own code:

```rust
use rusty::prelude::*;

let device = Device::cpu(); // or Device::best() for a GPU when present
let model = nn::Sequential::new(&device)
    .add(nn::Linear::new(2, 16, &device))
    .add(nn::Tanh)
    .add(nn::Linear::new(16, 2, &device));
let mut opt = optim::AdamW::new(model.parameters(), 0.05).weight_decay(0.0);

// XOR
let x = Tensor::from_data(&[0., 0., 0., 1., 1., 0., 1., 1.], &[4, 2], &device);
let y = Tensor::from_data(&[0., 1., 1., 0.], &[4], &device);
for _ in 0..200 {
    opt.zero_grad();
    let loss = F::cross_entropy(&model.forward(&x), &y);
    loss.backward();
    opt.step();
}
```

More runnable programs, with their real output, are in
[docs/use-cases.md](docs/use-cases.md): an MLP with a `DataLoader`, a
character-level transformer trained from scratch, a CPU-vs-GPU matmul and
attention comparison, and similarity search.

## Feature matrix

"Stable" means implemented, covered by tests, and used by the examples; the
API can still change before 1.0. "Experimental" means it runs but is less
tested, incomplete, or only checked on one machine.

| Area | Status | Notes |
|------|--------|-------|
| `Tensor` on GPU (wgpu) and CPU | **Stable** | `f32`; broadcasting element-wise ops, reductions, batched matmul, softmax, permute, slice/cat, gathers |
| Autograd | **Stable** | Reverse mode; every op's gradient is checked against finite differences, and GPU gradients against the CPU |
| `nn` layers | **Stable** | `Linear`, `Embedding`, `LayerNorm`, `RMSNorm`, `Dropout`, causal `MultiHeadAttention`, activations, `Sequential` |
| Optimizers and LR schedules | **Stable** | `SGD` (momentum, Nesterov), `AdamW`, gradient clipping, constant/step/cosine/warmup schedules |
| Safetensors save/load, state dicts | **Stable** | `rusty::serialize` |
| `DataLoader` | **Stable** | Batching and seeded shuffling; single-threaded |
| LoRA fine-tuning of Llama-style models (`rusty-cli`) | **Experimental** | Small models (TinyLlama class) on ~8 GB machines; see the [LoRA walkthrough](docs/examples/lora-sft.md) |
| Llama / LFM2-style blocks (`rusty-graph`) | **Experimental** | Attention with RoPE, SwiGLU MLP, RMSNorm, KV cache |
| Flash attention kernels (`rusty-backend`) | **Experimental** | Forward v2 and a backward kernel; see `tests/flash_grad_check.rs` |
| FP16 loss scaling, checkpoint/resume (`rusty-trainer`) | **Experimental** | |
| Terminal monitor, Hub download helpers | **Experimental** | `rusty-monitor`, `rusty-hub` |
| `f16` / `bf16` tensors in the `rusty` API | Planned | |
| Convolutions, sort/top-k, cumsum in the `rusty` API | Planned | |
| Faster small-op GPU path (batched dispatch) | Planned | Per-op submission makes tiny GPU workloads slower than the CPU |
| Multi-GPU / distributed training | Planned | `rusty-cli` has a file-based gradient sync flag (`--world-size`) that is a prototype |
| A CUDA backend | Not planned | wgpu is the point of the project |

## Architecture

```mermaid
flowchart TD
    cli[rusty-cli] --> trainer[rusty-trainer]
    cli --> monitor[rusty-monitor]
    hub[rusty-hub] --> graph
    trainer --> graph[rusty-graph]
    trainer --> autograd[rusty-autograd]
    trainer --> loader[rusty-loader]
    graph --> autograd
    graph --> loader
    autograd --> backend[rusty-backend]
    loader --> backend
    graph --> backend
    backend --> wgpu[wgpu: Metal / Vulkan / DX12]

    rusty["rusty (Tensor, autograd, nn, optim)"] --> kernels["own WGSL kernels + CPU reference"]
    kernels --> wgpu
```

The workspace has two layers. The **`rusty`** crate is the PyTorch-style tensor
API (it carries its own small set of generic kernels and a CPU device). The
**LLM stack** (`rusty-backend` through `rusty-cli`) has purpose-built kernels
for Llama-style models and the LoRA fine-tuning CLI. Details, including how a
tensor op reaches a WGSL kernel, are in [docs/architecture.md](docs/architecture.md).

| Crate | What it is |
|-------|------------|
| [`rusty`](rusty) | Tensor, autograd, `nn`, `optim`, `serialize`, `data` |
| [`rusty-backend`](rusty-backend) | wgpu context, WGSL kernels (matmul, RoPE, RMSNorm, flash attention, AdamW, ...), buffer pool |
| [`rusty-autograd`](rusty-autograd) | Gradient tape for the backend tensors |
| [`rusty-graph`](rusty-graph) | Transformer building blocks and the Llama/LFM2-style model builder |
| [`rusty-loader`](rusty-loader) | Safetensors (single and sharded) and tokenizer loading |
| [`rusty-trainer`](rusty-trainer) | Training loops, LoRA, loss scaling, checkpoints |
| [`rusty-cli`](rusty-cli) | `train`, `eval` and `infer` commands |
| [`rusty-monitor`](rusty-monitor) | Terminal dashboard |
| [`rusty-hub`](rusty-hub) | Hugging Face Hub download helpers |
| [`benchmarks`](benchmarks) | GPU kernel benchmarks |

## Fine-tuning a model

`rusty-cli` fine-tunes small Llama-style models with LoRA (response-only loss
and a chat template for SFT) and can run short pretraining on `.txt` or packed
`.bin` token files:

```bash
cargo run -p rusty-cli --release -- train \
  --lite --task sft \
  --model ./models/tinyllama \
  --data ./data/train.jsonl \
  --out ./outputs/run1
```

`--help` lists every flag. A step-by-step walkthrough with real output is in
[docs/examples/lora-sft.md](docs/examples/lora-sft.md) *(placeholder: written by another branch)*.
This path is experimental.

## Compared with candle, burn and tch

These are good projects; Rusty is not trying to replace them. The honest
picture, as of this writing:

| | Rusty | [candle](https://github.com/huggingface/candle) | [burn](https://github.com/tracel-ai/burn) | [tch-rs](https://github.com/LaurentMazare/tch-rs) |
|---|---|---|---|---|
| What it is | Small pure-Rust framework on wgpu | Minimalist Rust ML framework from Hugging Face | Full Rust deep-learning framework with swappable backends | Rust bindings to libtorch (PyTorch's C++ library) |
| GPU story | wgpu only: Metal, Vulkan, DX12 | CUDA and Metal, plus CPU | wgpu, CUDA, Metal, CPU and others through backends | Whatever libtorch supports, notably CUDA |
| Needs a C++/CUDA toolchain | No | For CUDA builds | Depends on backend | Yes, libtorch |
| Maturity | Young, pre-1.0, one maintainer | Widely used, many reference model implementations | Widely used, training-focused, large API | Mature, full PyTorch op coverage |
| Model zoo / ecosystem | Tiny | Large | Growing | The whole PyTorch ecosystem via TorchScript/weights |

Choose candle for inference of popular models, burn for a mature all-Rust
training framework, tch-rs when you need exact PyTorch semantics and CUDA.
Choose Rusty to learn how the stack works end to end, when you want a
dependency-light wgpu-only codebase you can read and modify, or to
experiment on a consumer GPU without installing CUDA. Rusty does not match
their op coverage, model support or hardening, and we have not benchmarked it
against them; kernel-level numbers for Rusty alone are in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Roadmap

- [x] Rebuilt tensor core with a CPU device, checked autograd, differentiable `nn` layers, LR schedules, safetensors state dicts
- [x] CI: fmt, clippy, tests on Linux and macOS, docs, MSRV, cargo-deny
- [ ] Reliable end-to-end LoRA SFT run, documented ([walkthrough](docs/examples/lora-sft.md))
- [ ] Published kernel benchmarks ([benchmarks/RESULTS.md](benchmarks/RESULTS.md))
- [ ] Faster GPU path for many small ops (command batching, bind-group caching)
- [ ] Parallel-reduction softmax and layer norm kernels
- [ ] `f16`/`bf16` tensors, convolutions, sort/top-k
- [ ] Unify the `rusty` tensor API with the LLM stack
- [ ] First crates.io release

## Contributing

Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers setup,
tests and the PR process; [docs/good-first-issues.md](docs/good-first-issues.md)
lists concrete starting points; [docs/writing-a-kernel.md](docs/writing-a-kernel.md)
walks through adding a GPU op.

## License

MIT, see [LICENSE](LICENSE).

## Acknowledgments

[wgpu](https://github.com/gfx-rs/wgpu) for the cross-platform GPU layer,
[safetensors](https://github.com/huggingface/safetensors) for the weight
format, and [FlashAttention](https://arxiv.org/abs/2205.14135) for the
attention algorithm.
