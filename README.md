<h1 align="center">Ferrotensor</h1>

<p align="center"><strong>A pure-Rust GPU ML framework: no CUDA, no Python, no C++ toolchain.</strong></p>

<p align="center">
  <a href="https://github.com/puranikyashaswin/ferrotensor/actions/workflows/ci.yml"><img src="https://github.com/puranikyashaswin/ferrotensor/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/rust-1.88+-orange" alt="Rust 1.88+">
  <img src="https://img.shields.io/badge/GPU-Metal%20%7C%20Vulkan%20%7C%20DX12-purple" alt="GPU backends">
  <img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue" alt="MIT OR Apache-2.0">
  <img src="https://img.shields.io/badge/status-pre--1.0-yellow" alt="pre-1.0">
</p>

Ferrotensor runs tensors, autograd and neural networks on whatever GPU you have
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
- Docs: [architecture](docs/architecture.md) · [writing a kernel](docs/writing-a-kernel.md) · [use cases](docs/use-cases.md) · [LoRA SFT walkthrough](docs/examples/lora-sft.md) · [benchmarks](benchmarks/RESULTS.md)


## Highlights

Measured on a MacBook Air (Apple M2, 8 GB unified memory) through Metal:

- **LoRA fine-tuning of a real model on a laptop.** SmolLM2-135M on 2,000
  Alpaca examples: held-out loss 2.43 → **1.65** (perplexity 11.3 → 5.2) in
  250 steps at 8.8 s/step, 3.3 GB peak memory, 3.7 MB adapter.
  [Full walkthrough with every command](docs/examples/lora-sft.md).
- **Hand-written WGSL kernels.** matmul reaches **611 GFLOPS** at 4096³
  (5× the first version); causal flash attention forward is 4× faster at
  sequence length 2048. [Benchmarks and method](benchmarks/RESULTS.md).
- **Checked, not assumed.** Every differentiable op is tested against finite
  differences on the CPU, and every GPU kernel against a CPU reference.

## Quickstart

You need Rust 1.88+ (`rustup` is enough). On macOS no other setup is needed.

```bash
git clone https://github.com/puranikyashaswin/ferrotensor.git
cd ferrotensor
cargo run --release -p ferrotensor --example quickstart
```

Output on an Apple M2 (the device line changes with your hardware):

```
device: Apple M2 (Metal)
x @ w = [7.5, 2.0, 18.0, 2.0] (shape [2, 2])
dL/dw = [5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
```

That is [`ferrotensor/examples/quickstart.rs`](ferrotensor/examples/quickstart.rs): a
matmul and its gradient on the best available device. No GPU? Use
`Device::cpu()`; the same code runs.

In your own code:

```rust
use ferrotensor::prelude::*;

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
| Safetensors save/load, state dicts | **Stable** | `ferrotensor::serialize` |
| `DataLoader` | **Stable** | Batching and seeded shuffling; single-threaded |
| LoRA fine-tuning of Llama-style models (`ferrotensor-cli`) | **Experimental** | Small models (TinyLlama class) on ~8 GB machines; see the [LoRA walkthrough](docs/examples/lora-sft.md) |
| Llama / LFM2-style blocks (`ferrotensor-graph`) | **Experimental** | Attention with RoPE, SwiGLU MLP, RMSNorm, KV cache |
| Flash attention kernels (`ferrotensor-backend`) | **Experimental** | Forward v2 and a backward kernel; see `tests/flash_grad_check.rs` |
| FP16 loss scaling, checkpoint/resume (`ferrotensor-trainer`) | **Experimental** | |
| Terminal monitor, Hub download helpers | **Experimental** | `ferrotensor-monitor`, `ferrotensor-hub` |
| `f16` / `bf16` tensors in the `ferrotensor` API | Planned | |
| Convolutions, sort/top-k, cumsum in the `ferrotensor` API | Planned | |
| Faster small-op GPU path (batched dispatch) | Planned | Per-op submission makes tiny GPU workloads slower than the CPU |
| Multi-GPU / distributed training | Planned | `ferrotensor-cli` has a file-based gradient sync flag (`--world-size`) that is a prototype |
| A CUDA backend | Not planned | wgpu is the point of the project |

## Architecture

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

    ferrotensor["ferrotensor (Tensor, autograd, nn, optim)"] --> kernels["own WGSL kernels + CPU reference"]
    kernels --> wgpu
```

The workspace has two layers. The **`ferrotensor`** crate is the PyTorch-style tensor
API (it carries its own small set of generic kernels and a CPU device). The
**LLM stack** (`ferrotensor-backend` through `ferrotensor-cli`) has purpose-built kernels
for Llama-style models and the LoRA fine-tuning CLI. Details, including how a
tensor op reaches a WGSL kernel, are in [docs/architecture.md](docs/architecture.md).

| Crate | What it is |
|-------|------------|
| [`ferrotensor`](ferrotensor) | Tensor, autograd, `nn`, `optim`, `serialize`, `data` |
| [`ferrotensor-backend`](ferrotensor-backend) | wgpu context, WGSL kernels (matmul, RoPE, RMSNorm, flash attention, AdamW, ...), buffer pool |
| [`ferrotensor-autograd`](ferrotensor-autograd) | Gradient tape for the backend tensors |
| [`ferrotensor-graph`](ferrotensor-graph) | Transformer building blocks and the Llama/LFM2-style model builder |
| [`ferrotensor-loader`](ferrotensor-loader) | Safetensors (single and sharded) and tokenizer loading |
| [`ferrotensor-trainer`](ferrotensor-trainer) | Training loops, LoRA, loss scaling, checkpoints |
| [`ferrotensor-cli`](ferrotensor-cli) | `train`, `eval` and `infer` commands |
| [`ferrotensor-monitor`](ferrotensor-monitor) | Terminal dashboard |
| [`ferrotensor-hub`](ferrotensor-hub) | Hugging Face Hub download helpers |
| [`benchmarks`](benchmarks) | GPU kernel benchmarks |

## Fine-tuning a model

`ferrotensor-cli` fine-tunes small Llama-style models with LoRA (response-only loss
and a chat template for SFT) and can run short pretraining on `.txt` or packed
`.bin` token files:

```bash
cargo run -p ferrotensor-cli --release -- train \
  --lite --task sft \
  --model ./models/tinyllama \
  --data ./data/train.jsonl \
  --out ./outputs/run1
```

`--help` lists every flag. A step-by-step walkthrough with real output is in
[docs/examples/lora-sft.md](docs/examples/lora-sft.md).
This path is experimental.

## Compared with candle, burn and tch

These are good projects; Ferrotensor is not trying to replace them. The honest
picture, as of this writing:

| | Ferrotensor | [candle](https://github.com/huggingface/candle) | [burn](https://github.com/tracel-ai/burn) | [tch-rs](https://github.com/LaurentMazare/tch-rs) |
|---|---|---|---|---|
| What it is | Small pure-Rust framework on wgpu | Minimalist Rust ML framework from Hugging Face | Full Rust deep-learning framework with swappable backends | Rust bindings to libtorch (PyTorch's C++ library) |
| GPU story | wgpu only: Metal, Vulkan, DX12 | CUDA and Metal, plus CPU | wgpu, CUDA, Metal, CPU and others through backends | Whatever libtorch supports, notably CUDA |
| Needs a C++/CUDA toolchain | No | For CUDA builds | Depends on backend | Yes, libtorch |
| Maturity | Young, pre-1.0, one maintainer | Widely used, many reference model implementations | Widely used, training-focused, large API | Mature, full PyTorch op coverage |
| Model zoo / ecosystem | Tiny | Large | Growing | The whole PyTorch ecosystem via TorchScript/weights |

Choose candle for inference of popular models, burn for a mature all-Rust
training framework, tch-rs when you need exact PyTorch semantics and CUDA.
Choose Ferrotensor to learn how the stack works end to end, when you want a
dependency-light wgpu-only codebase you can read and modify, or to
experiment on a consumer GPU without installing CUDA. Ferrotensor does not match
their op coverage, model support or hardening, and we have not benchmarked it
against them; kernel-level numbers for Ferrotensor alone are in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Roadmap

- [x] Rebuilt tensor core with a CPU device, checked autograd, differentiable `nn` layers, LR schedules, safetensors state dicts
- [x] CI: fmt, clippy, tests on Linux and macOS, docs, MSRV, cargo-deny
- [ ] Reliable end-to-end LoRA SFT run, documented ([walkthrough](docs/examples/lora-sft.md))
- [ ] Published kernel benchmarks ([benchmarks/RESULTS.md](benchmarks/RESULTS.md))
- [ ] Faster GPU path for many small ops (command batching, bind-group caching)
- [ ] Parallel-reduction softmax and layer norm kernels
- [ ] `f16`/`bf16` tensors, convolutions, sort/top-k
- [ ] Unify the `ferrotensor` tensor API with the LLM stack
- [ ] First crates.io release

## Contributing

Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers setup,
tests and the PR process; [docs/good-first-issues.md](docs/good-first-issues.md)
lists concrete starting points; [docs/writing-a-kernel.md](docs/writing-a-kernel.md)
walks through adding a GPU op.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. Unless you explicitly state otherwise, any contribution
intentionally submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual licensed as above, without any additional
terms or conditions.

## Acknowledgments

[wgpu](https://github.com/gfx-rs/wgpu) for the cross-platform GPU layer,
[safetensors](https://github.com/huggingface/safetensors) for the weight
format, and [FlashAttention](https://arxiv.org/abs/2205.14135) for the
attention algorithm.
