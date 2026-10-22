# Ferrotensor

Ferrotensor is a pure-Rust GPU ML framework. It runs tensors, autograd and
neural networks on whatever GPU you have (Apple Silicon through Metal, or
Vulkan/DX12 on Windows and Linux) using [wgpu](https://wgpu.rs) and
hand-written WGSL compute kernels. There is no CUDA, cuDNN or libtorch to
install: `cargo build` is the whole toolchain. Python bindings wrap the same
tensor API.

The same API also runs on a CPU device. Every differentiable op is tested
against finite differences on the CPU, and every GPU kernel against the CPU.

It is young: pre-1.0, `f32` only, single GPU, and small-scale by design.

## Where to start

| If you want to | Read |
|---|---|
| Install the crate, the CLI or the Python package | [Installation](installation.md) |
| Train a model from Python | [Python quickstart](python.md) |
| See runnable Rust programs and their real output | [Use cases](use-cases.md) |
| Fine-tune a real model with LoRA on an 8 GB laptop | [LoRA fine-tuning](examples/lora-sft.md) |
| Understand how a tensor op reaches a WGSL kernel | [Architecture](architecture.md) |
| Add your own GPU op | [Writing a kernel](writing-a-kernel.md) |
| Check the numbers and how they were measured | [Benchmarks](benchmarks.md) |

The Rust API reference is on [docs.rs](https://docs.rs/ferrotensor). The
source, issues and releases are on
[GitHub](https://github.com/puranikyashaswin/ferrotensor).

This guide is built from the documents in the repository, so it always
matches the files on the `main` branch.
