# Ferrotensor

Ferrotensor is a pure-Rust GPU ML framework. It runs tensors, autograd and
neural networks on the GPU using [wgpu](https://wgpu.rs) and hand-written
WGSL compute kernels, which run through Vulkan, Metal or DX12. It has been
tested on an NVIDIA A10G (24 GB) through Vulkan, an Apple M2 (8 GB) through
Metal, and software Vulkan (Mesa lavapipe) in CI; other GPUs are untested,
and [reports are welcome](testing-on-your-gpu.md). There is no CUDA, cuDNN
or libtorch to install: `cargo build` is the whole toolchain. Python
bindings wrap the same tensor API.

The same API also runs on a CPU device, which the tests use as the
reference for the GPU. Op gradients are tested against finite differences
on the CPU, and the LoRA loss and gradients against PyTorch autograd.

It is young: pre-1.0, single GPU, and small-scale by design. Arithmetic and
trainable parameters are `f32`; the CLI stores frozen LoRA base weights as `f16` by
default to save memory (`--f32-weights` keeps them `f32`).

## Where to start

| If you want to | Read |
|---|---|
| Install the crate, the CLI or the Python package | [Installation](installation.md) |
| Train a model from Python | [Python quickstart](python.md) |
| See runnable Rust programs and their real output | [Use cases](use-cases.md) |
| Fine-tune a real model with LoRA, step by step | [LoRA fine-tuning](examples/lora-sft.md) |
| Understand how a tensor op reaches a WGSL kernel | [Architecture](architecture.md) |
| Add your own GPU op | [Writing a kernel](writing-a-kernel.md) |
| Check the numbers and how they were measured | [Benchmarks](benchmarks.md) |

The Rust API reference is on [docs.rs](https://docs.rs/ferrotensor). The
source, issues and releases are on
[GitHub](https://github.com/puranikyashaswin/ferrotensor).

This guide is built from the documents in the repository, so it always
matches the files on the `main` branch.
