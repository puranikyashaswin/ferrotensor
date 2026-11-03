# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- A how-to guide on the website (`/guide/`): 19 step-by-step guides, from installing and checking your GPU to LoRA fine-tuning, training in Rust or Python, performance and troubleshooting. Each has the real output to expect, annotated, with the machine it came from.

### Changed
- `ferrotensor-backend`: uploading tensors no longer keeps a host-side staging copy of every upload until the next GPU submission. `WgpuContext::note_upload` counts uploaded bytes and every 256 MiB (`UPLOAD_RELEASE_BYTES`) waits for the copies so wgpu frees them; `synchronize` now also submits pending uploads. Loading a model held a staging copy of all its weights (6.8 GB for SmolLM2-1.7B in fp32); in the CI test, uploading 768 MiB grew peak memory by 1536 MiB before and is bounded at under 1.6x now.
- The README, CONTRIBUTING, the GPU testing guide, the architecture notes and the website's benchmarks and LoRA pages describe the NVIDIA results, the software-Vulkan CI job and op fusion; the README no longer says the Python package is missing from PyPI.

## [0.5.0] - 2026-11-02

### Added
- LoRA results on an NVIDIA A10G for SmolLM2-135M, 360M and 1.7B in the LoRA walkthrough (`docs/examples/lora-sft.md`): held-out loss 1.6521, 1.3228 and 1.0667 after 250 steps, at 1.33, 1.69 and 2.47 s per step. With the GPU changes below, SmolLM2-135M on the A10G went from 20.2 s to 1.33 s per step; its held-out loss is 1.6521 (1.6519 before, as on an Apple M2).
- Kernel and Tensor API benchmark results on an NVIDIA A10G and a Tesla T4 through Vulkan in `benchmarks/RESULTS.md`, with op fusion off and on.
- `scripts/lora-run.sh MODEL` runs the LoRA SFT walkthrough on any Llama-style Hugging Face model with the walkthrough's data and settings and writes a JSON summary (held-out loss before and after, loss curve, seconds per step, peak memory, greedy generations).
- Op fusion, off by default: with `FERROTENSOR_FUSION=1`, consecutive same-size element-wise ops on the GPU run as one `fused` dispatch of up to six steps instead of one dispatch each, still writing every intermediate. Results are checked bit-identical to unfused runs. See `docs/design/op-fusion.md`.
- The `WGPU_BACKEND` environment variable (`vulkan`, `metal`, `dx12`, `gl` or a comma list) restricts which backends `Device::new`/`gpu`/`best`, `WgpuContext::new` and `ferrotensor info` try.
- CI runs the GPU test suites and the autograd finite-difference checks on Mesa lavapipe (Vulkan on the CPU) for every PR, so the Vulkan path is checked without a GPU.
- `scripts/hardware-report.sh` runs `ferrotensor info --check`, the kernel benchmarks and the Tensor API benchmarks and writes one JSON report, so every machine is measured the same way. CI runs it in quick mode on software Vulkan.

### Changed
- `ferrotensor-autograd`: the backward passes of RMSNorm and RoPE run on the GPU (`ComputeEngine::rms_norm_backward`, `ComputeEngine::rope_theta_inverse`) instead of reading the input, weight and gradient back to the host. LoRA training made about 2,000 blocking GPU-to-host reads per step through these two ops.
- **Breaking**, `ferrotensor-trainer`: `masked_cross_entropy` computes the loss and its gradient on the GPU (`ComputeEngine::masked_cross_entropy`) and reads back only one loss per position. Before, it read the whole `[seq, vocab]` logits to the host and uploaded a gradient of the same size, about 50 MB each way per SmolLM2 sequence. It now takes the `&ComputeEngine` as a fourth argument.
- `ferrotensor-backend`: `ComputeEngine` records kernel dispatches and copies into one pending command encoder on the `WgpuContext` and submits them together (every 128 ops, and when a tensor is read or written) instead of one queue submission per op. `WgpuContext::flush` submits the pending work; call it before submitting your own commands on `queue`. `WgpuContext::synchronize` waits for the GPU to finish.
- `ferrotensor-backend`: reading a tensor no longer blocks in wgpu's `Maintain::Wait`, which in wgpu-core 0.19 gives up after 5 s and frees the still-running submission's resources; it polls until the GPU is done.
- `ferrotensor-trainer`: the global gradient norm sums squares on the GPU (`ComputeEngine::sum_squares_partial`) and reads back once per optimizer step instead of once per gradient tensor.

### Fixed
- `ferrotensor`: waiting for the GPU no longer falls back to wgpu's blocking `Maintain::Wait`, which in wgpu-core 0.19 gives up after 5 s and frees the still-running submission's resources. A batch of more than 5 s of GPU work could crash the driver.
- The LoRA walkthrough called `target/release/ferrotensor-cli`; the binary has been `ferrotensor` since the rename.

## [0.4.0] - 2026-11-01

### Added
- `ferrotensor info` lists every GPU adapter wgpu can open (name, vendor, backend, driver, the limits the kernels depend on) and marks the one Ferrotensor uses. `--check` runs matmul, softmax and RMSNorm on it against an f64 CPU reference and exits non-zero if one is out of tolerance; `--json` prints the same for bug reports.
- `ferrotensor-backend`: `adapters::list_adapters` and `adapters::self_check`, the library side of `ferrotensor info`.
- A guide to testing Ferrotensor on other GPUs (`docs/testing-on-your-gpu.md`), and a field for `ferrotensor info --check --json` output in the bug report template.

### Changed
- `ferrotensor`: the backward passes of `index_select` and `nn::Embedding` scatter-add their gradients on the GPU instead of reading ids and gradients back to the host. The sums run in a fixed order, so the gradients are deterministic and match the CPU bit for bit.
- `ferrotensor`: `cat` concatenates on the GPU, and `cross_entropy` with class-id targets runs forward and backward on the GPU. On an Apple M2, `cross_entropy` over `[4096, 32000]` went from 749.6 ms to 64.6 ms on the GPU and from 4226.9 ms to 517.2 ms on the CPU, and an MLP training step from 3.69 ms to 2.13 ms on the GPU.
- `cross_entropy` with class ids computes the loss per row (log-sum-exp minus the target logit), so its last bits can differ from 0.3.0.
- The README feature matrix and the website list what is measured and what is not: the LoRA row now names SmolLM2-135M, the one model measured.

## [0.3.0] - 2026-10-26

### Added
- `ferrotensor`: `manual_seed(seed)` (also in the prelude) seeds the generator behind `Tensor::randn`, `rand`, `uniform`, every `nn` layer's initialisation and `functional::dropout`, so a program gives the same numbers on every run. Unseeded behaviour is unchanged. Python: `ferrotensor.manual_seed(seed)`.

### Changed
- `ferrotensor`: reducing a whole tensor on the GPU (`sum()` and the other full reductions) uses a two-pass workgroup tree instead of one thread for every element. On an Apple M2, `sum()` of `[4096, 4096]` went from 2169 ms to 1.19 ms.
- `ferrotensor`: `argmax` and `argmin` run entirely on the GPU, so they no longer split the batched dispatch. `argmax_dim(-1)` of `[1024, 4096]` went from 8.40 ms to 0.99 ms on an M2. Ties resolve to the same index as on the CPU.
- The website takes its version from the workspace `Cargo.toml` and shows the newest CHANGELOG release.

### Removed
- `ferrotensor-graph`: `FlashAttention::forward_multi_head` and `FlashAttention::multi_head_forward`, deprecated in 0.2.1. Use `forward` (it already splits heads), projecting the input yourself.

### Fixed
- The Python MLP training test seeds its weight initialisation, so it no longer fails on a rare unlucky start.

## [0.2.1] - 2026-10-25

### Fixed
- `ferrotensor-graph`: `FlashAttention::memory_savings` printed `0MB` for short sequences because sizes were divided by 1,000,000 as integers. Sizes now print in binary units and are labelled per head, e.g. `8.0x less memory (128.0 KiB vs 1.0 MiB per head for seq=512)`.
- `ferrotensor`: `optim::clip_grad_norm` reads the gradient norm back once instead of once per parameter, so a GPU run syncs once. The returned norm and the clipping are unchanged.

### Security
- Updated tokio from 1.51.0 to 1.54.1 for [RUSTSEC-2026-0386](https://rustsec.org/advisories/RUSTSEC-2026-0386) (unsound `AsyncFd` constructors). Only `ferrotensor-monitor` uses tokio.

### Deprecated
- `ferrotensor-graph`: `FlashAttention::forward_multi_head` (same as `forward`) and `FlashAttention::multi_head_forward` (applies no projections). Both will be removed in 0.3.

## [0.2.0] - 2026-10-22

### Added
- **Python bindings** (`ferrotensor-py`, published to PyPI as `ferrotensor`): `Device`, `Tensor` with autograd and operators (`@` for matmul), `nn` layers (Linear, Embedding, LayerNorm, RMSNorm, Dropout, activations, Sequential), losses, SGD and AdamW with LR schedules, and safetensors state dicts. abi3 wheels for CPython 3.10+ on macOS arm64 and Linux x86_64, type stubs, a pytest suite and an MLP example.
- `ferrotensor`: `Tensor::tril` and `Tensor::triu`; the causal mask is built with `triu`.
- `ferrotensor`: a KV cache for `nn::MultiHeadAttention` (`forward_cached` with `nn::KvCache`). Decoding one token at a time matches the full causal forward within 1e-5 on CPU and GPU; `char_transformer` samples with it.
- `ferrotensor`: `DataLoader::epoch_prefetched` gathers batches on a background thread, in the same order as `epoch`.
- GPU parity tests for every `nn` layer and for safetensors round trips; a Tensor API benchmark (`benchmarks`, `tensor-api` binary) for CPU and GPU devices.
- CI builds and tests the Python wheels on every push and publishes them to PyPI from `v*` tags through trusted publishing.

### Changed
- `ferrotensor`: GPU ops are batched into one submission and one compute pass per sync point, reuse freed buffers, and read back without wgpu's 1 ms sleep on Metal. On an M2, 100 small element-wise ops go from 10.6 ms to 0.82 ms and an MLP training step (784-256-10, batch 64) from 10.8 ms to 3.98 ms; `char_transformer` trains 600 steps in 31.9 s (was 105.0 s). Results are bit for bit unchanged.
- `ferrotensor`: `Embedding::forward` gathers on the GPU without reading the ids back. Ids outside `[0, vocab)` now give zero rows and no gradient on both devices (before: a panic, or row 0 for negative ids).
- `ferrotensor`: `cross_entropy`, `nll_loss`, `accuracy` and `one_hot` take `impl Into<Targets>`: an f32 tensor of class ids or a slice, array or `Vec` of `i64`, `i32` or `usize`. Negative ids are skipped everywhere, so `accuracy` now ignores `-100` rows like the losses. Existing call forms still compile.
- `ferrotensor`: `DataLoader` stacks batches into one preallocated buffer instead of cloning each sample (`Dataset::with_sample`). An epoch of 20k × 784 floats at batch 128 goes from 63,923 allocations (251 MB) and 11.3 ms to 2,043 allocations (63 MB) and 6.4 ms on an M2.
- The `ferrotensor-backend` examples are ported to the current API and check their results against a CPU reference.
- Docs: install lines, a Python quickstart, batched dispatch in the architecture guide, refreshed timings, and new good first issues.

### Removed
- `ferrotensor`: the `metal` and `vulkan` Cargo features (no code read them) and the unused dependencies on the internal LLM crates, serde, serde_json, memmap2, log, env_logger and criterion.

## [0.1.0] - 2026-10-21

First public version.

### Added
- `ferrotensor`: tensor API with a CPU device and a wgpu GPU device, reverse-mode autograd checked against finite differences, differentiable `nn` layers (Linear, Embedding, LayerNorm, RMSNorm, Dropout, activations), SGD/Adam/AdamW, gradient clipping, LR schedules (cosine with warmup), `Dataset`/`DataLoader`, and safetensors save/load of module state dicts.
- `ferrotensor-backend`: wgpu compute backend (Metal / Vulkan / DX12) with hand-written WGSL kernels: register-blocked vec4 matmul, Flash Attention v2 forward and backward with causal offset, sliding window, masks and dropout, softmax, RMSNorm, RoPE, int8 dequantization, and a capped buffer pool. Every kernel is tested against a CPU reference.
- `ferrotensor-autograd`: GPU autograd tape used by the trainer.
- `ferrotensor-graph`: Llama / LFM2-style blocks (attention with RoPE, SwiGLU MLP, RMSNorm, KV cache, LoRA linear layers).
- `ferrotensor-loader`: safetensors and tokenizer loading for Llama-style Hugging Face checkpoints.
- `ferrotensor-trainer`: LoRA SFT with response-only loss and chat template, next-token pretraining, AdamW, FP16 loss scaling, gradient accumulation, checkpoint save and resume.
- `ferrotensor` command-line tool (`ferrotensor-cli` crate): `train`, `eval` and `generate` with temperature, top-k and top-p sampling and a KV cache.
- `ferrotensor-hub` (Hugging Face Hub downloads) and `ferrotensor-monitor` (terminal dashboard).
- Benchmark suite with recorded results (`benchmarks/RESULTS.md`) and an end-to-end LoRA walkthrough on SmolLM2-135M (`docs/examples/lora-sft.md`).
- CI (rustfmt, clippy with `-D warnings`, tests on Linux and macOS, rustdoc, MSRV, cargo-deny), release workflow for `v*` tags, issue and PR templates, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, architecture and kernel-writing guides.

### Changed
- Renamed the project from Rusty to Ferrotensor; crates are now `ferrotensor-*` and the CLI binary is `ferrotensor`.
- Licensed under MIT OR Apache-2.0 (previously MIT only).

[Unreleased]: https://github.com/puranikyashaswin/ferrotensor/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/puranikyashaswin/ferrotensor/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/puranikyashaswin/ferrotensor/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/puranikyashaswin/ferrotensor/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/puranikyashaswin/ferrotensor/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/puranikyashaswin/ferrotensor/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/puranikyashaswin/ferrotensor/releases/tag/v0.1.0
