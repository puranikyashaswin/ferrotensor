# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `ferrotensor info` lists every GPU adapter wgpu can open (name, vendor, backend, driver, the limits the kernels depend on) and marks the one Ferrotensor uses. `--check` runs matmul, softmax and RMSNorm on it against an f64 CPU reference and exits non-zero if one is out of tolerance; `--json` prints the same for bug reports.
- `ferrotensor-backend`: `adapters::list_adapters` and `adapters::self_check`, the library side of `ferrotensor info`.
- A guide to testing Ferrotensor on other GPUs (`docs/testing-on-your-gpu.md`), and a field for `ferrotensor info --check --json` output in the bug report template.

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

[Unreleased]: https://github.com/puranikyashaswin/ferrotensor/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/puranikyashaswin/ferrotensor/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/puranikyashaswin/ferrotensor/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/puranikyashaswin/ferrotensor/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/puranikyashaswin/ferrotensor/releases/tag/v0.1.0
