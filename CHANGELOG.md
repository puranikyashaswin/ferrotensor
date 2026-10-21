# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First public version. Nothing has been published to crates.io yet.

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

[Unreleased]: https://github.com/puranikyashaswin/ferrotensor/commits/main
