# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CI workflows: rustfmt, clippy, tests on Linux and macOS, rustdoc with broken-link checks, MSRV check and `cargo-deny`.
- Release workflow that builds `rusty-cli` for macOS arm64 and Linux x86_64 and attaches the binaries to a GitHub Release on `v*` tags.
- Dependabot for Cargo and GitHub Actions.
- Issue templates, pull request template, `SECURITY.md`, `CODE_OF_CONDUCT.md`.
- Crates.io package metadata (description, keywords, categories, readme, homepage) for all crates.

### Changed
- Workspace authors email corrected.
- Applied rustfmt across the workspace.

## [0.1.0]

Initial release of the Rusty workspace.

### Added
- `rusty-backend`: wgpu compute backend (Metal / Vulkan) with hand-written WGSL kernels, including Flash Attention v2 forward and backward, and a buffer pool.
- `rusty-autograd`: reverse-mode automatic differentiation with a gradient tape.
- `rusty-graph`: neural network modules (Embedding, Linear with optional LoRA, multi-head Attention with rotary embeddings, SwiGLU MLP, RMSNorm, KV cache) and a Llama / LFM2-style model builder (experimental).
- `rusty-loader`: safetensors model loading and tokenizer support.
- `rusty-trainer`: GPU AdamW, FP16 loss scaling, gradient accumulation and clipping, LoRA fine-tuning, SFT with response-only loss masking, next-token pretraining on `.txt` and packed `.bin` data, and checkpoint save and resume.
- `rusty-cli`: command-line entry point for training, evaluation and generation.
- `rusty-hub`: HuggingFace Hub integration.
- `rusty-monitor`: terminal dashboard for training runs.
- `rusty`: umbrella crate re-exporting the framework.
- `benchmarks`: benchmark binary.

[Unreleased]: https://github.com/puranikyashaswin/rusty/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/puranikyashaswin/rusty/releases/tag/v0.1.0
