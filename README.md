<p align="center">
  <img src="https://img.shields.io/badge/rust-1.75+-orange?style=for-the-badge&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/Metal-Apple%20Silicon-purple?style=for-the-badge&logo=apple" alt="Metal">
  <img src="https://img.shields.io/badge/GFLOPS-121+-green?style=for-the-badge" alt="Performance">
  <img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" alt="License">
</p>

<h1 align="center">Rusty</h1>

<h3 align="center">GPU-Accelerated ML Framework in Pure Rust</h3>

<p align="center">
  <strong>From custom GPU kernels to Llama architecture — understanding ML at every layer</strong>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-performance">Benchmarks</a> •
  <a href="#-examples">Examples</a>
</p>

---

## Overview

Rusty is a GPU-accelerated machine learning framework built entirely from scratch in Rust. The most dependable path today is the **World Trainer** lite path: LoRA SFT (response-only loss + chat template) and short-sequence pretrain/eval for smaller models on ~8GB consumer hardware. Full-scale / distributed training remains a later milestone.

**What makes this project unique:**

- **Custom GPU Kernels** — Hand-written WGSL compute shaders, not relying on cuBLAS or external libraries
- **LFM2 / Llama-style blocks** — Attention, SwiGLU MLP, RMSNorm, KV cache (experimental)
- **LoRA adapters** — Attach, train, save, and resume low-rank adapters
- **Cross-Platform GPU** — Runs on Metal (Apple Silicon) and Vulkan (Windows/Linux)

---

## Quick Start

Run a GPU demo in 30 seconds:

```bash
git clone https://github.com/puranikyashaswin/rusty.git
cd rusty
cargo run --example basic_tensor --release -p rusty
```

**Expected output:**
```
Rusty ML - Basic Tensor Example

[GPU] Apple M2 (Metal)

[INIT] Creating tensors...
       Tensor A: [32, 32]
       Tensor B: [32, 32]

[COMPUTE] Performing matrix multiplication...
          Result shape: [32, 32]
          First few values: [..., ..., ..., ..., ...]  (GPU-dependent)

[DONE] All operations completed successfully!
```

---

## Key Features

### GPU Compute Engine

Custom WGSL compute shaders optimized for ML workloads:

| Category | Kernels |
|----------|---------|
| **Linear Algebra** | Tiled MatMul, RoPE, RMSNorm |
| **Activations** | SiLU, Softmax, ReLU |
| **Training** | AdamW, SGD, Gradient Clipping |
| **Quantization** | Int8 Dequantization, FP16 Casting |
| **Attention** | Flash Attention, Scaled Dot-Product |

### Neural Network Layers

Production-ready building blocks:

- **Embedding** — Token embedding with vocabulary lookup
- **Linear** — Dense layers with optional LoRA adapters
- **Attention** — Multi-head attention with rotary embeddings
- **MLP** — SwiGLU feedforward network
- **LlamaBlock** — Transformer block (experimental)
- **Lfm2Model** — LFM2-style model builder used by the CLI

### Training Infrastructure

- Automatic differentiation with gradient tape
- GPU-accelerated AdamW optimizer
- Mixed precision loss scaling (FP16)
- Gradient accumulation and clipping
- LoRA for parameter-efficient fine-tuning
- SFT with chat-template tokenization and **response-only** loss masking
- Pretrain / next-token LM on `.txt` and packed `.bin` token streams
- Unified training checkpoints with transactional resume validation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│      rusty-cli          │      rusty-monitor                    │
│   Command Line Interface│   TUI Dashboard + Training Viz        │
├─────────────────────────────────────────────────────────────────┤
│      rusty-trainer      │      rusty-loader                     │
│      Training Loops     │      Safetensors + Tokenizer          │
├─────────────────────────────────────────────────────────────────┤
│                       rusty-graph                               │
│            Neural Networks: Attention, MLP, Llama               │
├─────────────────────────────────────────────────────────────────┤
│                      rusty-autograd                             │
│             Automatic Differentiation + Optimizers              │
├─────────────────────────────────────────────────────────────────┤
│                      rusty-backend                              │
│              GPU Compute Engine + WGSL Kernels                  │
├─────────────────────────────────────────────────────────────────┤
│                     Metal / Vulkan                              │
│                   Apple M1/M2/M3, GPUs                          │
└─────────────────────────────────────────────────────────────────┘
```

### Crate Overview

| Crate | Description |
|-------|-------------|
| **rusty-backend** | GPU compute engine with custom WGSL shaders |
| **rusty-graph** | Neural network layers (Attention, MLP, LlamaBlock) |
| **rusty-autograd** | Automatic differentiation and optimizers |
| **rusty-loader** | Safetensors and tokenizer loading |
| **rusty-trainer** | Training loops with mixed precision |
| **rusty-monitor** | TUI dashboard with real-time training visualization |
| **rusty-cli** | Command-line interface |

---

## Performance

Benchmarked on Apple M2 (Metal backend). MatMul numbers are from `benchmarks/RESULTS.md`; run locally to reproduce.

| Operation | Size | Throughput |
|-----------|------|------------|
| MatMul | 4096×4096 | **121 GFLOPS** |
| MatMul | 2048×2048 | 114 GFLOPS |
| MatMul | 1024×1024 | 109 GFLOPS |

<details>
<summary>Run benchmarks</summary>

```bash
cargo run -p rusty-benchmarks --release
```

</details>

---

## Examples

### Basic GPU Operations

```bash
cargo run --example basic_tensor --release -p rusty
```

Matrix multiplication, element-wise operations, and activations on GPU.

### Flash Attention

```bash
cargo run --example flash_attention --release -p rusty
```

Memory-efficient attention with O(N) memory instead of O(N²).

### LoRA Fine-tuning

```bash
cargo run --example lora_finetune --release -p rusty
```

Parameter-efficient training with low-rank adapters.

### Training Demo

Simulated training demo (loss curve only — no GPU backward pass):

```bash
cargo run -p rusty-cli --release -- --demo
```

---

## Fine-tuning Models

Rusty World Trainer is **lite-first**: LoRA SFT and short-sequence pretrain/continued LM on ~8GB MacBook-class hardware.

### Step 1: Download a Model

```bash
./scripts/download_model.sh tinyllama
```

Or manually:
```bash
pip install huggingface-hub
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ./models/tinyllama
```

### Step 2: Prepare Training Data

SFT (`--task sft`) expects `.json` / `.jsonl` with `{prompt,response}` (optional `system`). Chat template encoding is applied by default; loss is computed **only on response tokens**.

```json
[
  {"prompt": "Who are you?", "response": "I am an AI assistant."},
  {"prompt": "What can you do?", "response": "I can answer questions and assist with tasks."}
]
```

Pretrain (`--task pretrain`) accepts:

- `.txt` — one sample per non-empty line
- `.bin` — packed little-endian `u32` token stream (fixed windows)

### Step 3: Train / eval / infer (recommended CLI)

```bash
# 8GB MacBook SFT (LoRA, safe defaults, response-only loss + chat template)
cargo run -p rusty-cli --release -- train \
  --lite --task sft \
  --model ./models/tinyllama \
  --data ./data/train.jsonl \
  --out ./outputs/run1

# Continued pretrain / next-token LM on packed tokens
cargo run -p rusty-cli --release -- train \
  --lite --task pretrain \
  --model ./models/tinyllama \
  --data ./data/train.bin \
  --out ./outputs/lm1

# Eval (same masking rules as train)
cargo run -p rusty-cli --release -- eval \
  --task sft \
  --model ./models/tinyllama \
  --data ./data/val.jsonl

# Resume (validates task / seq-len / train-modules / model path)
cargo run -p rusty-cli --release -- train \
  --lite --task sft \
  --model ./models/tinyllama \
  --data ./data/train.jsonl \
  --resume ./outputs/run1/lora_adapters.safetensors \
  --out ./outputs/run1

# Interactive-style generation
cargo run -p rusty-cli --release -- infer \
  --model ./models/tinyllama \
  --prompt "Hello"
```

`--lite` / `--auto` set consumer-safe defaults (LoRA, fp16, grad-accum=8, seq-len=128, epochs=3, max-vram ~8GB) unless you override them explicitly. Silent demo-data fallback is **disabled** unless `--allow-fallback` is set. Use `--no-chat-template` only if you intentionally want raw prompt/response tokenization for SFT.

Legacy positional / `--train` flags still work for older scripts.

---

## Supported Models

| Model | Status |
|-------|--------|
| LLaMA / LLaMA-2 / LLaMA-3 | Experimental (single-shard, LFM2 builder) |
| TinyLlama | Best tested and recommended lightweight path |
| Mistral | Config detection only |
| Phi / Phi-2 / Phi-3 | Download script available |
| Qwen / Qwen-2 | Config detection only |
| Gemma / Gemma-2 | Download script available |

Sharded safetensors through `model.safetensors.index.json` are supported. Larger-model training paths remain experimental even when loading succeeds.

---

## Requirements

- **Rust** 1.75 or later
- **GPU**: Apple Silicon (M1/M2/M3) or Vulkan-capable GPU recommended
- **OS**: macOS, Linux, or Windows

For the best current experience, use a small model such as TinyLlama or Phi-mini class checkpoints, LoRA training, and conservative sequence length / batch size settings.

---

## Project Status

| Component | Status |
|-----------|--------|
| GPU Backend (wgpu) | Working — matmul, element-wise, activations |
| Custom WGSL Kernels | Working — core ops; attention/norm fixes in progress |
| LFM2 / Llama-style graph | Experimental — inference plus lightweight fine-tuning |
| LoRA adapters | Working — attach, train, save, resume |
| Autograd + Optimizers | Partial — low-level `rusty-autograd`; trainer integration for LoRA/SFT |
| Safetensors Loading | Working — single + sharded files |
| Mixed Precision (FP16) | Working — `GradScaler` with checkpointing |
| Flash Attention | Partial — forward V2; backward WIP |
| CUDA Backend | Planned |
| Distributed Training | Planned |

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/YOUR_USERNAME/rusty.git
git checkout -b feature/your-feature
cargo test --workspace
git commit -m "feat: description"
git push origin feature/your-feature
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgments

- [wgpu](https://github.com/gfx-rs/wgpu) — Cross-platform GPU API
- [safetensors](https://github.com/huggingface/safetensors) — Safe tensor serialization
- [Flash Attention](https://arxiv.org/abs/2205.14135) — Memory-efficient attention

---

<p align="center">
  <strong>Built with Rust</strong>
</p>
