# Rusty Examples

This directory contains runnable examples demonstrating the Rusty ML framework.

## Quick Start

All examples can be run with:

```bash
cargo run --example <example_name> --release -p rusty
```

## Available Examples

### 1. `basic_tensor` - GPU Tensor Operations

**Best for:** First-time users, verifying GPU setup

```bash
cargo run --example basic_tensor --release -p rusty
```

**Demonstrates:**
- GPU initialization with wgpu/Metal
- Creating tensors on GPU
- Matrix multiplication
- Element-wise operations (add, multiply)
- SiLU activation function

**Expected output** (values vary by GPU):
```
Rusty ML - Basic Tensor Example

[GPU] Apple M2 (Metal)

[INIT] Creating tensors...
       Tensor A: [32, 32]
       Tensor B: [32, 32]

[COMPUTE] Performing matrix multiplication...
          Result shape: [32, 32]
          First few values: [..., ..., ..., ..., ...]

[DONE] All operations completed successfully!
```

---

### 2. `flash_attention` - Memory-Efficient Attention

**Best for:** Understanding Flash Attention implementation

```bash
cargo run --example flash_attention --release -p rusty
```

**Demonstrates:**
- Flash Attention algorithm
- O(N) memory vs O(N²) standard attention
- Tiled computation on GPU
- Memory savings calculation

---

### 3. `lora_finetune` - LoRA Fine-tuning

**Best for:** Understanding parameter-efficient training

```bash
cargo run --example lora_finetune --release -p rusty
```

**Demonstrates:**
- LoRA adapter initialization
- Low-rank weight matrices
- Forward pass with LoRA
- Memory-efficient fine-tuning

---

## Adding New Examples

1. Create a new `.rs` file in this directory
2. Add the example to `rusty/Cargo.toml`:
   ```toml
   [[example]]
   name = "my_example"
   path = "examples/my_example.rs"
   ```
3. Run with: `cargo run --example my_example --release -p rusty`

---

## Troubleshooting

### "No GPU found"

Ensure you're on a supported platform:
- macOS with Apple Silicon (M1/M2/M3)
- Linux/Windows with Vulkan-capable GPU

### "Permission denied on Cargo.lock"

This can happen with sandbox issues. Try:
```bash
rm -f Cargo.lock
cargo build
```
