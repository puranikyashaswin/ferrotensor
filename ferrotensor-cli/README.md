# ferrotensor-cli

The `ferrotensor-cli` command line for [Ferrotensor](../README.md): LoRA fine-tuning
(`train`), evaluation (`eval`) and sampling (`generate`, alias `infer`) for
small Llama-style models such as SmolLM2 and TinyLlama, on any GPU wgpu
supports (Metal, Vulkan, DX12).

Install it from crates.io (the binary is called `ferrotensor`), or run it from
a checkout with `cargo run -p ferrotensor-cli --release --`:

```bash
cargo install ferrotensor-cli
ferrotensor --help
```

```bash
cargo run -p ferrotensor-cli --release -- --help            # commands
cargo run -p ferrotensor-cli --release -- train --help      # every training flag

# LoRA SFT with response-only loss and the ChatML template
cargo run -p ferrotensor-cli --release -- train \
  --model ./data/models/SmolLM2-135M \
  --data ./data/alpaca/train.jsonl --eval-data ./data/alpaca/val.jsonl \
  --out ./data/runs/sft --max-steps 250 --grad-accum 8 --lr 5e-4

# Sample from the fine-tuned adapter (rank/alpha/targets are read from the file)
cargo run -p ferrotensor-cli --release -- generate \
  --model ./data/models/SmolLM2-135M \
  --lora ./data/runs/sft/lora_adapters.safetensors \
  --prompt "Give three tips for staying healthy." --temperature 0.7 --top-p 0.9

# Held-out loss and perplexity
cargo run -p ferrotensor-cli --release -- eval \
  --model ./data/models/SmolLM2-135M --data ./data/alpaca/val.jsonl \
  --lora ./data/runs/sft/lora_adapters.safetensors
```

A full walkthrough with measured numbers is in
[docs/examples/lora-sft.md](../docs/examples/lora-sft.md).

- SFT data: `.jsonl` (one object per line) or a `.json` array of
  `{"prompt": ..., "response": ...}` objects, optional `"system"`.
- Pretraining data (`--task pretrain`): `.txt` lines, a `.json` array of
  strings or token ids, or packed little-endian `u32` tokens in `.bin`.
- Outputs in `--out`: `lora_adapters.safetensors` (+ `.state.json` for
  `--resume`), `metrics.jsonl` (one line per logged step / eval) and
  `training_state.json` (read by `ferrotensor-cli monitor`).
- `ferrotensor-cli demo` (or `--demo`) plays a simulated training UI; it does not
  train anything.

## Where GPU memory goes: `--memory-report`

`train --memory-report` prints a table of GPU memory by category after the
model loads and after each of the first 3 optimizer steps
(`--memory-report=N` for N steps), and writes the same numbers in bytes to
`OUT/memory_report.json`. Without the flag nothing is printed and training
is unchanged.

```bash
cargo run -p ferrotensor-cli --release -- train \
  --model ./data/models/SmolLM2-135M --data ./data/alpaca/train.jsonl \
  --out ./data/runs/mem --max-steps 3 --memory-report
```

Each table has two columns: the bytes at the highest point of that phase
(the model load, or one optimizer step of `--batch-size` x `--grad-accum`
sequences) and at the highest point seen so far in the run. The rows:

| Row | What it holds |
|---|---|
| base weights (frozen) | Model weights uploaded when the model is built, split into f16 (linear layers with the default f16 weights) and f32 (embedding, norms, or every weight with `--f32-weights`) |
| trainable parameters | The LoRA adapters, or the base weights `--train-modules` selects |
| optimizer state | The AdamW moments, two f32 tensors per trainable tensor |
| activations | Forward outputs and intermediates, kept by the autograd graph until the step's backward is done |
| parameter gradients | Gradients of the trainable parameters |
| backward scratch | Everything else created during backward: gradients of activations and kernel temporaries |
| loss (vocab-sized) | The logits, `d loss / d logits` and the logits' gradient |
| readback staging | Buffers that copy results to the host |
| other | Buffers created outside these phases (e.g. gradient-norm partial sums) |
| pool idle | Freed buffers the pool keeps for reuse; still allocated on the GPU |

Below the table: how much of the live bytes the tensors' shapes need
(wgpu rounds every buffer up to 4 bytes), the highest live plus pool-idle
total of the phase, and the 10 largest live buffers at the phase peak with
their category, dtype and shape.

The report counts only buffers Ferrotensor allocated through wgpu, so it is
lower than `nvidia-smi` or Activity Monitor. Not counted: driver and
allocator overhead and alignment, compiled pipelines and shaders, uniform
buffers, the staging copies wgpu makes for uploads (released every 256 MiB
of uploads), and buffers of the `ferrotensor` tensor API. A buffer is
counted under the phase that created it; the logits, the loss gradient and
the parameter gradients are moved to their own rows when the trainer knows
what they are, and the phase peak is corrected for that move.
