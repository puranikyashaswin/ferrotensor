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
| total (live tensors) | The sum of the rows above |
| pool idle (cap 512 MiB) | Freed buffers the pool keeps for reuse; still allocated on the GPU. Once the GPU has finished with them the pool keeps at most the cap shown; while it still runs the work recorded before they were freed, the pool keeps them above the cap so later ops can reuse them (see below) |
| total held (live + pool idle) | The highest live plus pool-idle bytes of the phase, and of the run so far |
| freed, awaiting GPU | Buffers Ferrotensor has dropped but wgpu may not have destroyed yet (see below); the highest of the phase and of the run |
| total incl. awaiting GPU | The highest live plus pool-idle plus awaiting-GPU bytes of the phase and of the run: the closest figure to what the GPU holds |

The rows down to pool idle are taken at the moment the live tensors peaked.
The last three are the highest each value reached, so they can come from
other moments and need not add up with the rows above.

A buffer Ferrotensor drops stays on the GPU until wgpu destroys it, which
it does only after the work using it has run and the device is maintained
(on every batch submission and device poll). Kernels are recorded and
submitted in batches, and the host usually records well ahead of the GPU.
Every tensor the pool does not keep (created from host data, or beyond its
cap with pool overflow off), every per-dispatch uniform, dummy and scratch
buffer, the readback copy after the read, and the temporary buffer of a
quantised weight upload stay on the GPU until the batch recorded when they
were dropped has finished; the report counts them as awaiting GPU until
that is known (at a later submission, or at the latest the next wait for
the GPU).

Pool overflow: dropping a buffer frees nothing until the GPU has finished
the work recorded before the drop, so freed tensor buffers beyond the
pool's cap are kept for reuse until then, and only then dropped (and
destroyed at once). A later op that needs a buffer of the same size takes
one of them instead of allocating a new one. Results are the same either
way. Set `FERROTENSOR_POOL_OVERFLOW=0` to drop buffers beyond the cap as
they are freed instead, e.g. to compare the two with `--memory-report`.

With `--memory-report`, every point where the report samples (after the
model load, before the first step and after each reported step) waits for
the GPU first, so the numbers are taken with no work in flight. This adds
waits that a run without the flag does not have: it can change timing,
never results. Without the flag there is no added wait.

Below the table: how much of the live bytes the tensors' shapes need
(wgpu rounds every buffer up to 4 bytes), the bytes held right after the
sample (live, pool idle, awaiting GPU), and the 10 largest live buffers at
the phase peak with their category, dtype and shape.

The report counts only buffers Ferrotensor allocated through wgpu, so it is
lower than `nvidia-smi` or Activity Monitor. Not counted: the driver's
context and its allocator's blocks, alignment and slack, compiled
pipelines, shaders and bind groups, the staging copies wgpu makes for
uploads (released every 256 MiB of uploads), and buffers of the
`ferrotensor` tensor API. A buffer is
counted under the phase that created it; the logits, the loss gradient and
the parameter gradients are moved to their own rows when the trainer knows
what they are, and the phase peak is corrected for that move.
