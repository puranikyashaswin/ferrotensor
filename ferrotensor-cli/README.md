# ferrotensor-cli

The `ferrotensor-cli` command line for [Ferrotensor](../README.md): LoRA fine-tuning
(`finetune`, `train`), evaluation (`eval`) and sampling (`generate`, alias `infer`) for
small Llama-style models such as SmolLM2 and TinyLlama, on the GPU through
wgpu (Vulkan, Metal or DX12). It has been tested on an NVIDIA A10G (24 GB)
through Vulkan, an Apple M2 (8 GB) through Metal, and software Vulkan (Mesa
lavapipe) in CI; other GPUs are untested, and
[reports are welcome](../docs/testing-on-your-gpu.md).

Install it from crates.io (the binary is called `ferrotensor`), or run it from
a checkout with `cargo run -p ferrotensor-cli --release --`:

```bash
cargo install ferrotensor-cli
ferrotensor --help
```

```bash
cargo run -p ferrotensor-cli --release -- --help            # commands
cargo run -p ferrotensor-cli --release -- train --help      # every training flag

# The short path: LoRA SFT with batch size, sequence length, gradient
# accumulation and checkpointing picked to fit this machine (see below)
cargo run -p ferrotensor-cli --release -- finetune \
  --model ./data/models/SmolLM2-135M \
  --data ./data/alpaca/train.jsonl --out ./data/runs/sft

# The explicit path: LoRA SFT (response-only loss, ChatML template) with
# every setting chosen by hand
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
  `--resume`; with `--train-modules` other than `lora` the file is
  `trained_weights.safetensors` and holds the f32 base weights that
  trained, which `eval --lora`, `generate --lora` and `--resume` load),
  `metrics.jsonl` (one line per logged step / eval) and
  `training_state.json` (read by `ferrotensor-cli monitor`).
- `classify-train` fine-tunes the model with LoRA as a text classifier from
  JSONL `{"text", "label"}` lines and `classify-predict` writes per-line
  class probabilities; see
  [docs/examples/classify.md](../docs/examples/classify.md).
- `ferrotensor-cli demo` (or `--demo`) plays a simulated training UI; it does not
  train anything.

## Scripting: `--json` and exit codes

`plan`, `finetune`, `train`, `eval`, `generate` and `info` take `--json`:
stdout then carries one JSON object when the command ends, and progress and
other text go to stderr (`generate` streams its text there too). On an error
stdout carries `{"error": "<message>"}`. The keys of each object are listed in
[AGENTS.md](../AGENTS.md#json-output).

Every command exits with 0 on success, 1 on a runtime error (I/O, the GPU, a
kernel out of tolerance) and 2 when the input is invalid or this machine
cannot meet the request: a missing model or data file, a flag value out of
range, no GPU adapter, or no configuration that fits. clap's usage errors
also exit with 2.

## Frozen base weights: f16, f32 or 4-bit (experimental)

By default the frozen linear-layer weights (attention, MLP and `lm_head`)
are stored as f16 and every product is computed in f32; `--f32-weights`
keeps them f32. Both compute the same model up to f16 rounding of the
weights.

`--q4-weights` (on `train`, `eval` and `generate`, off by default) is
experimental and changes the model's outputs. It stores
the frozen weights of every decoder-block projection (`q/k/v/o_proj`,
`gate/up/down_proj`) at 4 bits: affine codes in groups of 64 along the
input dimension, one f16 scale and one f16 minimum per group, about 4.5
bits per weight. They are quantised on the host while the model loads,
from the f16/f32/bf16 safetensors, and the matmul decodes them and
computes in f32. `lm_head` stays f16; the embedding, the norms and the
LoRA adapters stay f32.

This is not the same model. It computes with the decoded weights, so its
logits, loss and gradients differ from the f16 or f32 base, and from
PyTorch on the original checkpoint. The tests check that the 4-bit model
equals, to f32 tolerance, the same model loaded in f32 from weights that
were quantised and decoded on the host.

Measured on an NVIDIA A10G (24 GB) with the
[walkthrough recipe](../docs/examples/lora-sft.md) (250 steps, batch 1 × 8-step
accumulation), the held-out loss after training is worse than with f16 base
weights, and peak GPU memory (`nvidia-smi`, with `--checkpoint-blocks`) is lower:

| Model | Held-out loss after 250 steps, f16 | 4-bit | Peak GPU memory, f16 | 4-bit |
|---|---:|---:|---:|---:|
| SmolLM2-135M | 1.6521 | 1.8000 | 1,890 MiB | 1,698 MiB |
| SmolLM2-360M | 1.3228 | 1.4087 | 2,783 MiB | 1,998 MiB |
| SmolLM2-1.7B | 1.0640 | 1.1198 | 5,281 MiB | 3,741 MiB |

These are 250-step runs in separate sessions. In the 30-step confirmation run of
merged `main`, SmolLM2-1.7B with `--checkpoint-blocks` peaked at 5,313 MiB with f16 base
weights and 3,741 MiB with `--q4-weights`, and the held-out loss after 30 steps was
1.6245 with 4-bit weights against 1.4560 with f16 (2.6474 and 3.7470 before
training): 4-bit is still worse. Peaks for the same setting vary between sessions by
up to about 70 MiB.

The format is plain round-to-nearest in groups of 64, not the NF4 format
of QLoRA.

- `--q4-weights` conflicts with `--f32-weights`, and is an error when base
  weights train (`--train-modules` other than `lora`): 4-bit weights are
  never trainable.
- An adapter trained with `--q4-weights` learned to work with the 4-bit
  weights, quantisation error included. Evaluate and sample it with
  `--q4-weights` too. Loading it on the f16 or f32 base works (the shapes
  are the same) but gives a different model from the one that was
  trained, usually close to it; nothing checks or warns about this.
- Checkpoints that are already quantised (MLX-style `<name>.scales`
  tensors) are rejected with an error; load the f16/f32/bf16 checkpoint and
  pass `--q4-weights` instead.
- With `--memory-report`, the 4-bit weights appear under base weights as
  "of which 4-bit".

## Fine-tune in one command: `finetune`

`finetune` takes every `train` flag plus `--memory-mib`, `--vram-mib` and
`--force`. It runs the `plan` below for the model, prints the plan to
stderr, then trains in the same process.

```bash
ferrotensor finetune --model ./data/models/SmolLM2-360M --data train.jsonl
ferrotensor finetune --model ./data/models/SmolLM2-360M --data train.jsonl --seq-len 512
```

- The plan sets `--batch-size`, `--seq-len`, `--checkpoint-blocks` and
  `--grad-accum` (8 sequences per step), each only when it is not given.
  A given `--batch-size`, `--seq-len` or `--checkpoint-blocks` is kept and
  the plan is chosen around it. `--f32-weights`, `--q4-weights` and
  `--lora-rank` are passed to the plan as they are. A stderr line says
  which values came from the plan and which from the command line.
- When no configuration fits, it prints the estimate and the budget that
  would fit, exits with code 2 and does not train. `--force` trains anyway
  with the smallest configuration.
- After training it prints the estimated peak, the measured peak of the
  GPU buffers Ferrotensor tracks (without the driver context), and the
  difference in percent.
- `--lite` is refused; it sets memory flags itself.

## What fits this machine: `plan`

`plan` estimates the peak GPU memory of LoRA training and prints the
largest configuration expected to fit, with the `train` line to run it. It
reads only the model's `config.json` and does not train.

```bash
ferrotensor plan --model ./data/models/SmolLM2-360M
ferrotensor plan --model ./data/models/SmolLM2-360M --memory-mib 3000 --json
ferrotensor plan                 # the documented SmolLM2 sizes and which fit
```

- Budget: `--memory-mib` wins; else `--vram-mib`; else, on a GPU that
  shares system RAM (integrated GPUs, Apple silicon), 60% of the system
  RAM; else, on an NVIDIA GPU, the smallest `memory.total` that
  `nvidia-smi` lists; else an assumed 4,096 MiB, because wgpu does not
  report the memory of a discrete GPU. The output says which (`measured`,
  `assumed`, `user`).
- Choice: the longer sequence first (256, then 128, or only `--seq-len`),
  then no checkpointing before `--checkpoint-blocks`, then the larger batch
  (8, 4, 2, 1), keeping 15% of the budget free. Sequence length comes first
  because a shorter one cuts longer examples, which changes what is
  trained; batch size only changes speed, and the train line keeps 8
  sequences per step. Seq 128 is used only when nothing fits at 256, and a
  note says so. `--weights f16|f32|q4` (default f16) and `--lora-rank`
  (default 8) are fixed by the flags.
- Exit code 0 when a configuration fits, 2 when none does or the input is
  invalid, 1 on a runtime error.
- `--json` prints one object with the keys `adapter`, `backend`,
  `device_type`, `budget_mib`, `budget_source`, `model`, `parameters`,
  `chosen` (`batch_size`, `seq_len`, `rank`, `precision`,
  `checkpoint_blocks`), `estimated_peak_mib`, `margin`, `fits`,
  `train_command` and `notes`. Without `--model` the per-model keys are in
  a `models` array instead.

The peak is an estimate: base weights at their precision, LoRA adapters
with gradients and AdamW state, activations kept for backward (only block
inputs with `--checkpoint-blocks`), the logits and their gradient, and a
fixed 1,350 MiB for the driver context and buffer pool. Its constants are
calibrated on the eight A10G peaks in the table below (SmolLM2 135M, 360M
and 1.7B, f16 weights, sequence length 256); the estimate is within 13.3%
of each of them. Other GPUs, f32 and 4-bit weights, LFM2 models and other
sequence lengths are not calibrated.

## Less GPU memory: `--checkpoint-blocks`

`train --checkpoint-blocks` (off by default) keeps only each decoder
block's input during the forward pass and recomputes the block's forward
during backward, instead of keeping all its activations. It trades time for
memory. Measured on an NVIDIA A10G (24 GB) with SmolLM2, f16 base weights,
sequence length 256 and 30 optimizer steps, all in one session on merged
`main` (peak from `nvidia-smi`, seconds per step as the median over the 30 steps):

| Model | Peak GPU memory, off → on | Seconds per step, off → on |
|---|---:|---:|
| SmolLM2-135M, batch 1 × accum 8 | 2,306 → 1,890 MiB | 1.17 → 1.47 (+26%) |
| SmolLM2-360M, batch 1 × accum 8 | 3,579 → 2,783 MiB | 1.56 → 2.04 (+31%) |
| SmolLM2-1.7B, batch 1 × accum 8 | 8,142 → 5,313 MiB | 2.275 → 3.155 (+39%) |
| SmolLM2-135M, batch 8 × accum 8 | 9,441 → 2,767 MiB | 2.645 → 3.135 (+19%) |

The training loss and gradient norm, as printed at every step, and the
held-out loss were identical with the flag on and off. The unit tests in
`ferrotensor-trainer/tests/block_checkpointing.rs` compare the loss (to a
relative 1e-5) and the gradients (to 1e-6 plus a relative 1e-4) with and
without checkpointing.

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
| base weights (frozen) | Model weights uploaded when the model is built, split into f16 (linear layers with the default f16 weights), 4-bit (decoder-block projections with `--q4-weights`) and f32 (embedding, norms, or every weight with `--f32-weights`) |
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

## Where step time goes: `--profile`

`train --profile` counts the GPU work of every optimizer step. After the
run it prints the step with the median wall time and writes the same
numbers to `OUT/profile.json` (times in seconds, counts as
`<name>_per_step`). Counting adds a few atomic adds per op and no waits for
the GPU, so it does not change results. `--memory-report` waits for the
GPU in the first steps it reports, so leave it off when profiling.

```text
profile: median of <steps> optimizer steps by wall time
  step wall time             <ms> ms
  waiting on the GPU         <ms> ms
  in readbacks               <ms> ms
  dispatches                 <count> /step
  bind groups                <count> /step
  uniform buffers            <count> /step
  uniform bytes              <bytes> /step
  submits                    <count> /step
  synchronizes               <count> /step
  forced synchronizes        <count> /step
  upload bytes               <bytes> /step
  readbacks                  <count> /step
  readback bytes             <bytes> /step
```

The values in angle brackets are placeholders. Step time minus the time
waiting on the GPU is roughly the time the host spends recording work;
waiting includes the wait inside each readback, and forced synchronizes
are the waits that free upload staging memory every 256 MiB of uploads.
