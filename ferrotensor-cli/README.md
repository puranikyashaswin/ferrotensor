# ferrotensor-cli

The `ferrotensor-cli` command line for [Ferrotensor](../README.md): a GPU
check (`info`), memory planning (`plan`), LoRA fine-tuning (`finetune`, `train`),
evaluation (`eval`), sampling (`generate`, alias `infer`), conversation (`chat`),
adapter export and merging (`export-adapter`, `merge`), GGUF export for
llama.cpp and Ollama (`export-gguf`) and training a byte-level
model from scratch (`pretrain`), for
small Llama-style models such as SmolLM2 and TinyLlama, on the GPU through
wgpu (Vulkan, Metal or DX12). It also has `classify-train` and
`classify-predict` (text classification), `monitor`, `demo` and `benchmark`.
It has been tested on an NVIDIA A10G (24 GB)
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

# Check the GPU and the kernels on it
cargo run -p ferrotensor-cli --release -- info --check

# What fits this machine, without training
cargo run -p ferrotensor-cli --release -- plan --model ./data/models/SmolLM2-135M

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

# Talk to the fine-tuned model, one message per line (see below)
cargo run -p ferrotensor-cli --release -- chat \
  --model ./data/models/SmolLM2-135M \
  --lora ./data/runs/sft/lora_adapters.safetensors

# Held-out loss and perplexity
cargo run -p ferrotensor-cli --release -- eval \
  --model ./data/models/SmolLM2-135M --data ./data/alpaca/val.jsonl \
  --lora ./data/runs/sft/lora_adapters.safetensors

# A standalone model directory with the adapter folded in (see below)
cargo run -p ferrotensor-cli --release -- merge \
  --model ./data/models/SmolLM2-135M \
  --lora ./data/runs/sft/lora_adapters.safetensors \
  --out ./data/models/SmolLM2-135M-sft

# Or a PEFT adapter directory for PyTorch
cargo run -p ferrotensor-cli --release -- export-adapter \
  --model ./data/models/SmolLM2-135M \
  --lora ./data/runs/sft/lora_adapters.safetensors \
  --output ./data/runs/sft/peft

# Train a tiny byte-level model from random weights (see below)
cargo run -p ferrotensor-cli --release -- pretrain \
  --data corpus.txt --out ./data/models/tiny
```

A full walkthrough with measured numbers is in
[docs/examples/lora-sft.md](../docs/examples/lora-sft.md).

## Model directory or Hugging Face id

Every command's `--model` takes a model directory or a Hugging Face model id
such as `HuggingFaceTB/SmolLM2-135M` that is already in the local cache:

```bash
hf download HuggingFaceTB/SmolLM2-135M
ferrotensor chat --model HuggingFaceTB/SmolLM2-135M
```

- An existing directory always wins, so `--model org/name` uses `./org/name`
  when it exists.
- Otherwise an id (`org/name`, letters, digits, `-`, `_` and `.`) is looked up
  in the Hugging Face cache, and stderr shows the snapshot it chose once:
  `model: org/name (<snapshot path>)`. The cache is `HF_HUB_CACHE`, else
  `$HF_HOME/hub`, else `$XDG_CACHE_HOME/huggingface/hub`, else
  `~/.cache/huggingface/hub`. The snapshot is the one `refs/main` points to,
  or the smallest snapshot name when there is no `refs/main`. A snapshot
  without `config.json` exits with 2 and says the download looks incomplete.
- An id that is not cached exits with 2 and names the cache directory and the
  fetch command, `hf download org/name` (`huggingface-cli download org/name`
  on older installs). Ferrotensor never runs it and never downloads.
- After a LoRA `train` or `finetune`, stderr ends with a `next:` block: the
  `chat` and `merge` commands for this run, with `--model` as given.

- SFT data: `.jsonl` (one object per line) or a `.json` array of
  `{"prompt": ..., "response": ...}` objects, optional `"system"`.
- Pretraining data (`--task pretrain`): `.txt` lines, a `.json` array of
  strings or token ids, or packed little-endian `u32` tokens in `.bin`.
- Outputs in `--out`: `lora_adapters.safetensors` (+ `.state.json` and
  `.state.safetensors` for `--resume`; with `--train-modules` other than `lora` the file is
  `trained_weights.safetensors` and holds the f32 base weights that
  trained, which `eval --lora`, `generate --lora` and `--resume` load),
  `metrics.jsonl` (one line per logged step / eval) and
  `training_state.json` (read by `ferrotensor monitor`).
- `--train-modules` picks what trains: `lora` (the default), or base weights
  by group: `attn`, `mlp`, `norm`, `head`, `all` (every base weight but the
  token embedding), `embed` (only the token embedding) and `full` (`all`
  plus the embedding). A trained embedding costs a gradient and two AdamW
  moments of `vocab x hidden` f32 values each on the GPU, and the table is
  read back to the host for every forward pass while it trains. With tied
  embeddings `lm_head` is a separate copy, so training the embedding does
  not change `lm_head` and training `head` does not change the embedding.
- `classify-train` fine-tunes the model with LoRA as a text classifier from
  JSONL `{"text", "label"}` lines and `classify-predict` writes per-line
  class probabilities; see
  [docs/examples/classify.md](../docs/examples/classify.md).
- `ferrotensor demo` (or `--demo`) plays a simulated training UI; it does not
  train anything. `ferrotensor monitor` (or `--monitor`) is the terminal
  dashboard, and `ferrotensor benchmark` prints how to run the kernel
  benchmarks.

## Scripting: `--json` and exit codes

`plan`, `finetune`, `train`, `eval`, `generate`, `chat`, `merge`,
`export-gguf`, `pretrain` and `info` take `--json`: stdout then carries one JSON object when the
command ends, and progress and other text go to stderr (`generate` and `chat`
stream their text there too). On an error stdout carries
`{"error": "<message>"}`. `chat --json` is the exception: it prints one JSON
object per reply, each on its own line. `classify-train`, `classify-predict`,
`export-adapter`, `monitor`, `demo` and `benchmark` take no `--json`. The keys
of each object are listed in [AGENTS.md](../AGENTS.md#json-output).

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

`--q4-weights` (on every command that loads a model, off by default) is
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

## A standalone model: `merge`

`merge` folds a LoRA adapter into the base weights and writes a model
directory that loads without `--lora`, here and in Hugging Face
Transformers:

```bash
cargo run -p ferrotensor-cli --release -- merge \
  --model ./data/models/SmolLM2-135M \
  --lora ./data/runs/sft/lora_adapters.safetensors \
  --out ./data/models/SmolLM2-135M-sft
```

Every adapted weight becomes `W + alpha / rank * (A B)^T`, computed in f32
and stored in the base's dtype (F32, F16 or BF16); every other tensor is
copied byte for byte into one `model.safetensors`, also when the base is
sharded. `config.json`, `generation_config.json` and the tokenizer files are
copied unchanged. An adapter on `lm_head` of a model with tied embeddings
(SmolLM2) writes a separate `lm_head.weight` and sets
`tie_word_embeddings` to false, since the embeddings stay as they were.

The merge runs on the host without a GPU and streams: one tensor is read,
changed row by row and written before the next, so host memory stays near
one row plus the adapter instead of the model. `--lora` takes a Ferrotensor
adapter file or a PEFT adapter directory. `--out` must be new or empty.
`--json` prints `{"out_dir", "merged", "copied"}` (tensor counts).
Quantized bases (`quantization` or `quantization_config` in `config.json`,
or `.scales` tensors), files of trained base weights, and adapter tensors
whose base weight is missing or has another shape are refused before
anything is written.

## llama.cpp and Ollama: `export-gguf`

`export-gguf` writes a Llama-style model directory, such as a `merge`
output, as one GGUF version 3 file:

```bash
cargo run -p ferrotensor-cli --release -- export-gguf \
  --model ./data/models/SmolLM2-135M-sft --out ./data/SmolLM2-135M-sft.gguf
```

2-D weights are stored in f16 and 1-D tensors and norms in f32, as
llama.cpp's converter does; `--f32` stores every tensor in f32. The tensor
names, the dimension order, the reordered query and key rows and the
metadata keys follow llama.cpp's `convert_hf_to_gguf.py`. `--lora` adds an
adapter while writing, with the same maths as `merge` (the sum is rounded
once, to the file's type, not first to the base's). The export runs on the host and streams
one row at a time, like `merge`. `--json` prints
`{"out_file", "tensors", "merged"}`.

The tokenizer must be byte-level BPE with the pre-tokenizer of GPT-2 or of
SmolLM (llama.cpp `gpt-2` and `smollm`); SentencePiece-style tokenizers such
as TinyLlama's are refused. An existing `--out`, a `model_type` other than
`llama`, a scaled `rope_scaling` or `rope_parameters` (Llama 3.1 and
later), projection biases (Qwen2), a quantized base and tensors without a
llama.cpp name are refused too, with exit code 2. The file is written to the GGUF v3 specification;
loading in llama.cpp is checked on the next hardware run, with
`scripts/check_gguf.sh`, which compares a greedy completion of llama.cpp's
`llama-simple` with `ferrotensor generate`.

## Train from scratch: `pretrain`

`pretrain` trains a small byte-level Llama-style model from random weights
on a text file and leaves a model directory that `generate --raw` loads. It
is meant for learning and experiments: f32 weights, untied embeddings,
training defaults that are conservative and not tuned, and no claim about
speed or about quality beyond the CI test below.

```bash
cargo run -p ferrotensor-cli --release -- pretrain \
  --data corpus.txt --out ./data/models/tiny
cargo run -p ferrotensor-cli --release -- generate \
  --model ./data/models/tiny --raw --prompt "Once upon a time"
```

In order, it

1. writes a byte-level `tokenizer.json`: `<pad>` 0, `<s>` 1, `</s>` 2, then
   byte `b` as token `b + 3` (259 tokens), a BPE model without merges over
   the `ByteLevel` characters. Any UTF-8 text encodes to its bytes and
   decodes back exactly; `<s>` in the text stays three bytes;
2. writes `config.json` and, in `init/`, the initial model in the Hugging
   Face Llama layout from `--layers` (default 4), `--hidden` (128), `--heads` (4),
   `--mlp-hidden` (4 x hidden) and `--context` (256, written as
   `max_position_embeddings`): norm scales are ones, every other weight is
   drawn from a normal distribution with standard deviation 0.02 by a
   generator seeded with `--seed`, so a seed always gives the same model.
   The head size `--hidden / --heads` must be even and at most 128. The
   defaults give 1,116,032 parameters;
3. writes the bytes of `--data` as `tokens.bin` (little-endian u32, newlines
   kept, the file read once);
4. runs `train --task pretrain --train-modules full` from `init/`:
   every weight and the embedding train in f32 on windows of `--seq-len`
   tokens (default `--context`), with `--lr 1e-3`, `--warmup-steps 100`,
   `--max-steps 2000`, `--batch-size 8`, `--grad-accum 1` and
   `--save-every 500` unless given;
5. writes the trained weights as `model.safetensors`, linears transposed
   back to `[out, in]`. `trained_weights.safetensors` and the training state
   stay in `--out` for `--resume`.

`--resume` continues a run in the same `--out` from its
`trained_weights.safetensors` and training state; pass the flags of the
first run, with `--max-steps` counting the steps before the resume. Without
`--resume`, an `--out` that already holds `model.safetensors` or `init/` is
refused. A run that stops early (Ctrl-C, a crash) leaves no
`model.safetensors`, so `generate` refuses the directory and names
`pretrain --resume`; a resume with `--max-steps` at or below the step in
`training_state.json` trains nothing and exports the weights saved at that
step.
`--json` prints every `train` key plus `parameters` and `vocab_size`.
`--device`, `--checkpoint-blocks`, `--grad-clip`, `--weight-decay` and
`--log-every` work as in `train`, and `--seed` also seeds the initial weights;
`pretrain --help` lists every flag and default.

Host memory while training is about 20 bytes per byte of the corpus (the
token windows, held twice). GPU memory is the f32 weights, their gradients
and two AdamW moments (16 bytes per parameter) plus activations; `plan` does
not cover `pretrain`. Not included: a learned tokenizer, tied embeddings,
mixed precision, and document boundaries (the corpus is one stream).

The CI test trains a tiny model (2 layers, hidden 64, 4 heads, context 64,
164,544 parameters) on one sentence repeated 100 times, on software Vulkan
(llvmpipe): with `--lr 3e-3`, `--warmup-steps 20` and 150 steps the logged
training loss fell from 3.91 at step 25 to 0.0114 at step 150 (uniform
guessing is ln 259 = 5.56), and greedy `generate --raw` continues "the quick
brown" with " fox jumps". This shows the path learns; it says nothing about
quality on real text. The exported directory gives the same eval loss as
the trained weights it came from, and loads in Transformers with the same
loss within 1e-3.

## Sample from a model: `generate`

`generate` stops with exit code 2 when the prompt encodes to no tokens or is
longer than the model's `max_position_embeddings`. When the prompt fits but
the prompt plus `--max-new-tokens` does not, it lowers the number of new
tokens to what fits and says so on stderr. A model without a
`max_position_embeddings` entry is not checked.

## Talk to a model: `chat`

`chat` holds a multi-turn conversation with a base model or, with `--lora`,
a fine-tuned one. It reads one user message per line of stdin and streams
each reply as it is generated:

```bash
cargo run -p ferrotensor-cli --release -- chat \
  --model ./data/models/SmolLM2-135M \
  --lora ./data/runs/sft/lora_adapters.safetensors \
  --system "Answer in one sentence."
```

`/reset` clears the conversation, and `/exit`, `/quit` or the end of input
ends it. After each reply stderr gets the same line `generate` prints (prompt
tokens, new tokens, tokens per second). `--temperature`, `--top-k`,
`--top-p`, `--seed` and `--max-new-tokens` work as in `generate`.

Each turn encodes the whole conversation in ChatML: the system message,
then the user and assistant turns, then the assistant header. The KV cache
is kept across turns, so each turn processes only the new tokens: the new
message in the template, then the reply. When the conversation and
`--max-new-tokens` no longer fit the model's `max_position_embeddings`
(from `config.json`), the oldest exchanges are dropped, the system message
is kept, and stderr says so once; the cache then keeps only the start the
trimmed conversation shares with it and runs the rest. A single message
that does not fit alone is an error (exit code 2). A model without a
`max_position_embeddings` entry is not trimmed.

Only models whose tokenizer has the ChatML tokens `<|im_start|>` and
`<|im_end|>` are supported for now (SmolLM2 has them); for other models
`chat` exits with 2, and `generate --raw` takes a prompt verbatim.

Input can be piped, which is how scripts and the tests drive it. With
`--json` each reply is one JSON object per line on stdout,
`{"reply", "prompt_tokens", "cached_tokens", "new_tokens", "tokens_per_second"}`
(`cached_tokens`: prompt tokens taken from the cache, not run again), and the
streamed text goes to stderr.
