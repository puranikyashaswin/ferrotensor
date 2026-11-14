# LoRA SFT: SmolLM2 on Alpaca

This walkthrough fine-tunes a real Hugging Face checkpoint with LoRA,
evaluates it on held-out data, saves the adapter, reloads it and generates
text. Every number below was measured with the exact commands shown. The
first run is SmolLM2-135M on an Apple M2 (8 GB) with a 0.4 build; the
sections after it are on an NVIDIA A10G (24 GB), each labelled with the
build it was measured on.

| | |
|---|---|
| Machine | MacBook Air, Apple M2, 8 GB unified memory, macOS 27.2 |
| GPU backend | Metal through wgpu |
| Toolchain | rustc 1.99.1, release build of Ferrotensor 0.4 |
| Model | [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) (base model, Llama architecture, 30 layers, hidden 576, 9 query / 3 KV heads, vocab 49152, tied embeddings) |
| Data | [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned): 2,000 training and 64 held-out examples, as prepared below |
| Adapter | LoRA rank 8, alpha 16, on `q_proj`, `k_proj`, `v_proj`, `o_proj` of every layer (921,600 trainable parameters, 240 tensors) |

## Result

Measured on 0.4; later builds moved more of the backward pass and the loss
to the GPU, so the speed and memory figures in this section are not
current.

| | Before | After 250 steps |
|---|---|---|
| Held-out loss (64 examples, 2,744 response tokens) | 2.4250 | **1.6519** |
| Held-out perplexity | 11.30 | **5.22** |

- Wall time: 2,195.5 s for 250 optimizer steps (2,000 sequences, one
  epoch), including four evaluations: **8.8 s per step**, about 1.1 s per
  sequence.
- Peak memory footprint (`/usr/bin/time -l`): **3.29 GB** for training,
  2.24 GB for evaluation.
- Generation with the KV cache: about 13 tokens/s for the base model and
  8 to 9 tokens/s with the adapter applied (LoRA adds two small matmuls per
  projection).
- The adapter is 3.7 MB. Reloading it with `eval --lora` reproduces the
  held-out loss exactly (1.6519).

The base model has never seen the ChatML template, so before tuning it
echoes the prompt or loops; after tuning it answers and stops at
`<|im_end|>` on its own (greedy decoding, `--temperature 0`):

| Prompt | Base model | With the adapter |
|---|---|---|
| Give three tips for staying healthy. | `Give three tips for staying healthy.` then `Assistant` repeated until the 60-token limit | 1. Eat a balanced diet that includes a variety of fruits, vegetables, lean proteins, and whole grains. 2. Stay physically active by exercising for at least 30 minutes a day, five days a week. 3. Limit your intake of processed foods, sugary drinks, and unhealthy … (60-token limit) |
| Translate 'good morning' into French. | `Translate 'good morning' into German.` / `… Italian.` / `… Portuguese.` … (60-token limit) | `Bonjour, bonne journe!` (stops by itself after 8 tokens) |

A 135M-parameter model stays a 135M-parameter model: answers are fluent
but shallow, and sampled output can wander (for "Write a haiku about the
ocean." with `--temperature 0.7 --top-k 50 --top-p 0.9 --seed 1` it wrote
repetitive free verse, not a haiku).

### Loss curve

Training loss is the mean over the supervised (response) tokens of the
last 10 steps (80 sequences). It is noisy because each window is only
about 2,300 to 3,000 tokens; the held-out loss is the cleaner signal.

| Step | Train loss | Held-out loss | LR | Grad norm |
|---:|---:|---:|---:|---:|
| 0 | | 2.4250 | | |
| 10 | 2.6203 | | 2.50e-4 | 0.83 |
| 20 | 2.2851 | | 5.00e-4 | 2.04 |
| 30 | 2.1590 | | 4.98e-4 | 1.63 |
| 40 | 2.0081 | | 4.92e-4 | 1.09 |
| 50 | 2.0937 | 1.8145 | 4.81e-4 | 1.92 |
| 60 | 2.0069 | | 4.67e-4 | 1.18 |
| 70 | 1.7144 | | 4.50e-4 | 1.09 |
| 80 | 1.9706 | | 4.29e-4 | 1.13 |
| 90 | 1.9172 | | 4.05e-4 | 0.91 |
| 100 | 1.7948 | 1.6979 | 3.79e-4 | 1.30 |
| 110 | 1.6495 | | 3.50e-4 | 2.83 |
| 120 | 1.6975 | | 3.21e-4 | 0.97 |
| 130 | 1.7112 | | 2.90e-4 | 1.00 |
| 140 | 1.7910 | | 2.60e-4 | 1.44 |
| 150 | 1.9821 | 1.6702 | 2.29e-4 | 0.81 |
| 160 | 1.6392 | | 2.00e-4 | 0.69 |
| 170 | 1.7333 | | 1.71e-4 | 1.41 |
| 180 | 1.7810 | | 1.45e-4 | 1.12 |
| 190 | 1.7577 | | 1.21e-4 | 3.28 |
| 200 | 1.7130 | 1.6579 | 1.00e-4 | 2.04 |
| 210 | 1.7690 | | 8.28e-5 | 0.69 |
| 220 | 1.7831 | | 6.86e-5 | 1.16 |
| 230 | 1.8512 | | 5.83e-5 | 0.67 |
| 240 | 1.8602 | | 5.21e-5 | 0.83 |
| 250 | 1.6317 | 1.6519 | 5.00e-5 | 0.91 |

(Grad norm is the global norm of the last optimizer step in each window,
before clipping to 1.0.)

## On an NVIDIA GPU: three model sizes

`scripts/lora-run.sh MODEL` repeats this walkthrough (same data split, LoRA
rank 8 on q/k/v/o, lr 5e-4, 20 warmup steps, 250 steps of 8 sequences,
sequence length 256) on any Llama-style checkpoint. Run on an NVIDIA A10G
(24 GB, Vulkan, driver 595.112.07, rustc 1.99.1) at commit `1b6709f`:

| Model | Layers, hidden | Trainable params | Held-out loss before → after | Perplexity before → after | Time per step | Peak host memory | Adapter |
|---|---|---:|---|---|---:|---:|---:|
| [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | 30, 576 | 921,600 | 2.4250 → **1.6521** | 11.30 → **5.22** | 1.33 s | 2.06 GB | 3.7 MB |
| [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) | 32, 960 | 1,638,400 | 2.0852 → **1.3228** | 8.05 → **3.75** | 1.69 s | 4.75 GB | 6.6 MB |
| [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) | 24, 2048 | 3,145,728 | 2.6474 → **1.0667** | 14.12 → **2.91** | 2.47 s | 20.01 GB | 12.6 MB |

- Time per step is the median over the run, for one optimizer step of 8
  sequences; 250 steps of the 1.7B model take about 10 minutes.
- The 135M result matches the M2 run above: the same held-out loss before
  tuning (2.4250) and 1.6521 after, against 1.6519. The last digits differ
  because the loss and the gradient norm are now summed on the GPU. An
  earlier build that computed them on the CPU gave 1.6519 on the A10G too,
  at 20.2 s per step.
- With the adapter, all three answer instead of echoing the prompt.
  Greedy outputs for "Give three tips for staying healthy.": 135M as in the
  table above; 360M `1. Eat a balanced diet. 2. Get enough sleep. 3. Exercise
  regularly.`; 1.7B `1. Eat a balanced diet that includes fruits,
  vegetables, lean proteins, and whole grains. 2. Exercise regularly to
  maintain a healthy weight …`.
- Greedy generation with the KV cache runs at 35.9, 21.8 and 10.1 tokens/s
  for the base models.

### Batching, f16 base weights and lower host memory (0.6)

The same recipe on the same A10G at commit `b319ce7`, with the changes in 0.6:
uploads that no longer keep staging copies, `--batch-size`, and frozen base
weights stored as f16 (`--f16-weights`, the default since 0.6):

| Model | Run | Held-out loss after | Time per step | GPU memory (GiB) | Peak host memory |
|---|---|---|---:|---:|---:|
| SmolLM2-135M | f32 weights, `--batch-size 1 --grad-accum 8` | **1.6521** | 1.28 s | 3.8 | 1.35 GB |
| SmolLM2-135M | f32 weights, `--batch-size 8 --grad-accum 1` | **1.6521** | **0.57 s** | 16.3 | 1.30 GB |
| SmolLM2-1.7B | f32 weights, batch 1 | **1.0657** | 2.49 s | 14.7 | 5.57 GB |
| SmolLM2-1.7B | f16 weights, batch 1 | **1.0640** | **2.38 s** | **11.9** | **5.33 GB** |

- Batching eight sequences into one GPU pass trains SmolLM2-135M 2.2x
  faster per optimizer step with the identical held-out loss.
- f16 base weights cut SmolLM2-1.7B's GPU memory by 2.8 GiB and run slightly
  faster; the held-out loss moves by 0.0017.
- Peak host memory for SmolLM2-1.7B dropped from 20.01 GB to 5.33 GB.
- GPU memory is the peak reported by `nvidia-smi` during training
  (3,861, 16,708, 15,028 and 12,199 MiB).
- On this 0.6 build, SmolLM2-1.7B with `--batch-size 8` did not fit on the
  24 GB A10G (22.5 GiB usable): it stopped at 22,590 MiB. This was not
  re-measured after the memory work in 0.7 ([below](#less-gpu-memory-07)).
- The GPU memory figures in this table are from 0.6; for 0.7 see
  [Less GPU memory (0.7)](#less-gpu-memory-07).

### Against PyTorch + PEFT on CUDA, same GPU

[`benchmarks/baselines/pytorch_lora.py`](../../benchmarks/baselines/pytorch_lora.py)
runs the same recipe in PyTorch 2.15 + PEFT (CUDA 13.2): the same split,
ChatML formatting, response-only loss, LoRA initialization, optimizer,
schedule and clipping. Both ran on the same NVIDIA A10G, Ferrotensor with
the 0.6 build above (before the memory work in 0.7), one seed per run; GPU
memory is the `nvidia-smi` peak for both.

| Model | Run | Held-out loss | Time per step | GPU memory (MiB) |
|---|---|---|---:|---:|
| SmolLM2-135M | Ferrotensor, batch 1 | 2.4250 → **1.6521** | 1.28 s | 3,861 |
| | PyTorch, batch 1 | 2.4250 → **1.6494** | 1.08 s | 1,541 |
| | Ferrotensor, batch 8 | 2.4250 → **1.6521** | 0.57 s | 16,708 |
| | PyTorch, batch 8 | 2.4250 → **1.6493** | 0.19 s | 6,843 |
| SmolLM2-1.7B | Ferrotensor, f16 base weights, batch 1 | 2.6474 → **1.0640** | 2.38 s | 12,199 |
| | Ferrotensor, f32 base weights, batch 1 | 2.6474 → **1.0657** | 2.49 s | 15,028 |
| | PyTorch, f32, batch 1 | 2.6474 → **1.0608** | 0.93 s | 8,175 |
| | PyTorch, f32, batch 8 | 2.6474 → **1.0596** | 0.52 s | 21,103 |

- The held-out loss before training is identical to four decimals for both
  models, so the forward passes and the loss agree.
- After 250 steps Ferrotensor is within 0.005 of PyTorch: 0.0027 (135M,
  batch 1), 0.0028 (135M, batch 8), 0.0049 for 1.7B with f32 base weights
  (the same weights as PyTorch) and 0.0032 with f16. Each framework shuffles the data
  with its own generator, so this number can only match approximately;
  PyTorch's own batch-1 and batch-8 runs differ by 0.0012 on 1.7B.
- Speed: Ferrotensor is slower. It runs at 0.84× of PyTorch's speed for
  SmolLM2-135M at batch 1, 0.32× at batch 8, and 0.37× for SmolLM2-1.7B
  (f32 base weights), so it takes 1.2 to 3.1 times PyTorch's time per step.
  I think most of the gap is in the large matrix multiplications, where
  PyTorch uses cuBLAS and Ferrotensor one portable WGSL kernel; I have not
  profiled it kernel by kernel.
- PyTorch used less GPU memory in every configuration and fit SmolLM2-1.7B
  at batch 8, which this 0.6 build did not on a 24 GB card. There is no
  matched comparison with PyTorch after the memory work in 0.7.

## Less GPU memory (0.7)

0.7 lowers the GPU memory of LoRA training in three steps: backward hands
each intermediate gradient back to the buffer pool once it is used (on by
default), freed buffers above the pool's cap stay reusable until the GPU has
finished the work that freed them (on by default), and `--checkpoint-blocks`
recomputes each decoder block's forward pass during backward instead of
keeping its activations (opt-in). Peak GPU memory (`nvidia-smi`, sampled
every 200 ms, driver context included) on the same A10G with f16 base
weights, sequence length 256, 8-step gradient accumulation and 30 optimizer
steps; each column includes the ones before it:

| Model | 0.6 | + release gradients | + reuse freed buffers (0.7 default) | + `--checkpoint-blocks` |
|---|---:|---:|---:|---:|
| SmolLM2-135M, batch 1 | 3,451 MiB | 2,635 MiB | 2,362 MiB | 1,890 MiB |
| SmolLM2-360M, batch 1 | 5,411 MiB | 4,259 MiB | 3,595 MiB | 2,719 MiB |
| SmolLM2-1.7B, batch 1 | 11,607 MiB | 9,438 MiB | 8,222 MiB | 5,245 MiB |
| SmolLM2-135M, batch 8 | 15,758 MiB | 10,669 MiB | 9,401 MiB | 2,879 MiB |

<!-- Maintainer: this 0.6 column was measured at 364490b with 30 steps; the 0.6 table above (b319ce7, 250 steps) has 12,199 MiB for SmolLM2-1.7B with f16 weights. The difference is not explained yet. -->

- Losses are identical at every stage: the training loss and gradient norm
  as printed at all 30 steps, and the held-out loss after them (1.9882,
  1.6168, 1.4561 and 1.9030 for the four rows).
- `--checkpoint-blocks` costs time: 19% (135M, batch 8) to 39% (1.7B) more
  per step ([per model](../../ferrotensor-cli/README.md#less-gpu-memory---checkpoint-blocks)).
  Releasing gradients and reusing buffers did not slow the 1.7B run (2.375,
  2.31 and 2.275 s per step).
- The same setting measured in separate sessions differed by up to 192 MiB.
- `train --memory-report` shows where the memory goes. For SmolLM2-1.7B,
  the frozen base weights hold 3,648 MiB throughout; the adapters,
  their gradients and the AdamW state together stay under 50 MiB.
  [What each row means](../../ferrotensor-cli/README.md#where-gpu-memory-goes---memory-report).

## Correctness checks

Besides the full PyTorch run above, the LoRA training path is checked
against PyTorch autograd by
[`ferrotensor-trainer/tests/gradient_parity.rs`](../../ferrotensor-trainer/tests/gradient_parity.rs).
[`scripts/gen_gradient_fixtures.py`](../../scripts/gen_gradient_fixtures.py)
builds a tiny Llama-style model (2 layers, hidden 32, 4 query and 2 KV
heads) in plain PyTorch with LoRA (rank 4) on every linear layer, and stores
its logits, loss and the gradient of every trainable tensor. The test runs
the same batches on the GPU and compares every element within
1e-4 + 1e-3 × |expected|:

| Case | Gradient tensors | Loss difference | Largest logit difference | Largest gradient difference |
|---|---:|---:|---:|---:|
| One sequence | 30 | 4.77e-7 | 1.55e-6 | 5.66e-7 |
| Batch | 30 | 9.54e-7 | 1.31e-6 | 9.54e-7 |
| Loss mask | 30 | 4.77e-7 | 1.31e-6 | 9.24e-7 |
| Trainable base weights | 50 | 4.77e-7 | 1.31e-6 | 9.24e-7 |

Those differences were measured on the A10G. The test also runs on every
pull request on software Vulkan:

```bash
python3 scripts/gen_gradient_fixtures.py   # optional: the fixtures are committed
cargo test --release -p ferrotensor-trainer --test gradient_parity -- --ignored --test-threads=1
```

The fixture model is tiny, so this does not catch effects that only appear
at scale, such as accumulation error in long reductions; at SmolLM2 scale
the evidence is the held-out loss against PyTorch above.

## Reproduce it

`scripts/lora-run.sh HuggingFaceTB/SmolLM2-135M` runs steps 1 to 4 below with
the same settings and writes `data/runs/SmolLM2-135M/summary.json` (held-out
loss before and after, loss curve, seconds per step, peak memory, greedy
generations with and without the adapter). Pass another Llama-style model to
measure it the same way. The steps it runs:

### 1. Download the model and the data

Everything goes under `data/`, which is gitignored.

```bash
hf download HuggingFaceTB/SmolLM2-135M \
  model.safetensors config.json tokenizer.json tokenizer_config.json \
  --local-dir data/models/SmolLM2-135M
hf download yahma/alpaca-cleaned alpaca_data_cleaned.json \
  --repo-type dataset --local-dir data/alpaca-cleaned
```

### 2. Convert Alpaca to `{prompt, response}` JSONL

Keep short examples (prompt plus response at most 500 characters), shuffle
with seed 0, and use the first 64 for evaluation and the next 2,000 for
training:

```bash
mkdir -p data/alpaca
python3 - <<'EOF'
import json, random
rows = json.load(open("data/alpaca-cleaned/alpaca_data_cleaned.json"))
out = []
for r in rows:
    p = r["instruction"].strip()
    if r.get("input", "").strip():
        p += "\n\n" + r["input"].strip()
    resp = r["output"].strip()
    if len(p) + len(resp) <= 500 and resp:
        out.append({"prompt": p, "response": resp})
random.Random(0).shuffle(out)
for name, rs in (("val", out[:64]), ("train", out[64:2064])):
    with open(f"data/alpaca/{name}.jsonl", "w") as f:
        for x in rs:
            f.write(json.dumps(x) + "\n")
EOF
```

Each prompt is wrapped in the ChatML template
(`<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n`). The response
is followed by `<|im_end|>`, and only response tokens count toward the
loss.

### 3. Train

```bash
cargo build --release -p ferrotensor-cli
/usr/bin/time -l ./target/release/ferrotensor train \
  --model data/models/SmolLM2-135M \
  --data data/alpaca/train.jsonl --eval-data data/alpaca/val.jsonl \
  --out data/runs/sft --seq-len 256 \
  --lr 5e-4 --warmup-steps 20 --grad-accum 8 --max-steps 250 \
  --log-every 10 --eval-every 50 --save-every 50 \
  --sample-prompt "Give three tips for staying healthy."
```

Defaults that apply here:
- batch size 1, so 8 sequences per optimizer step;
- LoRA rank 8 and alpha 16 on q/k/v/o;
- gradient clipping at a global norm of 1.0;
- AdamW without weight decay;
- warmup, then cosine decay to lr/10.

The run writes the following to `data/runs/sft/`:
- `lora_adapters.safetensors`, the adapter with its rank, alpha and
  targets in the file metadata;
- `lora_adapters.safetensors.state.json` and `.state.safetensors`, the
  step and the AdamW moments for `--resume`. The moments are F32, 8 bytes
  per trainable parameter: 7.4 MB here. (Older versions stored them in
  the JSON, which took 50 MB; those files still resume.)
- `metrics.jsonl`, one line per logged step and evaluation;
- `training_state.json`.

The first lines of output:

```text
device: Apple M2 (Metal)
model:  data/models/SmolLM2-135M (30 layers, hidden 576, vocab 49152)
data:   2000 examples (sft task), 66890 supervised tokens, seq-len 256
train:  lora (921600 trainable params), lr 0.0005 warmup 20 cosine, 250 steps x 8 sequences
eval before: loss 2.4250 ppl 11.302 (2744 tokens)
  step |   loss | grad norm |       lr | tok/s | s/step
    10 | 2.6203 |     0.834 |  2.50e-4 |    26 |   8.72
```

`tok/s` counts supervised (response) tokens only. Prompt tokens are also
processed but not counted.

### 4. Reload the adapter: evaluate and generate

```bash
./target/release/ferrotensor eval --model data/models/SmolLM2-135M \
  --data data/alpaca/val.jsonl \
  --lora data/runs/sft/lora_adapters.safetensors
# eval: 64 examples, 2744 tokens, loss 1.6519, perplexity 5.217 (9.7s)

./target/release/ferrotensor generate --model data/models/SmolLM2-135M \
  --lora data/runs/sft/lora_adapters.safetensors \
  --prompt "Translate 'good morning' into French." --temperature 0
# Bonjour, bonne journe!
# [18 prompt + 8 new tokens, 7.8 tok/s, kv cache on]
```

Without `--lora`, the same commands give the base-model rows above.
`generate` also takes `--temperature`, `--top-k`, `--top-p`, `--seed`,
`--system`, `--max-new-tokens`, `--raw` (no chat template) and
`--no-kv-cache`. Greedy output is identical with and without the cache.

### 5. Resume

```bash
./target/release/ferrotensor train --model data/models/SmolLM2-135M \
  --data data/alpaca/train.jsonl --out data/runs/sft --seq-len 256 \
  --lr 5e-4 --warmup-steps 20 --grad-accum 8 --max-steps 300 \
  --resume data/runs/sft/lora_adapters.safetensors
```

Resuming restores the adapter, the AdamW moments and the step counter. It
then skips the batches already seen, in the same seeded order. A 2-step
continuation of this run printed `continuing at step 250` and kept
training normally (losses 1.4960 and 1.6540).

## Known limits

- Ferrotensor is slower than PyTorch on CUDA: 0.84×, 0.32× and 0.37× of its
  speed in the three measured runs.
- Every NVIDIA number here is from one GPU, an A10G. AMD, Intel and DX12 are
  untested, and the only Apple run is the 0.4 build above.
- The M2 run above predates moving the RMSNorm and RoPE backward passes and
  the loss to the GPU and batching kernel submissions; its speed and memory
  are not current.
- Peak host memory for SmolLM2-1.7B is 5.33 GB with every option measured
  (f16 or 4-bit base weights, with or without `--checkpoint-blocks`): the
  GPU memory work does not change it. Models above 1.7B are untested.
- The logits and their gradient are still materialised: 1,008 MiB at the
  peak for SmolLM2-135M at batch 8 with `--checkpoint-blocks`.
- 4-bit base weights (`--q4-weights`) are experimental and give a worse
  held-out loss than f16 after 250 steps: 1.8000 vs 1.6521 (135M), 1.4087
  vs 1.3228 (360M), 1.1198 vs 1.0640 (1.7B)
  ([details](../../ferrotensor-cli/README.md#frozen-base-weights-f16-f32-or-4-bit-experimental)).
- `--batch-size 1` is the default. `--batch-size N` runs N sequences through
  a Llama-style model in one pass, padded to the longest, with the same
  result as one at a time; the logits and their gradient grow with N (about
  50 MB each per SmolLM2 sequence of 256 tokens), so GPU memory grows with N.
- The tested architecture is Llama-style. Qwen3-style attention (per-head
  RMSNorm of queries and keys before RoPE, `q_norm`/`k_norm`) is implemented
  and checked on a small synthetic model against a CPU reference of the
  forward pass; real Qwen3 checkpoints are not measured yet. Released LFM2
  checkpoints are rejected because their per-head q/k layernorm
  (`q_layernorm`) is not wired up, and LFM2 conv blocks also lack the B/C
  gating and a backward pass.
