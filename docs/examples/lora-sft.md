# LoRA SFT on an 8 GB MacBook: SmolLM2-135M on Alpaca

This walkthrough fine-tunes a real Hugging Face checkpoint with LoRA,
evaluates it on held-out data, saves the adapter, reloads it and generates
text. Every number below was measured on the machine listed here, with the
exact commands shown.

| | |
|---|---|
| Machine | MacBook Air, Apple M2, 8 GB unified memory, macOS 27.2 |
| GPU backend | Metal through wgpu |
| Toolchain | rustc 1.99.1, release build |
| Model | [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) (base model, Llama architecture, 30 layers, hidden 576, 9 query / 3 KV heads, vocab 49152, tied embeddings) |
| Data | [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned): 2,000 training and 64 held-out examples, as prepared below |
| Adapter | LoRA rank 8, alpha 16, on `q_proj`, `k_proj`, `v_proj`, `o_proj` of every layer (921,600 trainable parameters, 240 tensors) |

## Result

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

## Correctness checks

Two checks were run against PyTorch with `transformers` before trusting the
loss numbers above. The scripts are not part of the repository.

- Forward pass: logits for a 20-token input matched
  `AutoModelForCausalLM` (float32) to about 1e-5 at every position, with the
  same argmax at all 20.
- Backward pass: with LoRA on q/k/v/o (rank 4) and a response mask, the
  loss matched (7.58104 vs 7.58115). Every one of the 240 adapter
  gradients matched autograd, with a worst relative error of 0.2%.

## Reproduce it

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
cargo build --release -p rusty-cli
/usr/bin/time -l ./target/release/rusty-cli train \
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
- `lora_adapters.safetensors.state.json`, the step and AdamW moments for
  `--resume` (50 MB, because the moments are stored as JSON);
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
./target/release/rusty-cli eval --model data/models/SmolLM2-135M \
  --data data/alpaca/val.jsonl \
  --lora data/runs/sft/lora_adapters.safetensors
# eval: 64 examples, 2744 tokens, loss 1.6519, perplexity 5.217 (9.7s)

./target/release/rusty-cli generate --model data/models/SmolLM2-135M \
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
./target/release/rusty-cli train --model data/models/SmolLM2-135M \
  --data data/alpaca/train.jsonl --out data/runs/sft --seq-len 256 \
  --lr 5e-4 --warmup-steps 20 --grad-accum 8 --max-steps 300 \
  --resume data/runs/sft/lora_adapters.safetensors
```

Resuming restores the adapter, the AdamW moments and the step counter. It
then skips the batches already seen, in the same seeded order. A 2-step
continuation of this run printed `continuing at step 250` and kept
training normally (losses 1.4960 and 1.6540).

## Known limits

- Speed is bound by kernel dispatch and host round trips. Every op is its
  own GPU submission, and RMSNorm and RoPE backward and the loss run on the
  CPU. The GPU is far from saturated; 8.8 s per 8-sequence step is a
  baseline, not a ceiling.
- Each step trains one sequence at a time. `--batch-size` only changes how
  many sequences share one optimizer step.
- The tested architecture is Llama-style. Released LFM2 checkpoints are
  rejected because per-head q/k layernorm is not implemented. LFM2 conv
  blocks also lack the B/C gating and a backward pass.
