# AGENTS.md

Ferrotensor is a Rust framework for tensors, autograd and LoRA fine-tuning of
small language models on the GPU, through wgpu and hand-written WGSL kernels
(Vulkan, Metal or DX12; no CUDA). The `ferrotensor` command (crate
`ferrotensor-cli`) plans, trains, evaluates, samples and merges LoRA adapters
for Llama-style models such as SmolLM2, TinyLlama and Qwen2 (Qwen2.5 checkpoints
are not measured yet), and trains a small
byte-level model from scratch.

## Install

```bash
cargo install ferrotensor-cli   # Rust 1.88+; the binary is `ferrotensor`
ferrotensor info --check --json # the GPU wgpu uses, and a kernel self-check
```

From a checkout, replace `ferrotensor` with `cargo run -p ferrotensor-cli --release --`.
A model is a Hugging Face checkpoint directory with `config.json`,
`tokenizer.json` and `*.safetensors`. `--model` also takes a Hugging Face model
id that is already in the cache (`hf download HuggingFaceTB/SmolLM2-135M`, then
`--model HuggingFaceTB/SmolLM2-135M`); an existing directory of that name wins,
and nothing is downloaded.

## The one-command path

```bash
ferrotensor plan --model DIR --json                       # what fits this GPU
ferrotensor finetune --model DIR --data train.jsonl --json # train with that plan
ferrotensor eval --model DIR --data val.jsonl --lora outputs/run/lora_adapters.safetensors --json
ferrotensor chat --model DIR --lora outputs/run/lora_adapters.safetensors --json < questions.txt
ferrotensor merge --model DIR --lora outputs/run/lora_adapters.safetensors --out outputs/merged --json
ferrotensor pretrain --data corpus.txt --out outputs/tiny --json  # from random weights; generate --raw loads it
```

`chat` needs a ChatML model such as SmolLM2 (TinyLlama's tokenizer lacks the
tokens); `generate --raw` works with any model.

`finetune` writes into `--out` (default `outputs/run`): `lora_adapters.safetensors`,
its `.state.json` and `.state.safetensors` for `--resume`, `metrics.jsonl` and
`training_state.json`.

## Data format

- SFT (default `--task sft`): `.jsonl`, one `{"prompt": ..., "response": ...}`
  object per line with an optional `"system"`, or a `.json` array of them.
  Loss is on response tokens only; prompts are wrapped in ChatML unless
  `--no-chat-template`.
- Pretraining (`--task pretrain`): `.txt` lines, a `.json` array of strings or
  token ids, or packed little-endian `u32` tokens in `.bin`.

## JSON output

With `--json`, stdout carries exactly one JSON object when the command ends;
progress and human text go to stderr. A float that was never measured (NaN)
is `null`. On an error stdout carries `{"error": "<message>"}` instead.

| Command | Keys |
|---|---|
| `plan` | `adapter`, `backend`, `device_type`, `budget_mib`, `budget_source`, `model`, `parameters`, `chosen`, `estimated_peak_mib`, `margin`, `fits`, `train_command`, `notes`; without `--model` the per-model keys are in a `models` array |
| `finetune` | `chosen` (`batch_size`, `seq_len`, `rank`, `precision`, `checkpoint_blocks`), `grad_accum`, `budget_mib`, `estimated_peak_mib`, `fits`, then every `train` key |
| `train` | `steps`, `first_loss`, `final_loss`, `eval_loss_before`, `eval_loss_after` (null without `--eval-data`), `measured_peak_mib`, `seconds`, `out_dir`, `adapter` (null on ranks above 0, which save nothing) |
| `eval` | `examples`, `tokens`, `loss`, `perplexity`, `seconds` |
| `generate` | `text`, `prompt_tokens`, `new_tokens`, `tokens_per_second`, `kv_cache` |
| `chat` | one object per reply, each on its own line: `reply`, `prompt_tokens`, `new_tokens`, `tokens_per_second` |
| `merge` | `out_dir`, `merged` (tensors with the adapter added), `copied` (tensors copied byte for byte) |
| `pretrain` | every `train` key, then `parameters` and `vocab_size` (259: a byte-level tokenizer) |
| `info` | `ferrotensor`, `os`, `arch`, `adapters` (one object per GPU, `selected` marks the one used), `checks` (null without `--check`), and `error` only on failure |

`chat` is the one command that prints more than one object: it reads one
user message per line of stdin and writes one JSON line per reply, so read
stdout line by line. An error ends the session with a one-line
`{"error": ...}` object.

`measured_peak_mib` counts the GPU buffers Ferrotensor tracks, not the driver
context, so it is below what `nvidia-smi` shows. `classify-train`,
`classify-predict`, `export-adapter`, `monitor`, `demo` and `benchmark` take
no `--json`.

## Exit codes

| Code | Meaning | What to do |
|---|---|---|
| 0 | Success | Read the JSON on stdout. |
| 1 | Runtime error: I/O, the GPU, a kernel out of tolerance | Read `error`; retrying with the same input gives the same result. |
| 2 | The request cannot be met on this machine, or the input is invalid | Change the request: fix the path or flag named in `error`; when nothing fits, use a smaller model, `--q4-weights`, or `--force` to try the smallest configuration. |
| 101 | A panic, with no JSON on stdout | wgpu panics when the GPU device is lost, most often because it ran out of memory (stderr then ends with a hint). Lower `--batch-size` or `--seq-len`, or use `finetune`. |

Code 2 covers: a missing model directory, `config.json` or data file; a
checkpoint the model builder cannot load (a missing or misshapen weight, or a
`config.json` feature the forward pass lacks: a `hidden_act` other than SiLU,
`rope_scaling`, sliding-window attention, a head size above 128); a model
id that is not in the Hugging Face cache or whose snapshot has no `config.json` (`error` names the cache and
`hf download <id>`); a flag
value out of range; a dataset with no usable examples; no GPU adapter;
`plan` or `finetune` finding no configuration that fits; `generate` with a
prompt that encodes to no tokens or is longer than the model's context; `chat`
refusing a tokenizer without ChatML tokens or a message longer than the
model's context; and `merge`
refusing its input (a non-empty `--out`, a quantized base, an adapter that
does not match the base); and `pretrain` refusing a model size the kernels
cannot run (`--hidden` not a multiple of `--heads`, an odd head size or one
above 128), a `--seq-len` above `--context`, an `--out` that already holds
`model.safetensors` or `init/` without `--resume`, or `--resume` without
`trained_weights.safetensors`. clap's usage errors
also exit with 2, with the usage text on stderr and nothing on stdout.

## Picking a model

Run `ferrotensor plan --json` (no `--model`) to see which documented SmolLM2
size fits; it decides from the GPU memory budget. `--memory-mib` sets the
budget; else on a GPU that shares system RAM (Apple silicon, integrated GPUs)
it is 60% of system RAM, on an NVIDIA GPU the `nvidia-smi` total, and
otherwise an assumed 4,096 MiB, so pass `--vram-mib` for other discrete GPUs.

Measured peak GPU memory (`nvidia-smi`, MiB) on one NVIDIA A10G (24 GB), f16
base weights, batch 1 x 8-step accumulation, sequence length 256:

| Model | Default | `--checkpoint-blocks` |
|---|---:|---:|
| SmolLM2-135M | 2,306 | 1,890 |
| SmolLM2-360M | 3,579 | 2,783 |
| SmolLM2-1.7B | 8,142 | 5,313 |

- 8 GB: an Apple M2 (8 GB) fine-tuned SmolLM2-135M with a 0.4 build; there are
  no current memory numbers for the Mac. Start with SmolLM2-135M.
- 16 GB: no 16 GB machine has been measured. The table above is the guide;
  `plan` adds `--checkpoint-blocks` when the default does not fit.
- 24 GB: all three sizes were measured on the A10G (24 GB). Models larger than
  SmolLM2-1.7B are untested.

The `plan` estimate is calibrated on these A10G peaks; its worst error there is
13.3%. Other GPUs, f32 and 4-bit weights and LFM2 models are not calibrated.

## Working on the code

- GitHub CI is the reference build: `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`,
  docs and MSRV. Format with `cargo fmt --all` before every commit.
- GPU tests are skipped unless `FERROTENSOR_RUN_GPU_TESTS=1`; CLI tests that
  need a GPU are `#[ignore]` and run in CI with `--ignored` on software Vulkan.
- No `unwrap`, `expect` or `panic!` outside tests; propagate errors with `?`.
  No `let _ =`, `.ok()` or `unwrap_or_default()` to silence a `Result`.
- In library code a lint is suppressed with `#[expect(lint, reason = "...")]`.
  Test and example files carry one `#![allow(...)]` at the top, or one
  `#[allow(...)]` on `mod tests`.
- `unsafe` is denied, except the loader's two documented memory-map sites.
- A comment says why the code does something. A doc comment is one summary sentence, plus
  `# Errors`, `# Panics` or `# Safety` when they apply.
- No new dependency without the maintainer's approval. One concern per PR.
- Tests assert concrete values against an independent reference.
- Every number in the docs comes from a run, with the machine named.
- The PR body says what changed and how it was tested, and ends with a
  `Changelog:` line that summarizes the change in one sentence.
