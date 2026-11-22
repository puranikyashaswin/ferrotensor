# AGENTS.md

Ferrotensor is a Rust framework for tensors, autograd and LoRA fine-tuning of
small language models on the GPU, through wgpu and hand-written WGSL kernels
(Vulkan, Metal or DX12; no CUDA). The `ferrotensor` command (crate
`ferrotensor-cli`) plans, trains, evaluates, samples and merges LoRA adapters
for Llama-style models such as SmolLM2, TinyLlama and Qwen2, and trains a small
model from scratch with a byte-level or a learned BPE tokenizer. One A10G run
of the commands, with the numbers, is in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md#real-hardware-run-16-november-2026).

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
ferrotensor export-gguf --model outputs/merged --out model.gguf --json  # for llama.cpp and Ollama
ferrotensor pretrain --data corpus.txt --out outputs/tiny --json  # from random weights; generate --raw loads it
ferrotensor pretrain --data corpus.txt --out outputs/tiny-bpe --tokenizer bpe --vocab-size 4096 --json  # BPE learned from the corpus
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
| `train` | `steps`, `first_loss`, `final_loss`, `eval_loss_before`, `eval_loss_after` (null without `--eval-data`), `measured_peak_mib`, `seconds`, `out_dir`, `adapter` (null on ranks above 0, which save nothing), and with `--q4-weights` only, `q4_weights` (`modules`, `skip`, `group`, `tensors`: the 4-bit selection and how many tensors it stored at 4 bits), and with `FERROTENSOR_EXPERIMENTAL=1` on a GPU with f16 cooperative matrices only, `loss_scaler` (`scale`, `skipped_steps`, `f16_overflows`: the dynamic loss scale at the end, the optimizer steps it skipped on overflow, and the matmul workgroups that clamped an input past the f16 range), and with `FERROTENSOR_AUTOTUNE` on in that setting only, `autotune` (`mode`, `seconds` spent on trials, `shapes_tuned`, `shapes_cached`, `shapes_open`, `budget_exhausted`, `cache_file`) |
| `eval` | `examples`, `tokens`, `loss`, `perplexity`, `seconds` |
| `generate` | `text`, `prompt_tokens`, `new_tokens`, `tokens_per_second` (decode speed: one-token forward passes after the prompt per second, prompt time left out; null when none ran), `prefill_seconds` (time to run the prompt), `kv_cache`, and with `--profile` only, `profile` (`kernel_timing`, `prefill` and `decode`: see [GPU time by kernel](#gpu-time-by-kernel---profile)) |
| `chat` | one object per reply, each on its own line: `reply`, `prompt_tokens` (the whole conversation), `cached_tokens` (prompt tokens taken from the KV cache of earlier turns, not run again), `new_tokens`, `tokens_per_second` and `prefill_seconds` (as for `generate`; with the kept cache the prefill runs only the uncached tokens) |
| `merge` | `out_dir`, `merged` (tensors with the adapter added), `copied` (tensors copied byte for byte) |
| `export-gguf` | `out_file`, `tensors` (tensors in the GGUF file), `merged` (tensors with the `--lora` adapter added; 0 without it) |
| `pretrain` | every `train` key, then `parameters`, `vocab_size` (259 for the byte-level tokenizer, `--vocab-size` for BPE) and `tokenizer` (`"byte"` or `"bpe"`) |
| `info` | `ferrotensor`, `os`, `arch`, `adapters` (one object per GPU, `selected` marks the one used, `kernel_timing` the mode `--profile` would time kernels in, or null, `cooperative_matrix` the cooperative-matrix shapes it offers: `m`, `n`, `k`, `ab_type`, `cr_type`, `saturating_accumulation`; the device asks for them only with `FERROTENSOR_EXPERIMENTAL=1`), `checks` (null without `--check`), and `error` only on failure |
| `q4-report` | `model`, `group`, `tensors` (one object per 2-D weight, by name: `name`, `shape` as stored, `relative_error`, `max_abs_error`, `groups`, `outlier_groups`, `outlier_fraction`) |

`chat` is the one command that prints more than one object: it reads one
user message per line of stdin and writes one JSON line per reply, so read
stdout line by line. An error ends the session with a one-line
`{"error": ...}` object.

### GPU time by kernel (`--profile`)

`train --profile` writes `OUT/profile.json` with the median step's totals
(`step_seconds`, `gpu_wait_seconds`, `<count>_per_step`) and, from
timestamp queries, `kernel_timing` (`"inside_passes"`, `"pass_boundaries"`
or null when the adapter has none), `kernel_gpu_seconds_per_step` and
`kernels`: one object per kernel, longest GPU time first, with `name`,
`dispatches_per_step` and `gpu_seconds_per_step`. `generate --profile
--json` adds `profile` with `kernel_timing`, `prefill` (`gpu_wait_seconds`,
`kernel_gpu_seconds`, `kernels` of `name`, `dispatches`, `gpu_seconds`) and
`decode` (`tokens`, `gpu_wait_seconds_per_token`,
`kernel_gpu_seconds_per_token`, `kernels` of `name`, `dispatches_per_token`,
`gpu_seconds_per_token`). The kernel sum attributes GPU time; it leaves out
idle gaps and is not an exact budget (see
[ferrotensor-cli/README.md](ferrotensor-cli/README.md#gpu-time-by-kernel)).

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
`rope_scaling`, `use_sliding_window: true`, a head size above 128); a model
id that is not in the Hugging Face cache or whose snapshot has no `config.json` (`error` names the cache and
`hf download <id>`); a flag
value out of range; a `--q4-modules` / `--q4-skip` selection that matches no
weight; a dataset with no usable examples; no GPU adapter;
`plan` or `finetune` finding no configuration that fits; `generate` with a
prompt that encodes to no tokens or is longer than the model's context; `chat`
refusing a tokenizer without ChatML tokens or a message longer than the
model's context; and `merge`
refusing its input (a non-empty `--out`, a quantized base, an adapter that
does not match the base); `export-gguf` refusing its input (an `--out` that
exists, a `model_type` other than `llama`, a config `check_supported`
refuses such as scaled RoPE, projection biases (Qwen2), a quantized base, a tokenizer that
is not byte-level BPE with the GPT-2 or SmolLM pre-tokenizer, an adapter that
does not match the base); and `pretrain` refusing a model size the kernels
cannot run (`--hidden` not a multiple of `--heads`, an odd head size or one
above 128), a `--seq-len` above `--context`, an `--out` that already holds
`model.safetensors` or `init/` without `--resume`, `--resume` without
`trained_weights.safetensors` or with a saved tokenizer of another size than
the flags ask for, `--vocab-size` without `--tokenizer bpe`, and for `bpe` a
`--vocab-size` of 259 or less, one the corpus has too few pairs for, or a
corpus that is not UTF-8. clap's usage errors
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
- 16 GB: on an NVIDIA Tesla T4 (16 GB) the planner chose batch 2 x
  accumulation 4 for SmolLM2-1.7B, which peaked at 11,138 MiB over 250 steps
  ([second card](benchmarks/RESULTS.md#second-card-nvidia-tesla-t4-16-gb)).
  `plan` adds `--checkpoint-blocks` when the default does not fit.
- 24 GB: all three sizes were measured on the A10G (24 GB). Models larger than
  SmolLM2-1.7B are untested. A 16 November 2026 run of
  `finetune` with no memory flags on the A10G is in
  [benchmarks/RESULTS.md](benchmarks/RESULTS.md#real-hardware-run-16-november-2026):
  SmolLM2-1.7B chose batch 4 x accumulation 2 and peaked at 15,064 MiB, and a
  budget of 3,000 MiB was refused with exit code 2.

The `plan` estimate is fitted to 24 peaks with 8 sequences per step, 15 on the
A10G and 9 on the T4: these, and batch 2, 4 and 8 runs of SmolLM2 and
Qwen2.5-0.5B. It is at most 4.6% below them and at most 12% above the 250-step
and single-pass peaks; 30-step runs that accumulate passes have not reached
their peak, and it is up to 35% above those. Other GPUs, f32 and 4-bit weights
and LFM2 models are not calibrated.

`finetune` and `plan --data` estimate for the rows the data pads to, not for
`--seq-len`: the 99th percentile of the tokenized lengths sets the margin
check, and a batch of the longest sequences must fit the budget without the
margin. On the A10G, SmolLM2-1.7B at batch 8 x 1 on a 2,000-example Alpaca
split (longest sequence 214 tokens) peaked at 19,296 MiB, and the estimate for
8 x 214 rows is 20,409 MiB, 5.8% above it.

## 4-bit weights (experimental)

`--q4-weights` still fails its quality gate: the untouched base model has a
higher held-out loss with it than with f16
([A10G run](benchmarks/RESULTS.md#4-bit-weights-the-gap-is-there-before-training)).
Use it only when nothing else fits. `--q4-modules KEYS`, `--q4-skip KEYS`
(keys as `--train-modules`, plus `layers.N` and `layers.A-B`) and
`--q4-group 16|32|64|128` exist to study where that loss comes from, and
`ferrotensor q4-report --model DIR --json` measures the 4-bit error of every
weight matrix on the host, without a GPU. See
[ferrotensor-cli/README.md](ferrotensor-cli/README.md#finding-where-the-4-bit-loss-comes-from).

## Cooperative-matrix matmuls (experimental)

With `FERROTENSOR_EXPERIMENTAL=1`, on a GPU whose adapter offers 16x16x16 f16
cooperative matrices with f32 accumulation and 32-thread subgroups, matmuls
against f16 base weights (at least 32 rows) round their f32 input to f16 and
run on those matrices; LoRA's thin matmuls stay in f32. `train` then turns on
dynamic loss scaling (start 65536, x2 after 2,000 good steps, x0.5 and a
skipped step on overflow) and reports it as `loss_scaler`. Any other value of
the variable, or another GPU, keeps the f32 path.

The tile and K split of each of those matmuls come from static rules that were
measured on one GPU. `FERROTENSOR_AUTOTUNE=1` times candidates instead, inside
the training step: each product (layout, kernel variant, rows rounded up to a
bucket, N and K) runs three timed trials per candidate as real work, read back
from GPU timestamp queries, and then keeps the fastest unless the static pick
is within 3%. The choices are saved to `autotune.txt` in the platform's cache
directory (`FERROTENSOR_AUTOTUNE_CACHE` sets the path), keyed by adapter,
driver and Ferrotensor version, and a later run starts from them; a file for
another adapter or a corrupt file is ignored with a warning on stderr.
`FERROTENSOR_AUTOTUNE=retune` ignores the file, and
`FERROTENSOR_AUTOTUNE_BUDGET` (seconds, default 30) stops new products from
starting trials once that much time went into them. A trial's K split sums its
slices in a different order, so the first steps of a tuning run differ from an
untuned run by rounding; with a cache they do not. Any other value, no
timestamp queries or no cooperative matrices keep the static rules exactly.

Under the same setting, `train` runs the q, k and v projections of an
attention block, and the gate and up projections of a feed-forward, as one
matmul each when every one is a bias-free frozen f16 weight with a LoRA
adapter. `FERROTENSOR_FUSE_QKV` and `FERROTENSOR_FUSE_GATEUP` (`on`, `off` or
`auto`, read once at start, default `auto`) choose per kind; `auto` fuses q, k
and v always and gate and up only when their summed width is at most 8,192
(SmolLM2-135M and 360M; on one A10G the gate and up group was 3% to 6% faster
than the separate projections at 360M and 5% slower at 1.7B); `off` runs the
unfused layers of before, bit for bit, and any other value is `auto` with a
warning. A fused group keeps the only copy of its weights, so the model holds
them once; the adapter file, `merge` and `export-gguf` still use the original
tensors and layout, and a pass with a KV cache copies the weights back out of
the group once.

## Working on the code

- GitHub CI is the reference build: `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`,
  docs and MSRV. Format with `cargo fmt --all` before every commit.
- GPU tests are skipped unless `FERROTENSOR_RUN_GPU_TESTS=1`; CLI tests that
  need a GPU are `#[ignore]` and run in CI with `--ignored` on software Vulkan,
  on Metal (GitHub's Apple silicon runners) and on DX12 (WARP; there without
  the `ferrotensor-cli` tests, which take too long, and on pull requests only
  the backend kernel and parity tests; the rest runs on `main`, weekly and by
  `workflow_dispatch`).
  The four Qwen2 parity tests also read `FERROTENSOR_QWEN2_FIXTURE` (CI sets
  it) and skip with a message when it is unset.
- No `unwrap`, `expect` or `panic!` outside tests; propagate errors with `?`.
  No `let _ =`, `.ok()` or `unwrap_or_default()` to silence a `Result`.
- In library code a lint is suppressed with `#[expect(lint, reason = "...")]`.
  Test and example files carry one `#![allow(...)]` at the top, or one
  `#[allow(...)]` on `mod tests`.
- `unsafe` is denied, except the loader's two documented memory-map sites and
  `enable_experimental_features` in `ferrotensor-backend/src/adapters.rs`, which
  opts in to wgpu's experimental features only with `FERROTENSOR_EXPERIMENTAL=1`,
  and `ShaderChecks::module` in `ferrotensor-backend/src/shader_checks.rs`, which
  builds the crate's own WGSL kernels with `create_shader_module_trusted`
  without naga's forced loop bounding, and without the other runtime checks
  `FERROTENSOR_SHADER_CHECKS` leaves out (`default` keeps all of wgpu's). With
  that variable unset, the cooperative-matrix kernel of products of 256 rows or
  more is also built without bounds checks: each of its accesses has its own
  guard (see `Coop::kernel`).
- A comment says why the code does something. A doc comment is one summary sentence, plus
  `# Errors`, `# Panics` or `# Safety` when they apply.
- No new dependency without the maintainer's approval. One concern per PR.
- Tests assert concrete values against an independent reference.
- Every number in the docs comes from a run, with the machine named.
- The PR body says what changed and how it was tested, and ends with a
  `Changelog:` line that summarizes the change in one sentence.
