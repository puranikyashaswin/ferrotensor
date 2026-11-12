# Using a Ferrotensor LoRA adapter from PyTorch (PEFT)

`ferrotensor train` writes its adapter in its own safetensors layout. For a
Llama-style model (SmolLM2, TinyLlama, Llama, Mistral) `export-adapter`
turns it into a [PEFT](https://github.com/huggingface/peft) adapter
directory, which `peft.PeftModel.from_pretrained` loads onto the same base
model in PyTorch. Ferrotensor also reads such a directory back, so an
adapter trained with PEFT can be evaluated or sampled here.

## Export

```bash
cargo build --release -p ferrotensor-cli
./target/release/ferrotensor export-adapter \
  --lora data/runs/sft/lora_adapters.safetensors \
  --model data/models/SmolLM2-135M \
  --output data/runs/sft/peft \
  --base-model HuggingFaceTB/SmolLM2-135M
```

This writes `data/runs/sft/peft/adapter_model.safetensors` and
`adapter_config.json`. Only the model's `config.json` is read; no GPU is
needed. `--base-model` becomes `base_model_name_or_path` in the config
(default: the `--model` path). `--format peft` is the default and only
format. `--adapter` is accepted as another name for `--lora`.

## Load in PyTorch

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.float32)
model = PeftModel.from_pretrained(base, "data/runs/sft/peft")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
```

`model.merge_and_unload()` folds the adapter into the base weights if you
want a plain `transformers` model.

## Load a PEFT adapter in Ferrotensor

`eval` and `generate` take a PEFT adapter directory wherever they take an
adapter file:

```bash
./target/release/ferrotensor generate --model data/models/SmolLM2-135M \
  --lora data/runs/sft/peft --prompt "Hi" --temperature 0
```

In Rust, `ferrotensor_trainer::peft::load_peft_adapter` returns the tensors
with their rank, alpha and targets, ready for
`ferrotensor_trainer::checkpoint::apply_checkpoint`;
`ferrotensor_trainer::peft::save_peft_adapter` exports straight from a model
in memory.

## What converts

| | Ferrotensor | PEFT |
|---|---|---|
| Key | `model.layers.0.self_attn.q_proj.lora_A.weight` | `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight` |
| `lora_A` | `[in, rank]` | `[rank, in]` |
| `lora_B` | `[rank, out]` | `[out, rank]` |
| Scale | `alpha / rank` | `lora_alpha / r` |

Supported targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, `down_proj` and `lm_head`, with one rank and alpha for all of
them, on every layer. LFM2 adapters, conv projections, mixed ranks and, on
import, DoRA, LoRA bias, `fan_in_fan_out`, `modules_to_save`,
`layers_to_transform` and `rank_pattern`/`alpha_pattern` are errors, not
silently skipped. On import, `use_rslora` is kept by storing
`alpha = lora_alpha * sqrt(r)`.

## Parameter names of Llama-style models

Ferrotensor now names the parameters of Llama-style models as Hugging Face
does: `input_layernorm`, `post_attention_layernorm` and `mlp.*` instead of
the LFM2 names `operator_norm`, `ffn_norm` and `feed_forward.*`. Attention
(`self_attn.*`), `lm_head`, `model.norm` and `model.embed_tokens` keep their
names; LFM2 models keep all of theirs. Adapter files and optimizer state
written with the old names still load (`apply_checkpoint` maps each old name
to the new one); a file holding both forms of one tensor, or a tensor the
model does not have, is an error.

## How this is checked

The software-Vulkan CI job trains a LoRA adapter on all eight targets of a
tiny random Llama model
([`ferrotensor-trainer/examples/peft_fixture.rs`](../../ferrotensor-trainer/examples/peft_fixture.rs)),
exports it with `export-adapter`, and runs
[`scripts/check_peft_export.py`](../../scripts/check_peft_export.py): it
loads the base model with `transformers`, the adapter with `peft`, checks
that every adapter tensor loaded, and compares the logits for fixed token
ids with the ones Ferrotensor computed. Pinned versions: torch 2.14.1 (CPU
wheel), transformers 5.20.2, peft 0.21.2, safetensors 0.8.0. To run it
yourself:

```bash
cargo run --release -p ferrotensor-trainer --example peft_fixture -- out
cargo run --release -p ferrotensor-cli -- export-adapter \
  --lora out/adapter.safetensors --model out/model --output out/peft
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install transformers==5.20.2 peft==0.21.2 safetensors==0.8.0
python scripts/check_peft_export.py --model out/model --adapter out/peft --logits out/logits.json
```

The GPU tests in
[`ferrotensor-trainer/tests/peft_export.rs`](../../ferrotensor-trainer/tests/peft_export.rs)
check that export then import gives bit-identical logits, that the exported
keys and shapes are exactly PEFT's, and that adapters with the old names
still load.
