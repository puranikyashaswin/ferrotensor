#!/usr/bin/env python3
"""Write a tiny random checkpoint per model family and its Hugging Face
transformers logits and gradients.

Each row of FAMILIES is one family: a transformers class, the config values
the family changes, a seed and a function that asserts, on the transformers
model, that the feature under test is live. Every weight, including biases
and norm scales, is random and of order 0.1 to 1, so a dropped bias or a
missing RoPE scaling moves the logits visibly. The config is written in the
layout of the configs on the Hugging Face hub and read back by transformers,
so both sides load the same file. A row has its own generator, so adding a row
does not change the others.

`ferrotensor-trainer/tests/family_parity.rs` loads each checkpoint on the GPU
and compares its logits, and the gradients of its LoRA adapters, with the
ones written here. CI runs this script and passes the output directory to
that test as FERROTENSOR_FAMILY_FIXTURES.

Usage (torch, transformers and safetensors installed):

    python scripts/gen_family_fixtures.py OUT_DIR [--only NAME ...]
    python scripts/gen_family_fixtures.py --real MODEL_DIR OUT_DIR

Output, for each family NAME:
    OUT_DIR/NAME/model/config.json         the checkpoint config
    OUT_DIR/NAME/model/model.safetensors   weights in Hugging Face layout,
                                           f32, without lm_head.weight when
                                           the embeddings are tied
    OUT_DIR/NAME/expected.safetensors      `logits` [1, seq, vocab] and
                                           `grad.<name>` [out, in] of the
                                           loss below for every decoder-block
                                           linear weight
    OUT_DIR/NAME/cases.json                input ids, targets, loss, versions

`--real` records transformers' float32 logits of a real checkpoint on 8
prompts instead: OUT_DIR holds `cases.json` (the token ids of each prompt,
from the checkpoint's own tokenizer, with the BOS token the tokenizer adds)
and `logits.safetensors` (`logits.<n>`: the `[tokens, vocab]` logits of prompt
n). `ferrotensor-trainer/tests/llama3_rope.rs` compares Ferrotensor's logits
with these.
"""

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import transformers
from safetensors.torch import save_file

VOCAB = 96
HIDDEN = 64
HEADS = 4
KV_HEADS = 2
HEAD_DIM = HIDDEN // HEADS
LAYERS = 2
FFN = 128

IDS = [3, 17, 5, 29, 0, 11, 23, 8, 60, 42, 7, 51]
TARGETS = IDS[1:] + [33]

LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

# What every row shares; a row adds or replaces keys.
BASE_CONFIG = {
    "attention_dropout": 0.0,
    "bos_token_id": 62,
    "eos_token_id": 63,
    "hidden_act": "silu",
    "hidden_size": HIDDEN,
    "initializer_range": 0.02,
    "intermediate_size": FFN,
    "num_attention_heads": HEADS,
    "num_hidden_layers": LAYERS,
    "num_key_value_heads": KV_HEADS,
    "tie_word_embeddings": True,
    "torch_dtype": "float32",
    "use_cache": False,
    "vocab_size": VOCAB,
}

QWEN2_THETA = 1_000_000.0
QWEN2_EPS = 1e-6

# A window of 4 tokens over the 12 of IDS: from the 5th token on, a token
# cannot see the whole sequence.
WINDOW = 4

MISTRAL_THETA = 10_000.0

LLAMA3_THETA = 500_000.0
LLAMA3_SCALING = {
    "factor": 8.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 64,
    "rope_type": "llama3",
}


def uniform(shape, scale, generator):
    return (torch.rand(shape, generator=generator) * 2 - 1) * scale


def check_qwen2(model, config):
    """Biased query, key and value projections and a RoPE base of 1e6."""
    attn = model.model.layers[0].self_attn
    assert attn.q_proj.bias is not None and attn.k_proj.bias is not None
    assert attn.v_proj.bias is not None and attn.o_proj.bias is None
    assert attn.q_proj.bias.abs().max() > 0.1, "the biases are not random"
    expected_inv_freq = 1.0 / QWEN2_THETA ** (
        torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    torch.testing.assert_close(model.model.rotary_emb.inv_freq, expected_inv_freq)


def check_llama3(model, config):
    """No biases, and the Llama 3 scaling is really applied: the table differs
    from plain RoPE in the divided pairs and in the blended one, and agrees in
    the kept one."""
    attn = model.model.layers[0].self_attn
    assert attn.q_proj.bias is None and attn.o_proj.bias is None
    plain = 1.0 / LLAMA3_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    ratio = model.model.rotary_emb.inv_freq / plain
    assert getattr(model.model.rotary_emb, "rope_type", "llama3") == "llama3"
    assert abs(ratio[0].item() - 1.0) < 1e-6, ratio
    assert abs(ratio[-1].item() - 1.0 / LLAMA3_SCALING["factor"]) < 1e-6, ratio
    assert (ratio[1].item() - 1.0 / LLAMA3_SCALING["factor"]) > 0.05 and ratio[1].item() < 0.95, ratio


def logits_without_window(model, window_layers):
    """The logits of `model` and of a copy with the window switched off.

    `window_layers` are the layers whose window must be live. The asserts
    read the window off the transformers layers themselves, and the second
    forward pass shows that it changes the output, so a config key that
    transformers ignored cannot pass."""
    for index, layer in enumerate(model.model.layers):
        expected = WINDOW if index in window_layers else None
        # Mistral's attention reads the window from the config at each call.
        actual = getattr(layer.self_attn, "sliding_window", model.config.sliding_window)
        assert actual == expected, (index, actual)
    ids = torch.tensor([IDS])
    with torch.no_grad():
        with_window = model(input_ids=ids).logits
        plain = copy.deepcopy(model)
        plain.config.sliding_window = None
        plain.config.use_sliding_window = False
        plain.config.layer_types = ["full_attention"] * LAYERS
        for layer in plain.model.layers:
            layer.self_attn.sliding_window = None
            if hasattr(layer, "attention_type"):
                layer.attention_type = "full_attention"
        if hasattr(plain.model, "has_sliding_layers"):
            plain.model.has_sliding_layers = False
        full = plain(input_ids=ids).logits
    moved = (with_window - full).abs().max().item()
    assert moved > 0.05, f"the window of {WINDOW} tokens does not change the logits ({moved})"
    # The first WINDOW tokens see everything either way.
    torch.testing.assert_close(with_window[0, :WINDOW], full[0, :WINDOW])
    return moved


def check_mistral(model, config):
    """No biases, and a window of 4 tokens on every layer: Mistral applies
    `sliding_window` whenever it is set."""
    attn = model.model.layers[0].self_attn
    assert attn.q_proj.bias is None and attn.o_proj.bias is None
    assert config.sliding_window == WINDOW
    expected_inv_freq = 1.0 / MISTRAL_THETA ** (
        torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    torch.testing.assert_close(model.model.rotary_emb.inv_freq, expected_inv_freq)
    logits_without_window(model, window_layers=set(range(LAYERS)))


def check_qwen2_window(model, config):
    """Qwen2 with `use_sliding_window` and `max_window_layers` 1: layer 0
    attends to the whole sequence and layer 1 to a window of 4 tokens."""
    assert config.use_sliding_window and config.sliding_window == WINDOW
    assert list(config.layer_types) == ["full_attention", "sliding_attention"], config.layer_types
    attn = model.model.layers[0].self_attn
    assert attn.q_proj.bias is not None and attn.q_proj.bias.abs().max() > 0.1
    logits_without_window(model, window_layers={1})


@dataclass
class Row:
    """One family: its transformers class, config and live-feature check."""

    cls: str
    config: dict
    check: Callable
    seed: int = 0


FAMILIES = {
    # Biased query, key and value projections, tied embeddings, a RoPE base of
    # 1e6, RMSNorm epsilon 1e-6, grouped-query attention with a head size of
    # 16, in the layout of the Qwen2.5 configs.
    "qwen2": Row(
        cls="Qwen2ForCausalLM",
        config={
            **BASE_CONFIG,
            "architectures": ["Qwen2ForCausalLM"],
            "max_position_embeddings": 128,
            "max_window_layers": LAYERS,
            "model_type": "qwen2",
            "rms_norm_eps": QWEN2_EPS,
            "rope_theta": QWEN2_THETA,
            "sliding_window": 128,
            "use_sliding_window": False,
        },
        check=check_qwen2,
    ),
    # Qwen2 with the sliding window on: `max_window_layers` 1 leaves layer 0
    # on full attention and puts layer 1 on a window of 4 tokens, which
    # transformers derives as `layer_types`.
    "qwen2_window": Row(
        cls="Qwen2ForCausalLM",
        config={
            **BASE_CONFIG,
            "architectures": ["Qwen2ForCausalLM"],
            "max_position_embeddings": 128,
            "max_window_layers": 1,
            "model_type": "qwen2",
            "rms_norm_eps": QWEN2_EPS,
            "rope_theta": QWEN2_THETA,
            "sliding_window": WINDOW,
            "use_sliding_window": True,
        },
        check=check_qwen2_window,
        seed=1,
    ),
    # Mistral: no biases, untied-style layout of Mistral-7B-v0.1 (here tied),
    # a RoPE base of 1e4 and a window of 4 tokens on every layer.
    "mistral": Row(
        cls="MistralForCausalLM",
        config={
            **BASE_CONFIG,
            "architectures": ["MistralForCausalLM"],
            "head_dim": HEAD_DIM,
            "max_position_embeddings": 128,
            "model_type": "mistral",
            "rms_norm_eps": 1e-5,
            "rope_theta": MISTRAL_THETA,
            "sliding_window": WINDOW,
        },
        check=check_mistral,
        seed=2,
    ),
    # The Llama 3 RoPE scaling and a RoPE base of 500000, as in Llama-3.2-1B.
    # `original_max_position_embeddings` 64 puts the 8 frequency pairs of a
    # head in all three regimes (kept, blended, divided by the factor), and
    # the 12 positions of the test input see the difference: unscaled RoPE
    # moves the logits visibly.
    "llama3": Row(
        cls="LlamaForCausalLM",
        config={
            **BASE_CONFIG,
            "architectures": ["LlamaForCausalLM"],
            "attention_bias": False,
            "head_dim": HEAD_DIM,
            "max_position_embeddings": 4096,
            "mlp_bias": False,
            "model_type": "llama",
            "pretraining_tp": 1,
            "rms_norm_eps": 1e-5,
            "rope_scaling": LLAMA3_SCALING,
            "rope_theta": LLAMA3_THETA,
        },
        check=check_llama3,
    ),
}


def write_family(name, row, out):
    family_dir = out / name
    model_dir = family_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(row.config, indent=2) + "\n")

    config = transformers.AutoConfig.from_pretrained(model_dir, attn_implementation="eager")
    model = getattr(transformers, row.cls)(config).to(torch.float32).eval()

    # Init by name suffix: norm scales near 1, biases and the embedding wide,
    # everything else of order 0.1.
    generator = torch.Generator().manual_seed(row.seed)
    with torch.no_grad():
        for param_name, param in model.named_parameters():
            if param_name.endswith("norm.weight"):
                param.copy_(1 + uniform(param.shape, 0.25, generator))
            elif param_name.endswith(".bias"):
                param.copy_(uniform(param.shape, 0.5, generator))
            elif param_name == "model.embed_tokens.weight":
                param.copy_(uniform(param.shape, 0.5, generator))
            else:
                param.copy_(uniform(param.shape, 0.2, generator))

    # The properties the Rust side relies on, checked on the transformers
    # model: the class is the row's, tying is as declared, and the feature
    # under test reached the model (a config key transformers ignored would
    # make the fixture test nothing).
    assert type(model).__name__ == row.cls, type(model).__name__
    tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    assert tied == row.config["tie_word_embeddings"], f"{name}: tying is not as declared"
    attn = model.model.layers[0].self_attn
    assert attn.q_proj.weight.shape == (HEADS * HEAD_DIM, HIDDEN)
    assert model.model.norm.variance_epsilon == row.config["rms_norm_eps"]
    row.check(model, config)

    ids = torch.tensor([IDS])
    logits = model(input_ids=ids).logits
    # Mean cross-entropy over every position, as masked_cross_entropy with a
    # full mask computes it.
    loss = torch.nn.functional.cross_entropy(logits[0], torch.tensor(TARGETS))
    loss.backward()

    expected = {"logits": logits.detach().contiguous()}
    for layer in range(LAYERS):
        for linear in LINEARS:
            weight = f"model.layers.{layer}.{linear}.weight"
            grad = model.get_parameter(weight).grad
            assert grad is not None and grad.abs().max() > 0, f"zero gradient for {weight}"
            expected[f"grad.{weight}"] = grad.contiguous()

    weights = {n: p.detach().contiguous() for n, p in model.named_parameters()}
    if tied:
        assert "lm_head.weight" not in weights, "tied lm_head is listed once, as embed_tokens"
    save_file(weights, model_dir / "model.safetensors")
    save_file(expected, family_dir / "expected.safetensors")
    meta = {
        "generator": "scripts/gen_family_fixtures.py",
        "family": name,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "ids": IDS,
        "targets": TARGETS,
        "loss": loss.item(),
    }
    (family_dir / "cases.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {family_dir}: loss {loss.item():.6f}, logits |max| {logits.abs().max().item():.3f}")


PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "def fibonacci(n):\n    if n < 2:\n        return n",
    "Once upon a time, in a small village by the sea, there lived",
    "The derivative of x squared with respect to x is",
    "Translate to German: Good morning, how are you?",
    "In 1969, the first humans walked on the surface of the",
    "A long sentence about the weather, the seasons, and how people feel about rain "
    "after many dry weeks in the middle of summer, which goes on for a while and ends here.",
]


def write_real_logits(model_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    ids, tensors = [], {}
    for n, prompt in enumerate(PROMPTS):
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = model(**encoded).logits[0].float().contiguous()
        ids.append(encoded["input_ids"][0].tolist())
        tensors[f"logits.{n}"] = logits
    (out_dir / "cases.json").write_text(json.dumps({"ids": ids}))
    save_file(tensors, str(out_dir / "logits.safetensors"))
    print(f"wrote {len(ids)} prompts to {out_dir}; vocab {logits.shape[1]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--only", nargs="+", choices=sorted(FAMILIES), metavar="NAME",
                        help="write these families only")
    parser.add_argument("--real", type=Path, metavar="MODEL_DIR",
                        help="record the logits of the real checkpoint in MODEL_DIR instead")
    args = parser.parse_args()

    if args.real:
        write_real_logits(args.real, args.out_dir)
        return
    torch.use_deterministic_algorithms(True)
    for name in args.only or FAMILIES:
        write_family(name, FAMILIES[name], args.out_dir)


if __name__ == "__main__":
    main()
