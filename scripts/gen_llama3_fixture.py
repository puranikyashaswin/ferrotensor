#!/usr/bin/env python3
"""Write a tiny random Llama 3 checkpoint and its Hugging Face transformers logits.

The model is `LlamaForCausalLM` from transformers with the Llama 3 RoPE
scaling (`rope_type: "llama3"`) and grouped-query attention (4 query heads, 2
key/value heads, head size 16), tied embeddings and a RoPE base of 500000, as
in Llama-3.2-1B. The scaling uses `original_max_position_embeddings` 64, so
the 8 frequency pairs of a head fall in all three regimes (kept, blended,
divided by the factor) and the 12 positions of the test input see the
difference: unscaled RoPE moves the logits visibly. Every weight is random and
of order 0.1 to 1. The config is written in the layout of the Llama 3.2
configs on the Hugging Face hub and read back by transformers, so both sides
load the same file.

`ferrotensor-trainer/tests/llama3_rope.rs` loads the checkpoint on the GPU and
compares its logits, and the gradients of its LoRA adapters, with the ones
written here. CI runs this script and passes the output directory to that test
as FERROTENSOR_LLAMA3_FIXTURE.

Usage (torch, transformers and safetensors installed):

    python scripts/gen_llama3_fixture.py OUT_DIR

Output:
    OUT_DIR/model/config.json         the checkpoint config
    OUT_DIR/model/model.safetensors   weights in Hugging Face layout, f32,
                                      without lm_head.weight (tied)
    OUT_DIR/expected.safetensors      `logits` [1, seq, vocab] and
                                      `grad.<name>` [out, in] of the loss
                                      below for every decoder-block linear
                                      weight
    OUT_DIR/cases.json                input ids, targets, loss, versions
"""

import json
import sys
from pathlib import Path

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
EPS = 1e-5
THETA = 500_000.0
SCALING = {
    "factor": 8.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 64,
    "rope_type": "llama3",
}

IDS = [3, 17, 5, 29, 0, 11, 23, 8, 60, 42, 7, 51]
TARGETS = IDS[1:] + [33]

LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 62,
    "eos_token_id": 63,
    "head_dim": HEAD_DIM,
    "hidden_act": "silu",
    "hidden_size": HIDDEN,
    "initializer_range": 0.02,
    "intermediate_size": FFN,
    "max_position_embeddings": 4096,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": HEADS,
    "num_hidden_layers": LAYERS,
    "num_key_value_heads": KV_HEADS,
    "pretraining_tp": 1,
    "rms_norm_eps": EPS,
    "rope_scaling": SCALING,
    "rope_theta": THETA,
    "tie_word_embeddings": True,
    "torch_dtype": "float32",
    "use_cache": False,
    "vocab_size": VOCAB,
}


def uniform(shape, scale, generator):
    return (torch.rand(shape, generator=generator) * 2 - 1) * scale


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: gen_llama3_fixture.py OUT_DIR")
    out = Path(sys.argv[1])
    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")

    torch.use_deterministic_algorithms(True)
    config = transformers.AutoConfig.from_pretrained(model_dir, attn_implementation="eager")
    model = transformers.LlamaForCausalLM(config).to(torch.float32).eval()

    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(1 + uniform(param.shape, 0.25, generator))
            elif name == "model.embed_tokens.weight":
                param.copy_(uniform(param.shape, 0.5, generator))
            else:
                param.copy_(uniform(param.shape, 0.2, generator))

    # The properties the Rust side relies on, checked on the transformers model.
    attn = model.model.layers[0].self_attn
    assert attn.q_proj.bias is None and attn.o_proj.bias is None
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), "untied"
    assert attn.q_proj.weight.shape == (HEADS * HEAD_DIM, HIDDEN)
    # The scaling is really applied: the table differs from plain RoPE in the
    # divided pairs and in the blended one, and agrees in the kept one.
    plain = 1.0 / THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    ratio = model.model.rotary_emb.inv_freq / plain
    assert getattr(model.model.rotary_emb, "rope_type", "llama3") == "llama3"
    assert abs(ratio[0].item() - 1.0) < 1e-6, ratio
    assert abs(ratio[-1].item() - 1.0 / SCALING["factor"]) < 1e-6, ratio
    assert (ratio[1].item() - 1.0 / SCALING["factor"]) > 0.05 and ratio[1].item() < 0.95, ratio
    assert model.model.norm.variance_epsilon == EPS

    ids = torch.tensor([IDS])
    logits = model(input_ids=ids).logits
    # Mean cross-entropy over every position, as masked_cross_entropy with a
    # full mask computes it.
    loss = torch.nn.functional.cross_entropy(logits[0], torch.tensor(TARGETS))
    loss.backward()

    expected = {"logits": logits.detach().contiguous()}
    for layer in range(LAYERS):
        for linear in LINEARS:
            name = f"model.layers.{layer}.{linear}.weight"
            grad = model.get_parameter(name).grad
            assert grad is not None and grad.abs().max() > 0, f"zero gradient for {name}"
            expected[f"grad.{name}"] = grad.contiguous()

    weights = {n: p.detach().contiguous() for n, p in model.named_parameters()}
    assert "lm_head.weight" not in weights, "tied lm_head is listed once, as embed_tokens"
    save_file(weights, model_dir / "model.safetensors")
    save_file(expected, out / "expected.safetensors")
    meta = {
        "generator": "scripts/gen_llama3_fixture.py",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "ids": IDS,
        "targets": TARGETS,
        "loss": loss.item(),
    }
    (out / "cases.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}: loss {loss.item():.6f}, logits |max| {logits.abs().max().item():.3f}")


if __name__ == "__main__":
    main()
