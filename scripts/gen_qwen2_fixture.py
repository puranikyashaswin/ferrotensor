#!/usr/bin/env python3
"""Write a tiny random Qwen2 checkpoint and its Hugging Face transformers logits.

The model is `Qwen2ForCausalLM` from transformers: biased query, key and
value projections, tied embeddings, a RoPE base of 1e6, RMSNorm epsilon
1e-6, grouped-query attention with a head size of 16. Every weight,
including the biases and norm scales, is random and of order 0.1 to 1, so a
dropped bias or norm moves the logits visibly. The config is written in the
layout of the Qwen2.5 configs on the Hugging Face hub and read back by
transformers, so both sides load the same file.

`ferrotensor-trainer/tests/qwen2_family.rs` loads the checkpoint on the GPU
and compares its logits, and the gradients of its LoRA adapters, with the
ones written here. CI runs this script and passes the output directory to
that test as FERROTENSOR_QWEN2_FIXTURE.

Usage (torch, transformers and safetensors installed):

    python scripts/gen_qwen2_fixture.py OUT_DIR

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
EPS = 1e-6
THETA = 1_000_000.0

IDS = [3, 17, 5, 29, 0, 11, 23, 8, 60, 42, 7, 51]
TARGETS = IDS[1:] + [33]

LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 62,
    "eos_token_id": 63,
    "hidden_act": "silu",
    "hidden_size": HIDDEN,
    "initializer_range": 0.02,
    "intermediate_size": FFN,
    "max_position_embeddings": 128,
    "max_window_layers": LAYERS,
    "model_type": "qwen2",
    "num_attention_heads": HEADS,
    "num_hidden_layers": LAYERS,
    "num_key_value_heads": KV_HEADS,
    "rms_norm_eps": EPS,
    "rope_theta": THETA,
    "sliding_window": 128,
    "tie_word_embeddings": True,
    "torch_dtype": "float32",
    "use_cache": False,
    "use_sliding_window": False,
    "vocab_size": VOCAB,
}


def uniform(shape, scale, generator):
    return (torch.rand(shape, generator=generator) * 2 - 1) * scale


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: gen_qwen2_fixture.py OUT_DIR")
    out = Path(sys.argv[1])
    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")

    torch.use_deterministic_algorithms(True)
    config = transformers.AutoConfig.from_pretrained(model_dir, attn_implementation="eager")
    model =transformers.Qwen2ForCausalLM(config).to(torch.float32).eval()

    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(1 + uniform(param.shape, 0.25, generator))
            elif name.endswith(".bias"):
                param.copy_(uniform(param.shape, 0.5, generator))
            elif name == "model.embed_tokens.weight":
                param.copy_(uniform(param.shape, 0.5, generator))
            else:
                param.copy_(uniform(param.shape, 0.2, generator))

    # The properties the Rust side relies on, checked on the transformers model.
    attn = model.model.layers[0].self_attn
    assert attn.q_proj.bias is not None and attn.k_proj.bias is not None
    assert attn.v_proj.bias is not None and attn.o_proj.bias is None
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), "untied"
    assert attn.q_proj.weight.shape == (HEADS * HEAD_DIM, HIDDEN)
    expected_inv_freq = 1.0 / THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    torch.testing.assert_close(model.model.rotary_emb.inv_freq, expected_inv_freq)
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
        "generator": "scripts/gen_qwen2_fixture.py",
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
