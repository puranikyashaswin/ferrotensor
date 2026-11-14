#!/usr/bin/env python3
"""Generate the gradient-parity fixtures for Ferrotensor's LoRA training path.

Builds the tiny Llama-style decoder of `ferrotensor-graph/tests/common`
(grouped-query attention, rotate-half RoPE, RMSNorm, SwiGLU MLP, untied LM
head) in plain PyTorch, adds a LoRA adapter to every linear layer exactly as
`Linear::project_2d` does (`y = x W + (x A B) * alpha / rank`), and records
the loss and the autograd gradient of every trainable tensor for a few
batches. `ferrotensor-trainer/tests/gradient_parity.rs` runs the same
batches on the GPU and compares.

The base weights are the ones `tiny_llama_weights()` produces in Rust (the
same xorshift generator, reproduced bit for bit below), so
`ferrotensor-graph/tests/model_correctness.rs` can also check the logits
written here against its independent CPU reference.

No PEFT: the LoRA math is written out so the fixture has no hidden
behaviour. Everything runs on the CPU in float32.

Usage (from the repository root, with torch, safetensors and numpy):

    python scripts/gen_gradient_fixtures.py

Output: ferrotensor-graph/tests/fixtures/gradient_parity/
    config.json            model config (as the Rust fixture writes it)
    model.safetensors      base weights, Hugging Face layout ([out, in])
    adapters.safetensors   LoRA factors in Ferrotensor layout:
                           `<prefix>.lora_a` [in, rank], `<prefix>.lora_b` [rank, out]
    expected.safetensors   `logits.<case>` [batch, seq, vocab] and
                           `grad.<case>.<name>` in Ferrotensor layout
                           (linear weights [in, out], LoRA as above)
    cases.json             input ids, targets, loss masks and expected loss
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

# Must match ferrotensor-graph/tests/common/mod.rs.
VOCAB = 32
HIDDEN = 32
HEADS = 4
KV_HEADS = 2
HEAD_DIM = HIDDEN // HEADS
LAYERS = 2
FFN = 64
EPS = 1e-6
THETA = 500.0

# Must match `lora_everywhere()` in the same file.
RANK = 4
ALPHA = 8.0
SCALE = ALPHA / RANK

OUT = Path(__file__).resolve().parent.parent / "ferrotensor-graph/tests/fixtures/gradient_parity"

MASK64 = (1 << 64) - 1

LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def pseudo_random(n, seed, scale):
    """`pseudo_random` of ferrotensor-graph/tests/common/mod.rs, bit for bit."""
    s = ((seed * 0x9E3779B97F4A7C15) & MASK64) | 1
    units = np.empty(n, dtype=np.float32)
    for i in range(n):
        s ^= s >> 12
        s ^= (s << 25) & MASK64
        s ^= s >> 27
        units[i] = np.float32(((s * 0x2545F4914F6CDD1D) & MASK64) >> 40)
    units = units / np.float32(1 << 24)
    return (units * np.float32(2.0) - np.float32(1.0)) * np.float32(scale)


def tiny_llama_weights():
    """`tiny_llama_weights()` of the Rust fixture: name -> float32 array."""
    weights = {}
    seed = 1

    def add(name, shape, scale, offset):
        nonlocal seed
        seed += 1
        values = pseudo_random(int(np.prod(shape)), seed, scale) + np.float32(offset)
        weights[name] = values.astype(np.float32).reshape(shape)

    add("model.embed_tokens.weight", [VOCAB, HIDDEN], 0.5, 0.0)
    q, kv = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    for l in range(LAYERS):
        p = f"model.layers.{l}"
        add(f"{p}.self_attn.q_proj.weight", [q, HIDDEN], 0.2, 0.0)
        add(f"{p}.self_attn.k_proj.weight", [kv, HIDDEN], 0.2, 0.0)
        add(f"{p}.self_attn.v_proj.weight", [kv, HIDDEN], 0.2, 0.0)
        add(f"{p}.self_attn.o_proj.weight", [HIDDEN, q], 0.2, 0.0)
        add(f"{p}.mlp.gate_proj.weight", [FFN, HIDDEN], 0.2, 0.0)
        add(f"{p}.mlp.up_proj.weight", [FFN, HIDDEN], 0.2, 0.0)
        add(f"{p}.mlp.down_proj.weight", [HIDDEN, FFN], 0.2, 0.0)
        add(f"{p}.input_layernorm.weight", [HIDDEN], 0.25, 1.0)
        add(f"{p}.post_attention_layernorm.weight", [HIDDEN], 0.25, 1.0)
    add("model.norm.weight", [HIDDEN], 0.25, 1.0)
    add("lm_head.weight", [VOCAB, HIDDEN], 0.2, 0.0)
    return weights


def adapter_prefixes():
    return [f"model.layers.{l}.{m}" for l in range(LAYERS) for m in LINEARS] + ["lm_head"]


def make_adapters(weights):
    """Non-zero LoRA factors (a zero B would make every gradient of A zero)."""
    adapters = {}
    seed = 1000
    for prefix in adapter_prefixes():
        out_dim, in_dim = weights[f"{prefix}.weight"].shape
        seed += 2
        a = pseudo_random(in_dim * RANK, seed, 0.3).reshape(in_dim, RANK)
        b = pseudo_random(RANK * out_dim, seed + 1, 0.3).reshape(RANK, out_dim)
        adapters[f"{prefix}.lora_a"] = a
        adapters[f"{prefix}.lora_b"] = b
    return adapters


# ---------------------------------------------------------------------------
# The model, written from the Rust code it mirrors
# ---------------------------------------------------------------------------

def rms_norm(x, g):
    # RMSNorm::forward -> Tensor::rms_norm: x / sqrt(mean(x^2) + eps) * g
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g


def linear(p, prefix, x):
    # Linear::project_2d: x @ W (W stored [in, out]) + (x @ A @ B) * alpha / rank
    out = x @ p[f"{prefix}.weight"].T
    return out + (x @ p[f"{prefix}.lora_a"] @ p[f"{prefix}.lora_b"]) * SCALE


def rope(x, heads):
    # apply_rope_theta_in_place: within each head, (x[i], x[i + d/2]) is
    # rotated by pos * theta^(-2i/d); token t of every row sits at position t.
    b, s, _ = x.shape
    x = x.view(b, s, heads, HEAD_DIM)
    half = HEAD_DIM // 2
    i = torch.arange(half, dtype=torch.float32)
    freq = torch.tensor(THETA, dtype=torch.float32).pow(-2.0 * i / HEAD_DIM)
    angle = torch.arange(s, dtype=torch.float32)[:, None] * freq[None, :]
    cos, sin = angle.cos()[None, :, None, :], angle.sin()[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).view(b, s, heads * HEAD_DIM)


def attention(q, k, v):
    # Tensor::flash_attention with causal = true, no mask, no dropout;
    # query head h reads key/value head h / (HEADS / KV_HEADS).
    b, s, _ = q.shape
    q = q.view(b, s, HEADS, HEAD_DIM).transpose(1, 2)
    k = k.view(b, s, KV_HEADS, HEAD_DIM).transpose(1, 2).repeat_interleave(HEADS // KV_HEADS, dim=1)
    v = v.view(b, s, KV_HEADS, HEAD_DIM).transpose(1, 2).repeat_interleave(HEADS // KV_HEADS, dim=1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(HEAD_DIM)
    causal = torch.ones(s, s, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    out = scores.softmax(-1) @ v
    return out.transpose(1, 2).reshape(b, s, HEADS * HEAD_DIM)


def forward(p, ids):
    # Lfm2Model::forward / forward_batch for an all-attention (Llama) model.
    x = p["model.embed_tokens.weight"][ids]
    for l in range(LAYERS):
        pre = f"model.layers.{l}"
        h = rms_norm(x, p[f"{pre}.input_layernorm.weight"])
        q = rope(linear(p, f"{pre}.self_attn.q_proj", h), HEADS)
        k = rope(linear(p, f"{pre}.self_attn.k_proj", h), KV_HEADS)
        v = linear(p, f"{pre}.self_attn.v_proj", h)
        x = x + linear(p, f"{pre}.self_attn.o_proj", attention(q, k, v))
        h = rms_norm(x, p[f"{pre}.post_attention_layernorm.weight"])
        gate = linear(p, f"{pre}.mlp.gate_proj", h)
        up = linear(p, f"{pre}.mlp.up_proj", h)
        x = x + linear(p, f"{pre}.mlp.down_proj", torch.nn.functional.silu(gate) * up)
    return linear(p, "lm_head", rms_norm(x, p["model.norm.weight"]))


def loss_fn(logits, targets, masks):
    # masked_cross_entropy (one row) and masked_cross_entropy_batch: each
    # sequence is averaged over its supervised positions, and the batch loss
    # is the sum of those means.
    per_pos = torch.nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB), targets.reshape(-1), reduction="none"
    ).view(targets.shape)
    m = masks.to(torch.float32)
    return ((per_pos * m).sum(-1) / m.sum(-1)).sum()


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

ROW0 = [3, 17, 5, 29, 0, 11, 23, 8]  # IDS of model_correctness.rs
ROW1 = [30, 2, 19, 7, 7, 26, 1, 14]  # its second batch row
TARGET0 = [17, 5, 29, 0, 11, 23, 8, 12]
TARGET1 = [2, 19, 7, 7, 26, 1, 14, 31]
FULL = [1] * 8
MASK0 = [0, 0, 1, 1, 1, 0, 1, 1]  # 5 supervised positions
MASK1 = [0, 1, 1, 1, 1, 1, 1, 0]  # 6, so the per-sequence means differ in weight

CASES = {
    # One sequence through Lfm2Model::forward + masked_cross_entropy.
    "single": dict(rows=[ROW0], targets=[TARGET0], masks=[FULL], trainable_base=False),
    # Two sequences through forward_batch + masked_cross_entropy_batch.
    "batch": dict(rows=[ROW0, ROW1], targets=[TARGET0, TARGET1], masks=[FULL, FULL],
                  trainable_base=False),
    # The same, with ignored positions.
    "masked": dict(rows=[ROW0, ROW1], targets=[TARGET0, TARGET1], masks=[MASK0, MASK1],
                   trainable_base=False),
    # The masked batch with trainable base weights: gradients of every
    # weight except the embedding (its lookup runs on the host and is not
    # differentiable in Ferrotensor).
    "full": dict(rows=[ROW0, ROW1], targets=[TARGET0, TARGET1], masks=[MASK0, MASK1],
                 trainable_base=True),
}


def ferrotensor_layout(name, grad):
    """Linear weights are [out, in] in PyTorch and [in, out] in Ferrotensor."""
    if name.endswith(".weight") and grad.dim() == 2:
        return grad.T.contiguous()
    return grad.contiguous()


def main():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    weights = tiny_llama_weights()
    adapters = make_adapters(weights)

    expected = {}
    cases_json = {}
    for case, spec in CASES.items():
        p = {n: torch.tensor(v) for n, v in {**weights, **adapters}.items()}
        trainable = [n for n in p if n.endswith(".lora_a") or n.endswith(".lora_b")]
        if spec["trainable_base"]:
            trainable += [n for n in weights if n != "model.embed_tokens.weight"]
        for n in trainable:
            p[n].requires_grad_(True)

        ids = torch.tensor(spec["rows"], dtype=torch.long)
        targets = torch.tensor(spec["targets"], dtype=torch.long)
        masks = torch.tensor(spec["masks"], dtype=torch.long)
        logits = forward(p, ids)
        loss = loss_fn(logits, targets, masks)
        loss.backward()

        expected[f"logits.{case}"] = logits.detach().contiguous()
        for n in sorted(trainable):
            g = p[n].grad
            assert g is not None and g.abs().max() > 0, f"{case}: zero gradient for {n}"
            expected[f"grad.{case}.{n}"] = ferrotensor_layout(n, g)
        cases_json[case] = dict(spec, loss=loss.item())
        print(f"{case:7s} loss {loss.item():.6f}  logits |max| {logits.abs().max().item():.3f}  "
              f"{len(trainable)} gradients, smallest |max| "
              f"{min(p[n].grad.abs().max().item() for n in trainable):.2e}")

    # Ferrotensor runs the base model without adapters too: base logits for
    # the first row, for the CPU-reference check in model_correctness.rs.
    with torch.no_grad():
        p = {n: torch.tensor(v) for n, v in weights.items()}
        p.update({n: torch.zeros_like(torch.tensor(v)) for n, v in adapters.items()})
        expected["logits.base"] = forward(p, torch.tensor([ROW0])).contiguous()

    OUT.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "intermediate_size": FFN,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "num_hidden_layers": LAYERS,
        "vocab_size": VOCAB,
        "rms_norm_eps": EPS,
        "rope_theta": THETA,
        "tie_word_embeddings": False,
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file({n: torch.tensor(v) for n, v in weights.items()}, OUT / "model.safetensors")
    save_file({n: torch.tensor(v) for n, v in adapters.items()}, OUT / "adapters.safetensors")
    save_file(expected, OUT / "expected.safetensors")
    meta = {
        "generator": "scripts/gen_gradient_fixtures.py",
        "torch": torch.__version__,
        "lora_rank": RANK,
        "lora_alpha": ALPHA,
        "cases": cases_json,
    }
    (OUT / "cases.json").write_text(json.dumps(meta, indent=2) + "\n")
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"wrote {OUT} ({total / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
