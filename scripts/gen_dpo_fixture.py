#!/usr/bin/env python3
"""Generate the DPO parity reference for Ferrotensor's LoRA training path.

Uses the tiny Llama-style model and LoRA adapters of the gradient-parity
fixture (`ferrotensor-graph/tests/fixtures/gradient_parity`, written by
`gen_gradient_fixtures.py`, whose PyTorch forward this script imports) and
computes, with PyTorch autograd in float32:

- the reference log-probability of each sequence: the same model with every
  LoRA B set to zero (the adapters off), under `torch.no_grad()`;
- the policy log-probability of each sequence: the model with its adapters;
- the DPO loss, the mean over pairs of
  `-log sigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)))`,
  where a sequence's log-probability is the sum of its target
  log-probabilities over the positions whose mask is 1;
- the gradient of that loss for every LoRA factor.

`ferrotensor-trainer/tests/dpo.rs` runs the same pairs on the GPU and
compares. Nothing is committed: CI writes the output to a temporary
directory and passes it to the test in `FERROTENSOR_DPO_FIXTURE`.

Usage (from the repository root, with torch, safetensors and numpy):

    python scripts/gen_dpo_fixture.py OUT_DIR

Output in OUT_DIR:
    cases.json             rows, targets, masks, beta, loss and log-probs
    expected.safetensors   `grad.<case>.<name>` in Ferrotensor layout
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_gradient_fixtures as g  # noqa: E402

FIXTURE = g.OUT

# Sequences alternate chosen and rejected: pair i is rows 2i and 2i + 1.
# Each pair shares a prompt; the mask marks the response positions, as the
# chat template masking of `finetune` does.
PROMPT = [3, 17, 5]
CHOSEN0 = PROMPT + [29, 0, 11, 23, 8]
REJECTED0 = PROMPT + [12, 9, 30, 4, 4]
CHOSEN1 = [30, 2, 19, 7, 7, 26, 1, 14]
REJECTED1 = [30, 2, 19, 21, 6, 16, 25, 14]


def shifted(row, last):
    return row[1:] + [last]


CASES = {
    # One pair, beta 0.1 (the default).
    "pair": dict(
        rows=[CHOSEN0, REJECTED0],
        targets=[shifted(CHOSEN0, 12), shifted(REJECTED0, 2)],
        masks=[[0, 0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1, 0]],
        beta=0.1,
    ),
    # Two pairs of different response lengths, beta 0.5.
    "batch": dict(
        rows=[CHOSEN0, REJECTED0, CHOSEN1, REJECTED1],
        targets=[shifted(CHOSEN0, 12), shifted(REJECTED0, 2),
                 shifted(CHOSEN1, 31), shifted(REJECTED1, 31)],
        masks=[[0, 0, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 1, 0],
               [0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]],
        beta=0.5,
    ),
}


def sequence_log_probs(logits, targets, masks):
    logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (logp * masks.to(logp.dtype)).sum(-1)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: gen_dpo_fixture.py OUT_DIR")
    out = Path(sys.argv[1])
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    weights = load_file(FIXTURE / "model.safetensors")
    adapters = load_file(FIXTURE / "adapters.safetensors")
    lora_meta = json.loads((FIXTURE / "cases.json").read_text())

    expected = {}
    cases_json = {}
    for case, spec in CASES.items():
        ids = torch.tensor(spec["rows"], dtype=torch.long)
        targets = torch.tensor(spec["targets"], dtype=torch.long)
        masks = torch.tensor(spec["masks"], dtype=torch.long)

        with torch.no_grad():
            ref = {**weights, **adapters}
            ref.update({n: torch.zeros_like(v) for n, v in adapters.items() if n.endswith(".lora_b")})
            ref_logp = sequence_log_probs(g.forward(ref, ids), targets, masks)

        p = {**weights, **{n: v.clone().requires_grad_(True) for n, v in adapters.items()}}
        pi_logp = sequence_log_probs(g.forward(p, ids), targets, masks)
        beta = spec["beta"]
        z = beta * ((pi_logp[0::2] - ref_logp[0::2]) - (pi_logp[1::2] - ref_logp[1::2]))
        loss = -torch.nn.functional.logsigmoid(z).mean()
        loss.backward()

        for n in sorted(adapters):
            grad = p[n].grad
            assert grad is not None and grad.abs().max() > 0, f"{case}: zero gradient for {n}"
            expected[f"grad.{case}.{n}"] = g.ferrotensor_layout(n, grad)
        cases_json[case] = dict(
            spec,
            loss=loss.item(),
            policy_log_probs=pi_logp.tolist(),
            reference_log_probs=ref_logp.tolist(),
        )
        print(f"{case:6s} loss {loss.item():.6f}  margins {z.tolist()}  "
              f"smallest gradient |max| {min(p[n].grad.abs().max().item() for n in adapters):.2e}")

    out.mkdir(parents=True, exist_ok=True)
    save_file(expected, out / "expected.safetensors")
    meta = {
        "generator": "scripts/gen_dpo_fixture.py",
        "torch": torch.__version__,
        "lora_rank": lora_meta["lora_rank"],
        "lora_alpha": lora_meta["lora_alpha"],
        "cases": cases_json,
    }
    (out / "cases.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
