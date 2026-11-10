#!/usr/bin/env python3
"""Check a Ferrotensor LoRA adapter exported to PEFT against PyTorch + PEFT.

Loads the base model with transformers, applies the exported adapter with
peft.PeftModel.from_pretrained, runs the token ids stored in the logits file
and compares the logits with the ones Ferrotensor computed for the same
adapter. Exits non-zero when they differ by more than --tolerance, when the
adapter does not load completely, or when the adapter barely changes the
logits (the comparison would then prove nothing).

    cargo run --release -p ferrotensor-trainer --example peft_fixture -- out
    cargo run --release -p ferrotensor-cli -- export-adapter \\
        --lora out/adapter.safetensors --model out/model --output out/peft
    python scripts/check_peft_export.py \\
        --model out/model --adapter out/peft --logits out/logits.json

Needs torch, transformers, peft and safetensors (CI pins torch 2.14.1 CPU,
transformers 5.20.2, peft 0.21.2, safetensors 0.8.0).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="base model directory")
    parser.add_argument("--adapter", required=True, help="PEFT adapter directory")
    parser.add_argument("--logits", required=True, help="logits.json written by Ferrotensor")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    expected = json.loads(Path(args.logits).read_text())
    ids = expected["input_ids"]
    vocab = expected["vocab_size"]
    reference = torch.tensor(expected["logits"], dtype=torch.float32).reshape(len(ids), vocab)

    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="eager"
    )
    base.eval()
    input_ids = torch.tensor([ids])
    with torch.no_grad():
        base_logits = base(input_ids).logits[0]

    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    # Every tensor of the file reached the model, and the model has no
    # adapter tensor the file lacks.
    saved = load_file(str(Path(args.adapter) / "adapter_model.safetensors"))
    # save_embedding_layers=False: with lm_head among the targets PEFT would
    # otherwise add the base lm_head weight, which is not part of an adapter.
    loaded = get_peft_model_state_dict(model, save_embedding_layers=False)
    if set(saved) != set(loaded):
        fail(
            "adapter keys differ: only in file "
            f"{sorted(set(saved) - set(loaded))}, only in model {sorted(set(loaded) - set(saved))}"
        )
    for name, tensor in saved.items():
        if tensor.shape != loaded[name].shape or not torch.equal(tensor, loaded[name].cpu()):
            fail(f"{name} was not loaded as stored")
    print(f"peft loaded all {len(saved)} adapter tensors from {args.adapter}")

    with torch.no_grad():
        logits = model(input_ids).logits[0]
    diff = (logits - reference).abs().max().item()
    base_diff = (base_logits - reference).abs().max().item()
    print(
        f"max |logits(PyTorch + PEFT) - logits(Ferrotensor)| = {diff:.3e} "
        f"over {reference.numel()} values (tolerance {args.tolerance:g})"
    )
    print(f"without the adapter the difference is {base_diff:.3e}")
    if base_diff <= 10 * args.tolerance:
        fail("the adapter changes the logits too little for this comparison to mean anything")
    if diff > args.tolerance:
        fail("PyTorch + PEFT and Ferrotensor disagree")
    print("OK")


if __name__ == "__main__":
    main()
