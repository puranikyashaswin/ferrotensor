#!/usr/bin/env python3
"""Record transformers' float32 logits of a real Llama 3 checkpoint on 8 prompts.

The output directory holds `cases.json` (the token ids of each prompt, from
the checkpoint's own tokenizer, with the BOS token the tokenizer adds) and
`logits.safetensors` (`logits.<n>`: the `[tokens, vocab]` logits of prompt n).
`ferrotensor-trainer/tests/llama3_rope.rs` compares Ferrotensor's logits with
these.

Usage (torch and transformers installed):

    python scripts/gen_llama3_real_logits.py MODEL_DIR OUT_DIR
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def main():
    model_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
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


if __name__ == "__main__":
    main()
