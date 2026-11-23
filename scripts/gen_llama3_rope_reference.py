#!/usr/bin/env python3
"""Print the Llama 3 RoPE inverse frequencies of transformers as Rust arrays.

Three configs go through `ROPE_INIT_FUNCTIONS["llama3"]`, the function
`_compute_llama3_parameters` of transformers' `modeling_rope_utils.py`, in
float32 as the library runs it:

* Llama-3.2-1B: head size 64, base 500000, factor 32, original context 8192
* Llama-3.1-8B: head size 128, base 500000, factor 8, original context 8192
* the tiny test model of `scripts/gen_llama3_fixture.py`: head size 16,
  base 500000, factor 8, original context 64

The output is pasted into the `matches_transformers` test of
`ferrotensor-graph/src/rope.rs`, which compares `Llama3Scaling::inv_freq`
with it.

Usage (torch and transformers installed):

    python scripts/gen_llama3_rope_reference.py
"""

import torch
import transformers
from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

CASES = [
    ("LLAMA_3_2_1B", 2048, 32, 64, 500_000.0, 32.0, 8192),
    ("LLAMA_3_1_8B", 4096, 32, 128, 500_000.0, 8.0, 8192),
    ("TINY", 64, 4, 16, 500_000.0, 8.0, 64),
]


def inv_freq(hidden, heads, head_dim, theta, factor, original):
    scaling = {
        "rope_type": "llama3",
        "factor": factor,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": original,
    }
    config = LlamaConfig(
        hidden_size=hidden,
        num_attention_heads=heads,
        head_dim=head_dim,
        rope_parameters={**scaling, "rope_theta": theta},
    )
    table, attention_factor = ROPE_INIT_FUNCTIONS["llama3"](config, "cpu")
    assert attention_factor == 1.0
    assert table.dtype == torch.float32 and table.shape == (head_dim // 2,)
    return table


def main():
    print(f"// torch {torch.__version__}, transformers {transformers.__version__}")
    for name, hidden, heads, head_dim, theta, factor, original in CASES:
        table = inv_freq(hidden, heads, head_dim, theta, factor, original)
        values = ", ".join(f"{v:.9e}" for v in table.tolist())
        print(f"// head size {head_dim}, base {theta}, factor {factor}, original context {original}")
        print(f"const {name}: [f32; {head_dim // 2}] = [{values}];")


if __name__ == "__main__":
    main()
