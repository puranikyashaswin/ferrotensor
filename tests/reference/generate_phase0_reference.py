#!/usr/bin/env python3
"""
Generate tiny PyTorch reference fixtures for Rusty Phase 0 parity checks.

Requires:
  pip install torch
"""

import json
from pathlib import Path

import torch


def main() -> None:
    torch.manual_seed(7)

    batch, seq, hidden, out_features = 2, 3, 4, 5
    eps = 1e-5

    x = torch.randn(batch, seq, hidden, dtype=torch.float32)
    weight = torch.randn(out_features, hidden, dtype=torch.float32)
    rms_weight = torch.randn(hidden, dtype=torch.float32)

    x_2d = x.reshape(batch * seq, hidden)
    linear = x_2d @ weight.t()
    linear_3d = linear.reshape(batch, seq, out_features)

    rms = torch.nn.functional.rms_norm(x, (hidden,), rms_weight, eps=eps)

    payload = {
        "meta": {
            "batch": batch,
            "seq": seq,
            "hidden": hidden,
            "out_features": out_features,
            "eps": eps,
            "seed": 7,
        },
        "input": x.tolist(),
        "linear_weight_hf_layout": weight.tolist(),
        "linear_output": linear_3d.tolist(),
        "rms_weight": rms_weight.tolist(),
        "rms_output": rms.tolist(),
    }

    out_path = Path(__file__).with_name("phase0_linear_rmsnorm.json")
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
