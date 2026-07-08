#!/usr/bin/env python3
"""
Generate Phase 0 linear/RMSNorm fixtures without PyTorch.

Uses a deterministic LCG so fixtures can be regenerated on any machine.
Produces the same schema as generate_phase0_reference.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


class Lcg:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
        return self.state

    def randn(self) -> float:
        # Box-Muller
        u1 = (self.next_u64() / 2**64) or 1e-12
        u2 = self.next_u64() / 2**64
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def reshape_3d(flat: list[float], b: int, s: int, h: int) -> list[list[list[float]]]:
    out: list[list[list[float]]] = []
    idx = 0
    for _ in range(b):
        batch = []
        for _ in range(s):
            row = flat[idx : idx + h]
            idx += h
            batch.append(row)
        out.append(batch)
    return out


def main() -> None:
    seed = 7
    batch, seq, hidden, out_features = 2, 3, 4, 5
    eps = 1e-5
    rng = Lcg(seed)

    x_flat = [rng.randn() for _ in range(batch * seq * hidden)]
    weight = [[rng.randn() for _ in range(hidden)] for _ in range(out_features)]  # HF [out, in]
    rms_weight = [rng.randn() for _ in range(hidden)]

    # Linear: flatten [B*S, H] @ W^T where W is HF [out, in]
    linear_flat: list[float] = []
    for row in range(batch * seq):
        base = row * hidden
        for o in range(out_features):
            acc = 0.0
            for i in range(hidden):
                acc += x_flat[base + i] * weight[o][i]
            linear_flat.append(acc)
    linear_3d = reshape_3d(linear_flat, batch, seq, out_features)

    # RMSNorm over last dim
    rms_out: list[list[list[float]]] = []
    for b in range(batch):
        batch_rows = []
        for s in range(seq):
            start = (b * seq + s) * hidden
            vals = x_flat[start : start + hidden]
            mean_sq = sum(v * v for v in vals) / hidden
            inv_rms = 1.0 / math.sqrt(mean_sq + eps)
            row = [vals[i] * inv_rms * rms_weight[i] for i in range(hidden)]
            batch_rows.append(row)
        rms_out.append(batch_rows)

    payload = {
        "meta": {
            "batch": batch,
            "seq": seq,
            "hidden": hidden,
            "out_features": out_features,
            "eps": eps,
            "seed": seed,
            "generator": "pure-python-lcg",
        },
        "input": reshape_3d(x_flat, batch, seq, hidden),
        "linear_weight_hf_layout": weight,
        "linear_output": linear_3d,
        "rms_weight": rms_weight,
        "rms_output": rms_out,
    }

    out_path = Path(__file__).with_name("phase0_linear_rmsnorm.json")
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
