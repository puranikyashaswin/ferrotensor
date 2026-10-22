"""Train a small MLP to separate two interleaved spirals.

python examples/mlp.py            # CPU
python examples/mlp.py --device gpu
"""

from __future__ import annotations

import argparse
import math
import os
import random
import tempfile

import ferrotensor as ft
import ferrotensor.nn.functional as F
from ferrotensor import nn, optim


def spirals(n_per_class: int, seed: int = 0) -> tuple[list[list[float]], list[int]]:
    rng = random.Random(seed)
    points, labels = [], []
    for label in (0, 1):
        for i in range(n_per_class):
            r = 0.1 + 2.0 * i / n_per_class
            theta = 2.5 * math.pi * i / n_per_class + label * math.pi
            noise = (rng.gauss(0, 0.05), rng.gauss(0, 0.05))
            points.append([r * math.cos(theta) + noise[0], r * math.sin(theta) + noise[1]])
            labels.append(label)
    return points, labels


class MLP(nn.Module):
    def __init__(self, hidden: int, device: ft.Device) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden, device=device),
            nn.Tanh(),
            nn.Linear(hidden, hidden, device=device),
            nn.Tanh(),
            nn.Linear(hidden, 2, device=device),
        )

    def forward(self, x: ft.Tensor) -> ft.Tensor:
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu, gpu or best")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    device = ft.Device(args.device)
    points, labels = spirals(100)
    x = ft.tensor(points, device=device)

    model = MLP(32, device)
    opt = optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    sched = optim.WarmupCosineLR(opt, warmup_steps=10, total_steps=args.steps, min_lr=0.002)
    print(f"device={device!r} parameters={model.num_parameters()}")

    for step in range(args.steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), labels)
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss {loss.item():.4f}  lr {opt.lr:.4f}")

    model.eval()
    with ft.no_grad():
        acc = F.accuracy(model(x), labels)
    print(f"train accuracy {acc:.3f}")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mlp.safetensors")
        model.save(path)
        restored = MLP(32, device)
        restored.load(path)
        with ft.no_grad():
            print(f"reloaded accuracy {F.accuracy(restored(x), labels):.3f}")


if __name__ == "__main__":
    main()
