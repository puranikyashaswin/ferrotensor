"""End to end: train a small MLP on a toy problem on the CPU."""

import math
import random

import ferrotensor as ft
import ferrotensor.nn.functional as F
from ferrotensor import nn, optim


def make_spirals(n_per_class=60, seed=0):
    rng = random.Random(seed)
    xs, ys = [], []
    for label in range(2):
        for i in range(n_per_class):
            r = i / n_per_class * 2.0 + 0.1
            theta = i / n_per_class * 2.5 * math.pi + label * math.pi
            xs.append([r * math.cos(theta) + rng.gauss(0, 0.05), r * math.sin(theta) + rng.gauss(0, 0.05)])
            ys.append(label)
    return ft.tensor(xs), ys


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


def test_mlp_loss_drops_and_fits(tmp_path):
    x, y = make_spirals()
    model = MLP()
    opt = optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    sched = optim.WarmupCosineLR(opt, warmup_steps=10, total_steps=300, min_lr=0.002)

    losses = []
    for _ in range(300):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])
    model.eval()
    with ft.no_grad():
        acc = F.accuracy(model(x), y)
    assert acc > 0.9, acc

    # The trained weights survive a save/load round trip.
    path = tmp_path / "mlp.safetensors"
    model.save(path)
    fresh = MLP()
    fresh.load(path)
    with ft.no_grad():
        assert F.accuracy(fresh(x), y) == acc


def test_regression_with_sgd():
    xs = [[i / 10] for i in range(-10, 11)]
    x = ft.tensor(xs)
    y = x * 3.0 + 0.5
    model = nn.Linear(1, 1)
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 1e-4 < first
    assert model.weight.item() == __import__("pytest").approx(3.0, abs=1e-2)
    assert model.bias.item() == __import__("pytest").approx(0.5, abs=1e-2)
