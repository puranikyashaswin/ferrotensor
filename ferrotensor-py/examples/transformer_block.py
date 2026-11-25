"""Train one pre-norm transformer block to reverse short sequences of token ids.

python examples/transformer_block.py            # CPU
python examples/transformer_block.py --device gpu

Position i has to output the token at position SEQ - 1 - i, which only
attention can move there: the MLP sees one position at a time. The script exits
with an error if the loss does not fall or if the model does not reverse
sequences it was not trained on.
"""

from __future__ import annotations

import argparse
import random

import ferrotensor as ft
import ferrotensor.nn.functional as F
from ferrotensor import nn, optim

VOCAB = 8
SEQ = 6


def batch(rng: random.Random, size: int) -> tuple[list[list[int]], list[list[int]]]:
    tokens = [[rng.randrange(VOCAB) for _ in range(SEQ)] for _ in range(size)]
    return tokens, [row[::-1] for row in tokens]


class Block(nn.Module):
    """x + attention(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, dim: int, heads: int, device: ft.Device) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, device=device)
        self.attn = nn.MultiHeadAttention(dim, heads, device=device)
        self.norm2 = nn.LayerNorm(dim, device=device)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim, device=device),
            nn.GELU(),
            nn.Linear(4 * dim, dim, device=device),
        )

    def forward(self, x: ft.Tensor) -> ft.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Reverser(nn.Module):
    def __init__(self, dim: int, heads: int, device: ft.Device) -> None:
        super().__init__()
        self.tokens = nn.Embedding(VOCAB, dim, device=device)
        self.positions = nn.Embedding(SEQ, dim, device=device)
        self.block = Block(dim, heads, device)
        self.norm = nn.LayerNorm(dim, device=device)
        self.head = nn.Linear(dim, VOCAB, device=device)

    def forward(self, tokens: list[list[int]]) -> ft.Tensor:
        x = self.tokens(tokens) + self.positions(list(range(SEQ)))
        return self.head(self.norm(self.block(x)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu, gpu or best")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    ft.manual_seed(0)
    rng = random.Random(0)
    device = ft.Device(args.device)
    model = Reverser(32, 4, device)
    opt = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0)
    print(f"device={device!r} parameters={model.num_parameters()}")

    first_loss = last_loss = float("nan")
    for step in range(args.steps):
        tokens, targets = batch(rng, 32)
        opt.zero_grad()
        loss = F.cross_entropy(model(tokens), targets)
        loss.backward()
        opt.step()
        last_loss = loss.item()
        if step == 0:
            first_loss = last_loss
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss {last_loss:.4f}")

    tokens, targets = batch(random.Random(1), 256)
    model.eval()
    with ft.no_grad():
        logits = model(tokens).reshape(256 * SEQ, VOCAB)
        acc = F.accuracy(logits, [t for row in targets for t in row])
    print(f"loss {first_loss:.4f} -> {last_loss:.4f}  held-out token accuracy {acc:.3f}")
    assert last_loss < 0.5 * first_loss, f"loss did not fall: {first_loss} -> {last_loss}"
    assert acc > 0.9, f"held-out token accuracy is only {acc:.3f}"


if __name__ == "__main__":
    main()
