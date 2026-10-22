"""Bad input raises Python exceptions, never a Rust panic."""

import pytest

import ferrotensor as ft
import ferrotensor.nn.functional as F
from ferrotensor import nn, optim


@pytest.mark.parametrize(
    "fn, exc, match",
    [
        (lambda: ft.tensor([[1, 2], [3]]), ValueError, "rectangular"),
        (lambda: ft.tensor([1, [2]]), ValueError, "rectangular"),
        (lambda: ft.tensor(["a"]), TypeError, "number"),
        (lambda: ft.tensor({"a": 1}), TypeError, "number"),
        (lambda: ft.zeros(-1), ValueError, "negative"),
        (lambda: ft.ones(2, 3) + ft.ones(4), ValueError, "broadcast"),
        (lambda: ft.ones(2, 3) @ ft.ones(2, 3), ValueError, "inner dimensions"),
        (lambda: ft.ones(2, 2, 2) @ ft.ones(3, 2, 2), ValueError, "batch"),
        (lambda: ft.tensor(1.0) @ ft.ones(1), ValueError, "at least 1"),
        (lambda: ft.ones(2, 3).reshape(4), ValueError, "invalid"),
        (lambda: ft.ones(2, 3).reshape(-1, -1), ValueError, "only one"),
        (lambda: ft.ones(2).sum(1), IndexError, "out of range"),
        (lambda: ft.ones(2).softmax(3), IndexError, "out of range"),
        (lambda: ft.ones(2, 2).permute(0, 0), ValueError, "permutation"),
        (lambda: ft.ones(2, 3).expand(4, 3), ValueError, "expand"),
        (lambda: ft.ones(3)[3], IndexError, "out of bounds"),
        (lambda: ft.ones(3)[0, 0], IndexError, "too many"),
        (lambda: ft.ones(3)["x"], TypeError, "indices"),
        (lambda: ft.ones(3).narrow(0, 2, 2), IndexError, "out of range"),
        (lambda: ft.ones(3).index_select(0, [3]), IndexError, "out of bounds"),
        (lambda: ft.ones(2).item(), ValueError, "one-element"),
        (lambda: ft.ones(2, requires_grad=True).backward(), ValueError, "one-element"),
        (lambda: ft.ones(2, requires_grad=True).backward(ft.ones(3)), ValueError, "shape"),
        (lambda: ft.ones(2).masked_fill(ft.ones(2), float("-inf")), ValueError, "finite"),
        (lambda: ft.ones(2).clamp(1, 0), ValueError, "greater"),
        (lambda: ft.cat([]), ValueError, "non-empty"),
        (lambda: ft.cat([ft.ones(2, 2), ft.ones(2, 3)]), ValueError, "differ"),
        (lambda: ft.stack([ft.ones(2), ft.ones(3)]), ValueError, "same shape"),
        (lambda: ft.arange(0, 1, 0), ValueError, "step"),
        (lambda: ft.Device("tpu"), ValueError, "unknown device"),
        (lambda: ft.zeros(1, device=3), TypeError, "device"),
        (lambda: len(ft.tensor(1.0)), TypeError, "0-d"),
        (lambda: nn.Linear(3, 2)(ft.ones(2, 4)), ValueError, "last dimension 3"),
        (lambda: nn.Linear(0, 2), ValueError, "positive"),
        (lambda: nn.Embedding(5, 2)([5]), IndexError, "embedding id"),
        (lambda: nn.Embedding(5, 2)([1.5]), IndexError, "embedding id"),
        (lambda: nn.LayerNorm(4)(ft.ones(2, 3)), ValueError, "trailing"),
        (lambda: nn.Dropout(1.0), ValueError, "probability"),
        (lambda: nn.MultiHeadAttention(6, 4), ValueError, "multiple"),
        (lambda: nn.MultiHeadAttention(8, 2)(ft.ones(8)), ValueError, "seq"),
        (lambda: F.mse_loss(ft.ones(2), ft.ones(3)), ValueError, "does not match"),
        (lambda: F.cross_entropy(ft.ones(2, 3), [0, 3]), IndexError, "class index"),
        (lambda: F.cross_entropy(ft.ones(2, 3), [0, 1, 2]), ValueError, "target shape"),
        (lambda: F.cross_entropy(ft.ones(2, 0), [0, 1]), ValueError, "class dimension"),
        (lambda: F.linear(ft.ones(2, 3), ft.ones(2, 2)), ValueError, "linear"),
        (
            lambda: F.scaled_dot_product_attention(ft.ones(2, 3), ft.ones(2, 4), ft.ones(2, 4)),
            ValueError,
            "leading",
        ),
        (lambda: optim.SGD([], lr=0.1), ValueError, "empty"),
        (lambda: optim.SGD([ft.ones(1)], lr=0.1), ValueError, "requires_grad"),
        (lambda: optim.SGD([ft.ones(1, requires_grad=True)], lr=-1), ValueError, "learning rate"),
        (lambda: optim.SGD([ft.ones(1, requires_grad=True)], lr=0.1, nesterov=True), ValueError, "nesterov"),
        (lambda: optim.AdamW([ft.ones(1, requires_grad=True)], betas=(1.0, 0.9)), ValueError, "betas"),
        (lambda: optim.clip_grad_norm([], 0.0), ValueError, "positive"),
        (lambda: optim.StepLR(None, step_size=1), ValueError, "base_lr"),
        (lambda: optim.StepLR(None, step_size=0, base_lr=1.0), ValueError, "step_size"),
    ],
)
def test_raises(fn, exc, match):
    with pytest.raises(exc, match=match):
        fn()


def test_errors_are_not_panics():
    # PanicException derives from BaseException, not Exception: make sure a
    # mismatch caught as Exception really is a normal Python error.
    try:
        ft.ones(2, 3) @ ft.ones(2, 3)
    except Exception as e:  # noqa: BLE001
        assert type(e).__name__ == "ValueError"
