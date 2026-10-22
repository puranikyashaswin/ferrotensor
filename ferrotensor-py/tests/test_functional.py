import math

import pytest

import ferrotensor as ft
import ferrotensor.nn.functional as F


def test_regression_losses():
    p = ft.tensor([1.0, 2.0, 3.0])
    t = ft.tensor([1.0, 0.0, 5.0])
    assert F.mse_loss(p, t).item() == pytest.approx(8 / 3)
    assert F.l1_loss(p, t).item() == pytest.approx(4 / 3)
    assert F.mse_loss(p, [1.0, 0.0, 5.0]).item() == pytest.approx(8 / 3)
    assert F.smooth_l1_loss(p, t, beta=1.0).item() == pytest.approx((0 + 1.5 + 1.5) / 3)


def test_cross_entropy_indices_and_probs():
    logits = ft.tensor([[2.0, 0.0], [0.0, 2.0]])
    expected = -math.log(math.exp(2) / (math.exp(2) + 1))
    assert F.cross_entropy(logits, [0, 1]).item() == pytest.approx(expected, rel=1e-5)
    assert F.cross_entropy(logits, ft.tensor([0, 1])).item() == pytest.approx(expected, rel=1e-5)
    # -100 is ignored
    assert F.cross_entropy(logits, [0, -100]).item() == pytest.approx(expected, rel=1e-5)
    probs = ft.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert F.cross_entropy(logits, probs).item() == pytest.approx(expected, rel=1e-5)
    seq = ft.zeros(2, 3, 4)
    assert F.cross_entropy(seq, [[0, 1, 2], [3, 0, 1]]).item() == pytest.approx(math.log(4), rel=1e-5)


def test_nll_and_binary_losses():
    logp = ft.tensor([[0.0, 0.0]]).log_softmax(-1)
    assert F.nll_loss(logp, [1]).item() == pytest.approx(math.log(2), rel=1e-5)
    p = ft.tensor([0.5, 0.5])
    assert F.binary_cross_entropy(p, [1.0, 0.0]).item() == pytest.approx(math.log(2), rel=1e-4)
    assert F.binary_cross_entropy_with_logits(ft.zeros(2), [1.0, 0.0]).item() == pytest.approx(
        math.log(2), rel=1e-4
    )


def test_cross_entropy_backward():
    logits = ft.zeros(1, 3, requires_grad=True)
    F.cross_entropy(logits, [2]).backward()
    assert logits.grad.tolist()[0] == pytest.approx([1 / 3, 1 / 3, -2 / 3], abs=1e-5)


def test_activations_and_norms():
    x = ft.tensor([[-1.0, 1.0, 3.0]])
    assert F.relu(x).tolist() == [[0.0, 1.0, 3.0]]
    assert F.leaky_relu(x, 0.5).tolist() == [[-0.5, 1.0, 3.0]]
    assert F.sigmoid(x).shape == F.tanh(x).shape == F.gelu(x).shape == F.silu(x).shape
    assert sum(F.softmax(x).tolist()[0]) == pytest.approx(1.0)
    assert F.log_softmax(x, dim=-1).exp().sum().item() == pytest.approx(1.0)
    y = F.layer_norm(x)
    assert abs(y.mean().item()) < 1e-5
    assert F.layer_norm(x, ft.ones(3), ft.zeros(3)).shape == (1, 3)
    assert F.rms_norm(x, ft.ones(3)).shape == (1, 3)


def test_dropout_functional():
    x = ft.ones(10)
    assert F.dropout(x, 0.5, training=False).tolist() == [1.0] * 10
    assert F.dropout(x, 0.0).tolist() == [1.0] * 10


def test_attention_and_misc():
    q = ft.randn(2, 4, 8)
    out = F.scaled_dot_product_attention(q, q, ft.randn(2, 4, 3), is_causal=True)
    assert out.shape == (2, 4, 3)
    assert F.one_hot([0, 2], 3).tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    assert F.accuracy(ft.tensor([[1.0, 0.0], [1.0, 0.0]]), [0, 1]) == 0.5
    assert F.accuracy(ft.tensor([[1.0, 0.0], [1.0, 0.0]]), ft.tensor([0, 1])) == 0.5
    assert F.accuracy(ft.tensor([[1.0, 0.0], [1.0, 0.0]]), [0, -100]) == 1.0
    a = ft.tensor([[1.0, 0.0]])
    assert F.cosine_similarity(a, a).tolist() == pytest.approx([1.0])
