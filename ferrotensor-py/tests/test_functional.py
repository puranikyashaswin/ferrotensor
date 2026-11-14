import math

import pytest

import ferrotensor as ft
import ferrotensor.nn.functional as F
from conftest import close


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


def test_activation_values():
    x = ft.tensor([-1.0, 1.0, 3.0])
    close(F.sigmoid(x).tolist(), [0.26894142, 0.73105858, 0.95257413])
    close(F.tanh(x).tolist(), [-0.76159416, 0.76159416, 0.99505475])
    close(F.silu(x).tolist(), [-0.26894142, 0.73105858, 2.85772238])
    close(F.gelu(x).tolist(), [-0.15880801, 0.84119199, 2.99636261], tol=1e-3)
    close(F.softmax(ft.tensor([[1.0, 2.0, 3.0]])).tolist(), [[0.09003057, 0.24472847, 0.66524096]])
    close(F.log_softmax(ft.tensor([[1.0, 2.0, 3.0]]), dim=0).tolist(), [[0.0, 0.0, 0.0]])
    close(F.leaky_relu(x).tolist(), [-0.01, 1.0, 3.0])


def test_layer_norm_values():
    x = ft.tensor([[1.0, 2.0, 3.0]])
    unit = 1 / math.sqrt(2 / 3)
    close(F.layer_norm(x).tolist(), [[-unit, 0.0, unit]], tol=1e-4)
    out = F.layer_norm(x, ft.tensor([2.0, 2.0, 2.0]), ft.tensor([1.0, 1.0, 1.0]))
    close(out.tolist(), [[1 - 2 * unit, 1.0, 1 + 2 * unit]], tol=1e-4)
    # A large eps pulls the result towards zero.
    assert abs(F.layer_norm(x, eps=100.0).tolist()[0][2]) < 0.2


def test_rms_norm_values():
    rms = math.sqrt(12.5)
    out = F.rms_norm(ft.tensor([[3.0, 4.0]]), ft.tensor([1.0, 2.0]))
    close(out.tolist(), [[3 / rms, 2 * 4 / rms]], tol=1e-4)
    close(F.rms_norm(ft.tensor([[3.0, 4.0]])).tolist(), [[3 / rms, 4 / rms]], tol=1e-4)


def test_linear_values():
    weight = ft.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    bias = ft.tensor([0.5, 0.5, 0.5])
    assert F.linear(ft.tensor([[1.0, 2.0]]), weight, bias).tolist() == [[1.5, 2.5, 3.5]]
    assert F.linear(ft.tensor([[1.0, 2.0]]), weight).tolist() == [[1.0, 2.0, 3.0]]
    batched = F.linear(ft.ones(2, 3, 2), weight)
    assert batched.shape == (2, 3, 3)
    assert batched.tolist()[1][2] == [1.0, 1.0, 2.0]


def test_scaled_dot_product_attention_values():
    q = ft.tensor([[1.0, 0.0]])
    k = ft.tensor([[1.0, 0.0], [0.0, 1.0]])
    v = ft.tensor([[10.0, 0.0], [0.0, 10.0]])
    first = 1 / (1 + math.exp(-1 / math.sqrt(2)))
    close(F.scaled_dot_product_attention(q, k, v).tolist(), [[10 * first, 10 * (1 - first)]], tol=1e-4)

    qk = ft.tensor([[1.0, 0.0], [0.0, 1.0]])
    values = ft.tensor([[1.0, 2.0], [3.0, 4.0]])
    causal = F.scaled_dot_product_attention(qk, qk, values, is_causal=True).tolist()
    close(causal[0], [1.0, 2.0])  # the first position only sees itself
    second = 1 / (1 + math.exp(-1 / math.sqrt(2)))  # weight of the second position
    close(causal[1], [1 * (1 - second) + 3 * second, 2 * (1 - second) + 4 * second], tol=1e-4)
    # Without the mask the first position also attends to the second.
    assert F.scaled_dot_product_attention(qk, qk, values).tolist()[0] != pytest.approx([1.0, 2.0])


def test_cosine_similarity_values():
    a = ft.tensor([[1.0, 0.0], [1.0, 1.0]])
    b = ft.tensor([[0.0, 1.0], [1.0, 1.0]])
    close(F.cosine_similarity(a, b).tolist(), [0.0, 1.0], tol=1e-5)
    close(F.cosine_similarity(a, b, dim=0).tolist(), [1 / math.sqrt(2)] * 2, tol=1e-5)
    close(F.cosine_similarity(a, -a).tolist(), [-1.0, -1.0], tol=1e-5)


def test_smooth_l1_beta_and_binary_losses():
    zero = ft.zeros(3)
    target = ft.tensor([0.5, 3.0, -4.0])
    # |d| < beta: 0.5 d^2 / beta, else |d| - 0.5 beta
    expected = (0.5 * 0.25 / 2 + (3 - 1) + (4 - 1)) / 3
    assert F.smooth_l1_loss(zero, target, beta=2.0).item() == pytest.approx(expected, rel=1e-5)
    p = ft.tensor([0.9, 0.2])
    assert F.binary_cross_entropy(p, [1.0, 0.0]).item() == pytest.approx(
        -(math.log(0.9) + math.log(0.8)) / 2, rel=1e-4
    )
    logits = ft.tensor([2.0, -1.0])
    assert F.binary_cross_entropy_with_logits(logits, [1.0, 0.0]).item() == pytest.approx(
        (math.log1p(math.exp(-2.0)) + math.log1p(math.exp(-1.0))) / 2, rel=1e-4
    )


def test_nll_loss_ignores_negative_targets():
    logp = ft.tensor([[0.0, 0.0], [0.0, 2.0]]).log_softmax(-1)
    assert F.nll_loss(logp, [1, -100]).item() == pytest.approx(math.log(2), rel=1e-5)
    assert F.nll_loss(logp, ft.tensor([0, 1])).item() == pytest.approx(
        (math.log(2) + math.log(1 + math.exp(-2))) / 2, rel=1e-4
    )


def test_cross_entropy_matches_nll_of_log_softmax():
    logits = ft.tensor([[0.3, -1.0, 2.0], [1.0, 1.0, 0.0]])
    target = [2, 0]
    ce = F.cross_entropy(logits, target).item()
    assert ce == pytest.approx(F.nll_loss(logits.log_softmax(-1), target).item(), rel=1e-5)


def test_one_hot_with_tensor_indices():
    assert F.one_hot(ft.tensor([2, 0]), 3).tolist() == [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    assert F.one_hot([1], 2, device="cpu").tolist() == [[0.0, 1.0]]


def test_dropout_functional_in_training():
    y = F.dropout(ft.ones(1000), 0.5)
    assert set(y.tolist()) <= {0.0, 2.0}
    assert 350 < y.tolist().count(0.0) < 650
    assert F.dropout(ft.ones(4), 0.25, training=False).tolist() == [1.0] * 4


def test_loss_functions_return_scalars_that_keep_grad():
    p = ft.ones(2, 3, requires_grad=True)
    for loss in (
        F.mse_loss(p, ft.zeros(2, 3)),
        F.l1_loss(p, ft.zeros(2, 3)),
        F.smooth_l1_loss(p, ft.zeros(2, 3)),
        F.cross_entropy(p, [0, 1]),
        F.binary_cross_entropy_with_logits(p, ft.zeros(2, 3)),
    ):
        assert loss.numel() == 1 and loss.requires_grad
