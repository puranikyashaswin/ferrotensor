import math

import pytest

import ferrotensor as ft
import ferrotensor.nn.functional as F
from conftest import close
from ferrotensor import nn


def test_backward_simple():
    w = ft.tensor([1.0, 2.0, 3.0], requires_grad=True)
    assert w.is_leaf and w.grad is None and w.grad_fn is None
    loss = (w * w).sum()
    assert loss.requires_grad and not loss.is_leaf and loss.grad_fn is not None
    loss.backward()
    assert w.grad.tolist() == [2.0, 4.0, 6.0]


def test_gradients_accumulate_and_reset():
    w = ft.ones(2, requires_grad=True)
    (w * 3).sum().backward()
    (w * 3).sum().backward()
    assert w.grad.tolist() == [6.0, 6.0]
    w.grad = None
    assert w.grad is None
    w.grad = ft.ones(2)
    assert w.grad.tolist() == [1.0, 1.0]


def test_matmul_and_broadcast_grads():
    a = ft.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = ft.tensor([1.0, -1.0], requires_grad=True)
    ((a @ a) + b).sum().backward()
    # d/da sum(a @ a) = 1 @ a^T + a^T @ 1
    assert a.grad.tolist() == [[7.0, 11.0], [9.0, 13.0]]
    assert b.grad.tolist() == [2.0, 2.0]


def test_backward_with_gradient():
    x = ft.tensor([1.0, 2.0], requires_grad=True)
    y = x * 2
    y.backward(ft.tensor([1.0, 10.0]))
    assert x.grad.tolist() == [2.0, 20.0]


def test_indexing_and_shape_ops_are_differentiable():
    x = ft.arange(6, requires_grad=True)
    y = x.reshape(2, 3)[:, 1:].T
    y.sum().backward()
    assert x.grad.tolist() == [0.0, 1.0, 1.0, 0.0, 1.0, 1.0]


def test_detach_stops_gradient():
    x = ft.ones(2, requires_grad=True)
    y = x.detach() * 3
    assert not y.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        y.sum().backward()


def test_requires_grad_toggles():
    x = ft.ones(2)
    assert x.requires_grad_() is x and x.requires_grad
    x.requires_grad = False
    assert not x.requires_grad
    y = ft.ones(2, requires_grad=True) * 2
    with pytest.raises(RuntimeError, match="leaf"):
        y.requires_grad = False


def test_no_grad_context_and_decorator():
    w = ft.ones(2, requires_grad=True)
    assert ft.is_grad_enabled()
    with ft.no_grad():
        assert not ft.is_grad_enabled()
        with ft.no_grad():
            pass
        assert not ft.is_grad_enabled()
        y = w * 2
    assert ft.is_grad_enabled()
    assert not y.requires_grad

    @ft.no_grad()
    def infer(t):
        return t * 2

    assert not infer(w).requires_grad
    assert ft.is_grad_enabled()


def test_no_grad_restores_on_exception():
    with pytest.raises(KeyError):
        with ft.no_grad():
            raise KeyError("x")
    assert ft.is_grad_enabled()


def test_finite_difference_check():
    def f(t):
        return ((t.tanh() * t).softmax(-1) * ft.tensor([1.0, 2.0, 3.0])).sum()

    vals = [0.3, -0.7, 1.1]
    x = ft.tensor(vals, requires_grad=True)
    f(x).backward()
    eps = 1e-2
    numeric = []
    for i in range(3):
        up = list(vals)
        down = list(vals)
        up[i] += eps
        down[i] -= eps
        numeric.append((f(ft.tensor(up)).item() - f(ft.tensor(down)).item()) / (2 * eps))
    close(x.grad.tolist(), numeric, tol=1e-2)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# name -> (function of a tensor, derivative of that function at x)
UNARY_GRADS = {
    "relu": (lambda t: t.relu(), lambda x: 1.0 if x > 0 else 0.0),
    "sigmoid": (lambda t: t.sigmoid(), lambda x: sigmoid(x) * (1 - sigmoid(x))),
    "tanh": (lambda t: t.tanh(), lambda x: 1 - math.tanh(x) ** 2),
    "silu": (lambda t: t.silu(), lambda x: sigmoid(x) * (1 + x * (1 - sigmoid(x)))),
    "exp": (lambda t: t.exp(), math.exp),
    "log": (lambda t: t.log(), lambda x: 1 / x),
    "sqrt": (lambda t: t.sqrt(), lambda x: 0.5 / math.sqrt(x)),
    "rsqrt": (lambda t: t.rsqrt(), lambda x: -0.5 * x**-1.5),
    "square": (lambda t: t.square(), lambda x: 2 * x),
    "reciprocal": (lambda t: t.reciprocal(), lambda x: -1 / x**2),
    "sin": (lambda t: t.sin(), math.cos),
    "cos": (lambda t: t.cos(), lambda x: -math.sin(x)),
    "abs": (lambda t: t.abs(), lambda x: 1.0 if x > 0 else -1.0),
    "neg": (lambda t: -t, lambda x: -1.0),
    "pow3": (lambda t: t**3, lambda x: 3 * x**2),
    "leaky_relu": (lambda t: t.leaky_relu(0.1), lambda x: 1.0 if x > 0 else 0.1),
    "clamp": (lambda t: t.clamp(0.8, 1.7), lambda x: 1.0 if 0.8 < x < 1.7 else 0.0),
}


@pytest.mark.parametrize("name", list(UNARY_GRADS))
def test_unary_gradient_matches_the_derivative(name):
    fn, deriv = UNARY_GRADS[name]
    points = [0.5, 1.5, 2.0]
    x = ft.tensor(points, requires_grad=True)
    fn(x).sum().backward()
    close(x.grad.tolist(), [deriv(p) for p in points], tol=1e-4)


def test_gradients_at_negative_inputs():
    x = ft.tensor([-2.0, -0.5], requires_grad=True)
    (x.relu() + x.abs() + x.leaky_relu(0.1)).sum().backward()
    close(x.grad.tolist(), [0.0 - 1.0 + 0.1] * 2)


def test_gelu_gradient_matches_finite_differences():
    points = [-1.0, 0.3, 1.5]
    x = ft.tensor(points, requires_grad=True)
    x.gelu().sum().backward()
    eps = 1e-2
    numeric = [
        (ft.tensor([p + eps]).gelu().item() - ft.tensor([p - eps]).gelu().item()) / (2 * eps) for p in points
    ]
    close(x.grad.tolist(), numeric, tol=1e-2)


def test_binary_op_gradients():
    a = ft.tensor([2.0, 6.0], requires_grad=True)
    b = ft.tensor([4.0, 3.0], requires_grad=True)
    (a / b).sum().backward()
    close(a.grad.tolist(), [0.25, 1 / 3])
    close(b.grad.tolist(), [-2 / 16, -6 / 9])

    a, b = ft.tensor([1.0, 5.0], requires_grad=True), ft.tensor([3.0, 2.0], requires_grad=True)
    a.maximum(b).sum().backward()
    assert a.grad.tolist() == [0.0, 1.0] and b.grad.tolist() == [1.0, 0.0]
    a, b = ft.tensor([1.0, 5.0], requires_grad=True), ft.tensor([3.0, 2.0], requires_grad=True)
    a.minimum(b).sum().backward()
    assert a.grad.tolist() == [1.0, 0.0] and b.grad.tolist() == [0.0, 1.0]

    a = ft.tensor([1.0, 2.0], requires_grad=True)
    ((a - 1) * 3 + 2 / a).sum().backward()
    close(a.grad.tolist(), [3 - 2.0, 3 - 0.5])


def test_reduction_gradients():
    x = ft.tensor([1.0, 2.0, 3.0, 6.0], requires_grad=True)
    x.mean().backward()
    assert x.grad.tolist() == [0.25] * 4
    x.grad = None
    x.var().backward()
    close(x.grad.tolist(), [-1.0, -0.5, 0.0, 1.5])

    m = ft.tensor([[1.0, 5.0], [7.0, 2.0]], requires_grad=True)
    m.max().backward()
    assert m.grad.tolist() == [[0.0, 0.0], [1.0, 0.0]]
    m.grad = None
    m.max(1).sum().backward()
    assert m.grad.tolist() == [[0.0, 1.0], [1.0, 0.0]]
    m.grad = None
    m.min(0).sum().backward()
    assert m.grad.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_softmax_gradients():
    x = ft.tensor([0.2, -0.4, 1.1], requires_grad=True)
    x.softmax(-1).sum().backward()
    close(x.grad.tolist(), [0.0, 0.0, 0.0], tol=1e-6)

    y = ft.tensor([0.0, 1.0], requires_grad=True)
    y.log_softmax(-1).sum().backward()
    soft = [sigmoid(-1.0), sigmoid(1.0)]
    close(y.grad.tolist(), [1 - 2 * soft[0], 1 - 2 * soft[1]])


def test_matmul_gradients_for_vectors_and_batches():
    a = ft.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    v = ft.tensor([2.0, 3.0], requires_grad=True)
    (v @ a).sum().backward()
    assert v.grad.tolist() == [3.0, 7.0]
    assert a.grad.tolist() == [[2.0, 2.0], [3.0, 3.0]]

    a.grad = None
    v.grad = None
    (a @ v).sum().backward()
    assert v.grad.tolist() == [4.0, 6.0]
    assert a.grad.tolist() == [[2.0, 3.0], [2.0, 3.0]]

    batch = ft.ones(2, 2, 3, requires_grad=True)
    weight = ft.ones(3, 2, requires_grad=True)
    (batch @ weight).sum().backward()
    assert batch.grad.tolist() == [[[2.0] * 3] * 2] * 2
    assert weight.grad.tolist() == [[4.0, 4.0]] * 3


def test_join_and_select_gradients():
    a = ft.ones(2, requires_grad=True)
    b = ft.ones(2, requires_grad=True)
    ft.cat([a, b * 2]).sum().backward()
    assert a.grad.tolist() == [1.0, 1.0] and b.grad.tolist() == [2.0, 2.0]
    a.grad = None
    b.grad = None
    (ft.stack([a, b * 3]) * ft.tensor([[1.0, 1.0], [2.0, 2.0]])).sum().backward()
    assert a.grad.tolist() == [1.0, 1.0] and b.grad.tolist() == [6.0, 6.0]

    a.grad = None
    b.grad = None
    ft.where(ft.tensor([1.0, 0.0]), a, b).sum().backward()
    assert a.grad.tolist() == [1.0, 0.0] and b.grad.tolist() == [0.0, 1.0]

    x = ft.tensor([1.0, 2.0], requires_grad=True)
    x.masked_fill(ft.tensor([0.0, 1.0]), 0.0).sum().backward()
    assert x.grad.tolist() == [1.0, 0.0]


def test_shape_op_gradients():
    x = ft.arange(4, requires_grad=True)
    x.index_select(0, [1, 1, 3]).sum().backward()
    assert x.grad.tolist() == [0.0, 2.0, 0.0, 1.0]
    x.grad = None
    x.narrow(0, 1, 2).sum().backward()
    assert x.grad.tolist() == [0.0, 1.0, 1.0, 0.0]
    x.grad = None
    first, second = x.chunk(2)
    (first + second * 2).sum().backward()
    assert x.grad.tolist() == [1.0, 1.0, 2.0, 2.0]
    x.grad = None
    pieces = x.split(1)
    (pieces[0] * 5 + pieces[3]).sum().backward()
    assert x.grad.tolist() == [5.0, 0.0, 0.0, 1.0]

    y = ft.tensor([1.0, 2.0], requires_grad=True)
    y.expand(3, 2).sum().backward()
    assert y.grad.tolist() == [3.0, 3.0]
    y.grad = None
    y.unsqueeze(0).squeeze(0).flatten().sum().backward()
    assert y.grad.tolist() == [1.0, 1.0]

    z = ft.arange(6, requires_grad=True)
    weights = ft.arange(6).reshape(3, 2)
    (z.reshape(2, 3).permute(1, 0) * weights).sum().backward()
    assert z.grad.tolist() == [0.0, 2.0, 4.0, 1.0, 3.0, 5.0]


def test_clone_passes_gradients_through():
    x = ft.tensor([1.0, 2.0], requires_grad=True)
    (x.clone() * 2).sum().backward()
    assert x.grad.tolist() == [2.0, 2.0]


def test_a_tensor_used_twice_sums_its_gradients():
    x = ft.tensor([3.0], requires_grad=True)
    y = x * x + x * 2
    y.backward()
    assert x.grad.tolist() == [8.0]


def test_detached_values_are_constants():
    x = ft.tensor([2.0], requires_grad=True)
    (x * x.detach()).sum().backward()
    assert x.grad.tolist() == [2.0]
    assert not x.detach().requires_grad and x.detach().is_leaf


def test_functional_loss_gradients():
    p = ft.tensor([1.0, 2.0, 3.0], requires_grad=True)
    t = ft.tensor([0.0, 3.0, 5.0])
    F.mse_loss(p, t).backward()
    close(p.grad.tolist(), [2 / 3, -2 / 3, -4 / 3])
    p.grad = None
    F.l1_loss(p, t).backward()
    close(p.grad.tolist(), [1 / 3, -1 / 3, -1 / 3])
    p.grad = None
    F.binary_cross_entropy_with_logits(p, [1.0, 0.0, 1.0]).backward()
    close(p.grad.tolist(), [(sigmoid(1.0) - 1) / 3, sigmoid(2.0) / 3, (sigmoid(3.0) - 1) / 3])


def test_layer_gradients_flow_to_every_parameter():
    model = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.LayerNorm(4), nn.Linear(4, 2))
    F.cross_entropy(model(ft.randn(5, 3)), [0, 1, 0, 1, 1]).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.grad.shape == p.shape, name
