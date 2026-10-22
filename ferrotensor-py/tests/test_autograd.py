import pytest

import ferrotensor as ft
from conftest import close


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
