import array
import math

import pytest

import ferrotensor as ft
from conftest import close


def test_from_nested_list():
    t = ft.tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3)
    assert t.ndim == 2 and t.dim() == 2 and t.numel() == 6 and len(t) == 2
    assert t.size() == (2, 3) and t.size(-1) == 3
    assert t.dtype == "float32"
    assert t.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_constructor_and_scalars():
    assert ft.Tensor([1.5]).tolist() == [1.5]
    s = ft.tensor(3.0)
    assert s.shape == () and s.tolist() == 3.0 and s.item() == 3.0 and float(s) == 3.0
    assert ft.tensor([]).shape == (0,)
    assert ft.tensor(((1, 2), (3, 4))).shape == (2, 2)


def test_from_buffer_like_objects():
    t = ft.tensor(array.array("f", [1.0, 2.0, 3.0]))
    assert t.tolist() == [1.0, 2.0, 3.0]
    t = ft.tensor(memoryview(array.array("d", [4.0, 5.0])))
    assert t.tolist() == [4.0, 5.0]
    assert ft.tensor(ft.ones(2)).tolist() == [1.0, 1.0]


def test_creation_functions():
    assert ft.zeros(2, 3).tolist() == [[0.0] * 3] * 2
    assert ft.ones((2,)).tolist() == [1.0, 1.0]
    assert ft.zeros([1, 2]).shape == (1, 2)
    assert ft.full((2, 2), 7).tolist() == [[7.0, 7.0], [7.0, 7.0]]
    assert ft.full(3, 1.5).tolist() == [1.5] * 3
    assert ft.eye(2).tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert ft.arange(4).tolist() == [0.0, 1.0, 2.0, 3.0]
    assert ft.arange(1, 2, 0.5).tolist() == [1.0, 1.5]
    assert ft.arange(3, 0, -1).tolist() == [3.0, 2.0, 1.0]
    assert ft.linspace(0, 1, 5).tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    base = ft.ones(2, 2)
    assert ft.zeros_like(base).tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert ft.ones_like(base).sum().item() == 4.0
    assert ft.full_like(base, 2).sum().item() == 8.0
    assert ft.zeros(3, requires_grad=True).requires_grad


def test_random():
    r = ft.rand(1000)
    assert r.shape == (1000,)
    assert 0.0 <= r.min().item() and r.max().item() < 1.0
    n = ft.randn(4000)
    assert abs(n.mean().item()) < 0.1
    assert 0.9 < n.var().item() < 1.1


def test_arithmetic_and_broadcasting():
    a = ft.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = ft.tensor([10.0, 20.0])
    assert (a + b).tolist() == [[11.0, 22.0], [13.0, 24.0]]
    assert (a - 1).tolist() == [[0.0, 1.0], [2.0, 3.0]]
    assert (1 - a).tolist() == [[0.0, -1.0], [-2.0, -3.0]]
    assert (2 * a).tolist() == [[2.0, 4.0], [6.0, 8.0]]
    assert (a / 2).tolist() == [[0.5, 1.0], [1.5, 2.0]]
    close((2 / a).tolist(), [[2.0, 1.0], [2 / 3, 0.5]])
    assert (a**2).tolist() == [[1.0, 4.0], [9.0, 16.0]]
    assert (-a).tolist() == [[-1.0, -2.0], [-3.0, -4.0]]
    assert abs(ft.tensor([-1.0, 2.0])).tolist() == [1.0, 2.0]
    assert a.add(b).mul(0.5).sub(1).div(2).shape == (2, 2)
    assert a.maximum(2.5).tolist() == [[2.5, 2.5], [3.0, 4.0]]
    assert a.minimum(b).tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_comparisons():
    a = ft.tensor([1.0, 2.0, 3.0])
    assert (a > 2).tolist() == [0.0, 0.0, 1.0]
    assert (a >= 2).tolist() == [0.0, 1.0, 1.0]
    assert (a < 2).tolist() == [1.0, 0.0, 0.0]
    assert (a <= 2).tolist() == [1.0, 1.0, 0.0]
    assert a.eq(2).tolist() == [0.0, 1.0, 0.0]
    assert a.gt(a).tolist() == [0.0, 0.0, 0.0]
    # == stays identity so tensors remain hashable.
    assert (a == a) is True
    assert len({a, a}) == 1


def test_matmul():
    a = ft.tensor([[1.0, 2.0], [3.0, 4.0]])
    v = ft.tensor([1.0, 1.0])
    assert (a @ a).tolist() == [[7.0, 10.0], [15.0, 22.0]]
    assert (a @ v).tolist() == [3.0, 7.0]
    assert (v @ a).tolist() == [4.0, 6.0]
    assert (v @ v).tolist() == 2.0
    batched = ft.ones(3, 2, 4) @ ft.ones(4, 5)
    assert batched.shape == (3, 2, 5) and batched.tolist()[0][0][0] == 4.0
    assert a.matmul(a).shape == (2, 2)


def test_unary_math():
    x = ft.tensor([-1.0, 0.0, 1.0])
    assert x.relu().tolist() == [0.0, 0.0, 1.0]
    close(x.leaky_relu(0.1).tolist(), [-0.1, 0.0, 1.0])
    close(x.sigmoid().tolist(), [0.26894142, 0.5, 0.73105858])
    close(x.tanh().tolist(), [-0.76159416, 0.0, 0.76159416])
    close(x.exp().tolist(), [0.36787944, 1.0, 2.71828183])
    close(x.silu().tolist(), [-0.26894142, 0.0, 0.73105858])
    close(x.gelu().tolist(), [-0.15880801, 0.0, 0.84119199], tol=1e-3)
    close(x.sin().tolist(), [-0.84147098, 0.0, 0.84147098])
    close(x.cos().tolist(), [0.54030231, 1.0, 0.54030231])
    assert x.sign().tolist() == [-1.0, 0.0, 1.0]
    assert x.abs().tolist() == [1.0, 0.0, 1.0]
    assert x.neg().tolist() == [1.0, -0.0, -1.0]
    assert x.square().tolist() == [1.0, 0.0, 1.0]
    assert x.clamp(min=0).tolist() == [0.0, 0.0, 1.0]
    assert x.clamp(-0.5, 0.5).tolist() == [-0.5, 0.0, 0.5]
    p = ft.tensor([1.0, 4.0])
    assert p.sqrt().tolist() == [1.0, 2.0]
    assert p.rsqrt().tolist() == [1.0, 0.5]
    assert p.reciprocal().tolist() == [1.0, 0.25]
    close(p.log().tolist(), [0.0, 1.38629436])
    assert p.pow(2).tolist() == [1.0, 16.0]


def test_softmax():
    x = ft.tensor([[1.0, 2.0, 3.0]])
    close(x.softmax(-1).tolist(), [[0.09003057, 0.24472847, 0.66524096]])
    close(x.log_softmax(dim=1).tolist(), [[-2.40760596, -1.40760596, -0.40760596]])


def test_reductions():
    x = ft.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert x.sum().item() == 21.0
    assert x.sum(0).tolist() == [5.0, 7.0, 9.0]
    assert x.sum(dim=-1, keepdim=True).tolist() == [[6.0], [15.0]]
    assert x.sum((0, 1)).tolist() == 21.0
    assert x.sum(keepdim=True).shape == (1, 1)
    assert x.mean().item() == 3.5
    assert x.mean(1).tolist() == [2.0, 5.0]
    assert x.max().item() == 6.0 and x.min().item() == 1.0
    assert x.max(0).tolist() == [4.0, 5.0, 6.0]
    assert x.min(1, keepdim=True).tolist() == [[1.0], [4.0]]
    close(x.var().item(), 35 / 12)
    assert x.var(1).tolist() == pytest.approx([2 / 3, 2 / 3])
    assert x.argmax().item() == 5.0
    assert x.argmin().item() == 0.0
    assert x.argmax(1).tolist() == [2.0, 2.0]
    assert x.argmin(0, keepdim=True).tolist() == [[0.0, 0.0, 0.0]]


def test_shape_ops():
    x = ft.arange(6)
    assert x.reshape(2, 3).shape == (2, 3)
    assert x.reshape((3, -1)).shape == (3, 2)
    assert x.view(-1, 2).shape == (3, 2)
    m = x.reshape(2, 3)
    assert m.flatten().shape == (6,)
    assert ft.ones(2, 3, 4).flatten(1).shape == (2, 12)
    assert ft.ones(1, 3, 1).squeeze().shape == (3,)
    assert ft.ones(1, 3, 1).squeeze(0).shape == (3, 1)
    assert ft.ones(1, 3, 1).squeeze(1).shape == (1, 3, 1)
    assert m.unsqueeze(0).shape == (1, 2, 3)
    assert m.unsqueeze(-1).shape == (2, 3, 1)
    assert m.transpose(0, 1).tolist() == [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]
    assert m.T.tolist() == m.transpose(0, 1).tolist()
    assert ft.ones(2, 3, 4).permute(2, 0, 1).shape == (4, 2, 3)
    assert ft.ones(2, 3, 4).permute((-1, 0, 1)).shape == (4, 2, 3)
    assert ft.tensor([[1.0], [2.0]]).expand(2, 3).tolist() == [[1.0] * 3, [2.0] * 3]
    assert ft.tensor([1.0, 2.0]).expand(3, -1).shape == (3, 2)
    assert m.narrow(1, 1, 2).tolist() == [[1.0, 2.0], [4.0, 5.0]]
    assert [c.shape for c in ft.arange(5).chunk(2)] == [(3,), (2,)]
    assert [s.tolist() for s in ft.arange(5).split(2)] == [[0.0, 1.0], [2.0, 3.0], [4.0]]
    assert m.index_select(1, [2, 0]).tolist() == [[2.0, 0.0], [5.0, 3.0]]
    assert m.index_select(0, ft.tensor([1])).tolist() == [[3.0, 4.0, 5.0]]


def test_indexing():
    m = ft.arange(12).reshape(3, 4)
    assert m[0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert m[-1, -1].tolist() == 11.0
    assert m[1:].shape == (2, 4)
    assert m[:, 1:3].tolist() == [[1.0, 2.0], [5.0, 6.0], [9.0, 10.0]]
    assert m[::2, ::-1].tolist() == [[3.0, 2.0, 1.0, 0.0], [11.0, 10.0, 9.0, 8.0]]
    assert m[5:].shape == (0, 4)


def test_cat_stack_where():
    a, b = ft.ones(2, 2), ft.zeros(1, 2)
    assert ft.cat([a, b]).shape == (3, 2)
    assert ft.cat((a, a), dim=1).shape == (2, 4)
    assert ft.stack([a, a]).shape == (2, 2, 2)
    assert ft.stack([a, a], dim=-1).shape == (2, 2, 2)
    cond = ft.tensor([1.0, 0.0, 1.0])
    assert ft.where(cond, ft.tensor([1.0, 2.0, 3.0]), 0.0).tolist() == [1.0, 0.0, 3.0]
    x = ft.tensor([[1.0, 2.0]])
    assert x.masked_fill(ft.tensor([[0.0, 1.0]]), -1e9).tolist() == [[1.0, -1e9]]


def test_detach_clone_copy():
    a = ft.tensor([1.0, 2.0])
    d = a.detach()
    c = a.clone()
    a.copy_(ft.tensor([5.0, 6.0]))
    assert d.tolist() == [5.0, 6.0]  # shares storage
    assert c.tolist() == [1.0, 2.0]  # own storage


def test_repr():
    assert repr(ft.tensor([1.0, 2.5])) == "tensor([1.0, 2.5])"
    assert repr(ft.ones(2, requires_grad=True)) == "tensor([1.0, 1.0], requires_grad=True)"
    assert "grad_fn=<" in repr(ft.ones(2, requires_grad=True) * 2)
    assert repr(ft.zeros(2000)) == "tensor(shape=[2000])"


def test_device_basics():
    cpu = ft.Device("cpu")
    assert cpu.is_cpu and not cpu.is_gpu and cpu.type == "cpu"
    assert cpu == ft.Device.cpu() and repr(cpu) == "Device('cpu')"
    assert cpu.name == "CPU" and cpu.backend == "Cpu"
    assert ft.zeros(2).device == cpu
    assert ft.zeros(2, device="cpu").device.is_cpu
    assert isinstance(ft.Device.gpu_available(), bool)
    best = ft.Device.best()
    assert best.is_gpu == ft.Device.gpu_available()
    assert ft.zeros(1).to("cpu").device == cpu
    assert ft.__version__


def test_manual_seed_makes_random_tensors_and_init_reproducible():
    from ferrotensor import nn

    def draw():
        return ft.randn(5).tolist(), nn.Linear(3, 2).weight.tolist()

    ft.manual_seed(123)
    first = draw()
    ft.manual_seed(123)
    assert draw() == first
    ft.manual_seed(124)
    assert draw()[0] != first[0]


def test_named_comparison_methods():
    a = ft.tensor([1.0, 2.0, 3.0])
    b = ft.tensor([2.0, 2.0, 2.0])
    assert a.lt(b).tolist() == [1.0, 0.0, 0.0]
    assert a.le(b).tolist() == [1.0, 1.0, 0.0]
    assert a.ge(b).tolist() == [0.0, 1.0, 1.0]
    assert a.gt(b).tolist() == [0.0, 0.0, 1.0]
    assert a.eq(b).tolist() == [0.0, 1.0, 0.0]
    assert a.lt(2.5).tolist() == [1.0, 1.0, 0.0]
    assert a.ge(ft.tensor([[1.0], [3.0]])).tolist() == [[1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]


def test_power_and_leaky_relu_defaults():
    x = ft.tensor([2.0, 3.0])
    close((x**0.5).tolist(), [2**0.5, 3**0.5])
    assert (x**3).tolist() == [8.0, 27.0]
    close((x**-1).tolist(), [0.5, 1 / 3])
    close(ft.tensor([-2.0, 3.0]).leaky_relu().tolist(), [-0.02, 3.0])


def test_where_with_tensor_branches():
    cond = ft.tensor([[1.0, 0.0], [0.0, 1.0]])
    other = ft.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert ft.where(cond, other, ft.tensor([10.0, 20.0])).tolist() == [[1.0, 20.0], [10.0, 4.0]]
    assert ft.where(cond, 5.0, other).tolist() == [[5.0, 2.0], [3.0, 5.0]]


def test_multi_dim_reductions():
    x = ft.arange(24).reshape(2, 3, 4)
    assert x.sum((0, 2)).tolist() == [60.0, 92.0, 124.0]
    assert x.mean((0, 2)).tolist() == [7.5, 11.5, 15.5]
    assert x.max((0, 2)).tolist() == [15.0, 19.0, 23.0]
    assert x.min((0, 2)).tolist() == [0.0, 4.0, 8.0]
    assert x.sum((0, 2), keepdim=True).shape == (1, 3, 1)
    assert x.sum([-1, 0]).tolist() == [60.0, 92.0, 124.0]
    close(x.var(-1).tolist(), [[1.25] * 3] * 2)
    assert x.var(1, keepdim=True).shape == (2, 1, 4)


def test_arg_reductions_with_keepdim():
    x = ft.tensor([[1.0, 5.0, 2.0], [7.0, 0.0, 3.0]])
    assert x.argmax(1, keepdim=True).tolist() == [[1.0], [0.0]]
    assert x.argmin(0).tolist() == [0.0, 1.0, 0.0]
    assert x.argmax(-1).tolist() == [1.0, 0.0]
    assert x.argmax().item() == 3.0 and x.argmin().item() == 4.0


def test_softmax_over_the_first_dim():
    x = ft.tensor([[1.0, 2.0], [3.0, 4.0]])
    low, high = 1 / (1 + math.exp(2)), 1 - 1 / (1 + math.exp(2))
    close(x.softmax(0).tolist(), [[low, low], [high, high]])
    close(x.log_softmax(0).tolist(), [[math.log(low)] * 2, [math.log(high)] * 2])
    assert x.softmax(1).sum(1).tolist() == pytest.approx([1.0, 1.0])


def test_chunk_split_narrow_and_index_select_along_dims():
    m = ft.arange(6).reshape(2, 3)
    assert [p.tolist() for p in m.chunk(3, dim=1)] == [[[0.0], [3.0]], [[1.0], [4.0]], [[2.0], [5.0]]]
    assert [p.shape for p in m.chunk(2, dim=-1)] == [(2, 2), (2, 1)]
    assert [p.shape for p in m.split(2, dim=1)] == [(2, 2), (2, 1)]
    assert m.split(2, 1)[1].tolist() == [[2.0], [5.0]]
    assert m.narrow(-1, 0, 2).tolist() == [[0.0, 1.0], [3.0, 4.0]]
    assert m.index_select(1, [-1]).tolist() == [[2.0], [5.0]]
    assert m.index_select(0, [1, 1, 0]).shape == (3, 3)


def test_creation_options():
    base = ft.ones(2)
    assert ft.zeros_like(base, requires_grad=True).requires_grad
    assert ft.ones_like(base, requires_grad=True).requires_grad
    assert ft.full_like(base, 3.0, requires_grad=True).tolist() == [3.0, 3.0]
    assert ft.full((2,), 1.0, requires_grad=True).requires_grad
    assert ft.arange(3, requires_grad=True).requires_grad
    assert ft.eye(3, requires_grad=True).requires_grad
    assert ft.randn(2, requires_grad=True).requires_grad
    assert ft.rand(2, requires_grad=True).requires_grad
    assert ft.arange(0, 1, 0.25).tolist() == [0.0, 0.25, 0.5, 0.75]
    assert ft.arange(5, 0).shape == (0,)
    assert ft.linspace(0, 1, 1).tolist() == [0.0]
    assert ft.linspace(0, 1, 0).shape == (0,)
    assert ft.rand((2, 3), device=ft.Device("cpu")).shape == (2, 3)
    assert ft.randn([4], device="cpu").shape == (4,)
    assert ft.zeros(0, 3).shape == (0, 3) and ft.zeros(0, 3).tolist() == []


def test_scalar_and_one_element_tensors():
    s = ft.tensor(2.0)
    assert s.ndim == 0 and s.dim() == 0 and s.numel() == 1 and s.size() == ()
    assert ft.tensor([[2.5]]).item() == 2.5 and float(ft.tensor([4.0])) == 4.0
    assert (s + 1).tolist() == 3.0
    assert ft.tensor([]).numel() == 0 and ft.tensor([]).tolist() == []


def test_to_and_cpu_on_the_cpu(cpu):
    x = ft.tensor([1.0, 2.0], requires_grad=True)
    y = x.to(cpu).cpu().to("cpu")
    assert y.device == cpu and y.tolist() == [1.0, 2.0]
    (y * 3).sum().backward()
    assert x.grad.tolist() == [3.0, 3.0]


def test_grad_fn_names_the_producing_op():
    w = ft.ones(2, requires_grad=True)
    assert w.grad_fn is None
    assert isinstance(((w * 2).sum()).grad_fn, str)
    assert (w * 2).grad_fn and (w + 1).grad_fn and w.exp().grad_fn
    assert "grad_fn" not in repr(w)
    assert ft.ones(2).grad_fn is None


def test_operators_do_not_modify_their_inputs():
    a = ft.tensor([1.0, 2.0])
    b = ft.tensor([3.0, 4.0])
    _ = a + b, a * b, a - b, a / b, a @ b, a.relu(), a.softmax(), a.reshape(2, 1), a.T, a.clamp(1.5, 1.6)
    assert a.tolist() == [1.0, 2.0] and b.tolist() == [3.0, 4.0]


def test_tensors_are_not_interned_by_equality():
    a, b = ft.ones(2), ft.ones(2)
    assert not (a == b)
    assert a.eq(b).tolist() == [1.0, 1.0]
    assert len({a, b}) == 2
