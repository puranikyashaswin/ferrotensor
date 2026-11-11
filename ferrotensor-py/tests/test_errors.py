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


def lr_setter(value):
    w = ft.ones(1, requires_grad=True)
    optim.SGD([w], lr=1.0).lr = value


@pytest.mark.parametrize(
    "fn, exc, match",
    [
        # tensors: dimensions and indices
        (lambda: ft.ones(2, 3).transpose(0, 2), IndexError, "out of range"),
        (lambda: ft.ones(2).unsqueeze(3), IndexError, "out of range"),
        (lambda: ft.ones(2).squeeze(2), IndexError, "out of range"),
        (lambda: ft.ones(2, 3).flatten(1, 0), ValueError, "after"),
        (lambda: ft.ones(3).chunk(0), ValueError, "positive"),
        (lambda: ft.ones(3).split(0), ValueError, "positive"),
        (lambda: ft.ones(3).chunk(2, dim=2), IndexError, "out of range"),
        (lambda: ft.ones(3).narrow(1, 0, 1), IndexError, "out of range"),
        (lambda: ft.ones(2, 3).index_select(5, [0]), IndexError, "out of range"),
        (lambda: ft.ones(2, 3).index_select(0, [0.5]), ValueError, "not an integer"),
        (lambda: ft.ones(2, 3).expand(3), ValueError, "fewer dimensions"),
        (lambda: ft.ones(3)[::0], ValueError, "step"),
        (lambda: ft.tensor(1.0)[0], IndexError, "too many"),
        (lambda: ft.ones(2, 3)[2], IndexError, "out of bounds"),
        (lambda: ft.ones(2, 3)[:, -4], IndexError, "out of bounds"),
        # tensors: shapes and values
        (lambda: ft.ones(3) @ ft.ones(4), ValueError, "inner dimensions"),
        (lambda: ft.ones(2, 3).copy_(ft.ones(3, 2)), ValueError, "does not match"),
        (lambda: setattr(ft.ones(2), "grad", ft.ones(3)), ValueError, "does not match"),
        (lambda: ft.tensor([]).max(), ValueError, "empty"),
        (lambda: ft.tensor([]).min(), ValueError, "empty"),
        (lambda: ft.tensor([]).argmax(), ValueError, "empty"),
        (lambda: ft.ones(2, 3).masked_fill(ft.ones(4), 1.0), ValueError, "broadcast"),
        (lambda: ft.where(ft.ones(2), ft.ones(3), 0.0), ValueError, "broadcast"),
        (lambda: ft.stack([ft.ones(2)], dim=3), IndexError, "out of range"),
        (lambda: ft.cat([ft.ones(2)], dim=1), IndexError, "out of range"),
        (lambda: ft.cat([1.0, 2.0]), TypeError, ""),
        # operands of the wrong type
        (lambda: ft.ones(2) + "a", TypeError, ""),
        (lambda: ft.ones(2) * [1.0, 2.0], TypeError, ""),
        (lambda: ft.ones(2) ** ft.ones(2), TypeError, ""),
        (lambda: ft.ones(2) @ [1.0, 2.0], TypeError, ""),
        (lambda: ft.ones(2).maximum("a"), TypeError, ""),
        (lambda: ft.zeros("a"), TypeError, ""),
        (lambda: ft.zeros(2.5), TypeError, ""),
        (lambda: ft.arange("x"), TypeError, ""),
        (lambda: ft.full((2,), "x"), TypeError, ""),
        (lambda: ft.tensor([1.0], requires_grad="yes"), TypeError, ""),
        # devices
        (lambda: ft.Device(5), TypeError, "(?i)str"),
        (lambda: ft.randn(2, device="tpu"), ValueError, "unknown device"),
        (lambda: ft.ones(2).to("tpu"), ValueError, "unknown device"),
        (lambda: ft.ones(2).to(5), TypeError, "device"),
        (lambda: ft.eye(2, device="metal"), ValueError, "unknown device"),
        (lambda: nn.Linear(2, 2, device="tpu"), ValueError, "unknown device"),
        (lambda: nn.Embedding(3, 2, device=3), TypeError, "device"),
        (lambda: F.one_hot([0], 2, device="tpu"), ValueError, "unknown device"),
        # layers
        (lambda: nn.Linear(2, 2)([1.0, 2.0]), TypeError, "Tensor"),
        (lambda: nn.Linear(2, 0), ValueError, "positive"),
        (lambda: nn.Embedding(0, 2), ValueError, "positive"),
        (lambda: nn.Embedding(5, 0), ValueError, "positive"),
        (lambda: nn.LayerNorm(0), ValueError, "positive"),
        (lambda: nn.LayerNorm(()), ValueError, "non-empty"),
        (lambda: nn.RMSNorm(0), ValueError, "positive"),
        (lambda: nn.RMSNorm(4)(ft.ones(2, 3)), ValueError, "last dimension 4"),
        (lambda: nn.MultiHeadAttention(8, 0), ValueError, "positive"),
        (lambda: nn.MultiHeadAttention(0, 1), ValueError, "multiple"),
        (lambda: nn.MultiHeadAttention(8, 2)(ft.ones(2, 3, 4, 8)), ValueError, "seq"),
        (lambda: nn.MultiHeadAttention(8, 2)(ft.ones(2, 3, 4)), ValueError, "seq"),
        (lambda: nn.Dropout(-0.1), ValueError, "probability"),
        (lambda: nn.Sequential(nn.Linear(2, 2))(ft.ones(3)), ValueError, "last dimension 2"),
        (lambda: nn.Sequential().append(3), TypeError, "Module"),
        # functional
        (lambda: F.smooth_l1_loss(ft.ones(2), ft.ones(2), beta=0.0), ValueError, "beta"),
        (lambda: F.l1_loss(ft.ones(2), ft.ones(3)), ValueError, "does not match"),
        (lambda: F.binary_cross_entropy(ft.ones(2), ft.ones(3)), ValueError, "does not match"),
        (lambda: F.binary_cross_entropy_with_logits(ft.ones(2), [1.0]), ValueError, "does not match"),
        (lambda: F.dropout(ft.ones(2), 1.5), ValueError, "probability"),
        (lambda: F.dropout(ft.ones(2), -0.1), ValueError, "probability"),
        (lambda: F.nll_loss(ft.ones(2, 3), [0, 5]), IndexError, "class index"),
        (lambda: F.nll_loss(ft.ones(2, 3), [0]), ValueError, "target shape"),
        (lambda: F.cross_entropy(ft.ones(2, 3), [0.5, 1]), IndexError, "class index"),
        (lambda: F.layer_norm(ft.ones(2, 3), ft.ones(2)), ValueError, "does not match"),
        (lambda: F.layer_norm(ft.ones(2, 3), None, ft.ones(4)), ValueError, "does not match"),
        (lambda: F.rms_norm(ft.ones(2, 3), ft.ones(4)), ValueError, "does not match"),
        (lambda: F.linear(ft.ones(3), ft.ones(3, 2)), ValueError, "at least 2"),
        (lambda: F.linear(ft.ones(2, 3), ft.ones(3, 2), ft.ones(5)), ValueError, "bias shape"),
        (lambda: F.one_hot([3], 3), IndexError, "out of bounds"),
        (lambda: F.one_hot([0.5], 3), ValueError, "not an integer"),
        (lambda: F.accuracy(ft.ones(2, 3), [0]), ValueError, "targets"),
        (lambda: F.accuracy(ft.ones(2, 3), [0, 3]), IndexError, "class index"),
        (lambda: F.accuracy(ft.ones(3), [0]), ValueError, "logits must be"),
        (lambda: F.cosine_similarity(ft.ones(2, 2), ft.ones(2, 3)), ValueError, "does not match"),
        (lambda: F.cosine_similarity(ft.ones(2, 2), ft.ones(2, 2), dim=2), IndexError, "out of range"),
        (lambda: F.scaled_dot_product_attention(ft.ones(3), ft.ones(3), ft.ones(3)), ValueError, "expects"),
        (
            lambda: F.scaled_dot_product_attention(ft.ones(2, 3), ft.ones(2, 3), ft.ones(3, 3)),
            ValueError,
            "expects",
        ),
        # optimizers and schedules
        (lambda: optim.SGD(5, lr=0.1), TypeError, "iterable"),
        (lambda: optim.SGD([1.0], lr=0.1), TypeError, "Tensor"),
        (lambda: optim.SGD([ft.ones(1, requires_grad=True)], lr=0.1, momentum=-1.0), ValueError, "non-negative"),
        (lambda: optim.SGD([ft.ones(1, requires_grad=True)], lr=0.1, weight_decay=-1.0), ValueError, "non-negative"),
        (lambda: optim.SGD([ft.ones(1, requires_grad=True)], lr=float("nan")), ValueError, "learning rate"),
        (lambda: optim.SGD([ft.ones(1, requires_grad=True)], lr=float("inf")), ValueError, "learning rate"),
        (lambda: optim.AdamW([ft.ones(1, requires_grad=True)], eps=0.0), ValueError, "eps"),
        (lambda: optim.AdamW([ft.ones(1, requires_grad=True)], weight_decay=-0.1), ValueError, "weight_decay"),
        (lambda: optim.AdamW([ft.ones(1, requires_grad=True)], max_grad_norm=0.0), ValueError, "max_grad_norm"),
        (lambda: optim.AdamW([ft.ones(1, requires_grad=True)], betas=(0.9, -0.1)), ValueError, "betas"),
        (lambda: lr_setter(-1.0), ValueError, "learning rate"),
        (lambda: optim.clip_grad_value([], 0.0), ValueError, "positive"),
        (lambda: optim.clip_grad_norm([], -1.0), ValueError, "positive"),
        (lambda: optim.CosineAnnealingLR(None, total_steps=4), ValueError, "base_lr"),
        (lambda: optim.WarmupCosineLR(None), ValueError, "peak_lr"),
        (lambda: optim.WarmupLinearLR(None), ValueError, "peak_lr"),
        (lambda: optim.ConstantLR(None), ValueError, "pass lr"),
        (lambda: optim.StepLR(None, step_size=1, base_lr=-1.0), ValueError, "learning rate"),
    ],
)
def test_more_errors_are_python_exceptions(fn, exc, match):
    with pytest.raises(exc, match=match):
        fn()


def test_requesting_the_gpu_works_or_raises_runtime_error():
    try:
        gpu = ft.Device("gpu")
    except RuntimeError as e:
        assert "no GPU adapter" in str(e)
        assert not ft.Device.gpu_available()
        with pytest.raises(RuntimeError, match="no GPU adapter"):
            ft.zeros(1, device="gpu")
        with pytest.raises(RuntimeError, match="no GPU adapter"):
            ft.Device.gpu()
        assert ft.Device.best().is_cpu
    else:
        assert gpu.is_gpu and ft.Device.gpu_available()
        assert ft.Device.best().is_gpu


def test_tensors_stay_usable_after_a_failed_operation():
    a = ft.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    with pytest.raises(ValueError):
        a @ ft.ones(3, 3)
    with pytest.raises(ValueError):
        a.reshape(5)
    with pytest.raises(IndexError):
        a[2]
    with pytest.raises(ValueError):
        a + ft.ones(3)
    assert a.tolist() == [[1.0, 2.0], [3.0, 4.0]] and a.grad is None
    (a * 2).sum().backward()
    assert a.grad.tolist() == [[2.0, 2.0], [2.0, 2.0]]


def test_a_failed_optimizer_construction_leaves_the_parameters_alone():
    w = ft.ones(2, requires_grad=True)
    (w * 3).sum().backward()
    with pytest.raises(ValueError):
        optim.SGD([w], lr=-1.0)
    with pytest.raises(ValueError):
        optim.AdamW([w, ft.ones(1)])
    assert w.tolist() == [1.0, 1.0] and w.grad.tolist() == [3.0, 3.0]
    opt = optim.SGD([w], lr=0.5)
    opt.step()
    assert w.tolist() == [-0.5, -0.5]


def test_a_failed_state_dict_load_copies_nothing():
    layer = nn.Linear(2, 1)
    before = layer.weight.tolist()
    with pytest.raises(ValueError, match="shape mismatch"):
        layer.load_state_dict({"weight": ft.ones(2, 1), "bias": ft.ones(5)})
    with pytest.raises(RuntimeError, match="missing keys"):
        layer.load_state_dict({"weight": ft.ones(2, 1)})
    assert layer.weight.tolist() == before and layer.bias.tolist() == [0.0]
