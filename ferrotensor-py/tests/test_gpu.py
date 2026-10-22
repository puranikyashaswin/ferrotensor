"""GPU parity tests; run with FERROTENSOR_GPU_TESTS=1 on a machine with a GPU."""

import pytest

import ferrotensor as ft
import ferrotensor.nn.functional as F
from conftest import close
from ferrotensor import nn, optim

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def gpu():
    return ft.Device("gpu")


def test_device(gpu):
    assert gpu.is_gpu and gpu.type == "gpu" and gpu == ft.Device.gpu()
    assert gpu.name and gpu.backend != "Cpu"
    assert ft.zeros(2, device=gpu).device == gpu


def test_ops_match_cpu(gpu):
    a = ft.randn(4, 5)
    b = ft.randn(5, 3)
    ga, gb = a.to(gpu), b.to(gpu)
    close((ga @ gb).tolist(), (a @ b).tolist(), tol=1e-4)
    close((ga.relu() * 2 + 1).softmax(-1).tolist(), (a.relu() * 2 + 1).softmax(-1).tolist(), tol=1e-5)
    close(ga.sum(1).tolist(), a.sum(1).tolist(), tol=1e-4)
    close(ga[1:, ::2].tolist(), a[1:, ::2].tolist())


def test_grads_match_cpu(gpu):
    w = ft.randn(3, 4)
    wc = w.detach().requires_grad_()
    wg = w.to(gpu).requires_grad_()
    (wc.tanh() @ wc.T).sum().backward()
    (wg.tanh() @ wg.T).sum().backward()
    assert wg.grad.device == gpu
    close(wg.grad.tolist(), wc.grad.tolist(), tol=1e-4)


def test_device_mismatch_raises(gpu):
    with pytest.raises(ValueError, match="same device"):
        ft.ones(2) + ft.ones(2, device=gpu)


def test_train_on_gpu(gpu):
    x = ft.randn(64, 4, device=gpu)
    y = (x.sum(1) > 0).tolist()
    model = nn.Sequential(nn.Linear(4, 16, device=gpu), nn.ReLU(), nn.Linear(16, 2, device=gpu))
    opt = optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
    losses = []
    for _ in range(100):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_save_load_gpu(gpu, tmp_path):
    model = nn.Linear(3, 2, device=gpu)
    model.save(tmp_path / "m.safetensors")
    other = nn.Linear(3, 2, device=gpu)
    other.load(tmp_path / "m.safetensors")
    assert other.weight.tolist() == model.weight.tolist()
    assert other.weight.device == gpu
