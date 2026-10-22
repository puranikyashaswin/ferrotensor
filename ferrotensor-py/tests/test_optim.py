import math

import pytest

import ferrotensor as ft
from ferrotensor import nn, optim


def quadratic_step(opt, w):
    opt.zero_grad()
    loss = (w * w).sum()
    loss.backward()
    opt.step()
    return loss.item()


def test_sgd_step_matches_formula():
    w = ft.tensor([1.0, -2.0], requires_grad=True)
    opt = optim.SGD([w], lr=0.1)
    quadratic_step(opt, w)
    assert w.tolist() == pytest.approx([0.8, -1.6])
    assert opt.lr == pytest.approx(0.1)
    opt.lr = 0.5
    assert opt.lr == 0.5
    assert "SGD(lr=0.5" in repr(opt)


def test_sgd_momentum_and_nesterov_converge():
    for kwargs in ({"momentum": 0.9}, {"momentum": 0.9, "nesterov": True}, {"weight_decay": 0.1}):
        w = ft.tensor([3.0], requires_grad=True)
        opt = optim.SGD([w], lr=0.05, **kwargs)
        for _ in range(200):
            quadratic_step(opt, w)
        assert abs(w.item()) < 1e-2, kwargs


def test_adamw_converges_and_counts_steps():
    w = ft.tensor([3.0, -3.0], requires_grad=True)
    opt = optim.AdamW([w], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0, max_grad_norm=10.0)
    for _ in range(300):
        quadratic_step(opt, w)
    assert opt.steps == 300
    assert max(abs(v) for v in w.tolist()) < 0.05


def test_optimizer_updates_module_parameters_in_place():
    model = nn.Linear(2, 1)
    before = model.weight.tolist()
    opt = optim.SGD(model.parameters(), lr=1.0)
    model(ft.ones(1, 2)).sum().backward()
    opt.step()
    assert model.weight.tolist() != before


def test_zero_grad():
    w = ft.ones(2, requires_grad=True)
    opt = optim.AdamW([w])
    (w * 2).sum().backward()
    opt.zero_grad()
    assert w.grad is None


def test_clip_grad():
    w = ft.tensor([3.0, 4.0], requires_grad=True)
    (w * w).sum().__mul__(0.5).backward()
    assert optim.clip_grad_norm([w], 1.0) == pytest.approx(5.0)
    assert w.grad.tolist() == pytest.approx([0.6, 0.8], abs=1e-5)
    optim.clip_grad_value([w], 0.7)
    assert w.grad.tolist() == pytest.approx([0.6, 0.7], abs=1e-5)


def test_schedulers():
    w = ft.ones(1, requires_grad=True)
    opt = optim.SGD([w], lr=1.0)
    sched = optim.WarmupCosineLR(opt, warmup_steps=2, total_steps=6, min_lr=0.0)
    assert opt.lr == pytest.approx(0.5)  # step 0 of warmup
    lrs = [sched.step() for _ in range(6)]
    assert lrs[0] == pytest.approx(1.0)
    assert lrs[-1] == pytest.approx(0.0, abs=1e-6)
    assert opt.lr == pytest.approx(lrs[-1])
    assert sched.get_last_lr() == lrs[-1]

    step = optim.StepLR(None, step_size=2, gamma=0.5, base_lr=1.0)
    assert [step.lr_at(i) for i in range(5)] == [1.0, 1.0, 0.5, 0.5, 0.25]
    cos = optim.CosineAnnealingLR(None, total_steps=4, base_lr=2.0)
    assert cos.lr_at(2) == pytest.approx(1.0)
    lin = optim.WarmupLinearLR(None, warmup_steps=1, total_steps=3, peak_lr=1.0)
    assert [lin.lr_at(i) for i in range(4)] == pytest.approx([1.0, 1.0, 0.5, 0.0])
    assert optim.ConstantLR(opt).lr_at(100) == opt.lr
    assert math.isfinite(optim.WarmupCosineLR(opt, 0, 10).lr_at(3))
