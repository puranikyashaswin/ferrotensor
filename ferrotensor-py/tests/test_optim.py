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


def test_sgd_momentum_follows_the_formula():
    w = ft.tensor([1.0], requires_grad=True)
    opt = optim.SGD([w], lr=0.1, momentum=0.9)
    quadratic_step(opt, w)  # grad 2.0: velocity 2.0, w = 1 - 0.2
    assert w.tolist() == pytest.approx([0.8])
    quadratic_step(opt, w)  # grad 1.6: velocity 0.9 * 2.0 + 1.6 = 3.4, w = 0.8 - 0.34
    assert w.tolist() == pytest.approx([0.46], abs=1e-5)


def test_sgd_weight_decay_is_added_to_the_gradient():
    w = ft.tensor([1.0], requires_grad=True)
    opt = optim.SGD([w], lr=0.1, weight_decay=0.5)
    quadratic_step(opt, w)  # grad 2.0 + 0.5 * 1.0 = 2.5
    assert w.tolist() == pytest.approx([0.75])


def test_sgd_skips_parameters_without_a_gradient():
    a = ft.tensor([1.0], requires_grad=True)
    b = ft.tensor([5.0], requires_grad=True)
    opt = optim.SGD([a, b], lr=0.1)
    (a * a).sum().backward()
    opt.step()
    assert a.tolist() == pytest.approx([0.8]) and b.tolist() == [5.0]
    opt.zero_grad()
    opt.step()  # nothing to apply
    assert a.tolist() == pytest.approx([0.8])


def test_adamw_first_step_moves_by_the_learning_rate():
    w = ft.tensor([1.0, -1.0], requires_grad=True)
    opt = optim.AdamW([w], lr=0.1, weight_decay=0.0)
    assert opt.steps == 0
    (w * w).sum().backward()
    opt.step()
    # Bias-corrected Adam makes the first update lr * sign(grad).
    assert w.tolist() == pytest.approx([0.9, -0.9], abs=1e-5)
    assert opt.steps == 1


def test_adamw_decays_weights_even_without_signal():
    w = ft.tensor([2.0], requires_grad=True)
    opt = optim.AdamW([w], lr=0.1, weight_decay=0.5)
    (w * 0).sum().backward()  # a zero gradient
    opt.step()
    assert w.tolist() == pytest.approx([2.0 * (1 - 0.1 * 0.5)], abs=1e-6)


def test_adamw_clips_the_gradient_before_stepping():
    clipped = ft.tensor([30.0, 40.0], requires_grad=True)
    free = ft.tensor([30.0, 40.0], requires_grad=True)
    for w, kwargs in ((clipped, {"max_grad_norm": 1.0}), (free, {})):
        opt = optim.AdamW([w], lr=0.1, weight_decay=0.0, **kwargs)
        w.grad = ft.tensor([30.0, 40.0])
        opt.step()
    # step() rescales the gradient in place to the maximum norm.
    assert clipped.grad.tolist() == pytest.approx([0.6, 0.8], abs=1e-4)
    assert free.grad.tolist() == [30.0, 40.0]


def test_learning_rate_setters_and_reprs():
    w = ft.ones(1, requires_grad=True)
    adam = optim.AdamW([w], lr=0.25)
    adam.lr = 0.125
    assert adam.lr == 0.125 and repr(adam) == "AdamW(lr=0.125, params=1)"
    assert repr(optim.SGD([w, w.detach().requires_grad_()], lr=1.0)) == "SGD(lr=1, params=2)"


def test_clip_grad_norm_edge_cases():
    a = ft.ones(1, requires_grad=True)
    b = ft.ones(1, requires_grad=True)
    assert optim.clip_grad_norm([a, b], 1.0) == 0.0  # no gradients yet
    a.grad = ft.tensor([3.0])
    b.grad = ft.tensor([4.0])
    assert optim.clip_grad_norm([a, b], 10.0) == pytest.approx(5.0)
    assert a.grad.tolist() == [3.0] and b.grad.tolist() == [4.0]  # under the limit: untouched
    assert optim.clip_grad_norm([a, b], 1.0) == pytest.approx(5.0)
    assert a.grad.tolist() == pytest.approx([0.6], abs=1e-4)  # the norm is global across parameters
    assert b.grad.tolist() == pytest.approx([0.8], abs=1e-4)
    optim.clip_grad_value([ft.ones(1)], 1.0)  # a tensor without a gradient is ignored


def test_clip_grad_value_clamps_both_signs():
    w = ft.ones(3, requires_grad=True)
    w.grad = ft.tensor([-5.0, 0.2, 5.0])
    optim.clip_grad_value([w], 1.0)
    assert w.grad.tolist() == pytest.approx([-1.0, 0.2, 1.0])


def test_schedulers_attached_to_an_optimizer():
    w = ft.ones(1, requires_grad=True)
    opt = optim.SGD([w], lr=1.0)
    step = optim.StepLR(opt, step_size=2, gamma=0.1)
    assert opt.lr == 1.0
    rates = [step.step() for _ in range(4)]
    assert rates == pytest.approx([1.0, 0.1, 0.1, 0.01])
    assert step.last_step == 4 and opt.lr == pytest.approx(0.01)

    opt.lr = 2.0
    cos = optim.CosineAnnealingLR(opt, total_steps=4, min_lr=0.5)
    assert cos.lr_at(0) == pytest.approx(2.0)
    assert cos.lr_at(2) == pytest.approx(1.25)
    assert cos.lr_at(4) == pytest.approx(0.5) and cos.lr_at(100) == pytest.approx(0.5)

    const = optim.ConstantLR(None, lr=0.3)
    assert const.lr_at(0) == const.lr_at(50) == pytest.approx(0.3)


def test_warmup_schedules_at_known_steps():
    cos = optim.WarmupCosineLR(None, warmup_steps=2, total_steps=6, peak_lr=1.0)
    assert cos.lr_at(0) == pytest.approx(0.5) and cos.lr_at(1) == pytest.approx(1.0)
    assert cos.lr_at(2) == pytest.approx(1.0) and cos.lr_at(4) == pytest.approx(0.5)
    assert cos.lr_at(6) == pytest.approx(0.0, abs=1e-6)
    lin = optim.WarmupLinearLR(None, warmup_steps=2, total_steps=6, peak_lr=1.0, end_lr=0.2)
    assert [lin.lr_at(i) for i in range(7)] == pytest.approx([0.5, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2])
    assert lin.lr_at(50) == pytest.approx(0.2)


def test_scheduler_changes_the_size_of_optimizer_steps():
    w = ft.tensor([1.0], requires_grad=True)
    opt = optim.SGD([w], lr=1.0)
    sched = optim.StepLR(opt, step_size=1, gamma=0.5)
    for expected in (0.5, 0.25):
        opt.zero_grad()
        (w * 1.0).sum().backward()
        before = w.item()
        opt.step()
        assert before - w.item() == pytest.approx(opt.lr)
        assert sched.step() == pytest.approx(expected)
