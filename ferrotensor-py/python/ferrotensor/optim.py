"""Optimizers, gradient clipping and learning-rate schedules.

>>> opt = ft.optim.AdamW(model.parameters(), lr=1e-3)
>>> sched = ft.optim.WarmupCosineLR(opt, warmup_steps=100, total_steps=1000)
>>> for step in range(1000):
...     opt.zero_grad()
...     loss = ...
...     loss.backward()
...     opt.step()
...     sched.step()
"""

from __future__ import annotations

from typing import Any, Optional

from ferrotensor._ferrotensor import SGD, AdamW, _Schedule, clip_grad_norm, clip_grad_value


def _base(optimizer: Any, lr: Optional[float], name: str) -> float:
    if lr is not None:
        return lr
    if optimizer is None:
        raise ValueError(f"pass {name} or an optimizer to take it from")
    return float(optimizer.lr)


class LRScheduler:
    """Sets ``optimizer.lr`` from a schedule indexed by optimizer step.

    Construction sets the rate for step 0; call :meth:`step` after each
    ``optimizer.step()``. ``optimizer`` may be ``None`` to use only
    :meth:`lr_at`.
    """

    def __init__(self, schedule: _Schedule, optimizer: Any = None) -> None:
        self._schedule = schedule
        self.optimizer = optimizer
        self.last_step = 0
        self._apply()

    def lr_at(self, step: int) -> float:
        """The learning rate at step ``step`` (0-based)."""
        return self._schedule.lr_at(step)

    def get_last_lr(self) -> float:
        """The learning rate for the current step."""
        return self.lr_at(self.last_step)

    def step(self) -> float:
        """Advance one step, update the optimizer, and return the new rate."""
        self.last_step += 1
        return self._apply()

    def _apply(self) -> float:
        lr = self.get_last_lr()
        if self.optimizer is not None:
            self.optimizer.lr = lr
        return lr


class ConstantLR(LRScheduler):
    """The same rate at every step."""

    def __init__(self, optimizer: Any = None, lr: Optional[float] = None) -> None:
        super().__init__(_Schedule.constant(_base(optimizer, lr, "lr")), optimizer)


class StepLR(LRScheduler):
    """``base_lr * gamma ** (step // step_size)``."""

    def __init__(
        self, optimizer: Any = None, step_size: int = 1, gamma: float = 0.1, base_lr: Optional[float] = None
    ) -> None:
        super().__init__(_Schedule.step(_base(optimizer, base_lr, "base_lr"), step_size, gamma), optimizer)


class CosineAnnealingLR(LRScheduler):
    """Cosine decay from ``base_lr`` to ``min_lr`` over ``total_steps``."""

    def __init__(
        self,
        optimizer: Any = None,
        total_steps: int = 1,
        min_lr: float = 0.0,
        base_lr: Optional[float] = None,
    ) -> None:
        super().__init__(
            _Schedule.cosine(_base(optimizer, base_lr, "base_lr"), min_lr, total_steps), optimizer
        )


class WarmupCosineLR(LRScheduler):
    """Linear warmup to ``peak_lr`` over ``warmup_steps``, then cosine decay to ``min_lr``."""

    def __init__(
        self,
        optimizer: Any = None,
        warmup_steps: int = 0,
        total_steps: int = 1,
        min_lr: float = 0.0,
        peak_lr: Optional[float] = None,
    ) -> None:
        peak = _base(optimizer, peak_lr, "peak_lr")
        super().__init__(_Schedule.warmup_cosine(peak, min_lr, warmup_steps, total_steps), optimizer)


class WarmupLinearLR(LRScheduler):
    """Linear warmup to ``peak_lr`` over ``warmup_steps``, then linear decay to ``end_lr``."""

    def __init__(
        self,
        optimizer: Any = None,
        warmup_steps: int = 0,
        total_steps: int = 1,
        end_lr: float = 0.0,
        peak_lr: Optional[float] = None,
    ) -> None:
        peak = _base(optimizer, peak_lr, "peak_lr")
        super().__init__(_Schedule.warmup_linear(peak, end_lr, warmup_steps, total_steps), optimizer)


__all__ = [
    "SGD",
    "AdamW",
    "ConstantLR",
    "CosineAnnealingLR",
    "LRScheduler",
    "StepLR",
    "WarmupCosineLR",
    "WarmupLinearLR",
    "clip_grad_norm",
    "clip_grad_value",
]
