"""PyTorch-style tensors, autograd, nn and optim on wgpu (Metal, Vulkan, DX12) or the CPU.

>>> import ferrotensor as ft
>>> x = ft.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
>>> (x * x).sum().backward()
>>> x.grad.tolist()
[[2.0, 4.0], [6.0, 8.0]]
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from ferrotensor._ferrotensor import (
    Device,
    Tensor,
    __version__,
    _no_grad_enter,
    _no_grad_exit,
    arange,
    cat,
    eye,
    full,
    full_like,
    is_grad_enabled,
    linspace,
    load,
    ones,
    ones_like,
    rand,
    randn,
    save,
    stack,
    tensor,
    where,
    zeros,
    zeros_like,
)

_F = TypeVar("_F", bound=Callable[..., Any])


class no_grad:
    """Disable autograd recording on this thread, as a context manager or decorator.

    >>> w = ft.ones(2, requires_grad=True)
    >>> with ft.no_grad():
    ...     y = w * 2
    >>> y.requires_grad
    False
    """

    def __enter__(self) -> None:
        _no_grad_enter()

    def __exit__(self, *exc: object) -> None:
        _no_grad_exit()

    def __call__(self, func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with no_grad():
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]


from ferrotensor import nn, optim  # noqa: E402  (they import names defined above)

__all__ = [
    "Device",
    "Tensor",
    "__version__",
    "arange",
    "cat",
    "eye",
    "full",
    "full_like",
    "is_grad_enabled",
    "linspace",
    "load",
    "nn",
    "no_grad",
    "ones",
    "ones_like",
    "optim",
    "rand",
    "randn",
    "save",
    "stack",
    "tensor",
    "where",
    "zeros",
    "zeros_like",
]
