"""Neural-network modules, in the style of ``torch.nn``.

Subclass :class:`Module` and assign layers, tensors with
``requires_grad=True``, or lists of modules as attributes; they are found
by :meth:`Module.parameters` and :meth:`Module.state_dict`.

Unlike PyTorch, ``Linear.weight`` has shape ``[in_features, out_features]``
(the layer computes ``x @ weight + bias``).
"""

from __future__ import annotations

from typing import Any, Iterator, NamedTuple, Sequence, Union, overload

from ferrotensor._ferrotensor import Device, Tensor, _Layer, load, save
from ferrotensor.nn import functional

DeviceLike = Union[Device, str, None]


class IncompatibleKeys(NamedTuple):
    """What :meth:`Module.load_state_dict` did not match."""

    missing_keys: list[str]
    unexpected_keys: list[str]


def Parameter(data: Tensor) -> Tensor:  # noqa: N802  (mirrors torch.nn.Parameter)
    """A leaf tensor sharing ``data``'s values, with ``requires_grad=True``."""
    return data.detach().requires_grad_()


class Module:
    """Base class for layers and models.

    >>> class MLP(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.fc1 = nn.Linear(4, 8)
    ...         self.fc2 = nn.Linear(8, 2)
    ...     def forward(self, x):
    ...         return self.fc2(self.fc1(x).relu())
    """

    training: bool

    def __init__(self) -> None:
        self.training = True

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must implement forward()")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    # ---- discovery -----------------------------------------------------

    def _local_parameters(self) -> Iterator[tuple[str, Tensor]]:
        for name, value in vars(self).items():
            if isinstance(value, Tensor) and value.requires_grad and value.is_leaf:
                yield name, value

    def named_children(self) -> Iterator[tuple[str, Module]]:
        """Direct submodules, including those in list, tuple or dict attributes."""
        for name, value in vars(self).items():
            if isinstance(value, Module):
                yield name, value
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield f"{name}.{i}", item
            elif isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, Module):
                        yield f"{name}.{key}", item

    def children(self) -> Iterator[Module]:
        for _, child in self.named_children():
            yield child

    def modules(self) -> Iterator[Module]:
        """This module and every submodule, depth first."""
        yield self
        for child in self.children():
            yield from child.modules()

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Tensor]]:
        """``(dotted name, parameter)`` pairs; the names are the state-dict keys."""
        params = [(prefix + name, p) for name, p in self._local_parameters()]
        for name, child in self.named_children():
            params.extend(child.named_parameters(f"{prefix}{name}."))
        return params

    def parameters(self) -> list[Tensor]:
        """Every trainable tensor (they share storage with the module)."""
        return [p for _, p in self.named_parameters()]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ---- modes and gradients ---------------------------------------------

    def _set_training(self, mode: bool) -> None:
        pass

    def train(self, mode: bool = True) -> Module:
        """Set training mode (dropout active) here and in every submodule."""
        self.training = mode
        self._set_training(mode)
        for child in self.children():
            child.train(mode)
        return self

    def eval(self) -> Module:
        """Set evaluation mode (dropout off)."""
        return self.train(False)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    # ---- state -----------------------------------------------------------

    def state_dict(self) -> dict[str, Tensor]:
        """Parameters by name, detached but sharing storage with the module."""
        return {name: p.detach() for name, p in self.named_parameters()}

    def load_state_dict(self, state_dict: dict[str, Tensor], strict: bool = True) -> IncompatibleKeys:
        """Copy values into the parameters in place (optimizers keep working).

        Raises ``RuntimeError`` for missing or unexpected keys when ``strict``,
        and ``ValueError`` when a shape differs. Nothing is copied on error.
        """
        own = dict(self.named_parameters())
        missing = [k for k in own if k not in state_dict]
        unexpected = [k for k in state_dict if k not in own]
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"error loading state dict: missing keys {missing}, unexpected keys {unexpected}"
            )
        for name, param in own.items():
            if name in state_dict and state_dict[name].shape != param.shape:
                raise ValueError(
                    f"shape mismatch for {name}: model has {param.shape}, "
                    f"state dict has {state_dict[name].shape}"
                )
        for name, param in own.items():
            if name in state_dict:
                param.copy_(state_dict[name])
        return IncompatibleKeys(missing, unexpected)

    def save(self, path: str) -> None:
        """Write :meth:`state_dict` to a safetensors file."""
        save(self.state_dict(), path)

    def load(self, path: str, strict: bool = True) -> IncompatibleKeys:
        """Load parameters from a safetensors file written by :meth:`save`."""
        return self.load_state_dict(load(path), strict=strict)

    # ---- printing --------------------------------------------------------

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        children = list(self.named_children())
        if not children:
            return f"{type(self).__name__}({self.extra_repr()})"
        lines = [f"{type(self).__name__}("]
        for name, child in children:
            body = repr(child).replace("\n", "\n  ")
            lines.append(f"  ({name}): {body}")
        lines.append(")")
        return "\n".join(lines)


class _RustModule(Module):
    """A module backed by a ferrotensor Rust layer."""

    _layer: _Layer

    def forward(self, input: Tensor) -> Tensor:
        return self._layer.forward(input)

    def _local_parameters(self) -> Iterator[tuple[str, Tensor]]:
        return iter(self._layer.named_parameters())

    def _param(self, name: str) -> Tensor | None:
        return dict(self._layer.named_parameters()).get(name)


class Linear(_RustModule):
    """``y = x @ weight + bias`` with ``weight`` of shape ``[in_features, out_features]``.

    Xavier-normal weights and zero bias. Accepts ``[..., in_features]`` input.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device: DeviceLike = None
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._layer = _Layer.linear(in_features, out_features, bias, device)

    @property
    def weight(self) -> Tensor:
        w = self._param("weight")
        assert w is not None
        return w

    @property
    def bias(self) -> Tensor | None:
        return self._param("bias")

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
        )


class Embedding(_RustModule):
    """Lookup table: integer ids (a tensor or nested list) to rows of ``weight``."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device: DeviceLike = None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._layer = _Layer.embedding(num_embeddings, embedding_dim, device)

    def forward(self, input: Any) -> Tensor:
        return self._layer.forward(input)

    @property
    def weight(self) -> Tensor:
        w = self._param("weight")
        assert w is not None
        return w

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"


class LayerNorm(_RustModule):
    """Layer normalization over the trailing ``normalized_shape`` dims, with affine weight and bias."""

    def __init__(
        self, normalized_shape: int | Sequence[int], eps: float = 1e-5, device: DeviceLike = None
    ) -> None:
        super().__init__()
        shape = [normalized_shape] if isinstance(normalized_shape, int) else list(normalized_shape)
        self.normalized_shape = tuple(shape)
        self.eps = eps
        self._layer = _Layer.layer_norm(shape, eps, device)

    @property
    def weight(self) -> Tensor:
        w = self._param("weight")
        assert w is not None
        return w

    @property
    def bias(self) -> Tensor:
        b = self._param("bias")
        assert b is not None
        return b

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}"


class RMSNorm(_RustModule):
    """RMS normalization over the last dimension (size ``dim``) with a learned scale."""

    def __init__(self, dim: int, eps: float = 1e-5, device: DeviceLike = None) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self._layer = _Layer.rms_norm(dim, eps, device)

    @property
    def weight(self) -> Tensor:
        w = self._param("weight")
        assert w is not None
        return w

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}"


class Dropout(_RustModule):
    """Zero entries with probability ``p`` in training mode (scaling the rest); identity in eval mode."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self._layer = _Layer.dropout(p)

    def _set_training(self, mode: bool) -> None:
        self._layer.set_training(mode)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class MultiHeadAttention(_RustModule):
    """Self-attention over ``[seq, embed_dim]`` or ``[batch, seq, embed_dim]`` input."""

    def __init__(
        self, embed_dim: int, num_heads: int, causal: bool = False, device: DeviceLike = None
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal
        self._layer = _Layer.attention(embed_dim, num_heads, causal, device)

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, causal={self.causal}"


class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.relu()


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        return input.leaky_relu(self.negative_slope)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class GELU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.gelu()


class SiLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.silu()


class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.sigmoid()


class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.tanh()


class Softmax(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return input.softmax(self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Identity(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input


class Flatten(Module):
    """Flatten dims ``start_dim..=end_dim`` (by default all but the batch dim)."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        return input.flatten(self.start_dim, self.end_dim)


class ModuleList(Module):
    """A list of modules whose parameters are named ``"0.weight"``, ``"1.weight"``, ..."""

    def __init__(self, modules: Sequence[Module] = ()) -> None:
        super().__init__()
        self._items: list[Module] = []
        for m in modules:
            self.append(m)

    def append(self, module: Module) -> ModuleList:
        if not isinstance(module, Module):
            raise TypeError(f"expected a Module, got {type(module).__name__}")
        self._items.append(module)
        return self

    def named_children(self) -> Iterator[tuple[str, Module]]:
        for i, m in enumerate(self._items):
            yield str(i), m

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._items)

    @overload
    def __getitem__(self, index: int) -> Module: ...
    @overload
    def __getitem__(self, index: slice) -> ModuleList: ...
    def __getitem__(self, index: int | slice) -> Module | ModuleList:
        if isinstance(index, slice):
            return type(self)(self._items[index])
        return self._items[index]


class Sequential(ModuleList):
    """Apply modules in order: ``Sequential(Linear(2, 8), ReLU(), Linear(8, 1))``."""

    def __init__(self, *modules: Module) -> None:
        super().__init__(modules)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return Sequential(*self._items[index])
        return self._items[index]

    def forward(self, input: Any) -> Any:
        for m in self._items:
            input = m(input)
        return input


__all__ = [
    "GELU",
    "Dropout",
    "Embedding",
    "Flatten",
    "Identity",
    "IncompatibleKeys",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "Module",
    "ModuleList",
    "MultiHeadAttention",
    "Parameter",
    "RMSNorm",
    "ReLU",
    "Sequential",
    "SiLU",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "functional",
]
