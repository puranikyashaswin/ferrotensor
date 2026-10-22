"""Type stubs for the compiled ``ferrotensor._ferrotensor`` extension.

All tensors are float32. Bad arguments raise ``ValueError`` (shapes,
values), ``IndexError`` (dims, indices) or ``TypeError``; any other failure
inside ferrotensor raises ``RuntimeError``.
"""

from __future__ import annotations

from os import PathLike
from typing import Any, Iterable, Sequence, Union, overload

__version__: str

DeviceLike = Union["Device", str, None]
"""A ``Device``, ``"cpu"``, ``"gpu"``, ``"best"`` or ``None`` (the CPU)."""
Number = Union[int, float]
Operand = Union["Tensor", int, float]
ShapeArg = Union[int, Sequence[int]]
Data = Any
"""A number, a nested list/tuple of numbers, a ``Tensor``, or any object with
``.tolist()`` (``array.array``, a NumPy array, a ``memoryview``)."""

class Device:
    """Where tensors live: the CPU, or a GPU through wgpu (Metal, Vulkan, DX12).

    ``Device("gpu")`` raises ``RuntimeError`` when no GPU adapter is
    available. The GPU is opened once per process and shared.
    """

    def __init__(self, kind: str = "cpu") -> None: ...
    @staticmethod
    def cpu() -> Device: ...
    @staticmethod
    def gpu() -> Device:
        """The GPU; raises ``RuntimeError`` when no adapter is available."""
    @staticmethod
    def best() -> Device:
        """The GPU when available, otherwise the CPU."""
    @staticmethod
    def gpu_available() -> bool: ...
    @property
    def type(self) -> str:
        """``"cpu"`` or ``"gpu"``."""
    @property
    def is_cpu(self) -> bool: ...
    @property
    def is_gpu(self) -> bool: ...
    @property
    def name(self) -> str:
        """``"CPU"`` or the GPU adapter name."""
    @property
    def backend(self) -> str:
        """``"Cpu"`` or the wgpu backend, e.g. ``"Metal"``."""
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class Tensor:
    """A float32 tensor on the CPU or a GPU, with reverse-mode autograd.

    Ops broadcast like NumPy. Dimensions may be negative.
    """

    def __init__(self, data: Data, *, device: DeviceLike = None, requires_grad: bool = False) -> None: ...

    # metadata
    @property
    def shape(self) -> tuple[int, ...]: ...
    @overload
    def size(self) -> tuple[int, ...]: ...
    @overload
    def size(self, dim: int) -> int: ...
    @property
    def ndim(self) -> int: ...
    def dim(self) -> int: ...
    def numel(self) -> int: ...
    @property
    def dtype(self) -> str:
        """Always ``"float32"``."""
    @property
    def device(self) -> Device: ...
    def __len__(self) -> int: ...

    # autograd
    @property
    def requires_grad(self) -> bool: ...
    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """Only leaf tensors can change this (``RuntimeError`` otherwise)."""
    def requires_grad_(self, requires_grad: bool = True) -> Tensor:
        """Set ``requires_grad`` in place and return ``self``."""
    @property
    def is_leaf(self) -> bool:
        """True unless the tensor was produced by a recorded op."""
    @property
    def grad_fn(self) -> str | None:
        """Name of the op that produced this tensor, or ``None``."""
    @property
    def grad(self) -> Tensor | None:
        """The accumulated gradient (``None`` until ``backward`` reaches this leaf)."""
    @grad.setter
    def grad(self, value: Tensor | None) -> None: ...
    def backward(self, gradient: Tensor | None = None) -> None:
        """Accumulate gradients into every leaf's ``.grad``.

        Without ``gradient`` the tensor must have one element. Raises
        ``RuntimeError`` when the tensor does not require grad.
        """
    def detach(self) -> Tensor:
        """The same values (shared storage), cut from the autograd graph."""
    def clone(self) -> Tensor:
        """A copy with its own storage (differentiable)."""
    def copy_(self, src: Tensor) -> Tensor:
        """Overwrite the values in place with ``src`` (same shape, not recorded)."""

    # data out
    def tolist(self) -> Any:
        """Nested lists of floats; a float for a 0-D tensor."""
    def item(self) -> float:
        """The value of a one-element tensor (``ValueError`` otherwise)."""
    def __float__(self) -> float: ...
    def to(self, device: Device | str) -> Tensor:
        """Copy to ``device`` (differentiable)."""
    def cpu(self) -> Tensor: ...

    # operators
    def __add__(self, other: Operand) -> Tensor: ...
    def __radd__(self, other: Operand) -> Tensor: ...
    def __sub__(self, other: Operand) -> Tensor: ...
    def __rsub__(self, other: Number) -> Tensor: ...
    def __mul__(self, other: Operand) -> Tensor: ...
    def __rmul__(self, other: Operand) -> Tensor: ...
    def __truediv__(self, other: Operand) -> Tensor: ...
    def __rtruediv__(self, other: Number) -> Tensor: ...
    def __pow__(self, exponent: Number) -> Tensor: ...
    def __matmul__(self, other: Tensor) -> Tensor:
        """Matrix product; batched for 3-D+ and NumPy-style for 1-D operands."""
    def __neg__(self) -> Tensor: ...
    def __abs__(self) -> Tensor: ...
    def __lt__(self, other: Operand) -> Tensor:  # type: ignore[override]
        """1.0 where true, else 0.0."""
    def __le__(self, other: Operand) -> Tensor: ...  # type: ignore[override]
    def __gt__(self, other: Operand) -> Tensor: ...  # type: ignore[override]
    def __ge__(self, other: Operand) -> Tensor: ...  # type: ignore[override]
    def __hash__(self) -> int:
        """By identity; ``==`` also compares identity (use ``eq()`` for values)."""
    def __getitem__(self, index: int | slice | tuple[int | slice, ...]) -> Tensor:
        """Index with ints and slices (any step); differentiable."""

    # element-wise
    def add(self, other: Operand) -> Tensor: ...
    def sub(self, other: Operand) -> Tensor: ...
    def mul(self, other: Operand) -> Tensor: ...
    def div(self, other: Operand) -> Tensor: ...
    def pow(self, exponent: Number) -> Tensor: ...
    def matmul(self, other: Tensor) -> Tensor: ...
    def maximum(self, other: Operand) -> Tensor: ...
    def minimum(self, other: Operand) -> Tensor: ...
    def gt(self, other: Operand) -> Tensor:
        """1.0 where ``self > other``, else 0.0."""
    def lt(self, other: Operand) -> Tensor: ...
    def ge(self, other: Operand) -> Tensor: ...
    def le(self, other: Operand) -> Tensor: ...
    def eq(self, other: Operand) -> Tensor: ...
    def neg(self) -> Tensor: ...
    def abs(self) -> Tensor: ...
    def relu(self) -> Tensor: ...
    def leaky_relu(self, negative_slope: float = 0.01) -> Tensor: ...
    def gelu(self) -> Tensor: ...
    def silu(self) -> Tensor: ...
    def sigmoid(self) -> Tensor: ...
    def tanh(self) -> Tensor: ...
    def exp(self) -> Tensor: ...
    def log(self) -> Tensor: ...
    def sqrt(self) -> Tensor: ...
    def rsqrt(self) -> Tensor: ...
    def square(self) -> Tensor: ...
    def reciprocal(self) -> Tensor: ...
    def sin(self) -> Tensor: ...
    def cos(self) -> Tensor: ...
    def sign(self) -> Tensor: ...
    def clamp(self, min: float | None = None, max: float | None = None) -> Tensor: ...
    def softmax(self, dim: int = -1) -> Tensor: ...
    def log_softmax(self, dim: int = -1) -> Tensor: ...
    def masked_fill(self, mask: Tensor, value: float) -> Tensor:
        """Replace entries where ``mask`` is non-zero with the finite ``value``."""

    # reductions (dim: an int, a sequence of ints, or None for all)
    def sum(self, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor: ...
    def mean(self, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor: ...
    def max(self, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
        """Largest values (no indices; see ``argmax``)."""
    def min(self, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor: ...
    def var(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Population variance (divides by n, unlike ``torch.var``'s default)."""
    def argmax(self, dim: int | None = None, keepdim: bool = False) -> Tensor:
        """Indices as a float tensor; flattened index when ``dim`` is None."""
    def argmin(self, dim: int | None = None, keepdim: bool = False) -> Tensor: ...

    # shape
    def reshape(self, *shape: ShapeArg) -> Tensor:
        """New shape with the same element count; one entry may be -1."""
    def view(self, *shape: ShapeArg) -> Tensor: ...
    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor: ...
    def squeeze(self, dim: int | None = None) -> Tensor: ...
    def unsqueeze(self, dim: int) -> Tensor: ...
    def permute(self, *dims: ShapeArg) -> Tensor: ...
    def transpose(self, dim0: int, dim1: int) -> Tensor: ...
    @property
    def T(self) -> Tensor:
        """The tensor with its dimensions reversed."""
    def expand(self, *sizes: ShapeArg) -> Tensor:
        """Broadcast to a larger shape; -1 keeps a dimension."""
    def narrow(self, dim: int, start: int, length: int) -> Tensor: ...
    def chunk(self, chunks: int, dim: int = 0) -> list[Tensor]: ...
    def split(self, split_size: int, dim: int = 0) -> list[Tensor]: ...
    def index_select(self, dim: int, index: Tensor | Sequence[int]) -> Tensor: ...

def tensor(data: Data, *, device: DeviceLike = None, requires_grad: bool = False) -> Tensor:
    """Create a tensor from a number, nested list, or object with ``.tolist()``."""

def zeros(*size: ShapeArg, device: DeviceLike = None, requires_grad: bool = False) -> Tensor: ...
def ones(*size: ShapeArg, device: DeviceLike = None, requires_grad: bool = False) -> Tensor: ...
def randn(*size: ShapeArg, device: DeviceLike = None, requires_grad: bool = False) -> Tensor:
    """Standard-normal samples."""

def rand(*size: ShapeArg, device: DeviceLike = None, requires_grad: bool = False) -> Tensor:
    """Uniform samples in ``[0, 1)``."""

def full(
    size: ShapeArg, fill_value: float, *, device: DeviceLike = None, requires_grad: bool = False
) -> Tensor: ...
def zeros_like(input: Tensor, *, requires_grad: bool = False) -> Tensor: ...
def ones_like(input: Tensor, *, requires_grad: bool = False) -> Tensor: ...
def full_like(input: Tensor, fill_value: float, *, requires_grad: bool = False) -> Tensor: ...
def arange(
    start: float,
    end: float | None = None,
    step: float = 1.0,
    *,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    """``[start, end)`` in steps of ``step``; ``arange(n)`` is ``0..n-1``."""

def linspace(
    start: float, end: float, steps: int, *, device: DeviceLike = None, requires_grad: bool = False
) -> Tensor:
    """``steps`` evenly spaced values from ``start`` to ``end`` inclusive."""

def eye(n: int, *, device: DeviceLike = None, requires_grad: bool = False) -> Tensor: ...
def cat(tensors: Iterable[Tensor], dim: int = 0) -> Tensor:
    """Join tensors along an existing dimension."""

def stack(tensors: Iterable[Tensor], dim: int = 0) -> Tensor:
    """Join same-shape tensors along a new dimension."""

def where(condition: Tensor, input: Operand, other: Operand) -> Tensor:
    """``input`` where ``condition`` is non-zero, else ``other``.

    Computed as ``input * c + other * (1 - c)``, so both must be finite.
    """

def is_grad_enabled() -> bool: ...
def _no_grad_enter() -> None: ...
def _no_grad_exit() -> None: ...
def save(tensors: dict[str, Tensor], path: str | PathLike[str]) -> None:
    """Write tensors to a safetensors file (as F32)."""

def load(path: str | PathLike[str], *, device: DeviceLike = None) -> dict[str, Tensor]:
    """Read a safetensors file (F32, F16, BF16 or F64 converted to float32)."""

class _Layer:
    """A Rust layer; use the classes in ``ferrotensor.nn`` instead."""

    @staticmethod
    def linear(
        in_features: int, out_features: int, bias: bool = True, device: DeviceLike = None
    ) -> _Layer: ...
    @staticmethod
    def embedding(num_embeddings: int, embedding_dim: int, device: DeviceLike = None) -> _Layer: ...
    @staticmethod
    def layer_norm(
        normalized_shape: Sequence[int], eps: float = 1e-5, device: DeviceLike = None
    ) -> _Layer: ...
    @staticmethod
    def rms_norm(dim: int, eps: float = 1e-5, device: DeviceLike = None) -> _Layer: ...
    @staticmethod
    def dropout(p: float = 0.5) -> _Layer: ...
    @staticmethod
    def attention(
        embed_dim: int, num_heads: int, causal: bool = False, device: DeviceLike = None
    ) -> _Layer: ...
    def forward(self, x: Any) -> Tensor: ...
    def named_parameters(self) -> list[tuple[str, Tensor]]: ...
    def set_training(self, training: bool) -> None: ...

# losses and functional layers (re-exported by ferrotensor.nn.functional)
def mse_loss(input: Tensor, target: Tensor | Data) -> Tensor: ...
def l1_loss(input: Tensor, target: Tensor | Data) -> Tensor: ...
def smooth_l1_loss(input: Tensor, target: Tensor | Data, beta: float = 1.0) -> Tensor: ...
def cross_entropy(input: Tensor, target: Tensor | Data) -> Tensor:
    """Mean cross-entropy of logits ``[.., C]``.

    ``target`` holds class indices with shape ``[..]`` (negative indices are
    ignored, e.g. ``-100`` for padding) or probabilities with shape ``[.., C]``.
    """

def nll_loss(input: Tensor, target: Tensor | Data) -> Tensor:
    """Mean negative log-likelihood of class indices under log-probabilities ``[.., C]``."""

def binary_cross_entropy(input: Tensor, target: Tensor | Data) -> Tensor: ...
def binary_cross_entropy_with_logits(input: Tensor, target: Tensor | Data) -> Tensor: ...
def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor: ...
def layer_norm(
    input: Tensor, weight: Tensor | None = None, bias: Tensor | None = None, eps: float = 1e-5
) -> Tensor:
    """Layer normalization over the last dimension."""

def rms_norm(input: Tensor, weight: Tensor | None = None, eps: float = 1e-5) -> Tensor:
    """RMS normalization over the last dimension."""

def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """``input @ weight + bias`` with ``weight`` of shape ``[in, out]``."""

def scaled_dot_product_attention(
    query: Tensor, key: Tensor, value: Tensor, is_causal: bool = False
) -> Tensor:
    """``softmax(q k^T / sqrt(d)) v`` for q ``[.., Lq, d]``, k ``[.., Lk, d]``, v ``[.., Lk, dv]``."""

def one_hot(indices: Tensor | Sequence[int], num_classes: int, *, device: DeviceLike = None) -> Tensor: ...
def accuracy(logits: Tensor, target: Tensor | Sequence[int]) -> float:
    """Fraction of rows of ``[N, C]`` logits whose argmax is the target."""

def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = -1) -> Tensor: ...

# optim
class SGD:
    """Stochastic gradient descent with optional momentum, Nesterov and weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None: ...
    def step(self) -> None:
        """Update every parameter that has a ``.grad``."""
    def zero_grad(self) -> None:
        """Set every parameter's ``.grad`` to ``None``."""
    lr: float

class AdamW:
    """Adam with decoupled weight decay; ``max_grad_norm`` clips gradients before each step."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        max_grad_norm: float | None = None,
    ) -> None: ...
    def step(self) -> None: ...
    def zero_grad(self) -> None: ...
    lr: float
    @property
    def steps(self) -> int:
        """Number of steps taken."""

def clip_grad_norm(params: Iterable[Tensor], max_norm: float) -> float:
    """Scale gradients so their global L2 norm is at most ``max_norm``; returns the norm before clipping."""

def clip_grad_value(params: Iterable[Tensor], clip_value: float) -> None:
    """Clamp every gradient entry to ``[-clip_value, clip_value]``."""

class _Schedule:
    @staticmethod
    def constant(lr: float) -> _Schedule: ...
    @staticmethod
    def step(base_lr: float, step_size: int, gamma: float) -> _Schedule: ...
    @staticmethod
    def cosine(base_lr: float, min_lr: float, total_steps: int) -> _Schedule: ...
    @staticmethod
    def warmup_cosine(peak_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> _Schedule: ...
    @staticmethod
    def warmup_linear(peak_lr: float, end_lr: float, warmup_steps: int, total_steps: int) -> _Schedule: ...
    def lr_at(self, step: int) -> float: ...
