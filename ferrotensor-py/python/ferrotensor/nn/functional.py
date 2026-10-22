"""Functional ops and losses, in the style of ``torch.nn.functional``.

Losses average over all elements (or, for class-index targets, over the
non-ignored positions).
"""

from __future__ import annotations

from ferrotensor._ferrotensor import (
    Tensor,
    accuracy,
    binary_cross_entropy,
    binary_cross_entropy_with_logits,
    cosine_similarity,
    cross_entropy,
    dropout,
    l1_loss,
    layer_norm,
    linear,
    mse_loss,
    nll_loss,
    one_hot,
    rms_norm,
    scaled_dot_product_attention,
    smooth_l1_loss,
)


def relu(input: Tensor) -> Tensor:
    """``max(input, 0)``."""
    return input.relu()


def leaky_relu(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    """``input`` where positive, else ``negative_slope * input``."""
    return input.leaky_relu(negative_slope)


def gelu(input: Tensor) -> Tensor:
    """Gaussian error linear unit."""
    return input.gelu()


def silu(input: Tensor) -> Tensor:
    """``input * sigmoid(input)``."""
    return input.silu()


def sigmoid(input: Tensor) -> Tensor:
    """Logistic sigmoid."""
    return input.sigmoid()


def tanh(input: Tensor) -> Tensor:
    """Hyperbolic tangent."""
    return input.tanh()


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    """Softmax along ``dim``."""
    return input.softmax(dim)


def log_softmax(input: Tensor, dim: int = -1) -> Tensor:
    """Log-softmax along ``dim``."""
    return input.log_softmax(dim)


__all__ = [
    "accuracy",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "cosine_similarity",
    "cross_entropy",
    "dropout",
    "gelu",
    "l1_loss",
    "layer_norm",
    "leaky_relu",
    "linear",
    "log_softmax",
    "mse_loss",
    "nll_loss",
    "one_hot",
    "relu",
    "rms_norm",
    "scaled_dot_product_attention",
    "sigmoid",
    "silu",
    "smooth_l1_loss",
    "softmax",
    "tanh",
]
