"""Common utility functions."""
from jax.nn import logsumexp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype


@jaxtyped
@beartype
def softmax(x: Float[Array, "..."], sharpness: float = 0.05):
    """Return the soft maximum of the given vector"""
    return 1 / sharpness * logsumexp(sharpness * x)


@jaxtyped
@beartype
def softmin(x: Float[Array, "..."], sharpness: float = 0.05):
    """Return the soft minimum of the given vector"""
    return -softmax(-x, sharpness)
