"""Common utility functions."""
from beartype import beartype
from jax.nn import logsumexp
from jax.nn import log_sigmoid
from jaxtyping import Array, Float, jaxtyped


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


def log_smooth_uniform(x, x_min, x_max, smoothing):
    """Return the smooth approximation of uniform distribution for log probabilities"""
    return log_sigmoid(smoothing * (x - x_min)) + log_sigmoid(smoothing * (x_max - x))
