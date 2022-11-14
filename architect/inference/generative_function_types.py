"""
Define types for pre- and post-compilation generative functions.

A generative function is a function that models a stochastic process by making random
choices during its execution.
"""
from typing import Any, Tuple
from typing_extensions import Protocol

import jax.numpy as jnp

from .choice_database import ChoiceDatabase


class GenerativeFunction(Protocol):
    """
    A generative function takes a vector of inputs and the values of any fixed random
    choices, and it returns an output vector, the log likelihood of that output, the
    trace containing the values of any random choices made, and the database tracking
    the indices of choices in the trace.
    """

    def __call__(
        self,
        *args: Any,
        selection: jnp.ndarray,
        predetermined_values: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, ChoiceDatabase]:
        ...


class CompiledGenerativeFunction(Protocol):
    """
    Before using a generative function, it needs to be "compiled" to not return the
    choice database, since that's not a JAX-compatible type.
    """

    def __call__(
        self,
        *args: Any,
        selection: jnp.ndarray,
        predetermined_values: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ...
