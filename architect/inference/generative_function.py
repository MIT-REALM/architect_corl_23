"""
Define generative function types.

A generative function is a function that simulates some stochastic process, usually
by making random choices from some prior distribution. To be useful for probabilistic
programming, a generative function should record the choices made as well as the log
probability of each choice (allowing us to compute the log probability of an overall
execution). These choices (and their logprobs) should be recorded in a *trace*, which is
returned alongside the function result.

In addition to returning a trace alongside its result, a generative function should
take (in addition to its regular arguments) a choice map that optionally specifies the
values of particular random choices made by the function. This will allow us to sample
from the posterior distribution of the function conditioned on some observations.

Log probabilities need not be normalized.

TODO How to make probabilistic programming compatible with JAX? I think I need to
restrict to finite support programs (i.e. we can enumerate the choices made and that
list is constant from execution to execution)?
"""
from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from immutables import Map
from jax._src.prng import PRNGKeyArray
from typing_extensions import Protocol

#####################################
# Types
#####################################

# Random choices are tracked using unique identifiers. Choice values can be
# either scalar or multivariate, but in either case are represented using DeviceArrays
Choice = str
Value = jnp.DeviceArray


# A generative function maps arguments and a map of conditioned choices to a result,
# a map of all choices (including conditioned ones) made during execution, and a log
# probability of execution.
#
# We'll call the map of conditioned choices the "choicemap", and we'll call the map of
# choices made during execution the "trace".
class GenerativeFunction(Protocol):
    def __call__(
        self, *args: Any, choicemap: Map, key: PRNGKeyArray
    ) -> Tuple[Any, Map, jnp.DeviceArray]:
        ...


#####################################
# Helper functions
#####################################

# Whenever a generative function wants to make a random choice, it needs to check
# whether that choice has been pre-determined in the choicemap, so let's add a helper
# function to take care of that
def make_random_choice(
    choice: Choice,
    choicemap: Map,
    trace: Map,
    random_fn: Callable[[PRNGKeyArray], jnp.DeviceArray],
    logprob_fn: Callable[[Value], jnp.DeviceArray],
    prng_key: PRNGKeyArray,
) -> Tuple[Value, Map, jnp.DeviceArray]:
    """
    Make a random choice, or load a pre-determined choice if specified in the choicemap.

    Saves the choice to the trace along with its value and log probability. Returns the
    extended trace and consumes the given PRNG Key.

    args:
        choice: the string ID of the choice being made
        choicemap: specifies any pre-determined choices
        trace: the function trace
        random_function: the function that generates the value of the
            random choice.
        logprob_fn: the function that returns the log probability of the chosen value.
    returns:
        the value of the random choice, the new trace, and the log probability of that
        choice
    """
    # If we're assigning a choice that has been made before, that's an error
    if choice in trace:
        raise RuntimeError(f"Cannot assign to existing choice: {choice}")

    # If the choice has been specified in the choicemap, use that value
    if choice in choicemap:
        value = choicemap[choice]
    else:  # Otherwise, make a random choice
        value = random_fn(prng_key)

    # Either way, make sure to save the value in the trace
    # and return the value, trace, and logprob of the choice
    logprob = logprob_fn(value)
    trace = trace.set(choice, value)
    return value, trace, logprob
