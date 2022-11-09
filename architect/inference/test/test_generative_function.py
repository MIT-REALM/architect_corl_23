import jax
import jax.numpy as jnp
from jax.random import normal
import jax.scipy.stats as jaxstats
from jax._src.prng import PRNGKeyArray

from architect.inference import (
    Choice,
    Choicemap,
    Trace,
    Value,
    make_random_choice,
    trace_logprob,
)


def example_generative_function(
    prior_mean: jnp.DeviceArray, choicemap: Choicemap, key: PRNGKeyArray
):
    """
    An example generative function that makes some random choices.

    These choices are organizes hierarchically as follows:
        z_1 ~ N(prior_mean, 1.0)
        z_2 ~ N(z_1, 1.0)

    args:
        prior_mean: the mean of z_1
        choicemap: the choicemap for this generative function
        key: the key to use to generate random numbers

    returns:
        z_2
    """
    # All generative functions need to start by making a trace
    trace = {}

    # First choose a population mean from the prior mean
    random_fn = lambda key: normal(key) + prior_mean
    logprob_fn = lambda value: jaxstats.norm.logpdf(value, loc=prior_mean, scale=1.0)
    key, subkey = jax.random.split(key)
    z_1, trace = make_random_choice(
        "z_1", choicemap, trace, random_fn, logprob_fn, subkey
    )

    # Then choose a sample with this population mean
    random_fn = lambda key: normal(key) + z_1
    logprob_fn = lambda value: jaxstats.norm.logpdf(value, loc=z_1, scale=1.0)
    key, subkey = jax.random.split(key)
    z_2, trace = make_random_choice(
        "z_2", choicemap, trace, random_fn, logprob_fn, subkey
    )

    return z_2, trace


def test_example_generative_function():
    """Test the example generative function"""
    key = jax.random.PRNGKey(0)
    prior_mean = jnp.array(0.0)

    # First check if the function works with an empty choicemap
    _, trace = example_generative_function(prior_mean, {}, key)

    # Make sure all the right choices have been made
    assert "z_1" in trace and "z_2" in trace

    # Now see what happens if we provide some choices
    _, trace = example_generative_function(prior_mean, {"z_1": jnp.array(0.0)}, key)
    assert trace["z_1"][0] == 0.0

    _, trace = example_generative_function(
        prior_mean, {"z_1": jnp.array(0.0), "z_2": jnp.array(0.0)}, key
    )
    assert trace["z_1"][0] == 0.0
    assert trace["z_2"][0] == 0.0
