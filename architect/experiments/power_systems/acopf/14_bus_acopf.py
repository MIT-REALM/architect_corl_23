import time

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import seaborn as sns
from beartype import beartype
from beartype.typing import Tuple, Union
from jaxtyping import Array, Float, Integer, jaxtyped

from architect.systems.power_systems.acopf import Dispatch, Network
from architect.systems.power_systems.example_systems.acopf.ieee_14_bus import (
    make_14_bus_network,
)


@jaxtyped
@beartype
def inference_loop(
    rng_key: Integer[Array, "..."],
    kernel,
    initial_state,
    num_samples: int,
) -> Tuple[Union[Dispatch, Network], Float[Array, "..."]]:
    """
    Run the given kernel for the specified number of steps, returning the samples.

    args:
        rng_key: JAX rng key
        kernel: the MCMC kernel to run
        initial_state: the starting state for the sampler
        num_samples: how many samples to take
    """

    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jrandom.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states.position, states.logprob


if __name__ == "__main__":
    # Make the test system
    L = 100.0
    sys = make_14_bus_network(L)

    # Let's start by trying to solve the ACOPF problem for the nominal system
    network = sys.nominal_network
    prior_logprob = sys.dispatch_prior_logprob
    # Remember the negative to make lower costs more likely!
    posterior_logprob = lambda dispatch: -sys(dispatch, network).potential

    # Make a MALA kernel for MCMC sampling
    mala_step_size = 1e-4
    n_samples = 10_000_000
    warmup_samples = 1_000_000
    n_chains = 1
    mala = blackjax.mala(
        lambda x: prior_logprob(x) + posterior_logprob(x), mala_step_size
    )

    # Make some random keys to use as we go
    key = jrandom.PRNGKey(0)

    # Initialize the dispatch randomly
    key, dispatch_key = jrandom.split(key)
    dispatch_keys = jrandom.split(dispatch_key, n_chains)
    initial_dispatch = jax.vmap(sys.sample_random_dispatch)(dispatch_keys)

    # Initialize the sampler
    initial_state = jax.vmap(mala.init)(initial_dispatch)

    # Run the chains
    key, sample_key = jrandom.split(key)
    sample_keys = jrandom.split(sample_key, n_chains)
    t_start = time.perf_counter()
    samples, logprobs = jax.vmap(inference_loop, in_axes=(0, None, 0, None))(
        sample_keys, mala.step, initial_state, n_samples + warmup_samples
    )
    t_end = time.perf_counter()
    print(f"Ran {n_chains} chains of {n_samples:,} samples in {t_end - t_start:.2f} s")

    # Get the most likely dispatch from each chain
    most_likely_idx = jnp.argmax(logprobs, axis=-1)
    best_dispatch = jtu.tree_map(
        lambda leaf: leaf[jnp.arange(0, n_chains), most_likely_idx],
        samples,
    )
    result = jax.vmap(sys, in_axes=(0, None))(best_dispatch, network)

    # Plot the results
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))

    # Plot the violations at the best dispatch from each chain
    sns.swarmplot(
        data=[
            result.P_violation,
            result.Q_violation,
            result.V_violation,
        ],
        ax=axs[0, 0],
    )
    axs[0, 0].set_xticklabels(["P", "Q", "V"])
    axs[0, 0].set_ylabel("Constraint Violation")

    sns.histplot(x=result.generation_cost, ax=axs[0, 1])
    axs[0, 1].set_xlabel("Generation cost")

    # Plot the chain convergence
    axs[1, 0].plot(logprobs.T)
    axs[1, 0].set_xlabel("# Samples")
    axs[1, 0].set_ylabel("Overall log probability")
    axs[1, 0].vlines(
        warmup_samples, *axs[1, 0].get_ylim(), colors="k", linestyles="dashed"
    )

    # Plot the logprob histogram
    sns.histplot(x=logprobs[:, warmup_samples:].flatten(), ax=axs[1, 1])
    axs[1, 1].set_xlabel("Log probability")

    plt.savefig(
        (
            f"plts/acopf/14bus/L_{L:0.1e}_{n_samples}_samples_"
            f"{n_chains}_chains_lr_{mala_step_size:0.1e}.png"
        )
    )
