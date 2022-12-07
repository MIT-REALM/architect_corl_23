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
from architect.systems.power_systems.example_systems.acopf.simple_3_bus import (
    make_3_bus_network,
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
    repair_steps = 5
    repair_lr = 1e-6
    sys = make_3_bus_network(L)

    # Let's start by trying to solve the ACOPF problem for the nominal system
    network = sys.nominal_network
    prior_logprob = sys.dispatch_prior_logprob
    # Remember the negative to make lower costs more likely!
    posterior_logprob = lambda dispatch: -sys(
        dispatch, network, repair_steps, repair_lr
    ).potential

    # Make a MALA kernel for MCMC sampling
    mala_step_size = 1e-5
    n_samples = 30_000
    warmup_samples = 20_000
    n_chains = 10
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
    result = jax.vmap(sys, in_axes=(0, None, None, None))(
        best_dispatch, network, 100, repair_lr
    )

    # Plot the results
    fig = plt.figure(figsize=(16, 16), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["constraints", "cost"],
            ["trace", "logprob_hist"],
            ["generation", "generation"],
        ]
    )

    # Plot the violations at the best dispatch from each chain
    sns.swarmplot(
        data=[
            result.P_violation,
            result.Q_violation,
            result.V_violation,
        ],
        ax=axs["constraints"],
    )
    axs["constraints"].set_xticklabels(["P", "Q", "V"])
    axs["constraints"].set_ylabel("Constraint Violation")

    sns.histplot(x=result.generation_cost, ax=axs["cost"])
    axs["cost"].set_xlabel("Generation cost")

    # Plot the chain convergence
    axs["trace"].plot(logprobs.T)
    axs["trace"].set_xlabel("# Samples")
    axs["trace"].set_ylabel("Overall log probability")
    axs["trace"].vlines(
        warmup_samples, *axs["trace"].get_ylim(), colors="k", linestyles="dashed"
    )

    # Plot the logprob histogram
    sns.histplot(x=logprobs[:, warmup_samples:].flatten(), ax=axs["logprob_hist"])
    axs["logprob_hist"].set_xlabel("Log probability")

    # Plot the generations along with their limits
    bus = jnp.arange(sys.n_bus)
    P_min, P_max = sys.bus_active_limits.T
    for i in range(n_chains):
        P = result.P[i, :]
        lower_error = P - P_min
        upper_error = P_max - P
        errs = jnp.vstack((lower_error, upper_error))
        axs["generation"].errorbar(
            bus,
            P,
            yerr=errs,
            linestyle="None",
            marker="o",
            markersize=10,
            linewidth=3.0,
            capsize=10.0,
            capthick=3.0,
        )
    axs["generation"].set_xlabel("Active power injection (p.u.)")

    plt.savefig(
        (
            f"plts/acopf/3bus/L_{L:0.1e}_{n_samples}_samples_"
            f"{n_chains}_chains_mala_step_{mala_step_size:0.1e}_"
            f"repair_steps_{repair_steps}_repair_lr_{repair_lr:0.1e}"
            ".png"
        )
    )
