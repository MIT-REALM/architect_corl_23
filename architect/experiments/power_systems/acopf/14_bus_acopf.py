import json
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
from jaxtyping import Array, Float, Integer, jaxtyped, Shaped

from architect.systems.power_systems.acopf import Dispatch, Network
from architect.systems.power_systems.example_systems.acopf.load_test_networks import (
    load_test_network,
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
    L = 50.0
    sys = load_test_network("case14", penalty=L)

    # Let's start by trying to solve the ACOPF problem for the nominal system
    network = sys.network_spec.nominal_network
    prior_logprob = sys.dispatch_prior_logprob
    # Remember the negative to make lower costs more likely!
    posterior_logprob = lambda dispatch: -sys(dispatch, network).potential

    # Make a MALA kernel for MCMC sampling
    mala_step_size = 1e-5
    n_samples = 200
    warmup_samples = 200
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
    result = jax.vmap(sys, in_axes=(0, None))(best_dispatch, network)

    # Plot the results
    fig = plt.figure(figsize=(32, 16), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["constraints", "cost"],
            ["trace", "logprob_hist"],
            ["generation", "voltage"],
        ]
    )

    # Plot the violations at the best dispatch from each chain
    sns.swarmplot(
        data=[
            result.P_gen_violation.sum(axis=-1),
            result.Q_gen_violation.sum(axis=-1),
            result.P_load_violation.sum(axis=-1),
            result.Q_load_violation.sum(axis=-1),
            result.V_violation.sum(axis=-1),
        ],
        ax=axs["constraints"],
    )
    axs["constraints"].set_xticklabels(["Pg", "Qg", "Pd", "Qd", "V"])
    axs["constraints"].set_ylabel("Constraint Violation")

    # Plot generation cost vs constraint violation
    total_constraint_violation = (
        result.P_gen_violation.sum(axis=-1)
        + result.Q_gen_violation.sum(axis=-1)
        + result.P_load_violation.sum(axis=-1)
        + result.Q_load_violation.sum(axis=-1)
        + result.V_violation.sum(axis=-1)
    )
    axs["cost"].scatter(result.generation_cost, total_constraint_violation)
    axs["cost"].set_xlabel("Generation cost")
    axs["cost"].set_ylabel("Total constraint violation")

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
    bus = sys.gen_spec.buses
    P_min, P_max = sys.gen_spec.P_limits.T
    for i in range(n_chains):
        P = result.dispatch.gen.P[i, :]
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
    axs["generation"].set_ylabel("$P_g$ (p.u.)")

    # Plot the voltages along with their limits
    bus = jnp.arange(sys.n_bus)
    V_min, V_max = sys.bus_voltage_limits.T
    for i in range(n_chains):
        V = result.voltage_amplitudes[i, :]
        lower_error = V - V_min
        upper_error = V_max - V
        errs = jnp.vstack((lower_error, upper_error))
        axs["voltage"].errorbar(
            bus,
            V,
            yerr=errs,
            linestyle="None",
            marker="o",
            markersize=10,
            linewidth=3.0,
            capsize=10.0,
            capthick=3.0,
        )
    axs["voltage"].set_ylabel("$|V|$ (p.u.)")

    filename = (
        f"results/acopf/14bus/dispatch_L_{L:0.1e}_{n_samples}_samples_"
        f"{n_chains}_chains_mala_step_{mala_step_size:0.1e}"
    )
    plt.savefig(filename + ".png")

    # Save the dispatch
    with open(filename + ".json", "w") as f:
        json.dump(
            {"dispatch": best_dispatch._asdict(), "network": network._asdict()},
            f,
            default=lambda x: x.tolist() if isinstance(x, Shaped[Array, "..."]) else x,
        )
