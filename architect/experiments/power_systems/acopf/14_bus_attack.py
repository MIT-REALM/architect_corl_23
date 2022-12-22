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
from jaxtyping import Array, Float, Integer, jaxtyped

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
    L = 100.0
    repair_steps = 5
    repair_lr = 1e-6
    sys = load_test_network("case14", penalty=L)

    # Let's assume that we've already solved for a feasible dispatch for the ACOPF
    # problem, so we can load it from a file. Pick one dispatch arbitrarily
    # from the proposed solutions
    dispatch_filename = (
        "results/acopf/14bus/dispatch_L_1.0e+02_30000_samples_"
        "10_chains_mala_step_1.0e-05_repair_steps_5_repair_lr_1.0e-06.json"
    )
    with open(dispatch_filename, "r") as f:
        dispatch_json = json.load(f)
        dispatch = Dispatch(
            jnp.array(dispatch_json["best_dispatch"]["voltage_amplitudes"])[0],
            jnp.array(dispatch_json["best_dispatch"]["voltage_angles"])[0],
        )

    # Now the problem is to find a likely perturbation to the network that will
    # cause this dispatch to fail
    prior_logprob = sys.network_prior_logprob
    # Positive to make higher costs more likely!
    posterior_logprob = lambda network: sys(
        dispatch, network, repair_steps, repair_lr
    ).potential

    # Make a MALA kernel for MCMC sampling
    mala_step_size = 1e-5
    n_samples = 2_000
    warmup_samples = 2_000
    n_chains = 10
    mala = blackjax.mala(
        lambda x: prior_logprob(x) + posterior_logprob(x), mala_step_size
    )

    # Make some random keys to use as we go
    key = jrandom.PRNGKey(0)

    # Initialize the network randomly
    key, network_key = jrandom.split(key)
    network_keys = jrandom.split(network_key, n_chains)
    initial_network = jax.vmap(sys.sample_random_network)(network_keys)

    # Initialize the sampler
    initial_state = jax.vmap(mala.init)(initial_network)

    # Run the chains
    key, sample_key = jrandom.split(key)
    sample_keys = jrandom.split(sample_key, n_chains)
    t_start = time.perf_counter()
    samples, logprobs = jax.vmap(inference_loop, in_axes=(0, None, 0, None))(
        sample_keys, mala.step, initial_state, n_samples + warmup_samples
    )
    t_end = time.perf_counter()
    print(f"Ran {n_chains} chains of {n_samples:,} samples in {t_end - t_start:.2f} s")

    # Get the most likely network from each chain
    most_likely_idx = jnp.argmax(logprobs, axis=-1)
    best_network = jtu.tree_map(
        lambda leaf: leaf[jnp.arange(0, n_chains), most_likely_idx],
        samples,
    )
    # The dispatcher is allowed to locally repair their dispatch in response
    result = jax.vmap(sys, in_axes=(None, 0, None, None))(
        dispatch, best_network, 10_000, repair_lr
    )

    # Plot the results
    fig = plt.figure(figsize=(32, 16), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["constraints", "violation_vs_logprob"],
            ["trace", "logprob_hist"],
            ["conductances", "susceptances"],
        ]
    )

    # Plot the violations found by each chain
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

    prior = jax.vmap(prior_logprob)(best_network)
    violation = result.P_violation + result.Q_violation + result.V_violation
    sns.scatterplot(x=prior, y=100 * violation, ax=axs["violation_vs_logprob"])
    axs["violation_vs_logprob"].set_xlabel("Prior log likelihood")
    axs["violation_vs_logprob"].set_ylabel("Severity (MW)")

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

    # Plot the line conductances vs their nominal values.
    # Scale the size of markers by the extent of total violation caused
    total_violation = result.P_violation + result.Q_violation + result.V_violation
    min_violation, max_violation = total_violation.min(), total_violation.max()
    violation_range = max_violation - min_violation
    markersize = 1 + 200 * (total_violation - min_violation) / violation_range
    line = jnp.arange(sys.n_line)
    nominal_conductances = sys.nominal_network.line_conductances
    nominal_susceptances = sys.nominal_network.line_susceptances
    axs["conductances"].scatter(
        line,
        nominal_conductances,
        marker="_",
        color="k",
        s=500,
        label="Nominal Conductance",
    )
    axs["susceptances"].scatter(
        line,
        nominal_susceptances,
        marker="_",
        color="k",
        s=500,
        label="Nominal Susceptance",
    )
    for i in range(n_chains):
        conductances = best_network.line_conductances[i]
        susceptances = best_network.line_susceptances[i]
        axs["conductances"].scatter(line, conductances, marker="o", s=markersize[i])
        axs["susceptances"].scatter(line, susceptances, marker="o", s=markersize[i])

    axs["conductances"].set_ylabel("Conductance (p.u.)")
    axs["conductances"].set_xlabel("Line")
    axs["conductances"].set_xticks(line)
    axs["conductances"].set_xticklabels([f"{i}->{j}" for i, j in sys.lines])
    axs["susceptances"].set_ylabel("Susceptance (p.u.)")
    axs["susceptances"].set_xlabel("Line")
    axs["susceptances"].set_xticks(line)
    axs["susceptances"].set_xticklabels([f"{i}->{j}" for i, j in sys.lines])

    filename = (
        f"results/acopf/14bus/attack_L_{L:0.1e}_{n_samples}_samples_"
        f"{n_chains}_chains_mala_step_{mala_step_size:0.1e}_"
        f"repair_steps_{repair_steps}_repair_lr_{repair_lr:0.1e}"
    )
    plt.show()
    # plt.savefig(filename + ".png")

    # # Save the attack
    # network_serializable = {
    #     "line_conductances": best_network.line_conductances.tolist(),
    #     "line_susceptances": best_network.line_susceptances.tolist(),
    # }
    # with open(filename + ".json", "w") as f:
    #     json.dump({"best_network": network_serializable}, f)
