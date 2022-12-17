import json
import time

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jax.nn import logsumexp
import matplotlib.pyplot as plt
import seaborn as sns
from beartype import beartype
from beartype.typing import Tuple, Union, Callable
from jaxtyping import Array, Float, Integer, jaxtyped

from architect.systems.power_systems.acopf import Dispatch, Network, ACOPF
from architect.systems.power_systems.example_systems.acopf.ieee_14_bus import (
    make_14_bus_network,
)


# Define convenience types
Params = Union[Dispatch, Network]
Likelihood = Callable[[Params], Float[Array, ""]]


def softmax(x: Float[Array, "..."], smoothing: float = 5):
    """Return the soft maximum of the given vector"""
    return 1 / smoothing * logsumexp(smoothing * x)


def softmin(x: Float[Array, "..."], smoothing: float = 5):
    """Return the soft minimum of the given vector"""
    return -softmax(-x, smoothing)


@jaxtyped
@beartype
def compile_dispatch_logprobs(
    sys: ACOPF,
    networks: Network,
    repair_steps: int = 5,
    repair_lr: float = 1e-6,
) -> Tuple[Likelihood, Likelihood]:
    """
    Compile the prior and posterior log likelihood functions for the dispatch.

    args:
        sys: the ACOPF problem being solved
        networks: the set of network contingencies to optimize against
        repair_steps: how many local repair steps the operator is permitted
        repair_lr: the stepsize for local repair steps
    returns:
        A tuple containing the prior and posterior log likelihood functions
    """
    prior_logprob = sys.dispatch_prior_logprob

    # Negative to make lower costs more likely!
    posterior_logprob_single_network = lambda dispatch, network: sys(
        dispatch, network, repair_steps, repair_lr
    ).potential
    posterior_logprob = lambda dispatch: -jax.vmap(
        posterior_logprob_single_network, in_axes=(None, 0)
    )(dispatch, networks).mean()

    return prior_logprob, posterior_logprob


@jaxtyped
@beartype
def compile_network_logprobs(
    sys: ACOPF,
    dispatches: Dispatch,
    repair_steps: int = 5,
    repair_lr: float = 1e-6,
) -> Tuple[Likelihood, Likelihood]:
    """
    Compile the prior and posterior log likelihood functions for the network.

    args:
        sys: the ACOPF problem being solved
        dispatches: the set of dispatches to optimize against
        repair_steps: how many local repair steps the operator is permitted
        repair_lr: the stepsize for local repair steps
    returns:
        A tuple containing the prior and posterior log likelihood functions
    """
    prior_logprob = sys.network_prior_logprob

    # Positive to make higher costs more likely!
    posterior_logprob_single_dispatch = lambda dispatch, network: sys(
        dispatch, network, repair_steps, repair_lr
    ).potential
    posterior_logprob = lambda network: softmin(
        jax.vmap(posterior_logprob_single_dispatch, in_axes=(0, None))(
            dispatches, network
        )
    )

    return prior_logprob, posterior_logprob


@jaxtyped
@beartype
def run_chain(
    rng_key: Integer[Array, "..."],
    kernel,
    initial_state,
    num_samples: int,
) -> Tuple[Union[Dispatch, Network], Float[Array, "..."]]:
    """
    Run the given kernel for the specified number of steps, returning last sample
    and associated log likelihood.

    args:
        rng_key: JAX rng key
        kernel: the MCMC kernel to run
        initial_state: the starting state for the sampler
        num_samples: how many samples to take
    returns:
        the last state of the chain and associated log probability
    """

    @jax.jit
    def one_step(_, carry):
        # unpack carry
        state, rng_key = carry

        # Take a step
        rng_key, subkey = jrandom.split(rng_key)
        state, _ = kernel(subkey, state)

        # Re-pack the carry and return
        carry = (state, rng_key)
        return carry

    initial_carry = (initial_state, rng_key)
    states, _ = jax.lax.fori_loop(0, num_samples, one_step, initial_carry)

    return states.position, states.logprob


@jaxtyped
@beartype
def run_smc(
    rng_key: Integer[Array, "..."],
    sys: ACOPF,
    dispatches: Dispatch,
    networks: Network,
    num_smc_steps: int,
    num_substeps: int,
    step_size: float,
    repair_steps: int = 5,
    repair_lr: float = 1e-6,
) -> Tuple[Dispatch, Float[Array, " n"], Network, Float[Array, " n"]]:
    """
    Run the sequential monte carlo algorithm for SC-ACOPF

    args:
        rng_key: JAX rng key
        sys: the ACOPF system to optimize
        dispatches: the initial population of dispatches
        networks: the initial population of networks
        num_smc_steps: how many steps to take for the top-level SMC algorithm
        num_substeps: the number of steps to run for each set of MCMC chains within
            this step
        stepsize: the size of MCMC step to take on each substep
        repair_steps: how many local repair steps the operator is permitted
        repair_lr: the stepsize for local repair steps
    returns:
        - The updated population of dispatches
        - The overall log probabilities for the dispatches (before the contingencies are
            generated)
        - The updated population of network contingencies
        - The overall log probabilities for the contingencies
    """
    # Make the function to run one step and jit it
    @jax.jit
    def one_smc_step(_, carry):
        # Unpack the carry
        rng_key, dispatches, _, networks, _ = carry

        # Compile dispatch logprobs and MCMC kernel
        dispatch_prior, dispatch_posterior = compile_dispatch_logprobs(
            sys, networks, repair_steps, repair_lr
        )
        dispatch_sampler = blackjax.mala(
            lambda x: dispatch_prior(x) + dispatch_posterior(x), step_size
        )
        # Initialize the dispatch chains
        initial_dispatch_state = jax.vmap(dispatch_sampler.init)(dispatches)
        # Run dispatch chains and update dispatches
        n_dispatch_chains = dispatches.voltage_amplitudes.shape[0]
        rng_key, sample_key = jrandom.split(rng_key)
        sample_keys = jrandom.split(sample_key, n_dispatch_chains)
        dispatches, dispatch_logprobs = jax.vmap(run_chain, in_axes=(0, None, 0, None))(
            sample_keys, dispatch_sampler.step, initial_dispatch_state, num_substeps
        )

        # Compile network logprobs
        network_prior, network_posterior = compile_network_logprobs(
            sys, dispatches, repair_steps, repair_lr
        )
        network_sampler = blackjax.mala(
            lambda x: network_prior(x) + network_posterior(x), step_size
        )
        # Initialize the network chains
        initial_network_state = jax.vmap(network_sampler.init)(networks)
        # Run network chains and update networks
        n_network_chains = networks.line_conductances.shape[0]
        rng_key, sample_key = jrandom.split(rng_key)
        sample_keys = jrandom.split(sample_key, n_network_chains)
        networks, network_logprobs = jax.vmap(run_chain, in_axes=(0, None, 0, None))(
            sample_keys, network_sampler.step, initial_network_state, num_substeps
        )

        # Return the updated carry
        return (rng_key, dispatches, dispatch_logprobs, networks, network_logprobs)

    # Run the appropriate number of steps
    n_dispatches = dispatches.voltage_amplitudes.shape[0]
    n_networks = networks.line_conductances.shape[0]
    initial_carry = (
        rng_key,
        dispatches,
        jnp.zeros(n_dispatches),
        networks,
        jnp.zeros(n_networks),
    )
    _, dispatches, dispatch_logprobs, networks, network_logprobs = jax.lax.fori_loop(
        0, num_smc_steps, one_smc_step, initial_carry
    )

    return dispatches, dispatch_logprobs, networks, network_logprobs


if __name__ == "__main__":
    # Make some random keys to use as we go
    key = jrandom.PRNGKey(0)

    # Make the test system
    L = 100.0
    repair_steps = 5
    repair_lr = 1e-6
    sys = make_14_bus_network(L)

    # Define the SMC parameters
    n_dispatches = 10
    n_networks = 10
    n_mcmc_substeps = 1000
    mcmc_step_size = 1e-5
    n_smc_steps = 10

    # Initialize dispatch and network populations from the prior
    key, dispatch_key = jrandom.split(key)
    dispatch_keys = jrandom.split(dispatch_key, n_dispatches)
    initial_dispatches = jax.vmap(sys.sample_random_dispatch)(dispatch_keys)

    key, network_key = jrandom.split(key)
    network_keys = jrandom.split(network_key, n_networks)
    initial_networks = jax.vmap(sys.sample_random_network)(network_keys)

    print(
        (
            f"Initialized {initial_dispatches.voltage_amplitudes.shape[0]} dispatches "
            f"and {initial_networks.line_conductances.shape[0]} network contingencies."
        )
    )

    # Run the SMC algorithm
    t_start = time.perf_counter()
    dispatches, dispatch_logprobs, networks, network_logprobs = run_smc(
        key,
        sys,
        initial_dispatches,
        initial_networks,
        n_smc_steps,
        n_mcmc_substeps,
        mcmc_step_size,
        repair_steps,
        repair_lr,
    )
    t_end = time.perf_counter()
    print(
        (
            f"Ran {n_smc_steps} SMC rounds of {n_mcmc_substeps} substeps "
            f"in {t_end - t_start:.2f} s"
        )
    )

    # Get the most likely dispatch
    most_likely_dispatch_idx = jnp.argmax(dispatch_logprobs, axis=-1)
    best_dispatch = jtu.tree_map(
        lambda leaf: leaf[most_likely_dispatch_idx],
        dispatches,
    )

    # Evaluate the performance of this dispatch against each proposed contingency,
    # allowing the operator to locally repair as much as they can
    result = jax.vmap(sys, in_axes=(None, 0, None, None))(
        best_dispatch, networks, 10_000, repair_lr
    )

    # Plot the results
    fig = plt.figure(figsize=(32, 16), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["constraints", "violation_vs_logprob"],
            # ["trace", "logprob_hist"],
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

    prior = jax.vmap(sys.network_prior_logprob)(networks)
    violation = result.P_violation + result.Q_violation + result.V_violation
    sns.scatterplot(x=prior, y=100 * violation, ax=axs["violation_vs_logprob"])
    axs["violation_vs_logprob"].set_xlabel("Prior log likelihood")
    axs["violation_vs_logprob"].set_ylabel("Severity (MW)")

    # # Plot the chain convergence
    # axs["trace"].plot(logprobs.T)
    # axs["trace"].set_xlabel("# Samples")
    # axs["trace"].set_ylabel("Overall log probability")
    # axs["trace"].vlines(
    #     warmup_samples, *axs["trace"].get_ylim(), colors="k", linestyles="dashed"
    # )

    # # Plot the logprob histogram
    # sns.histplot(x=logprobs[:, warmup_samples:].flatten(), ax=axs["logprob_hist"])
    # axs["logprob_hist"].set_xlabel("Log probability")

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
    for i in range(n_networks):
        conductances = networks.line_conductances[i]
        susceptances = networks.line_susceptances[i]
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
        f"results/acopf/14bus/mean_scopf_L_{L:0.1e}_{n_smc_steps}_smc_steps_"
        f"{n_dispatches}_dispatch_{n_networks}_networks_mala_"
        f"step_{mcmc_step_size:0.1e}_{n_mcmc_substeps}_substeps_"
        f"repair_steps_{repair_steps}_repair_lr_{repair_lr:0.1e}"
    )
    # plt.show()
    plt.savefig(filename + ".png")

    # Save the contingencies and dispatch
    network_serializable = {
        "line_conductances": networks.line_conductances.tolist(),
        "line_susceptances": networks.line_susceptances.tolist(),
    }
    dispatch_serializable = {
        "voltage_amplitudes": best_dispatch.voltage_amplitudes.tolist(),
        "voltage_angles": best_dispatch.voltage_angles.tolist(),
    }
    with open(filename + ".json", "w") as f:
        json.dump(
            {"contingencies": network_serializable, "dispatch": dispatch_serializable},
            f,
        )
