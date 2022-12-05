import time

import seaborn as sns
import blackjax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
from beartype import beartype
from beartype.typing import Callable, Tuple
from jaxtyping import Array, Integer, PyTree, jaxtyped

from architect.components.specifications.stl.formula import (
    STLFormula,
    STLPredicate,
    STLUntimedAlways,
)
from architect.systems.power_systems.centralized_dynamic_dispatch import (
    CentralizedDynamicDispatch,
)
from architect.systems.power_systems.example_systems import make_3_bus_system


def make_service_constraint() -> STLFormula:
    """Define the STL constraint that a certain fraction of load is always served."""
    # Encode the constraint that 95% of the load is served as an STL formula
    lower_bound = 0.95
    load_mostly_served = STLPredicate(lambda load_served: load_served, lower_bound)
    p_load_always_mostly_served = STLUntimedAlways(load_mostly_served)

    return p_load_always_mostly_served


def ep_filter_spec(sys: CentralizedDynamicDispatch) -> PyTree[bool]:
    """Returns a PyTree filter for the exogenous parameters for this experiment."""
    # Line limits and intermittent generator limits
    filter_spec = jtu.tree_map(lambda _: False, sys)
    filter_spec = eqx.tree_at(
        lambda tree: (
            tree.intermittent_limits_prediction_err,
            tree.intermittent_true_limits,
        ),
        filter_spec,
        replace=(True, True),
    )

    return filter_spec


def make_simulate_fn(
    sys: CentralizedDynamicDispatch,
    exogenous_filter_spec: PyTree,
) -> Callable[[PyTree], PyTree]:
    """
    Make a function to simulate system behavior given exogenous parameters.

    args:
        sys: the CentralizedDynamicDispatch system under test
        exogenous_filter_spec: the filter specifying which parts of the system are
            exogenous factors
    """
    # Extract the non-exogenous part of the system
    non_exogenous_parameters = eqx.filter(sys, exogenous_filter_spec, inverse=True)

    # Define the simulation function
    @jax.jit
    def simulate_fn(exogenous_parameters):
        # Combine the exogenous parameters with the non-exogenous parameters
        sys_to_test = eqx.combine(non_exogenous_parameters, exogenous_parameters)

        # Simulate the behavior of the system starting from zero initial generation
        # and 50% state of charge in all storage.
        behavior = sys_to_test.simulate(
            jnp.zeros((sys.num_dispatchables,)),
            jnp.zeros((sys.num_intermittents,)),
            jnp.zeros((sys.num_storage_resources,)),
            sys.storage_charge_limits.mean(axis=-1),
        )

        return behavior

    return simulate_fn


def make_logprob_fns(
    sys: CentralizedDynamicDispatch,
    exogenous_filter_spec: PyTree,
    prediction_error_stddev: float = 0.05,
    wind_generation_limits_stddev: float = 0.05,
    smoothing: float = 10.0,
    maximize_failure: bool = True,
):
    """
    Make the prior and posterior log probability functions.

    args:
        sys: the CentralizedDynamicDispatch system under test
        exogenous_filter_spec: the filter specifying which parts of the system are
            exogenous factors
        prediction_error_stddev: the standard deviation of prediction error
        wind_generation_limits_stddev: the standard deviation of wind upper limits
        smoothing: the smoothing parameter for the STL specification (higher = stiffer)
        maximize_failure: if True, prioritize sampling cases where there is more
            failure. Otherwise, treat all failures similarly.
    returns:
        Two functions (one for the prior logprob, one for the posterior)
    """
    # Make the STL formula we'll use to assess service level
    p_load_always_mostly_served = make_service_constraint()

    # For now, assume all exogenous params follow a multivariate gaussian distribution
    @jax.jit
    def prior_logprob_fn(exogenous_parameters):
        prediction_errors = exogenous_parameters.intermittent_limits_prediction_err
        limits = exogenous_parameters.intermittent_true_limits
        T = prediction_errors.shape[0]
        logprob = jax.scipy.stats.multivariate_normal.logpdf(
            prediction_errors.squeeze(),
            mean=jnp.zeros(T),
            cov=prediction_error_stddev ** 2 * jnp.eye(T),
        )
        logprob += jax.scipy.stats.multivariate_normal.logpdf(
            limits.squeeze(),
            mean=jnp.ones(T),
            cov=wind_generation_limits_stddev ** 2 * jnp.eye(T),
        )
        return logprob

    # The posterior logprob conditions on the event that a failure occurs
    simulate_fn = make_simulate_fn(sys, exogenous_filter_spec)

    @jax.jit
    def posterior_logprob_fn(exogenous_parameters):
        behavior = simulate_fn(exogenous_parameters)

        # Compute the robustness of the service specification
        load_served = behavior["load_served"].reshape(1, -1)
        t = jnp.arange(0, load_served.size).reshape(1, -1)
        signal = jnp.vstack((t, load_served))
        robustness = p_load_always_mostly_served(signal, smoothing=smoothing)[
            1, 0
        ]  # take robustness at the start of the trace

        # If we're not trying to maximize failure, flatten negative robustness values
        if not maximize_failure:
            robustness = jax.nn.softplus(robustness)

        # lower robustness -> higher logprob -> higher probability
        return -robustness

    return prior_logprob_fn, posterior_logprob_fn


@jaxtyped
@beartype
def inference_loop(
    rng_key: Integer[Array, "..."],
    kernel,
    initial_state,
    num_samples: int,
) -> Tuple[CentralizedDynamicDispatch, PyTree]:
    """
    Run the given kernel for the specified number of steps, returning the samples.

    args:
        rng_key: JAX rng key
        kernel: the MCMC kernel to run
        initial_state: the starting state for the sampler
        num_samples: how many samples to take
    returns:
        The sampled exogenous parameters and log probabilities
    """

    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states.position, states.logprob


if __name__ == "__main__":
    # Make a simple 3-bus system
    T_horizon = 2
    T_sim = 8
    sys = make_3_bus_system(T_horizon, T_sim)

    # Figure out which parameters are exogenous
    exogenous_filter_spec = ep_filter_spec(sys)

    # Compile log likelihood functions
    prediction_error_stddev = 0.2
    wind_generation_limits_stddev = 0.2
    smoothing = 1e3
    prior_logprob, posterior_logprob = make_logprob_fns(
        sys,
        exogenous_filter_spec,
        prediction_error_stddev=prediction_error_stddev,
        wind_generation_limits_stddev=wind_generation_limits_stddev,
        smoothing=smoothing,
        maximize_failure=True,
    )

    # Make a MALA kernel for MCMC sampling
    mala_step_size = 2e-3
    n_samples = 10_000
    n_chains = 2
    mala = blackjax.mala(
        lambda x: prior_logprob(x) + posterior_logprob(x), mala_step_size
    )

    # Initialize the exogenous parameters randomly
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    baseline_exogenous_parameters = eqx.filter(sys, exogenous_filter_spec)
    prediction_error = jax.random.multivariate_normal(
        subkey, jnp.zeros(T_sim), prediction_error_stddev ** 2 * jnp.eye(T_sim)
    ).reshape(-1, 1)
    wind_generation_limits = jax.random.multivariate_normal(
        subkey, jnp.ones(T_sim), wind_generation_limits_stddev ** 2 * jnp.eye(T_sim)
    ).reshape(-1, 1)
    initial_position = eqx.tree_at(
        lambda spec: [
            spec.intermittent_limits_prediction_err,
            spec.intermittent_true_limits,
        ],
        baseline_exogenous_parameters,
        replace=(prediction_error, wind_generation_limits),
    )
    initial_state = mala.init(initial_position)

    # Sample
    single_chain = lambda key: inference_loop(key, mala.step, initial_state, n_samples)

    keys = jax.random.split(key, n_chains)
    start = time.perf_counter()
    # samples = [single_chain(key) for key in keys]
    samples, logprob = single_chain(keys[0])
    end = time.perf_counter()
    print(f"Sampled {n_samples} steps in {(end - start):.2f} s")

    # Simulate the worst likely disturbance and plot the results
    worst_disturbance_idx = jnp.argmax(logprob)
    worst_likely_disturbance = jtu.tree_map(
        lambda leaf: leaf[worst_disturbance_idx] if leaf is not None else leaf, samples
    )
    simulate_fn = make_simulate_fn(sys, exogenous_filter_spec)
    behavior = simulate_fn(worst_likely_disturbance)

    load_served = behavior["load_served"]
    dispatchable_schedule = behavior["dispatchable_schedule"]
    intermittent_schedule = behavior["intermittent_schedule"]
    limits = worst_likely_disturbance.intermittent_true_limits[:-T_horizon]
    predictions = worst_likely_disturbance.intermittent_limits_prediction_err[
        :-T_horizon
    ]

    # behaviors = jax.vmap(simulate_fn)(samples)
    # spec = make_service_constraint()

    # @jax.jit
    # def robustness_fn(load_served):
    #     t = jnp.arange(T_sim - T_horizon) * 5
    #     signal = jnp.vstack((t.reshape(1, -1), load_served))
    #     robustness = spec(signal, smoothing=smoothing)[1, 0]
    #     return robustness

    # load_served = behaviors["load_served"]
    # dispatchable_schedule = behaviors["dispatchable_schedule"]
    # intermittent_schedule = behaviors["intermittent_schedule"]
    # robustness = jax.vmap(robustness_fn)(load_served)

    # Plot results
    fig, ((hist_ax, worst_ax), (r_ax, _)) = plt.subplots(2, 2, figsize=(16, 8))

    # Plot the logprobability histogram
    sns.histplot(x=logprob, ax=hist_ax)
    hist_ax.set_xlabel("Log probability")

    # Visualize the convergence of the sampler
    r_ax.plot(logprob)
    r_ax.set_ylabel("Log probability")
    r_ax.set_xlabel("# steps")

    # Plot the system performance for the worst trace found
    t = jnp.arange(T_sim - T_horizon) * 5
    worst_ax.plot(t, load_served, label="All generation")
    worst_ax.plot(t, dispatchable_schedule.sum(axis=-1), label="Diesel")
    worst_ax.plot(
        t,
        intermittent_schedule.sum(axis=-1),
        label="Scheduled wind",
        color="green",
    )
    worst_ax.plot(
        t, limits.sum(axis=-1), label="Wind limit (true)", color="green", linestyle="--"
    )
    worst_ax.plot(
        t,
        limits.sum(axis=-1) + predictions.sum(axis=-1),
        label="Wind limit (predicted)",
        color="green",
        linestyle=":",
    )
    worst_ax.set_xlabel("Time (min)")
    worst_ax.set_ylabel("Power / total load")
    worst_ax.legend()

    plt.savefig(
        f"plts/dispatch/3bus/sigma_2e-1_{n_samples}_steps_lr_{mala_step_size:0.1e}.png"
    )
