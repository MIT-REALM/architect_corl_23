from math import ceil

import blackjax
import blackjax.smc.resampling as resampling
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
from jaxtyping import PyTree

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
        lambda tree: (tree.intermittent_limits_prediction_err,),
        filter_spec,
        replace=(True,),
    )

    return filter_spec


def make_simulate_fn(
    sys: CentralizedDynamicDispatch,
    exogenous_filter_spec: PyTree,
):
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
        smoothing: the smoothing parameter for the STL specification (higher = stiffer)
        maximize_failure: if True, prioritize sampling cases where there is more
            failure. Otherwise, treat all failures similarly.
    returns:
        Two functions (one for the prior logprob, one for the posterior)
    """
    # Make the STL formula we'll use to assess service level
    p_load_always_mostly_served = make_service_constraint()

    # We assume that the prediction errors for intermittent capacity limits follow a
    # Gaussian distribution with zero mean and given variance and are independent.
    @jax.jit
    def prior_logprob_fn(exogenous_parameters):
        prediction_errors = exogenous_parameters.intermittent_limits_prediction_err
        T = prediction_errors.shape[0]
        return jax.scipy.stats.multivariate_normal.logpdf(
            prediction_errors,
            mean=jnp.zeros(T),
            cov=prediction_error_stddev ** 2 * jnp.eye(T),
        )

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

        return robustness

    return prior_logprob_fn, posterior_logprob_fn


def smc_inference_loop(rng_key, smc_kernel, initial_state, n_tempering_steps: int):
    """
    Run the tempered SMC algorithm.

    Smoothly increase the tempering parameter from 0 to 1 in n_tempering_steps
    """

    @jax.jit
    def one_step(carry, lmbda):
        i, state, key = carry
        key, subkey = jax.random.split(key, 2)
        state, _ = smc_kernel(subkey, state, lmbda)
        new_carry = i + 1, state, key
        return new_carry, None

    (n_iter, final_state, _), _ = jax.lax.scan(
        one_step, (0, initial_state, rng_key), jnp.linspace(0, 1.0, n_tempering_steps)
    )

    return n_iter, final_state


if __name__ == "__main__":
    # Make a simple 3-bus system
    T_horizon = 2
    T_sim = 8
    sys = make_3_bus_system(T_horizon, T_sim)

    # Figure out which parameters are exogenous
    exogenous_filter_spec = ep_filter_spec(sys)

    # Compile log likelihood functions
    prediction_error_stddev = 0.05
    prior_logprob, posterior_logprob = make_logprob_fns(
        sys,
        exogenous_filter_spec,
        prediction_error_stddev=prediction_error_stddev,
        smoothing=10.0,
        maximize_failure=True,
    )

    # Make a tempered SMC kernel (with MALA inner kernel) to sample failures
    n_samples = 100
    n_tempering_steps = 10
    mala_parameters = {"step_size": 1e-4}
    tempered = blackjax.tempered_smc(
        prior_logprob,
        posterior_logprob,
        blackjax.mala,
        mala_parameters,
        resampling.systematic,
        mcmc_iter=100,
    )

    # Initialize the exogenous parameters randomly
    baseline_exogenous_parameters = eqx.filter(sys, exogenous_filter_spec)

    def generate_random_exogenous_parameters(key):
        prediction_error = jax.random.multivariate_normal(
            key, jnp.zeros(T_sim), prediction_error_stddev ** 2 * jnp.eye(T_sim)
        )
        return eqx.tree_at(
            lambda spec: [
                spec.intermittent_limits_prediction_err,
            ],
            baseline_exogenous_parameters,
            replace=(prediction_error,),
        )

    key = jax.random.PRNGKey(0)
    initial_smc_state = jax.vmap(generate_random_exogenous_parameters)(
        jax.random.split(key, n_samples)
    )
    initial_smc_state = tempered.init(initial_smc_state)

    # Run SMC
    n_iter, smc_samples = smc_inference_loop(
        key, tempered.step, initial_smc_state, n_tempering_steps
    )
    print("Number of steps in the adaptive algorithm: ", n_iter.item())

    # Simulate and plot the results
    simulate_fn = make_simulate_fn(sys, exogenous_filter_spec)
    behaviors = jax.vmap(simulate_fn)(smc_samples.particles)

    load_served = behaviors["load_served"]
    dispatchable_schedule = behaviors["dispatchable_schedule"]
    intermittent_schedule = behaviors["intermittent_schedule"]

    spec = make_service_constraint()

    def robustness_fn(load_served):
        t = jnp.arange(T_sim - T_horizon) * 5
        signal = jnp.vstack((t.reshape(1, -1), load_served))
        robustness = spec(signal)[1, 0]
        return robustness

    robustness = jax.vmap(robustness_fn)(load_served)

    # Plot the histogram of robustnesses and the worst disturbance
    fig, (hist_ax, worst_ax) = plt.subplots(1, 2, figsize=(16, 8))

    hist_ax.hist(robustness, bins=ceil(n_samples / 10))

    worst_disturbance_idx = jnp.argmin(robustness)
    t = jnp.arange(T_sim - T_horizon) * 5
    worst_ax.plot(t, load_served[worst_disturbance_idx], label="All generation")
    worst_ax.plot(
        t, dispatchable_schedule.sum(axis=-1)[worst_disturbance_idx], label="Diesel"
    )
    worst_ax.plot(
        t, intermittent_schedule.sum(axis=-1)[worst_disturbance_idx], label="Wind"
    )
    worst_ax.set_xlabel("Time (min)")
    worst_ax.set_ylabel("Fraction of load served")
    worst_ax.legend()

    plt.savefig(
        f"plts/3_bus_contingency_{n_samples}_particles_{n_tempering_steps}_steps.png"
    )
