import jax
import jax.numpy as jnp

from architect.problem_definition import DesignParameters

from architect.design.examples.agv_localization.agv_exogenous_parameters import (
    AGVExogenousParameters,
)
from architect.design.examples.agv_localization.agv_simulator import agv_simulate


def test_agv_simulator():
    # Test out the simulation
    T = 2
    dt = 0.5
    time_steps = int(T / dt)

    # Define the exogenous parameters
    observation_noise_covariance = jnp.diag(jnp.array([0.1, 0.01, 0.01]))
    actuation_noise_covariance = dt ** 2 * jnp.diag(jnp.array([0.001, 0.001, 0.01]))
    initial_state_mean = jnp.array([-2.2, 0.5, 0.0])
    initial_state_covariance = 0.001 * jnp.eye(3)
    ep = AGVExogenousParameters(
        time_steps,
        initial_state_mean,
        initial_state_covariance,
        actuation_noise_covariance,
        observation_noise_covariance,
    )

    # Define the design parameters
    beacon_locations = jnp.array([-1.6945883, -1.0, 0.0, -0.8280163]).reshape(2, 2)
    control_gains = jnp.array([2.535058, 0.09306894])

    dp = DesignParameters(control_gains.size + beacon_locations.size)
    dp.set_values(jnp.concatenate((control_gains, beacon_locations.reshape(-1))))

    # Sample some exogenous parameters
    prng_key = jax.random.PRNGKey(0)
    prng_key, subkey = jax.random.split(prng_key)
    exogenous_sample = ep.sample(subkey)

    true_states, _, _, _ = agv_simulate(
        dp.get_values(),
        exogenous_sample,
        observation_noise_covariance,
        actuation_noise_covariance,
        initial_state_mean,
        initial_state_covariance,
        time_steps,
        dt,
    )

    assert true_states.shape[0] == time_steps
