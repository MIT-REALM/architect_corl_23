import jax
import jax.numpy as jnp

from architect.design.examples.turtle_stl.tbstl_exogenous_parameters import (
    TBSTLExogenousParameters,
)
from architect.design.examples.turtle_stl.tbstl_simulator import (
    tbstl_simulate_dt
)


def test_satellite_sim():
    # Test the simulation
    t_sim = 10.0
    dt = 0.1
    T = int(t_sim // dt)
    planned_trajectory = jnp.zeros((T, 2)) + 1.0

    ep = TBSTLExogenousParameters()
    prng_key = jax.random.PRNGKey(0)
    prng_key, subkey = jax.random.split(prng_key)
    start_state = ep.sample(subkey)

    # Simulate
    state_trace_dt = tbstl_simulate_dt(planned_trajectory, start_state, T, dt)

    assert state_trace_dt.shape[0] == T
