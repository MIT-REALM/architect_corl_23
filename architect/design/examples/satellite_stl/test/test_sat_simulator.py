import jax.numpy as jnp

from architect.design.examples.satellite_stl.sat_simulator import (
    sat_simulate,
    sat_simulate_dt,
)


def test_satellite_sim():
    # Test the simulation
    t_sim = 1.0
    dt = 0.1
    substeps = 5
    T = int(t_sim // dt)
    start_state = jnp.zeros((6,)) + 1.0
    K = jnp.array(
        [
            [100.0, 0.0, 0.0, 100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0, 0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0, 0.0, 0.0, 100.0],
        ]
    )
    planned_trajectory = jnp.zeros((T, 9))
    dps = jnp.concatenate((K.reshape(-1), planned_trajectory.reshape(-1)))

    state_trace_ct, _ = sat_simulate(dps, start_state, T, dt, substeps)
    state_trace_dt, _ = sat_simulate_dt(dps, start_state, T, dt)

    assert state_trace_ct.shape == state_trace_dt.shape
    assert state_trace_ct.shape[0] == T
