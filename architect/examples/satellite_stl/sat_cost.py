from functools import partial

import jax
import jax.numpy as jnp

import architect.components.specifications.stl as stl

from .sat_simulator import sat_simulate_dt


@partial(jax.jit, static_argnames=["specification"])
def sat_cost(
    design_params: jnp.ndarray,
    exogenous_sample: jnp.ndarray,
    specification: stl.STLFormula,
    specification_weight: float,
    time_steps: int,
    dt: float,
) -> jnp.ndarray:
    """Compute the cost based on given beacon locations.

    args:
        design_params: a (6 * 3 + time_steps * (3 + 6)) array of design parameters.
            The first 6 * 3 values define a gain matrix for state-feedback control
            u = -K(x - x_planned). Each later group of 3 + 6 values is a control input
            and target state at that timestep, defining a trajectory.
        exogenous_sample: (6,) array containing initial states.
                          Can be generated by SatExogenousParameters.sample
        specification: jit-wrapped STL formula specifying the planning problem
        specification_weight: how much to weight the STL robustness in the cost
        time_steps: the number of steps to simulate
        dt: the duration of each time step
    returns:
        scalar cost
    """
    # Get the state trace and control effort from simulating
    state_trace, total_effort = sat_simulate_dt(
        design_params,
        exogenous_sample,
        time_steps,
        dt,
    )

    # Get the robustness of the formula on this state trace
    t = jnp.linspace(0.0, time_steps * dt, state_trace.shape[0]).reshape(1, -1)
    signal = jnp.vstack((t, state_trace.T))  # type: ignore
    robustness = specification(signal)[1, 0]

    # Compute cost based on combination of STL formula robustness and control effort
    cost = -robustness + total_effort / specification_weight
    return cost
