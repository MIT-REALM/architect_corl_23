import jax.numpy as jnp
from beartype import beartype

from architect.systems.power_systems.acopf import ACOPF, Network


@beartype
def make_3_bus_network(penalty: float = 100.0) -> ACOPF:
    """
    Define a 3-bus test system.

    This network has a load bus (0) connected to 2 generator buses (1 and 2).

    The generators at bus 1 and 2 can supply 1 and 0.5 units of active power each,
    but the generator at bus 2 is much cheaper.

    The load demands between 0.8 and 1.0 units of active power and has a large benefit
    to model an inelastic load.

    args:
        penalty coefficient
    """
    nominal_network = Network(
        lines=jnp.array([[0, 1], [0, 2]]),
        line_conductances=jnp.array([0.0, 0.0]),
        line_susceptances=jnp.array([1.0, 1.0]),
        shunt_conductances=jnp.array([0.0, 0.0, 0.0]),
        shunt_susceptances=jnp.array([0.0, 0.0, 0.0]),
    )
    return ACOPF(
        nominal_network,
        # Limits
        bus_active_limits=jnp.array([[-1.0, -0.8], [0.0, 1.0], [0.0, 0.5]]),
        bus_reactive_limits=jnp.array([[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]]),
        bus_voltage_limits=jnp.array([[0.9, 1.1], [0.9, 1.1], [0.9, 1.1]]),
        # Costs
        bus_active_linear_costs=jnp.array([-10.0, 5.0, 1.0]),
        bus_active_quadratic_costs=jnp.array([0.0, 0.0, 0.0]),
        bus_reactive_linear_costs=jnp.array([0.0, 0.1, 0.1]),
        constraint_penalty=penalty,
        # Failures
        prob_line_failure=jnp.array([0.1, 0.1]),
        line_conductance_stddev=jnp.array([0.1, 0.1]),
        line_susceptance_stddev=jnp.array([0.1, 0.1]),
    )
