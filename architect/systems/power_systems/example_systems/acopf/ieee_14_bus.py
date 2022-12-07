import jax.numpy as jnp
from beartype import beartype

from architect.systems.power_systems.acopf import ACOPF, Network


@beartype
def make_14_bus_network(penalty: float = 100.0) -> ACOPF:
    """
    Define a test system based on the IEEE 14-bus system
    https://icseg.iti.illinois.edu/ieee-14-bus-system/

    We remove the loads at bus 1, 2, and 5 so that each node has either
    only loads or only generators.

    args:
        penalty coefficient
    """
    n_bus = 14
    n_line = 20
    nominal_network = Network(
        lines=jnp.array(
            [
                [0, 1],
                [0, 4],
                [1, 4],
                [1, 3],
                [1, 2],
                [2, 3],
                [3, 4],
                [3, 6],
                [3, 8],
                [4, 5],
                [5, 10],
                [5, 11],
                [5, 12],
                [6, 7],
                [6, 8],
                [8, 9],
                [8, 13],
                [9, 10],
                [11, 12],
                [12, 13],
            ]
        ),
        line_conductances=jnp.array([0.01 for _ in range(n_line)]),
        line_susceptances=jnp.array([1.0 for _ in range(n_line)]),
        shunt_conductances=jnp.array([0.0 for _ in range(n_bus)]),
        shunt_susceptances=jnp.array([0.0 for _ in range(n_bus)]),
    )
    generator_buses = [0, 1, 2, 5, 7]
    generator_max = 5.0
    generator_min = 0.0
    max_Q = 2.0
    load_min = -2.0
    load_max = -1.0
    return ACOPF(
        nominal_network,
        # Limits
        bus_active_limits=jnp.array(
            [
                [generator_min, generator_max]
                if i in generator_buses
                else [load_min, load_max]
                for i in range(n_bus)
            ]
        ),
        bus_reactive_limits=jnp.array([[-max_Q, max_Q] for _ in range(n_bus)]),
        bus_voltage_limits=jnp.array([[0.9, 1.1] for _ in range(n_bus)]),
        # Costs
        bus_active_linear_costs=jnp.array(
            [1.0 if i in generator_buses else -10.0 for i in range(n_bus)]
        ),
        bus_active_quadratic_costs=jnp.array([0.0 for _ in range(n_bus)]),
        bus_reactive_linear_costs=jnp.array(
            [0.1 if i in generator_buses else 0.0 for i in range(n_bus)]
        ),
        constraint_penalty=penalty,
        # Failures
        prob_line_failure=jnp.array([0.1 for _ in range(n_line)]),
        line_conductance_stddev=jnp.array([0.1 for _ in range(n_line)]),
        line_susceptance_stddev=jnp.array([0.1 for _ in range(n_line)]),
    )
