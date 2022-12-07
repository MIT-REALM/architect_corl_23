import jax
import jax.numpy as jnp
from beartype import beartype

from architect.systems.power_systems.acopf import ACOPF, Network


@beartype
def make_14_bus_network(penalty: float = 10.0) -> ACOPF:
    """
    Define a test system based on the IEEE 14-bus system.

    Image here: icseg.iti.illinois.edu/ieee-14-bus-system/
    Documented data here: github.com/MATPOWER/matpower/blob/master/data/case14.m
    Data format here: github.com/MATPOWER/matpower/blob/a908aeb33d023ee9354f660a8f551c6dc5a7dcdd/lib/caseformat.m  # noqa

    All bus indices are zero-indexed, so 1 less than typically found in the MATLAB-heavy
    power systems literature :D

    args:
        penalty coefficient
    """
    n_bus = 14
    n_line = 20

    # Define the normalizing constant for powers
    base_MVA = 100  # mega volt-ampere

    # Define the connections between buses
    lines = jnp.array(
        [
            [0, 1],
            [0, 4],
            [1, 2],
            [1, 3],
            [1, 4],
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
    )
    line_resistances = jnp.array(
        [
            0.01938,
            0.05403,
            0.04699,
            0.05811,
            0.05695,
            0.06701,
            0.01335,
            0.0,
            0.0,
            0.0,
            0.09498,
            0.12291,
            0.06615,
            0.0,
            0.0,
            0.03181,
            0.12711,
            0.08205,
            0.22092,
            0.17093,
        ]
    )
    line_reactances = jnp.array(
        [
            0.05917,
            0.22304,
            0.19797,
            0.17632,
            0.17388,
            0.17103,
            0.04211,
            0.20912,
            0.55618,
            0.25202,
            0.1989,
            0.25581,
            0.13027,
            0.17615,
            0.11001,
            0.0845,
            0.27038,
            0.19207,
            0.19988,
            0.34802,
        ]
    )
    shunt_conductance = jnp.zeros((n_bus,))
    shunt_reactance = jnp.zeros((n_bus,))
    shunt_reactance = shunt_reactance.at[8].set(19.0)
    shunt_conductance = shunt_conductance / base_MVA
    shunt_reactance = shunt_reactance / base_MVA

    # The line data is provided as impedance, so we need to convert to admittances
    line_impedance = line_resistances + 1j * line_reactances
    line_admittance = 1 / line_impedance
    line_conductances = line_admittance.real
    line_susceptances = line_admittance.imag

    # Now we can load this data into our representation
    nominal_network = Network(
        lines=lines,
        line_conductances=line_conductances,
        line_susceptances=line_susceptances,
        shunt_conductances=shunt_conductance,
        shunt_susceptances=shunt_reactance,
    )

    # We need to load in all of the bus data as well for generator capacities
    # and load ranges
    generator_buses = jnp.array([0, 1, 2, 5, 7])
    P_gen_max = jnp.array([332.4, 140.0, 100.0, 100.0, 100.0])
    P_gen_min = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    Q_gen_max = jnp.array([10.0, 50.0, 40.0, 24.0, 24.0])
    Q_gen_min = jnp.array([0.0, -40.0, 0.0, -6.0, -6.0])
    gen_quad_costs = jnp.array([0.043, 0.25, 0.01, 0.01, 0.01])
    gen_linear_costs = jnp.array([20.0, 20.0, 40.0, 40.0, 40.0])
    load_buses = jnp.array([1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
    P_load_max = jnp.array(
        [25.0, 95.0, 50.0, 10.0, 15.0, 30.0, 10.0, 5.0, 5.0, 15.0, 15.0]
    )
    P_load_min = jnp.array([12.5, 45.0, 25.0, 5.0, 7.5, 15.0, 5.0, 2.5, 2.5, 7.5, 7.5])
    Q_load_max = jnp.zeros_like(P_load_max) + 20.0
    Q_load_min = -Q_load_max
    load_benefit = 100.0 + jnp.zeros_like(P_load_max)
    reactive_power_cost = jnp.zeros((n_bus,))

    # Convert these into per-bus limits
    bus_P_limits = jnp.zeros((n_bus, 2))
    bus_P_limits = bus_P_limits.at[generator_buses, 0].add(P_gen_min)
    bus_P_limits = bus_P_limits.at[generator_buses, 1].add(P_gen_max)
    bus_P_limits = bus_P_limits.at[load_buses, 0].add(-P_load_max)
    bus_P_limits = bus_P_limits.at[load_buses, 1].add(-P_load_min)
    bus_P_limits = bus_P_limits / base_MVA

    bus_Q_limits = jnp.zeros((n_bus, 2))
    bus_Q_limits = bus_Q_limits.at[generator_buses, 0].add(Q_gen_min)
    bus_Q_limits = bus_Q_limits.at[generator_buses, 1].add(Q_gen_max)
    bus_Q_limits = bus_Q_limits.at[load_buses, 0].add(-Q_load_max)
    bus_Q_limits = bus_Q_limits.at[load_buses, 1].add(-Q_load_min)
    bus_Q_limits = bus_Q_limits / base_MVA

    bus_V_limits = jnp.zeros((n_bus, 2))
    bus_V_limits = bus_V_limits.at[:, 0].set(0.94)
    bus_V_limits = bus_V_limits.at[:, 1].set(1.06)

    bus_P_linear_costs = jnp.zeros((n_bus,))
    bus_P_linear_costs = bus_P_linear_costs.at[generator_buses].add(gen_linear_costs)
    bus_P_linear_costs = bus_P_linear_costs.at[load_buses].add(-load_benefit)

    bus_P_quad_costs = jnp.zeros((n_bus,))
    bus_P_quad_costs = bus_P_quad_costs.at[generator_buses].add(gen_quad_costs)

    return ACOPF(
        nominal_network,
        # Limits
        bus_active_limits=bus_P_limits,
        bus_reactive_limits=bus_Q_limits,
        bus_voltage_limits=bus_V_limits,
        # Costs
        bus_active_linear_costs=bus_P_linear_costs,
        bus_active_quadratic_costs=bus_P_quad_costs,
        bus_reactive_linear_costs=reactive_power_cost,
        constraint_penalty=penalty,
        # Failures
        prob_line_failure=jnp.array([0.1 for _ in range(n_line)]),
        line_conductance_stddev=jnp.array([0.1 for _ in range(n_line)]),
        line_susceptance_stddev=jnp.array([0.1 for _ in range(n_line)]),
    )


if __name__ == "__main__":
    sys = make_14_bus_network()
    key = jax.random.PRNGKey(0)
    dispatch = sys.sample_random_dispatch(key)
    sys(dispatch, sys.nominal_network)
