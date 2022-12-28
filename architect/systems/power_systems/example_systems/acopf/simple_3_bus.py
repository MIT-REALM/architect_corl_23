import jax
import jax.numpy as jnp
from beartype import beartype

from architect.systems.power_systems.acopf import ACOPF
from architect.systems.power_systems.acopf_types import (
    InterconnectionSpecification,
    Network,
    NetworkSpecification,
)


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
        line_conductances=jnp.array([0.0, 0.0]),
        line_susceptances=jnp.array([1.0, 1.0]),
    )
    network_spec = NetworkSpecification(
        nominal_network,
        lines=jnp.array([[0, 1], [0, 2]]),
        shunt_conductances=jnp.array([0.0, 0.0, 0.0]),
        shunt_susceptances=jnp.array([0.0, 0.0, 0.0]),
        transformer_tap_ratios=jnp.ones(2),
        transformer_phase_shifts=jnp.zeros(2),
        charging_susceptances=jnp.zeros(2),
        prob_line_failure=jnp.array([0.1, 0.1]),
        line_conductance_stddev=jnp.array([0.1, 0.1]),
        line_susceptance_stddev=jnp.array([0.1, 0.1]),
    )
    gen_spec = InterconnectionSpecification(
        buses=jnp.array([1, 2]),
        P_limits=jnp.array([[0.0, 1.0], [0.0, 0.5]]),
        Q_limits=jnp.array([[-2.0, 2.0]]),
        P_linear_costs=jnp.array([5.0, 1.0]),
        P_quadratic_costs=jnp.array([0.0, 0.0]),
    )
    load_spec = InterconnectionSpecification(
        buses=jnp.array([0]),
        P_limits=jnp.array([[-1.0, -0.8]]),
        Q_limits=jnp.array([[0.0, 0.0]]),
        P_linear_costs=jnp.array([0.0]),
        P_quadratic_costs=jnp.array([0.0]),
    )
    bus_voltage_limits = jnp.array([[0.9, 1.1], [0.9, 1.1], [0.9, 1.1]])

    return ACOPF(
        gen_spec,
        load_spec,
        network_spec,
        bus_voltage_limits,
        ref_bus_idx=1,
        constraint_penalty=penalty,
    )


if __name__ == "__main__":
    sys = make_3_bus_network()
    key = jax.random.PRNGKey(0)
    dispatch = sys.sample_random_dispatch(key)
    r = sys(dispatch, sys.network_spec.nominal_network)
    print(r.acopf_residual)
