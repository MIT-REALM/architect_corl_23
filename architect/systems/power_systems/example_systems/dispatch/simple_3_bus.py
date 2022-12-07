import matplotlib.pyplot as plt
import jax.numpy as jnp
from jaxtyping import Float, Array
from beartype.typing import Optional

from architect.systems.power_systems import CentralizedDynamicDispatch


def make_3_bus_system(
    T_horizon: int,
    T_sim: int,
    prediction_errors: Optional[Float[Array, " T_sim ni"]] = None,
) -> CentralizedDynamicDispatch:
    """
    Construct a simple 3-bus test system

    Power units MW, Time units are 10 min

    args:
        T_horizon: horizon for dynamic dispatch
        T_sim: time horizon for the simulation
        prediction_errors: prediction errors in intermittent generation
    """
    if prediction_errors is None:
        prediction_errors = jnp.array([[0.0]] * T_sim)

    return CentralizedDynamicDispatch(
        T_horizon=T_horizon,
        T_sim=T_sim,
        # --------------
        # Dispatchable generators have 1.0 MW capacity each, but they are
        # expensive: LCOE is 87 $/MWh = 14.5 $/p.u.
        # Modeled on the diesel resources in the Azores case study
        dispatchable_names=["diesel"],
        dispatchable_costs=jnp.array([[0.0, 43.5]]),
        dispatchable_limits=jnp.array([[0.0, 1.0]]),
        dispatchable_ramp_rates=jnp.array([0.1]),
        # --------------
        # Intermittent generators have 1.0 MW max capacity and can ramp fast.
        # LCOE is 87 $/MWh = 14.5 $/p.u.. We'll add variation and error later.
        # Modeled on wind in the Azores case study
        intermittent_names=["wind"],
        intermittent_costs=jnp.array([[0.0, 14.5]]),
        intermittent_ramp_rates=jnp.array([1.0]),
        intermittent_true_limits=jnp.array(
            # [[1.0] if t < 10 else [0.8] for t in range(T_sim)]
            [[1.0] for _ in range(T_sim)]
        ),
        intermittent_limits_prediction_err=prediction_errors,
        # --------------
        # In the baseline system we have no storage (although some might be added
        # for mitigation). We represent this as a "dormant" storage resource co-located
        # with wind.
        storage_names=["storage"],
        storage_costs=jnp.array([[0.0, 10.0]]),
        storage_power_limits=jnp.array([[-0.1, 0.1]]),
        storage_charge_limits=jnp.array([[0.0, 0.0]]),
        storage_ramp_rates=jnp.array([0.1]),
        # --------------
        # We'll model the load as inelastic (very high price)
        load_names=["load"],
        load_benefits=jnp.array([1e3]),
        load_true_limits=jnp.array(
            [
                [
                    [0.0, 1.0],
                ]
            ]
            * T_sim
        ),
        load_limits_prediction_err=jnp.array([[0.0]] * T_sim),
        # --------------
        num_buses=3,
        dispatchables_at_bus={
            0: ["diesel"],
            1: [],
            2: [],
        },
        intermittents_at_bus={
            0: [],
            1: ["wind"],
            2: [],
        },
        storage_at_bus={
            0: [],
            1: [],
            2: ["storage"],
        },
        loads_at_bus={
            0: [],
            1: [],
            2: ["load"],
        },
        # --------------
        lines=[
            (0, 2),
            (1, 2),
        ],
        nominal_line_limits=jnp.array([1.0 for _ in range(2)]),
        line_failure_times=jnp.array([100.0 for _ in range(2)]),
        line_failure_durations=jnp.array([5.0 for _ in range(2)]),
    )


if __name__ == "__main__":
    horizon = 2
    sim = 20
    sys = make_3_bus_system(horizon, sim)
    output = sys.simulate(
        jnp.zeros((sys.num_dispatchables,)),
        jnp.zeros((sys.num_intermittents,)),
        jnp.zeros((sys.num_storage_resources,)),
        sys.storage_charge_limits.mean(axis=-1),
    )

    t = jnp.arange(sim - horizon) * 5
    plt.plot(t, output["load_served"], label="All generation")
    plt.plot(t, output["dispatchable_schedule"].sum(axis=-1), label="Diesel")
    plt.plot(t, output["intermittent_schedule"].sum(axis=-1), label="Wind")
    plt.xlabel("Time (min)")
    plt.ylabel("Fraction of load served")
    plt.legend()
    plt.show()
