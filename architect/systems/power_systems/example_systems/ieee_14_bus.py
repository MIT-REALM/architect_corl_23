import matplotlib.pyplot as plt
import jax.numpy as jnp

from architect.systems.power_systems import CentralizedDynamicDispatch


def make_ieee_14_bus_system(T_horizon: int, T_sim: int) -> CentralizedDynamicDispatch:
    """
    Construct the 14-bus IEEE test system.

    Power units MW, Time units are 10 min

    args:
        T_horizon: horizon for dynamic dispatch
        T_sim: time horizon for the simulation
    """
    return CentralizedDynamicDispatch(
        T_horizon=T_horizon,
        T_sim=T_sim,
        # --------------
        # Dispatchable generators have 0.5 MW capacity each, but have slow ramp
        # rates (10% of capacity per step). LCOE is 87 $/MWh = 14.5 $/p.u.
        # Modeled on the hydro resources in the Azores case study
        dispatchable_names=["hydro1", "hydro2", "hydro3"],
        dispatchable_costs=jnp.array([[0.0, 14.5], [0.0, 14.5], [0.0, 14.5]]),
        dispatchable_limits=jnp.array([[0.0, 0.5], [0.0, 0.5], [0.0, 0.5]]),
        dispatchable_ramp_rates=jnp.array([0.05, 0.05, 0.05]),
        # --------------
        # Intermittent generators have 1 MW max capacity and can ramp at 70% per step.
        # LCOE is the same as for hydro. We'll add variation and prediction error later.
        intermittent_names=["wind1", "wind2"],
        intermittent_costs=jnp.array([[0.0, 14.5], [0.0, 14.5]]),
        intermittent_ramp_rates=jnp.array([0.7, 0.7]),
        intermittent_true_limits=jnp.array([[[0.0, 1.0], [0.0, 1.0]]] * T_sim),
        intermittent_limits_prediction_err=jnp.array([[0.0, 0.0]] * T_sim),
        # --------------
        # In the baseline system we have no storage (although some might be added
        # for mitigation). We represent this as 3 "dormant" storage resources.
        storage_names=["storage1", "storage2", "storage3"],
        storage_costs=jnp.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]),
        storage_power_limits=jnp.array([[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]]),
        storage_charge_limits=jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        storage_ramp_rates=jnp.array([0.1, 0.1, 0.1]),
        # --------------
        # We'll model these loads as inelastic (very high prices)
        load_names=[f"load{i+1}" for i in range(11)],
        load_benefits=jnp.array([1e3 for _ in range(11)]),
        load_true_limits=jnp.array(
            [
                [
                    [0.0, 0.15],
                    [0.0, 0.2],
                    [0.0, 0.1],
                    [0.0, 0.15],
                    [0.0, 0.2],
                    [0.0, 0.1],
                    [0.0, 0.15],
                    [0.0, 0.2],
                    [0.0, 0.1],
                    [0.0, 0.15],
                    [0.0, 0.2],
                ]
            ]
            * T_sim
        ),
        load_limits_prediction_err=jnp.array([[0.0 for _ in range(11)]] * T_sim),
        # --------------
        num_buses=14,
        dispatchables_at_bus={
            0: [],
            1: ["hydro1"],
            2: [],
            3: [],
            4: [],
            5: ["hydro2"],
            6: [],
            7: ["hydro3"],
            8: [],
            9: [],
            10: [],
            11: [],
            12: [],
            13: [],
        },
        intermittents_at_bus={
            0: ["wind1"],
            1: [],
            2: ["wind2"],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
            10: [],
            11: [],
            12: [],
            13: [],
        },
        storage_at_bus={
            0: [],
            1: [],
            2: [],
            3: ["storage1"],
            4: ["storage2"],
            5: [],
            6: [],
            7: [],
            8: [],
            9: [],
            10: ["storage3"],
            11: [],
            12: [],
            13: [],
        },
        loads_at_bus={
            0: [],
            1: ["load1"],
            2: ["load2"],
            3: ["load3"],
            4: ["load4"],
            5: ["load5"],
            6: [],
            7: [],
            8: ["load6"],
            9: ["load7"],
            10: ["load8"],
            11: ["load9"],
            12: ["load10"],
            13: ["load11"],
        },
        # --------------
        lines=[
            (0, 1),
            (0, 4),
            (1, 4),
            (1, 3),
            (1, 2),
            (2, 3),
            (3, 4),
            (3, 6),
            (3, 8),
            (4, 5),
            (5, 10),
            (5, 11),
            (5, 12),
            (6, 7),
            (6, 8),
            (8, 9),
            (8, 13),
            (9, 10),
            (11, 12),
            (12, 13),
        ],
        nominal_line_limits=jnp.array([0.7 for _ in range(20)]),
        line_failure_times=jnp.array([10.0 for _ in range(20)]),
        line_failure_durations=jnp.array([10.0 for _ in range(20)]),
    )


if __name__ == "__main__":
    horizon = 2
    sim = 30
    sys = make_ieee_14_bus_system(horizon, sim)
    output = sys.simulate(
        jnp.zeros((sys.num_dispatchables,)),
        jnp.zeros((sys.num_intermittents,)),
        jnp.zeros((sys.num_storage_resources,)),
        sys.storage_charge_limits.mean(axis=-1),
    )

    t = jnp.arange(sim - horizon) * 5
    plt.plot(t, output["load_served"], label="Fraction of demand served")
    plt.xlabel("Time (min)")
    plt.ylabel("Fraction of load served")
    plt.show()
