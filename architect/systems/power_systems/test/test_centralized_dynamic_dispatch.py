import pytest

import jax.numpy as jnp

from architect.systems.power_systems import CentralizedDynamicDispatch


@pytest.fixture(params=[1, 2, 3, 5])
def two_bus_system(request):
    return make_two_bus_system(request.param)


def make_two_bus_system(horizon):
    """
    Make a simple 2-bus example.

    Bus 0 has a dispatchable generator and an intermittent generator.
    Bus 1 has a load and storage.

    args:
        horizon: the number of steps to consider
    """
    T_sim = 2 * horizon
    two_bus_system = CentralizedDynamicDispatch(
        T_horizon=horizon,
        T_sim=T_sim,
        dispatchable_names=["nuclear_1"],
        dispatchable_costs=jnp.array([[1.0, 1.0]]),
        dispatchable_limits=jnp.array([[0.0, 1.0]]),
        dispatchable_ramp_rates=jnp.array([0.1]),
        # --------------
        intermittent_names=["wind_1"],
        intermittent_costs=jnp.array([[0.5, 0.5]]),
        intermittent_ramp_rates=jnp.array([1.0]),
        intermittent_true_limits=jnp.array([[0.5]] * T_sim),
        intermittent_limits_prediction_err=jnp.array([[0.1]] * T_sim),
        # --------------
        storage_names=["storage_1"],
        storage_costs=jnp.array([[10.0, 10.0]]),
        storage_power_limits=jnp.array([[-0.1, 0.1]]),
        storage_charge_limits=jnp.array([[0.2, 0.3]]),
        storage_ramp_rates=jnp.array([10.0]),
        # --------------
        load_names=["load_1"],
        load_benefits=jnp.array([1e3]),
        load_true_limits=jnp.array([[[0.0, 1.5]]] * T_sim),
        load_limits_prediction_err=jnp.array([[0.1]] * T_sim),
        # --------------
        num_buses=2,
        dispatchables_at_bus={0: ["nuclear_1"], 1: []},
        intermittents_at_bus={0: ["wind_1"], 1: []},
        storage_at_bus={0: [], 1: ["storage_1"]},
        loads_at_bus={0: [], 1: ["load_1"]},
        # --------------
        lines=[(0, 1)],
        nominal_line_limits=jnp.array([1.2]),
        line_failure_times=jnp.array([T_sim], dtype=jnp.float32),
        line_failure_durations=jnp.array([4], dtype=jnp.float32),
    )

    return two_bus_system


def test_CentralizedDynamicDispatch___post_init__(
    two_bus_system: CentralizedDynamicDispatch,
):
    """Test that the derived values have consistent values"""
    assert two_bus_system.num_generators == (
        two_bus_system.num_dispatchables
        + two_bus_system.num_intermittents
        + two_bus_system.num_storage_resources
    )
    assert (
        two_bus_system.num_injections
        == two_bus_system.num_generators + two_bus_system.num_loads
    )
    assert two_bus_system.num_decision_variables == two_bus_system.T_horizon * (
        two_bus_system.num_injections
        + two_bus_system.num_lines
        + two_bus_system.num_storage_resources
    )


def test_CentralizedDynamicDispatch_make_quadratic_cost_matrix(
    two_bus_system: CentralizedDynamicDispatch,
):
    Q = two_bus_system.make_quadratic_cost_matrix()

    # Check the shape of Q
    n = two_bus_system.num_decision_variables
    assert Q.shape == (n, n)

    # Make sure that Q is PSD
    Q_eig = jnp.linalg.eigvals(Q)
    assert Q_eig.min() >= 0.0


def test_CentralizedDynamicDispatch_make_linear_cost_vector(
    two_bus_system: CentralizedDynamicDispatch,
):
    c = two_bus_system.make_linear_cost_vector()

    # Check the shape of c
    n = two_bus_system.num_decision_variables
    assert c.shape == (n,)


def test_CentralizedDynamicDispatch_make_constraints(
    two_bus_system: CentralizedDynamicDispatch,
):
    initial_dispatchable_generation = jnp.array([0.0])
    initial_intermittent_generation = jnp.array([0.0])
    initial_storage_power = jnp.array([0.0])
    initial_storage_state_of_charge = jnp.array([0.25])
    predicted_load_limits = jnp.array([[[0.0, 1.5]]] * two_bus_system.T_sim)
    predicted_intermittent_limits = jnp.array([[0.5]] * two_bus_system.T_sim)
    A, lb, ub = two_bus_system.make_constraints(
        initial_dispatchable_generation,
        initial_intermittent_generation,
        initial_storage_power,
        initial_storage_state_of_charge,
        predicted_load_limits,
        predicted_intermittent_limits,
        two_bus_system.nominal_line_limits,
    )

    # Check the shapes
    n_vars = two_bus_system.num_decision_variables
    T = two_bus_system.T_horizon
    n_constraints = T * (
        1  # overall power conservation
        + 4  # power limits on 2 generators, 1 storage, and 1 load
        + 3  # ramp rate limits on 2 generators and 1 storage (per step)
        + 1  # flow limit on 1 tie line
        + 2  # power conservation at 2 buses
        + 1  # storage state of charge dynamics (per step)
        + 1  # storage min/max state of charge limits
    )
    assert A.shape == (n_constraints, n_vars)
    assert lb.shape == (n_constraints,)
    assert ub.shape == (n_constraints,)

    # I manually defined these expected arrays
    A_power_conservation = jnp.array([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    A_generation_limits = jnp.array(
        [
            [1, 0, 0, 0, 0, 0],  # nuclear_1 generation limits
            [0, 1, 0, 0, 0, 0],  # wind_1 generation limits
            [0, 0, 1, 0, 0, 0],  # storage_1 generation limits
            [0, 0, 0, 1, 0, 0],  # load_1 load limits
        ]
    )
    A_ramp_limits = jnp.array(
        [
            [1, 0, 0, 0, 0, 0],  # nuclear_1 ramp rate limits
            [0, 1, 0, 0, 0, 0],  # wind_1 ramp rate limits
            [0, 0, 1, 0, 0, 0],  # storage_1 ramp rate limits
        ]
    )
    A_ramp_limits_prev = -A_ramp_limits
    A_flow_limits = jnp.array([[0, 0, 0, 0, 1, 0]])
    A_incompressibility_b0 = jnp.array(
        [
            [1, 1, 0, 0, -1, 0],  # flow leaving bus 0 = sum of injections at bus 0
        ]
    )
    A_incompressibility_b1 = jnp.array(
        [
            [0, 0, 1, 1, 1, 0],  # flow leaving bus 1 = sum of injections at bus 1
        ]
    )
    A_current_step_soc_dynamics = jnp.array(
        [
            [0, 0, -1, 0, 0, -1],  # power injection from storage + current SOC
        ]
    )
    A_prev_step_soc_dynamics = jnp.array([[0, 0, 0, 0, 0, 1]])

    # Assemble A_expected from blocks
    A_power_conservation_all_steps = jnp.block(
        [
            [
                A_power_conservation if i == t else jnp.zeros_like(A_power_conservation)
                for i in range(T)
            ]
            for t in range(T)
        ]
    )
    A_generation_limits_all_steps = jnp.block(
        [
            [
                A_generation_limits if i == t else jnp.zeros_like(A_generation_limits)
                for i in range(T)
            ]
            for t in range(T)
        ]
    )
    A_ramp_limits_all_steps = jnp.block(
        [
            [
                A_ramp_limits
                if i == t
                else A_ramp_limits_prev
                if i == t - 1
                else jnp.zeros_like(A_ramp_limits)
                for i in range(T)
            ]
            for t in range(T)
        ]
    )
    A_flow_limits_all_steps = jnp.block(
        [
            [
                A_flow_limits if i == t else jnp.zeros_like(A_flow_limits)
                for i in range(T)
            ]
            for t in range(T)
        ]
    )
    A_incompressibility_b0_all_steps = jnp.block(
        [
            [
                A_incompressibility_b0
                if i == t
                else jnp.zeros_like(A_incompressibility_b0)
                for i in range(T)
            ]
            for t in range(T)
        ]
    )
    A_incompressibility_b1_all_steps = jnp.block(
        [
            [
                A_incompressibility_b1
                if i == t
                else jnp.zeros_like(A_incompressibility_b1)
                for i in range(T)
            ]
            for t in range(T)
        ]
    )
    A_soc_dynamics_all_steps = jnp.block(
        [
            [
                A_current_step_soc_dynamics
                if i == t
                else A_prev_step_soc_dynamics
                if i == t - 1
                else jnp.zeros_like(A_current_step_soc_dynamics)
                for i in range(T)
            ]
            for t in range(T)
        ]
    )
    A_soc_limits = jnp.block(
        [
            [
                A_prev_step_soc_dynamics
                if i == t
                else jnp.zeros_like(A_prev_step_soc_dynamics)
                for i in range(T)
            ]
            for t in range(T)
        ]
    )
    A_expected = jnp.vstack(
        (
            A_power_conservation_all_steps,
            A_generation_limits_all_steps,
            A_ramp_limits_all_steps,
            A_flow_limits_all_steps,
            A_incompressibility_b0_all_steps,
            A_incompressibility_b1_all_steps,
            A_soc_dynamics_all_steps,
            A_soc_limits,
        )
    )

    # Compare with the expected matrix
    assert jnp.allclose(A, A_expected)

    lb_expected = jnp.hstack(
        T * [jnp.array([0])]  # overall power conservation
        + T
        * [
            two_bus_system.dispatchable_limits[:, 0],
            0 * predicted_intermittent_limits[0, :],  # lower lim assumed 0 for now
            two_bus_system.storage_power_limits[:, 0],
            -predicted_load_limits[0, :, 1],
        ]
        + T
        * [
            -two_bus_system.dispatchable_ramp_rates,
            -two_bus_system.intermittent_ramp_rates,
            -two_bus_system.storage_ramp_rates,
        ]
        + T
        * [
            -two_bus_system.nominal_line_limits,
        ]
        + T * [jnp.array([0]), jnp.array([0])]  # bus flow incompressibility
        + [-initial_storage_state_of_charge]
        + ((T - 1) * [jnp.array([0])] if T > 1 else [])
        + T * [two_bus_system.storage_charge_limits[:, 0]]
    )
    assert jnp.allclose(lb, lb_expected)

    ub_expected = jnp.hstack(
        T * [jnp.array([0])]  # overall power conservation
        + T
        * [
            two_bus_system.dispatchable_limits[:, 1],
            predicted_intermittent_limits[0, :],  # these are constant for now
            two_bus_system.storage_power_limits[:, 1],
            -predicted_load_limits[0, :, 0],
        ]
        + T
        * [
            two_bus_system.dispatchable_ramp_rates,
            two_bus_system.intermittent_ramp_rates,
            two_bus_system.storage_ramp_rates,
        ]
        + T
        * [
            two_bus_system.nominal_line_limits,
        ]
        + T * [jnp.array([0]), jnp.array([0])]  # bus flow incompressibility
        + [-initial_storage_state_of_charge]
        + ((T - 1) * [jnp.array([0])] if T > 1 else [])
        + T * [two_bus_system.storage_charge_limits[:, 1]]
    )
    assert jnp.allclose(ub, ub_expected)


def test_CentralizedDynamicDispatch_solve_dynamic_dispatch(
    two_bus_system: CentralizedDynamicDispatch,
):
    initial_dispatchable_generation = jnp.array([0.0])
    initial_intermittent_generation = jnp.array([0.0])
    initial_storage_power = jnp.array([0.0])
    initial_storage_state_of_charge = jnp.array([0.25])
    predicted_load_limits = jnp.array([[[0.0, 1.5]]] * two_bus_system.T_sim)
    predicted_intermittent_limits = jnp.array([[0.5]] * two_bus_system.T_sim)
    result = two_bus_system.solve_dynamic_dispatch(
        initial_dispatchable_generation,
        initial_intermittent_generation,
        initial_storage_power,
        initial_storage_state_of_charge,
        predicted_load_limits,
        predicted_intermittent_limits,
        two_bus_system.nominal_line_limits,
    )

    expected_keys = [
        "dispatchable_schedule",
        "intermittent_schedule",
        "storage_discharge_schedule",
        "load_schedule",
        "line_flow_schedule",
        "storage_soc_schedule",
        "x",
        "Q",
        "c",
        "A",
        "lb",
        "ub",
    ]
    for key in expected_keys:
        assert key in result


def test_CentralizedDynamicDispatch_get_current_line_limits(
    two_bus_system: CentralizedDynamicDispatch,
):
    # Check that the line limits way before and after the failure are unaffected
    tol = 0.01
    t_before = -10
    lim_before = two_bus_system.get_current_line_limits(t_before)
    t_after = 100
    lim_after = two_bus_system.get_current_line_limits(t_after)

    assert jnp.allclose(lim_before, two_bus_system.nominal_line_limits, atol=tol)
    assert jnp.allclose(lim_after, two_bus_system.nominal_line_limits, atol=tol)

    # Make sure that the limit decreases mid-failure
    t_mid_failure = (
        two_bus_system.line_failure_times + two_bus_system.line_failure_durations / 2.0
    )
    lim = two_bus_system.get_current_line_limits(t_mid_failure.item())
    assert (lim <= 0.1).all()


if __name__ == "__main__":
    sys = make_two_bus_system(1)
    initial_dispatchable_generation = jnp.array([0.0])
    initial_intermittent_generation = jnp.array([0.0])
    initial_storage_power = jnp.array([0.0])
    initial_storage_state_of_charge = jnp.array([0.25])
    sol = sys.solve_dynamic_dispatch(
        initial_dispatchable_generation,
        initial_intermittent_generation,
        initial_storage_power,
        initial_storage_state_of_charge,
    )

    for k in sol.keys():
        print(f"{k}: {sol[k]}")
