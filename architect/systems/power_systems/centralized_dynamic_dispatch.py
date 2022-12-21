from dataclasses import field

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from beartype import beartype
from beartype.typing import Dict, List, Tuple
from jax.nn import logsumexp, sigmoid
from jaxopt import BoxOSQP
from jaxtyping import Array, Float, PyTree, jaxtyped

from architect.problem_definition import AutonomousSystem


@beartype
class CentralizedDynamicDispatch(AutonomousSystem):
    """
    Solves centralized dynamic dispatch at every time step over a fixed horizon.

    Assumes an elastic load and three types of generation resource: fully generators,
    intermittent generators (which may be dispatched within some partially-predictable
    range via curtailment), and dispatchable storage. All generators are modeled with
    finite ramp rates.
    """

    T_horizon: int
    T_sim: int

    # All costs are 2-element arrays where the first element is the quadratic factor
    # and the second element is the linear factor.

    dispatchable_names: List[str]
    dispatchable_costs: Float[Array, "nd 2"]
    dispatchable_limits: Float[Array, "nd 2"]
    dispatchable_ramp_rates: Float[Array, " nd"]
    num_dispatchables: int = field(init=False)

    intermittent_names: List[str]
    intermittent_costs: Float[Array, "ni 2"]
    intermittent_ramp_rates: Float[Array, " ni"]
    num_intermittents: int = field(init=False)
    # Intermittent generators have a series of maximum generation limits and a series of
    # prediction errors
    intermittent_true_limits: Float[Array, "Tsim ni"]
    intermittent_limits_prediction_err: Float[Array, "Tsim ni"]

    storage_names: List[str]
    storage_costs: Float[Array, "ns 2"]
    storage_power_limits: Float[Array, "ns 2"]
    storage_charge_limits: Float[Array, "ns 2"]
    storage_ramp_rates: Float[Array, " ns"]
    num_storage_resources: int = field(init=False)

    load_names: List[str]
    load_benefits: Float[Array, " nl"]
    num_loads: int = field(init=False)
    # Loads have a series of true limits and a series of prediction errors
    load_true_limits: Float[Array, "Tsim nl 2"]
    load_limits_prediction_err: Float[Array, "Tsim nl"]

    num_buses: int
    dispatchables_at_bus: Dict[int, List[str]]
    intermittents_at_bus: Dict[int, List[str]]
    storage_at_bus: Dict[int, List[str]]
    loads_at_bus: Dict[int, List[str]]

    lines: List[Tuple[int, int]]
    nominal_line_limits: Float[Array, " nl"]
    num_lines: int = field(init=False)
    # Lines fail at the times in this array, and no failure occurs if
    # these times exceed T_sim
    line_failure_times: Float[Array, " nl"]
    line_failure_durations: Float[Array, " nl"]

    num_generators: int = field(init=False)
    num_injections: int = field(init=False)
    num_decision_variables: int = field(init=False)
    num_variables_per_step: int = field(init=False)

    # TODO: should we add congestion costs?

    @property
    def dp_filter_spec(self) -> PyTree[bool]:
        """Returns a PyTree filter selecting only the design parameters."""
        # No design parameters yet
        # TODO: add storage capacity
        filter_spec = jtu.tree_map(lambda _: False, self)

        return filter_spec

    @property
    def ep_filter_spec(self) -> PyTree[bool]:
        """Returns a PyTree filter selecting only the exogenous parameters."""
        # Line limits and intermittent generator limits
        filter_spec = jtu.tree_map(lambda _: False, self)
        filter_spec = eqx.tree_at(
            lambda spec: [
                spec.line_failure_times,
                spec.intermittent_limits_prediction_err,
                spec.intermittent_true_limits,
                spec.load_limits_prediction_err,
                spec.load_true_limits,
            ],
            filter_spec,
            replace=True,
        )

        return filter_spec

    def __post_init__(self) -> None:
        # Sanity check
        if self.T_sim < self.T_horizon:
            raise ValueError(
                "T_sim ({}) must be >= T_horizon ({})!".format(
                    self.T_sim, self.T_horizon
                )
            )

        # For convenience, get the number of different things in the network
        self.num_dispatchables = len(self.dispatchable_names)
        self.num_intermittents = len(self.intermittent_names)
        self.num_storage_resources = len(self.storage_names)
        self.num_lines = len(self.lines)
        self.num_loads = len(self.load_names)

        # The decision variables of the dynamic dispatch problem are:
        #   - Generation for each generator and storage resource at each step
        #   - Load consumption for each load at each time step
        #   - Flows through each line at each step
        #   - Charge in each storage resource
        #
        # For convenience, we encode both generation and consumption as power injection
        # (positive when generated, negative when consumed)
        #
        # We'll assemble these into a decision vector
        #   x_k = [generations, storage rates, loads, flows, charges]
        # and concatenate [x_0, x_1, ..., x_T_horizon]
        self.num_generators = (
            self.num_dispatchables + self.num_intermittents + self.num_storage_resources
        )
        self.num_injections = self.num_generators + self.num_loads
        self.num_variables_per_step = (
            self.num_injections + self.num_lines + self.num_storage_resources
        )
        self.num_decision_variables = self.T_horizon * self.num_variables_per_step

    @jaxtyped
    @beartype
    def make_quadratic_cost_matrix(self) -> Float[Array, "nx nx"]:
        """Assemble the quadratic cost matrix for the dispatch QP"""
        # We don't have any cross-terms in our cost, so Q is just diagonal
        dispatchable_quadratic_costs = self.dispatchable_costs[:, 0]
        intermittent_quadratic_costs = self.intermittent_costs[:, 0]
        storage_quadratic_costs = self.storage_costs[:, 0]
        load_quadratic_costs = jnp.zeros((self.num_loads,))
        congestion_quadratic_costs = jnp.zeros((self.num_lines,))
        storage_soc_quadratic_costs = jnp.zeros((self.num_storage_resources,))
        Q = jnp.diag(
            jnp.hstack(
                self.T_horizon
                * (
                    dispatchable_quadratic_costs,
                    intermittent_quadratic_costs,
                    storage_quadratic_costs,
                    load_quadratic_costs,
                    congestion_quadratic_costs,
                    storage_soc_quadratic_costs,
                )
            )
        )
        return Q

    @jaxtyped
    @beartype
    def make_linear_cost_vector(self) -> Float[Array, " nx"]:
        """Assemble the quadratic cost matrix for the dispatch QP"""
        dispatchable_linear_costs = self.dispatchable_costs[:, 1]
        intermittent_linear_costs = self.intermittent_costs[:, 1]
        storage_linear_costs = self.storage_costs[:, 1]
        load_linear_costs = self.load_benefits
        congestion_linear_costs = jnp.zeros((self.num_lines,))
        storage_soc_linear_costs = jnp.zeros((self.num_storage_resources,))
        c = jnp.hstack(
            self.T_horizon
            * (
                dispatchable_linear_costs,
                intermittent_linear_costs,
                storage_linear_costs,
                load_linear_costs,
                congestion_linear_costs,
                storage_soc_linear_costs,
            )
        )
        return c

    @jaxtyped
    @beartype
    def add_to_constraints(
        self,
        A: Float[Array, "nc nx"],
        lb: Float[Array, " nc"],
        ub: Float[Array, " nc"],
        new_A_rows: Float[Array, "n_new nx"],
        new_lbs: Float[Array, " n_new"],
        new_ubs: Float[Array, " n_new"],
    ) -> Tuple[
        Float[Array, " nc+n_new nx"],
        Float[Array, " nc+n_new"],
        Float[Array, " nc+n_new"],
    ]:
        """
        Add new rows to the constraint matrix A.

        args:
            A: existing constraint matrix
            lb: existing lower bound vector
            ub: existing upper bound vector
            new_A_rows: new rows to append to A
            new_lbs: new lower bounds to append to lb
            new_ubs: new lower bounds to append to ub
        returns:
            The new A matrix and lb and ub vectors.
        """
        A = jnp.vstack((A, new_A_rows))
        lb = jnp.hstack((lb, new_lbs))
        ub = jnp.hstack((ub, new_ubs))

        return A, lb, ub

    @jaxtyped
    @beartype
    def make_constraints(
        self,
        initial_dispatchable_power: Float[Array, " nd"],
        initial_intermittent_power: Float[Array, " ni"],
        initial_storage_power: Float[Array, " ns"],
        initial_storage_state_of_charge: Float[Array, " ns"],
        predicted_load_limits: Float[Array, "Th nl 2"],
        predicted_intermittent_limits: Float[Array, "Th ni"],
        line_limits: Float[Array, " nline"],
    ) -> Tuple[Float[Array, "nc nx"], Float[Array, " nc"], Float[Array, " nc"]]:
        """
        Assemble the constraint matrix for the dynamic dispatch problem.

        This problem has the following constraints:
          - Overall power conservation (total power injection = 0)
          - Dispatchable and intermittent generation limits
          - Storage charge/discharge limits
          - Load limits
          - Ramp rate limits
          - Max flow constraints
          - Power conservation at each bus (injections + flows)
          - Storage charge dynamics
          - Storage min/max state of charge

        The constraints for jaxopt.BoxOSQP have the form Ax = z and lb <= z <= ub.

        args:
            initial_dispatchable_power: initial power output of dispatchable resources
            initial_intermittent_power: initial power output of intermittent resources
            initial_storage_power: initial power output of storage resources
            initial_storage_state_of_charge: initial state of charge of storage
            predicted_load_limits: predicted min/max load limits over the horizon
            predicted_intermittent_limits: predicted min/max intermittent generation
                limits over the horizon
            line_limits: current line limits (assumed constant over the horizon)
        returns:
            A, lb, and ub in the expression above.
        """
        # Create the constraint matrix and bound vectors starting with one dummy row
        # (we'll remove this row once all the other constraints are added)
        A = jnp.zeros((1, self.num_decision_variables))
        lb = jnp.zeros((1,))
        ub = jnp.zeros((1,))

        # Add power conservation constraints for each timestep
        for t in range(self.T_horizon):
            # Sum of generation = sum of loads
            k = t * self.num_variables_per_step
            variable_start = k
            variable_end = k + self.num_injections
            new_A_rows = jnp.zeros((1, self.num_decision_variables))
            new_A_rows = new_A_rows.at[:, variable_start:variable_end].set(1.0)
            new_lb = jnp.array([0.0])
            new_ub = jnp.array([0.0])
            A, lb, ub = self.add_to_constraints(A, lb, ub, new_A_rows, new_lb, new_ub)

        # Add power injection min/max limits for each timestep
        for t in range(self.T_horizon):
            # min <= injection <= max
            k = t * self.num_variables_per_step
            variable_start = k
            variable_end = k + self.num_injections
            new_A_rows = jnp.zeros((self.num_injections, self.num_decision_variables))
            new_A_rows = new_A_rows.at[:, variable_start:variable_end].set(
                jnp.eye(self.num_injections)
            )
            new_lbs = jnp.hstack(
                (
                    self.dispatchable_limits[:, 0],
                    0.0 * predicted_intermittent_limits[t, :],
                    self.storage_power_limits[:, 0],
                    # minus max load, since loads are negative injection
                    -predicted_load_limits[t, :, 1],
                )
            )
            new_ubs = jnp.hstack(
                (
                    self.dispatchable_limits[:, 1],
                    predicted_intermittent_limits[t, :],
                    self.storage_power_limits[:, 1],
                    # minus min load, since loads are negative injection
                    -predicted_load_limits[t, :, 0],
                )
            )
            A, lb, ub = self.add_to_constraints(A, lb, ub, new_A_rows, new_lbs, new_ubs)

        # Add max ramp rate limits for first timestep (relative to the initial power)
        # -max + initial <= injection <= max + initial
        ramp_rates = jnp.hstack(
            (
                self.dispatchable_ramp_rates,
                self.intermittent_ramp_rates,
                self.storage_ramp_rates,
            )
        )
        new_A_rows = jnp.zeros((self.num_generators, self.num_decision_variables))
        new_A_rows = new_A_rows.at[:, : self.num_generators].set(
            jnp.eye(self.num_generators)
        )
        initial_generation = jnp.hstack(
            (
                initial_dispatchable_power,
                initial_intermittent_power,
                initial_storage_power,
            )
        )
        A, lb, ub = self.add_to_constraints(
            A,
            lb,
            ub,
            new_A_rows,
            initial_generation - ramp_rates,
            initial_generation + ramp_rates,
        )

        # Add max ramp rate limits for each subsequent timestep
        for t in range(self.T_horizon - 1):
            # -max <= injection[k+1] - injection[k] <= max
            k = t * self.num_variables_per_step
            k_next = (t + 1) * self.num_variables_per_step
            variable_start = k
            variable_end = k + self.num_generators
            next_var_start = k_next
            next_var_end = k_next + self.num_generators
            new_A_rows = jnp.zeros((self.num_generators, self.num_decision_variables))
            new_A_rows = new_A_rows.at[:, next_var_start:next_var_end].set(
                jnp.eye(self.num_generators)
            )
            new_A_rows = new_A_rows.at[:, variable_start:variable_end].set(
                -jnp.eye(self.num_generators)
            )
            A, lb, ub = self.add_to_constraints(
                A, lb, ub, new_A_rows, -ramp_rates, ramp_rates
            )

        # Add flow limits for each timestep
        for t in range(self.T_horizon):
            # -max <= flow <= max
            k = t * self.num_variables_per_step
            variable_start = k + self.num_injections
            variable_end = k + self.num_injections + self.num_lines
            new_A_rows = jnp.zeros((self.num_lines, self.num_decision_variables))
            new_A_rows = new_A_rows.at[:, variable_start:variable_end].set(
                jnp.eye(self.num_lines)
            )
            A, lb, ub = self.add_to_constraints(
                A, lb, ub, new_A_rows, -line_limits, line_limits
            )

        # Constrain total injections at each bus to equal total flow from each bus
        # at each timestep
        for bus_idx in range(self.num_buses):
            # Get the resources at this bus
            dispatchable_names_at_bus = self.dispatchables_at_bus[bus_idx]
            dispatchable_at_bus = [
                self.dispatchable_names.index(name)
                for name in dispatchable_names_at_bus
            ]
            intermittent_names_at_bus = self.intermittents_at_bus[bus_idx]
            intermittent_at_bus = [
                self.intermittent_names.index(name)
                for name in intermittent_names_at_bus
            ]
            storage_names_at_bus = self.storage_at_bus[bus_idx]
            storage_at_bus = [
                self.storage_names.index(name) for name in storage_names_at_bus
            ]
            load_names_at_bus = self.loads_at_bus[bus_idx]
            loads_at_bus = [self.load_names.index(name) for name in load_names_at_bus]
            lines_leaving_bus = [
                i for i, line in enumerate(self.lines) if line[0] == bus_idx
            ]
            lines_entering_bus = [
                i for i, line in enumerate(self.lines) if line[1] == bus_idx
            ]

            # Convert generator/load/line indices to variable indices.
            # x = [dispatchable, intermittent, storage, loads, flows]
            offset = self.num_dispatchables
            intermittent_at_bus = [var_idx + offset for var_idx in intermittent_at_bus]
            offset += self.num_intermittents
            storage_at_bus = [var_idx + offset for var_idx in storage_at_bus]
            offset += self.num_storage_resources
            loads_at_bus = [var_idx + offset for var_idx in loads_at_bus]
            offset += self.num_loads
            lines_leaving_bus = [var_idx + offset for var_idx in lines_leaving_bus]
            lines_entering_bus = [var_idx + offset for var_idx in lines_entering_bus]

            for t in range(self.T_horizon):
                # Sum of injections + flows entering bus - flows leaving bus = 0
                # Remember that loads are negative injections.
                injections_at_bus = (
                    dispatchable_at_bus
                    + intermittent_at_bus
                    + storage_at_bus
                    + loads_at_bus
                )

                # Offset all variable indices to align with current timestep
                k = t * self.num_variables_per_step
                injections_at_bus_t = [k + var_idx for var_idx in injections_at_bus]
                lines_leaving_bus_t = [k + var_idx for var_idx in lines_leaving_bus]
                lines_entering_bus_t = [k + var_idx for var_idx in lines_entering_bus]

                new_A_rows = jnp.zeros((1, self.num_decision_variables))
                new_A_rows = new_A_rows.at[:, injections_at_bus_t].set(1.0)
                new_A_rows = new_A_rows.at[:, lines_entering_bus_t].set(1.0)
                new_A_rows = new_A_rows.at[:, lines_leaving_bus_t].set(-1.0)
                new_lb = jnp.array([0.0])
                new_ub = jnp.array([0.0])
                A, lb, ub = self.add_to_constraints(
                    A, lb, ub, new_A_rows, new_lb, new_ub
                )

        # Enforce charge dynamics at the initial timestep
        # initial_charge - power = next_charge
        # or -next_charge - power = -initial_charge
        # (negatives make it consistent with later steps)
        power_start_idx = self.num_dispatchables + self.num_intermittents
        power_end_idx = power_start_idx + self.num_storage_resources
        next_charge_start_idx = self.num_injections + self.num_lines
        next_charge_end_idx = next_charge_start_idx + self.num_storage_resources
        new_A_rows = jnp.zeros(
            (self.num_storage_resources, self.num_decision_variables)
        )
        new_A_rows = new_A_rows.at[:, power_start_idx:power_end_idx].set(
            -jnp.eye(self.num_storage_resources)
        )
        new_A_rows = new_A_rows.at[:, next_charge_start_idx:next_charge_end_idx].set(
            -jnp.eye(self.num_storage_resources)
        )
        A, lb, ub = self.add_to_constraints(
            A,
            lb,
            ub,
            new_A_rows,
            -initial_storage_state_of_charge,
            -initial_storage_state_of_charge,
        )

        # Add constraints to enforce charge dynamics at all subsequent timesteps
        # prev_charge - power = current_charge
        # or prev_charge - current_charge - power = 0
        for t in range(1, self.T_horizon):
            k = t * self.num_variables_per_step
            k_prev = (t - 1) * self.num_variables_per_step
            power_start_idx = k + self.num_dispatchables + self.num_intermittents
            power_end_idx = power_start_idx + self.num_storage_resources
            current_charge_start_idx = k + self.num_injections + self.num_lines
            current_charge_end_idx = (
                current_charge_start_idx + self.num_storage_resources
            )
            prev_charge_start_idx = k_prev + self.num_injections + self.num_lines
            prev_charge_end_idx = prev_charge_start_idx + self.num_storage_resources
            new_A_rows = jnp.zeros(
                (self.num_storage_resources, self.num_decision_variables)
            )
            new_A_rows = new_A_rows.at[:, power_start_idx:power_end_idx].set(
                -jnp.eye(self.num_storage_resources)
            )
            new_A_rows = new_A_rows.at[
                :, current_charge_start_idx:current_charge_end_idx
            ].set(-jnp.eye(self.num_storage_resources))
            new_A_rows = new_A_rows.at[
                :, prev_charge_start_idx:prev_charge_end_idx
            ].set(jnp.eye(self.num_storage_resources))
            new_lbs = jnp.zeros((self.num_storage_resources,))
            new_ubs = jnp.zeros((self.num_storage_resources,))
            A, lb, ub = self.add_to_constraints(A, lb, ub, new_A_rows, new_lbs, new_ubs)

        # Add constraints for min/max storage charge
        # min <= charge <= max
        for t in range(self.T_horizon):
            k = t * self.num_variables_per_step
            current_charge_start_idx = k + self.num_injections + self.num_lines
            current_charge_end_idx = (
                current_charge_start_idx + self.num_storage_resources
            )
            new_A_rows = jnp.zeros(
                (self.num_storage_resources, self.num_decision_variables)
            )
            new_A_rows = new_A_rows.at[
                :, current_charge_start_idx:current_charge_end_idx
            ].set(jnp.eye(self.num_storage_resources))
            A, lb, ub = self.add_to_constraints(
                A,
                lb,
                ub,
                new_A_rows,
                self.storage_charge_limits[:, 0],
                self.storage_charge_limits[:, 1],
            )

        # Remove the dummy row from the constraints
        A = A[1:, :]
        lb = lb[1:]
        ub = ub[1:]

        return A, lb, ub

    @jaxtyped
    @beartype
    def solve_dynamic_dispatch(
        self,
        initial_dispatchable_power: Float[Array, " nd"],
        initial_intermittent_power: Float[Array, " ni"],
        initial_storage_power: Float[Array, " ns"],
        initial_storage_state_of_charge: Float[Array, " ns"],
        predicted_load_limits: Float[Array, "Th nl 2"],
        predicted_intermittent_limits: Float[Array, "Th ni"],
        line_limits: Float[Array, " nline"],
    ) -> PyTree[Float[Array, "..."]]:
        """
        Formulate and solve the dynamic dispatch QP.

        args:
            initial_dispatchable_power: initial power output of dispatchable resources
            initial_intermittent_power: initial power output of intermittent resources
            initial_storage_power: initial power output of storage resources
            initial_storage_state_of_charge: initial state of charge of storage
            predicted_load_limits: predicted min/max load limits over the horizon
            predicted_intermittent_limits: predicted min/max intermittent generation
                limits over the horizon
            line_limits: current line limits (assumed constant over the horizon)

        returns:
            injection schedule for all generators, storage, and loads; charge schedule
            for all storage; and line flows for all lines
        """
        # The cost has the form x.T Q x + c.T x
        Q = self.make_quadratic_cost_matrix()
        c = self.make_linear_cost_vector()

        # The constraints have the form A x = z, lb <= z <= ub
        A, lb, ub = self.make_constraints(
            initial_dispatchable_power,
            initial_intermittent_power,
            initial_storage_power,
            initial_storage_state_of_charge,
            predicted_load_limits,
            predicted_intermittent_limits,
            line_limits,
        )

        print(
            (
                f"Constructed QP with {self.num_decision_variables} decision variables"
                f" and {A.shape[0]} constraints."
            )
        )
        print(f"\tQ: {Q.shape}, c: {c.shape}")
        print(f"\tA: {A.shape}, lb: {lb.shape}, ub: {ub.shape}")

        # Create a solver
        qp = BoxOSQP()
        sol = qp.run(params_obj=(Q, c), params_eq=A, params_ineq=(lb, ub))

        x_solution = sol.params.primal[0]

        # Extract the results
        dispatchable_schedule = jnp.zeros((self.T_horizon, self.num_dispatchables))
        intermittent_schedule = jnp.zeros((self.T_horizon, self.num_intermittents))
        storage_discharge_schedule = jnp.zeros(
            (self.T_horizon, self.num_storage_resources)
        )
        load_schedule = jnp.zeros((self.T_horizon, self.num_loads))
        line_flow_schedule = jnp.zeros((self.T_horizon, self.num_lines))
        storage_soc_schedule = jnp.zeros((self.T_horizon, self.num_storage_resources))

        for t in range(self.T_horizon):
            k = t * self.num_variables_per_step
            dispatchable_schedule = dispatchable_schedule.at[t, :].set(
                x_solution[k : k + self.num_dispatchables]
            )
            k += self.num_dispatchables
            intermittent_schedule = intermittent_schedule.at[t, :].set(
                x_solution[k : k + self.num_intermittents]
            )
            k += self.num_intermittents
            storage_discharge_schedule = storage_discharge_schedule.at[t, :].set(
                x_solution[k : k + self.num_storage_resources]
            )
            k += self.num_storage_resources
            load_schedule = load_schedule.at[t, :].set(
                -x_solution[k : k + self.num_loads]
            )
            k += self.num_loads
            line_flow_schedule = line_flow_schedule.at[t, :].set(
                x_solution[k : k + self.num_lines]
            )
            k += self.num_lines
            storage_soc_schedule = storage_soc_schedule.at[t, :].set(
                x_solution[k : k + self.num_storage_resources]
            )
            k += self.num_storage_resources

        return {
            "dispatchable_schedule": dispatchable_schedule,
            "intermittent_schedule": intermittent_schedule,
            "storage_discharge_schedule": storage_discharge_schedule,
            "load_schedule": load_schedule,
            "line_flow_schedule": line_flow_schedule,
            "storage_soc_schedule": storage_soc_schedule,
            "x": x_solution,
            "Q": Q,
            "c": c,
            "A": A,
            "lb": lb,
            "ub": ub,
        }

    def get_current_line_limits(self, t: int) -> Float[Array, " nl"]:
        """
        Get the current line limits accounting for failures.

        args:
            t: current timestep
        """
        # Model a failure as a transient decrease in line capacity
        smoothing = 50.0
        pre_failure_scaling = sigmoid(2 * (self.line_failure_times - t))
        recovery_times = self.line_failure_times + self.line_failure_durations
        post_failure_scaling = sigmoid(2 * (t - recovery_times))
        scale_factor = (
            1
            / smoothing
            * logsumexp(
                smoothing * jnp.vstack((pre_failure_scaling, post_failure_scaling)),
                axis=0,
            )
        )

        return scale_factor * self.nominal_line_limits

    @jaxtyped
    @beartype
    def simulate(
        self,
        initial_dispatchable_power: Float[Array, " nd"],
        initial_intermittent_power: Float[Array, " ni"],
        initial_storage_power: Float[Array, " ns"],
        initial_storage_state_of_charge: Float[Array, " ns"],
    ) -> PyTree[Float[Array, "..."]]:
        """
        Simulate the performance of the centralized dynamic dispatch system.

        args:
            initial_dispatchable_power: initial power output of dispatchable resources
            initial_intermittent_power: initial power output of intermittent resources
            initial_storage_power: initial power output of storage resources
            initial_storage_state_of_charge: initial state of charge of storage

        returns:
            - generation schedule for each generation/storage resource at each timestep
            - fraction of total load served at each timestep
        """

        @jax.jit
        def simulate_step(carry, t):
            # Extract previous state
            (
                prev_dispatchable_power,
                prev_intermittent_power,
                prev_storage_power,
                prev_storage_state_of_charge,
            ) = carry
            # Get relevant window of predictions
            predicted_load_limits = jax.lax.dynamic_slice_in_dim(
                self.load_true_limits, t, self.T_horizon, axis=0
            )
            predicted_load_errs = jax.lax.dynamic_slice_in_dim(
                self.load_limits_prediction_err, t + 1, self.T_horizon - 1, axis=0
            )
            predicted_load_limits = predicted_load_limits.at[1:, :, :].add(
                predicted_load_errs.reshape(self.T_horizon - 1, -1, 1)
            )

            predicted_intermittent_limits = jax.lax.dynamic_slice_in_dim(
                self.intermittent_true_limits, t, self.T_horizon, axis=0
            )
            predicted_intermittent_errs = jax.lax.dynamic_slice_in_dim(
                self.intermittent_limits_prediction_err,
                t + 1,
                self.T_horizon - 1,
                axis=0,
            )
            predicted_intermittent_limits = predicted_intermittent_limits.at[1:, :].add(
                predicted_intermittent_errs
            )

            # Update the line limits for the current time
            line_limits = self.get_current_line_limits(t)

            # Solve the dynamic dispatch problem
            result = self.solve_dynamic_dispatch(
                prev_dispatchable_power,
                prev_intermittent_power,
                prev_storage_power,
                prev_storage_state_of_charge,
                predicted_load_limits,
                predicted_intermittent_limits,
                line_limits,
            )

            # Extract results
            dispatchable_power = result["dispatchable_schedule"][0]
            intermittent_power = result["intermittent_schedule"][0]
            storage_power = result["storage_discharge_schedule"][0]
            storage_state_of_charge = result["storage_soc_schedule"][0]
            load_schedule = result["load_schedule"][0]

            # Record total % load served (as a fraction of maximum demand)
            load_served = load_schedule.sum() / self.load_true_limits[t, :, 1].sum()

            # Return the next carry and the output from this step
            next_carry = (
                dispatchable_power,
                intermittent_power,
                storage_power,
                storage_state_of_charge,
            )
            output = {
                "load_served": load_served,
                "dispatchable_schedule": dispatchable_power,
                "intermittent_schedule": intermittent_power,
                "storage_discharge_schedule": storage_power,
                "storage_soc_schedule": storage_state_of_charge,
            }

            return next_carry, output

        # Simulate and return the output
        initial_carry = (
            initial_dispatchable_power,
            initial_intermittent_power,
            initial_storage_power,
            initial_storage_state_of_charge,
        )
        _, output = jax.lax.scan(
            simulate_step, initial_carry, jnp.arange(self.T_sim - self.T_horizon)
        )

        return output
