import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax.nn import log_sigmoid, sigmoid, relu
from jaxtyping import Array, Float, Integer, jaxtyped

from architect.problem_definition import AutonomousSystem


# Utility type
PRNGKeyArray = Integer[Array, "2"]


# @jaxtyped
# @beartype  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class Dispatch(NamedTuple):
    """
    Dispatched voltage amplitude and angles for AC power flow.

    args:
        voltage_amplitudes: voltage amplitude at each bus
        voltage_angles: voltage angle at each bus
    """

    voltage_amplitudes: Float[Array, " n_bus"]
    voltage_angles: Float[Array, " n_bus"]


# @jaxtyped
# @beartype  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class Network(NamedTuple):
    """
    Representation of network connectivity for AC power flow.

    Includes admittances between nodes, but not shunt admittances.

    args:
        line_conductances: n_lines-element array of conductances connecting buses.
            If [i, j] is the k-th row of `lines`, then the k-th entry of `conductances`
            contains the conductance connecting bus i and bus j.
        line_susceptances: n_lines-element array of susceptances connecting buses.
            If [i, j] is the k-th row of `lines`, then the k-th entry of `susceptances`
            contains the susceptance connecting bus i and bus j.
    """

    line_conductances: Float[Array, " n_lines"]
    line_susceptances: Float[Array, " n_lines"]

    @property
    def n_line(self):
        """Return the number of lines in this network"""
        return self.line_conductances.shape[0]

    def Y(
        self,
        lines: Integer[Array, "n_lines 2"],
        shunt_conductances: Float[Array, " n_bus"],
        shunt_susceptances: Float[Array, " n_bus"],
    ):
        """
        Compute the nodal admittance matrix for this network.

        args:
            lines: n_lines x 2 array of integers specifying bus connections. If [i, j]
                is a row in this array, then buses i and j will be connected. Each
                connection should be specified only once (so if [i, j] is included,
                it is not necessary to include [j, i] as well)
            shunt_conductances: n_lines-element array of shunt conductances connecting
                buses to ground.
            shunt_susceptances: n_lines-element array of shunt susceptances connecting
                buses to ground.
        """
        # Clip all line and shunt conductances to make sure they're non-negative
        line_conductances = jnp.clip(self.line_conductances, a_min=0.0)
        shunt_conductances = jnp.clip(shunt_conductances, a_min=0.0)

        # Assemble conductance and susceptance into complex admittance
        line_admittances = line_conductances + 1j * self.line_susceptances
        shunt_admittances = shunt_conductances + 1j * shunt_susceptances

        # Assemble the admittance matrix from the given data
        # For off-diagonal elements, Y_ij = negative admittance connecting bus i and j
        # For diagonal elements, Y_ii = shunt admittance at bus i - sum of off-diagonal
        #   entries
        n_bus = shunt_conductances.shape[0]
        Y_off_diagonal = jsparse.BCOO(
            (-1.0 * line_admittances, lines), shape=(n_bus, n_bus)
        )
        Y_off_diagonal = Y_off_diagonal + Y_off_diagonal.T

        diagonal_indices = jnp.arange(0, n_bus).reshape(-1, 1)
        diagonal_indices = jnp.hstack((diagonal_indices, diagonal_indices))
        diagonal_entries = shunt_admittances - Y_off_diagonal.sum(axis=1).todense()
        Y_diagonal = jsparse.BCOO(
            (diagonal_entries, diagonal_indices), shape=(n_bus, n_bus)
        )

        Y = Y_diagonal + Y_off_diagonal

        return Y


# @jaxtyped
# @beartype  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class ACOPFResult(NamedTuple):
    """
    Result of running the ACOPF algorithm.

    Includes the voltage amplitude, voltage angle, active power injection, and reactive
    power injection at each bus. Also includes the total constraint violations, cost,
    augmented cost ("potential")
    """

    dispatch: Dispatch
    network: Network

    P: Float[Array, " n_bus"]
    Q: Float[Array, " n_bus"]

    P_violation: Float[Array, ""]
    Q_violation: Float[Array, ""]
    V_violation: Float[Array, ""]

    generation_cost: Float[Array, ""]
    potential: Float[Array, ""]


@jaxtyped
@beartype
class ACOPF(AutonomousSystem):
    """
    Defines an autonomous system modelling AC optimal power flow.

    # TODO support thermal limits on lines

    args:
        nominal_network: the nominal admittances of the network lines
        lines: n_lines x 2 array of integers specifying bus connections. If [i, j]
            is a row in this array, then buses i and j will be connected. Each
            connection should be specified only once (so if [i, j] is included,
            it is not necessary to include [j, i] as well)
        shunt_conductances: n_lines-element array of shunt conductances connecting
            buses to ground.
        shunt_susceptances: n_lines-element array of shunt susceptances connecting
            buses to ground.
        bus_active_limits: n_bus x 2 array of (min, max) active power limits at each
            bus (positive for generation, negative for load)
        bus_reactive_limits: n_bus x 2 array of (min, max) reactive power limits at
            each bus (positive for generation, negative for load)
        bus_voltage_limits: n_bus x 2 array of (min, max) voltage amplitude at each bus
        bus_active_linear_costs: n_bus-element array of linear cost per p.u. active
            power at each bus (positive for generation cost, negative for load benefit)
        bus_active_quadratic_costs: n_bus-element array of quadratic cost per p.u.
            active power squared at each bus
        bus_reactive_linear_costs: n_bus-element array of linear cost per p.u. reactive
            power magnitude at each bus
        constraint_penalty: scalar penalty coefficient for violating power, voltage, or
            angle constraints
        prob_line_failure: n_line-element array of prior probabilities of line failures
        line_conductance_stddev: n_line-element array of standard deviation of line
            conductances in both the nominal and failure cases.
        line_susceptance_stddev: n_line-element array of standard deviation of line
            susceptances in both the nominal and failure cases.
    """

    nominal_network: Network
    lines: Integer[Array, "n_lines 2"]
    shunt_conductances: Float[Array, " n_bus"]
    shunt_susceptances: Float[Array, " n_bus"]

    bus_active_limits: Float[Array, "n_bus 2"]
    bus_reactive_limits: Float[Array, "n_bus 2"]
    bus_voltage_limits: Float[Array, "n_bus 2"]

    bus_active_linear_costs: Float[Array, " n_bus"]
    bus_active_quadratic_costs: Float[Array, " n_bus"]
    bus_reactive_linear_costs: Float[Array, " n_bus"]

    constraint_penalty: float

    prob_line_failure: Float[Array, " n_line"]
    line_conductance_stddev: Float[Array, " n_line"]
    line_susceptance_stddev: Float[Array, " n_line"]

    @property
    def n_bus(self) -> int:
        """Return the number of buses in the network"""
        return self.shunt_conductances.shape[0]

    @property
    def n_line(self) -> int:
        """Return the number of lines in the network"""
        return self.nominal_network.n_line

    @jaxtyped
    @beartype
    def power_injections(
        self, dispatch: Dispatch, network: Network
    ) -> Tuple[Float[Array, " n_bus"], Float[Array, " n_bus"]]:
        """
        Compute the real and reactive power injections at each bus.

        args:
            dispatch: the voltage amplitudes and angles
            network: the network to use to solve the injections
        returns:
            a tuple of real and reactive power injections at each bus
        """
        # Compute the complex voltage at each bus
        V = dispatch.voltage_amplitudes * jnp.exp(dispatch.voltage_angles * 1j)

        # Compute the complex current and power that results from the proposed voltages
        # TODO@dawsonc convert to sparse admittance matrix once
        # https://github.com/google/jax/issues/13118 is resolved
        Y = network.Y(self.lines, self.shunt_conductances, self.shunt_susceptances)
        Y = Y.todense()
        # current = jsparse.sparsify(lambda Y, V: Y @ V)(
        #     Y, V
        # )
        current = jnp.dot(Y, V)
        S = V * current.conj()

        # Extract real and reactive power from complex power
        P, Q = S.real, S.imag

        return P, Q

    @jaxtyped
    @beartype
    def generation_cost(
        self, P: Float[Array, " n_bus"], Q: Float[Array, " n_bus"]
    ) -> Float[Array, ""]:
        """
        Compute the cost of generation given power injections.
        """
        quad_cost = self.bus_active_quadratic_costs * P ** 2
        lin_cost = self.bus_active_linear_costs * P
        q_cost = self.bus_reactive_linear_costs * jnp.abs(Q)
        return (quad_cost + lin_cost + q_cost).sum()

    @jaxtyped
    @beartype
    def constraint_violations(
        self, dispatch: Dispatch, P: Float[Array, " n_bus"], Q: Float[Array, " n_bus"]
    ) -> Float[Array, " n_constraints"]:
        """
        Compute an array of constraint violations for:
        - Active power limits
        - Reactive power limits
        - Voltage amplitude limits
        """
        constraint_violations = []

        # Active power limits
        P_min, P_max = self.bus_active_limits.T
        P_violation = (relu(P - P_max) + relu(P_min - P)).sum()
        constraint_violations.append(P_violation)

        # Reactive power limits
        Q_min, Q_max = self.bus_reactive_limits.T
        Q_violation = (relu(Q - Q_max) + relu(Q_min - Q)).sum()
        constraint_violations.append(Q_violation)

        # Voltage amplitude limits
        V_min, V_max = self.bus_voltage_limits.T
        V_violation = (
            relu(dispatch.voltage_amplitudes - V_max)
            + relu(V_min - dispatch.voltage_amplitudes)
        ).sum()
        constraint_violations.append(V_violation)

        return jnp.stack(constraint_violations)

    @jaxtyped
    @beartype
    def repair_dispatch(
        self,
        dispatch: Dispatch,
        network: Network,
        num_steps: int = 10,
        step_size: float = 1e-4,
    ) -> Dispatch:
        """
        Use gradient descent to locally modify the dispatch to satisfy the constraints.

        args:
            dispatch: the voltage dispatch to repair
            network: the network in use
            num_steps: how many gradient steps to take
            step_size: the size of the repair steps
        returns:
            a repaired dispatch (not guaranteed to satisfy the constraints, but should
            be closer to feasible)
        """
        # Define the loss function to repair
        def loss(dispatch):
            P, Q = self.power_injections(dispatch, network)
            violation = self.constraint_violations(dispatch, P, Q).sum()
            return violation

        # Define a function to execute one repair step
        V_min, V_max = self.bus_voltage_limits.T

        def repair_step(_, dispatch):
            # Take a gradient step
            grads = jax.grad(loss)(dispatch)
            dispatch = jtu.tree_map(lambda d, g: d - step_size * g, dispatch, grads)
            # Project voltage amplitudes to limits
            fixed_voltage_amplitudes = jnp.clip(
                dispatch.voltage_amplitudes, a_min=V_min, a_max=V_max
            )
            dispatch = Dispatch(fixed_voltage_amplitudes, dispatch.voltage_angles)
            return dispatch

        # Execute the step function a bunch of times and return the result
        repaired_dispatch = jax.lax.fori_loop(0, num_steps, repair_step, dispatch)
        return repaired_dispatch

    @jaxtyped
    @beartype
    def __call__(
        self,
        dispatch: Dispatch,
        network: Network,
        repair_steps: int = 10,
        repair_lr: float = 1e-4,
    ) -> ACOPFResult:
        """
        Computes the power flows associated with the proposed bus voltages and angles.

        args:
            dispatch: the voltage dispatch to use when solving power flow
            network: the network to use for solving power flow
            repair_steps: how many steps to take to correct infeasible dispatches
            repair_lr: the size of the repair steps
        returns:
            an ACOPFResult including the repaired dispatch
        """
        # Repair the dispatch using gradient descent (if necessary) to satisfy the
        # constraints
        dispatch = self.repair_dispatch(dispatch, network, repair_steps, repair_lr)

        # Compute the power injections for the repaired dispatch
        P, Q = self.power_injections(dispatch, network)

        # Compute constraint violations
        constraint_violations = self.constraint_violations(dispatch, P, Q)
        total_violation = constraint_violations.sum()

        # Compute the net cost of generation
        generation_cost = self.generation_cost(P, Q)

        # Compute a normalizing factor for generation cost in the mean-P case
        P_min, P_max = self.bus_active_limits.T
        Q_min, Q_max = self.bus_reactive_limits.T
        P_mean = (P_max - P_min) / 2.0
        Q_mean = (Q_max - Q_min) / 2.0
        cost_normalization = jnp.abs(self.generation_cost(P_mean, Q_mean))

        # Compute a normalizing factor for the constraints as well
        constraint_normalization = jnp.maximum(jnp.abs(P_max), jnp.abs(P_min)).sum()

        # Combine cost and violations to get the potential
        potential = (
            generation_cost / cost_normalization
            + self.constraint_penalty * total_violation / constraint_normalization
        )

        result = ACOPFResult(
            dispatch,
            network,
            P,
            Q,
            constraint_violations[0],
            constraint_violations[1],
            constraint_violations[2],
            generation_cost,
            potential,
        )

        return result

    @jaxtyped
    @beartype
    def dispatch_prior_logprob(self, dispatch: Dispatch) -> Float[Array, ""]:
        """
        Compute the prior log probability of the given dispatch.

        Probability is not necessarily normalized.

        We assume that the prior distribution for the dispatch is uniform with respect
        to voltage amplitude between the specified min and max values, while angle is
        assumed to be Gaussian centered at 0 with standard deviation pi/4.

        We use a smooth approximation of the uniform distribution.
        """
        # Define the smoothing constant
        b = 50.0

        def log_smooth_uniform(x, x_min, x_max):
            return log_sigmoid(b * (x - x_min)) + log_sigmoid(b * (x_max - x))

        V_min, V_max = self.bus_voltage_limits.T
        logprob = log_smooth_uniform(dispatch.voltage_amplitudes, V_min, V_max).sum()

        sigma_theta = jnp.pi / 4
        logprob += jax.scipy.stats.multivariate_normal.logpdf(
            dispatch.voltage_angles,
            mean=jnp.zeros(self.n_bus),
            cov=sigma_theta ** 2 * jnp.eye(self.n_bus),
        )

        return logprob

    @jaxtyped
    @beartype
    def sample_random_dispatch(self, key: PRNGKeyArray) -> Dispatch:
        """
        Sample a random dispatch from the uniform distribution

            |V| x theta \sim [V_min, V_max] x N(0, pi/4)

        args:
            key: PRNG key to use for sampling
        """
        amplitude_key, angle_key = jrandom.split(key)

        V_min, V_max = self.bus_voltage_limits.T
        voltage_amplitudes = jrandom.uniform(
            amplitude_key, shape=(self.n_bus,), minval=V_min, maxval=V_max
        )

        sigma_theta = jnp.pi / 4
        voltage_angles = jrandom.multivariate_normal(
            angle_key,
            mean=jnp.zeros(self.n_bus),
            cov=sigma_theta ** 2 * jnp.eye(self.n_bus),
        )

        return Dispatch(voltage_amplitudes, voltage_angles)

    @jaxtyped
    @beartype
    def network_prior_logprob(self, network: Network) -> Float[Array, ""]:
        """
        Compute the prior log probability of the given network.

        Probability is not necessarily normalized.

        We use a mixture model for failures:

            With probability 1 - self.prob_line_failure, the conductance and susceptance
            are sampled from Gaussians with mean at the nominal value and standard
            deviation self.line_conductance_stddev and self.line_susceptance_stddev.

            With probability 1 - self.prob_line_failure, the conductance and susceptance
            are sampled from Gaussians with zero mean at the nominal value and standard
            deviation self.line_conductance_stddev and self.line_susceptance_stddev.
            Conductances are clipped to be non-negative

        Failures are assumed to be indepedendent.

        We assume that shunt admittances are not subject to failures.
        """
        # Unpack the network conductances and susceptances
        line_conductances = network.line_conductances
        line_susceptances = network.line_susceptances

        # Get the nominal line conductance and susceptance
        nominal_line_conductances = self.nominal_network.line_conductances
        nominal_line_susceptances = self.nominal_network.line_susceptances

        # The PDF of the mixture is a convex combination of the nominal- and
        # failure-case PDFs. Both have a similar form based on a element-wise Gaussian
        gaussian_pdf = lambda y, y_nom, sigma: jax.scipy.stats.norm.pdf(
            y, loc=y_nom, scale=sigma
        )

        # Compute the nominal-case likelihood
        nominal_likelihood = jax.vmap(gaussian_pdf)(
            line_conductances, nominal_line_conductances, self.line_conductance_stddev
        ) * jax.vmap(gaussian_pdf)(
            line_susceptances, nominal_line_susceptances, self.line_susceptance_stddev
        )

        # Compute the failure-case likelihood, where we add a term to smoothly clip
        # conductances to be non-negative
        def smooth_clip(x, x_min):
            b = 50.0  # smoothing
            return sigmoid(b * (x - x_min))

        failure_likelihood = (
            jax.vmap(gaussian_pdf)(
                line_conductances,
                jnp.zeros_like(nominal_line_conductances),
                self.line_conductance_stddev,
            )
            * jax.vmap(gaussian_pdf)(
                line_susceptances,
                jnp.zeros_like(nominal_line_susceptances),
                self.line_susceptance_stddev,
            )
            * smooth_clip(line_conductances, 0.0)
        )

        # Combine the likeihoods by weighting the probability of failure of each line
        p_failure = self.prob_line_failure
        p_nominal = 1 - p_failure
        likelihood = p_nominal * nominal_likelihood + p_failure * failure_likelihood

        # Return the overall log likelihood
        logprob = jnp.log(likelihood).sum()

        return logprob

    @jaxtyped
    @beartype
    def sample_random_network(self, key: PRNGKeyArray) -> Network:
        """
        Sample a random network.

        We use a mixture model for failures:

            With probability 1 - self.prob_line_failure, the conductance and susceptance
            are sampled from Gaussians with mean at the nominal value and standard
            deviation self.line_conductance_stddev and self.line_susceptance_stddev.

            With probability 1 - self.prob_line_failure, the conductance and susceptance
            are sampled from Gaussians with zero mean at the nominal value and standard
            deviation self.line_conductance_stddev and self.line_susceptance_stddev.
            Conductances are clipped to be non-negative

        args:
            key: PRNG key to use for sampling
        """
        failure_key, nominal_key, failed_key = jrandom.split(key, 3)

        # Sample failure states for each line
        line_failures = jrandom.bernoulli(
            failure_key, self.prob_line_failure, shape=(self.nominal_network.n_line,)
        )

        # Sample nominal- and failure-case scale factors for each line
        conductance_key, susceptance_key = jrandom.split(nominal_key)
        nominal_conductances = jrandom.multivariate_normal(
            conductance_key,
            mean=self.nominal_network.line_conductances,
            cov=self.line_conductance_stddev * jnp.eye(self.n_line),
        )
        nominal_susceptances = jrandom.multivariate_normal(
            susceptance_key,
            mean=self.nominal_network.line_susceptances,
            cov=self.line_susceptance_stddev * jnp.eye(self.n_line),
        )
        conductance_key, susceptance_key = jrandom.split(failed_key)
        failure_conductances = jrandom.multivariate_normal(
            conductance_key,
            mean=jnp.zeros(self.n_line),
            cov=self.line_conductance_stddev * jnp.eye(self.n_line),
        )
        failure_susceptances = jrandom.multivariate_normal(
            susceptance_key,
            mean=jnp.zeros(self.n_line),
            cov=self.line_susceptance_stddev * jnp.eye(self.n_line),
        )

        # Combine the nominal and failure-mode parameters
        conductances = jnp.where(
            line_failures, failure_conductances, nominal_conductances
        )
        susceptances = jnp.where(
            line_failures, failure_susceptances, nominal_susceptances
        )

        # Clip conductances to be positive
        conductances = jnp.clip(conductances, a_min=0.0)

        # Return a new network (shunt admittances are unchanged from nominal)
        return Network(
            conductances,
            susceptances,
        )
