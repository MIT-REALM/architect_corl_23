"""Define a simulator for the F16 system"""
import math

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple
from jax_f16.f16 import F16
from jaxtyping import Array, Float, jaxtyped
from tqdm import tqdm

from architect.types import PRNGKeyArray
from architect.utils import softmin

# Constants
FLIGHT_DECK = 1000.0  # ft


class F16Result(NamedTuple):
    """
    The result of an F16 simulation

    args:
        states: the state of the aircraft over time
        potential: the potential/cost assigned to this rollout
    """

    states: Float[Array, "T nx"]
    potential: Float[Array, ""]


def are_wings_level(x_f16):
    "are the wings level?"
    phi = x_f16[F16.PHI]
    radsFromWingsLevel = round(phi / (2 * math.pi))
    return abs(phi - (2 * math.pi) * radsFromWingsLevel) < 0.087


def is_roll_rate_low(x_f16):
    "is the roll rate low enough to switch to pull?"
    p = x_f16[F16.P]
    return abs(p) < 0.17


def is_above_flight_deck(x_f16):
    "is the aircraft above the flight deck?"
    alt = x_f16[F16.H]
    return alt >= FLIGHT_DECK


def is_nose_high_enough(x_f16):
    "is the nose high enough?"

    theta = x_f16[F16.THETA]
    alpha = x_f16[F16.ALPHA]

    # Determine which angle is "level" (0, 360, 720, etc)
    radsFromNoseLevel = round((theta - alpha) / (2 * math.pi))

    # Evaluate boolean
    return (theta - alpha) - 2 * math.pi * radsFromNoseLevel > 0.0


def roll_wings_level(x_f16):
    "get commands to roll level."

    phi = x_f16[F16.PHI]
    p = x_f16[F16.P]

    rv = jnp.array(F16.trim_control())

    # Determine which angle is "level" (0, 360, 720, etc)
    radsFromWingsLevel = round(phi / (2 * math.pi))

    # PD Control until phi == pi * radsFromWingsLevel
    ps = -(phi - (2 * math.pi) * radsFromWingsLevel) * 4.0 - p * 2.0

    # Build commands to roll wings level
    rv = rv.at[1].set(ps)

    return rv


def pull_nose_level(self):
    "get commands to pull up"
    rv = jnp.array(F16.trim_control())
    rv = rv.at[0].set(5.0)
    return rv


def gcas_controller(x: Float[Array, " nx"]) -> Float[Array, " nu"]:
    """Implement a version of the GCAS controller from the original F16 implementation.

    github.com/stanleybak/AeroBenchVVPython
    """
    going_to_crash = jnp.logical_and(
        jnp.logical_not(is_nose_high_enough(x)),
        jnp.logical_not(is_above_flight_deck(x)),
    )

    need_to_level_wings = jnp.logical_not(
        jnp.logical_and(is_roll_rate_low(x), are_wings_level(x))
    )

    do_nothing = jnp.array(F16.trim_control())
    pull_up_command = pull_nose_level(x)
    roll_level_command = roll_wings_level(x)

    # If we are going to crash, we need to do something.
    # Roll level if not level, then pull up
    avoid_crash = roll_level_command
    avoid_crash += jnp.where(
        need_to_level_wings, jnp.zeros_like(pull_up_command), pull_up_command
    )

    # If we are going to crash, don't; otherwise, do nothing
    command = jnp.where(going_to_crash, avoid_crash, do_nothing)

    return command


class ResidualControl(eqx.Module):
    """Define an MLP for the residual control input"""

    layers: list

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        n_in, n_hidden, n_out = F16.NX, 32, F16.NU
        self.layers = [
            eqx.nn.Linear(n_in, n_hidden, key=key1),
            eqx.nn.Linear(n_hidden, n_hidden, key=key2),
            eqx.nn.Linear(n_hidden, n_hidden, key=key2),
            eqx.nn.Linear(n_hidden, n_out, key=key3),
        ]

    def __call__(self, x, max_input: float = 1.0):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))

        u = self.layers[-1](x)

        # Saturate the throttle from 0 to 1
        u = u.at[4].set(jax.nn.sigmoid(u[4]))

        # Saturate other controls from -10 to 10
        u = u.at[:4].set(10 * jax.nn.tanh(u[:4]))

        return u


def closed_loop_dynamics(
    x: Float[Array, " nx"],
    residual_controller: ResidualControl,
) -> Float[Array, " nx"]:
    """
    The closed loop dynamics with GCAS and residual controller.

    args:
        x: the state of the aircraft
        residual_controller: the residual controller

    returns:
        the derivative of the state
    """
    u_gcas = gcas_controller(x)
    u_residual = residual_controller(x)
    u = 0.5 * u_gcas + 0.5 * u_residual
    # u = u_residual

    return F16().xdot(x, u)


def sample_initial_states(key: PRNGKeyArray) -> Float[Array, " nx"]:
    """
    Sample initial states for the simulation.

    Equivalent to case 3S from Heidlauf et al.
    """
    x = jnp.array(F16.trim_state())

    # Sample a random initial altitude
    x = x.at[F16.H].set(3650.0 + jax.random.normal(key) * 100.0)

    # Sample random initial roll
    x = x.at[F16.PHI].set(jax.random.normal(key) * jnp.pi / 8)

    # Sample random initial pitch
    x = x.at[F16.THETA].set(-1 * jnp.pi / 5 + jax.random.normal(key) * jnp.pi / 5)

    return x


def initial_state_logprior(x: Float[Array, " nx"]) -> Float[Array, ""]:
    """Compute the log prior probability for the given initial state."""

    # Sample a random initial altitude
    logprior = jax.scipy.stats.norm.logpdf(x[F16.H], 3650.0, 100.0)

    # Sample random initial roll
    logprior += jax.scipy.stats.norm.logpdf(x[F16.PHI], 0.0, jnp.pi / 8)

    # Sample random initial pitch
    logprior += jax.scipy.stats.norm.logpdf(x[F16.THETA], -1 * jnp.pi / 5, jnp.pi / 5)

    return logprior


@jaxtyped
@beartype
def simulate(
    initial_state: Float[Array, " nx"],
    residual_controller: ResidualControl,
    dt: float = 0.01,
    duration: float = 15.0,
) -> F16Result:
    """
    Simulate an F16 with GCAS and residual control.

    args:
        initial_state: the initial state of the aircraft
        residual_controller: the residual controller

    returns:
        the result of the simulation
    """

    # Define a function to step the simulation
    def step(carry, t):
        # Unpack the carry
        x = carry["x"]

        # get the state derivative and integrate
        xdot = closed_loop_dynamics(x, residual_controller)

        # Integrate the state derivative
        x = x + xdot * dt

        # Pack the carry
        carry = {"x": x}

        # Return the carry for the next step and output states as well
        return carry, x

    # Define the initial carry and inputs
    carry = {"x": initial_state}

    # Run the loop
    _, x = jax.lax.scan(step, carry, jnp.arange(0.0, duration, dt))

    # Get the minimum altitudetempering
    alt = x[:, F16.H]
    min_alt = softmin(alt, sharpness=10.0)

    # Potential is high when min altitude is low
    potential = jax.nn.elu(200.0 - min_alt) / FLIGHT_DECK  # Normalize by flight deck
    # # Add a penalty for deviation from level at end of simulation
    # potential += x[-1, F16.PHI] ** 2 / jnp.pi**2
    # potential += x[-1, F16.THETA] ** 2 / jnp.pi**2

    return F16Result(states=x, potential=potential)


if __name__ == "__main__":
    # # Sample a large number of points
    # N = int(5e2)
    # key = jax.random.PRNGKey(0)
    # key, x_key = jax.random.split(key)
    # x_keys = jax.random.split(x_key, N)
    # x = jax.vmap(sample_initial_states)(x_keys)

    # # Simulate all of these initial states using the hand-derived GCAS controller
    # sim_fn = lambda x: simulate(x, gcas_controller).states
    # print("Generating training set...")
    # x_train = jax.vmap(sim_fn)(x)

    # # Train a neural controller to match these
    # controller = ResidualControl(key)
    # lr = 1e-3
    # iters = 500

    # def loss_fn(pi, x_train):
    #     u_gcas = jax.vmap(jax.vmap(gcas_controller))(x_train)
    #     u_pi = jax.vmap(jax.vmap(pi))(x_train)
    #     mse = jnp.mean(jnp.square(u_gcas - u_pi))
    #     return mse

    # def step(controller):
    #     grads = jax.jit(jax.grad(loss_fn))(controller, x_train)
    #     return jax.tree_util.tree_map(lambda pi, g: pi - lr * g, controller, grads)

    # print("Training...")
    # for i in tqdm(range(iters)):
    #     controller = step(controller)
    #     if i % 10 == 0:
    #         print(f"Iter {i}: {loss_fn(controller, x_train)}")

    # eqx.tree_serialise_leaves("gcas_neural.eqx", controller)

    # Test the simulation
    print("Testing...")
    key = jax.random.PRNGKey(0)

    # Load the controller
    key, subkey = jax.random.split(key)
    controller = ResidualControl(subkey)
    controller = eqx.tree_deserialise_leaves("gcas_neural.eqx", controller)

    initial_state = jnp.array(F16.trim_state())
    # Point towards the ground
    initial_state = initial_state.at[F16.THETA].set(-0.6)

    result = simulate(initial_state, controller, dt=0.01, duration=15.0)
    print(result.potential)

    # plot the results, showing altitude over time
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(6, 6))

    axes[0].plot(result.states[:, F16.H])
    axes[0].set_ylabel("Altitude (ft)")
    axes[0].set_xlabel("Time (s)")

    # Also plot the roll angle and rate
    axes[1].plot(result.states[:, F16.PHI], label="Roll Angle")
    axes[1].plot(result.states[:, F16.P], label="Roll Rate")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()

    plt.show()
