"""Implement an oracle for the drone landing problem."""
import functools

import jax
import jax.nn
import jax.numpy as jnp
import jax.random as jrandom
import optax
from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jaxtyping import Array, Bool, Float, jaxtyped

from architect.systems.components.sensing.vision.render import (
    CameraExtrinsics,
    CameraIntrinsics,
)
from architect.systems.drone_landing.env import DroneLandingEnv, DroneState
from architect.types import PRNGKeyArray


@functools.partial(jax.jit, static_argnums=(2,))
def rollout(
    initial_state: DroneState,
    action_sequence: Float[Array, "T 2"],
    env: DroneLandingEnv,
) -> Float[Array, ""]:
    """
    Rollout the environment and get the cost of the rollout.
    
    Args:
        x_init: the initial state of the drone.
        action_sequence: the sequence of actions to take.
        env: the environment.

    Returns:
        The cost of the rollout.
    """
    # Create a function to take one step with the policy
    def step(state, action):
        # Update the state using the dynamics
        new_drone_state = env.drone_dynamics(state.drone_state, action)
        new_state = DroneState(new_drone_state, state.tree_locations, state.wind_speed)

        # Get the running cost (squared distance to goal)
        target = jnp.array([0.0, 0.0, 0.5, 0.0])
        cost = jnp.sum((new_drone_state - target) ** 2) * env._dt

        # Add the collision cost
        min_distance_to_obstacle = env._scene.check_for_collision(
            new_drone_state[:3],  # trim out yaw; not needed for collision checking
            new_state.wind_speed,
            new_state.tree_locations,
            50,
        )
        cost += 1e2 * jax.nn.sigmoid(
            -25 * min_distance_to_obstacle
        )

        return new_state, cost

    # Run the rollout
    _, costs = jax.lax.scan(step, initial_state, action_sequence)
    
    # Return the total cost
    return jnp.mean(costs)


@functools.partial(jax.jit, static_argnums=(1, 2,))
def oracle_policy(
    initial_state: DroneState,
    horizon: int,
    env: DroneLandingEnv,
    key: PRNGKeyArray,
) -> Float[Array, "horizon 2"]:
    """
    A policy that uses total information about the environment to compute the optimal
    action sequence.

    Args:
        initial_state: the initial state of the environment.
        horizon: the number of steps to take in the rollout.
        env: the environment.
        key: a random number generator key.
    """
    # Define the loss and grad function
    loss_and_grad = jax.value_and_grad(rollout, argnums=1)

    # Initialize a random action sequence
    action_sequence = jrandom.normal(key, (horizon, 2))

    # Define an optimizer
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(action_sequence)
