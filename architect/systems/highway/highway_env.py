"""Manage the state of the ego car and all other vehicles on the highway."""
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jaxtyping import Float, Array, jaxtyped

from architect.systems.components.sensing.vision.render import (
    CameraIntrinsics,
    CameraExtrinsics,
)
from architect.systems.components.sensing.vision.util import look_at
from architect.systems.highway.highway_scene import HighwayScene


@beartype
class HighwayObs(NamedTuple):
    """Observations returned from the highway environment.

    Attributes:
        speed: the forward speed of the ego vehicle
        depth_image: a depth image rendered from the front of the ego vehicle
    """

    speed: Float[Array, ""]
    depth_image: Float[Array, "res_x res_y"]


@beartype
class HighwayState(NamedTuple):
    """The state of the ego vehicle and all other vehicles on the highway.

    Attributes:
        ego_state: the state of the ego vehicle
        other_states: the states of all other vehicles
    """

    ego_state: Float[Array, " n_states"]
    other_states: Float[Array, "n_other n_states"]

    @property
    def car_states(self) -> Float[Array, "n_cars n_states"]:
        """Return the states of all cars in the scene."""
        return jnp.concatenate([self.ego_state[None], self.other_states], axis=0)


class HighwayEnv:
    """
    The highway includes one ego vehicle and some number of other vehicles. Each vehicle
    has state [x, y, theta, v] (position, heading, forward speed) and simple dynamics
    [1] with control inputs [a, delta] (acceleration, steering angle).

    The scene yields observations of the current vehicle's forward speed and the
    depth image rendered from the front of the ego vehicle.

    References:
        [1] https://msl.cs.uiuc.edu/planning/node658.html

    Args:
        highway_scene: a representation of the underlying highway scene.
        camera_intrinsics: the intrinsics of the camera mounted to the ego agent.
        dt: the time step to use for simulation.
    """

    _highway_scene: HighwayScene
    _camera_intrinsics: CameraIntrinsics
    _dt: float
    _axle_length: float = 1.0

    def __init__(
        self,
        highway_scene: HighwayScene,
        camera_intrinsics: CameraIntrinsics,
        dt: float,
    ):
        """Initialize the environment."""
        self._highway_scene = highway_scene
        self._camera_intrinsics = camera_intrinsics
        self._dt = dt

    @jaxtyped
    @beartype
    def car_dynamics(
        self,
        state: Float[Array, " n_states"],
        action: Float[Array, " n_actions"],
    ) -> Float[Array, " n_states"]:
        """Compute the dynamics of the car.

        Args:
            state: the current state of the car [x, y, theta, v]
            action: the control action to take [acceleration, steering angle]

        Returns:
            The next state of the car
        """
        # Unpack the state and action
        x, y, theta, v = state
        a, delta = action

        # Clip the steering angle
        delta = jnp.clip(delta, -0.4 * jnp.pi, 0.4 * jnp.pi)

        # Compute the next state
        x_next = x + v * jnp.cos(theta) * self._dt
        y_next = y + v * jnp.sin(theta) * self._dt
        theta_next = theta + v * jnp.tan(delta) / self._axle_length * self._dt
        v_next = v + a * self._dt

        return jnp.array([x_next, y_next, theta_next, v_next])

    @jaxtyped
    @beartype
    def step(
        self,
        state: HighwayState,
        ego_action: Float[Array, " n_actions"],
        other_actions: Float[Array, "n_other n_actions"],
    ) -> Tuple[HighwayState, HighwayObs, Float[Array, ""]]:
        """Take a step in the environment.

        Args:
            state: the current state of the environment
            ego_action: the control action to take for the ego vehicle (acceleration and
                steering angle)
            other_actions: the control action to take for all other vehicles
                (acceleration and steering angle)

        Returns:
            The next state of the environment, the observations, and the reward
        """
        # Unpack the state
        ego_state, other_states = state

        # Compute the next state of the ego and other vehicles
        next_ego_state = self.car_dynamics(ego_state, ego_action)
        next_other_states = jax.vmap(self.car_dynamics)(other_states, other_actions)
        next_state = HighwayState(next_ego_state, next_other_states)

        # Compute the reward
        reward = self.reward(next_state)

        # Compute the observations from a camera placed on the ego vehicle
        ego_x, ego_y, ego_theta = next_ego_state
        camera_origin = jnp.array([ego_x, ego_y, self._highway_scene.car.height / 2])
        ego_heading_vector = jnp.array([jnp.cos(ego_theta), jnp.sin(ego_theta), 0])
        extrinsics = CameraExtrinsics(
            camera_origin=camera_origin,
            camera_R_to_world=look_at(
                camera_origin=camera_origin,
                target=camera_origin + ego_heading_vector,
                up=jnp.array([0, 0, 1.0]),
            )
        )
        obs = self._highway_scene.render_depth(
            self._camera_intrinsics, extrinsics, next_state.car_states
        )

        return next_state, obs, reward
