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
        non_ego_states: the states of all other vehicles
    """

    ego_state: Float[Array, " n_states"]
    non_ego_states: Float[Array, "n_non_ego n_states"]


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
        max_render_dist: the maximum distance to render in the depth image.
        render_sharpness: the sharpness of the scene.
    """

    _highway_scene: HighwayScene
    _camera_intrinsics: CameraIntrinsics
    _dt: float
    _max_render_dist: float
    _render_sharpness: float

    _axle_length: float = 1.0

    def __init__(
        self,
        highway_scene: HighwayScene,
        camera_intrinsics: CameraIntrinsics,
        dt: float,
        max_render_dist: float = 30.0,
        render_sharpness: float = 100.0,
    ):
        """Initialize the environment."""
        self._highway_scene = highway_scene
        self._camera_intrinsics = camera_intrinsics
        self._dt = dt
        self._max_render_dist = max_render_dist
        self._render_sharpness = render_sharpness

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
        non_ego_actions: Float[Array, "n_non_ego n_actions"],
    ) -> Tuple[HighwayState, HighwayObs, Float[Array, ""], bool]:
        """Take a step in the environment.

        Args:
            state: the current state of the environment
            ego_action: the control action to take for the ego vehicle (acceleration and
                steering angle)
            non_ego_actions: the control action to take for all other vehicles
                (acceleration and steering angle)

        Returns:
            The next state of the environment, the observations, the reward, and a
            boolean indicating whether the episode is done. Episodes end when the ego
            car crashes into any other object in the scene.
        """
        # Unpack the state
        ego_state, non_ego_states = state

        # Compute the next state of the ego and other vehicles
        next_ego_state = self.car_dynamics(ego_state, ego_action)
        next_non_ego_states = jax.vmap(self.car_dynamics)(
            non_ego_states, non_ego_actions
        )
        next_state = HighwayState(next_ego_state, next_non_ego_states)

        # Compute the reward
        # reward = self.reward(next_state)  # TODO@dawsonc
        reward = jnp.array(0.0)

        # Compute the observations from a camera placed on the ego vehicle
        ego_x, ego_y, ego_theta, ego_v = next_ego_state
        camera_origin = jnp.array([ego_x, ego_y, self._highway_scene.car.height / 2])
        ego_heading_vector = jnp.array([jnp.cos(ego_theta), jnp.sin(ego_theta), 0])
        extrinsics = CameraExtrinsics(
            camera_origin=camera_origin,
            camera_R_to_world=look_at(
                camera_origin=camera_origin,
                target=camera_origin + ego_heading_vector,
                up=jnp.array([0, 0, 1.0]),
            ),
        )
        depth_image = self._highway_scene.render_depth(
            self._camera_intrinsics,
            extrinsics,
            next_non_ego_states[:, :3],  # trim out speed; not needed for rendering
            max_dist=self._max_render_dist,
            sharpness=self._render_sharpness,
        )
        obs = HighwayObs(ego_v, depth_image)

        # TODO@dawsonc
        # crashed = self._highway_scene.check_for_collision(
        #     next_ego_state[:3],  # trim out speed; not needed for collision checking
        #     next_non_ego_states[:, :3],
        # )
        # done = crashed
        done = False

        return next_state, obs, reward, done
