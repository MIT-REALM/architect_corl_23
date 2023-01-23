import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn import logsumexp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from architect.systems.hide_and_seek.hide_and_seek_types import (
    MultiAgentTrajectory,
    HideAndSeekResult,
)


@jaxtyped
@beartype
class Game(eqx.Module):
    """
    The hide and seek game engine

    args:
        initial_seeker_positions: initial positions of seekers
        initial_hider_positions: initial positions of hiders
        duration: the length of the game (seconds)
        dt: the timestep of the game's simulation (seconds)
        sensing_range: the sensing range of each seeker (meters)
        seeker_max_speed: the max speed of the seekers (m/s)
        hider_max_speed: the max speed of the hiders (m/s)
        b: parameter used to smooth min/max
    """

    initial_seeker_positions: Float[Array, "n_seeker 2"]
    initial_hider_positions: Float[Array, "n_hider 2"]

    duration: float = 100.0
    dt: float = 0.1

    sensing_range: float = 0.5

    seeker_max_speed: float = 0.5
    hider_max_speed: float = 0.5

    b: float = 5.0

    @jaxtyped
    @beartype
    def __call__(
        self,
        seeker_trajectory: MultiAgentTrajectory,
        hider_trajectory: MultiAgentTrajectory,
    ) -> HideAndSeekResult:
        """
        Play out a game of hide and seek.

        args:
            seeker_trajectory: trajectories for the seekers to follow
            hider_trajectory: trajectories for the hiders to follow
        """
        # Define a function to execute one step in the game
        def step(carry, t):
            # Unpack carry
            current_seeker_positions, current_hider_positions = carry

            # Get current target positions for hiders and seekers
            seeker_target = seeker_trajectory(t / self.duration)
            hider_target = hider_trajectory(t / self.duration)

            # Steer towards these targets (clip with max speeds)
            v_seeker = seeker_target - current_seeker_positions
            seeker_speed = jnp.linalg.norm(v_seeker)
            v_seeker = jnp.where(
                seeker_speed >= self.seeker_max_speed,
                v_seeker * self.seeker_max_speed / seeker_speed,
                v_seeker,
            )
            v_hider = hider_target - current_hider_positions
            hider_speed = jnp.linalg.norm(v_hider)
            v_hider = jnp.where(
                hider_speed >= self.hider_max_speed,
                v_hider * self.hider_max_speed / hider_speed,
                v_hider,
            )

            seeker_positions = current_seeker_positions + self.dt * v_seeker
            hider_positions = current_hider_positions + self.dt * v_hider

            # Return an updated carry and output
            output = (seeker_positions, hider_positions)
            return output, output

        # Execute the game simulation to get a trace of hider and seeker positions
        _, (seeker_positions, hider_positions) = jax.lax.scan(
            step,
            (self.initial_seeker_positions, self.initial_hider_positions),
            jnp.arange(0, self.duration, self.dt),
        )

        # Compute the cost as the (smoothed) minimum squared distance between each hider
        # and all seekers
        softmin = lambda x: -1 / self.b * logsumexp(-self.b * x, axis=-1)
        distance = lambda x, y: jnp.linalg.norm(x - y, axis=-1) ** 2
        distance_to_all_seekers_fn = lambda p: jax.vmap(distance, in_axes=(1, None))(
            seeker_positions, p
        )
        distance_to_all_seekers = jax.vmap(distance_to_all_seekers_fn, in_axes=(1,))(
            hider_positions
        )
        min_distance_to_any_seeker = softmin(softmin(distance_to_all_seekers))
        cost = min_distance_to_any_seeker.sum() - self.sensing_range

        return HideAndSeekResult(
            seeker_trajectory=seeker_trajectory,
            hider_trajectory=hider_trajectory,
            game_duration=self.duration,
            seeker_positions=seeker_positions,
            hider_positions=hider_positions,
            potential=cost,
            hiding_margin=distance_to_all_seekers,
        )
