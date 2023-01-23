import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from beartype import beartype
from beartype.typing import List, NamedTuple
from jax.nn import log_sigmoid
from jaxtyping import Array, Float, jaxtyped
from scipy.special import binom

from architect.types import PRNGKeyArray


# @jaxtyped
# @beartype
class Trajectory2D(NamedTuple):
    """
    The trajectory for a single robot, represented by a fixed-degree Bezier curve.

    Time is normalized to [0, 1]

    args:
        p: the array of control points for the Bezier curve
    """

    p: Float[Array, "T 2"]

    @property
    def n(self):
        """Return the degree of this Bezier curve"""
        return self.p.shape[0] - 1

    @property
    def i(self):
        """Return the degree indices"""
        return np.arange(self.n + 1)

    @property
    def coefficients(self):
        """Return the degree indices"""
        return binom(self.n, self.i)

    @jax.jit
    @jaxtyped
    @beartype
    def __call__(self, t: Float[Array, ""]) -> Float[Array, "2"]:
        """Return the point along the trajectory at the given time"""
        # Bezier curves have an explicit form
        # see https://en.wikipedia.org/wiki/B%C3%A9zier_curve
        return jnp.sum(
            self.coefficients * (1 - t) ** (self.n - self.i) * t ** self.i * self.p.T,
            axis=-1,
        )


# @jaxtyped
# @beartype
class MultiAgentTrajectory(NamedTuple):
    """
    The trajectory for a swarm of robots.

    args:
        trajectories: the list of trajectories for each robot.
    """

    trajectories: List[Trajectory2D]

    @jaxtyped
    def __call__(self, t: Float[Array, ""]) -> Float[Array, "N 2"]:
        """Return the waypoints for each agent at a given time (linearly interpolate)"""
        return jnp.array([traj(t) for traj in self.trajectories])


@jaxtyped
@beartype
class Arena(eqx.Module):
    """
    The arena in which hide and seek takes place

    args:
        width: width of the arena (m)
        height: height of the arena (m)
        buffer: how far robots should keep from the walls
        smoothing: higher values -> sharper approximation of a uniform distribution
    """

    width: float
    height: float
    buffer: float

    smoothing: float = 20.0

    @jaxtyped
    @beartype
    def trajectory_prior_logprob(self, traj: Trajectory2D) -> Float[Array, ""]:
        """
        Compute the prior log probability of the given trajectory.

        Probability is not necessarily normalized.

        We assume that the prior distribution for the trajectory is uniform within the
        boundaries of the arena.

        We use a smooth approximation of the uniform distribution.
        """

        def log_smooth_uniform(x, x_min, x_max):
            return log_sigmoid(self.smoothing * (x - x_min)) + log_sigmoid(
                self.smoothing * (x_max - x)
            )

        # The arena coordinates are centered at (0, 0)
        x_min, x_max = -self.width / 2.0 + self.buffer, self.width / 2.0 - self.buffer
        y_min, y_max = -self.height / 2.0 + self.buffer, self.height / 2.0 - self.buffer

        logprob_x = log_smooth_uniform(traj.p[:, 0], x_min, x_max).sum()
        logprob_y = log_smooth_uniform(traj.p[:, 1], y_min, y_max).sum()

        return logprob_x + logprob_y

    @jaxtyped
    @beartype
    def sample_random_trajectory(
        self,
        key: PRNGKeyArray,
        start_p: Float[Array, " 2"],
        T: int = 4,
    ) -> Trajectory2D:
        """
        Sample a random trajectory from the uniform distribution within the arena

        args:
            key: PRNG key to use for sampling
            start_p: the starting position for the trajectory
            T: number of steps to include in the trajectory
        """
        # The arena coordinates are centered at (0, 0)
        x_min, x_max = -self.width / 2.0 + self.buffer, self.width / 2.0 - self.buffer
        y_min, y_max = -self.height / 2.0 + self.buffer, self.height / 2.0 - self.buffer

        # Sample all points except the start point randomly
        x_key, y_key = jrandom.split(key)
        x = jrandom.uniform(x_key, shape=(T - 1, 1), minval=x_min, maxval=x_max)
        y = jrandom.uniform(y_key, shape=(T - 1, 1), minval=y_min, maxval=y_max)
        p = jnp.hstack((x, y))

        # Prepend the start point
        p = jnp.vstack((start_p.reshape(1, 2), p))

        return Trajectory2D(p)

    @jaxtyped
    @beartype
    def multi_trajectory_prior_logprob(
        self, multi_traj: MultiAgentTrajectory
    ) -> Float[Array, ""]:
        """
        Compute the prior log probability of the given multi-agent trajectory.
        """
        return sum(
            [self.trajectory_prior_logprob(traj) for traj in multi_traj.trajectories]
        )

    @jaxtyped
    @beartype
    def sample_random_multi_trajectory(
        self, key: PRNGKeyArray, start_ps: Float[Array, "n 2"], T: int = 4
    ) -> MultiAgentTrajectory:
        """
        Sample a random multi-agent trajectory

        args:
            key: PRNG key to use for sampling
            start_ps: n_agents x 2 array of starting positions
            T: number of steps to include in the trajectory
        """
        n = start_ps.shape[0]
        keys = jrandom.split(key, n)
        return MultiAgentTrajectory(
            [
                self.sample_random_trajectory(k, start_p, T=T)
                for k, start_p in zip(keys, start_ps)
            ]
        )


class HideAndSeekResult(NamedTuple):
    """
    Result of playing a game of hide and seek

    Includes:
        The trajectories used by seekers and hiders
        The duration of the game
        The positions of all seekers and hiders throughout the game
        The potential value (i.e. cost for seekers)
    """

    seeker_trajectory: MultiAgentTrajectory
    hider_trajectory: MultiAgentTrajectory

    game_duration: float

    seeker_positions: Float[Array, "T 2"]
    hider_positions: Float[Array, "T 2"]

    potential: Float[Array, ""]
