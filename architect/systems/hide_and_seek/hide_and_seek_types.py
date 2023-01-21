import jax.numpy as jnp
from jax.nn import log_sigmoid
from beartype.typing import NamedTuple, List
from jaxtyping import (
    Array,
    Float,
    # jaxtyped,
)


# @jaxtyped
# @beartype  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class Trajectory2D(NamedTuple):
    """
    The trajectory for a single robot (given by a set of timed waypoints)

    args:
        t: monotonically increasing timestamps for each waypoint
        p: the waypoints to track
    """

    t: Float[Array, " T"]
    p: Float[Array, "T 2"]

    def __call__(self, t: Float[Array, ""]) -> Float[Array, "2"]:
        """Return the waypoint at a given time (linearly interpolate)"""
        # Interpolate x and y separately
        p_t = jnp.zeros((2,))
        p_t = p_t.at[0].set(jnp.interp(t, self.t, self.x[:, 0]))
        p_t = p_t.at[1].set(jnp.interp(t, self.t, self.x[:, 1]))
        return p_t


# @jaxtyped
# @beartype  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class MultiAgentTrajectory(NamedTuple):
    """
    The trajectory for a swarm of robots.

    args:
        trajectories: the list of trajectories for each robot.
    """

    trajectories: List[Trajectory2D]

    def __call__(self, t: Float[Array, ""]) -> List[Float[Array, "2"]]:
        """Return the waypoints for each agent at a given time (linearly interpolate)"""
        return [traj(t) for traj in self.trajectories]


# @jaxtyped
# @beartype  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class Arena(NamedTuple):
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
        x_min, x_max = -self.width / 2.0, self.width
        y_min, y_max = -self.height / 2.0, self.height

        logprob_x = log_smooth_uniform(traj.p[:, 0], x_min, x_max).sum()
        logprob_y = log_smooth_uniform(traj.p[:, 1], y_min, y_max).sum()

        return logprob_x + logprob_y

    @jaxtyped
    @beartype
    def sample_random_trajectory(
        self, key: PRNGKeyArray, T: int = 10, max_t: float = 10.0
    ) -> Trajectory2D:
        """
        Sample a random trajectory from the uniform distribution within the arena

        args:
            key: PRNG key to use for sampling
            T: number of steps to include in the trajectory
            max_t: timestamp of the last waypoint
        """
        # The arena coordinates are centered at (0, 0)
        x_min, x_max = -self.width / 2.0, self.width
        y_min, y_max = -self.height / 2.0, self.height

        x_key, y_key = jrandom.split(key)
        x = jrandom.uniform(x_key, shape=(T, 1), minval=x_min, maxval=x_max)
        y = jrandom.uniform(y_key, shape=(T, 1), minval=y_min, maxval=y_max)
        p = jnp.hstack((x, y))
        t = jnp.linspace(0.0, max_t, T)

        return Trajectory2D(t, p)

    @jaxtyped
    @beartype
    def multi_trajectory_prior_logprob(
        self, multi_traj: MultiAgentTrajectory
    ) -> Float[Array, ""]:
        """
        Compute the prior log probability of the given multi-agent trajectory.
        """
        return sum([self.trajectory_prior_logprob(traj) for traj in multi_traj])

    @jaxtyped
    @beartype
    def sample_random_multi_trajectory(
        self, key: PRNGKeyArray, n: int, T: int = 10, max_t: float = 10.0
    ) -> Trajectory2D:
        """
        Sample a random multi-agent trajectory

        args:
            key: PRNG key to use for sampling
            n: number of agents
            T: number of steps to include in the trajectory
            max_t: timestamp of the last waypoint
        """
        keys = jrandom.split(key, n)
        return MultiAgentTrajectory(
            [self.sample_random_trajectory(k, T=T, max_t=max_t) for k in keys]
        )
