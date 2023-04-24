import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from architect.systems.hide_and_seek.hide_and_seek import Game
from architect.systems.hide_and_seek.hide_and_seek_types import (
    MultiAgentTrajectory, Trajectory2D)


def test_Game(plot=False):
    # Create the game
    game = Game(
        jnp.array([[0.0, 0.0], [0.0, 1.0]]),
        jnp.array([[1.0, 0.5], [1.0, 1.5]]),
        duration=100.0,
        dt=0.1,
        sensing_range=0.5,
        seeker_max_speed=0.5,
        hider_max_speed=0.5,
        b=5.0,
    )

    # Define test trajectories
    p1 = jnp.array([[0.0, 0.0], [0.9, 0.5], [1.0, 0.0]])
    traj1 = Trajectory2D(p1)
    p2 = jnp.array([[0.0, 1.0], [0.9, 0.5], [1.0, 1.0]])
    traj2 = Trajectory2D(p2)
    seeker_traj = MultiAgentTrajectory([traj1, traj2])

    p1 = jnp.array([[1.0, 0.5], [0.0, 0.5]])
    traj1 = Trajectory2D(p1)
    p2 = jnp.array([[1.0, 1.5], [0.0, 1.5]])
    traj2 = Trajectory2D(p2)
    hider_traj = MultiAgentTrajectory([traj1, traj2])

    # Evaluate the game
    result = game(seeker_traj, hider_traj)

    # Plot if requested
    if plot:
        # Plot initial setup
        plt.scatter(
            game.initial_seeker_positions[:, 0],
            game.initial_seeker_positions[:, 1],
            color="b",
            marker="x",
            label="Initial seeker positions",
        )
        plt.scatter(
            game.initial_hider_positions[:, 0],
            game.initial_hider_positions[:, 1],
            color="r",
            marker="x",
            label="Initial hider positions",
        )

        # Plot planned trajectories
        t = jnp.linspace(0, 1, 100)
        for traj in seeker_traj.trajectories:
            pts = jax.vmap(traj)(t)
            plt.plot(pts[:, 0], pts[:, 1], "b:")

        for traj in hider_traj.trajectories:
            pts = jax.vmap(traj)(t)
            plt.plot(pts[:, 0], pts[:, 1], "r:")

        # Plot resulting trajectories
        for i in range(game.initial_seeker_positions.shape[0]):
            plt.plot(
                result.seeker_positions[:, i, 0], result.seeker_positions[:, i, 1], "b-"
            )

        for i in range(game.initial_hider_positions.shape[0]):
            plt.plot(
                result.hider_positions[:, i, 0], result.hider_positions[:, i, 1], "r-"
            )

        plt.legend()
        plt.show()


if __name__ == "__main__":
    test_Game(plot=True)
