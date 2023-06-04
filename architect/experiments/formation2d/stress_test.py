import argparse
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from tqdm import tqdm

from architect.systems.formation2d.simulator import WindField, simulate
from architect.systems.hide_and_seek.hide_and_seek_types import (
    Arena,
    MultiAgentTrajectory,
    Trajectory2D,
)

if __name__ == "__main__":
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_prefix", type=str)
    parser.add_argument("--N", type=int, nargs="?", default=10_000)
    parser.add_argument("--batches", type=int, nargs="?", default=10)
    args = parser.parse_args()

    # Hyperparameters
    file_prefix = args.file_prefix
    N = args.N
    batches = args.batches

    prng_key = jrandom.PRNGKey(0)

    # Test the quality of the solution (and the predicted failure modes)
    # by sampling a whole heckin' bunch of exogenous parameters

    # First load the design and identified failure modes
    dp_filename = file_prefix + ".json"
    with open(dp_filename, "r") as f:
        saved_data = json.load(f)

    ep_filename = file_prefix + "_final_eps.eqx"
    n_chains = 5
    dummy = jax.vmap(lambda _: WindField(prng_key))(jnp.arange(n_chains))
    wind = eqx.tree_deserialise_leaves(ep_filename, dummy)

    # Create the game
    width = saved_data["width"]
    height = saved_data["height"]
    R = saved_data["R"]
    n = saved_data["n"]
    duration = saved_data["duration"]
    T = saved_data["T"]
    max_wind_thrust = saved_data["max_wind_thrust"]
    arena = Arena(width, height, R)
    initial_states = jnp.stack(
        (
            jnp.zeros(n) - width / 2.0 + R,
            jnp.linspace(-height / 2.0 + R, height / 2.0 - R, n),
            jnp.zeros(n),
            jnp.zeros(n),
        )
    ).T
    goal_com_position = jnp.array([width / 2.0 - R, 0.0])

    # Wrap the simulator function
    simulate_fn = lambda dp, ep: simulate(
        dp,
        initial_states,
        ep,
        goal_com_position,
        max_wind_thrust=max_wind_thrust,
        duration=duration,
        dt=0.05,
        communication_range=R,
    )

    traj_data = jnp.array(saved_data["trajs"]["trajectories"])
    final_dps = MultiAgentTrajectory([Trajectory2D(traj) for traj in traj_data])

    # See how the design performs on the identified failure modes
    result = jax.vmap(simulate_fn, in_axes=(None, 0))(final_dps, wind)
    predicted_worst_case = result.potential.max()

    # Save the results to a file
    save_filename = file_prefix + "_predicted_failures.npz"
    with open(save_filename, "wb") as f:
        jnp.save(f, jnp.array(result.potential).reshape(-1))

    # See how the chosen dispatch performs against a BUNCH of test cases
    stress_test_potentials = []
    stress_test_worst_case = []
    n_gt_predicted = []
    sim_fn = jax.jit(jax.vmap(simulate_fn, in_axes=(None, 0)))
    print("Running stress test")
    for i in tqdm(range(batches)):
        prng_key, stress_test_key = jrandom.split(prng_key)
        stress_test_keys = jrandom.split(stress_test_key, N)
        stress_test_eps = jax.vmap(WindField)(stress_test_keys)
        stress_test_result = sim_fn(final_dps, stress_test_eps)
        stress_test_worst_case.append(stress_test_result.potential.max())
        n_gt_predicted.append(
            (stress_test_result.potential > predicted_worst_case).sum()
        )
        stress_test_potentials.append(stress_test_result.potential)

    print("")
    print(file_prefix)
    print(f"Testes on {N * batches} random eps")
    print(f"Predicted worst case potential: {predicted_worst_case}")
    print(f"Worst case identified by stress test: {max(stress_test_worst_case)}")
    print(f"{100 * sum(n_gt_predicted) / (N * batches)}% are worse than predicted")

    # Save the results to a file
    save_filename = file_prefix + "_stress_test.npz"
    with open(save_filename, "wb") as f:
        jnp.save(f, jnp.array(stress_test_potentials).reshape(-1))
