"""Test the highway simulation"""
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt

from architect.experiments.highway.predict_and_mitigate import (
    sample_non_ego_actions,
    simulate,
)
from architect.experiments.highway.train_highway_agent import make_highway_env
from architect.systems.highway.driving_policy import DrivingPolicy

if __name__ == "__main__":
    # Make a PRNG key (#sorandom)
    prng_key = jrandom.PRNGKey(0)

    # Make the environment to use
    image_shape = (32, 32)
    env = make_highway_env(image_shape)
    env._collision_penalty = 10.0

    # Load the driving policy
    dummy_policy = DrivingPolicy(jrandom.PRNGKey(0), image_shape)
    policy = eqx.tree_deserialise_leaves(
        "results/highway/initial_policy.eqx", dummy_policy
    )
    dp, static_policy = eqx.partition(policy, eqx.is_array)

    # Initialize some fixed initial states
    # Hack: reset covariances to remove variation in initial state
    env._initial_state_covariance = 1e-3 * env._initial_state_covariance
    env._shading_light_direction_covariance = (
        1e-3 * env._shading_light_direction_covariance
    )
    prng_key, initial_state_key = jrandom.split(prng_key)
    initial_state = env.reset(initial_state_key)

    # Initialize some random actions for the non-ego vehicles
    prng_key, ep_key = jrandom.split(prng_key)
    noise_scale = 0.5
    T = 60
    initial_eps = sample_non_ego_actions(ep_key, env, T, 2, noise_scale)

    # Get the norm of the gradient of actions wrt actions for different simulation
    # lengths
    potential_fn = lambda actions, t: simulate(
        env, policy, initial_state, actions, static_policy, t
    ).potential
    ts = [1, 2, 4, 8, 16, 32, 60]
    grad_norms = []
    for t in ts:
        actions = initial_eps[:t]
        grad = jax.grad(potential_fn)(actions, t)
        grad_norms.append(jnp.linalg.norm(grad))
        print(f"{t}: {grad_norms[-1]}")

    print(ts)
    print(grad_norms)

    plt.plot(ts, grad_norms)
