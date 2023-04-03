"""Code to predict and mitigate failure modes in the highway scenario."""
import argparse
import json
import os
import time
import shutil

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from beartype.typing import NamedTuple
from jax.nn import sigmoid
from jaxtyping import Array, Float, Shaped

from architect.engines import predict_and_mitigate_failure_modes
from architect.experiments.highway.train_highway_agent import make_highway_env
from architect.systems.highway.driving_policy import DrivingPolicy
from architect.systems.highway.highway_env import HighwayEnv, HighwayObs, HighwayState
from architect.utils import softmin


class SimulationResults(NamedTuple):
    """A class for storing the results of a simulation."""

    potential: Float[Array, ""]
    initial_obs: HighwayObs
    final_obs: HighwayObs


def simulate(
    env: HighwayEnv,
    policy: DrivingPolicy,
    initial_state: HighwayState,
    static_policy: DrivingPolicy,
    max_steps: int = 60,
) -> Float[Array, ""]:
    """Simulate the highway environment.

    Disables randomness in the policy and environment (all randomness should be
    factored out into the initial_state argument).

    If the environment terminates before `max_steps` steps, it will not be reset and
    all further reward will be zero.

    Args:
        env: The environment to simulate.
        policy: The parts of the policy that are design parameters.
        initial_state: The initial state of the environment.
        static_policy: the parts of the policy that are not design parameters.
        max_steps: The maximum number of steps to simulate.

    Returns:
        SimulationResults object
    """
    # Merge the policy back together
    policy = eqx.combine(policy, static_policy)

    @jax.checkpoint
    def step(carry, key):
        # Unpack the carry
        action, state, already_done = carry

        # PRNG key management. These don't have any effect, but they need to be passed
        # to the environment and policy.
        step_subkey, action_subkey = jrandom.split(key)

        # Fix non-ego actions
        non_ego_actions = jnp.zeros((state.non_ego_states.shape[0], 2))

        # Take a step in the environment using the action carried over from the previous
        # step.
        next_state, next_observation, reward, done = env.step(
            state, action, non_ego_actions, step_subkey
        )

        # Compute the action for the next step
        next_action, _, _ = policy(next_observation, action_subkey, deterministic=True)

        # If the environment has already terminated, set the reward to zero.
        reward = jax.lax.cond(already_done, lambda: 0.0, lambda: reward)
        already_done = jnp.logical_or(already_done, done)

        # Don't step if the environment has terminated
        next_action = jax.lax.cond(already_done, lambda: action, lambda: next_action)
        next_state = jax.lax.cond(already_done, lambda: state, lambda: next_state)

        next_carry = (next_action, next_state, already_done)
        output = reward
        return next_carry, output

    # Get the initial observation and action
    initial_obs = env.get_obs(initial_state)
    initial_action, _, _ = policy(initial_obs, jrandom.PRNGKey(0), deterministic=True)

    # Transform and rollout!
    keys = jrandom.split(jrandom.PRNGKey(0), max_steps)
    (_, final_state, _), reward = jax.lax.scan(
        step, (initial_action, initial_state, False), keys
    )

    # Get the final observation
    final_obs = env.get_obs(final_state)

    # The potential is the negative of the (soft) minimum reward observed
    potential = -softmin(reward)

    return SimulationResults(potential, initial_obs, final_obs)


if __name__ == "__main__":
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--savename", type=str, default="overtake_predict_only")
    parser.add_argument("--image_w", type=int, nargs="?", default=32)
    parser.add_argument("--image_h", type=int, nargs="?", default=32)
    parser.add_argument("--L", type=float, nargs="?", default=1.0)
    parser.add_argument("--dp_mcmc_step_size", type=float, nargs="?", default=1e-5)
    parser.add_argument("--ep_mcmc_step_size", type=float, nargs="?", default=1e-6)
    parser.add_argument("--num_rounds", type=int, nargs="?", default=30)
    parser.add_argument("--num_mcmc_steps_per_round", type=int, nargs="?", default=10)
    parser.add_argument("--num_chains", type=int, nargs="?", default=10)
    parser.add_argument("--quench_rounds", type=int, nargs="?", default=0)
    parser.add_argument("--disable_gradients", action="store_true")
    parser.add_argument("--disable_stochasticity", action="store_true")
    boolean_action = argparse.BooleanOptionalAction
    parser.add_argument("--repair", action=boolean_action, default=False)
    parser.add_argument("--predict", action=boolean_action, default=True)
    parser.add_argument("--temper", action=boolean_action, default=False)
    parser.add_argument("--dp_grad_clip", type=float, nargs="?", default=float("inf"))
    parser.add_argument("--ep_grad_clip", type=float, nargs="?", default=float("inf"))
    parser.add_argument("--dont_normalize_gradients", action="store_true")
    args = parser.parse_args()

    # Hyperparameters
    L = args.L
    dp_mcmc_step_size = args.dp_mcmc_step_size
    ep_mcmc_step_size = args.ep_mcmc_step_size
    num_rounds = args.num_rounds
    num_mcmc_steps_per_round = args.num_mcmc_steps_per_round
    num_chains = args.num_chains
    use_gradients = not args.disable_gradients
    use_stochasticity = not args.disable_stochasticity
    repair = args.repair
    predict = args.predict
    temper = args.temper
    quench_rounds = args.quench_rounds
    dp_grad_clip = args.dp_grad_clip
    ep_grad_clip = args.ep_grad_clip
    normalize_gradients = not args.dont_normalize_gradients

    print(f"Running prediction/mitigation on overtake with hyperparameters:")
    print(f"\tmodel_path = {args.model_path}")
    print(f"\timage dimensions (w x h) = {args.image_w} x {args.image_h}")
    print(f"\tL = {L}")
    print(f"\tdp_mcmc_step_size = {dp_mcmc_step_size}")
    print(f"\tep_mcmc_step_size = {ep_mcmc_step_size}")
    print(f"\tnum_rounds = {num_rounds}")
    print(f"\tnum_mcmc_steps_per_round = {num_mcmc_steps_per_round}")
    print(f"\tnum_chains = {num_chains}")
    print(f"\tuse_gradients = {use_gradients}")
    print(f"\tuse_stochasticity = {use_stochasticity}")
    print(f"\trepair = {repair}")
    print(f"\tpredict = {predict}")
    print(f"\ttemper = {temper}")
    print(f"\tquench_rounds = {quench_rounds}")
    print(f"\tdp_grad_clip = {dp_grad_clip}")
    print(f"\tep_grad_clip = {ep_grad_clip}")
    print(f"\tnormalize_gradients = {normalize_gradients}")

    # Add exponential tempering if using
    t = jnp.linspace(0, 1, num_rounds)
    tempering_schedule = 1 - jnp.exp(-5 * t) if temper else None

    # Make a PRNG key (#sorandom)
    prng_key = jrandom.PRNGKey(0)

    # Make the environment to use
    image_shape = (args.image_h, args.image_w)
    env = make_highway_env(image_shape)

    # Load the model (key doesn't matter; we'll replace all leaves with the saved
    # parameters), duplicating the model for each chain. We'll also split partition
    # out just the continuous parameters, which will be our design parameters
    dummy_policy = DrivingPolicy(jrandom.PRNGKey(0), image_shape)
    load_policy = lambda _: eqx.tree_deserialise_leaves(args.model_path, dummy_policy)
    get_dps = lambda _: eqx.partition(load_policy(_), eqx.is_array)[0]
    initial_dps = eqx.filter_vmap(get_dps)(jnp.arange(num_chains))
    # Also save out the static part of the policy
    _, static_policy = eqx.partition(load_policy(None), eqx.is_array)

    # Initialize some random initial states
    prng_key, initial_state_key = jrandom.split(prng_key)
    initial_state_keys = jrandom.split(initial_state_key, num_chains)
    initial_eps = eqx.filter_vmap(env.reset)(initial_state_keys)

    # Run the prediction+mitigation process
    t_start = time.perf_counter()
    dps, eps, dp_logprobs, ep_logprobs = predict_and_mitigate_failure_modes(
        prng_key,
        initial_dps,
        initial_eps,
        dp_logprior_fn=lambda dp: jnp.array(0.0),  # uniform prior over policies
        ep_logprior_fn=env.overall_prior_logprob,
        potential_fn=lambda dp, ep: L * simulate(env, dp, ep, static_policy).potential,
        num_rounds=num_rounds,
        num_mcmc_steps_per_round=num_mcmc_steps_per_round,
        dp_mcmc_step_size=dp_mcmc_step_size,
        ep_mcmc_step_size=ep_mcmc_step_size,
        use_gradients=use_gradients,
        use_stochasticity=use_stochasticity,
        repair=repair,
        predict=predict,
        quench_rounds=quench_rounds,
        tempering_schedule=tempering_schedule,
        dp_grad_clip=dp_grad_clip,
        ep_grad_clip=ep_grad_clip,
        normalize_gradients=normalize_gradients,
    )
    t_end = time.perf_counter()
    print(
        f"Ran {num_rounds:,} rounds with {num_chains} chains in {t_end - t_start:.2f} s"
    )

    # Select the policy that performs best against all predicted failures before
    # the final round (choose from all chains)
    if repair:
        most_likely_dps_idx = jnp.argmax(dp_logprobs[-1], axis=-1)
        final_dps = jtu.tree_map(lambda leaf: leaf[-1, most_likely_dps_idx], dps)
    else:
        # Just pick one policy arbitrarily if we didn't optimize the policies.
        final_dps = jtu.tree_map(lambda leaf: leaf[-1, 0], dps)

    # Evaluate this single policy against all failures
    final_eps = jtu.tree_map(lambda leaf: leaf[-1], eps)

    # # TODO for debugging plots, just use the initial policy and eps
    # t_end = 0.0
    # t_start = 0.0
    # final_dps = jtu.tree_map(lambda leaf: leaf[-1], initial_dps)
    # final_eps = initial_eps
    # dp_logprobs = jnp.zeros((num_rounds, num_chains))
    # ep_logprobs = jnp.zeros((num_rounds, num_chains))
    # # TODO debugging bit ends here

    # Evaluate the solutions proposed by the prediction+mitigation algorithm
    result = eqx.filter_vmap(
        lambda dp, ep: simulate(env, dp, ep, static_policy), in_axes=(None, 0)
    )(final_dps, final_eps)

    # Plot the results
    fig = plt.figure(figsize=(64, 32), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["non_ego_initial_states", "lighting", "trace", "color_1", "color_2"],
            ["initial_obs", "initial_obs", "initial_obs", "initial_obs", "initial_obs"],
            ["final_obs", "final_obs", "final_obs", "final_obs", "final_obs"],
            ["reward", "reward", "reward", "reward", "reward"],
        ],
        per_subplot_kw={
            "lighting": {"projection": "polar"},
            "color_1": {"projection": "3d"},
            "color_2": {"projection": "3d"},
        },
    )

    # Plot the chain convergence
    if predict:
        axs["trace"].plot(ep_logprobs)
        axs["trace"].set_ylabel("Log probability after contingency update")
    else:
        axs["trace"].plot(dp_logprobs)
        axs["trace"].set_ylabel("Log probability after repair")

    axs["trace"].set_xlabel("# Samples")

    # Plot the initial states of the non-ego agents
    axs["non_ego_initial_states"].set_title("Initial states of non-ego agents")
    axs["non_ego_initial_states"].set_xlabel("Car 1 velocity (m/s)")
    axs["non_ego_initial_states"].set_ylabel("Car 2 velocity (m/s)")
    axs["non_ego_initial_states"].scatter(
        final_eps.non_ego_states[:, 0, 3],  # first car velocity
        final_eps.non_ego_states[:, 1, 3],  # second car velocity
        c=result.reward,
    )

    # Plot the initial lighting conditions
    axs["lighting"].set_title("Lighting conditions")
    light_distance = jnp.linalg.norm(final_eps.shading_light_direction, axis=-1)
    inclination = jnp.arccos(final_eps.shading_light_direction[:, 2] / light_distance)
    azimuth = jnp.sign(final_eps.shading_light_direction[:, 1]) * jnp.arccos(
        final_eps.shading_light_direction[:, 0]
        / jnp.linalg.norm(final_eps.shading_light_direction[:, :2], axis=-1)
    )
    axs["lighting"].scatter(
        azimuth,
        inclination,
        c=result.reward,
    )
    axs["lighting"].set_xlabel("Azimuth")
    axs["lighting"].set_ylabel("Inclination")

    axs["color_1"].scatter(
        final_eps.non_ego_colors[:, 0, 0],
        final_eps.non_ego_colors[:, 0, 1],
        final_eps.non_ego_colors[:, 0, 2],
        c=result.reward,
    )
    axs["color_1"].set_title("Car 1 color")
    axs["color_1"].set_xlabel("R")
    axs["color_1"].set_ylabel("G")
    axs["color_1"].set_zlabel("B")

    axs["color_2"].scatter(
        final_eps.non_ego_colors[:, 1, 0],
        final_eps.non_ego_colors[:, 1, 1],
        final_eps.non_ego_colors[:, 1, 2],
        c=result.reward,
    )
    axs["color_2"].set_title("Car 2 color")
    axs["color_2"].set_xlabel("R")
    axs["color_2"].set_ylabel("G")
    axs["color_2"].set_zlabel("B")

    # Plot the reward across all failure cases
    axs["reward"].plot(result.reward, "ko")
    axs["reward"].set_ylabel("Reward")

    # Plot the initial RGB observations tiled on the same subplot
    axs["initial_obs"].imshow(
        jnp.concatenate(result.initial_obs.color_image.transpose(0, 2, 1, 3), axis=1)
    )
    # And do the same for the final observations
    axs["final_obs"].imshow(
        jnp.concatenate(result.final_obs.color_image.transpose(0, 2, 1, 3), axis=1)
    )

    if use_gradients and use_stochasticity:
        alg_type = "mala"
    elif use_gradients and not use_stochasticity:
        alg_type = "gd"
    elif not use_gradients and use_stochasticity:
        alg_type = "rmh"
    else:
        alg_type = "static"
    filename = (
        f"results/{args.savename}/L_{L:0.1e}_"
        f"{num_rounds * num_mcmc_steps_per_round}_samples_"
        f"{quench_rounds}_quench_{'tempered_' if temper else ''}"
        f"{num_chains}_chains_step_dp_{dp_mcmc_step_size:0.1e}_"
        f"ep_{ep_mcmc_step_size:0.1e}_{alg_type}"
    )
    print(f"Saving results to: {filename}")
    os.makedirs(f"results/{args.savename}", exist_ok=True)
    plt.savefig(filename + ".png")

    # Save the final design parameters (joined back into the full policy)
    final_policy = eqx.combine(final_dps, static_policy)
    eqx.tree_serialise_leaves(filename + ".eqx", final_policy)

    # # Save the trace of policies
    # eqx.tree_serialise_leaves(filename + "_trace.eqx", dps)

    # Save the initial policy
    shutil.copyfile(args.model_path, f"results/{args.savename}/" + "initial_policy.eqx")

    # Save the hyperparameters
    with open(filename + ".json", "w") as f:
        json.dump(
            {
                "initial_states": final_eps._asdict(),
                "time": t_end - t_start,
                "L": L,
                "dp_mcmc_step_size": dp_mcmc_step_size,
                "ep_mcmc_step_size": ep_mcmc_step_size,
                "num_rounds": num_rounds,
                "num_mcmc_steps_per_round": num_mcmc_steps_per_round,
                "num_chains": num_chains,
                "use_gradients": use_gradients,
                "use_stochasticity": use_stochasticity,
                "repair": repair,
                "predict": predict,
                "quench_rounds": quench_rounds,
                "tempering_schedule": tempering_schedule,
            },
            f,
            default=lambda x: x.tolist() if isinstance(x, Shaped[Array, "..."]) else x,
        )
