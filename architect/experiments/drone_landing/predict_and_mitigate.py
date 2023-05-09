"""Code to predict and mitigate failure modes in the drone scenario."""
import argparse
import json
import os
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib.pyplot as plt
from beartype.typing import NamedTuple
from jaxtyping import Array, Float, Shaped

from architect.engines import predict_and_mitigate_failure_modes
from architect.engines.reinforce import init_sampler as init_reinforce_sampler
from architect.engines.reinforce import make_kernel as make_reinforce_kernel
from architect.engines.samplers import init_sampler as init_mcmc_sampler
from architect.engines.samplers import make_kernel as make_mcmc_kernel
from architect.experiments.drone_landing.train_drone_agent import make_drone_landing_env
from architect.systems.drone_landing.env import DroneLandingEnv, DroneObs, DroneState
from architect.systems.drone_landing.policy import DroneLandingPolicy
from architect.types import PRNGKeyArray
from architect.utils import softmin


class SimulationResults(NamedTuple):
    """A class for storing the results of a simulation."""

    potential: Float[Array, ""]
    initial_obs: DroneObs
    final_obs: DroneObs
    drone_traj: Float[Array, "T 4"]
    tree_locations: Float[Array, "num_trees 2"]
    wind_speed: Float[Array, " 2"]


def simulate(
    env: DroneLandingEnv,
    policy: DroneLandingPolicy,
    initial_state: DroneState,
    static_policy: DroneLandingPolicy,
    max_steps: int = 80,
) -> Float[Array, ""]:
    """Simulate the drone landing environment.

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

        # Take a step in the environment using the action carried over from the previous
        # step.
        next_state, next_observation, reward, done = env.step(state, action, key)

        # Compute the action for the next step
        next_action = policy(next_observation)

        # If the environment has already terminated, set the reward to zero.
        reward = jax.lax.cond(already_done, lambda: 0.0, lambda: reward)
        already_done = jnp.logical_or(already_done, done)

        # Don't step if the environment has terminated
        next_action = jax.lax.cond(already_done, lambda: action, lambda: next_action)
        next_state = jax.lax.cond(already_done, lambda: state, lambda: next_state)

        next_carry = (next_action, next_state, already_done)
        output = (reward, state)
        return next_carry, output

    # Get the initial observation and action
    initial_obs = env.get_obs(initial_state)
    initial_action = policy(initial_obs)

    # Transform and rollout!
    keys = jrandom.split(jrandom.PRNGKey(0), max_steps)
    (_, final_state, _), (reward, state_traj) = jax.lax.scan(
        step, (initial_action, initial_state, False), keys
    )

    # Get the final observation
    final_obs = env.get_obs(final_state)

    # The potential is the negative of the (soft) minimum reward observed, plus a
    # penalty based on how far the drone is from the target.
    potential = (
        -softmin(reward, sharpness=1.0) + 5 * (final_state.drone_state[:2] ** 2).sum()
    )

    return SimulationResults(
        potential,
        initial_obs,
        final_obs,
        drone_traj=state_traj.drone_state,
        tree_locations=initial_state.tree_locations,
        wind_speed=initial_state.wind_speed,
    )


if __name__ == "__main__":
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--savename", type=str, default="drone_cond")
    parser.add_argument("--seed", type=int, nargs="?", default=0)
    parser.add_argument("--T", type=int, nargs="?", default=80)
    parser.add_argument("--L", type=float, nargs="?", default=1.0)
    parser.add_argument("--num_trees", type=int, nargs="?", default=10)
    parser.add_argument("--failure_level", type=float, nargs="?", default=25.0)
    parser.add_argument("--dp_mcmc_step_size", type=float, nargs="?", default=1e-5)
    parser.add_argument("--ep_mcmc_step_size", type=float, nargs="?", default=1e-5)
    parser.add_argument("--num_rounds", type=int, nargs="?", default=100)
    parser.add_argument("--num_steps_per_round", type=int, nargs="?", default=10)
    parser.add_argument("--num_chains", type=int, nargs="?", default=10)
    parser.add_argument("--quench_rounds", type=int, nargs="?", default=0)
    parser.add_argument("--disable_gradients", action="store_true")
    parser.add_argument("--disable_stochasticity", action="store_true")
    parser.add_argument("--disable_mh", action="store_true")
    parser.add_argument("--reinforce", action="store_true")
    boolean_action = argparse.BooleanOptionalAction
    parser.add_argument("--repair", action=boolean_action, default=False)
    parser.add_argument("--predict", action=boolean_action, default=True)
    parser.add_argument("--temper", action=boolean_action, default=True)
    parser.add_argument("--grad_clip", type=float, nargs="?", default=float("inf"))
    parser.add_argument("--dont_normalize_gradients", action="store_true")
    args = parser.parse_args()

    # Hyperparameters
    L = args.L
    seed = args.seed
    T = args.T
    num_trees = args.num_trees
    failure_level = args.failure_level
    dp_mcmc_step_size = args.dp_mcmc_step_size
    ep_mcmc_step_size = args.ep_mcmc_step_size
    num_rounds = args.num_rounds
    num_steps_per_round = args.num_steps_per_round
    num_chains = args.num_chains
    use_gradients = not args.disable_gradients
    use_stochasticity = not args.disable_stochasticity
    use_mh = not args.disable_mh
    reinforce = args.reinforce
    repair = args.repair
    predict = args.predict
    temper = args.temper
    quench_rounds = args.quench_rounds
    grad_clip = args.grad_clip
    normalize_gradients = not args.dont_normalize_gradients

    print("Running prediction/mitigation on drone landing with hyperparameters:")
    print(f"\tmodel_path = {args.model_path}")
    print(f"\tseed = {seed}")
    print(f"\timage dimensions (w x h) = 32 x 32")
    print(f"\tT = {T}")
    print(f"\tL = {L}")
    print(f"\tnum_trees = {num_trees}")
    print(f"\tfailure_level = {failure_level}")
    print(f"\tdp_mcmc_step_size = {dp_mcmc_step_size}")
    print(f"\tep_mcmc_step_size = {ep_mcmc_step_size}")
    print(f"\tnum_rounds = {num_rounds}")
    print(f"\tnum_steps_per_round = {num_steps_per_round}")
    print(f"\tnum_chains = {num_chains}")
    print(f"\tuse_gradients = {use_gradients}")
    print(f"\tuse_stochasticity = {use_stochasticity}")
    print(f"\tuse_mh = {use_mh}")
    print(f"\trepair = {repair}")
    print(f"\tpredict = {predict}")
    print(f"\ttemper = {temper}")
    print(f"\tquench_rounds = {quench_rounds}")
    print(f"\tgrad_clip = {grad_clip}")
    print(f"\tnormalize_gradients = {normalize_gradients}")
    print(
        f"\tUsing alternative algorithm? {reinforce}",
        f"(reinforce = {reinforce})",
    )

    # Add exponential tempering if using
    t = jnp.linspace(0, 1, num_rounds)
    tempering_schedule = 1 - jnp.exp(-40 * t) if temper else None

    # Make a PRNG key (#sorandom)
    prng_key = jrandom.PRNGKey(seed)

    # Make the environment to use
    image_shape = (32, 32)
    env = make_drone_landing_env(image_shape, num_trees)
    env._collision_penalty = 2e2

    # Load the model (key doesn't matter; we'll replace all leaves with the saved
    # parameters), duplicating the model for each chain. We'll also split partition
    # out just the continuous parameters, which will be our design parameters
    dummy_policy = DroneLandingPolicy(jrandom.PRNGKey(0), image_shape)
    load_policy = lambda _: eqx.tree_deserialise_leaves(args.model_path, dummy_policy)
    get_dps = lambda _: eqx.partition(load_policy(_), eqx.is_array)[0]
    initial_dps = eqx.filter_vmap(get_dps)(jnp.arange(num_chains))
    # Also save out the static part of the policy
    _, static_policy = eqx.partition(load_policy(None), eqx.is_array)

    # Initialize some initial states (these serve as our initial exogenous parameters)
    prng_key, initial_state_key = jrandom.split(prng_key)
    initial_state_keys = jrandom.split(initial_state_key, num_chains)
    initial_eps = jax.vmap(env.reset)(initial_state_keys)

    # Choose which sampler to use
    if reinforce:
        init_sampler_fn = init_reinforce_sampler
        make_kernel_fn = lambda logprob_fn, step_size, _: make_reinforce_kernel(
            logprob_fn,
            step_size,
            perturbation_stddev=0.05,
            baseline_update_rate=0.5,
        )
    else:
        # This sampler yields either MALA, GD, or RMH depending on whether gradients
        # and/or stochasticity are enabled
        init_sampler_fn = lambda params, logprob_fn: init_mcmc_sampler(
            params,
            logprob_fn,
            normalize_gradients,
        )
        make_kernel_fn = lambda logprob_fn, step_size, stochasticity: make_mcmc_kernel(
            logprob_fn,
            step_size,
            use_gradients,
            stochasticity,
            grad_clip,
            normalize_gradients,
            use_mh,
        )

    if reinforce:
        alg_type = "reinforce_l2c_0.05_step"
    elif use_gradients and use_stochasticity and use_mh:
        alg_type = "mala"
    elif use_gradients and use_stochasticity and not use_mh:
        alg_type = "ula"
    elif use_gradients and not use_stochasticity:
        alg_type = "gd"
    elif not use_gradients and use_stochasticity and use_mh:
        alg_type = "rmh"
    elif not use_gradients and use_stochasticity and not use_mh:
        alg_type = "random_walk"
    else:
        alg_type = "static"

    # Run the prediction+mitigation process
    t_start = time.perf_counter()
    dps, eps, dp_logprobs, ep_logprobs = predict_and_mitigate_failure_modes(
        prng_key,
        initial_dps,
        initial_eps,
        dp_logprior_fn=lambda _: jnp.array(0.0),  # uniform prior over policies
        ep_logprior_fn=lambda ep: env.initial_state_logprior(ep),
        potential_fn=lambda dp, ep: -L
        * jax.nn.elu(failure_level - simulate(env, dp, ep, static_policy, T).potential),
        # potential_fn=lambda dp, ep: L
        # * simulate(env, dp, ep, static_policy, T).potential,
        init_sampler=init_sampler_fn,
        make_kernel=make_kernel_fn,
        num_rounds=num_rounds,
        num_mcmc_steps_per_round=num_steps_per_round,
        dp_mcmc_step_size=dp_mcmc_step_size,
        ep_mcmc_step_size=ep_mcmc_step_size,
        use_stochasticity=use_stochasticity,
        repair=repair,
        predict=predict,
        quench_rounds=quench_rounds,
        tempering_schedule=tempering_schedule,
        logging_prefix=f"{args.savename}/{alg_type}[{os.getpid()}]",
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
        lambda dp, ep: simulate(env, dp, ep, static_policy, T),
        in_axes=(None, 0),
    )(final_dps, final_eps)

    # Plot the results
    fig = plt.figure(figsize=(32, 16), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["trace", "wind_speed", "trajectory", "trajectory", "trajectory"],
            ["initial_obs", "initial_obs", "initial_obs", "initial_obs", "initial_obs"],
            ["final_obs", "final_obs", "final_obs", "final_obs", "final_obs"],
            ["reward", "reward", "reward", "reward", "reward"],
        ],
    )

    # Plot the chain convergence
    if predict:
        axs["trace"].plot(ep_logprobs)
        axs["trace"].set_ylabel("Log probability after contingency update")
    else:
        axs["trace"].plot(dp_logprobs)
        axs["trace"].set_ylabel("Log probability after repair")

    axs["trace"].set_xlabel("# Samples")

    # Plot the trajectories for each case, color-coded by potential
    max_potential = jnp.max(result.potential)
    min_potential = jnp.min(result.potential)
    normalized_potential = (result.potential - min_potential) / (
        max_potential - min_potential
    )
    axs["trajectory"].axhline(4.0, linestyle="--", color="k")
    axs["trajectory"].axhline(-4.0, linestyle="--", color="k")
    axs["trajectory"].scatter(0.0, 0.0, marker="x", color="k", label="Goal")
    for chain_idx in range(num_chains):
        axs["trajectory"].plot(
            result.drone_traj[chain_idx, :, 0].T,
            result.drone_traj[chain_idx, :, 1].T,
            linestyle="-",
            color=plt.cm.plasma_r(normalized_potential[chain_idx]),
            label="Ego" if chain_idx == 0 else None,
        )
        for tree_location in result.tree_locations[chain_idx]:
            axs["trajectory"].add_patch(
                plt.Circle(
                    tree_location,
                    radius=0.5,
                    color=plt.cm.plasma_r(normalized_potential[chain_idx]),
                    fill=False,
                )
            )

    axs["trajectory"].set_xlabel("x")
    axs["trajectory"].set_ylabel("y")
    axs["trajectory"].set_aspect("equal")
    axs["trajectory"].set_ylim([-4.1, 4.1])
    axs["trajectory"].set_xlim([-15.0, 1.0])

    # Plot the wind speed for each case, color-coded by potential
    axs["wind_speed"].scatter(
        result.wind_speed[:, 0],
        result.wind_speed[:, 1],
        marker="o",
        color=plt.cm.plasma_r(normalized_potential),
    )
    axs["wind_speed"].set_xlabel("Wind speed x")
    axs["wind_speed"].set_ylabel("Wind speed y")

    # Plot the reward across all failure cases
    axs["reward"].plot(result.potential, "ko")
    axs["reward"].set_ylabel("Potential (negative min reward)")

    # Plot the initial RGB observations tiled on the same subplot
    axs["initial_obs"].imshow(
        jnp.concatenate(result.initial_obs.image.transpose(0, 2, 1, 3), axis=1)
    )
    # And do the same for the final observations
    axs["final_obs"].imshow(
        jnp.concatenate(result.final_obs.image.transpose(0, 2, 1, 3), axis=1)
    )

    save_dir = (
        f"results/{args.savename}/{'predict' if predict else ''}"
        f"{'_' if repair else ''}{'repair' if repair else ''}/"
        f"L_{L:0.1e}/"
        f"{num_rounds * num_steps_per_round}_samples_{num_rounds}x{num_steps_per_round}/"
        f"{num_chains}_chains/"
        f"{quench_rounds}_quench/"
        f"dp_{dp_mcmc_step_size:0.1e}/"
        f"ep_{ep_mcmc_step_size:0.1e}/"
        f"{'grad_norm' if normalize_gradients else 'no_grad_norm'}/"
        f"grad_clip_{grad_clip}/"
    )
    filename = save_dir + f"{alg_type}{'_tempered_40' if temper else ''}_{seed}"
    print(f"Saving results to: {filename}")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(filename + ".png")

    # Save the final design parameters (joined back into the full policy)
    final_policy = eqx.combine(final_dps, static_policy)
    eqx.tree_serialise_leaves(filename + ".eqx", final_policy)

    # Save the trace of policies
    eqx.tree_serialise_leaves(filename + "_trace.eqx", dps)

    # Save the initial policy
    try:
        shutil.copyfile(
            args.model_path, f"results/{args.savename}/" + "initial_policy.eqx"
        )
    except shutil.SameFileError:
        pass

    # Save the hyperparameters
    with open(filename + ".json", "w") as f:
        json.dump(
            {
                "eps": final_eps._asdict(),
                "eps_trace": eps._asdict(),
                "num_trees": num_trees,
                "time": t_end - t_start,
                "L": L,
                "T": T,
                "failure_level": failure_level,
                "dp_mcmc_step_size": dp_mcmc_step_size,
                "ep_mcmc_step_size": ep_mcmc_step_size,
                "num_rounds": num_rounds,
                "num_steps_per_round": num_steps_per_round,
                "num_chains": num_chains,
                "use_gradients": use_gradients,
                "use_stochasticity": use_stochasticity,
                "repair": repair,
                "predict": predict,
                "quench_rounds": quench_rounds,
                "tempering_schedule": tempering_schedule,
                "ep_logprobs": ep_logprobs,
                "dp_logprobs": dp_logprobs,
            },
            f,
            default=lambda x: x.tolist() if isinstance(x, Shaped[Array, "..."]) else x,
        )
