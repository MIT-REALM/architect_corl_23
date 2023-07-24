#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:48:16 2023

@author: rchanna
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib.pyplot as plt
from jax.config import config
from jaxtyping import Array, Shaped

import equinox as eqx

from architect.engines import predict_and_mitigate_failure_modes
from architect.engines.samplers import init_sampler as init_mcmc_sampler
from architect.engines.samplers import make_kernel as make_mcmc_kernel
from architect.systems.turtle_bot.turtle_bot_game import Game
from architect.systems.turtle_bot.turtle_bot_types import (
    Arena,
    Environment_state as Env,
    Policy,
)

config.update("jax_debug_nans", True)


if __name__ == "__main__":
    # n_targets = Arena.get_n_targets()

    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure_level", type=float, nargs="?", default=5.0)
    parser.add_argument(
        "--N", type=int, nargs="?", default=5
    )  # number of initial conditions
    parser.add_argument("--memory_length", type=int, nargs="?", default=3)
    parser.add_argument("--n_targets", type=int, nargs="?", default=1)
    parser.add_argument("--L", type=float, nargs="?", default=10.0)
    parser.add_argument("--width", type=float, nargs="?", default=20.0)
    parser.add_argument("--height", type=float, nargs="?", default=20.0)
    parser.add_argument("--duration", type=float, nargs="?", default=200.0)
    parser.add_argument(
        "--dp_mcmc_step_size", type=float, nargs="?", default=1e-2
    )  # dp = design parameters
    parser.add_argument(
        "--ep_mcmc_step_size", type=float, nargs="?", default=1e-2
    )  # ep = exogeneous parameters
    parser.add_argument("--num_rounds", type=int, nargs="?", default=30)
    parser.add_argument("--num_mcmc_steps_per_round", type=int, nargs="?", default=1)
    parser.add_argument("--num_chains", type=int, nargs="?", default=2)
    parser.add_argument("--quench_rounds", type=int, nargs="?", default=25)
    parser.add_argument("--disable_gradients", action="store_true")
    parser.add_argument("--disable_stochasticity", action="store_true")
    boolean_action = argparse.BooleanOptionalAction
    parser.add_argument("--repair", action=boolean_action, default=True)
    parser.add_argument("--predict", action=boolean_action, default=True)
    parser.add_argument("--temper", action=boolean_action, default=False)
    parser.add_argument("--grad_clip", type=float, nargs="?", default=float("inf"))
    args = parser.parse_args()

    # Hyperparameters
    n_targets = args.n_targets
    # target_pos = args.target_pos
    # sigma = args.sigma
    N = args.N
    memory_length = args.memory_length

    width = args.width
    height = args.height
    # buffer = args.buffer
    L = args.L
    duration = args.duration
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
    grad_clip = args.grad_clip
    failure_level = args.failure_level

    print("Running TurtleBot with hyperparameters:")
    print(f"\tN = {N}")
    print(f"\tmemory_length = {memory_length}")
    print(f"\tn_targets = {n_targets}")
    print(f"\tL = {L}")
    print(f"\twidth = {width}")
    print(f"\theight = {height}")
    # print(f"\tbuffer = {buffer}")
    print(f"\tduration = {duration}")
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
    print(f"\tgrad_clip = {grad_clip}")
    print(f"\tfailure_level = {failure_level}")

    # Add exponential tempering if using    #what is tempering?
    t = jnp.linspace(0, 1, num_rounds)
    tempering_schedule = 1 - jnp.exp(-5 * t) if temper else None

    # Create the game
    arena = Arena(width, height, n_targets)  # creates the hiding/searching space

    # Other options of ways in which I can randomly initialize the policy
    # =============================================================================
    #     # randomize policy
    #     prng_key = jax.random.PRNGKey(1)
    #     prng_key, subkey = jax.random.split(prng_key)
    #     policy = Policy(subkey,memory_length)
    #
    #     # or randomize policy like this? TODO: CHECK THIS
    #     prng_key, policy_key = jrandom.split(prng_key)
    #     policy_keys = jrandom.split(policy, num_chains)
    #     policy = jax.vmap(Policy)(policy_keys)
    # =============================================================================

    # initialize x_inits
    # let's say x_init_spread = 1
    x_init_spread = 10

    prng_key = jax.random.PRNGKey(0)
    x_inits = x_init_spread * jax.random.uniform(prng_key, shape=(N, 3))

    policy = arena.sample_random_policy(prng_key, memory_length)
    target_pos = arena.sample_random_target_pos(prng_key)
    sigma = arena.sample_random_sigma(prng_key)
    exogenous_env = Env(target_pos, sigma, x_inits)

    # TODO: make keyword arguments to avoid beartype error. Or put in same position as Game init method
    game = Game(
        # x_inits = x_inits,
        # x_hists = x_hists,
        # conc_hists = conc_hists,
        memory_length=memory_length,
        n_targets=n_targets,
        duration=duration,
        dt=0.1,
    )

    # Define a joint logprob for the exogenous paramters
    def joint_ep_prior_logprob(ep):
        target_pos = ep[0]
        sigma = ep[1]
        x_inits = ep[2]

        return (
            arena.target_pos_prior_logprob(target_pos)
            + arena.sigma_prior_logprob(sigma)
            + arena.x_inits_prior_logprob(x_inits, x_init_spread)
        )

    # Initialize the dp randomly
    def sample_random_dp(prng_key):
        return arena.sample_random_policy(prng_key, memory_length)

    prng_key, policy_key = jrandom.split(prng_key)
    policy_keys = jrandom.split(policy_key, num_chains)
    import pdb

    pdb.set_trace()
    init_design_params = jax.vmap(sample_random_dp)(policy_keys)

    prng_key, env_key = jrandom.split(prng_key)
    env_keys = jrandom.split(env_key, num_chains)
    batched_init_targets = jax.vmap(arena.sample_random_target_pos)(env_keys)
    batched_init_sigma = jax.vmap(arena.sample_random_sigma)(env_keys)

    def sample_x_init(prng_key):
        return arena.sample_random_x_inits(prng_key, N, x_init_spread)

    batched_init_x_inits = jax.vmap(sample_x_init)(env_keys)
    # import pdb
    # pdb.set_trace()

    # This sampler yields either MALA, GD, or RMH depending on whether gradients and/or
    # stochasticity are enabled
    init_sampler_fn = lambda params, logprob_fn: init_mcmc_sampler(
        params,
        logprob_fn,
        False,  # don't normalize gradients
    )
    make_kernel_fn = lambda logprob_fn, step_size, stochasticity: make_mcmc_kernel(
        logprob_fn,
        step_size,
        use_gradients,
        stochasticity,
        grad_clip,
        False,  # don't normalize gradients
        True,  # use metroplis-hastings
    )

    # Run the prediction+mitigation process # is this the falsification process? or adversarial-like in finding and fixing failure points?
    t_start = time.perf_counter()

    dps, eps, dp_logprobs, ep_logprobs = predict_and_mitigate_failure_modes(
        prng_key,
        init_design_params,  # policy, #TODO: the type of dp and ep is supposed to be a ndarray of floats. How to fix that?
        (batched_init_targets, batched_init_sigma, batched_init_x_inits),
        dp_logprior_fn=arena.policy_prior_logprob_uniform,
        ep_logprior_fn=joint_ep_prior_logprob,
        ep_potential_fn=lambda dp, ep: L * game(ep[0], ep[1], ep[2], dp).potential,
        dp_potential_fn=lambda dp, ep: -L * game(ep[0], ep[1], ep[2], dp).potential,
        init_sampler=init_sampler_fn,
        make_kernel=make_kernel_fn,
        num_rounds=num_rounds,
        num_mcmc_steps_per_round=num_mcmc_steps_per_round,
        dp_mcmc_step_size=dp_mcmc_step_size,
        ep_mcmc_step_size=ep_mcmc_step_size,
        use_stochasticity=use_stochasticity,
        repair=repair,
        predict=predict,
        quench_rounds=quench_rounds,
        quench_dps=False,  # quench both dps and eps
        tempering_schedule=tempering_schedule,
    )
    t_end = time.perf_counter()
    print(
        f"Ran {num_rounds:,} rounds with {num_chains} chains in {t_end - t_start:.2f} s"
    )

    # The same approach can be used here

    # TODO: working until here!
    # Select the policy that performs best against all environment states
    # predicted before the final round (choose from all chains)
    if repair:
        most_likely_dps_idx = jnp.argmax(dp_logprobs[-1], axis=-1)
        final_dps = jtu.tree_map(lambda leaf: leaf[-1, most_likely_dps_idx], dps)
    else:
        # Just pick one policy arbitrarily
        final_dps = jtu.tree_map(lambda leaf: leaf[-1, 0], dps)
    # Evaluate this against all contingencies
    final_eps = jtu.tree_map(lambda leaf: leaf[-1], eps)
    result = jax.vmap(game, in_axes=(0, 0, 0, None))(
        final_eps[0], final_eps[1], final_eps[2], final_dps
    )
    # For later, save the index of the worst contingency
    worst_eps_idx = jnp.argmax(result.potential)
    print(result.target_pos)
    # Plot the results
    fig = plt.figure(figsize=(8, 4), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["arena", "arena"],
            ["arena", "arena"],
            ["trace", "trace"],
        ]
    )
    axs["arena"].set_title(
        f"n_targets = {n_targets}, target_pos = {target_pos}, use_gradients = {use_gradients}, use_stochasticity = {use_stochasticity}, num_rounds = {num_rounds}"
    )

    # Plot the arena
    axs["arena"].plot(
        [-width / 2, -width / 2, width / 2, width / 2, -width / 2],
        [-height / 2, height / 2, height / 2, -height / 2, -height / 2],
        "k-",
    )

    # =============================================================================
    #     # Plot initial setup
    print(target_pos, "target_pos")
    print(sigma, "sigma")
    print((result.xs.shape), "RESULT SHAPE")
    print((result.xs[0, 1, :, 0]), "XS")
    print((result.xs[8, 1, :, 0]), "XS")

    delta = 0.025
    x = jnp.arange(-10.0, 10.0, delta)  # Change the range so it works in all cases
    y = jnp.arange(-10.0, 10.0, delta)
    X, Y = jnp.meshgrid(x, y)

    # def dummy_fn(x: jnp.ndarray, y:jnp.ndarray):
    #     return exogenous_env.get_concentration(x, y, n_targets)
    # data = jax.vmap(jax.vmap(dummy_fn))(X,Y)
    # axs["arena"].contour(X,Y, data)
    # axs["arena"].set_aspect("equal")

    # TODO: ????
    # for i in range(N):
    #     for j in range(num_chains):
    #         axs["arena"].plot(result.target_pos[j,:,0], result.target_pos[j,:,1], "m*")
    #         axs["arena"].plot(result.xs[j,i,:,0], result.xs[j,i,:,1])
    #         axs["arena"].plot(result.xs[j,i,0,0], result.xs[j,i,0,1], "bo")
    #         axs["arena"].plot(result.xs[j,i,-1,0], result.xs[j,i,-1,1], "ro")
    colors = ["r", "b", "c", "g", "k", "y"]
    for j in range(num_chains):
        # TODO: CHECK THIS STUFF!!!
        col = colors[j % (len(colors))]

        chain_env = Env(result.target_pos[j], result.sigma[j], batched_init_x_inits[j])

        def dummy_fn(x: jnp.ndarray, y: jnp.ndarray):
            return chain_env.get_concentration(x, y, n_targets)

        data = jax.vmap(jax.vmap(dummy_fn))(X, Y)
        axs["arena"].contour(X, Y, data, levels=[0.5], colors=col)

        axs["arena"].plot(
            result.target_pos[j, :, 0],
            result.target_pos[j, :, 1],
            color=col,
            marker="*",
        )
        for i in range(N):
            axs["arena"].plot(result.xs[j, i, :, 0], result.xs[j, i, :, 1], color=col)
            axs["arena"].plot(
                result.xs[j, i, 0, 0], result.xs[j, i, 0, 1], color=col, marker="^"
            )
            axs["arena"].plot(
                result.xs[j, i, -1, 0], result.xs[j, i, -1, 1], color=col, marker="o"
            )

    #
    # =============================================================================
    # # THIS PART SHOULD STAY THE SAME
    # # Plot the chain convergence
    if predict:
        axs["trace"].plot(ep_logprobs)
        axs["trace"].set_ylabel("Log probability after contingency update")
        # print (ep_logprobs, "EP LOGPROBS")
        # print (eps, "EPS")
    else:
        axs["trace"].plot(dp_logprobs)
        axs["trace"].set_ylabel("Log probability after repair")
        # print (dp_logprobs, "DP LOGPROBS")
    axs["trace"].set_xlabel("# Samples")
    print(result.potential.shape, "POTENTIAL")
    # import pdb
    # pdb.set_trace()
    plt.show()

    import pdb

    pdb.set_trace()

# SAVING THE FILE
# =============================================================================
#
# #    experiment_type = "hide_and_seek_disturbance"
#     if use_gradients and use_stochasticity:
#         alg_type = "mala"
#     elif use_gradients and not use_stochasticity:
#         alg_type = "gd"
#     elif not use_gradients and use_stochasticity:
#         alg_type = "rmh"
#     else:
#         alg_type = "static"
#     case_name = f"{n_seekers}_seekers_{n_hiders}_hiders"
#     filename = (
#         f"results/{experiment_type}/{case_name}/L_{L:0.1e}_{T}_T_"
#         f"{num_rounds * num_mcmc_steps_per_round}_samples_"
#         f"{num_chains}_chains_step_dp_{dp_mcmc_step_size:0.1e}_"
#         f"ep_{ep_mcmc_step_size:0.1e}_{alg_type}"
#         f"{'_repair' if repair else ''}"
#         f"{'_predict' if predict else ''}"
#     )
#     print(f"Saving results to: {filename}")
#     os.makedirs(f"results/{experiment_type}/{case_name}", exist_ok=True)
#     plt.savefig(filename + ".png")
#
#     # Save the dispatch
#     with open(filename + ".json", "w") as f:
#         json.dump(
#             {
#                 "seeker_traj": {
#                     "trajectories": [traj.p.tolist() for traj in final_dps.trajectories]
#                 },
#                 "hider_trajs": {
#                     "trajectories": [
#                         traj.p.tolist() for traj in final_eps[0].trajectories
#                     ]
#                 },
#                 "disturbance_trajs": {
#                     "trajectories": [
#                         traj.p.tolist() for traj in final_eps[1].trajectories
#                     ]
#                 },
#                 "best_hider_traj_idx": worst_eps_idx,
#                 "time": t_end - t_start,
#                 "n_seekers": n_seekers,
#                 "n_hiders": n_hiders,
#                 "T": T,
#                 "width": width,
#                 "height": height,
#                 "buffer": buffer,
#                 "duration": duration,
#                 "dp_mcmc_step_size": dp_mcmc_step_size,
#                 "ep_mcmc_step_size": ep_mcmc_step_size,
#                 "num_rounds": num_rounds,
#                 "num_mcmc_steps_per_round": num_mcmc_steps_per_round,
#                 "num_chains": num_chains,
#                 "use_gradients": use_gradients,
#                 "use_stochasticity": use_stochasticity,
#                 "repair": repair,
#                 "predict": predict,
#                 "quench_rounds": quench_rounds,
#                 "tempering_schedule": tempering_schedule,
#             },
#             f,
#             default=lambda x: x.tolist() if isinstance(x, Shaped[Array, "..."]) else x,
#         )
#
#     # Save the trace of design parameters
#     with open(filename + "_dp_trace.json", "w") as f:
#         json.dump(
#             {
#                 "seeker_trajs": dps.trajectories,
#                 "num_rounds": num_rounds,
#                 "num_chains": num_chains,
#                 "n_seekers": n_seekers,
#                 "n_hiders": n_hiders,
#                 "width": width,
#                 "height": height,
#                 "buffer": buffer,
#                 "duration": duration,
#             },
#             f,
#             default=lambda x: x.tolist() if isinstance(x, Shaped[Array, "..."]) else x,
#         )
#
# =============================================================================
