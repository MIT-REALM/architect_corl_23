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

from architect.engines import predict_and_mitigate_failure_modes
from architect.engines.samplers import init_sampler as init_mcmc_sampler
from architect.engines.samplers import make_kernel as make_mcmc_kernel
from architect.systems.turtle_bot.turtle_bot_game import Game
from architect.systems.turtle_bot.turtle_bot_types import Arena, Environment_state, Policy

config.update("jax_debug_nans", True)


if __name__ == "__main__":
    
    n_targets = Arena.get_n_targets()
    target_pos = Environment_state.get_target_pos()
    sigma = Environment_state.get_sigma()
    
    
    # Set up arguments
    parser = argparse.ArgumentParser()
    #parser.add_argument("--n_targets", type=int, nargs="?", default=1)
    #parser.add_argument("--target_pos", type=jnp.ndarray, nargs="?", default=?)# ADD DEFAULT
    #parser.add_argument("--sigma", type=jnp.ndarray, nargs="?", default=?) #ADD DEFAULT
    
    parser.add_argument("--N", type=int, nargs="?", default = 20) # number of initial conditions
    parser.add_argument("--memory_length", type=int, nargs="?", default = 3)
    parser.add_arguemnt("--n_targets", type=int, nargs="?", default = 1)
    
    parser.add_argument("--width", type=float, nargs="?", default=4 * 3.2)
    parser.add_argument("--height", type=float, nargs="?", default=2 * 2.0)
    parser.add_argument("--duration", type=float, nargs="?", default=50.0)
    parser.add_argument("--dp_mcmc_step_size", type=float, nargs="?", default=1e-2) # dp = design parameters
    parser.add_argument("--ep_mcmc_step_size", type=float, nargs="?", default=1e-2) # ep = exogeneous parameters
    parser.add_argument("--num_rounds", type=int, nargs="?", default=100)
    parser.add_argument("--num_mcmc_steps_per_round", type=int, nargs="?", default=10)
    parser.add_argument("--num_chains", type=int, nargs="?", default=10)
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
    #target_pos = args.target_pos
    #sigma = args.sigma
    N = args.N
    memory_length = args.memory_length
    
    width = args.width
    height = args.height
    buffer = args.buffer
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

    # print("Running HideAndSeek with hyperparameters:")
    # print(f"\tn_seekers = {n_seekers}")
    # print(f"\tn_hiders = {n_hiders}")
    # print(f"\tT = {T}")
    # print(f"\tL = {L}")
    # print(f"\twidth = {width}")
    # print(f"\theight = {height}")
    # print(f"\tbuffer = {buffer}")
    # print(f"\tduration = {duration}")
    # print(f"\tdp_mcmc_step_size = {dp_mcmc_step_size}")
    # print(f"\tep_mcmc_step_size = {ep_mcmc_step_size}")
    # print(f"\tnum_rounds = {num_rounds}")
    # print(f"\tnum_mcmc_steps_per_round = {num_mcmc_steps_per_round}")
    # print(f"\tnum_chains = {num_chains}")
    # print(f"\tuse_gradients = {use_gradients}")
    # print(f"\tuse_stochasticity = {use_stochasticity}")
    # print(f"\trepair = {repair}")
    # print(f"\tpredict = {predict}")
    # print(f"\ttemper = {temper}")
    # print(f"\tquench_rounds = {quench_rounds}")
    # print(f"\tgrad_clip = {grad_clip}")

    # Add exponential tempering if using    #what is tempering?
    t = jnp.linspace(0, 1, num_rounds)
    tempering_schedule = 1 - jnp.exp(-5 * t) if temper else None

    # Create the game
    arena = Arena(width, height, n_targets) # creates the hiding/searching space
    
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
    
    # TODO: can you go over again how the key will change each time?
    key = jax.random.PRNGKey(1)
    policy = Arena.sample_random_policy(key, memory_length)
    
    
    # initialize x_inits
    # let's say x_init_spread = 1
    x_init_spread = 1
    # TODO: Again, how will the key change?
    prng_key = jax.random.PRNGKey(0)
    x_inits = x_init_spread * jax.random.normal(prng_key, shape=(N, 3))  # TODO: HOW TO MAKE THIS EXOGENOUS PARAMETER
    
    
    # initialize x_hists
    def make_x_hist(x_init):
        x_hist = jnp.array([x_init for i in range(memory_length)])
        return x_hist
    make_x_hists = jax.vmap(make_x_hist)
    x_hists = make_x_hists(x_inits)
    
    
    # initialize conc_hists
    def make_conc_hist(x_init):
        conc = Environment_state.get_concentration(x_init[0], x_init[1])
        conc_hist = jnp.array([conc for i in range(memory_length)])
        return conc_hist
    make_conc_hists = jax.vmap(make_conc_hist)
    conc_hists = make_conc_hists(x_inits)    
    
    game = Game(
        policy,
        x_inits,
        x_hists,
        conc_hists,
        memory_length,
        duration = duration,
        dt = 0.1,
    )
        

    # Define a joint logprob for the exogenous paramters
        
    def ep_prior_logprob(ep): 
        n_targets = ep[0]
        target_pos = ep[1]
        sigma = ep[2]
        return (arena.n_targets_prior_logprob(n_targets) 
                + arena.target_pos_prior_logprob(target_pos)
                + arena.sigma_prior_logprob(sigma))


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
        policy,
        (n_targets, target_pos, sigma),
        dp_logprior_fn=arena.multi_trajectory_prior_logprob,
        ep_logprior_fn=joint_ep_prior_logprob,
        ep_potential_fn=lambda dp, ep: L * game(dp, ep[0], ep[1]).potential,
        dp_potential_fn=lambda dp, ep: -L * game(dp, ep[0], ep[1]).potential,
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

# =============================================================================
# 
#     # Select the seeker trajectory that performs best against all hider strategies
#     # predicted before the final round (choose from all chains)
#     if repair:
#         most_likely_dps_idx = jnp.argmax(dp_logprobs[-1], axis=-1)
#         final_dps = jtu.tree_map(lambda leaf: leaf[-1, most_likely_dps_idx], dps)
#     else:
#         # Just pick one dispatch arbitrarily
#         final_dps = jtu.tree_map(lambda leaf: leaf[-1, 0], dps)
#     # Evaluate this against all contingencies
#     final_eps = jtu.tree_map(lambda leaf: leaf[-1], eps)
#     result = jax.vmap(game, in_axes=(None, 0, 0))(final_dps, final_eps[0], final_eps[1])
#     # For later, save the index of the worst contingency
#     worst_eps_idx = jnp.argmax(result.potential)
# 
# =============================================================================
    # Plot the results
    fig = plt.figure(figsize=(32, 16), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["arena", "arena"],
            ["arena", "arena"],
            ["trace", "trace"],
        ]
    )
    axs["arena"].set_title(f"n_targets = {n_targets}, target_pos = {target_pos}, use_gradients = {use_gradients}, use_stochasticity = {use_stochasticity}, num_rounds = {num_rounds}")
    
    # Plot the arena
    axs["arena"].plot(
        [-width / 2, -width / 2, width / 2, width / 2, -width / 2],
        [-height / 2, height / 2, height / 2, -height / 2, -height / 2],
        "k-",
    )

# =============================================================================
#     # Plot initial setup
#     axs["arena"].scatter(
#         game.initial_seeker_positions[:, 0],
#         game.initial_seeker_positions[:, 1],
#         color="b",
#         marker="x",
#         s=25,
#         label="Initial seeker positions",
#     )
#     axs["arena"].scatter(
#         game.initial_hider_positions[:, 0],
#         game.initial_hider_positions[:, 1],
#         color="r",
#         marker="x",
#         s=25,
#         label="Initial hider positions",
#     )
# 
#     axs["arena"].legend()
#     axs["arena"].set_aspect("equal")
# 
#     # Plot planned seeker trajectories
#     t = jnp.linspace(0, 1, 100)
#     for traj in final_dps.trajectories:
#         pts = jax.vmap(traj)(t)
#         axs["arena"].plot(pts[:, 0], pts[:, 1], "b:") # the dotted blue line
#         axs["arena"].scatter(traj.p[:, 0], traj.p[:, 1], s=10, color="b", marker="o")
# 
#     # Plot realized seeker trajectories
#     for i in range(game.initial_seeker_positions.shape[0]):
#         axs["arena"].plot(
#             result.seeker_positions[0, :, i, 0],
#             result.seeker_positions[0, :, i, 1],
#             "b-", # solid blue line
#         )
# 
#     # Plot realized hider trajectories
#     for i in range(game.initial_hider_positions.shape[0]):
#         max_U = result.potential.max()
#         min_U = result.potential.min()
#         for j in range(num_chains):
#             # Make more successful evasions more apparent
#             potential = result.potential[j]
#             alpha = 0.1 + 0.9 * ((potential - min_U) / (1e-3 + max_U - min_U)).item()
#             axs["arena"].plot(
#                 result.hider_positions[j, :, i, 0],
#                 result.hider_positions[j, :, i, 1],
#                 "r-",
#                 linewidth=1,
#                 alpha=alpha,
#             )
#             
# =============================================================================
    # THIS PART SHOULD STAY THE SAME
    # Plot the chain convergence
    if predict:
        axs["trace"].plot(ep_logprobs)
        axs["trace"].set_ylabel("Log probability after contingency update")
    else:
        axs["trace"].plot(dp_logprobs)
        axs["trace"].set_ylabel("Log probability after repair")

    axs["trace"].set_xlabel("# Samples")



#    experiment_type = "hide_and_seek_disturbance"
    if use_gradients and use_stochasticity:
        alg_type = "mala"
    elif use_gradients and not use_stochasticity:
        alg_type = "gd"
    elif not use_gradients and use_stochasticity:
        alg_type = "rmh"
    else:
        alg_type = "static"
    case_name = f"{n_seekers}_seekers_{n_hiders}_hiders"
    filename = (
        f"results/{experiment_type}/{case_name}/L_{L:0.1e}_{T}_T_"
        f"{num_rounds * num_mcmc_steps_per_round}_samples_"
        f"{quench_rounds}_quench_{'tempered_' if temper else ''}"
        f"{num_chains}_chains_step_dp_{dp_mcmc_step_size:0.1e}_"
        f"ep_{ep_mcmc_step_size:0.1e}_{alg_type}"
        f"{'_repair' if repair else ''}"
        f"{'_predict' if predict else ''}"
    )
    print(f"Saving results to: {filename}")
    os.makedirs(f"results/{experiment_type}/{case_name}", exist_ok=True)
    plt.savefig(filename + ".png")

    # Save the dispatch
    with open(filename + ".json", "w") as f:
        json.dump(
            {
                "seeker_traj": {
                    "trajectories": [traj.p.tolist() for traj in final_dps.trajectories]
                },
                "hider_trajs": {
                    "trajectories": [
                        traj.p.tolist() for traj in final_eps[0].trajectories
                    ]
                },
                "disturbance_trajs": {
                    "trajectories": [
                        traj.p.tolist() for traj in final_eps[1].trajectories
                    ]
                },
                "best_hider_traj_idx": worst_eps_idx,
                "time": t_end - t_start,
                "n_seekers": n_seekers,
                "n_hiders": n_hiders,
                "T": T,
                "width": width,
                "height": height,
                "buffer": buffer,
                "duration": duration,
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

    # Save the trace of design parameters
    with open(filename + "_dp_trace.json", "w") as f:
        json.dump(
            {
                "seeker_trajs": dps.trajectories,
                "num_rounds": num_rounds,
                "num_chains": num_chains,
                "n_seekers": n_seekers,
                "n_hiders": n_hiders,
                "width": width,
                "height": height,
                "buffer": buffer,
                "duration": duration,
            },
            f,
            default=lambda x: x.tolist() if isinstance(x, Shaped[Array, "..."]) else x,
        )
