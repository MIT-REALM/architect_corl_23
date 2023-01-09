import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import seaborn as sns
from jaxtyping import Array, Shaped

from architect.engines import predict_and_mitigate_failure_modes
from architect.systems.power_systems.load_test_network import load_test_network


if __name__ == "__main__":
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, nargs="?", default="case14")
    parser.add_argument("--L", type=float, nargs="?", default=100.0)
    parser.add_argument("--mcmc_step_size", type=float, nargs="?", default=1e-6)
    parser.add_argument("--num_rounds", type=int, nargs="?", default=10)
    parser.add_argument("--num_mcmc_steps_per_round", type=int, nargs="?", default=20)
    parser.add_argument("--num_chains", type=int, nargs="?", default=10)
    parser.add_argument("--quench_rounds", type=int, nargs="?", default=5)
    parser.add_argument("--disable_gradients", action="store_true")
    parser.add_argument("--disable_stochasticity", action="store_true")
    parser.add_argument("--repair", nargs="?", const=True, default=True)
    parser.add_argument("--predict", nargs="?", const=True, default=False)
    args = parser.parse_args()

    # Hyperparameters
    case_name = args.case_name
    L = args.L
    mcmc_step_size = args.mcmc_step_size
    num_rounds = args.num_rounds
    num_mcmc_steps_per_round = args.num_mcmc_steps_per_round
    num_chains = args.num_chains
    use_gradients = not args.disable_gradients
    use_stochasticity = not args.disable_stochasticity
    repair = args.repair
    predict = args.predict
    quench_rounds = args.quench_rounds

    print(f"Running ACOPF on {case_name} with hyperparameters:")
    print(f"\tcase_name = {case_name}")
    print(f"\tL = {L}")
    print(f"\tmcmc_step_size = {mcmc_step_size}")
    print(f"\tnum_rounds = {num_rounds}")
    print(f"\tnum_mcmc_steps_per_round = {num_mcmc_steps_per_round}")
    print(f"\tnum_chains = {num_chains}")
    print(f"\tuse_gradients = {use_gradients}")
    print(f"\tuse_stochasticity = {use_stochasticity}")
    print(f"\trepair = {repair}")
    print(f"\tpredict = {predict}")
    print(f"\tquench_rounds = {quench_rounds}")

    # Load the test case
    sys = load_test_network(case_name, penalty=L)

    # Make a PRNG key (#sorandom)
    prng_key = jrandom.PRNGKey(0)

    # Start by solving for an optimal dispatch with the nominal network
    network = sys.network_spec.nominal_network_state
    # Add a batch dimension so we have a "population" of 1 network
    exogenous_params = jtu.tree_map(lambda leaf: jnp.expand_dims(leaf, 0), network)

    # Initialize the dispatch randomly
    prng_key, dispatch_key = jrandom.split(prng_key)
    dispatch_keys = jrandom.split(dispatch_key, num_chains)
    design_params = jax.vmap(sys.sample_random_dispatch)(dispatch_keys)

    print("Setup complete.")

    # Run the prediction+mitigation process
    t_start = time.perf_counter()
    dps, eps, dp_logprobs, ep_logprobs = predict_and_mitigate_failure_modes(
        prng_key,
        design_params,
        exogenous_params,
        dp_logprior_fn=sys.dispatch_prior_logprob,
        ep_logprior_fn=sys.network_prior_logprob,
        potential_fn=lambda dp, ep: sys(dp, ep).potential,
        num_rounds=num_rounds,
        num_mcmc_steps_per_round=num_mcmc_steps_per_round,
        mcmc_step_size=mcmc_step_size,
        use_gradients=use_gradients,
        use_stochasticity=use_stochasticity,
        repair=repair,
        predict=predict,
        quench_rounds=quench_rounds,
    )
    t_end = time.perf_counter()
    print(
        f"Ran {num_rounds:,} rounds of {num_chains} chains in {t_end - t_start:.2f} s."
    )

    # Get the performance with the design parameters after the last round
    # (use the nominal network)
    final_dps = jtu.tree_map(
        lambda leaf: leaf[-1], dps  # take the value of all chains from the last round
    )
    result = jax.vmap(sys, in_axes=(0, None))(final_dps, network)

    # Plot the results
    fig = plt.figure(figsize=(32, 16), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["constraints", "cost"],
            ["trace", "trace"],
            ["generation", "voltage"],
        ]
    )

    # Plot the violations at the best dispatch from each chain
    sns.swarmplot(
        data=[
            result.P_gen_violation.sum(axis=-1),
            result.Q_gen_violation.sum(axis=-1),
            result.P_load_violation.sum(axis=-1),
            result.Q_load_violation.sum(axis=-1),
            result.V_violation.sum(axis=-1),
        ],
        ax=axs["constraints"],
    )
    axs["constraints"].set_xticklabels(["Pg", "Qg", "Pd", "Qd", "V"])
    axs["constraints"].set_ylabel("Constraint Violation")

    # Plot generation cost vs constraint violation
    total_constraint_violation = (
        result.P_gen_violation.sum(axis=-1)
        + result.Q_gen_violation.sum(axis=-1)
        + result.P_load_violation.sum(axis=-1)
        + result.Q_load_violation.sum(axis=-1)
        + result.V_violation.sum(axis=-1)
    )
    axs["cost"].scatter(result.generation_cost, total_constraint_violation)
    axs["cost"].set_xlabel("Generation cost")
    axs["cost"].set_ylabel("Total constraint violation")

    # Plot the chain convergence
    axs["trace"].plot(dp_logprobs)
    axs["trace"].set_xlabel("# Samples")
    axs["trace"].set_ylabel("Overall log probability")

    # Plot the generations along with their limits
    bus = sys.gen_spec.buses
    P_min, P_max = sys.gen_spec.P_limits.T
    for i in range(num_chains):
        P = result.dispatch.gen.P[i, :]
        lower_error = P - P_min
        upper_error = P_max - P
        errs = jnp.vstack((lower_error, upper_error))
        axs["generation"].errorbar(
            bus,
            P,
            yerr=errs,
            linestyle="None",
            marker="o",
            markersize=10,
            linewidth=3.0,
            capsize=10.0,
            capthick=3.0,
        )
    axs["generation"].set_ylabel("$P_g$ (p.u.)")

    # Plot the voltages along with their limits
    bus = jnp.arange(sys.n_bus)
    V_min, V_max = sys.bus_voltage_limits.T
    for i in range(num_chains):
        V = result.voltage_amplitudes[i, :]
        lower_error = V - V_min
        upper_error = V_max - V
        errs = jnp.vstack((lower_error, upper_error))
        axs["voltage"].errorbar(
            bus,
            V,
            yerr=errs,
            linestyle="None",
            marker="o",
            markersize=10,
            linewidth=3.0,
            capsize=10.0,
            capthick=3.0,
        )
    axs["voltage"].set_ylabel("$|V|$ (p.u.)")

    filename = (
        f"results/acopf/{case_name}/dispatch_L_{L:0.1e}_"
        f"{num_rounds * num_mcmc_steps_per_round}_samples_"
        f"{quench_rounds}_quench_rounds_"
        f"{num_chains}_chains_mala_step_{mcmc_step_size:0.1e}"
    )
    print(f"Saving results to: {filename}")
    os.makedirs(f"results/acopf/{case_name}", exist_ok=True)
    plt.savefig(filename + ".png")

    # Save the dispatch
    with open(filename + ".json", "w") as f:
        json.dump(
            {
                "case": case_name,
                "dispatch": final_dps._asdict(),
                "network": network._asdict(),
                "time": t_end - t_start,
                "L": L,
                "mcmc_step_size": mcmc_step_size,
                "num_rounds": num_rounds,
                "num_mcmc_steps_per_round": num_mcmc_steps_per_round,
                "num_chains": num_chains,
                "use_gradients": use_gradients,
                "use_stochasticity": use_stochasticity,
                "repair": repair,
                "predict": predict,
                "quench_rounds": quench_rounds,
            },
            f,
            default=lambda x: x.tolist() if isinstance(x, Shaped[Array, "..."]) else x,
        )
