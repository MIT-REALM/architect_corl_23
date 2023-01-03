import json
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
    # Hyperparameters
    case_name = "case14"
    L = 100.0
    mcmc_step_size = 1e-6
    num_rounds = 1  # 10
    num_mcmc_steps_per_round = 20
    num_chains = 11
    use_gradients = True
    use_stochasticity = True
    repair = True
    predict = True
    quench_rounds = 0  # 5

    # Load the test case
    sys = load_test_network(case_name, penalty=L)

    # Make a PRNG key (#sorandom)
    prng_key = jrandom.PRNGKey(0)

    # Initialize the dispatch randomly
    prng_key, dispatch_key = jrandom.split(prng_key)
    dispatch_keys = jrandom.split(dispatch_key, num_chains)
    design_params = jax.vmap(sys.sample_random_dispatch)(dispatch_keys)

    # Initialize the network randomly
    prng_key, network_key = jrandom.split(prng_key)
    network_keys = jrandom.split(network_key, num_chains)
    exogenous_params = jax.vmap(sys.sample_random_network)(network_keys)

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
        f"Ran {num_rounds:,} rounds with {num_chains} chains in {t_end - t_start:.2f} s"
    )

    # Select the dispatch that performs best against all contingencies predicted before
    # the final round (choose from all chains)
    most_likely_dps_idx = jnp.argmax(dp_logprobs[-1], axis=-1)
    final_dps = jtu.tree_map(
        lambda leaf: leaf[-1, most_likely_dps_idx], dps
    )
    # Evaluate this against all contingencies
    final_eps = jtu.tree_map(lambda leaf: leaf[-1], eps)
    result = jax.vmap(sys, in_axes=(None, 0))(final_dps, final_eps)

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
    axs["trace"].plot(ep_logprobs)
    axs["trace"].set_xlabel("# Samples")
    axs["trace"].set_ylabel("Log probability after contingency update")

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
        f"results/acopf/{case_name}/sc_dispatch_L_{L:0.1e}_"
        f"{num_rounds * num_mcmc_steps_per_round}_samples_"
        f"{quench_rounds}_quench_rounds_"
        f"{num_chains}_chains_mala_step_{mcmc_step_size:0.1e}"
    )
    plt.savefig(filename + ".png")

    # Save the dispatch
    with open(filename + ".json", "w") as f:
        json.dump(
            {
                "case": case_name,
                "dispatch": final_dps._asdict(),
                "network": final_eps._asdict(),
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
