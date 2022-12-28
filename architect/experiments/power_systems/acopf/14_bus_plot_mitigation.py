import json

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

from architect.systems.power_systems.acopf import Dispatch, Network
from architect.systems.power_systems.load_test_network import (
    load_test_network,
)


if __name__ == "__main__":
    jnp.set_printoptions(linewidth=220)  # for easy printing

    # Make the test system
    L = 100.0
    repair_steps = 5
    repair_lr = 1e-6
    sys = load_test_network("case14", penalty=L)

    # Load the dispatch and contingencies
    sol_filename = (
        "results/acopf/14bus/mean_scopf_L_1.0e+02_100_smc_steps_10_dispatch_10_networks"
        "_mala_step_1.0e-05_1000_substeps_repair_steps_5_repair_lr_1.0e-06.json"
    )
    with open(sol_filename, "r") as f:
        sol = json.load(f)
        dispatch = Dispatch(
            jnp.array(sol["dispatch"]["voltage_amplitudes"]),
            jnp.array(sol["dispatch"]["voltage_angles"]),
        )
        networks = jax.vmap(Network)(
            jnp.array(sol["contingencies"]["line_conductances"]),
            jnp.array(sol["contingencies"]["line_susceptances"]),
        )

    # Also compare to the non-mitigated dispatch
    old_dispatch_filename = (
        "results/acopf/14bus/dispatch_L_1.0e+02_30000_samples_"
        "10_chains_mala_step_1.0e-05_repair_steps_5_repair_lr_1.0e-06.json"
    )
    with open(old_dispatch_filename, "r") as f:
        old_dispatch_json = json.load(f)
        old_dispatch = Dispatch(
            jnp.array(old_dispatch_json["best_dispatch"]["voltage_amplitudes"])[0],
            jnp.array(old_dispatch_json["best_dispatch"]["voltage_angles"])[0],
        )

    # Allow local repair
    steps = 10_000
    lr = 1e-6
    old_dispatch = sys.repair_dispatch(old_dispatch, sys.nominal_network, steps, lr)
    dispatch = sys.repair_dispatch(dispatch, sys.nominal_network, steps, lr)

    # Get the power injections for old and mitigated dispatches on the nominal network
    P_old, _ = sys.power_injections(old_dispatch, sys.nominal_network)
    P_new, _ = sys.power_injections(dispatch, sys.nominal_network)

    # Print the dispatches
    P_min, P_max = sys.bus_active_limits.T
    print("---------------------")
    print("Non-robust dispatch (nominal network)")
    print(["P_min: "] + [f"{p:5.2f}" for p in P_min])
    print(["P_old: "] + [f"{p:5.2f}" for p in P_old])
    print(["P_max: "] + [f"{p:5.2f}" for p in P_max])
    generation = P_old / P_old[P_old >= 0].sum()
    print(["gen %: "] + [f"{p * 100:5.0f}" for p in generation])
    print("---------------------")
    print("Robust dispatch injections (nominal network)")
    print(["P_min: "] + [f"{p:5.2f}" for p in P_min])
    print(["P_new: "] + [f"{p:5.2f}" for p in P_new])
    print(["P_max: "] + [f"{p:5.2f}" for p in P_max])
    generation = P_new / P_new[P_new >= 0].sum()
    print(["gen %: "] + [f"{p * 100:5.0f}" for p in generation])
    print("---------------------")

    # Print the outages in the repaired case
    nominal_conductances = sys.nominal_network.line_conductances
    nominal_susceptances = sys.nominal_network.line_susceptances
    results = jax.vmap(sys, in_axes=(None, 0, None, None))(
        dispatch, networks, steps, lr
    )
    violations = results.P_violation + results.Q_violation + results.V_violation
    threshold = 1e-3
    unrecoverable = violations > threshold
    print("Unrecoverable contingencies for robust dispatch")
    for i in unrecoverable.nonzero()[0]:
        print(f"- Contingency idx {i} with total violation {violations[i]:.3f}")
        print(f"\tP violation {results.P_violation[i]:.3f}")
        print(f"\tQ violation {results.Q_violation[i]:.3f}")
        print(f"\tV violation {results.V_violation[i]:.3f}")
        print("\t{}".format(["P_min: "] + [f"{p:5.2f}" for p in P_min]))
        print("\t{}".format(["P_new: "] + [f"{p:5.2f}" for p in results.P[i]]))
        print("\t{}".format(["P_max: "] + [f"{p:5.2f}" for p in P_max]))

        conductances = networks.line_conductances[i]
        susceptances = networks.line_susceptances[i]
        outages = jnp.logical_or(
            jnp.abs(conductances) < 0.1 * jnp.abs(nominal_conductances),
            jnp.abs(susceptances) < 0.1 * jnp.abs(nominal_susceptances),
        )
        print(f"\tOutages: {sys.lines[outages.nonzero()].tolist()}")

    # Plot dispatches
    sns.set_theme(context="talk", style="ticks")
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["generation"],
        ]
    )
    bus = jnp.arange(sys.n_bus)

    # Plot the generation limits
    lower_error = P_old - P_min
    upper_error = P_max - P_old
    errs = jnp.vstack((lower_error, upper_error))
    axs["generation"].errorbar(
        bus,
        P_old,
        yerr=errs,
        linestyle="None",
        marker="o",
        markersize=0,
        linewidth=3.0,
        capsize=0.0,
        capthick=3.0,
        color="k",
        label="Limits",
    )
    axs["generation"].set_ylabel("Net active power injection (p.u.)")
    axs["generation"].set_xlabel("Bus #")

    # Plot old dispatch
    axs["generation"].scatter(
        bus,
        P_old,
        marker=0,
        s=150,
        linewidth=3,
        color="#d95f02",
        label="Initial optimal dispatch",
    )

    # Plot new dispatch
    axs["generation"].scatter(
        bus,
        P_new,
        marker=1,
        s=150,
        linewidth=3,
        color="#1b9e77",
        label="Dispatch with mitigation",
    )

    axs["generation"].hlines(
        [0], bus.min(), bus.max(), linestyles=":", colors="k", alpha=0.5
    )

    axs["generation"].legend()
    axs["generation"].set_xticks(bus)

    sns.despine(trim=True)
    plt.show()
