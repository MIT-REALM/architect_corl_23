import json
import time

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import seaborn as sns
from beartype import beartype
from beartype.typing import Tuple, Union
from jaxtyping import Array, Float, Integer, jaxtyped

from architect.systems.power_systems.acopf import Dispatch, Network
from architect.systems.power_systems.example_systems.acopf.ieee_14_bus import (
    make_14_bus_network,
)


if __name__ == "__main__":
    jnp.set_printoptions(linewidth=220)  # for easy printing

    # Make the test system
    L = 100.0
    repair_steps = 5
    repair_lr = 1e-6
    sys = make_14_bus_network(L)

    # Load the dispatch and contingencies
    sol_filename = (
        "results/acopf/14bus/scopf_L_1.0e+02_100_smc_steps_10_dispatch_10_networks"
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
    
    # Get the power injections for old and mitigated dispatches on the nominal network
    P_old, _ = sys.power_injections(old_dispatch, sys.nominal_network)
    P_new, _ = sys.power_injections(dispatch, sys.nominal_network)

    # Print the dispatches
    P_min, P_max = sys.bus_active_limits.T
    print("---------------------")
    print("Non-robust dispatch")
    print(["P_min: "] + [f"{p:5.2f}" for p in P_min])
    print(["P_old: "] + [f"{p:5.2f}" for p in P_old])
    print(["P_max: "] + [f"{p:5.2f}" for p in P_max])
    generation = P_old / P_old[P_old >= 0].sum()
    print(["gen %: "] + [f"{p * 100:5.0f}" for p in generation])
    print("---------------------")
    print("Robust dispatch injections")
    print(["P_min: "] + [f"{p:5.2f}" for p in P_min])
    print(["P_new: "] + [f"{p:5.2f}" for p in P_new])
    print(["P_max: "] + [f"{p:5.2f}" for p in P_max])
    generation = P_new / P_new[P_new >= 0].sum()
    print(["gen %: "] + [f"{p * 100:5.0f}" for p in generation])
    print("---------------------")
