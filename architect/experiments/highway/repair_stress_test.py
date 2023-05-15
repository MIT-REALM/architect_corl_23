"""A script for analyzing the results of the repair experiments."""
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyemd
import scipy
import seaborn as sns
from jaxtyping import Array, Shaped
from tqdm import tqdm

from architect.experiments.highway.predict_and_mitigate import (
    non_ego_actions_prior_logprob,
    sample_non_ego_actions,
    simulate,
)
from architect.experiments.highway.train_highway_agent import make_highway_env
from architect.systems.highway.driving_policy import DrivingPolicy
from architect.systems.highway.highway_env import HighwayState

# How many monte carlo trials to use to compute true failure rate
N = 100
BATCHES = 10
# should we re-run the analysis (True) or just load the previously-saved summary (False)
REANALYZE = True
# path to save summary data to in predict_repair folder
SUMMARY_PATH = "results/highway_lqr/predict_repair_1.0/stress_test.json"
# Define data sources from individual experiments
SEEDS = [0, 1, 2, 3]
DATA_SOURCES = {
    "mala_tempered": {
        "path_prefix": "results/highway_lqr/predict_repair_1.0/noise_5.0e-01/L_1.0e+00/100_samples/10_chains/0_quench/dp_1.0e-03/ep_1.0e-03/mala_20tempered",
        "display_name": "Ours (tempered)",
    },
    "rmh": {
        "path_prefix": "results/highway_lqr/predict_repair_1.0/noise_5.0e-01/L_1.0e+00/100_samples/10_chains/0_quench/dp_1.0e-03/ep_1.0e-03/rmh",
        "display_name": "ROCUS",
    },
    # "gd": {
    #     "path_prefix": "results/highway_lqr/predict_repair_1.0/noise_5.0e-01/L_1.0e+00/100_samples/10_chains/0_quench/dp_1.0e-03/ep_1.0e-03/gd",
    #     "display_name": "ML",
    # },
    # "reinforce": {
    #     "path_prefix": "results/highway_lqr/predict_repair_1.0/noise_5.0e-01/L_1.0e+00/100_samples/10_chains/0_quench/dp_1.0e-03/ep_1.0e-03/reinforce_l2c",
    #     "display_name": "L2C",
    # },
}


def load_data_sources_from_json():
    """Load data sources from a JSON file."""
    loaded_data = {}

    # Load the results from each data source
    for alg in DATA_SOURCES:
        loaded_data[alg] = []
        for seed in SEEDS:
            with open(DATA_SOURCES[alg]["path_prefix"] + f"_{seed}" + ".json") as f:
                data = json.load(f)

                new_data = {
                    "display_name": DATA_SOURCES[alg]["display_name"],
                    "final_eps": jnp.array(data["action_trajectory"]),
                    "eps_trace": jax.tree_util.tree_map(
                        lambda x: jnp.array(x),
                        data["action_trajectory_trace"],
                        is_leaf=lambda x: isinstance(x, list),
                    ),
                    "failure_level": data["failure_level"],
                    "noise_scale": data["noise_scale"],
                    "initial_state": jax.tree_util.tree_map(
                        lambda x: jnp.array(x),
                        data["initial_state"],
                        is_leaf=lambda x: isinstance(x, list),
                    ),
                }
                new_data["T"] = new_data["eps_trace"].shape[2]

            # Also load in the design parameters
            image_shape = (32, 32)
            dummy_policy = DrivingPolicy(jax.random.PRNGKey(0), image_shape)
            full_policy = eqx.tree_deserialise_leaves(
                DATA_SOURCES[alg]["path_prefix"] + f"_{seed}" + ".eqx", dummy_policy
            )
            dp, static_policy = eqx.partition(full_policy, eqx.is_array)
            new_data["dp"] = dp
            new_data["static_policy"] = static_policy

            loaded_data[alg].append(new_data)

    return loaded_data


def monte_carlo_test(N, batches, loaded_data, alg, seed):
    """Stress test the given policy using N samples in batches"""
    image_shape = (32, 32)
    env = make_highway_env(image_shape)
    initial_state = HighwayState(
        ego_state=loaded_data[alg][seed]["initial_state"]["ego_state"],
        non_ego_states=loaded_data[alg][seed]["initial_state"]["non_ego_states"],
        shading_light_direction=loaded_data[alg][seed]["initial_state"][
            "shading_light_direction"
        ],
        non_ego_colors=loaded_data[alg][seed]["initial_state"]["non_ego_colors"],
    )

    cost_fn = lambda ep: simulate(
        env,
        loaded_data[alg][seed]["dp"],
        initial_state,
        ep,
        loaded_data[alg][seed]["static_policy"],
        loaded_data[alg][seed]["T"],
    ).potential
    cost_fn = jax.jit(cost_fn)

    sample_fn = lambda key: sample_non_ego_actions(
        key,
        env,
        horizon=loaded_data[alg][seed]["T"],
        n_non_ego=2,
        noise_scale=0.5,
    )

    # Sample N environmental parameters at random
    key = jax.random.PRNGKey(0)
    costs = []
    print("Running stress test...")
    for _ in tqdm(range(batches)):
        key, subkey = jax.random.split(key)
        initial_state_keys = jax.random.split(subkey, N)
        eps = jax.vmap(sample_fn)(initial_state_keys)
        costs.append(jax.vmap(cost_fn)(eps))

    return jnp.concatenate(costs)


if __name__ == "__main__":
    # Activate seaborn styling
    sns.set_theme(context="paper", style="whitegrid")

    if REANALYZE or not os.path.exists(SUMMARY_PATH):
        # Load the data
        data = load_data_sources_from_json()

        # Compute costs for each seed for each algorithm
        for alg in data:
            for seed, result in enumerate(data[alg]):
                print(f"Computing costs for {alg} seed {seed}")
                result["costs"] = monte_carlo_test(N, BATCHES, data, alg, seed)

        # Extract the summary data we want to save
        summary_data = {}
        for alg in data:
            summary_data[alg] = []
            for seed in SEEDS:
                summary_data[alg].append(
                    {
                        "display_name": data[alg][seed]["display_name"],
                        "costs": data[alg][seed]["costs"],
                        "failure_level": data[alg][seed]["failure_level"],
                    }
                )

        # Save the data
        with open(SUMMARY_PATH, "w") as f:
            json.dump(
                summary_data,
                f,
                default=lambda x: x.tolist()
                if isinstance(x, Shaped[Array, "..."])
                else x,
            )
    else:
        # Load the data
        with open(SUMMARY_PATH, "rb") as f:
            summary_data = json.load(f)

            for alg in DATA_SOURCES:
                for result in summary_data[alg]:
                    result["costs"] = jnp.array(result["costs"])

    # Post-process into a dataframe
    df = pd.DataFrame()
    for alg in DATA_SOURCES:
        for result in summary_data[alg]:
            df = df.append(
                pd.DataFrame(
                    {
                        "Algorithm": result["display_name"],
                        "Cost": result["costs"].flatten(),
                    }
                )
            )

    # Plot!
    plt.figure(figsize=(12, 8))
    sns.violinplot(
        x="Algorithm",
        y="Cost",
        data=df,
    )
    plt.gca().set_xlabel("")
    # plt.savefig('results/highway_lqr/predict_repair_1.0/seed_0.png') #saving images to file for each seed
    plt.show()
