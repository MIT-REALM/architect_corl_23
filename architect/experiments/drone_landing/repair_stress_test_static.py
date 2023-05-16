"""A script for analyzing the results of the prediction experiments."""
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

from architect.experiments.drone_landing.predict_and_mitigate import simulate
from architect.experiments.drone_landing.train_drone_agent import make_drone_landing_env
from architect.systems.drone_landing.env import DroneState
from architect.systems.drone_landing.policy import DroneLandingPolicy

# How many monte carlo trials to use to compute true failure rate
N = 100
BATCHES = 10
# should we re-run the analysis (True) or just load the previously-saved summary (False)
REANALYZE = False
# path to save summary data to in predict_repair_0.1 folder
SUMMARY_PATH = (
    "results/drone_landing_smooth/predict_repair_0.1/stress_test_dp_1.0e-02_+0.json"
)
# Define data sources from individual experiments
SEEDS = [0, 1, 2, 3]
DATA_SOURCES = {
    "mala_tempered": {
        "path_prefix": "results/drone_landing_smooth/predict_repair_0.1/L_1.0e+00/50_samples_10x5/10_chains/5_quench/dp_1.0e-02/ep_1.0e-02/grad_norm/grad_clip_inf/mala_tempered_40",
        "display_name": "Ours (tempered)",
    },
    "rmh": {
        "path_prefix": "results/drone_landing_smooth/predict_repair_0.1/L_1.0e+00/50_samples_10x5/10_chains/0_quench/dp_1.0e-02/ep_1.0e-02/grad_norm/grad_clip_inf/rmh",
        "display_name": "ROCUS",
    },
    # "gd": {
    #     "path_prefix": "results/drone_landing_smooth/predict_repair_0.1/L_1.0e+00/50_samples_10x5/10_chains/0_quench/dp_1.0e-03/ep_3.0e-03/grad_norm/grad_clip_inf/gd",
    #     "display_name": "ML",
    # },
    "reinforce": {
        "path_prefix": "results/drone_landing_smooth/predict_repair_0.1/L_1.0e+00/50_samples_10x5/10_chains/0_quench/dp_1.0e-03/ep_1.0e-03/grad_norm/grad_clip_inf/reinforce_l2c_0.05_step",
        "display_name": "L2C",
    },
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
                    "eps_trace": jax.tree_util.tree_map(
                        lambda x: jnp.array(x),
                        data["eps_trace"],
                        is_leaf=lambda x: isinstance(x, list),
                    ),
                    "T": data["T"],
                    "num_trees": data["num_trees"],
                    "failure_level": data["failure_level"],
                }

            # Also load in the design parameters
            image_shape = (32, 32)
            dummy_policy = DroneLandingPolicy(jax.random.PRNGKey(0), image_shape)
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
    env = make_drone_landing_env(image_shape, loaded_data[alg][seed]["num_trees"])
    cost_fn = lambda ep: simulate(
        env,
        loaded_data[alg][seed]["dp"],
        ep,
        loaded_data[alg][seed]["static_policy"],
        loaded_data[alg][seed]["T"],
    ).potential
    cost_fn = jax.jit(cost_fn)

    # Sample N environmental parameters at random
    key = jax.random.PRNGKey(0)
    costs = []
    print("Running stress test...")
    for _ in tqdm(range(batches)):
        key, subkey = jax.random.split(key)
        initial_state_keys = jax.random.split(subkey, N)
        eps = jax.vmap(env.reset)(initial_state_keys)
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
    algs = []
    costs = []
    seeds = []
    for alg in DATA_SOURCES:
        for seed, result in enumerate(summary_data[alg]):
            algs += [result["display_name"]] * len(result["costs"])
            costs += result["costs"].flatten().tolist()
            seeds += [seed] * len(result["costs"])

    df = pd.DataFrame({"Algorithm": algs, "Cost": costs, "Seed": seeds})

    # Assign maximum cost to any nans
    df["Cost"] = df["Cost"].fillna(df["Cost"].max())

    # Count failures
    failure_level = summary_data["mala_tempered"][0]["failure_level"]
    failure_level = 50.0
    df["Failure"] = df["Cost"] >= failure_level

    # Print failure rates
    print("Mean cost")
    print(df.groupby(["Algorithm"])["Cost"].mean())
    print("cost std err")
    print(
        df.groupby(["Algorithm", "Seed"])["Cost"].mean().groupby(["Algorithm"]).std()
        / 2
    )
    print("Cost (75th)")
    print(df.groupby(["Algorithm"])["Cost"].quantile(0.75))
    print("Cost (25th)")
    print(df.groupby(["Algorithm"])["Cost"].quantile(0.25))

    print(f"Failure rate level={failure_level}")
    print(df.groupby(["Algorithm"])["Failure"].mean())
    print(f"Failure rate {failure_level} stderr")
    print(
        df.groupby(["Algorithm", "Seed"])["Failure"].mean().groupby(["Algorithm"]).std()
        / 2
    )

    # # Plot!
    # plt.figure(figsize=(12, 8))
    # sns.swarmplot(
    #     x="Algorithm",
    #     y="Cost",
    #     # showfliers=False,
    #     # outlier_prop=1e-7,
    #     # # flier_kws={"s": 20},
    #     data=df,
    # )
    # plt.gca().set_xlabel("")
    # # plt.savefig('results/drone_landing_smooth/predict_repair_0.1/seed_0.png') #saving images to file for each seed
    # plt.show()
