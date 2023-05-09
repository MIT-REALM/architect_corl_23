"""A script for analyzing the results of the prediction experiments."""
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jaxtyping import Array, Shaped

from architect.experiments.highway.predict_and_mitigate import (
    non_ego_actions_prior_logprob,
    simulate,
)
from architect.experiments.highway.train_highway_agent import make_highway_env
from architect.systems.highway.driving_policy import DrivingPolicy
from architect.systems.highway.highway_env import HighwayState

# should we re-run the analysis (True) or just load the previously-saved summary (False)
REANALYZE = False
# path to save summary data to
SUMMARY_PATH = "results/highway_lqr/predict/convergence_summary.json"
# Define data sources from individual experiments
SEEDS = [0]
DATA_SOURCES = {
    "mala_tempered": {
        "path_prefix": "results/highway_lqr/predict/noise_5.0e-01/L_1.0e+00/200_samples/10_chains/0_quench/dp_1.0e-03/ep_1.0e-03/mala_tempered",
        "display_name": "RADIUM (ours)",
    },
    "rmh": {
        "path_prefix": "results/highway_lqr/predict/noise_5.0e-01/L_1.0e+00/200_samples/10_chains/0_quench/dp_1.0e-03/ep_1.0e-03/rmh",
        "display_name": "ROCUS",
    },
    "gd": {
        "path_prefix": "results/highway_lqr/predict/noise_5.0e-01/L_1.0e+00/200_samples/10_chains/0_quench/dp_1.0e-03/ep_1.0e-03/gd",
        "display_name": "ML",
    },
    "reinforce": {
        "path_prefix": "results/highway_lqr/predict/noise_5.0e-01/L_1.0e+00/200_samples/10_chains/0_quench/dp_1.0e-03/ep_1.0e-03/reinforce_l2c",
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
                DATA_SOURCES[alg]["path_prefix"] + f"_{seed}" + ".eqx",
                dummy_policy,
            )
            dp, static_policy = eqx.partition(full_policy, eqx.is_array)
            new_data["dp"] = dp
            new_data["static_policy"] = static_policy

            loaded_data[alg].append(new_data)

    return loaded_data


def get_costs(loaded_data):
    """Get the cost at each step of each algorithm in the loaded data."""
    # Pre-compile the same cost function for all
    alg = "mala_tempered"
    image_shape = (32, 32)
    env = make_highway_env(image_shape)
    initial_state = HighwayState(
        ego_state=loaded_data[alg]["initial_state"]["ego_state"],
        non_ego_states=loaded_data[alg]["initial_state"]["non_ego_states"],
        shading_light_direction=loaded_data[alg]["initial_state"][
            "shading_light_direction"
        ],
        non_ego_colors=loaded_data[alg]["initial_state"]["non_ego_colors"],
    )

    cost_fn = lambda ep: simulate(
        env,
        loaded_data[alg]["dp"],
        initial_state,
        ep,
        loaded_data[alg]["static_policy"],
        loaded_data[alg]["T"],
    ).potential
    cost_fn = jax.jit(cost_fn)
    ep_logprior_fn = jax.jit(
        lambda ep: non_ego_actions_prior_logprob(
            ep, env, loaded_data[alg]["noise_scale"]
        )
    )

    for alg in loaded_data:
        for i, result in enumerate(loaded_data[alg]):
            print(f"Computing costs for {alg}, seed {i}...")
            eps = result["eps_trace"]

            result["ep_costs"] = jax.vmap(jax.vmap(cost_fn))(eps)
            result["ep_logpriors"] = jax.vmap(jax.vmap(ep_logprior_fn))(eps)
            result["ep_logprobs"] = result["ep_logpriors"] - jax.nn.elu(
                result["failure_level"] - result["ep_costs"]
            )

    return loaded_data


if __name__ == "__main__":
    # Activate seaborn styling
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)

    if REANALYZE or not os.path.exists(SUMMARY_PATH):
        # Load the data
        data = load_data_sources_from_json()

        # Compute costs
        data = get_costs(data)

        # Extract the summary data we want to save
        summary_data = {}
        for alg in data:
            summary_data[alg] = []
            for result in data[alg]:
                summary_data[alg].append(
                    {
                        "display_name": data[alg]["display_name"],
                        "ep_costs": data[alg]["ep_costs"],
                        "ep_logpriors": data[alg]["ep_logpriors"],
                        "ep_logprobs": data[alg]["ep_logprobs"],
                        "failure_level": data[alg]["failure_level"],
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

            for alg in summary_data:
                for result in summary_data[alg]:
                    result["ep_costs"] = jnp.array(result["ep_costs"])
                    result["ep_logpriors"] = jnp.array(result["ep_logpriors"])
                    result["ep_logprobs"] = jnp.array(result["ep_logprobs"])

    # Post-process
    for alg in DATA_SOURCES:
        for result in summary_data[alg]:
            failure_level = result["failure_level"]
            costs = result["ep_costs"]
            num_failures = (costs >= failure_level).sum(axis=-1)
            # Cumulative max = 'how many failures have we seen so far?'
            num_failures = jax.lax.cummax(num_failures)
            # Add a 0 at the start (randomly sampling 10 failures gives 0 failures at step 0)
            num_failures = jnp.concatenate([jnp.zeros(1), num_failures])
            result["num_failures"] = num_failures

    # Make into pandas dataframe
    iters = pd.Series([], dtype=int)
    logprobs = pd.Series([], dtype=float)
    costs = pd.Series([], dtype=float)
    num_failures = pd.Series([], dtype=float)
    algs = pd.Series([], dtype=str)
    for alg in DATA_SOURCES:
        for result in summary_data[alg]:
            num_iters = result["ep_logprobs"].shape[0]
            num_chains = result["ep_logprobs"].shape[1]

            # Add the number of failures discovered initially
            iters = pd.concat(
                [iters, pd.Series(jnp.zeros(num_chains, dtype=int))], ignore_index=True
            )
            num_failures = pd.concat(
                [
                    num_failures,
                    pd.Series([float(result["num_failures"][0])] * num_chains),
                ],
                ignore_index=True,
            )
            logprobs = pd.concat(
                [logprobs, pd.Series(jnp.zeros(num_chains))], ignore_index=True
            )
            costs = pd.concat(
                [costs, pd.Series(jnp.zeros(num_chains))], ignore_index=True
            )
            algs = pd.concat(
                [algs, pd.Series([result["display_name"]] * num_chains)],
                ignore_index=True,
            )

            # Add the data for the rest of the iterations
            for i in range(num_iters):
                iters = pd.concat(
                    [iters, pd.Series(jnp.zeros(num_chains, dtype=int) + i + 1)],
                    ignore_index=True,
                )
                logprobs = pd.concat(
                    [logprobs, pd.Series(result["ep_logprobs"][i, :])],
                    ignore_index=True,
                )
                costs = pd.concat(
                    [
                        costs,
                        pd.Series(
                            -jax.nn.elu(
                                result["failure_level"] - result["ep_costs"][i, :]
                            )
                        ),
                    ],
                    ignore_index=True,
                )
                num_failures = pd.concat(
                    [
                        num_failures,
                        pd.Series([float(result["num_failures"][i + 1])] * num_chains),
                    ],
                    ignore_index=True,
                )
                algs = pd.concat(
                    [algs, pd.Series([result["display_name"]] * num_chains)],
                    ignore_index=True,
                )

    df = pd.DataFrame()
    df["Diffusion steps"] = iters
    df["Overall log likelihood"] = logprobs
    df["$[J^* - J]_+$"] = costs
    df["# failures discovered"] = num_failures
    df["Algorithm"] = algs

    # Plot!
    plt.figure(figsize=(12, 8))
    # sns.barplot(
    #     data=df[(df["Diffusion steps"] % 10 == 0)],
    #     x="Diffusion steps",
    #     y="# failures discovered",
    #     hue="Algorithm",
    # )
    sns.lineplot(
        data=df,
        x="Diffusion steps",
        y="# failures discovered",
        hue="Algorithm",
        linewidth=3,
    )

    plt.show()
