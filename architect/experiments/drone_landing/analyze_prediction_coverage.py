"""A script for analyzing the results of the prediction experiments."""
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jaxtyping import Array, Shaped

from architect.experiments.drone_landing.predict_and_mitigate import simulate
from architect.experiments.drone_landing.train_drone_agent import make_drone_landing_env
from architect.systems.drone_landing.env import DroneState
from architect.systems.drone_landing.policy import DroneLandingPolicy

# should we re-run the analysis (True) or just load the previously-saved summary (False)
REANALYZE = False
# path to save summary data to
SUMMARY_PATH = "results/drone_landing_smooth/predict/summary.json"
# Define data sources from individual experiments
DATA_SOURCES = {
    "mala_tempered": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_1.0e-02/ep_1.0e-02/grad_norm/grad_clip_inf/mala_tempered",
        "display_name": "Ours (tempered)",
    },
    "mala": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_1.0e-02/ep_1.0e-02/grad_norm/grad_clip_inf/mala",
        "display_name": "Ours (no temper)",
    },
    # "rmh_tempered": {
    #     "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_1.0e-02/ep_1.0e-02/grad_norm/grad_clip_inf/rmh_tempered",
    #     "display_name": "Ours (no gradients, tempered)",
    # },
    "rmh": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_1.0e-02/ep_1.0e-02/grad_norm/grad_clip_inf/rmh",
        "display_name": "ROCUS",
    },
    # "ula": {
    #     "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_1.0e-02/ep_1.0e-02/grad_norm/grad_clip_inf/ula",
    #     "display_name": "DiffScene",
    # },
    # "gd_tempered": {
    #     "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_3.0e-03/ep_3.0e-03/grad_norm/grad_clip_inf/gd_tempered",
    #     "display_name": "AdvGrad (tempered)",
    # },
    "gd": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_3.0e-03/ep_3.0e-03/grad_norm/grad_clip_inf/gd",
        "display_name": "ML",
    },
    "reinforce": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_1.0e-03/ep_1.0e-03/grad_norm/grad_clip_inf/reinforce_l2c_0.05_step",
        "display_name": "L2C",
    },
}


def load_data_sources_from_json():
    """Load data sources from a JSON file."""
    loaded_data = {}

    # Load the results from each data source
    for alg in DATA_SOURCES:
        with open(DATA_SOURCES[alg]["path_prefix"] + ".json") as f:
            data = json.load(f)

            loaded_data[alg] = {
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
            DATA_SOURCES[alg]["path_prefix"] + ".eqx", dummy_policy
        )
        dp, static_policy = eqx.partition(full_policy, eqx.is_array)
        loaded_data[alg]["dp"] = dp
        loaded_data[alg]["static_policy"] = static_policy

    return loaded_data


def get_costs(loaded_data):
    """Get the cost at each step of each algorithm in the loaded data."""
    # Pre-compile the same cost function for all
    alg = "mala_tempered"
    image_shape = (32, 32)
    env = make_drone_landing_env(image_shape, loaded_data[alg]["num_trees"])
    cost_fn = lambda ep: simulate(
        env,
        loaded_data[alg]["dp"],
        ep,
        loaded_data[alg]["static_policy"],
        loaded_data[alg]["T"],
    ).potential
    cost_fn = jax.jit(cost_fn)
    ep_logprior_fn = jax.jit(env.initial_state_logprior)

    for alg in loaded_data:
        print(f"Computing costs for {alg}...")
        eps = DroneState(
            drone_state=loaded_data[alg]["eps_trace"]["drone_state"],
            tree_locations=loaded_data[alg]["eps_trace"]["tree_locations"],
            wind_speed=loaded_data[alg]["eps_trace"]["wind_speed"],
        )
        loaded_data[alg]["ep_costs"] = jax.vmap(jax.vmap(cost_fn))(eps)
        loaded_data[alg]["ep_logpriors"] = jax.vmap(jax.vmap(ep_logprior_fn))(eps)
        loaded_data[alg]["ep_logprobs"] = loaded_data[alg]["ep_logpriors"] - jax.nn.elu(
            loaded_data[alg]["failure_level"] - loaded_data[alg]["ep_costs"]
        )

    return loaded_data


if __name__ == "__main__":
    # Activate seaborn styling
    sns.set_theme(context="paper", style="whitegrid")

    if REANALYZE or not os.path.exists(SUMMARY_PATH):
        # Load the data
        data = load_data_sources_from_json()

        # Compute costs
        data = get_costs(data)

        # Extract the summary data we want to save
        summary_data = {}
        for alg in data:
            summary_data[alg] = {
                "display_name": data[alg]["display_name"],
                "ep_costs": data[alg]["ep_costs"],
                "ep_logpriors": data[alg]["ep_logpriors"],
                "ep_logprobs": data[alg]["ep_logprobs"],
                "failure_level": data[alg]["failure_level"],
            }

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
                summary_data[alg]["ep_costs"] = jnp.array(summary_data[alg]["ep_costs"])
                summary_data[alg]["ep_logpriors"] = jnp.array(
                    summary_data[alg]["ep_logpriors"]
                )
                summary_data[alg]["ep_logprobs"] = jnp.array(
                    summary_data[alg]["ep_logprobs"]
                )

    # Post-process
    for alg in DATA_SOURCES:
        failure_level = summary_data[alg]["failure_level"]
        costs = summary_data[alg]["ep_costs"]
        num_failures = (costs >= failure_level).sum(axis=-1)
        # Cumulative max = how many discovered so far
        num_failures = jax.lax.cummax(num_failures)
        # Add a 1 at the start (randomly sampling 10 failures gives 1 failure at step 0)
        num_failures = jnp.concatenate([jnp.ones(1), num_failures])
        summary_data[alg]["num_failures"] = num_failures

    # Plot!
    fig, ax = plt.subplots(figsize=(6, 4))
    for alg in DATA_SOURCES:
        # Plot the failure rate
        h = ax.plot(
            np.arange(summary_data[alg]["num_failures"].shape[0]),
            summary_data[alg]["num_failures"],
            label=summary_data[alg]["display_name"],
        )

    # Set the axis labels
    ax.set_xlabel("Steps")
    ax.set_ylabel("# failures discovered")

    # Set the legend
    ax.legend()

    plt.show()
