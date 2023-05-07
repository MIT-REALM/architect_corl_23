"""A script for analyzing the results of the prediction experiments."""
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from architect.experiments.drone_landing.predict_and_mitigate import simulate
from architect.experiments.drone_landing.train_drone_agent import make_drone_landing_env
from architect.systems.drone_landing.env import DroneState
from architect.systems.drone_landing.policy import DroneLandingPolicy

# Define data sources
DATA_SOURCES = {
    "mala_tempered": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_1.0e-02/ep_1.0e-02/grad_norm/grad_clip_inf/mala_tempered",
        "display_name": "Ours (tempered)",
    },
    "mala": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_1.0e-02/ep_1.0e-02/grad_norm/grad_clip_inf/mala",
        "display_name": "Ours (no temper)",
    },
    "rmh_tempered": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_1.0e-02/ep_1.0e-02/grad_norm/grad_clip_inf/rmh_tempered",
        "display_name": "Ours (no gradients, tempered)",
    },
    "rmh": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_1.0e-02/ep_1.0e-02/grad_norm/grad_clip_inf/rmh",
        "display_name": "ROCUS",
    },
    "ula": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_1.0e-02/ep_1.0e-02/grad_norm/grad_clip_inf/ula",
        "display_name": "DiffScene",
    },
    "gd_tempered": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_3.0e-03/ep_3.0e-03/grad_norm/grad_clip_inf/gd_tempered",
        "display_name": "AdvGrad (tempered)",
    },
    "gd": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_3.0e-03/ep_3.0e-03/grad_norm/grad_clip_inf/gd",
        "display_name": "AdvGrad (no tempering)",
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

            # Convert the ep logprobs to numpy arrays
            ep_logprobs = jnp.array(data["ep_logprobs"])
            # Correct for tempering
            if data["tempering_schedule"] is not None:
                ep_logprobs = ep_logprobs[1:] / jnp.array(data["tempering_schedule"])[
                    1:
                ].reshape(-1, 1)
            else:
                ep_logprobs = ep_logprobs[1:]

            loaded_data[alg] = {
                "ep_logprobs": ep_logprobs,
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

    for alg in loaded_data:
        print(f"Computing costs for {alg}...")
        eps = DroneState(
            drone_state=loaded_data[alg]["eps_trace"]["drone_state"],
            tree_locations=loaded_data[alg]["eps_trace"]["tree_locations"],
            wind_speed=loaded_data[alg]["eps_trace"]["wind_speed"],
        )
        loaded_data[alg]["ep_costs"] = jax.vmap(jax.vmap(cost_fn))(eps)

    return loaded_data


if __name__ == "__main__":
    # Activate seaborn styling
    sns.set_theme(context="paper", style="whitegrid")

    # Load the data
    data = load_data_sources_from_json()

    # Compute costs
    data = get_costs(data)

    # Plot the mean and range of logprobs for each algorithm
    fig, ax = plt.subplots(figsize=(6, 4))
    for alg in data:
        # Plot the mean and range of logprobs
        h = ax.plot(
            np.arange(data[alg]["ep_costs"].shape[0]),
            data[alg]["ep_costs"].mean(axis=-1),
            label=data[alg]["display_name"],
        )
        ax.fill_between(
            np.arange(data[alg]["ep_costs"].shape[0]),
            data[alg]["ep_costs"].min(axis=-1),
            data[alg]["ep_costs"].max(axis=-1),
            color=h[0].get_color(),
            alpha=0.2,
        )

    ax.hlines(
        data[alg]["failure_level"],
        0,
        data[alg]["ep_costs"].shape[0],
        color="k",
        linestyle="--",
        label="Failure level",
    )

    # Set the axis labels
    ax.set_xlabel("Steps")
    ax.set_ylabel("Mean cost of predicted failures")

    # Set the legend
    ax.legend()

    plt.show()
