"""A script for analyzing the results of the prediction experiments."""
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    # "gd": {
    #     "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_3.0e-03/ep_3.0e-03/grad_norm/grad_clip_inf/gd",
    #     "display_name": "AdvGrad (no tempering)",
    # },
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
            ep_logprobs = np.array(data["ep_logprobs"])
            # Correct for tempering
            ep_logprobs = ep_logprobs[1:] / np.array(data["tempering_schedule"])[
                1:
            ].reshape(-1, 1)

            loaded_data[alg] = {
                "ep_logprobs": ep_logprobs,
                "display_name": DATA_SOURCES[alg]["display_name"],
            }

    return loaded_data


if __name__ == "__main__":
    # Activate seaborn styling
    sns.set_theme(context="paper", style="whitegrid")

    # Load the data
    data = load_data_sources_from_json()

    # Plot the mean and range of logprobs for each algorithm
    fig, ax = plt.subplots(figsize=(6, 4))
    for alg in data:
        # Plot the mean and range of logprobs
        h = ax.plot(
            np.arange(data[alg]["ep_logprobs"].shape[0]),
            data[alg]["ep_logprobs"].mean(axis=-1),
            label=data[alg]["display_name"],
        )
        ax.fill_between(
            np.arange(data[alg]["ep_logprobs"].shape[0]),
            data[alg]["ep_logprobs"].min(axis=-1),
            data[alg]["ep_logprobs"].max(axis=-1),
            color=h[0].get_color(),
            alpha=0.2,
        )

    # Set the axis labels
    ax.set_xlabel("Steps")
    ax.set_ylabel("Log likelihood of failure")

    # Set the legend
    ax.legend()

    plt.show()
