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

from architect.experiments.simple_grasping.predict_and_mitigate import (
    GraspExogenousParams,
    ep_logprior,
    sample_ep,
    simulate,
)
from architect.systems.simple_grasping.policy import AffordancePredictor

# How many monte carlo trials to use to compute true failure rate
N = 1000
BATCHES = 200
# should we re-run the analysis (True) or just load the previously-saved summary (False)
REANALYZE = True
# path to save summary data to
SUMMARY_PATH = "results/grasping_bowl/predict/coverage_summary.json"
# path to load convergence data from
CONVERGENCE_SUMMARY_PATH = (
    "results/grasping_bowl/predict/convergence_summary_gradnorm_mcmc_1.0e-01.json"
)
# Define data sources from individual experiments
SEEDS = [0, 1, 2, 3]

# These data sources are from results/highway, can be changed to results/highway_lqr
DATA_SOURCES = {
    "mala_tempered": {
        "path_prefix": "results/grasping_bowl/predict/L_1.0e+01/20_samples_20x1/10_chains/0_quench/dp_1.0e-01/ep_1.0e-01/grad_norm/grad_clip_inf/mala_tempered_40",
        "display_name": "RADIUM (ours)",
    },
    "rmh": {
        "path_prefix": "results/grasping_bowl/predict/L_1.0e+01/20_samples_20x1/10_chains/0_quench/dp_1.0e-01/ep_1.0e-01/grad_norm/grad_clip_inf/rmh",
        "display_name": "ROCUS",
    },
    "gd": {
        "path_prefix": "results/grasping_bowl/predict/L_1.0e+01/20_samples_20x1/10_chains/0_quench/dp_1.0e-01/ep_1.0e-01/grad_norm/grad_clip_inf/gd",
        "display_name": "ML",
    },
    "reinforce": {
        "path_prefix": "results/grasping_bowl/predict/L_1.0e+01/20_samples_20x1/10_chains/0_quench/dp_1.0e-01/ep_1.0e-01/grad_norm/grad_clip_inf/reinforce_l2c_0.05_step",
        "display_name": "L2C",
    },
}


# This is the same function as in drone_landing/analyze_prediction_coverage
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
                    "object_type": data["object_type"],
                    "failure_level": data["failure_level"],
                }

            # Also load in the design parameters
            dummy_policy = AffordancePredictor(jax.random.PRNGKey(0))
            full_policy = eqx.tree_deserialise_leaves(
                DATA_SOURCES[alg]["path_prefix"] + f"_{seed}" + ".eqx", dummy_policy
            )
            dp, static_policy = eqx.partition(full_policy, eqx.is_array)
            new_data["dp"] = dp
            new_data["static_policy"] = static_policy

            loaded_data[alg].append(new_data)

    return loaded_data


# This is the same function as in drone_landing/analyze_prediction_coverage, added initial state from highway/analyze_prediction_converge
def monte_carlo_test(N, batches, loaded_data):
    """Stress test the given policy using N samples in batches"""
    # Pre-compile the same cost function for all
    alg = "mala_tempered"
    cost_fn = lambda ep: simulate(
        loaded_data[alg][0]["object_type"],
        ep,
        loaded_data[alg][0]["dp"],
        loaded_data[alg][0]["static_policy"],
    ).potential
    cost_fn = jax.jit(cost_fn)

    sample_fn = lambda key: sample_ep(key)

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

    # Load convergence data (has costs for each algorithm)
    with open(CONVERGENCE_SUMMARY_PATH, "rb") as f:
        convergence_summary_data = json.load(f)

        for alg in convergence_summary_data:
            for result in convergence_summary_data[alg]:
                result["ep_costs"] = jnp.array(result["ep_costs"])
                result["ep_logpriors"] = jnp.array(result["ep_logpriors"])
                result["ep_logprobs"] = jnp.array(result["ep_logprobs"])

    if REANALYZE or not os.path.exists(SUMMARY_PATH):
        # Load the data
        data = load_data_sources_from_json()

        # Compute costs
        monte_carlo_costs = monte_carlo_test(N, BATCHES, data)

        # Extract the summary data we want to save
        summary_data = {
            "random_sample_costs": monte_carlo_costs,
            "N": N,
            "batches": BATCHES,
        }
        for alg in data:
            summary_data[alg] = []
            for result in convergence_summary_data[alg]:
                summary_data[alg].append(
                    {
                        "display_name": result["display_name"],
                        "ep_costs": result["ep_costs"],
                        "ep_logpriors": result["ep_logpriors"],
                        "ep_logprobs": result["ep_logprobs"],
                        "failure_level": result["failure_level"],
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

            summary_data["random_sample_costs"] = jnp.array(
                summary_data["random_sample_costs"]
            )
            for alg in DATA_SOURCES:
                for result in summary_data[alg]:
                    result["ep_costs"] = jnp.array(result["ep_costs"])
                    result["ep_logpriors"] = jnp.array(result["ep_logpriors"])
                    result["ep_logprobs"] = jnp.array(result["ep_logprobs"])

    # Post-process into a dataframe
    df = pd.DataFrame(
        {
            "Algorithm": "Monte Carlo from prior",
            "Cost": summary_data["random_sample_costs"],
        }
    )
    for alg in DATA_SOURCES:
        for result in summary_data[alg]:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "Algorithm": result["display_name"],
                            "Cost": result["ep_costs"].flatten(),
                        }
                    ),
                ]
            )

    # Compute wassertein distance of each algorithm from the prior
    bins = 10
    gt, gt_bins = jnp.histogram(
        summary_data["random_sample_costs"],
        bins,
        density=True,
        weights=jnp.exp(-jax.nn.elu(2.5 - summary_data["random_sample_costs"])),
    )
    gt_bin_centers = (gt_bins[1:] + gt_bins[:-1]) / 2.0
    print(
        "Wasserstein distance from ground truth (importance sampling w. "
        + f"N={summary_data['N'] * summary_data['batches']}):"
    )
    for alg in DATA_SOURCES:
        costs = jnp.concatenate(
            [x["ep_costs"][0:].reshape(-1) for x in summary_data[alg]]
        )
        hist, hist_bins = jnp.histogram(costs, bins, density=True)
        hist_bin_centers = (hist_bins[1:] + hist_bins[:-1]) / 2.0

        distance = lambda x, y: jnp.linalg.norm(x - y)
        distance_to_gt = lambda x: jax.vmap(distance, in_axes=(0, None))(
            gt_bin_centers, x
        )
        distance_matrix = jax.vmap(distance_to_gt)(hist_bin_centers)

        wasserstein = pyemd.emd(
            np.array(gt, dtype=np.float64),
            np.array(hist, dtype=np.float64),
            np.array(distance_matrix, dtype=np.float64),
        )
        print(f"{DATA_SOURCES[alg]['display_name']}: {wasserstein}")

    # Also compute and print the JS divergence
    print(f"JS divergence from ground truth (importance sampling w. {N * BATCHES}):")
    for alg in DATA_SOURCES:
        costs = jnp.concatenate(
            [x["ep_costs"][0:].reshape(-1) for x in summary_data[alg]]
        )
        # re-compute the histogram to use the same bins as the ground truth
        hist, hist_bins = jnp.histogram(costs, bins, density=True)

        # Convert to numpy
        gt = np.array(gt, dtype=np.float64)
        hist = np.array(hist, dtype=np.float64)
        js = scipy.spatial.distance.jensenshannon(gt, hist)
        print(f"{DATA_SOURCES[alg]['display_name']} JS: {js}")

    # Plot!
    plt.figure(figsize=(12, 8))
    sns.violinplot(
        x="Algorithm",
        y="Cost",
        # showfliers=False,
        # outlier_prop=1e-7,
        # flier_kws={"s": 20},
        data=df,
    )
    plt.gca().set_xlabel("")

    plt.show()
