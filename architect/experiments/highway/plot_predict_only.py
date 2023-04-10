"""Plot the failure modes that we've predicted"""
import json
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.tree_util as jtu

if __name__ == "__main__":
    # Load the data from the JSON file
    random_sample_results_path = "results/overtake_predict_only/L_1.0e+00_1_samples_0_quench_10_chains_step_dp_1.0e-05_ep_1.0e-06_mala.json"
    with open(random_sample_results_path, "r") as f:
        random_samples = json.load(f)["initial_states"]
    
    predicted_failure_results_path = "results/overtake_predict_only/L_1.0e+01_1000_samples_0_quench_tempered_10_chains_step_dp_1.0e-05_ep_1.0e-06_mala.json"
    with open(predicted_failure_results_path, "r") as f:
        predicted_failures = json.load(f)["initial_states"]
    
    # Plot the change
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["ego_initial_states", "non_ego_1_initial_states", "non_ego_2_initial_states"],
            ["lighting", "color_1", "color_2"],
        ],
    )

    # Change in initial ego state
    random_ego_states = jnp.array(random_samples["ego_state"])
    predicted_ego_states = jnp.array(predicted_failures["ego_state"])
    difference = predicted_ego_states - random_ego_states
    axs["ego_initial_states"].plot(difference, label=["x", "y", "theta", "v"])
    axs["ego_initial_states"].legend()
    axs["ego_initial_states"].set_title("Ego initial state")

    # Change in initial non-ego states
    random_non_ego_states = jnp.array(random_samples["non_ego_states"])
    predicted_non_ego_states = jnp.array(predicted_failures["non_ego_states"])
    difference = predicted_non_ego_states - random_non_ego_states
    axs["non_ego_1_initial_states"].plot(difference[:, 0], label=["x", "y", "theta", "v"])
    axs["non_ego_1_initial_states"].set_title("Non-ego 1 (closer)")
    axs["non_ego_1_initial_states"].legend()

    axs["non_ego_2_initial_states"].plot(difference[:, 1], label=["x", "y", "theta", "v"])
    axs["non_ego_2_initial_states"].set_title("Non-ego 2 (further)")
    axs["non_ego_2_initial_states"].legend()

    # Change in lighting
    random_lighting = jnp.array(random_samples["shading_light_direction"])
    predicted_lighting = jnp.array(predicted_failures["shading_light_direction"])
    difference = predicted_lighting - random_lighting
    axs["lighting"].plot(difference, label=["x", "y", "z"])
    axs["lighting"].legend()
    axs["lighting"].set_title("Lighting direction")

    # Change in color
    random_color = jnp.array(random_samples["non_ego_colors"])
    predicted_color = jnp.array(predicted_failures["non_ego_colors"])
    difference = predicted_color - random_color
    axs["color_1"].plot(difference[:, 0], label=["r", "g", "b"])
    axs["color_1"].set_title("Color 1 (closer)")
    axs["color_1"].legend()

    axs["color_2"].plot(difference[:, 1], label=["r", "g", "b"])
    axs["color_2"].set_title("Color 2 (further)")
    axs["color_2"].legend()
    
    plt.show()
