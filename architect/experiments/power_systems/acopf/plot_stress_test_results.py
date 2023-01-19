import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import jax.numpy as jnp

data = [
    {
        "Algorithm": "GD-NoAdv",
        "Worst case (predicted)": 0.0,
        "Worst case (observed)": 2.689948,
        "Likelihood of underestimate": 0.33210,
        "generation_cost": 0.54969215,
        "stress_test_results": "results/acopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_gd_stress_test.npz",  # noqa
    },
    {
        "Algorithm": "GD",
        "Worst case (predicted)": 0.0,
        "Worst case (observed)": 2.6572380,
        "Likelihood of underestimate": 0.17741300,
        "generation_cost": 0.549297,
        "stress_test_results": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_gd_stress_test.npz",  # noqa
    },
    {
        "Algorithm": "RMH",
        "Worst case (predicted)": 3.318828105,
        "Worst case (observed)": 5.32660293,
        "Likelihood of underestimate": 0.0002,
        "generation_cost": 1.487012,
        "stress_test_results": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_rmh_stress_test.npz",  # noqa
    },
    {
        "Algorithm": "Ours",
        "Worst case (predicted)": 0.35271,
        "Worst case (observed)": 0.2735613,
        "Likelihood of underestimate": 0.00,
        "generation_cost": 1.4132679,
        "stress_test_results": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_mala_stress_test.npz",  # noqa
    },
]

L = 100

tidy_data = []
for packet in data:
    tidy_data.append(
        {
            "Algorithm": packet["Algorithm"],
            "Constraint violation": packet["Worst case (predicted)"],
            "Type": "Predicted",
        }
    )
    tidy_data.append(
        {
            "Algorithm": packet["Algorithm"],
            "Constraint violation": packet["Worst case (observed)"],
            "Type": "Observed",
        }
    )

if __name__ == "__main__":
    results = pd.DataFrame(data)

    sns.set_theme(style="whitegrid", context="paper")
    fig = plt.figure(figsize=(9, 4), constrained_layout=True)

    # Load distributions from files
    potentials = []
    for entry in data:
        potentials.append(jnp.load(entry["stress_test_results"]))
    potentials = jnp.array(potentials)
    N = potentials.shape[1]
    potentials_labels = (
        ["GD-NoAdv" for i in range(N)]
        + ["GD" for i in range(N)]
        + ["RMH" for i in range(N)]
        + ["Ours" for i in range(N)]
    )

    # Plot the distributions of potentials
    ax = sns.boxenplot(
        x=potentials_labels,
        y=potentials.reshape(-1),
        # showfliers=False,
        flier_kws={"s": 20},
    )

    plt.scatter(
        results["Algorithm"],
        L * results["Worst case (predicted)"] + results["generation_cost"],
        marker="X",
        s=100,
        linewidth=4,
        label="Predicted worst-case",
        color="orangered",
    )
    # plt.scatter(
    #     results["Algorithm"],
    #     L * results["Worst case (observed)"] + results["generation_cost"],
    #     marker="_",
    #     s=300,
    #     linewidth=4,
    #     label="Worst-case (observed)",
    # )
    ax.scatter([], [], s=20, color="k", marker="d", label="Observed worst-case")
    plt.ylabel("Cost")
    plt.yscale("log")
    plt.legend()

    plt.show()
