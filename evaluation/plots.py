# evaluation/plots.py

import numpy as np
import matplotlib.pyplot as plt

def set_experience_xticks(num_experiences):
    plt.xticks(
        ticks=range(num_experiences),
        labels=[f"Exp {i+1}" for i in range(num_experiences)]
    )

def to_percent(values):
    values = list(values)

    # If values are in 0–1 scale, convert to 0–100
    if max(values) <= 1.0:
        values = [v * 100 for v in values]

    return values


def plot_method_curves(all_results, metric_key):
    markers = ["o", "s", "^", "D", "x", "*", "P", "X", "v", "<", ">"]
    linestyles = ["-", "--", "-.", ":"]

    for idx, (name, result) in enumerate(all_results.items()):
        values = result[metric_key]
        values = to_percent(values)

        plt.plot(
            range(len(values)),
            values,
            label=name,
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            linewidth=2,
            markersize=7,
            alpha=0.9,
        )

# ===============================
# SINGLE-RUN PLOTS
# ===============================

def plot_result_matrix(R, title="Accuracy Matrix R"):
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(R, aspect="auto", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy (%)")

    ax.set_xlabel("Evaluated Experience")
    ax.set_ylabel("Training Experience")
    ax.set_title(title)

    ax.set_xticks(range(R.shape[1]))
    ax.set_yticks(range(R.shape[0]))

    ax.set_xticklabels([f"Exp {i+1}" for i in range(R.shape[1])])
    ax.set_yticklabels([f"Exp {i+1}" for i in range(R.shape[0])])

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            val = R[i, j]

            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val * 100:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if val < 0.5 else "black",
                    fontsize=9,
                )

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_experience(R):
    T = R.shape[0]

    plt.figure(figsize=(8, 5))

    for j in range(T):
        plt.plot(
            range(T),
            R[:, j] * 100,
            marker="o",
            label=f"Experience {j}",
        )

    set_experience_xticks(T)
    plt.xlabel("Training Experience")
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Experience Accuracy")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_three_protocols(task_accs, growing_accs, fixed_accs):
    task_accs = np.array(task_accs) * 100
    growing_accs = np.array(growing_accs) * 100
    fixed_accs = np.array(fixed_accs) * 100

    plt.figure(figsize=(7, 5))

    plt.plot(task_accs, marker="o", label="Current Task Accuracy")
    plt.plot(growing_accs, marker="o", label="Growing Test Set Accuracy")
    plt.plot(fixed_accs, marker="o", label="Fixed Test Set Accuracy")

    num_experiences = len(task_accs)
    set_experience_xticks(num_experiences)
    plt.xlabel("Training Experience")
    plt.ylabel("Accuracy (%)")
    plt.title("Task vs Growing vs Fixed Accuracy")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ===============================
# MULTI-APPROACH COMPARISON
# ===============================

def plot_metric_comparison(all_results, metric_name):
    names = list(all_results.keys())
    values = [all_results[n]["metrics"][metric_name] for n in names]

    plt.figure(figsize=(8, 5))
    plt.bar(names, values)

    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Comparison Across Strategies")

    plt.xticks(rotation=30)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def plot_fixed_comparison(all_results):
    plt.figure(figsize=(8, 5))

    plot_method_curves(all_results, "fixed_accs")

    num_experiences = len(next(iter(all_results.values()))["fixed_accs"])
    set_experience_xticks(num_experiences)
    plt.xlabel("Training Experience")
    plt.ylabel("Accuracy (%)")
    plt.title("Fixed Test Accuracy Comparison")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_growing_comparison(all_results):
    plt.figure(figsize=(8, 5))

    plot_method_curves(all_results, "growing_accs")

    num_experiences = len(next(iter(all_results.values()))["growing_accs"])
    set_experience_xticks(num_experiences)
    plt.xlabel("Training Experience")
    plt.ylabel("Accuracy (%)")
    plt.title("Growing Test Accuracy Comparison")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_task_comparison(all_results):
    plt.figure(figsize=(8, 5))

    plot_method_curves(all_results, "task_accs")

    num_experiences = len(next(iter(all_results.values()))["task_accs"])
    set_experience_xticks(num_experiences)
    plt.xlabel("Training Experience")
    plt.ylabel("Accuracy (%)")
    plt.title("Current Task Accuracy Comparison")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_radar_metrics(
    all_results,
    metric_names=None,
    title="Radar Metric Comparison",
):
    """
    Radar plot comparing methods across summary metrics.

    Supports:
    - list of metric names
    - dict mapping:
        {
            "real metric key": "short label"
        }
    """

    if metric_names is None:
        metric_names = {
            "A Metric": "A",
            "ACC Metric": "ACC",
            "BWT (Backward Transfer)": "BWT",
            "FWT (Forward Transfer)": "FWT",
            "Forgetting": "FGT",
            "REM (Remembering)": "REM",
            "MS (Memory Score)": "MS",
            "SSS (Sample Storage Score)": "SSS",
            "CE (Compute Efficiency)": "CE",
        }

    # ---------------------------------------------------
    # Handle dict mapping vs list input
    # ---------------------------------------------------

    if isinstance(metric_names, dict):
        metric_keys = list(metric_names.keys())
        labels = list(metric_names.values())
    else:
        metric_keys = metric_names
        labels = metric_names

    num_vars = len(labels)

    # ---------------------------------------------------
    # Radar angles
    # ---------------------------------------------------

    angles = np.linspace(
        0,
        2 * np.pi,
        num_vars,
        endpoint=False,
    ).tolist()

    angles += angles[:1]

    # ---------------------------------------------------
    # Figure
    # ---------------------------------------------------

    plt.figure(figsize=(8, 8))

    ax = plt.subplot(111, polar=True)

    # ---------------------------------------------------
    # Plot each method
    # ---------------------------------------------------

    for method_name, result in all_results.items():

        metrics = result["metrics"]

        values = []

        for metric in metric_keys:

            value = metrics.get(metric, 0.0)

            # Handle lists
            if isinstance(value, (list, tuple, np.ndarray)):
                value = value[-1]

            value = float(value)

            # Convert 0-1 scale to percentage
            if abs(value) <= 1.0:
                value *= 100

            values.append(value)

        # Close polygon
        values += values[:1]

        ax.plot(
            angles,
            values,
            linewidth=2,
            marker="o",
            label=method_name,
        )

        ax.fill(
            angles,
            values,
            alpha=0.08,
        )

    # ---------------------------------------------------
    # Labels
    # ---------------------------------------------------

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Hide radial ticks 
    ax.set_yticks([])

    ax.set_title(
        title,
        fontsize=14,
        pad=20,
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.25, 1.1),
    )

    plt.tight_layout()
    plt.show()