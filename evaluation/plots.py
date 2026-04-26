# evaluation/plots.py

import numpy as np
import matplotlib.pyplot as plt


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

    for name, result in all_results.items():
        plt.plot(
            np.array(result["fixed_accs"]) * 100,
            marker="o",
            label=name,
        )

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

    for name, result in all_results.items():
        plt.plot(
            np.array(result["growing_accs"]) * 100,
            marker="o",
            label=name,
        )

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

    for name, result in all_results.items():
        plt.plot(
            np.array(result["task_accs"]) * 100,
            marker="o",
            label=name,
        )

    plt.xlabel("Training Experience")
    plt.ylabel("Accuracy (%)")
    plt.title("Current Task Accuracy Comparison")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()