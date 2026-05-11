# evaluation/plots.py

import numpy as np
import matplotlib.pyplot as plt


def set_experience_xticks(num_experiences):
    plt.xticks(
        ticks=range(num_experiences),
        labels=[f"Exp {i + 1}" for i in range(num_experiences)],
    )


def to_percent(values):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return values

    valid = values[~np.isnan(values)]

    if valid.size > 0 and np.nanmax(np.abs(valid)) <= 1.0:
        values = values * 100

    return values


def plot_method_curves(all_results, metric_key):
    markers = ["o", "s", "^", "D", "x", "*", "P", "X", "v", "<", ">"]
    linestyles = ["-", "--", "-.", ":"]

    for idx, (name, result) in enumerate(all_results.items()):
        values = to_percent(result[metric_key])
        x = np.arange(len(values))
        valid = ~np.isnan(values)

        plt.plot(
            x[valid],
            values[valid],
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
    R = np.asarray(R, dtype=float)
    R_percent = R * 100

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(R_percent, aspect="auto", vmin=0, vmax=100)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy (%)")

    ax.set_xlabel("Evaluated Experience")
    ax.set_ylabel("Training Experience")
    ax.set_title(title)

    ax.set_xticks(range(R.shape[1]))
    ax.set_yticks(range(R.shape[0]))

    ax.set_xticklabels([f"Exp {i + 1}" for i in range(R.shape[1])])
    ax.set_yticklabels([f"Exp {i + 1}" for i in range(R.shape[0])])

    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            val = R_percent[i, j]

            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if val < 50 else "black",
                    fontsize=9,
                )

    plt.tight_layout()
    plt.show()


def plot_accuracy_by_experience(R):
    R = np.asarray(R, dtype=float)
    T = R.shape[0]

    plt.figure(figsize=(8, 5))

    for j in range(T):
        values = R[:, j] * 100
        x = np.arange(T)
        valid = ~np.isnan(values)

        plt.plot(
            x[valid],
            values[valid],
            marker="o",
            label=f"Experience {j + 1}",
        )

    set_experience_xticks(T)
    plt.xlabel("Training Experience")
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Experience Accuracy")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_three_protocols(task_accs, growing_accs, fixed_accs):
    task_accs = to_percent(task_accs)
    growing_accs = to_percent(growing_accs)
    fixed_accs = to_percent(fixed_accs)

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
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ===============================
# MULTI-APPROACH COMPARISON
# ===============================

def plot_metric_comparison(
    all_results,
    metric_names=None,
    title="Summary Metric Comparison",
):
    if metric_names is None:
        metric_names = {
            "A Metric": "A ↑",
            "ACC Metric": "ACC ↑",
            "BWT (Backward Transfer)": "BWT ↑",
            "BWT+ (Positive Backward Transfer)": "BWT+ ↑",
            "FWT (Forward Transfer)": "FWT ↑",
            "Forgetting": "FGT ↓",
            "REM (Remembering)": "REM ↑",
            "MS (Memory Score)": "MS ↑",
            "SSS (Sample Storage Score)": "SSS ↑",
            "CE (Compute Efficiency)": "CE ↑",
        }

    if isinstance(metric_names, dict):
        metric_keys = list(metric_names.keys())
        labels = list(metric_names.values())
    else:
        metric_keys = metric_names
        labels = metric_names

    # Metrics where lower is better
    lower_is_better = {
        "Forgetting",
    }

    methods = list(all_results.keys())

    raw = np.array(
        [
            [
                all_results[m]["metrics"].get(metric, np.nan)
                for m in methods
            ]
            for metric in metric_keys
        ],
        dtype=float,
    )

    display = raw.copy()

    # Convert 0-1 range metrics to percentages
    for i in range(display.shape[0]):
        valid = display[i][~np.isnan(display[i])]

        if (
            valid.size > 0
            and np.nanmax(np.abs(valid)) <= 1.0
        ):
            display[i] *= 100

    # Row-wise normalization
    normalized = np.full_like(
        display,
        np.nan,
        dtype=float,
    )

    for i, metric in enumerate(metric_keys):
        row = display[i]
        valid = row[~np.isnan(row)]

        if valid.size == 0:
            continue

        min_val = np.min(valid)
        max_val = np.max(valid)

        if max_val == min_val:
            normalized[i] = 0.5
        else:
            normalized[i] = (
                (row - min_val)
                / (max_val - min_val)
            )

        # Lower forgetting is better
        if metric in lower_is_better:
            normalized[i] = 1.0 - normalized[i]

    fig, ax = plt.subplots(figsize=(12, 6))

    masked = np.ma.masked_invalid(normalized)

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#F0F0F0")

    im = ax.imshow(
        masked,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap=cmap,
    )

    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(metric_keys)))

    ax.set_xticklabels(
        methods,
        rotation=25,
        ha="right",
        fontsize=10,
    )

    ax.set_yticklabels(
        labels,
        fontsize=10,
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    ax.set_xlabel(
        "Approaches",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_ylabel(
        "Metrics",
        fontsize=11,
        fontweight="bold",
    )

    # Add values inside cells
    for i in range(len(metric_keys)):
        for j in range(len(methods)):
            value = display[i, j]
            score = normalized[i, j]

            text = (
                "nan"
                if np.isnan(value)
                else f"{value:.2f}"
            )

            if np.isnan(score):
                text_color = "black"
            elif score > 0.75:
                text_color = "white"
            else:
                text_color = "black"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

    cbar = plt.colorbar(im, ax=ax)

    cbar.set_label(
        "Relative Metric Performance",
        fontsize=10,
        fontweight="bold",
    )

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
    plt.grid(True, alpha=0.4)
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
    plt.grid(True, alpha=0.4)
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
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_radar_metrics(
    all_results,
    metric_names=None,
    title="Radar Metric Comparison",
):
    if metric_names is None:
        metric_names = {
            "A Metric": "A",
            "ACC Metric": "ACC",
            "BWT (Backward Transfer)": "BWT",
            "BWT+ (Positive Backward Transfer)": "BWT+",
            "FWT (Forward Transfer)": "FWT",
            "Forgetting": "FGT",
            "REM (Remembering)": "REM",
            "MS (Memory Score)": "MS",
            "SSS (Sample Storage Score)": "SSS",
            "CE (Compute Efficiency)": "CE",
        }

    if isinstance(metric_names, dict):
        metric_keys = list(metric_names.keys())
        labels = list(metric_names.values())
    else:
        metric_keys = metric_names
        labels = metric_names

    methods = list(all_results.keys())

    raw = np.array(
        [
            [
                all_results[m]["metrics"].get(metric, np.nan)
                for metric in metric_keys
            ]
            for m in methods
        ],
        dtype=float,
    )

    display = raw.copy()

    for j in range(display.shape[1]):
        valid = display[:, j][~np.isnan(display[:, j])]

        if valid.size > 0 and np.nanmax(np.abs(valid)) <= 1.0:
            display[:, j] *= 100

    normalized = np.full_like(display, np.nan, dtype=float)

    for j, metric in enumerate(metric_keys):
        col = display[:, j]
        valid = col[~np.isnan(col)]

        if valid.size == 0:
            continue

        min_val = np.min(valid)
        max_val = np.max(valid)

        if max_val == min_val:
            normalized[:, j] = 0.5
        else:
            normalized[:, j] = (col - min_val) / (max_val - min_val)

        if metric == "Forgetting":
            normalized[:, j] = 1.0 - normalized[:, j]

    normalized = np.nan_to_num(normalized, nan=0.0)

    num_vars = len(labels)

    angles = np.linspace(
        0,
        2 * np.pi,
        num_vars,
        endpoint=False,
    ).tolist()

    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for idx, method_name in enumerate(methods):
        values = normalized[idx].tolist()
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

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])

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