import re
import numpy as np
import matplotlib.pyplot as plt


def _sorted_cols_by_exp(cols):
    def extract_exp_num(col):
        match = re.search(r"Exp(\d+)", col)
        return int(match.group(1)) if match else float("inf")

    return sorted(cols, key=extract_exp_num)


def _get_exp_columns(results_df, prefix):
    cols = [c for c in results_df.columns if c.startswith(prefix)]
    return _sorted_cols_by_exp(cols)


def _exp_number_from_col(col):
    match = re.search(r"Exp(\d+)", col)
    return int(match.group(1)) if match else None


def _last_valid_value(series):
    vals = series.dropna()
    return vals.iloc[-1] if len(vals) > 0 else np.nan


def _first_valid_value(series):
    vals = series.dropna()
    return vals.iloc[0] if len(vals) > 0 else np.nan


# =========================================================
# SINGLE-RUN SUMMARY METRICS
# =========================================================

def compute_single_run_plot_metrics(results_df):
    """
    Build one compact metric dictionary for a single run.
    Good for bar plots and spider plots.
    """
    metrics = {}

    if "test_acc_stream" in results_df.columns:
        metrics["Stream Accuracy"] = _last_valid_value(results_df["test_acc_stream"])

    if "test_loss_stream" in results_df.columns:
        metrics["Stream Loss"] = _last_valid_value(results_df["test_loss_stream"])

    if "BWT_stream" in results_df.columns:
        metrics["Stream BWT"] = _last_valid_value(results_df["BWT_stream"])

    if "FWT_stream" in results_df.columns:
        metrics["Stream FWT"] = _last_valid_value(results_df["FWT_stream"])

    if "Forgetting_stream" in results_df.columns:
        metrics["Stream Forgetting"] = _last_valid_value(results_df["Forgetting_stream"])

    acc_cols = _get_exp_columns(results_df, "test_acc_Exp")
    if acc_cols:
        final_accs = []
        for col in acc_cols:
            val = _last_valid_value(results_df[col])
            if not np.isnan(val):
                final_accs.append(val)
        if final_accs:
            metrics["Avg Task Accuracy"] = float(np.mean(final_accs))

    forgetting_cols = _get_exp_columns(results_df, "Forgetting_Exp")
    if forgetting_cols:
        final_forgetting = []
        for col in forgetting_cols:
            val = _last_valid_value(results_df[col])
            if not np.isnan(val):
                final_forgetting.append(val)
        if final_forgetting:
            metrics["Avg Task Forgetting"] = float(np.mean(final_forgetting))

    bwt_cols = _get_exp_columns(results_df, "BWT_Exp")
    if bwt_cols:
        final_bwt = []
        for col in bwt_cols:
            val = _last_valid_value(results_df[col])
            if not np.isnan(val):
                final_bwt.append(val)
        if final_bwt:
            metrics["Avg Task BWT"] = float(np.mean(final_bwt))

    fwt_cols = _get_exp_columns(results_df, "FWT_Exp")
    if fwt_cols:
        useful_fwt = []
        for col in fwt_cols:
            val = _first_valid_value(results_df[col])
            if not np.isnan(val):
                useful_fwt.append(val)
        if useful_fwt:
            metrics["Avg Task FWT"] = float(np.mean(useful_fwt))

    if "train_time_epoch" in results_df.columns:
        vals = results_df["train_time_epoch"].dropna()
        if len(vals) > 0:
            metrics["Avg Epoch Time"] = float(vals.mean())

    return metrics


def normalize_single_run_metrics(metric_dict):
    """
    Normalize metrics to [0, 1] for spider plot and normalized bar plot.

    Higher is better:
    - accuracy
    - BWT
    - FWT

    Lower is better:
    - forgetting
    - loss
    - time
    """
    normalized = {}

    for name, value in metric_dict.items():
        if np.isnan(value):
            normalized[name] = np.nan
            continue

        lname = name.lower()

        if "accuracy" in lname:
            normalized[name] = np.clip(value, 0.0, 1.0)

        elif "bwt" in lname or "fwt" in lname:
            clipped = np.clip(value, -1.0, 1.0)
            normalized[name] = (clipped + 1.0) / 2.0

        elif "forgetting" in lname or "loss" in lname:
            clipped = np.clip(value, 0.0, 1.0)
            normalized[name] = 1.0 - clipped

        elif "time" in lname:
            normalized[name] = 1.0 / (1.0 + max(value, 0.0))

        else:
            normalized[name] = value

    return normalized


# =========================================================
# PER TEST SET: Accuracy + Forgetting
# =========================================================

def plot_accuracy_forgetting_per_testset(
    results_df,
    acc_ylim=(0.0, 1.0),
    forgetting_ylim=None,
    figsize=(8, 5),
):
    """
    One plot per test set:
    - X-axis: training experience
    - Left Y-axis: accuracy
    - Right Y-axis: forgetting
    """

    acc_cols = _get_exp_columns(results_df, "test_acc_Exp")
    forgetting_cols = _get_exp_columns(results_df, "Forgetting_Exp")

    if not acc_cols:
        print("No accuracy columns found.")
        return

    if forgetting_ylim is None:
        all_vals = []
        for col in forgetting_cols:
            vals = results_df[col].dropna().values
            all_vals.extend(vals.tolist())

        if all_vals:
            fmin, fmax = min(all_vals), max(all_vals)
            pad = 0.05 * (fmax - fmin + 1e-8)
            forgetting_ylim = (fmin - pad, fmax + pad)
        else:
            forgetting_ylim = (0.0, 1.0)

    x = results_df["experience"].values

    for acc_col in acc_cols:
        exp_id = _exp_number_from_col(acc_col)
        forgetting_col = f"Forgetting_Exp{exp_id:03d}"

        acc_y = results_df[acc_col].values
        forgetting_y = (
            results_df[forgetting_col].values
            if forgetting_col in results_df.columns
            else np.full_like(acc_y, np.nan)
        )

        fig, ax1 = plt.subplots(figsize=figsize)

        l1 = ax1.plot(x, acc_y, marker="o", label="Accuracy")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(acc_ylim)
        ax1.set_xlabel("Training Experience")
        ax1.set_xticks(x)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        l2 = ax2.plot(x, forgetting_y, marker="s", linestyle="--", label="Forgetting")
        ax2.set_ylabel("Forgetting")
        ax2.set_ylim(forgetting_ylim)

        plt.title(f"Test Set T{exp_id + 1}")

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels)

        plt.tight_layout()
        plt.show()


# =========================================================
# ALL ACCURACY CURVES
# =========================================================

def plot_all_accuracy(results_df, ylim=(0.0, 1.0), figsize=(8, 5)):
    acc_cols = _get_exp_columns(results_df, "test_acc_Exp")
    x = results_df["experience"].values

    plt.figure(figsize=figsize)

    for col in acc_cols:
        plt.plot(x, results_df[col].values, marker="o", label=col)

    if "test_acc_stream" in results_df.columns:
        plt.plot(
            x,
            results_df["test_acc_stream"].values,
            linestyle="--",
            linewidth=2,
            label="Overall"
        )

    plt.title("Test Accuracy over Experiences")
    plt.xlabel("Experience")
    plt.ylabel("Accuracy")
    plt.ylim(ylim)
    plt.xticks(x)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# ALL FORGETTING CURVES
# =========================================================

def plot_all_forgetting(results_df, figsize=(8, 5)):
    forgetting_cols = _get_exp_columns(results_df, "Forgetting_Exp")
    x = results_df["experience"].values

    plt.figure(figsize=figsize)

    for col in forgetting_cols:
        plt.plot(x, results_df[col].values, marker="s", linestyle="--", label=col)

    if "Forgetting_stream" in results_df.columns:
        plt.plot(
            x,
            results_df["Forgetting_stream"].values,
            linestyle="-.",
            linewidth=2,
            label="Forgetting_stream"
        )

    plt.title("Forgetting over Experiences")
    plt.xlabel("Experience")
    plt.ylabel("Forgetting")
    plt.xticks(x)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# SINGLE MODEL SUMMARY BAR PLOT
# =========================================================

def plot_single_model_summary_bars(
    results_df,
    title="Single Model Final Metric Summary",
    figsize=(10, 6),
    normalize_for_display=False,
):
    """
    Bar plot for one model/run showing the main final metrics.
    """
    metric_dict = compute_single_run_plot_metrics(results_df)

    if not metric_dict:
        print("No metrics found to plot.")
        return

    if normalize_for_display:
        plot_dict = normalize_single_run_metrics(metric_dict)
        ylabel = "Normalized Score"
    else:
        plot_dict = metric_dict
        ylabel = "Metric Value"

    names = list(plot_dict.keys())
    values = list(plot_dict.values())

    print("\n================ SINGLE MODEL BAR PLOT METRICS ================\n")
    for k, v in plot_dict.items():
        print(f"{k}: {v:.6f}")

    plt.figure(figsize=figsize)
    plt.bar(names, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================================================
# SINGLE MODEL SPIDER / RADAR PLOT
# =========================================================

def plot_single_model_spider(
    results_df,
    title="Single Model Spider Plot",
    figsize=(8, 8),
):
    """
    Spider/radar plot for one model.
    Uses normalized metrics so all metrics fit on one common scale.
    """
    raw_metrics = compute_single_run_plot_metrics(results_df)

    if not raw_metrics:
        print("No metrics found to plot.")
        return

    metrics = normalize_single_run_metrics(raw_metrics)

    names = list(metrics.keys())
    values = list(metrics.values())

    filtered = [(n, v) for n, v in zip(names, values) if not np.isnan(v)]
    if not filtered:
        print("All metrics are NaN. Nothing to plot.")
        return

    names = [x[0] for x in filtered]
    values = [x[1] for x in filtered]

    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]

    print("\n================ SINGLE MODEL SPIDER PLOT METRICS ================\n")
    print("Spider plot uses normalized scores so all metrics fit on one common scale.")
    print("For loss / forgetting / time, lower raw values become higher normalized scores.\n")
    for k, v in zip(names, values[:-1]):
        print(f"{k}: {v:.6f}")

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)

    plt.title(title, y=1.08)
    plt.tight_layout()
    plt.show()


# =========================================================
# MAIN ENTRY FUNCTIONS
# =========================================================

def plot_single_experiment(results_df):
    """
    Your original single-experiment plotting flow plus summary plots.
    """
    print("\n================ PLOTTING RESULTS ================\n")

    print("1. All accuracy curves")
    plot_all_accuracy(results_df)

    print("2. All forgetting curves")
    plot_all_forgetting(results_df)

    print("3. Per-task accuracy + forgetting plots")
    plot_accuracy_forgetting_per_testset(results_df)

    print("4. Raw summary bar plot")
    plot_single_model_summary_bars(
        results_df,
        title="Single Model Final Metric Summary (Raw Values)",
        normalize_for_display=False,
    )

    print("5. Normalized summary bar plot")
    plot_single_model_summary_bars(
        results_df,
        title="Single Model Final Metric Summary (Normalized)",
        normalize_for_display=True,
    )

    print("6. Spider plot")
    plot_single_model_spider(
        results_df,
        title="Single Model Spider Plot",
    )

    print("\nDone.\n")


def plot_single_model_summary(results_df):
    """
    Only the summary views for one model:
    - raw bar plot
    - normalized bar plot
    - spider plot
    """
    print("\n================ PLOTTING SINGLE MODEL SUMMARY ================\n")

    print("1. Raw summary bar plot")
    plot_single_model_summary_bars(
        results_df,
        title="Single Model Final Metric Summary (Raw Values)",
        normalize_for_display=False,
    )

    print("2. Normalized summary bar plot")
    plot_single_model_summary_bars(
        results_df,
        title="Single Model Final Metric Summary (Normalized)",
        normalize_for_display=True,
    )

    print("3. Spider plot")
    plot_single_model_spider(
        results_df,
        title="Single Model Spider Plot",
    )

    print("\nDone.\n")