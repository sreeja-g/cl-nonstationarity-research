import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_metrics(res, exp_id):
    """
    Convert one Avalanche result dictionary into one clean row.
    """
    row = {"experience": exp_id}

    for k, v in res.items():
        # =====================================================
        # Accuracy
        # =====================================================
        if "Top1_Acc_Exp/eval_phase/test_stream" in k:
            exp_name = k.split("/")[-1]   # Exp000, Exp001, ...
            row[f"test_acc_{exp_name}"] = v
        elif "Top1_Acc_Stream/eval_phase/test_stream" in k:
            row["test_acc_stream"] = v

        # =====================================================
        # Loss
        # =====================================================
        elif "Loss_Exp/eval_phase/test_stream" in k:
            exp_name = k.split("/")[-1]
            row[f"test_loss_{exp_name}"] = v
        elif "Loss_Stream/eval_phase/test_stream" in k:
            row["test_loss_stream"] = v

        # =====================================================
        # Timing
        # =====================================================
        elif "Time_Epoch/train_phase/train_stream" in k:
            row["train_time_epoch"] = v

        # =====================================================
        # CPU Usage
        # =====================================================
        elif "CPUUsage_Exp/eval_phase/test_stream" in k:
            exp_name = k.split("/")[-1]
            row[f"cpu_usage_{exp_name}"] = v

        # =====================================================
        # Disk Usage
        # =====================================================
        elif "DiskUsage_Exp/eval_phase/test_stream" in k:
            exp_name = k.split("/")[-1]
            row[f"test_disk_{exp_name}"] = v
        elif "DiskUsage_Stream/eval_phase/test_stream" in k:
            row["test_disk_stream"] = v

        # =====================================================
        # Continual Learning Metrics
        # =====================================================
        elif "StreamBWT/eval_phase/test_stream" in k:
            row["BWT_stream"] = v
        elif "ExperienceBWT/eval_phase/test_stream" in k:
            exp_name = k.split("/")[-1]
            row[f"BWT_{exp_name}"] = v

        elif "StreamForwardTransfer/eval_phase/test_stream" in k:
            row["FWT_stream"] = v
        elif "ExperienceForwardTransfer/eval_phase/test_stream" in k:
            exp_name = k.split("/")[-1]
            row[f"FWT_{exp_name}"] = v

        elif "StreamForgetting/eval_phase/test_stream" in k:
            row["Forgetting_stream"] = v
        elif "ExperienceForgetting/eval_phase/test_stream" in k:
            exp_name = k.split("/")[-1]
            row[f"Forgetting_{exp_name}"] = v

        # =====================================================
        # Confusion Matrix
        # =====================================================
        elif "ConfusionMatrix_Stream/eval_phase/test_stream" in k:
            row["confusion_matrix"] = v

    return row


def build_results_df(results):
    """
    results: list of Avalanche result dictionaries
    """
    clean_results = [extract_metrics(res, i) for i, res in enumerate(results)]
    results_df = pd.DataFrame(clean_results)
    return results_df


def get_metric_groups(results_df):
    """
    Return grouped columns for easier display.
    """
    metric_groups = {
        "test_acc_exp": [c for c in results_df.columns if c.startswith("test_acc_Exp")],
        "test_loss_exp": [c for c in results_df.columns if c.startswith("test_loss_Exp")],
        "bwt_exp": [c for c in results_df.columns if c.startswith("BWT_Exp")],
        "fwt_exp": [c for c in results_df.columns if c.startswith("FWT_Exp")],
        "forgetting_exp": [c for c in results_df.columns if c.startswith("Forgetting_Exp")],
        "stream_metrics": [c for c in results_df.columns if c.endswith("_stream")],
        "other_metrics": [c for c in results_df.columns if c in ["train_time_epoch"]],
        "resource_metrics": (
            [c for c in results_df.columns if c.startswith("cpu_usage_Exp")] +
            [c for c in results_df.columns if c.startswith("test_disk_Exp")]
        ),
    }
    return metric_groups


def print_basic_explanations():
    print("\n================ METRIC MEANINGS ================\n")
    print("Experience-level metrics (Exp):")
    print("  These show task-wise performance over time.")
    print("  Example: test_acc_Exp002 means accuracy on task/experience 2.")
    print()
    print("Stream-level metrics (_stream):")
    print("  These summarize overall performance across the full test stream.")
    print("  These are usually the most useful final comparison numbers.")
    print()
    print("Important continual learning metrics:")
    print("  Accuracy      -> how well the model performs")
    print("  Loss          -> prediction error")
    print("  Forgetting    -> how much old tasks degraded after learning new ones")
    print("  BWT           -> backward transfer from learning later tasks")
    print("  FWT           -> forward transfer before a task is learned")
    print()


def print_fwt_summary(results_df, fwt_cols):
    print("\n================ FORWARD TRANSFER SUMMARY ================\n")
    print("Forward transfer for a task is measured only once: right before that task is trained.")
    print("So for FWT_Exp00k, the meaningful value is the first non-NaN entry in that column.\n")

    useful_vals = []

    for col in fwt_cols:
        task_id = int(col.split("Exp")[-1])
        first_valid_idx = results_df[col].first_valid_index()

        if first_valid_idx is not None:
            val = results_df.loc[first_valid_idx, col]
            useful_vals.append(val)

            print(
                f"Task {task_id}: FWT = {val:.6f} "
                f"(measured after Experience {first_valid_idx}, before training Task {task_id})"
            )

            if val > 0:
                print(f"  Interpretation: previous tasks helped the model on Task {task_id} before it was learned.\n")
            elif val < 0:
                print(f"  Interpretation: previous tasks hurt the model on Task {task_id} before it was learned.\n")
            else:
                print(f"  Interpretation: previous tasks had no effect on Task {task_id} before it was learned.\n")

    if useful_vals:
        print(f"Average useful FWT across tasks: {np.mean(useful_vals):.6f}\n")


def display_metric_tables(results_df, display_fn=None):
    """
    display_fn:
      - pass display in notebook
      - leave as None to use print
    """
    if display_fn is None:
        def display_fn(x):
            print(x)

    metric_groups = get_metric_groups(results_df)

    for group_name, cols in metric_groups.items():
        if not cols:
            continue

        if group_name == "fwt_exp":
            print_fwt_summary(results_df, cols)
            continue

        print(f"\n================ {group_name.upper()} MATRIX ================\n")

        if group_name == "test_acc_exp":
            print("Interpretation: rows = training experience completed, columns = evaluation task.")
            print("Higher values are better. Diagonal values often show current-task performance.")
            print("Older-task drops across later rows suggest forgetting.\n")

        elif group_name == "test_loss_exp":
            print("Interpretation: lower values are better.")
            print("If loss increases for earlier tasks after later training, that suggests forgetting.\n")

        elif group_name == "bwt_exp":
            print("Interpretation: backward transfer after learning future tasks.")
            print("Positive BWT means later learning helped earlier tasks.")
            print("Negative BWT often indicates forgetting.\n")

        elif group_name == "forgetting_exp":
            print("Interpretation: how much each old task performance dropped from its best past value.")
            print("Lower is better. Higher forgetting means worse retention.\n")

        elif group_name == "stream_metrics":
            print("Interpretation: overall metrics across the entire test stream.")
            print("These are the most important final summary metrics for comparing methods.\n")

        elif group_name == "other_metrics":
            print("Interpretation: training-time metadata such as epoch time.\n")

        elif group_name == "resource_metrics":
            print("Interpretation: system/resource tracking. Useful, but secondary for CL research.\n")

        subset = results_df[["experience"] + cols]
        display_fn(subset)


def _last_valid_value(series):
    vals = series.dropna()
    return vals.iloc[-1] if len(vals) > 0 else np.nan


def _first_valid_value(series):
    vals = series.dropna()
    return vals.iloc[0] if len(vals) > 0 else np.nan


def compute_average_final_accuracy_across_tasks(results_df):
    """
    Average of the final non-NaN value from each test_acc_Exp column.
    This is often a better task-level final accuracy summary than stream accuracy alone.
    """
    acc_cols = [c for c in results_df.columns if c.startswith("test_acc_Exp")]
    final_vals = []

    for col in acc_cols:
        val = _last_valid_value(results_df[col])
        if not pd.isna(val):
            final_vals.append(val)

    return float(np.mean(final_vals)) if final_vals else np.nan


def compute_average_final_forgetting_across_tasks(results_df):
    """
    Average of the final non-NaN value from each Forgetting_Exp column.
    Lower is better.
    """
    forgetting_cols = [c for c in results_df.columns if c.startswith("Forgetting_Exp")]
    final_vals = []

    for col in forgetting_cols:
        val = _last_valid_value(results_df[col])
        if not pd.isna(val):
            final_vals.append(val)

    return float(np.mean(final_vals)) if final_vals else np.nan


def compute_average_final_bwt_across_tasks(results_df):
    """
    Average of the final non-NaN value from each BWT_Exp column.
    """
    bwt_cols = [c for c in results_df.columns if c.startswith("BWT_Exp")]
    final_vals = []

    for col in bwt_cols:
        val = _last_valid_value(results_df[col])
        if not pd.isna(val):
            final_vals.append(val)

    return float(np.mean(final_vals)) if final_vals else np.nan


def compute_average_useful_fwt_across_tasks(results_df):
    """
    Average of the first meaningful non-NaN value from each FWT_Exp column.
    """
    fwt_cols = [c for c in results_df.columns if c.startswith("FWT_Exp")]
    useful_vals = []

    for col in fwt_cols:
        val = _first_valid_value(results_df[col])
        if not pd.isna(val):
            useful_vals.append(val)

    return float(np.mean(useful_vals)) if useful_vals else np.nan


def compute_single_experiment_summary(results_df):
    """
    Return a compact final summary for one run.
    """
    summary = {}

    if "test_acc_stream" in results_df.columns:
        summary["final_test_acc_stream"] = _last_valid_value(results_df["test_acc_stream"])

    if "test_loss_stream" in results_df.columns:
        summary["final_test_loss_stream"] = _last_valid_value(results_df["test_loss_stream"])

    if "BWT_stream" in results_df.columns:
        summary["final_BWT_stream"] = _last_valid_value(results_df["BWT_stream"])

    if "FWT_stream" in results_df.columns:
        summary["final_FWT_stream"] = _last_valid_value(results_df["FWT_stream"])

    if "Forgetting_stream" in results_df.columns:
        summary["final_Forgetting_stream"] = _last_valid_value(results_df["Forgetting_stream"])

    if "train_time_epoch" in results_df.columns:
        vals = results_df["train_time_epoch"].dropna()
        summary["avg_train_time_epoch"] = vals.mean() if len(vals) > 0 else np.nan

    summary["avg_final_accuracy_across_tasks"] = compute_average_final_accuracy_across_tasks(results_df)
    summary["avg_final_forgetting_across_tasks"] = compute_average_final_forgetting_across_tasks(results_df)
    summary["avg_final_bwt_across_tasks"] = compute_average_final_bwt_across_tasks(results_df)
    summary["avg_useful_FWT_from_exp_cols"] = compute_average_useful_fwt_across_tasks(results_df)

    return summary


def print_single_experiment_summary(results_df):
    print("\n================ FINAL SINGLE-EXPERIMENT SUMMARY ================\n")
    print("These are the main final numbers you will usually compare across models or methods.\n")

    summary = compute_single_experiment_summary(results_df)

    if "final_test_acc_stream" in summary:
        print(f"Final stream accuracy           : {summary['final_test_acc_stream']:.6f}")
        print("  Overall accuracy across the whole test stream after the final experience.\n")

    if "avg_final_accuracy_across_tasks" in summary:
        print(f"Average final task accuracy     : {summary['avg_final_accuracy_across_tasks']:.6f}")
        print("  Mean of the final accuracy values from all test_acc_Exp columns.")
        print("  This tells you how well the model performs across individual tasks at the end.\n")

    if "final_test_loss_stream" in summary:
        print(f"Final stream loss               : {summary['final_test_loss_stream']:.6f}")
        print("  Overall loss across the whole test stream after the final experience.\n")

    if "final_Forgetting_stream" in summary:
        print(f"Final stream forgetting         : {summary['final_Forgetting_stream']:.6f}")
        print("  Lower is better. High forgetting means old tasks were not retained well.\n")

    if "avg_final_forgetting_across_tasks" in summary:
        print(f"Average final task forgetting   : {summary['avg_final_forgetting_across_tasks']:.6f}")
        print("  Mean of the final forgetting values from all Forgetting_Exp columns.")
        print("  Lower is better. This is often more interpretable than only using stream forgetting.\n")

    if "final_BWT_stream" in summary:
        print(f"Final stream BWT                : {summary['final_BWT_stream']:.6f}")
        print("  Positive is good. Negative often indicates later learning hurt earlier tasks.\n")

    if "avg_final_bwt_across_tasks" in summary:
        print(f"Average final task BWT          : {summary['avg_final_bwt_across_tasks']:.6f}")
        print("  Mean of the final BWT values from all BWT_Exp columns.\n")

    if "final_FWT_stream" in summary:
        print(f"Final stream FWT                : {summary['final_FWT_stream']:.6f}")
        print("  Measures transfer to unseen tasks before they were trained.\n")

    if "avg_useful_FWT_from_exp_cols" in summary:
        print(f"Average task-wise FWT           : {summary['avg_useful_FWT_from_exp_cols']:.6f}")
        print("  This uses the first meaningful non-NaN value from each FWT_Exp column.\n")

    if "avg_train_time_epoch" in summary:
        print(f"Average epoch time              : {summary['avg_train_time_epoch']:.6f}")
        print("  Mean training time per epoch across experiences.\n")


def plot_last_confusion_matrix(results_df):
    if "confusion_matrix" not in results_df.columns:
        print("No confusion_matrix column found.")
        return

    valid_items = results_df["confusion_matrix"].dropna()
    if len(valid_items) == 0:
        print("No confusion matrix values found.")
        return

    idx = valid_items.index[-1]
    cm = np.array(valid_items.loc[idx])

    print("\n================ FINAL CONFUSION MATRIX ================\n")
    print(f"Showing confusion matrix after Experience {idx}.")
    print("Rows = true classes, columns = predicted classes.")
    print("Stronger diagonal values usually mean better classification.\n")

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - After Experience {idx}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.show()


def analyze_single_experiment(results, display_fn=None):
    """
    Full single-experiment analysis pipeline.

    Parameters
    ----------
    results : list
        List of Avalanche result dicts, usually one dict per experience eval.
    display_fn : callable
        Use display in notebook, or leave None to print tables.

    Example notebook usage
    ----------------------
    from IPython.display import display
    results_df = analyze_single_experiment(results, display_fn=display)

    Example script usage
    --------------------
    results_df = analyze_single_experiment(results)
    """
    print("\n========================================================")
    print("BUILDING SINGLE-EXPERIMENT ANALYSIS")
    print("========================================================\n")

    results_df = build_results_df(results)

    print("Created results dataframe.")
    print(f"Number of rows (experiences): {len(results_df)}")
    print(f"Number of columns (metrics):  {len(results_df.columns)}\n")

    print("Columns found:")
    for col in results_df.columns:
        print(f"  - {col}")

    print_basic_explanations()

    print("\n================ RAW RESULTS DATAFRAME ================\n")
    if display_fn is None:
        print(results_df)
    else:
        display_fn(results_df)

    display_metric_tables(results_df, display_fn=display_fn)
    print_single_experiment_summary(results_df)
    plot_last_confusion_matrix(results_df)

    return results_df