import numpy as np


def A(R):
    """
    Final Average Accuracy.

    A = average accuracy after learning all experiences.
    Uses final row of R.
    """

    final_row = R[-1, :]
    values = [v for v in final_row if not np.isnan(v)]

    return float(np.mean(values)) if values else np.nan


def ACC(R):
    """
    Accuracy over time.

    ACC = average of lower-triangle entries R[i, j], where i >= j.
    Measures performance throughout continual learning.
    """

    N = R.shape[0]
    values = []

    for i in range(N):
        for j in range(i + 1):
            if not np.isnan(R[i, j]):
                values.append(R[i, j])

    return float(np.mean(values)) if values else np.nan


def BWT(R):
    """
    Backward Transfer.

    BWT = average of R[i, j] - R[j, j] for i > j.
    Negative BWT means forgetting.
    Positive BWT means later learning helped earlier tasks.
    """

    N = R.shape[0]
    values = []

    for i in range(1, N):
        for j in range(i):
            if not np.isnan(R[i, j]) and not np.isnan(R[j, j]):
                values.append(R[i, j] - R[j, j])

    return float(np.mean(values)) if values else np.nan


def BWT_plus(R):
    """
    Positive Backward Transfer.

    BWT+ = max(BWT, 0)
    """

    bwt = BWT(R)

    if np.isnan(bwt):
        return np.nan

    return float(max(bwt, 0.0))


def FWT(R):
    """
    Forward Transfer.

    FWT = average of upper-triangle entries R[i, j], where i < j.

    Requires fixed evaluation protocol because future-task values are needed.
    """

    N = R.shape[0]
    values = []

    for i in range(N):
        for j in range(i + 1, N):
            if not np.isnan(R[i, j]):
                values.append(R[i, j])

    return float(np.mean(values)) if values else np.nan


def forgetting(R):
    """
    Forgetting.

    Forgetting = |min(BWT, 0)|

    If BWT is positive, forgetting is 0.
    """

    bwt = BWT(R)

    if np.isnan(bwt):
        return np.nan

    return float(abs(min(bwt, 0.0)))


def REM(R):
    """
    Remembering score.

    REM = 1 - |min(BWT, 0)|
    REM = 1 - Forgetting
    """

    fg = forgetting(R)

    if np.isnan(fg):
        return np.nan

    return float(1.0 - fg)


def M(model):
    """
    Raw model size: total number of parameters.
    """

    return int(sum(p.numel() for p in model.parameters()))


def memory_score(model_sizes):
    """
    MS = min(1, mean(M_1 / M_i))
    """

    model_sizes = np.array(model_sizes, dtype=float)

    if len(model_sizes) == 0:
        return np.nan

    first_size = model_sizes[0]

    if first_size <= 0:
        return np.nan

    ratios = first_size / model_sizes

    return float(min(1.0, np.mean(ratios)))


def sample_storage_score(memory_sizes, dataset_memory_size):
    """
    SSS = 1 - min(1, mean(memory_i / dataset_memory_size))
    """

    memory_sizes = np.array(memory_sizes, dtype=float)

    if len(memory_sizes) == 0:
        return np.nan

    if dataset_memory_size is None or dataset_memory_size <= 0:
        return np.nan

    ratio = np.mean(memory_sizes / dataset_memory_size)

    return float(1.0 - min(1.0, ratio))


def compute_efficiency_from_ops(forward_backward_ops, total_train_ops):
    """
    CE = min(1, mean(forward_backward_ops_i / total_train_ops_i))

    forward_backward_ops_i:
        estimated normal forward-backward cost

    total_train_ops_i:
        estimated full method cost including replay, routing,
        regularization, memory update, etc.
    """

    forward_backward_ops = np.array(forward_backward_ops, dtype=float)
    total_train_ops = np.array(total_train_ops, dtype=float)

    if len(forward_backward_ops) == 0:
        return np.nan

    if len(forward_backward_ops) != len(total_train_ops):
        raise ValueError(
            "forward_backward_ops and total_train_ops must have same length"
        )

    if np.any(total_train_ops <= 0):
        raise ValueError("total_train_ops must be positive")

    ratios = forward_backward_ops / total_train_ops

    return float(min(1.0, np.mean(ratios)))


def compute_summary_metrics(
    R,
    model,
    train_times,
    model_sizes,
    memory_sizes,
    dataset_memory_size,
    forward_backward_ops,
    total_train_ops,
):
    if len(train_times) != R.shape[0]:
        raise ValueError("train_times must have one value per experience")

    if len(model_sizes) != R.shape[0]:
        raise ValueError("model_sizes must have one value per experience")

    if len(memory_sizes) != R.shape[0]:
        raise ValueError("memory_sizes must have one value per experience")

    if len(forward_backward_ops) != R.shape[0]:
        raise ValueError("forward_backward_ops must have one value per experience")

    if len(total_train_ops) != R.shape[0]:
        raise ValueError("total_train_ops must have one value per experience")

    return {
        "A Metric": A(R),
        "ACC Metric": ACC(R),
        "Model Size": M(model),
        "BWT (Backward Transfer)": BWT(R),
        "BWT+ (Positive Backward Transfer)": BWT_plus(R),
        "FWT (Forward Transfer)": FWT(R),
        "Forgetting": forgetting(R),
        "REM (Remembering)": REM(R),
        "MS (Memory Score)": memory_score(model_sizes),
        "SSS (Sample Storage Score)": sample_storage_score(
            memory_sizes,
            dataset_memory_size,
        ),
        "CE (Compute Efficiency)": compute_efficiency_from_ops(
            forward_backward_ops,
            total_train_ops,
        ),
        "Total Train Time": float(np.sum(train_times)),
        "Average Train Time": float(np.mean(train_times)),
        "Estimated Baseline Ops": float(np.sum(forward_backward_ops)),
        "Estimated Total Ops": float(np.sum(total_train_ops)),
    }