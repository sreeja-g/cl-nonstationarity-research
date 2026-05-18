import numpy as np


def A(R):
    """
    Final Average Accuracy across all tasks.

    A = average accuracy after learning all experiences.
    Uses final row of R.
    """

    final_row = R[-1, :]
    values = final_row[~np.isnan(final_row)]

    return float(np.mean(values)) if values.size > 0 else np.nan


def ACC(R):
    """
    Average Continual Accuracy.

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
    Backward Transfer (BWT)

    R[i, j] = performance on task j after learning task i

    BWT compares:
        final performance on old tasks
        vs
        performance immediately after learning them

    Negative -> forgetting
    Positive -> positive backward transfer
    """

    N = R.shape[0]
    values = []

    final_row = N - 1

    for j in range(N - 1):
        if not np.isnan(R[final_row, j]) and not np.isnan(R[j, j]):
            values.append(R[final_row, j] - R[j, j])

    return float(np.mean(values)) if values else np.nan


def BWT_plus(R):
    """
    Positive Backward Transfer (BWT+)

    BWT+ = max(BWT, 0)

    Measures beneficial backward transfer only.
    """

    bwt = BWT(R)

    if np.isnan(bwt):
        return np.nan

    return float(max(bwt, 0.0))


def FWT(R, baseline=None):
    """
    Forward Transfer (FWT)

    R[i, j] = performance on task j after learning task i

    FWT for task j:
        R[j-1, j] - baseline[j]

    baseline[j]:
        performance on task j before learning anything
        random-init / untrained model performance
    """

    N = R.shape[0]
    values = []

    if baseline is None:
        return np.nan

    baseline = np.asarray(baseline, dtype=float)

    if baseline.size != N:
        raise ValueError("baseline must have one value per task")

    for j in range(1, N):
        if not np.isnan(R[j - 1, j]) and not np.isnan(baseline[j]):
            values.append(R[j - 1, j] - baseline[j])

    return float(np.mean(values)) if values else np.nan


def forgetting(R):
    """
    Forgetting (FG)

    Standard forgetting:
        average drop from best past performance
        to final performance.

    FG_j = max_k(R[k, j]) - R[N-1, j]
    """

    N = R.shape[0]
    values = []

    final_row = N - 1

    for j in range(N - 1):
        past_scores = R[j:final_row + 1, j]
        past_scores = past_scores[~np.isnan(past_scores)]

        if past_scores.size == 0:
            continue

        best_score = np.max(past_scores)
        final_score = R[final_row, j]

        if not np.isnan(final_score):
            values.append(best_score - final_score)

    return float(np.mean(values)) if values else np.nan


def REM(R):
    """
    Remembering Score (REM)

    REM = 1 - Forgetting

    Higher is better.
    """

    fg = forgetting(R)

    if np.isnan(fg):
        return np.nan

    return float(1.0 - fg)


def M(model):
    """
    Raw model size.

    Returns total number of trainable and non-trainable parameters.
    """

    return int(sum(p.numel() for p in model.parameters()))


def memory_score(model_sizes):
    """
    Memory Score (MS)

    MS = min(1, average(M_1 / M_i))

    M_1:
        baseline model size

    M_i:
        model size during continual learning

    Higher is better:
        1 -> no model growth
        lower values -> larger model memory overhead
    """

    model_sizes = np.asarray(model_sizes, dtype=float)

    if model_sizes.size == 0:
        return np.nan

    if np.any(model_sizes <= 0):
        raise ValueError("model_sizes must be positive")

    first_size = model_sizes[0]
    ratios = first_size / model_sizes

    return float(min(1.0, np.mean(ratios)))


def sample_storage_score(buffer_sizes, lifetime_data_size):
    """
    Sample Storage Score (SSS)

    SSS = 1 - min(1, average(buffer_i / lifetime_data_size))

    Higher is better:
        1 -> minimal replay storage
        0 -> replay storage equals/exceeds lifetime dataset size
    """

    buffer_sizes = np.asarray(buffer_sizes, dtype=float)

    if buffer_sizes.size == 0:
        return np.nan

    if lifetime_data_size is None or lifetime_data_size <= 0:
        return np.nan

    if np.any(buffer_sizes < 0):
        raise ValueError("buffer_sizes must be non-negative")

    #ratio = np.mean(buffer_sizes) / float(lifetime_data_size)
    ratio = np.max(buffer_sizes) / lifetime_data_size

    return float(1.0 - min(1.0, ratio))


def compute_efficiency_from_ops(forward_backward_ops, total_train_ops):
    """
    Compute Efficiency (CE)

    CE = min(1, average(forward_backward_ops_i / total_train_ops_i))

    forward_backward_ops_i:
        standard forward-backward training cost

    total_train_ops_i:
        total method cost including replay, routing,
        regularization, memory updates, etc.

    Higher is better:
        1 -> no additional overhead
        lower values -> higher computational overhead
    """

    forward_backward_ops = np.asarray(forward_backward_ops, dtype=float)
    total_train_ops = np.asarray(total_train_ops, dtype=float)

    if forward_backward_ops.size == 0:
        return np.nan

    if forward_backward_ops.size != total_train_ops.size:
        raise ValueError(
            "forward_backward_ops and total_train_ops must have same length"
        )

    if np.any(forward_backward_ops < 0):
        raise ValueError("forward_backward_ops must be non-negative")

    if np.any(total_train_ops <= 0):
        raise ValueError("total_train_ops must be positive")

    ratios = forward_backward_ops / total_train_ops

    return float(min(1.0, np.mean(ratios)))


def compute_summary_metrics(
    R,
    model,
    train_times,
    model_sizes,
    buffer_sizes,
    lifetime_data_size,
    forward_backward_ops,
    total_train_ops,
    baseline=None,
):
    """
    Compute all summary metrics for continual learning experiments.
    """

    R = np.asarray(R, dtype=float)

    if len(train_times) != R.shape[0]:
        raise ValueError("train_times must have one value per experience")

    if len(model_sizes) != R.shape[0]:
        raise ValueError("model_sizes must have one value per experience")

    if len(buffer_sizes) != R.shape[0]:
        raise ValueError("buffer_sizes must have one value per experience")

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
        "FWT (Forward Transfer)": FWT(R, baseline),
        "Forgetting": forgetting(R),
        "REM (Remembering)": REM(R),
        "MS (Memory Score)": memory_score(model_sizes),
        "SSS (Sample Storage Score)": sample_storage_score(
            buffer_sizes,
            lifetime_data_size,
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