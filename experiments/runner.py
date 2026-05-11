# evaluation/experiment_runner.py

import os
import time
import json
import numpy as np

from evaluation.result_matrix import ResultMatrix
from evaluation.continual_metrics import compute_summary_metrics


def _to_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _extract_metric(last_metrics, metric_name):
    matches = [
        v for k, v in last_metrics.items()
        if metric_name in k
    ]

    if len(matches) == 0:
        raise KeyError(f"Could not find metric: {metric_name}")

    return _to_float(matches[-1])


def get_lifetime_data_size(benchmark):
    return sum(len(exp.dataset) for exp in benchmark.train_stream)


def get_test_experience_sizes(benchmark):
    return np.asarray(
        [len(exp.dataset) for exp in benchmark.test_stream],
        dtype=float,
    )


def weighted_mean_accuracy(accs, sizes):
    accs = np.asarray(accs, dtype=float)
    sizes = np.asarray(sizes, dtype=float)

    valid = ~np.isnan(accs)

    accs = accs[valid]
    sizes = sizes[valid]

    if accs.size == 0 or np.sum(sizes) == 0:
        return np.nan

    return float(np.sum(accs * sizes) / np.sum(sizes))


def estimate_ops(
    train_size,
    model_size,
    replay_size=0,
    regularization_overhead=0.0,
    routing_overhead=0.0,
    memory_update_overhead=0.0,
    regularization_multiplier=None,
    routing_multiplier=None,
    memory_update_multiplier=None,
):
    """
    Estimate relative training compute.

    Supports both old names:
        regularization_multiplier, routing_multiplier, memory_update_multiplier

    and new names:
        regularization_overhead, routing_overhead, memory_update_overhead
    """

    if regularization_multiplier is not None:
        regularization_overhead = regularization_multiplier - 1.0

    if routing_multiplier is not None:
        routing_overhead = routing_multiplier - 1.0

    if memory_update_multiplier is not None:
        memory_update_overhead = memory_update_multiplier

    forward_backward_ops = train_size * model_size

    effective_train_size = train_size + replay_size

    base_training_cost = effective_train_size * model_size

    overhead_multiplier = (
        1.0
        + regularization_overhead
        + routing_overhead
    )

    total_train_ops = base_training_cost * overhead_multiplier

    total_train_ops += (
        memory_update_overhead
        * train_size
        * model_size
    )

    return float(forward_backward_ops), float(total_train_ops)


def run_continual_experiment(
    strategy,
    benchmark,
    output_dir="outputs/experiment",
    regularization_multiplier=1.0,
    routing_multiplier=1.0,
    memory_update_multiplier=0.0,
    baseline_accs=None,
    save_outputs=False,
):
    os.makedirs(output_dir, exist_ok=True)

    n_experiences = len(benchmark.train_stream)
    result_matrix = ResultMatrix(n_experiences)

    train_times = []
    model_sizes = []
    buffer_sizes = []

    forward_backward_ops = []
    total_train_ops = []

    task_accs = []
    growing_accs = []
    fixed_accs = []

    test_sizes = get_test_experience_sizes(benchmark)

    for train_exp_id, train_exp in enumerate(benchmark.train_stream):
        print("=" * 70)
        print(f"Training experience {train_exp_id + 1}/{n_experiences}")
        print("=" * 70)

        start_time = time.time()
        strategy.train(train_exp)
        train_times.append(time.time() - start_time)

        last_metrics = strategy.evaluator.get_last_metrics()

        model_size = int(_extract_metric(last_metrics, "ModelSizeMetric"))
        replay_size = int(_extract_metric(last_metrics, "ReplaySizeMetric"))

        model_sizes.append(model_size)
        buffer_sizes.append(replay_size)

        train_size = len(train_exp.dataset)

        fb_ops, method_ops = estimate_ops(
            train_size=train_size,
            model_size=model_size,
            replay_size=replay_size,
            regularization_multiplier=regularization_multiplier,
            routing_multiplier=routing_multiplier,
            memory_update_multiplier=memory_update_multiplier,
        )

        forward_backward_ops.append(fb_ops)
        total_train_ops.append(method_ops)

        print("Evaluating on full test stream")
        eval_results = strategy.eval(benchmark.test_stream)

        result_matrix.update(
            train_exp_id=train_exp_id,
            eval_results=eval_results,
        )

        R_partial = result_matrix.get()
        i = train_exp_id

        task_acc = float(R_partial[i, i])

        growing_acc = weighted_mean_accuracy(
            accs=R_partial[i, : i + 1],
            sizes=test_sizes[: i + 1],
        )

        fixed_acc = weighted_mean_accuracy(
            accs=R_partial[i, :],
            sizes=test_sizes,
        )

        task_accs.append(task_acc)
        growing_accs.append(growing_acc)
        fixed_accs.append(fixed_acc)

    R = result_matrix.get()

    lifetime_data_size = get_lifetime_data_size(benchmark)

    metrics = compute_summary_metrics(
        R=R,
        model=strategy.model,
        train_times=train_times,
        model_sizes=model_sizes,
        buffer_sizes=buffer_sizes,
        lifetime_data_size=lifetime_data_size,
        forward_backward_ops=forward_backward_ops,
        total_train_ops=total_train_ops,
        baseline=baseline_accs,
    )

    if save_outputs:
        np.save(os.path.join(output_dir, "accuracy_matrix.npy"), R)

        np.save(os.path.join(output_dir, "task_accs.npy"), np.asarray(task_accs))
        np.save(os.path.join(output_dir, "growing_accs.npy"), np.asarray(growing_accs))
        np.save(os.path.join(output_dir, "fixed_accs.npy"), np.asarray(fixed_accs))

        np.save(os.path.join(output_dir, "model_sizes.npy"), np.asarray(model_sizes))
        np.save(os.path.join(output_dir, "buffer_sizes.npy"), np.asarray(buffer_sizes))

        np.save(
            os.path.join(output_dir, "forward_backward_ops.npy"),
            np.asarray(forward_backward_ops),
        )

        np.save(
            os.path.join(output_dir, "total_train_ops.npy"),
            np.asarray(total_train_ops),
        )

        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    return {
        "R": R,
        "metrics": metrics,
        "train_times": train_times,
        "model_sizes": model_sizes,
        "buffer_sizes": buffer_sizes,
        "memory_sizes": buffer_sizes,  # optional backward compatibility
        "forward_backward_ops": forward_backward_ops,
        "total_train_ops": total_train_ops,
        "lifetime_data_size": lifetime_data_size,
        "dataset_memory_size": lifetime_data_size,  # optional backward compatibility
        "test_sizes": test_sizes,
        "task_accs": task_accs,
        "growing_accs": growing_accs,
        "fixed_accs": fixed_accs,
        "baseline_accs": baseline_accs,
    }


def run_joint_experiment(
    strategy,
    benchmark,
    output_dir="outputs/joint",
    save_outputs=False,
):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Training JointTraining on full train stream")
    print("=" * 70)

    start_time = time.time()
    strategy.train(benchmark.train_stream)
    train_time = time.time() - start_time

    print("Evaluating JointTraining on full test stream")
    eval_results = strategy.eval(benchmark.test_stream)

    last_metrics = strategy.evaluator.get_last_metrics()
    model_size = int(_extract_metric(last_metrics, "ModelSizeMetric"))

    stream_acc = None

    for key, value in eval_results.items():
        if "Top1_Acc_Stream" in key:
            stream_acc = _to_float(value)
            break

    if stream_acc is None:
        raise KeyError("Could not find Top1_Acc_Stream in eval_results")

    n_experiences = len(benchmark.test_stream)

    task_accs = [stream_acc] * n_experiences
    growing_accs = [stream_acc] * n_experiences
    fixed_accs = [stream_acc] * n_experiences

    train_times = [train_time]
    model_sizes = [model_size] * n_experiences
    buffer_sizes = [0] * n_experiences

    lifetime_data_size = get_lifetime_data_size(benchmark)

    forward_backward_ops = [
        lifetime_data_size * model_size
    ]

    total_train_ops = [
        lifetime_data_size * model_size
    ]

    metrics = {
        "A Metric": stream_acc,
        "ACC Metric": np.nan,
        "Model Size": model_size,
        "BWT (Backward Transfer)": np.nan,
        "BWT+ (Positive Backward Transfer)": np.nan,
        "FWT (Forward Transfer)": np.nan,
        "Forgetting": np.nan,
        "REM (Remembering)": np.nan,
        "MS (Memory Score)": 1.0,
        "SSS (Sample Storage Score)": 1.0,
        "CE (Compute Efficiency)": 1.0,
        "Total Train Time": float(train_time),
        "Average Train Time": float(train_time),
        "Estimated Baseline Ops": float(np.sum(forward_backward_ops)),
        "Estimated Total Ops": float(np.sum(total_train_ops)),
    }

    if save_outputs:
        np.save(os.path.join(output_dir, "task_accs.npy"), np.asarray(task_accs))
        np.save(os.path.join(output_dir, "growing_accs.npy"), np.asarray(growing_accs))
        np.save(os.path.join(output_dir, "fixed_accs.npy"), np.asarray(fixed_accs))

        np.save(os.path.join(output_dir, "model_sizes.npy"), np.asarray(model_sizes))
        np.save(os.path.join(output_dir, "buffer_sizes.npy"), np.asarray(buffer_sizes))

        np.save(
            os.path.join(output_dir, "forward_backward_ops.npy"),
            np.asarray(forward_backward_ops),
        )

        np.save(
            os.path.join(output_dir, "total_train_ops.npy"),
            np.asarray(total_train_ops),
        )

        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    return {
        "R": None,
        "metrics": metrics,
        "train_times": train_times,
        "model_sizes": model_sizes,
        "buffer_sizes": buffer_sizes,
        "memory_sizes": buffer_sizes,
        "forward_backward_ops": forward_backward_ops,
        "total_train_ops": total_train_ops,
        "lifetime_data_size": lifetime_data_size,
        "dataset_memory_size": lifetime_data_size,
        "task_accs": task_accs,
        "growing_accs": growing_accs,
        "fixed_accs": fixed_accs,
        "eval_results": eval_results,
    }