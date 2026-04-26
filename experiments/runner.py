import os
import time
import json
import numpy as np

from evaluation.result_matrix import ResultMatrix
from evaluation.continual_metrics import compute_summary_metrics


def _extract_metric(last_metrics, metric_name):
    matches = [
        v for k, v in last_metrics.items()
        if metric_name in k
    ]

    if len(matches) == 0:
        raise KeyError(f"Could not find metric: {metric_name}")

    value = matches[-1]

    if hasattr(value, "item"):
        value = value.item()

    return value


def get_dataset_memory_size(benchmark):
    return sum(len(exp.dataset) for exp in benchmark.train_stream)


def get_test_experience_sizes(benchmark):
    return np.array(
        [len(exp.dataset) for exp in benchmark.test_stream],
        dtype=float,
    )


def weighted_mean_accuracy(accs, sizes):
    accs = np.array(accs, dtype=float)
    sizes = np.array(sizes, dtype=float)

    valid = ~np.isnan(accs)
    accs = accs[valid]
    sizes = sizes[valid]

    if len(accs) == 0 or np.sum(sizes) == 0:
        return np.nan

    return float(np.sum(accs * sizes) / np.sum(sizes))


def estimate_ops(
    train_size,
    model_size,
    replay_size=0,
    regularization_multiplier=1.0,
    routing_multiplier=1.0,
    memory_update_multiplier=0.0,
):
    baseline_ops = train_size * model_size

    effective_train_size = train_size + replay_size

    total_ops = (
        effective_train_size
        * model_size
        * regularization_multiplier
        * routing_multiplier
    )

    total_ops += memory_update_multiplier * train_size * model_size

    return baseline_ops, total_ops


def run_continual_experiment(
    strategy,
    benchmark,
    output_dir="outputs/experiment",
    regularization_multiplier=1.0,
    routing_multiplier=1.0,
    memory_update_multiplier=0.0,
    save_outputs=False,
):
    os.makedirs(output_dir, exist_ok=True)

    n_experiences = len(benchmark.train_stream)
    result_matrix = ResultMatrix(n_experiences)

    train_times = []
    model_sizes = []
    memory_sizes = []

    forward_backward_ops = []
    total_train_ops = []

    task_accs = []
    growing_accs = []
    fixed_accs = []

    test_sizes = get_test_experience_sizes(benchmark)

    for train_exp_id, train_exp in enumerate(benchmark.train_stream):
        print("=" * 70)
        print(f"Training experience {train_exp_id}")
        print("=" * 70)

        start_time = time.time()
        strategy.train(train_exp)
        train_times.append(time.time() - start_time)

        last_metrics = strategy.evaluator.get_last_metrics()

        model_size = int(_extract_metric(last_metrics, "ModelSizeMetric"))
        replay_size = int(_extract_metric(last_metrics, "ReplaySizeMetric"))

        model_sizes.append(model_size)
        memory_sizes.append(replay_size)

        train_size = len(train_exp.dataset)

        baseline_ops, method_ops = estimate_ops(
            train_size=train_size,
            model_size=model_size,
            replay_size=replay_size,
            regularization_multiplier=regularization_multiplier,
            routing_multiplier=routing_multiplier,
            memory_update_multiplier=memory_update_multiplier,
        )

        forward_backward_ops.append(baseline_ops)
        total_train_ops.append(method_ops)

        print("Evaluating on full test stream for R matrix")
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

    dataset_memory_size = get_dataset_memory_size(benchmark)

    metrics = compute_summary_metrics(
        R=R,
        model=strategy.model,
        train_times=train_times,
        model_sizes=model_sizes,
        memory_sizes=memory_sizes,
        dataset_memory_size=dataset_memory_size,
        forward_backward_ops=forward_backward_ops,
        total_train_ops=total_train_ops,
    )

    if save_outputs:
        np.save(os.path.join(output_dir, "accuracy_matrix.npy"), R)

        np.save(os.path.join(output_dir, "task_accs.npy"), np.array(task_accs))
        np.save(os.path.join(output_dir, "growing_accs.npy"), np.array(growing_accs))
        np.save(os.path.join(output_dir, "fixed_accs.npy"), np.array(fixed_accs))

        np.save(os.path.join(output_dir, "model_sizes.npy"), np.array(model_sizes))
        np.save(os.path.join(output_dir, "memory_sizes.npy"), np.array(memory_sizes))

        np.save(
            os.path.join(output_dir, "forward_backward_ops.npy"),
            np.array(forward_backward_ops),
        )

        np.save(
            os.path.join(output_dir, "total_train_ops.npy"),
            np.array(total_train_ops),
        )

        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    return {
        "R": R,
        "metrics": metrics,
        "train_times": train_times,
        "model_sizes": model_sizes,
        "memory_sizes": memory_sizes,
        "forward_backward_ops": forward_backward_ops,
        "total_train_ops": total_train_ops,
        "dataset_memory_size": dataset_memory_size,
        "test_sizes": test_sizes,
        "task_accs": task_accs,
        "growing_accs": growing_accs,
        "fixed_accs": fixed_accs,
    }


def run_joint_experiment(
    strategy,
    benchmark,
    output_dir="outputs/joint",
    save_outputs=False,
):
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    print("=" * 70)
    print("Training JointTraining on full train stream")
    print("=" * 70)

    strategy.train(benchmark.train_stream)

    train_time = time.time() - start_time

    print("Evaluating JointTraining on full test stream")
    eval_results = strategy.eval(benchmark.test_stream)

    last_metrics = strategy.evaluator.get_last_metrics()

    model_size = int(_extract_metric(last_metrics, "ModelSizeMetric"))

    stream_acc = None
    for key, value in eval_results.items():
        if "Top1_Acc_Stream" in key:
            stream_acc = float(value)
            break

    if stream_acc is None:
        raise KeyError("Could not find Top1_Acc_Stream in eval_results")

    n_experiences = len(benchmark.test_stream)

    task_accs = [stream_acc] * n_experiences
    growing_accs = [stream_acc] * n_experiences
    fixed_accs = [stream_acc] * n_experiences

    train_times = [train_time] * n_experiences
    model_sizes = [model_size] * n_experiences
    memory_sizes = [0] * n_experiences

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
        "Total Train Time": train_time,
        "Average Train Time": train_time,
    }

    if save_outputs:
        np.save(os.path.join(output_dir, "task_accs.npy"), np.array(task_accs))
        np.save(os.path.join(output_dir, "growing_accs.npy"), np.array(growing_accs))
        np.save(os.path.join(output_dir, "fixed_accs.npy"), np.array(fixed_accs))

        np.save(os.path.join(output_dir, "model_sizes.npy"), np.array(model_sizes))
        np.save(os.path.join(output_dir, "memory_sizes.npy"), np.array(memory_sizes))

        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    return {
        "R": None,
        "metrics": metrics,
        "train_times": train_times,
        "model_sizes": model_sizes,
        "memory_sizes": memory_sizes,
        "task_accs": task_accs,
        "growing_accs": growing_accs,
        "fixed_accs": fixed_accs,
        "eval_results": eval_results,
    }