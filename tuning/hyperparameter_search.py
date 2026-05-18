# tuning/hyperparameter_search.py

import os
import json
import itertools
import numpy as np

from benchmarks.benchmark_factory import build_benchmark
from models.model_factory import build_model
from evaluation.evaluator_factory import build_evaluator
from strategies.strategy_factory import build_strategy
from experiments.runner import run_continual_experiment
from utils.seed import set_seed


# -------------------------------------------------------
# Config splitting
# -------------------------------------------------------

EXPERIMENT_CONFIG_KEYS = {
    "regularization_multiplier",
    "routing_multiplier",
    "memory_update_multiplier",
    "regularization_overhead",
    "routing_overhead",
    "memory_update_overhead",
    "baseline_accs",
}


def split_config(config):
    """
    Split a hyperparameter config into:

    1. strategy_config:
       Parameters used to build the Avalanche strategy.
       Example:
           lr, train_mb_size, train_epochs, optimizer_name,
           plugin_names, ewc_lambda, mem_size, batch_size_mem

    2. experiment_config:
       Parameters used by run_continual_experiment.
       Example:
           regularization_multiplier,
           routing_multiplier,
           memory_update_multiplier

    This is important because compute-efficiency multipliers should not
    be passed to build_strategy. They must be passed to run_continual_experiment.
    """

    strategy_config = {}
    experiment_config = {}

    for key, value in config.items():
        if key in EXPERIMENT_CONFIG_KEYS:
            experiment_config[key] = value
        else:
            strategy_config[key] = value

    return strategy_config, experiment_config


def make_grid(search_space):
    keys = list(search_space.keys())
    values = list(search_space.values())

    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*values)
    ]


def convert_to_json_safe(obj):
    """
    Make objects safe for json.dump.

    Handles:
    - numpy scalars
    - numpy arrays
    - nested dicts/lists/tuples
    """

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, dict):
        return {
            key: convert_to_json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [convert_to_json_safe(value) for value in obj]

    if isinstance(obj, tuple):
        return [convert_to_json_safe(value) for value in obj]

    return obj


def run_single_config(
    config,
    run_id,
    benchmark_name,
    model_name,
    strategy_name,
    n_experiences,
    seed,
    output_root,
    model_kwargs=None,
    save_outputs=False,
):
    if model_kwargs is None:
        model_kwargs = {}

    strategy_config, experiment_config = split_config(config)

    set_seed(seed)

    benchmark = build_benchmark(
        benchmark_name=benchmark_name,
        n_experiences=n_experiences,
        seed=seed,
    )

    model = build_model(
        model_name=model_name,
        **model_kwargs,
    )

    output_dir = os.path.join(output_root, f"run_{run_id}")
    log_dir = os.path.join(output_dir, "logs")

    evaluator = build_evaluator(
        log_dir=log_dir,
        use_interactive=False,
        use_text=True,
        use_tensorboard=True,
    )

    strategy = build_strategy(
        strategy_name=strategy_name,
        model=model,
        evaluator=evaluator,
        **strategy_config,
    )

    result = run_continual_experiment(
        strategy=strategy,
        benchmark=benchmark,
        output_dir=output_dir,
        save_outputs=save_outputs,
        **experiment_config,
    )

    return result


def grid_search(
    approach_name,
    search_space,
    benchmark_name="split_mnist",
    model_name="avalanche_simplemlp",
    strategy_name="naive",
    n_experiences=10,
    seed=1234,
    model_kwargs=None,
    output_root="outputs/hyperparameter_tuning",
    score_key="A Metric",
    save_outputs=False,
):
    configs = make_grid(search_space)
    all_results = []

    search_output_root = os.path.join(
        output_root,
        approach_name,
        "grid_search",
    )

    os.makedirs(search_output_root, exist_ok=True)

    for run_id, config in enumerate(configs):
        print("\n" + "#" * 80)
        print(f"Grid Search | Approach: {approach_name}")
        print(f"Config {run_id + 1}/{len(configs)}")
        print("Full config:")
        print(config)

        strategy_config, experiment_config = split_config(config)

        print("Strategy config:")
        print(strategy_config)

        print("Experiment config:")
        print(experiment_config)

        print("#" * 80)

        result = run_single_config(
            config=config,
            run_id=run_id,
            benchmark_name=benchmark_name,
            model_name=model_name,
            strategy_name=strategy_name,
            n_experiences=n_experiences,
            seed=seed,
            output_root=search_output_root,
            model_kwargs=model_kwargs,
            save_outputs=save_outputs,
        )

        score = result["metrics"][score_key]

        all_results.append(
            {
                "run_id": run_id,
                "approach_name": approach_name,
                "config": config,
                "strategy_config": strategy_config,
                "experiment_config": experiment_config,
                "score_key": score_key,
                "score": float(score),
                "metrics": result["metrics"],
            }
        )

    best_result = max(
        all_results,
        key=lambda x: x["score"],
    )

    summary = {
        "approach_name": approach_name,
        "score_key": score_key,
        "best_result": best_result,
        "all_results": all_results,
    }

    with open(
        os.path.join(search_output_root, "grid_search_summary.json"),
        "w",
    ) as f:
        json.dump(
            convert_to_json_safe(summary),
            f,
            indent=4,
        )

    return best_result, all_results


def run_repeated_evaluation(
    approach_name,
    best_config,
    seeds,
    benchmark_name="split_mnist",
    model_name="avalanche_simplemlp",
    strategy_name="naive",
    n_experiences=10,
    model_kwargs=None,
    output_root="outputs/hyperparameter_tuning",
    score_key="A Metric",
    save_outputs=False,
):
    final_results = []

    final_output_root = os.path.join(
        output_root,
        approach_name,
        "repeated_evaluation",
    )

    os.makedirs(final_output_root, exist_ok=True)

    for run_id, seed in enumerate(seeds):
        print("\n" + "=" * 80)
        print(f"Repeated Evaluation | Approach: {approach_name}")
        print(f"Seed {seed} | Run {run_id + 1}/{len(seeds)}")
        print("Best config:")
        print(best_config)

        strategy_config, experiment_config = split_config(best_config)

        print("Strategy config:")
        print(strategy_config)

        print("Experiment config:")
        print(experiment_config)

        print("=" * 80)

        result = run_single_config(
            config=best_config,
            run_id=run_id,
            benchmark_name=benchmark_name,
            model_name=model_name,
            strategy_name=strategy_name,
            n_experiences=n_experiences,
            seed=seed,
            output_root=final_output_root,
            model_kwargs=model_kwargs,
            save_outputs=save_outputs,
        )

        final_results.append(
            {
                "run_id": run_id,
                "seed": seed,
                "approach_name": approach_name,
                "config": best_config,
                "strategy_config": strategy_config,
                "experiment_config": experiment_config,
                "score_key": score_key,
                "score": float(result["metrics"][score_key]),
                "metrics": result["metrics"],
                "result": result,
            }
        )

    scores = np.asarray(
        [r["score"] for r in final_results],
        dtype=float,
    )

    summary = {
        "approach_name": approach_name,
        "best_config": best_config,
        "score_key": score_key,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "sem_score": float(np.std(scores) / np.sqrt(len(scores))),
        "num_runs": len(final_results),
        "seeds": list(seeds),
    }

    with open(
        os.path.join(final_output_root, "repeated_evaluation_summary.json"),
        "w",
    ) as f:
        json.dump(
            convert_to_json_safe(summary),
            f,
            indent=4,
        )

    return summary, final_results


def tune_and_evaluate(
    approach_name,
    search_space,
    seeds,
    benchmark_name="split_mnist",
    model_name="avalanche_simplemlp",
    strategy_name="naive",
    n_experiences=10,
    tuning_seed=1234,
    model_kwargs=None,
    output_root="outputs/hyperparameter_tuning",
    score_key="A Metric",
    save_outputs=False,
):
    best_result, grid_results = grid_search(
        approach_name=approach_name,
        search_space=search_space,
        benchmark_name=benchmark_name,
        model_name=model_name,
        strategy_name=strategy_name,
        n_experiences=n_experiences,
        seed=tuning_seed,
        model_kwargs=model_kwargs,
        output_root=output_root,
        score_key=score_key,
        save_outputs=save_outputs,
    )

    best_config = best_result["config"]

    repeated_summary, repeated_results = run_repeated_evaluation(
        approach_name=approach_name,
        best_config=best_config,
        seeds=seeds,
        benchmark_name=benchmark_name,
        model_name=model_name,
        strategy_name=strategy_name,
        n_experiences=n_experiences,
        model_kwargs=model_kwargs,
        output_root=output_root,
        score_key=score_key,
        save_outputs=save_outputs,
    )

    return {
        "approach_name": approach_name,
        "best_result": best_result,
        "best_config": best_config,
        "grid_results": grid_results,
        "repeated_summary": repeated_summary,
        "repeated_results": repeated_results,
    }