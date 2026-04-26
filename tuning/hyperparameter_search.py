import itertools
import numpy as np

from hds_cl.benchmarks.benchmark_factory import build_benchmark
from hds_cl.models.model_factory import build_model
from hds_cl.evaluation.evaluator_factory import build_evaluator
from hds_cl.strategies.strategy_factory import build_strategy
from hds_cl.experiments.runner import run_continual_experiment


def grid_search(
    search_space,
    benchmark_name="split_mnist",
    model_name="simple_mlp",
    strategy_name="naive",
    n_experiences=5,
    seed=1234,
    output_root="outputs/tuning",
):
    keys = list(search_space.keys())
    values = list(search_space.values())

    configs = [
        dict(zip(keys, combination))
        for combination in itertools.product(*values)
    ]

    all_results = []

    for run_id, config in enumerate(configs):
        print("\n" + "#" * 80)
        print(f"Running config {run_id}")
        print(config)
        print("#" * 80)

        benchmark = build_benchmark(
            benchmark_name=benchmark_name,
            n_experiences=n_experiences,
            seed=seed,
        )

        model = build_model(
            model_name=model_name,
            num_classes=10,
        )

        log_dir = f"{output_root}/run_{run_id}/logs"
        output_dir = f"{output_root}/run_{run_id}"

        evaluator = build_evaluator(log_dir=log_dir)

        strategy = build_strategy(
            strategy_name=strategy_name,
            model=model,
            evaluator=evaluator,
            **config,
        )

        result = run_continual_experiment(
            strategy=strategy,
            benchmark=benchmark,
            protocol="growing",
            output_dir=output_dir,
        )

        avg_accuracy = result["metrics"]["Average Accuracy"]

        all_results.append({
            "run_id": run_id,
            "config": config,
            "score": avg_accuracy,
            "metrics": result["metrics"],
        })

    best_result = max(all_results, key=lambda x: x["score"])

    return best_result, all_results