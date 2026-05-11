from avalanche.benchmarks.classic import (
    SplitMNIST,
    SplitCIFAR10,
    PermutedMNIST,
)


def build_benchmark(
    benchmark_name: str,
    n_experiences: int = 5,
    seed: int = 1234,
    return_task_id: bool = False,
):
    benchmark_name = benchmark_name.lower().strip()

    if benchmark_name == "split_mnist":
        return SplitMNIST(
            n_experiences=n_experiences,
            seed=seed,
            return_task_id=return_task_id,
        )

    if benchmark_name == "split_cifar10":
        return SplitCIFAR10(
            n_experiences=n_experiences,
            seed=seed,
            return_task_id=return_task_id,
        )

    if benchmark_name == "permuted_mnist":
        return PermutedMNIST(
            n_experiences=n_experiences,
            seed=seed,
            return_task_id=return_task_id,
        )

    raise ValueError(f"Unknown benchmark_name: {benchmark_name}")