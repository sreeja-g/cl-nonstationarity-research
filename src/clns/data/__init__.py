"""Benchmark data layer: ``build_benchmark(cfg_benchmark, data_dir, seed) -> BenchmarkBundle``."""

from __future__ import annotations

from .common import BenchmarkBundle

BENCHMARKS = ("imagenet100", "clear10", "domainnet", "core50_nic", "synthetic")


def build_benchmark(cfg_benchmark, data_dir: str, seed: int) -> BenchmarkBundle:
    """Build the benchmark named ``cfg_benchmark.name`` (lazy per-benchmark imports)."""
    name = cfg_benchmark.name
    if name == "imagenet100":
        from .imagenet100 import build_imagenet100 as fn
    elif name == "clear10":
        from .clear10 import build_clear10 as fn
    elif name == "domainnet":
        from .domainnet import build_domainnet as fn
    elif name == "core50_nic":
        from .core50 import build_core50 as fn
    elif name == "synthetic":
        from .synthetic import build_synthetic as fn
    else:
        raise ValueError(f"Unknown benchmark '{name}'. Choose from {BENCHMARKS}.")
    return fn(cfg_benchmark, str(data_dir), int(seed))


__all__ = ["BenchmarkBundle", "build_benchmark", "BENCHMARKS"]
