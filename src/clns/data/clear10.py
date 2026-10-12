"""CLEAR-10 domain-incremental benchmark (iid protocol only).

Avalanche's ``CLEAR(..., evaluation_protocol="streaming")`` makes the test stream a duplicate of
the train stream (train == test data -> leakage), so only ``"iid"`` (70:30 split inside each
time bucket, split seed in 0..4) is allowed. Source: avalanche/benchmarks/classic/clear.py (v0.6.0).
Labels: 10 concepts + BACKGROUND = 11 classes.
Requires ``<root>/.done`` (``scripts/download_data.sh clear10``) unless allow_download /
require_done_marker=false. Train/test file disjointness is asserted per bucket.
Task order: the time order of the buckets, the same for every training seed (``task_order`` =
bucket indices, ``task_order_sha256`` in the meta).
"""

from __future__ import annotations

from pathlib import Path

from .common import (
    BenchmarkBundle,
    assert_disjoint,
    build_transforms,
    dataset_classes,
    finalize,
    record_task_order,
    require_data_root,
)

_SEEDS = (0, 1, 2, 3, 4)


def _check_clear_disjoint(root: Path, split_seed: int) -> None:
    """Train/test file disjointness per bucket and overall (iid protocol).

    The CLScenario returned by Avalanche's ``CLEAR`` does not expose file paths, so the same
    split lists are re-read with ``_CLEARImage(..., split="train"|"test", seed=split_seed)``
    and ``get_paths_and_targets(root_appended=False)`` (one list of (path, target) per bucket),
    exactly as ``avalanche/benchmarks/classic/clear.py`` (v0.6.0) builds the benchmark.
    Private Avalanche API (pinned avalanche-lib==0.6.0).
    """
    from avalanche.benchmarks.datasets.clear import _CLEARImage

    kw = dict(root=root, data_name="clear10", download=True, verbose=False, seed=split_seed)
    train = _CLEARImage(split="train", **kw).get_paths_and_targets(root_appended=False)
    test = _CLEARImage(split="test", **kw).get_paths_and_targets(root_appended=False)
    if len(train) != len(test):
        raise ValueError(f"CLEAR-10: {len(train)} train buckets vs {len(test)} test buckets")
    for b, (tr, te) in enumerate(zip(train, test)):
        assert_disjoint([str(p) for p, _ in tr], [str(p) for p, _ in te], f"CLEAR-10 bucket {b}")
    assert_disjoint(
        [str(p) for bucket in train for p, _ in bucket],
        [str(p) for bucket in test for p, _ in bucket],
        "CLEAR-10 (all buckets)",
    )


def build_clear10(cfg, data_dir: str, seed: int) -> BenchmarkBundle:
    protocol = str(cfg.get("protocol", "iid")).lower()
    if protocol != "iid":
        raise ValueError(
            f"CLEAR-10 protocol '{protocol}' is not allowed: the Avalanche 'streaming' protocol "
            f"uses the train stream as test stream (leakage). Use benchmark.protocol=iid."
        )
    split_seed = int(cfg.get("split_seed", 0))
    if split_seed not in _SEEDS:
        raise ValueError(f"CLEAR split_seed must be one of {_SEEDS}, got {split_seed}")

    root = Path(data_dir) / cfg.get("data_subdir", "clear10")
    require_data_root(root, cfg, "CLEAR-10", "clear10")

    from avalanche.benchmarks.classic.clear import CLEAR  # heavy import, keep lazy

    train_tf, eval_tf = build_transforms(cfg)
    bm = CLEAR(
        data_name="clear10",
        evaluation_protocol="iid",
        seed=split_seed,
        train_transform=train_tf,
        eval_transform=eval_tf,
        dataset_root=root,
    )
    _check_clear_disjoint(root, split_seed)
    train_list = [e.dataset for e in bm.train_stream]
    test_list = [e.dataset for e in bm.test_stream]
    if len(train_list) != int(cfg.n_experiences) or len(test_list) != len(train_list):
        raise ValueError(
            f"CLEAR-10 returned {len(train_list)} train / {len(test_list)} test experiences, "
            f"expected {cfg.n_experiences}"
        )
    classes = sorted({c for d in train_list for c in dataset_classes(d)})
    num_classes = max(classes) + 1

    meta = {
        "protocol": "iid",
        "split_seed": split_seed,
        "classes": classes,
        "n_experiences_full": len(train_list),
        "dataset_root": str(root),
    }
    # The time order of the buckets IS the benchmark: identical for every seed (seed only
    # changes initialisation / data order within an experience).
    record_task_order(meta, list(range(len(train_list))))
    benchmark = finalize(
        train_list, test_list, cfg, seed, meta,
        test_identities=[f"bucket:{b}" for b in range(len(test_list))],
    )
    return BenchmarkBundle(
        benchmark=benchmark,
        scenario="domain_il",
        num_classes=num_classes,
        class_order=None,
        meta=meta,
    )
