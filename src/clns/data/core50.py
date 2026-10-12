"""CORe50 NIC (default NICv2-79) benchmark with first-appearance test groups.

Avalanche's CORe50 (classic/core50.py, v0.6.0) exposes ONE shared test experience (sessions
3, 7, 10) -- a fixed test set. For the forgetting estimand (RESEARCH_LOG 7.2.1) the test set is
partitioned by the training experience in which each class FIRST appears: test experience j holds
the test frames of the classes first seen in train experience ``test_exp_to_train_exp[j]``;
empty groups are dropped. Train and test streams therefore generally have different lengths.

Validation: ``val_fraction > 0`` raises (frame-level splits leak; session hold-out is TODO).
Data completeness: ``<root>/.done`` from ``scripts/download_data.sh core50`` is required.

Task order follows the seed: ``benchmark.run: null`` uses Avalanche NIC run = training seed
(:func:`resolve_core50_run`); recorded as ``run`` / ``task_order`` (``[run, class set of every
training batch]``; the run is included because two runs could share batch class sets while their
frames differ) / ``task_order_sha256`` in the meta, plus ``test_filelist_sha256`` (hash of the
run's ``test_filelist.txt``), which lands in the shift result's ``split_hashes``.

Note: Avalanche only sets ``benchmark.classes_order`` for the "nc" variant, so the class order for
NIC variants is derived here from first appearance (ascending class id within an experience).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .common import (
    BenchmarkBundle,
    assert_disjoint,
    build_transforms,
    dataset_classes,
    finalize,
    record_task_order,
    require_data_root,
    sha256_file,
)


def partition_test_by_first_appearance(
    train_class_sets: Sequence[Sequence[int]],
    test_class_to_idx: Mapping[int, Sequence[int]],
    max_experiences: int | None = None,
):
    """Group test indices by the train experience where each class first appears.

    :param train_class_sets: classes present in each (full) train experience.
    :param test_class_to_idx: test class -> indices into the shared test dataset.
    :param max_experiences: keep only groups whose first appearance is < this value.
    :return: (test_exp_to_train_exp, groups, indices) -- parallel lists, one entry per non-empty
        group: train experience k, sorted classes, sorted test indices.
    :raises ValueError: if a test class never appears in training.
    """
    first_appearance: dict[int, int] = {}
    for k, classes in enumerate(train_class_sets):
        for c in sorted(int(x) for x in classes):
            first_appearance.setdefault(c, k)
    unseen = sorted(int(c) for c in test_class_to_idx if int(c) not in first_appearance)
    if unseen:
        raise ValueError(f"Test classes never seen in training: {unseen}")

    n_keep = len(train_class_sets) if max_experiences is None else int(max_experiences)
    exp_map: list[int] = []
    groups: list[list[int]] = []
    indices: list[list[int]] = []
    for k in range(min(n_keep, len(train_class_sets))):
        classes = sorted(c for c, kk in first_appearance.items() if kk == k)
        present = [c for c in classes if c in test_class_to_idx]
        idxs = sorted(int(i) for c in present for i in test_class_to_idx[c])
        if idxs:
            exp_map.append(k)
            groups.append(present)
            indices.append(idxs)
    return exp_map, groups, indices


def _check_core50_disjoint(root: Path, variant: str, run: int) -> str:
    """Assert no frame of the test filelist (sessions 3, 7, 10) is in any train filelist.

    Returns the sha256 of the test filelist (:func:`sha256_file`, CRLF-normalised).

    Re-reads the same filelists ``avalanche/benchmarks/classic/core50.py`` (v0.6.0) uses:
    ``<root>/<scen2dirs[variant] without '/'>/run<run>/train_batch_XX_filelist.txt`` and
    ``.../test_filelist.txt`` (object_lvl=True), parsed with ``default_flist_reader``.
    """
    from avalanche.benchmarks.classic.core50 import nbatch, scen2dirs
    from avalanche.benchmarks.utils.datasets_from_filelists import default_flist_reader

    base = root / (scen2dirs[variant][:-1] + "/" + "run" + str(run))
    train_paths: list[str] = []
    for b in range(nbatch[variant]):
        flist = base / f"train_batch_{str(b).zfill(2)}_filelist.txt"
        train_paths.extend(p for p, _ in default_flist_reader(flist))
    test_paths = [p for p, _ in default_flist_reader(base / "test_filelist.txt")]
    assert_disjoint(train_paths, test_paths, f"CORe50 {variant} run {run}")
    return sha256_file(base / "test_filelist.txt")


def resolve_core50_run(cfg, seed: int) -> int:
    """Avalanche NIC run (batch ordering) used for training seed ``seed``.

    ``benchmark.run: null`` (default) -> run = seed, so the task order follows the seed (runs
    0, 1, 2 for seeds 0, 1, 2); an explicit int pins one run for every seed. Must be in 0..9
    (the predefined Avalanche orderings).
    """
    run = cfg.get("run", None)
    run = int(seed) if run is None else int(run)
    if not 0 <= run <= 9:
        raise ValueError(
            f"CORe50 run must be in 0..9, got {run} (benchmark.run=null uses run = seed)"
        )
    return run


def build_core50(cfg, data_dir: str, seed: int) -> BenchmarkBundle:
    variant = str(cfg.get("variant", "nicv2_79"))
    run = resolve_core50_run(cfg, seed)
    if variant not in ("nicv2_79", "nicv2_196", "nicv2_391", "nic"):
        raise ValueError(f"Unsupported CORe50 NIC variant '{variant}'")

    if float(cfg.get("val_fraction", 0.0) or 0.0) > 0.0:
        raise ValueError(
            "CORe50: val_fraction > 0 is not supported. A random frame-level split of the "
            "training batches leaks near-duplicate consecutive frames of the same session into "
            "validation; a session hold-out validation split is required (TODO, not "
            "implemented). Use benchmark.val_fraction=0.0."
        )

    root = Path(data_dir) / cfg.get("data_subdir", "core50")
    require_data_root(root, cfg, "CORe50", "core50")

    from avalanche.benchmarks.classic.core50 import CORe50  # heavy import, keep lazy

    train_tf, eval_tf = build_transforms(cfg)
    bm = CORe50(
        scenario=variant,
        run=run,
        object_lvl=True,
        mini=False,
        train_transform=train_tf,
        eval_transform=eval_tf,
        dataset_root=root,
    )
    test_filelist_sha = _check_core50_disjoint(root, variant, run)
    train_full = [e.dataset for e in bm.train_stream]
    if len(bm.test_stream) != 1:
        raise ValueError("Expected a single shared CORe50 test experience")
    test_full = bm.test_stream[0].dataset

    max_exp = cfg.get("max_experiences")
    max_exp = None if max_exp is None else int(max_exp)
    train_classes = [dataset_classes(d) for d in train_full]
    exp_map, groups, indices = partition_test_by_first_appearance(
        train_classes, test_full.targets.val_to_idx, max_exp
    )
    test_list = [test_full.subset(idx) for idx in indices]

    # class order = order of first appearance (derived; Avalanche only sets it for 'nc').
    seen: list[int] = []
    for classes in train_classes[: (max_exp or len(train_classes))]:
        seen.extend(c for c in classes if c not in seen)

    meta = {
        "protocol": "nic",
        "variant": variant,
        "run": run,
        "class_order": seen,
        "n_experiences_full": len(train_full),
        "test_exp_to_train_exp": exp_map,
        "first_appearance_groups": groups,  # aligned with test experiences
        "train_stream_classes": train_classes,
        "note": "test experiences are first-appearance class groups; train/test lengths may differ",
        "dataset_root": str(root),
        "test_filelist_sha256": test_filelist_sha,
    }
    # Task order = the run + its sequence of training batches (fingerprinted by their class sets).
    record_task_order(meta, [run, train_classes])
    # Test already truncated by first-appearance filtering; do not truncate it again.
    benchmark = finalize(train_full, test_list, cfg, seed, meta, truncate_test=False)
    return BenchmarkBundle(
        benchmark=benchmark,
        scenario="class_il",
        num_classes=50,
        class_order=seen,
        meta=meta,
    )
