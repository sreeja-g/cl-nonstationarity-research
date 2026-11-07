"""CLEAR-10 domain-incremental benchmark (iid protocol only).

Avalanche's ``CLEAR(..., evaluation_protocol="streaming")`` makes the test stream a duplicate of
the train stream (train == test data -> leakage), so only ``"iid"`` is allowed. For
``data_name="clear10"`` (no underscore) Avalanche (datasets/clear/clear.py, v0.6.0,
``_load_metadata_new``) reads the released ``train/`` and ``test/`` folders as they are: the test
set is the separate released test folder, there is NO 70:30 split and the ``seed`` argument is
only checked against ``SEED_LIST`` (the 7:3 ``testset_ratio_0.3/split_<seed>`` file lists belong
to the old ``clear10_...`` names). ``split_seed`` therefore has no effect and must be 0.

AVALANCHE BUG (v0.6.0, ``_CLEARImage._load_metadata_new``, clear.py ~line 338): ``samples = []``
is created once per split OUTSIDE the ``for bucket`` loop and the same list object is appended
for every bucket, so with ``CLEAR(...)`` every train and every test experience would hold the
images of ALL buckets (no domain shift at all). This module therefore does NOT use ``CLEAR(...)``:
:func:`read_clear_buckets` builds fresh per-bucket ``(path, class index)`` lists from
``<root>/<split>/labeled_metadata.json``, ``class_names.txt`` and the per-class metadata files
(same parsing as Avalanche's ``_load_metadata_new``, paths ``<split>/<IMG_PATH>`` relative to the
root), buckets in ascending integer key order (ids recorded as ``bucket_ids`` in the meta), and
the experiences are ``PathsDataset`` s like in the other loaders. Labels: 10 concepts +
BACKGROUND = 11 classes.

Requires ``<root>/.done`` (``scripts/download_data.sh clear10``) unless allow_download /
require_done_marker=false (Avalanche is no longer used, so nothing is downloaded here). Checks
(:func:`check_clear_buckets`): buckets of a split pairwise disjoint, bucket sizes sum to the
unique files, train/test disjoint by full path and by file name, same bucket ids in train and test,
every image exists, number of buckets == ``n_experiences``.
Task order: the time order of the buckets, the same for every training seed (``task_order`` =
bucket indices, ``task_order_sha256`` in the meta).
"""

from __future__ import annotations

import json
from pathlib import Path

from avalanche.benchmarks.utils.datasets_from_filelists import PathsDataset

from .common import (
    BenchmarkBundle,
    assert_disjoint,
    basename_key,
    build_transforms,
    dataset_classes,
    finalize,
    make_cls_dataset,
    record_task_order,
    require_data_root,
)

_SEEDS = (0, 1, 2, 3, 4)


def read_clear_buckets(root: Path, split: str):
    """Per-bucket ``(root-relative path, class index)`` lists of one released split.

    Mirrors Avalanche's ``_load_metadata_new`` (datasets/clear/clear.py, v0.6.0) but with a FRESH
    list per bucket. Returns ``(bucket_ids, class_names, buckets)``; buckets are in ascending
    integer key order of ``labeled_metadata.json``.
    """
    folder = Path(root) / split
    meta_file = folder / "labeled_metadata.json"
    if not meta_file.is_file():
        raise FileNotFoundError(f"CLEAR-10: missing {meta_file}")
    labeled = json.loads(meta_file.read_text(encoding="utf-8"))
    class_names = [
        c.strip() for c in (folder / "class_names.txt").read_text().splitlines() if c.strip()
    ]
    try:
        keys = sorted(labeled, key=int)
    except ValueError as e:
        raise ValueError(f"CLEAR-10 {split}: non-integer bucket keys {list(labeled)}") from e
    buckets = []
    for key in keys:
        samples: list[tuple[str, int]] = []  # fresh per bucket (the Avalanche bug shares one)
        for class_idx, name in enumerate(class_names):
            meta = json.loads((folder / labeled[key][name]).read_text(encoding="utf-8"))
            for v in meta.values():
                samples.append((f"{split}/{v['IMG_PATH']}", class_idx))
        buckets.append(samples)
    return [int(k) for k in keys], class_names, buckets


def check_clear_buckets(root: Path, train, test, n_experiences: int) -> None:
    """Leakage / sanity checks on the train and test bucket lists (see the module docstring)."""
    for name, buckets in (("train", train), ("test", test)):
        if len(buckets) != n_experiences:
            raise ValueError(
                f"CLEAR-10 {name}: {len(buckets)} buckets, expected {n_experiences}"
            )
        seen: set[str] = set()
        for b, bucket in enumerate(buckets):
            paths = [p for p, _ in bucket]
            if not paths:
                raise ValueError(f"CLEAR-10 {name} bucket {b} is empty")
            if len(set(paths)) != len(paths):
                raise ValueError(f"CLEAR-10 {name} bucket {b} lists a file twice")
            assert_disjoint(sorted(seen), paths, f"CLEAR-10 {name}: bucket {b} vs earlier buckets")
            seen.update(paths)
        if len(seen) != sum(len(x) for x in buckets):  # defensive: sizes sum to unique files
            raise ValueError(f"CLEAR-10 {name}: buckets overlap")
    tr_all = [p for bucket in train for p, _ in bucket]
    te_all = [p for bucket in test for p, _ in bucket]
    assert_disjoint(tr_all, te_all, "CLEAR-10 (all buckets)")
    assert_disjoint(tr_all, te_all, "CLEAR-10 (file names)", key=basename_key)
    missing = [p for p in tr_all + te_all if not (Path(root) / p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"CLEAR-10: {len(missing)} listed images missing, e.g. {missing[:3]}"
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
    if split_seed != 0:
        raise ValueError(
            f"CLEAR-10: split_seed={split_seed} has no effect (Avalanche 0.6.0 reads the fixed "
            f"released train/ and test/ folders for data_name='clear10'; the seed is only "
            f"checked against SEED_LIST) -- use benchmark.split_seed=0. A seed other than 0 "
            f"would be recorded as a different split that is in fact identical."
        )

    root = Path(data_dir) / cfg.get("data_subdir", "clear10")
    require_data_root(root, cfg, "CLEAR-10", "clear10")

    ids_tr, names_tr, tr_buckets = read_clear_buckets(root, "train")
    ids_te, names_te, te_buckets = read_clear_buckets(root, "test")
    if ids_tr != ids_te or names_tr != names_te:
        raise ValueError(
            f"CLEAR-10 train/test differ in buckets ({ids_tr} vs {ids_te}) or class names"
        )
    check_clear_buckets(root, tr_buckets, te_buckets, int(cfg.n_experiences))

    train_tf, eval_tf = build_transforms(cfg)

    def _datasets(buckets, group):
        return [
            make_cls_dataset(
                PathsDataset(root, files), targets=[y for _, y in files],
                train_transform=train_tf, eval_transform=eval_tf, group=group,
            )
            for files in buckets
        ]

    train_list = _datasets(tr_buckets, "train")
    test_list = _datasets(te_buckets, "eval")
    classes = sorted({c for d in train_list for c in dataset_classes(d)})
    num_classes = max(classes) + 1

    meta = {
        "protocol": "iid",
        "split_seed": split_seed,
        "classes": classes,
        "bucket_ids": ids_tr,
        "class_names": names_tr,
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
