"""ImageNet-100 class-incremental benchmark.

Class list: data_splits/imagenet100/classes.txt (PODNet/UCIR-style subset, "UCIR equivalence
(verify)"). Train data = ILSVRC2012 ``train/<wnid>/*``; test data = ILSVRC2012 official ``val``
(``val/<wnid>/*`` -- val must be reorganised into wnid sub-folders). Validation (if requested)
is carved from TRAIN only, in ``finalize``.

Completeness: ``<root>/.done`` (from ``scripts/download_data.sh imagenet100``) is required (see
``require_data_root``); every listed wnid must have a train and a val folder, and every val folder
exactly ``benchmark.val_images_per_class`` (default 50 -> 5000 val images) images.

Class order (follows the training seed; :func:`imagenet100_class_order`) over the sorted wnid
list, with ``class_order_seed`` = 1993 by default:
  * seed 0: ``np.random.RandomState(class_order_seed).permutation(100)`` -- the UCIR/PODNet order
    (``np.random.seed(1993); np.random.shuffle(order)`` gives the same permutation), so seed 0
    stays comparable with the literature; equivalence with the published order is "(verify)".
  * seed k > 0: ``np.random.default_rng([class_order_seed, k]).permutation(100)`` (the same
    convention as the DomainNet domain order).
Each seed therefore sees its own sequence of tasks (20 x 5 classes by default): task order and
initialisation vary together across seeds. The order is recorded in the meta (``class_order``,
``task_order``, ``task_order_sha256``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.utils.datasets_from_filelists import PathsDataset

from .common import (
    BenchmarkBundle,
    assert_disjoint,
    basename_key,
    build_transforms,
    check_manifest,
    finalize,
    make_cls_dataset,
    read_list_file,
    record_task_order,
    require_data_root,
    splits_dir,
)

_EXTS = (".jpeg", ".jpg", ".png")
DEFAULT_CLASS_ORDER_SEED = 1993


def _list_split(root: Path, split: str, wnids: list[str]) -> list[tuple[str, int]]:
    files: list[tuple[str, int]] = []
    for label, wnid in enumerate(wnids):
        d = root / split / wnid
        if not d.is_dir():
            raise FileNotFoundError(
                f"Missing ImageNet directory '{d}'. Expected ILSVRC2012/{split}/<wnid>/ layout "
                f"(val must be organised into wnid sub-folders)."
            )
        names = sorted(p.name for p in d.iterdir() if p.name.lower().endswith(_EXTS))
        if not names:
            raise FileNotFoundError(f"No images found in '{d}'")
        files.extend((f"{split}/{wnid}/{n}", label) for n in names)
    return files


def _check_val_counts(test_files: list[tuple[str, int]], wnids: list[str], per_class: int) -> None:
    counts = [0] * len(wnids)
    for _, y in test_files:
        counts[y] += 1
    bad = [(wnids[c], n) for c, n in enumerate(counts) if n != per_class]
    if bad:
        raise ValueError(
            f"ImageNet-100 val is incomplete/corrupt: expected {per_class} images per class "
            f"({per_class * len(wnids)} total), got {sum(counts)}; offending classes "
            f"(wnid, count): {bad[:10]}. Re-run `bash scripts/download_data.sh imagenet100`."
        )


def imagenet100_class_order(class_order_seed: int, seed: int = 0, n: int = 100) -> list[int]:
    """Class order of training seed ``seed`` (see the module docstring).

    Seed 0 keeps the committed UCIR/PODNet permutation; other seeds use a ``default_rng`` seeded
    with ``[class_order_seed, seed]``, so every (class_order_seed, seed) pair has its own order.
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    if seed == 0:
        perm = np.random.RandomState(int(class_order_seed)).permutation(n)
    else:
        perm = np.random.default_rng([int(class_order_seed), seed]).permutation(n)
    return [int(c) for c in perm]


def build_imagenet100(cfg, data_dir: str, seed: int) -> BenchmarkBundle:
    classes_file = splits_dir() / "imagenet100" / "classes.txt"
    classes_sha = check_manifest(classes_file, cfg.get("classes_sha256"))
    wnids = read_list_file(classes_file)
    if len(wnids) != 100 or len(set(wnids)) != 100:
        raise ValueError(f"{classes_file} must list 100 unique wnids, got {len(wnids)}")

    n_exp = int(cfg.n_experiences)
    if 100 % n_exp != 0:
        raise ValueError(f"n_experiences={n_exp} must divide 100 classes")

    root = Path(data_dir) / cfg.get("data_subdir", "ILSVRC2012")
    require_data_root(root, cfg, "ImageNet-100", "imagenet100")
    train_files = _list_split(root, "train", wnids)
    test_files = _list_split(root, "val", wnids)
    _check_val_counts(test_files, wnids, int(cfg.get("val_images_per_class", 50)))
    # Paths differ by construction (train/... vs val/...); compare file names instead, which
    # catches train images copied into val folders (ILSVRC train/val names never coincide).
    assert_disjoint(
        [f for f, _ in train_files], [f for f, _ in test_files], "ImageNet-100", key=basename_key
    )

    train_tf, eval_tf = build_transforms(cfg)
    train_ds = make_cls_dataset(
        PathsDataset(root, train_files),
        targets=[y for _, y in train_files],
        train_transform=train_tf,
        eval_transform=eval_tf,
        group="train",
    )
    test_ds = make_cls_dataset(
        PathsDataset(root, test_files),
        targets=[y for _, y in test_files],
        train_transform=train_tf,
        eval_transform=eval_tf,
        group="eval",
    )

    order_seed = cfg.get("class_order_seed", DEFAULT_CLASS_ORDER_SEED)
    order_seed = DEFAULT_CLASS_ORDER_SEED if order_seed is None else int(order_seed)
    class_order = imagenet100_class_order(order_seed, seed)
    # class_incremental_benchmark: avalanche/benchmarks/scenarios/supervised.py (v0.6.0);
    # `class_order` and `seed` are mutually exclusive there, so we pass only class_order.
    scen = class_incremental_benchmark(
        {"train": train_ds, "test": test_ds},
        class_order=class_order,
        num_experiences=n_exp,
    )
    # PM amendment A1: reuse this scenario's per-experience datasets, then finalize.
    train_list = [e.dataset for e in scen.train_stream]
    test_list = [e.dataset for e in scen.test_stream]

    meta = {
        "protocol": "class_incremental",
        "classes_file": str(classes_file),
        "classes_sha256": classes_sha,
        "class_order": class_order,
        "class_order_seed": order_seed,
        "n_experiences_full": n_exp,
        "classes_per_experience": 100 // n_exp,
        "seed": seed,
    }
    record_task_order(meta, class_order)
    benchmark = finalize(train_list, test_list, cfg, seed, meta)
    return BenchmarkBundle(
        benchmark=benchmark,
        scenario="class_il",
        num_classes=100,
        class_order=class_order,
        meta=meta,
    )
