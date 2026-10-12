"""Tiny synthetic class-incremental benchmark for tests / smoke runs (no I/O).

Random 3 x s x s tensors (s = ``image_size``, default 32), 10 classes, ``n_experiences``
(default 5) experiences of 10 / n classes. Class-conditional mean shift makes the problem
learnable. The class order follows the training seed like ImageNet-100
(:func:`synthetic_class_order`, ``class_order_seed`` default 1993): seed 0 =
``RandomState(class_order_seed).permutation(10)``, seed k > 0 =
``default_rng([class_order_seed, k]).permutation(10)``; the pixel data also come from the seed.
No ``.done`` marker is required (nothing is read from disk).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .common import BenchmarkBundle, finalize, make_cls_dataset, record_task_order

_NUM_CLASSES = 10
DEFAULT_CLASS_ORDER_SEED = 1993


def _make_split(classes: list[int], per_class: int, gen: torch.Generator, size: int):
    xs, ys = [], []
    for c in classes:
        x = torch.randn(per_class, 3, size, size, generator=gen) * 0.5 + 0.1 * (c - 4.5)
        xs.append(x)
        ys.extend([c] * per_class)
    return TensorDataset(torch.cat(xs), torch.tensor(ys, dtype=torch.long))


def synthetic_class_order(class_order_seed: int, seed: int = 0) -> list[int]:
    """Class order of training seed ``seed`` (same convention as ImageNet-100)."""
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    if seed == 0:
        perm = np.random.RandomState(int(class_order_seed)).permutation(_NUM_CLASSES)
    else:
        perm = np.random.default_rng([int(class_order_seed), seed]).permutation(_NUM_CLASSES)
    return [int(c) for c in perm]


def build_synthetic(cfg, data_dir: str, seed: int) -> BenchmarkBundle:
    n_exp = int(cfg.get("n_experiences", 5))
    if _NUM_CLASSES % n_exp != 0:
        raise ValueError(f"n_experiences={n_exp} must divide {_NUM_CLASSES}")
    per = _NUM_CLASSES // n_exp
    n_train = int(cfg.get("train_per_class", 20))
    n_test = int(cfg.get("test_per_class", 10))
    size = int(cfg.get("image_size", 32) or 32)

    order_seed = cfg.get("class_order_seed", DEFAULT_CLASS_ORDER_SEED)
    order_seed = DEFAULT_CLASS_ORDER_SEED if order_seed is None else int(order_seed)
    class_order = synthetic_class_order(order_seed, seed)

    gen = torch.Generator().manual_seed(int(seed))
    train_list, test_list = [], []
    for e in range(n_exp):
        classes = class_order[e * per : (e + 1) * per]
        tr = _make_split(classes, n_train, gen, size)
        te = _make_split(classes, n_test, gen, size)
        train_list.append(make_cls_dataset(tr, tr.tensors[1].tolist(), group="train"))
        test_list.append(make_cls_dataset(te, te.tensors[1].tolist(), group="eval"))

    meta = {
        "protocol": "synthetic_class_incremental",
        "class_order": class_order,
        "class_order_seed": order_seed,
        "image_size": size,
        "n_experiences_full": n_exp,
        "seed": int(seed),
    }
    record_task_order(meta, class_order)
    benchmark = finalize(train_list, test_list, cfg, int(seed), meta)
    return BenchmarkBundle(
        benchmark=benchmark,
        scenario="class_il",
        num_classes=_NUM_CLASSES,
        class_order=class_order,
        meta=meta,
    )
