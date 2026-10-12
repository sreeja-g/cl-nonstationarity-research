"""Shared helpers for the benchmark data layer.

Avalanche API facts relied upon (verified against tag v0.6.0, not executed here):
  * ``benchmark_from_datasets(**streams)`` and ``split_validation_class_balanced``:
    avalanche/benchmarks/scenarios/dataset_scenario.py
  * ``_make_taskaware_classification_dataset`` (private):
    avalanche/benchmarks/utils/classification_dataset.py
  * ``with_classes_timeline(scenario)``: avalanche/benchmarks/scenarios/supervised.py; adds
    ``classes_in_this_experience`` (a ``set``), ``previous_classes``, ``classes_seen_so_far``,
    ``future_classes`` to (copies of) every experience and rebuilds EagerCLStreams.
  * per-experience data: ``benchmark.<name>_stream[i].dataset`` (an AvalancheDataset);
    class labels: ``dataset.targets`` (DataAttribute, supports ``.uniques``, ``.val_to_idx``);
    task labels: ``dataset.targets_task_labels``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from avalanche.benchmarks.scenarios.dataset_scenario import (
    benchmark_from_datasets,
    split_validation_class_balanced,
)
from avalanche.benchmarks.scenarios.supervised import with_classes_timeline
from avalanche.benchmarks.utils.classification_dataset import (
    _make_taskaware_classification_dataset,
)


@dataclass
class BenchmarkBundle:
    benchmark: Any  # avalanche CLScenario with .train_stream, .test_stream, optional .valid_stream
    scenario: str  # "class_il" | "domain_il"
    num_classes: int  # total output units
    class_order: list[int] | None  # class-IL only
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Paths / hashing
# --------------------------------------------------------------------------- #
def splits_dir() -> Path:
    """Directory holding committed split files (``<repo>/data_splits``).

    Override with env ``CLNS_SPLITS_DIR``. The default assumes an editable (src-layout)
    install: this file is ``<repo>/src/clns/data/common.py``.
    """
    env = os.environ.get("CLNS_SPLITS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "data_splits"


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a text manifest, with CRLF normalised to LF (git autocrlf safe)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def check_manifest(path: str | Path, sha256: str | None) -> str:
    """Return the manifest hash; raise ValueError if ``sha256`` is given and differs."""
    actual = sha256_file(path)
    if sha256 is not None and actual != str(sha256).lower():
        raise ValueError(f"Manifest {path} has sha256 {actual}, expected {sha256}")
    return actual


def read_list_file(path: str | Path) -> list[str]:
    """Non-empty, non-comment ('#') lines of a list file, stripped."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def record_task_order(meta: dict, order: Sequence[Any]) -> None:
    """Record the run's task order in ``meta`` (``task_order`` + ``task_order_sha256``).

    The task order follows the training seed (ImageNet-100 class order, DomainNet domain order,
    CORe50 NIC run); CLEAR-10 keeps its time order. The hash ends in ``sha256``, so it is one of
    the ``split_hashes`` that ``clns.shift.measure_stream`` records and
    ``clns.analysis.correlation_study.shift_mismatch`` compares: a shift matrix measured on
    another order never pairs with the run. ``order`` must be JSON-serialisable (ints, strings or
    nested lists of them).
    """
    import json

    order = [list(o) if isinstance(o, (list, tuple)) else o for o in order]
    meta["task_order"] = order
    canon = json.dumps(order, separators=(",", ":"), default=str)
    meta["task_order_sha256"] = hashlib.sha256(canon.encode("utf-8")).hexdigest()


DONE_MARKER = ".done"


def require_data_root(root: str | Path, cfg, name: str, download_name: str) -> Path:
    """Refuse missing or incomplete data before any loader touches it.

    * ``<root>/.done`` (written by ``scripts/download_data.sh <download_name>`` after a complete
      download) must exist, unless ``benchmark.allow_download=true`` (CLEAR/CORe50: Avalanche
      hard-codes download=True and may then fetch the data itself) or
      ``benchmark.require_done_marker=false``.
    * Without the marker requirement the directory must at least exist and be non-empty.
    """
    root = Path(root)
    hint = f"Run `bash scripts/download_data.sh {download_name}`"
    if bool(cfg.get("allow_download", False)):
        return root
    present = root.is_dir() and any(root.iterdir())
    if not present:
        raise FileNotFoundError(
            f"{name} data not found at '{root}'. {hint} (or, for CLEAR/CORe50, set "
            f"benchmark.allow_download=true to let Avalanche download it)."
        )
    if bool(cfg.get("require_done_marker", True)) and not (root / DONE_MARKER).is_file():
        raise FileNotFoundError(
            f"{name} data at '{root}' has no '{DONE_MARKER}' completion marker (download "
            f"incomplete, or not made by the download script). {hint}. If the data is known to "
            f"be complete, create '{root / DONE_MARKER}' or set "
            f"benchmark.require_done_marker=false."
        )
    return root


def basename_key(path) -> str:
    """File name without directories (``a/b/c.JPEG`` -> ``c.JPEG``); '/' or '\\' separated."""
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def assert_disjoint(
    train_paths: Sequence[str], test_paths: Sequence[str], what: str = "", key=None
) -> None:
    """Raise ValueError (listing a few overlaps) if any file is in both train and test.

    ``key`` maps each path to the identity that is compared (default: the path string). Use
    :func:`basename_key` when train and test live under different split folders (e.g.
    ``train/<wnid>/x.JPEG`` vs ``val/<wnid>/x.JPEG``), where full paths can never collide.
    """
    key = key or str
    overlap = set(map(key, train_paths)) & set(map(key, test_paths))
    if overlap:
        shown = sorted(overlap)[:5]
        where = f" in {what}" if what else ""
        raise ValueError(
            f"Train/test leakage{where}: {len(overlap)} shared files, e.g. {shown}"
        )


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
NORMALIZATION = {
    # torchvision ImageNet statistics: scratch arms, and the statistics the Places365 (CSAIL)
    # and torchvision ImageNet checkpoints were trained with.
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


def build_transform(name: str | None, image_size: int, normalization: str = "imagenet"):
    """Named transform presets (config values ``train_transform`` / ``eval_transform``).

    None / "none"  -> no transform (data already tensors, e.g. synthetic)
    imagenet_train -> RandomResizedCrop + flip + normalize
    imagenet_eval  -> Resize(size / 0.875) + CenterCrop + normalize
    resize_train   -> Resize((s, s)) + flip + normalize   (non-object-centric images)
    resize_eval    -> Resize((s, s)) + normalize
    ``normalization`` (config ``benchmark.normalization``): "imagenet" (default). train.py
    copies a pretrained arm's ``model.normalization`` / ``model.image_size`` here.
    """
    if name is None or str(name).lower() == "none":
        return None
    from torchvision import transforms as T

    norm_name = str(normalization or "imagenet").lower()
    if norm_name not in NORMALIZATION:
        raise ValueError(
            f"Unknown normalization '{normalization}'. Choose from {sorted(NORMALIZATION)}."
        )
    mean, std = NORMALIZATION[norm_name]
    norm = T.Normalize(mean=mean, std=std)
    s = int(image_size)
    presets = {
        "imagenet_train": [T.RandomResizedCrop(s), T.RandomHorizontalFlip(), T.ToTensor(), norm],
        "imagenet_eval": [T.Resize(int(round(s / 0.875))), T.CenterCrop(s), T.ToTensor(), norm],
        "resize_train": [T.Resize((s, s)), T.RandomHorizontalFlip(), T.ToTensor(), norm],
        "resize_eval": [T.Resize((s, s)), T.ToTensor(), norm],
    }
    if name not in presets:
        raise ValueError(f"Unknown transform '{name}'. Choose from {sorted(presets)} or none.")
    return T.Compose(presets[name])


def build_transforms(cfg) -> tuple[Any, Any]:
    size = int(cfg.get("image_size", 224))
    norm = cfg.get("normalization", "imagenet")
    return (
        build_transform(cfg.get("train_transform"), size, norm),
        build_transform(cfg.get("eval_transform"), size, norm),
    )


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def make_cls_dataset(ds, targets=None, train_transform=None, eval_transform=None, group="train"):
    """Wrap a raw torch dataset (or AvalancheDataset) as a task-label-0 classification dataset.

    ``train_transform``/``eval_transform`` become the 'train'/'eval' transform groups and
    ``group`` selects the initial one. Uses Avalanche's private
    ``_make_taskaware_classification_dataset`` (v0.6.0, classification_dataset.py).
    """
    kwargs: dict[str, Any] = dict(task_labels=0)
    if targets is not None:
        kwargs["targets"] = list(targets)
    if train_transform is not None or eval_transform is not None:
        kwargs["transform_groups"] = {
            "train": (train_transform, None),
            "eval": (eval_transform, None),
        }
        kwargs["initial_transform_group"] = group
    return _make_taskaware_classification_dataset(ds, **kwargs)


def force_task_label_zero(ds):
    """Return ``ds`` with every task label set to 0 (keeps targets and transform groups)."""
    return _make_taskaware_classification_dataset(ds, task_labels=0)


def subsample_per_class(ds, n: int, seed: int, stable_key: str | None = None):
    """Keep at most ``n`` samples per class (deterministic given ``seed``).

    ``stable_key`` given: every class ``c`` draws from its own generator seeded with
    :func:`stable_subsample_seed` ``(seed, f"{stable_key}|class:{c}")``, so a class keeps the same
    images whichever experience / stream position it lands in (its images must keep their
    relative order, which ``subset`` of a common parent dataset guarantees).
    """
    n = int(n)
    if n < 1:
        raise ValueError("subsample_per_class must be >= 1")
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    val_to_idx = ds.targets.val_to_idx
    for cls in sorted(val_to_idx):
        idxs = list(val_to_idx[cls])
        if stable_key is not None:
            rng = np.random.default_rng(
                stable_subsample_seed(seed, f"{stable_key}|class:{int(cls)}")
            )
        if len(idxs) > n:
            pick = rng.choice(len(idxs), size=n, replace=False)
            idxs = [idxs[i] for i in sorted(pick.tolist())]
        keep.extend(idxs)
    keep.sort()
    return ds.subset(keep)


def dataset_classes(ds) -> list[int]:
    return sorted(int(c) for c in ds.targets.uniques)


def stable_subsample_seed(subsample_seed: int, identity: str) -> int:
    """64-bit seed from ``subsample_seed`` and a STABLE identity string (sha256-derived)."""
    h = hashlib.sha256(f"{int(subsample_seed)}|{identity}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


def _normalise_classes_timeline(scenario) -> None:
    """Store ``classes_in_this_experience`` as a sorted list of ints (not a ``set``).

    ``with_classes_timeline`` sets it to ``dataset.targets.uniques`` (a ``set``), whose
    iteration order is arbitrary and which numpy turns into a 0-d object array (e.g. in
    ``np.setdiff1d``); a sorted list is deterministic and array-safe for any consumer.
    ``EagerCLStream[i]`` returns the stored experience object, so the assignment below is
    persistent.
    """
    for stream in scenario.streams.values():
        for i in range(len(stream)):
            exp = stream[i]
            exp.classes_in_this_experience = sorted(
                int(c) for c in exp.classes_in_this_experience
            )


def finalize(
    train_list: Sequence[Any],
    test_list: Sequence[Any],
    cfg,
    seed: int,
    meta: dict,
    truncate_test: bool = True,
    test_identities: Sequence[str] | None = None,
):
    """Uniform post-processing used by every loader.

    Order: task label -> 0, truncate to ``max_experiences`` (test too iff ``truncate_test``),
    ``subsample_per_class`` (train: from the training ``seed``; TEST: from the fixed
    ``benchmark.subsample_seed``, default 0, seeded PER CLASS by a stable identity -- class-IL:
    the class id; domain-IL: ``test_identities[j]`` (the domain / bucket name the loader passes)
    plus the class id -- never by the experience position, so every training seed, whatever its
    task order or grouping of classes into experiences, is evaluated on the same test images),
    validation carved from TRAIN only (``val_fraction`` > 0, class-balanced, eval
    transforms), then ``benchmark_from_datasets`` + ``with_classes_timeline`` (every experience
    of every stream gets ``classes_in_this_experience`` etc.; needed by the eval-time
    class mask). Fills size bookkeeping and ``meta["train_classes"]`` (sorted classes
    of each final training experience) in ``meta``. Returns the CLScenario.
    """
    train = [force_task_label_zero(d) for d in train_list]
    test = [force_task_label_zero(d) for d in test_list]

    max_exp = cfg.get("max_experiences")
    if max_exp is not None:
        max_exp = int(max_exp)
        if max_exp < 1:
            raise ValueError("max_experiences must be >= 1 or null")
        train = train[:max_exp]
        if truncate_test:
            test = test[:max_exp]

    sub = cfg.get("subsample_per_class")
    subsample_seed = int(cfg.get("subsample_seed", 0) or 0)
    if sub is not None:
        train = [subsample_per_class(d, sub, seed + i) for i, d in enumerate(train)]
        if test_identities is None:  # class-IL: a class id is unique to one experience
            ids = ["class_il"] * len(test)
        else:
            ids = [str(x) for x in test_identities][: len(test)]
            if len(ids) != len(test) or len(set(ids)) != len(ids):
                raise ValueError(f"need {len(test)} unique test identities, got {ids}")
        test = [
            subsample_per_class(d, sub, subsample_seed, stable_key=ident)
            for d, ident in zip(test, ids)
        ]

    streams: dict[str, list] = {}
    val_fraction = float(cfg.get("val_fraction", 0.0) or 0.0)
    if val_fraction > 0.0:
        new_train, valid = [], []
        # split_validation_class_balanced shuffles with the *global* torch RNG, so seed it
        # per experience and restore the caller's RNG state afterwards.
        state = torch.get_rng_state()
        try:
            for i, d in enumerate(train):
                torch.manual_seed(seed * 1000 + i)
                tr, va = split_validation_class_balanced(val_fraction, d)
                new_train.append(tr)
                valid.append(va.eval())
        finally:
            torch.set_rng_state(state)
        train = new_train
        streams["valid"] = valid
    streams["train"] = train
    streams["test"] = test

    meta["n_train_experiences"] = len(train)
    meta["n_test_experiences"] = len(test)
    meta["train_sizes"] = [len(d) for d in train]
    # Classes of every FINAL training experience (after truncation/subsampling/validation
    # split); consumed by the class-IL eval mask (EvalClassMaskPlugin via build_evaluator).
    meta["train_classes"] = [dataset_classes(d) for d in train]
    meta["test_sizes"] = [len(d) for d in test]
    if "valid" in streams:
        meta["valid_sizes"] = [len(d) for d in streams["valid"]]
    meta.setdefault("test_exp_to_train_exp", list(range(len(test))))
    meta["val_fraction"] = val_fraction
    meta["subsample_per_class"] = sub
    meta["subsample_seed"] = subsample_seed
    meta["max_experiences"] = max_exp
    scenario = with_classes_timeline(benchmark_from_datasets(**streams))
    _normalise_classes_timeline(scenario)
    return scenario
