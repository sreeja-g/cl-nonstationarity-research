"""Fixed probe set + penultimate-feature dumps for offline CKA (representation drift).

After every training experience train.py extracts the model's penultimate features (the input
of the LAST ``nn.Linear`` in module order: ResNet ``fc``, MobileNetV2 ``classifier[-1]``) on a
fixed probe set and writes them to ``<run_dir>/probe/`` (one file per row, see below). Linear
CKA between rows (``clns.shift.metrics.linear_cka``) is computed offline by the analysis.

Probe set (:func:`build_probe_set`): ``max_images`` test images in total, allocated per CLASS
over the whole test stream: every class gets ``max_images // n_classes`` images, the remainder
going to the classes with the smallest ``sha256(f"{probe_seed}:class:{c}")``. Within a class the
images are ranked by ``sha256(f"{probe_seed}:{image id}")`` (image id =
:func:`clns.shift.feature_cache.sample_ids`, made relative to ``data_root``) and, when the class
occurs in several test experiences (domain-IL), interleaved round-robin over them (by rank inside
each (class, experience), ties by hash). The selection depends only on ``probe_seed`` and the SET
of test images of each class, never on the training seed, the task order or how the test set is
split into experiences, so the probe images are identical across seeds/runs of a benchmark (e.g.
CORe50, whose first-appearance test groups change with the run; ~41 groups for NICv2-79, one
10-class group then mostly 1-class groups, which an even per-experience split would starve).
Classes with fewer images than their quota contribute all of them (nothing is redistributed).
``test_exp`` / ``index`` record where each chosen image sits in THIS run's test stream.

Storage (``<run_dir>/probe/``; every file written to a temporary file, fsynced, ``os.replace``-d):
  * ``meta.npz`` written once (later writers verify it is identical):
      ``labels`` int64 (P,), ``test_exp`` int64 (P,), ``index`` int64 (P,) (index inside that
      test experience's dataset), ``image_id`` unicode (P,), ``probe_seed`` int64 scalar;
  * ``row_{r:04d}.npz`` one per row: ``features`` float16 (P, D) after training experience r
    (0-based) and ``exp_index`` int64 scalar (= r).
On resume :func:`truncate_probe` deletes the rows ``>= n_done`` (like the accuracy matrix), so the
rows are exactly ``0 .. n_done-1`` after every committed experience. :func:`load_probe` stacks
the rows to ``features`` (n_rows, P, D) and RAISES unless they are exactly ``0 .. n_rows-1``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

PROBE_DIR = "probe"
META_FILE = "meta.npz"
_ROW_RE = re.compile(r"^row_(\d+)\.npz$")


@dataclass
class ProbeSet:
    dataset: torch.utils.data.Dataset  # concatenation of eval-mode subsets
    labels: np.ndarray
    test_exp: np.ndarray
    index: np.ndarray
    probe_seed: int
    image_id: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def meta_arrays(self) -> dict:
        ids = self.image_id if self.image_id is not None else np.asarray([], dtype=str)
        return {
            "labels": np.asarray(self.labels, dtype=np.int64),
            "test_exp": np.asarray(self.test_exp, dtype=np.int64),
            "index": np.asarray(self.index, dtype=np.int64),
            "image_id": np.asarray(ids, dtype=str),
            "probe_seed": np.asarray(self.probe_seed, dtype=np.int64),
        }


def _quotas(total: int, n_parts: int) -> list[int]:
    base, rem = divmod(int(total), int(n_parts))
    return [base + (1 if j < rem else 0) for j in range(n_parts)]


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_probe_set(
    test_stream, max_images: int, probe_seed: int, data_root: str | Path | None = None
) -> ProbeSet:
    """Fixed probe set over ``test_stream`` (see the module docstring).

    Every chosen image is taken from the experience dataset in its EVAL transform group.
    ``data_root`` makes path-based image ids relative (moving the data keeps the selection).
    """
    from clns.shift.feature_cache import sample_ids

    max_images = int(max_images)
    probe_seed = int(probe_seed)
    if max_images < 1:
        raise ValueError(f"max_images must be >= 1, got {max_images}")
    n_test = len(test_stream)
    if n_test < 1:
        raise ValueError("empty test stream")
    datasets, targets = [], []
    # class -> test experience -> [(hash key, position, image id)]
    by_class: dict[int, dict[int, list[tuple[str, int, str]]]] = {}
    for j in range(n_test):
        ds = test_stream[j].dataset.eval()
        ys = [int(y) for y in ds.targets]
        ids = sample_ids(ds, data_root)
        datasets.append(ds)
        targets.append(ys)
        for pos, (y, sid) in enumerate(zip(ys, ids)):
            by_class.setdefault(y, {}).setdefault(j, []).append(
                (_hash(f"{probe_seed}:{sid}"), pos, sid)
            )
    classes = sorted(by_class, key=lambda c: _hash(f"{probe_seed}:class:{c}"))
    quota = dict(zip(classes, _quotas(max_images, len(classes))))
    chosen: list[tuple[int, int, str]] = []  # (test_exp, position, image id)
    for c in sorted(by_class):
        ranked = []
        for j, items in by_class[c].items():
            items.sort()
            ranked.extend((rank, h, j, pos, sid) for rank, (h, pos, sid) in enumerate(items))
        ranked.sort()  # round-robin over the class's experiences, ties by hash
        chosen.extend((j, pos, sid) for _, _, j, pos, sid in ranked[: quota[c]])
    chosen.sort()
    subsets, labels, exps, index = [], [], [], []
    for j in range(n_test):
        pos = [p for jj, p, _ in chosen if jj == j]
        if not pos:
            continue
        subsets.append(datasets[j].subset(pos))
        labels.extend(targets[j][p] for p in pos)
        exps.extend([j] * len(pos))
        index.extend(pos)
    return ProbeSet(
        dataset=torch.utils.data.ConcatDataset(subsets),
        labels=np.asarray(labels, dtype=np.int64),
        test_exp=np.asarray(exps, dtype=np.int64),
        index=np.asarray(index, dtype=np.int64),
        probe_seed=probe_seed,
        image_id=np.asarray([sid for _, _, sid in chosen], dtype=str),
    )


def penultimate_layer(model: nn.Module) -> nn.Linear:
    """The last ``nn.Linear`` in module order (the classification head)."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise ValueError(f"{type(model).__name__} has no nn.Linear head to probe")
    return last


@torch.no_grad()
def extract_probe(
    model: nn.Module, probe: ProbeSet, device, batch_size: int = 256, num_workers: int = 0
) -> np.ndarray:
    """(P, D) float32 penultimate features of the probe set.

    Eval mode + no_grad; every module's train/eval flag is restored afterwards, so the strategy
    sees the model exactly as it left it.
    """
    head = penultimate_layer(model)
    captured: list[torch.Tensor] = []

    def _hook(_module, inputs):
        captured.append(inputs[0].detach().float().flatten(1).cpu())

    flags = {m: m.training for m in model.modules()}
    handle = head.register_forward_pre_hook(_hook)
    try:
        model.eval()
        loader = torch.utils.data.DataLoader(
            probe.dataset, batch_size=int(batch_size), shuffle=False,
            num_workers=int(num_workers),
        )
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            model(x.to(device))
    finally:
        handle.remove()
        for m, flag in flags.items():
            m.training = flag
    feats = torch.cat(captured, dim=0).numpy()
    if feats.shape[0] != len(probe):
        raise RuntimeError(f"probe extraction returned {feats.shape[0]} rows, expected "
                           f"{len(probe)}")
    return feats


def _write_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _row_files(probe_dir: Path) -> dict[int, Path]:
    out = {}
    for f in probe_dir.glob("row_*.npz"):
        m = _ROW_RE.match(f.name)
        if m:
            out[int(m.group(1))] = f
    return out


def row_path(probe_dir: str | Path, r: int) -> Path:
    return Path(probe_dir) / f"row_{int(r):04d}.npz"


def write_probe_meta(probe_dir: str | Path, probe: ProbeSet) -> None:
    """Write ``meta.npz`` once; if it exists, refuse a different probe set (ValueError)."""
    probe_dir = Path(probe_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    path = probe_dir / META_FILE
    meta = probe.meta_arrays()
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            for k, v in meta.items():
                if k not in z.files or not np.array_equal(z[k], v):
                    raise ValueError(
                        f"{path}: stored probe '{k}' differs from the current probe set"
                    )
        return
    _write_npz(path, **meta)


def write_probe_row(probe_dir: str | Path, r: int, feats: np.ndarray, probe: ProbeSet) -> Path:
    """Write row ``r`` (features after training experience ``r``) as ``row_{r:04d}.npz``.

    Writes/verifies ``meta.npz`` first, so rows of different probe sets never mix.
    """
    feats = np.asarray(feats, dtype=np.float16)
    if feats.ndim != 2 or feats.shape[0] != len(probe):
        raise ValueError(f"features must be ({len(probe)}, D), got {feats.shape}")
    write_probe_meta(probe_dir, probe)
    path = row_path(probe_dir, r)
    _write_npz(path, features=feats, exp_index=np.asarray(int(r), dtype=np.int64))
    return path


def load_probe(probe_dir: str | Path) -> dict:
    """Meta arrays + ``features`` (n_rows, P, D) float16 + ``exp_index`` (n_rows,).

    Raises ValueError unless the row files are exactly ``row_0000 .. row_{n-1}`` with matching
    ``exp_index`` and shapes (a gap means rows would not align with experiences).
    """
    probe_dir = Path(probe_dir)
    meta_path = probe_dir / META_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(f"no probe meta at {meta_path}")
    with np.load(meta_path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    rows = _row_files(probe_dir)
    if sorted(rows) != list(range(len(rows))):
        raise ValueError(
            f"{probe_dir}: probe rows {sorted(rows)} are not exactly 0..{len(rows) - 1} "
            "(probe disabled for some experiences, or a stale/missing row); rows would not "
            "align with experiences"
        )
    n_img = int(out["labels"].shape[0])
    feats, exp_index = [], []
    for r in range(len(rows)):
        with np.load(rows[r], allow_pickle=False) as z:
            f, e = z["features"], int(z["exp_index"])
        if e != r or f.ndim != 2 or f.shape[0] != n_img or (feats and f.shape != feats[0].shape):
            raise ValueError(f"{rows[r]}: exp_index {e} / shape {f.shape} inconsistent with "
                             f"row {r} of a {n_img}-image probe")
        feats.append(f)
        exp_index.append(e)
    out["features"] = np.stack(feats) if feats else np.zeros((0, n_img, 0), dtype=np.float16)
    out["exp_index"] = np.asarray(exp_index, dtype=np.int64)
    return out


def truncate_probe(probe_dir: str | Path, n_done: int) -> None:
    """Delete rows ``>= n_done`` (rows after the last checkpoint are stale) and leftover
    ``*.tmp`` files; ``n_done <= 0`` removes ``meta.npz`` too. Missing directory: no-op."""
    probe_dir = Path(probe_dir)
    if not probe_dir.is_dir():
        return
    n_done = int(n_done)
    for f in probe_dir.glob("*.tmp"):
        f.unlink()
    for r, f in _row_files(probe_dir).items():
        if r >= n_done:
            f.unlink()
    if n_done <= 0 and (probe_dir / META_FILE).exists():
        (probe_dir / META_FILE).unlink()


__all__ = [
    "META_FILE",
    "PROBE_DIR",
    "ProbeSet",
    "build_probe_set",
    "extract_probe",
    "load_probe",
    "penultimate_layer",
    "row_path",
    "truncate_probe",
    "write_probe_meta",
    "write_probe_row",
]
