"""CORe50 NIC (default NICv2-79) benchmark with first-appearance test groups.

Avalanche's CORe50 (classic/core50.py, v0.6.0) exposes ONE shared test experience (sessions
3, 7, 10) -- a fixed test set. For the forgetting estimand (RESEARCH_LOG 7.2.1) the test set is
partitioned by the training experience in which each class FIRST appears: test experience j holds
the test frames of the classes first seen in train experience ``test_exp_to_train_exp[j]``;
empty groups are dropped. Train and test streams therefore generally have different lengths.

Validation: ``val_fraction > 0`` raises (a frame-level split leaks near-duplicate consecutive
frames). Use ``benchmark.val_sessions`` instead: whole training sessions (of {1, 2, 4, 5, 6, 8, 9,
11}; sessions 3, 7, 10 are the test set) are held out of EVERY training batch and form the
validation stream (:func:`holdout_sessions`); the frames of a held-out session are never trained
on, so tuning trains on the remaining sessions only. Validation experiences use the same
first-appearance grouping as the test stream (``valid_exp_to_train_exp``); their frames come
from ALL batches (also later ones), but the classes were already seen in training. Production
runs keep ``val_sessions: null`` (all training sessions are used, no validation stream).
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

import re
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


TRAIN_SESSIONS = (1, 2, 4, 5, 6, 8, 9, 11)  # CORe50 test sessions are 3, 7, 10

_SESSION_DIR = re.compile(r"^s(\d+)/")
_SESSION_FILE = re.compile(r"^C_(\d+)_")


def core50_session(path) -> int:
    """Acquisition session (1..11) of a CORe50 frame path ``s<S>/o<O>/C_<SS>_<OO>_<FFF>.png``.

    The session is read from the leading ``s<S>/`` directory and cross-checked with the file
    name's ``C_<SS>_`` (the Avalanche CORe50 filelists, v0.6.0, are relative to
    ``core50_128x128``); backslashes are normalised to ``/``. Raises ValueError on any other
    format or on a mismatch.
    """
    p = str(path).replace("\\", "/")
    m = _SESSION_DIR.match(p)
    f = _SESSION_FILE.match(p.rsplit("/", 1)[-1])
    if m is None or f is None:
        raise ValueError(f"Unrecognised CORe50 frame path '{path}' (expected s<S>/o<O>/C_<SS>_...)")
    if int(m.group(1)) != int(f.group(1)):
        raise ValueError(
            f"CORe50 path '{path}': session directory s{m.group(1)} != file name session "
            f"C_{f.group(1)}"
        )
    return int(m.group(1))


def _core50_base(root: Path, variant: str, run: int) -> Path:
    """Filelist directory used by ``avalanche/benchmarks/classic/core50.py`` (v0.6.0)."""
    from avalanche.benchmarks.classic.core50 import scen2dirs

    return root / (scen2dirs[variant][:-1] + "/" + "run" + str(run))


def _read_train_paths(root: Path, variant: str, run: int) -> list[list[str]]:
    """Frame paths of every train batch, in filelist order (one list per batch).

    Re-reads the filelists Avalanche's CORe50 (v0.6.0) uses:
    ``<root>/<scen2dirs[variant] without '/'>/run<run>/train_batch_XX_filelist.txt``
    (object_lvl=True), parsed with ``default_flist_reader``.
    ``create_generic_benchmark_from_filelists`` builds one ``FilelistDataset`` per file, in the
    same order (scenarios/deprecated/generic_benchmark_creation.py), so ``paths[b][i]`` is
    sample ``i`` of ``train_stream[b].dataset``.
    """
    from avalanche.benchmarks.classic.core50 import nbatch
    from avalanche.benchmarks.utils.datasets_from_filelists import default_flist_reader

    base = _core50_base(root, variant, run)
    return [
        [p for p, _ in default_flist_reader(base / f"train_batch_{str(b).zfill(2)}_filelist.txt")]
        for b in range(nbatch[variant])
    ]


def _check_core50_disjoint(root: Path, variant: str, run: int) -> str:
    """Assert no frame of the test filelist (sessions 3, 7, 10) is in any train filelist.

    Returns the sha256 of the test filelist (:func:`sha256_file`, CRLF-normalised).
    """
    from avalanche.benchmarks.utils.datasets_from_filelists import default_flist_reader

    base = _core50_base(root, variant, run)
    train_paths = [p for batch in _read_train_paths(root, variant, run) for p in batch]
    test_paths = [p for p, _ in default_flist_reader(base / "test_filelist.txt")]
    assert_disjoint(train_paths, test_paths, f"CORe50 {variant} run {run}")
    return sha256_file(base / "test_filelist.txt")


def parse_val_sessions(value) -> list[int] | None:
    """``benchmark.val_sessions`` -> sorted list of ints, or None (no validation stream).

    Accepts only a list/tuple (or OmegaConf ListConfig) of non-empty ints; a scalar, a string, a
    bool or an empty list raises a ValueError (``val_sessions: 1`` would otherwise fail obscurely
    or, as a string, split into characters).
    """
    if value is None:
        return None
    from omegaconf import ListConfig

    if not isinstance(value, (list, tuple, ListConfig)):
        raise ValueError(
            f"CORe50 benchmark.val_sessions must be a list of ints like [1] or null, got "
            f"{value!r} ({type(value).__name__})"
        )
    items = list(value)
    if not items:
        raise ValueError("CORe50 val_sessions is empty; use null for no validation stream")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in items):
        raise ValueError(f"CORe50 val_sessions must contain ints only, got {items!r}")
    return sorted({int(x) for x in items})


def core50_session_check(
    paths: Sequence[Sequence[str]], val_sessions: Sequence[int] | None = None,
    n_used: int | None = None,
) -> list[dict]:
    """Per-batch session counts of the first ``n_used`` batches (all if None).

    One dict per batch: ``batch``, ``sessions`` ({session: n frames}), ``n_frames``,
    ``n_heldout`` / ``n_kept`` for ``val_sessions`` and ``empties_batch`` (True if holding the
    sessions out leaves the batch without training frames). Use it on the real filelists to
    verify the session choice (NICv2 batches are whole (object, session) units, so a batch may
    consist of a single session); see the CLI ``--check-sessions`` below.
    """
    held = set(int(s) for s in (val_sessions or ()))
    out = []
    for b, ps in enumerate(paths if n_used is None else paths[: int(n_used)]):
        counts: dict[int, int] = {}
        for p in ps:
            s = core50_session(p)
            counts[s] = counts.get(s, 0) + 1
        n_held = sum(c for s, c in counts.items() if s in held)
        out.append({
            "batch": b, "sessions": dict(sorted(counts.items())), "n_frames": len(ps),
            "n_heldout": n_held, "n_kept": len(ps) - n_held,
            "empties_batch": len(ps) > 0 and len(ps) == n_held,
        })
    return out


def holdout_sessions(
    train_full: Sequence,
    paths: Sequence[Sequence[str]],
    val_sessions: Sequence[int],
    n_used: int | None = None,
):
    """Hold whole sessions out of every training batch.

    :param train_full: one dataset per train batch.
    :param paths: frame paths per batch, aligned with ``train_full`` (:func:`_read_train_paths`).
    :param val_sessions: sessions to hold out (subset of :data:`TRAIN_SESSIONS`).
    :param n_used: number of leading batches that are trained on (``max_experiences``); only
        these must keep at least one frame. A batch beyond that range may become empty (the
        stream keeps its length; such a batch is never used).
    :return: (train_kept, heldout) -- per batch the dataset without / with the held-out
        sessions' frames (a held-out dataset may be empty).
    :raises ValueError: unknown session, nothing held out, empty used training batch, length
        mismatch, or any validation frame also in the kept training frames.
    """
    sessions = sorted({int(s) for s in val_sessions})
    bad = [s for s in sessions if s not in TRAIN_SESSIONS]
    if bad:
        raise ValueError(
            f"CORe50 val_sessions {bad} not in the training sessions {list(TRAIN_SESSIONS)} "
            f"(3, 7, 10 are the test sessions)"
        )
    if len(paths) != len(train_full):
        raise ValueError(f"{len(paths)} filelists vs {len(train_full)} train batches")
    n_used = len(train_full) if n_used is None else int(n_used)
    held = set(sessions)
    kept_ds, held_ds = [], []
    kept_paths: list[str] = []
    held_paths: list[str] = []
    for b, (ds, ps) in enumerate(zip(train_full, paths)):
        if len(ps) != len(ds):
            raise ValueError(f"CORe50 batch {b}: {len(ps)} filelist entries vs {len(ds)} samples")
        sess = [core50_session(p) for p in ps]
        keep = [i for i, s in enumerate(sess) if s not in held]
        out = [i for i, s in enumerate(sess) if s in held]
        if not keep and b < n_used:
            raise ValueError(
                f"CORe50 val_sessions={sessions} leave training batch {b} without frames; "
                f"choose other sessions"
            )
        kept_ds.append(ds.subset(keep))
        held_ds.append(ds.subset(out))
        kept_paths.extend(ps[i] for i in keep)
        held_paths.extend(ps[i] for i in out)
    if not held_paths:
        raise ValueError(f"CORe50 val_sessions={sessions}: no frame of these sessions in train")
    assert_disjoint(kept_paths, held_paths, f"CORe50 validation sessions {sessions}")
    return kept_ds, held_ds


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
            "validation. Use benchmark.val_fraction=0.0 and benchmark.val_sessions=[1] "
            "(whole-session hold-out) for a validation stream."
        )
    val_sessions = parse_val_sessions(cfg.get("val_sessions", None))

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
    heldout = None
    if val_sessions is not None:
        train_full, heldout = holdout_sessions(
            train_full, _read_train_paths(root, variant, run), val_sessions, max_exp
        )
    train_classes = [dataset_classes(d) for d in train_full]
    exp_map, groups, indices = partition_test_by_first_appearance(
        train_classes, test_full.targets.val_to_idx, max_exp
    )
    test_list = [test_full.subset(idx) for idx in indices]

    valid_list = None
    valid_map: list[int] = []
    if heldout is not None:
        # Validation groups: held-out frames of ALL batches, grouped by the (reduced) train
        # experience where their class first appears -- the same grouping as the test stream.
        from avalanche.benchmarks.utils.utils import concat_datasets

        valid_full = concat_datasets([d for d in heldout if len(d) > 0])
        valid_map, _, v_indices = partition_test_by_first_appearance(
            train_classes, valid_full.targets.val_to_idx, max_exp
        )
        valid_list = [valid_full.subset(idx) for idx in v_indices]

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
    if heldout is not None:
        meta["val_sessions"] = val_sessions
        meta["valid_exp_to_train_exp"] = valid_map
        meta["heldout_frames_per_batch"] = [len(d) for d in heldout]
    # Task order = the run + its sequence of training batches (fingerprinted by their class sets).
    record_task_order(meta, [run, train_classes])
    # Test already truncated by first-appearance filtering; do not truncate it again.
    benchmark = finalize(
        train_full, test_list, cfg, seed, meta, truncate_test=False, valid_list=valid_list
    )
    return BenchmarkBundle(
        benchmark=benchmark,
        scenario="class_il",
        num_classes=50,
        class_order=seen,
        meta=meta,
    )


def main(argv=None) -> int:
    """``python -m clns.data.core50 --check-sessions --root <data_dir>/core50 --run 0 --n-used 3
    --val-sessions 1``: report, per used batch, the frames per session and whether holding the
    sessions out would empty a batch (exit 1 if any used batch is emptied or nothing is held out).
    """
    import argparse

    ap = argparse.ArgumentParser(description="Check CORe50 session hold-out")
    ap.add_argument("--check-sessions", action="store_true", required=True)
    ap.add_argument("--root", required=True, help="CORe50 root (contains NIC_v2_79/, ...)")
    ap.add_argument("--variant", default="nicv2_79")
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--n-used", type=int, default=None, help="leading batches that are trained on")
    ap.add_argument("--val-sessions", type=int, nargs="+", default=[1])
    args = ap.parse_args(argv)

    paths = _read_train_paths(Path(args.root), args.variant, args.run)
    rows = core50_session_check(paths, args.val_sessions, args.n_used)
    for r in rows:
        flag = "  <-- EMPTIED" if r["empties_batch"] else ""
        print(f"batch {r['batch']:3d}: frames={r['n_frames']:6d} heldout={r['n_heldout']:6d} "
              f"kept={r['n_kept']:6d} sessions={r['sessions']}{flag}")
    emptied = [r["batch"] for r in rows if r["empties_batch"]]
    total_held = sum(r["n_heldout"] for r in rows)
    print(f"val_sessions={args.val_sessions}: {total_held} held-out frames in the checked "
          f"batches; batches emptied: {emptied or 'none'}")
    return 1 if emptied or total_held == 0 else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
