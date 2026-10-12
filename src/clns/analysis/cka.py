"""Representation drift: linear CKA between consecutive probe rows of every finished run.

train.py writes the penultimate features of a fixed probe set after every training experience
(``<run_dir>/probe/row_XXXX.npz`` + ``meta.npz``, :mod:`clns.evaluation.probe`). For each run
and each transition k (row k = after training experience k, row k+1 = after k+1) this computes
the debiased linear CKA (:func:`clns.shift.metrics.linear_cka`, float32 features) between the
two rows:

* ``cka_all``: over all probe images;
* ``cka_old``: over the probe images of OLD TASKS: images whose test experience ``test_exp``
  (probe ``meta.npz``) maps, through the run's ``test_exp_to_train_exp``
  (``accuracy_matrix.json``), to a training experience ``<= k``. Domain-IL (identity mapping):
  old = domains 0..k (test_exp <= k). Class-IL (identity mapping): old = the classes of train
  experiences 0..k. CORe50 NIC: old = the first-appearance test groups introduced in train
  experiences 0..k. Without ``accuracy_matrix.json`` / its mapping the identity is assumed
  (``old_def`` says which).

A CKA value needs at least ``MIN_CKA_N`` = 25 images; below that it is NaN and the column
``reason`` says why. Notes for interpretation: debiased (unbiased-HSIC) CKA can be NEGATIVE, and
for independent representations its sampling noise has sd of roughly 1/sqrt(n), so values of
order 1/sqrt(n_old) are not distinguishable from 0.

Usage::

    python -m clns.analysis.cka --runs-dir <out>/runs --out <dir>      # -> <dir>/cka.csv

Runs without ``DONE`` or without a probe directory are skipped with a warning.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MIN_CKA_N = 25
COLUMNS = ["benchmark", "model", "method", "seed", "k", "cka_all", "cka_old", "n_probe",
           "n_old", "old_def", "reason"]


def _test_exp_to_train_exp(run: Path):
    """The run's test_exp -> train_exp mapping (``accuracy_matrix.json``); None if unknown."""
    acc = run / "accuracy_matrix.json"
    if not acc.exists():
        return None
    with open(acc, encoding="utf-8") as fh:
        mp = json.load(fh).get("test_exp_to_train_exp")
    return None if mp is None else np.asarray(mp, dtype=np.int64)


def _cka(x: np.ndarray, y: np.ndarray) -> float:
    from clns.shift.metrics import linear_cka

    if x.shape[0] < MIN_CKA_N:
        return float("nan")
    return float(linear_cka(x, y))


def run_cka(run_dir) -> pd.DataFrame:
    """CKA rows (one per transition k) of one run dir ``.../<bench>/<model>/<method>/seed<k>``."""
    from clns.evaluation.probe import PROBE_DIR, load_probe

    run = Path(run_dir)
    m = re.fullmatch(r"seed(\d+)", run.name)
    probe = load_probe(run / PROBE_DIR)
    feats = probe["features"].astype(np.float32)
    labels = np.asarray(probe["labels"], dtype=np.int64)
    test_exp = np.asarray(probe["test_exp"], dtype=np.int64)
    mapping = _test_exp_to_train_exp(run)
    old_def = "test_exp_to_train_exp<=k"
    if mapping is None:
        mapping = np.arange(int(test_exp.max()) + 1 if test_exp.size else 0)
        old_def = "test_exp<=k (identity: no test_exp_to_train_exp)"
    if test_exp.size and test_exp.max() >= len(mapping):
        raise ValueError(f"probe test_exp {int(test_exp.max())} outside the run's mapping "
                         f"(length {len(mapping)})")
    train_exp_of_image = mapping[test_exp] if test_exp.size else test_exp
    rows = []
    for k in range(feats.shape[0] - 1):
        a, b = feats[k], feats[k + 1]
        old = train_exp_of_image <= k
        n_old = int(old.sum())
        reasons = []
        if len(labels) < MIN_CKA_N:
            reasons.append(f"cka_all: n_probe={len(labels)} < {MIN_CKA_N}")
        if n_old < MIN_CKA_N:
            reasons.append(f"cka_old: n_old={n_old} < {MIN_CKA_N}")
        rows.append({
            "benchmark": run.parent.parent.parent.name, "model": run.parent.parent.name,
            "method": run.parent.name, "seed": int(m.group(1)) if m else -1, "k": k,
            "cka_all": _cka(a, b), "cka_old": _cka(a[old], b[old]),
            "n_probe": int(len(labels)), "n_old": n_old, "old_def": old_def,
            "reason": "; ".join(reasons),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def collect(runs_dir) -> tuple[pd.DataFrame, dict]:
    """CKA table over every finished run under ``runs_dir`` (+ counts of skipped runs)."""
    counts = {"runs_found": 0, "runs_used": 0, "skipped_unfinished": 0, "skipped_no_probe": 0,
              "skipped_bad_probe": 0}
    frames = []
    for run in sorted(Path(runs_dir).glob("*/*/*/seed*")):
        if not run.is_dir() or re.fullmatch(r"seed\d+", run.name) is None:
            continue
        counts["runs_found"] += 1
        if not (run / "DONE").exists():
            log.warning("WARNING: skipping unfinished run (no DONE): %s", run)
            counts["skipped_unfinished"] += 1
            continue
        if not (run / "probe" / "meta.npz").exists():
            log.warning("WARNING: skipping run without probe (train.probe.enabled=false?): %s",
                        run)
            counts["skipped_no_probe"] += 1
            continue
        try:
            frames.append(run_cka(run))
        except ValueError as exc:
            log.warning("WARNING: skipping run with inconsistent probe rows: %s (%s)", run, exc)
            counts["skipped_bad_probe"] += 1
            continue
        counts["runs_used"] += 1
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return table, counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Linear CKA between consecutive probe rows (representation drift)."
    )
    ap.add_argument("--runs-dir", required=True, help="<output_dir>/runs")
    ap.add_argument("--out", required=True, help="output directory (cka.csv)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    table, counts = collect(args.runs_dir)
    print(f"runs: {counts}")
    if table.empty:
        print(f"error: no finished run with probe features under {args.runs_dir}",
              file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "cka.csv", index=False)
    print(f"wrote {out / 'cka.csv'} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
