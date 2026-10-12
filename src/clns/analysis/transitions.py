"""Shared transition table: accuracy matrices + shift matrices -> one row per transition.

Used by :mod:`clns.analysis.correlation_study` (Spearman, descriptive) and
:mod:`clns.analysis.hypotheses` (H1/H2 regressions). Both must pair forgetting with shift the
same way, so the pairing lives here.

Accuracy matrix convention: ``a[i][j]`` = accuracy on TEST experience j after training on TRAIN
experience i (shape T_train x T_test). ``test_exp_to_train_exp[j]`` = m_j, the train experience
that introduced test experience j (identity for most benchmarks; shorter than T_train for CORe50,
RESEARCH_LOG 7.2.1).

Transition k = the step from train experience k to k+1; its shift is ``M[k, k+1]`` of the shift
matrix measured by ``clns.shift.measure_stream`` (once per seed: the task order follows the
seed, except for CLEAR-10 whose time order is seed-independent). Two tables:

* consecutive (primary): one row per (run, k); forgetting = one-step drop
  ``a[m_j][j] - a[m_j+1][j]`` averaged over the test experiences j with m_j = k, shift =
  ``M[k, k+1]``;
* all pairs (secondary, :func:`all_pairs_table`): one row per (run, old test experience j, later
  transition t >= m_j); forgetting = ``a[t][j] - a[t+1][j]``, shift = ``M[m_j, t+1]`` (distance
  old task -> new task). It contains the consecutive pairs (t = m_j) as a subset.

Both carry ``a_tt`` (mean accuracy on the test experiences introduced at train experience k+1,
measured right after training on it: plasticity) and ``t`` (= k, the position in the stream).

A run is matched to a shift result by :func:`shift_mismatch` / :func:`select_shift`: same T,
class order, split hashes and ``task_order_sha256``. The shift of a run is therefore only used if
it was measured on the run's own task order.

Transition identity (:func:`transition_id`) is the actual PAIR OF TASKS, by scenario: domain or
bucket names (``task_order``) for domain-IL, class sets (``train_classes``) for class-IL, and run
+ batch index + class set for CORe50 NIC, so runs share a transition iff they train the same task
pair (all methods and arms of a seed do; different seeds only if their orders agree).

The all-pairs table needs a shift result measured with ``measure_stream --pairs full``
(:func:`shift_is_full`); with the default ``--pairs consecutive`` only ``M[k, k+1]`` exists.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

FORGETTING_MODES = ("one_step", "cumulative")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_accuracy_matrices(runs_dir) -> list[dict]:
    """Load finished runs under ``runs_dir/<benchmark>/<model>/<method>/seed<k>/``.

    Runs without a ``DONE`` marker are skipped with a warning. Each record has keys
    benchmark, model, method, seed, path, rows (np.ndarray T_train x T_test),
    test_exp_to_train_exp (list[int]), scenario, random_init, benchmark_meta (from
    ``provenance.json``; None if absent), eval_mask_unseen (``config.yaml``; default True).
    """
    runs_dir = Path(runs_dir)
    records = []
    for p in sorted(runs_dir.glob("*/*/*/seed*/accuracy_matrix.json")):
        run = p.parent
        m = re.fullmatch(r"seed(\d+)", run.name)
        if m is None:
            continue
        if not (run / "DONE").exists():
            warnings.warn(f"skipping unfinished run (no DONE): {run}", stacklevel=2)
            continue
        with open(p, encoding="utf-8") as fh:
            blob = json.load(fh)
        rows = np.asarray(blob["rows"], dtype=float)
        mapping = blob.get("test_exp_to_train_exp")
        if mapping is None:
            mapping = list(range(rows.shape[1]))
        bmeta = None
        prov = run / "provenance.json"
        if prov.exists():
            with open(prov, encoding="utf-8") as fh:
                bmeta = json.load(fh).get("benchmark_meta")
        records.append(
            {
                "benchmark": run.parent.parent.parent.name,
                "model": run.parent.parent.name,
                "method": run.parent.name,
                "seed": int(m.group(1)),
                "path": str(p),
                "rows": rows,
                "test_exp_to_train_exp": [int(v) for v in mapping],
                "scenario": blob.get("scenario"),
                "random_init": blob.get("random_init"),
                "benchmark_meta": bmeta,
                "eval_mask_unseen": _eval_mask_unseen(run),
            }
        )
    return records


def _eval_mask_unseen(run: Path) -> bool:
    """``train.eval_mask_unseen`` of the run (``config.yaml``); True (the default) if absent."""
    cfg = run / "config.yaml"
    if not cfg.exists():
        return True
    try:
        import yaml

        with open(cfg, encoding="utf-8") as fh:
            blob = yaml.safe_load(fh) or {}
        val = (blob.get("train") or {}).get("eval_mask_unseen", True)
    except Exception:  # unreadable config: fall back to the default
        return True
    return bool(True if val is None else val)


def load_shift(shift_dir, benchmark: str, seed: int) -> dict | None:
    """Shift matrix + meta of ``<shift_dir>/<benchmark>/seed<seed>/``; None if not present.

    Layout written by ``clns.shift.measure_stream``:
    ``<shift_dir>/<benchmark>/seed<k>/{shift_matrix.npz, shift_meta.json}``.
    """
    d = Path(shift_dir) / benchmark / f"seed{seed}"
    npz, meta = d / "shift_matrix.npz", d / "shift_meta.json"
    if not npz.exists() or not meta.exists():
        return None
    with open(meta, encoding="utf-8") as fh:
        m = json.load(fh)
    return {"matrix": np.load(npz)["matrix"], "meta": m, "dir": str(d), "seed": int(seed)}


def load_shift_candidates(shift_dir, benchmark: str) -> list[dict]:
    """All shift results of ``benchmark`` (every ``seed<k>`` directory), sorted by seed."""
    root = Path(shift_dir) / benchmark
    if not root.is_dir():
        return []
    out = []
    for d in root.glob("seed*"):
        m = re.fullmatch(r"seed(\d+)", d.name)
        if m is None:
            continue
        sh = load_shift(shift_dir, benchmark, int(m.group(1)))
        if sh is not None:
            out.append(sh)
    return sorted(out, key=lambda sh: sh["seed"])


def _order(o):
    return None if o is None else [int(c) for c in o]


def shift_mismatch(shift: dict, rec: dict) -> str | None:
    """Why ``shift`` does NOT describe the run's benchmark instance; None if it does.

    Compared against the run's ``provenance.json`` ``benchmark_meta``: T (shift) == T_train
    (run); equal class order (absent on both sides, domain-IL, counts as equal); every split
    hash the shift records equals the run's value where the run records it; and, when the shift
    records ``task_order_sha256``, the run MUST record the same one (a missing value is a
    mismatch, so legacy runs without a recorded task order cannot pair with an arbitrary order).
    The benchmark seed is NOT compared: the class order is fixed by ``benchmark.class_order_seed``
    / the task order. A run without ``benchmark_meta`` cannot be verified and never matches.
    """
    meta = shift["meta"]
    t_shift = int(meta.get("T", np.asarray(shift["matrix"]).shape[0]))
    t_train = int(np.asarray(rec["rows"]).shape[0])
    if t_shift != t_train:
        return f"shift T={t_shift} != run T_train={t_train}"
    bmeta = rec.get("benchmark_meta")
    if bmeta is None:
        return "run has no provenance benchmark_meta; cannot verify class order / split hashes"
    if _order(meta.get("class_order")) != _order(bmeta.get("class_order")):
        return "class_order mismatch between run and shift matrix"
    sha = meta.get("task_order_sha256")
    if sha is not None:
        if bmeta.get("task_order_sha256") is None:
            return "run records no task_order_sha256 (pre-v2 run); cannot verify the task order"
        if str(bmeta["task_order_sha256"]) != str(sha):
            return "task_order_sha256 mismatch between run and shift matrix"
    for k, v in (meta.get("split_hashes") or {}).items():
        if k in bmeta and str(bmeta[k]) != str(v):
            return f"split hash {k} mismatch between run and shift matrix"
    return None


def check_shift_matches_run(shift: dict, rec: dict) -> None:
    """Raise ValueError unless the shift matrix belongs to the run's benchmark instance."""
    why = shift_mismatch(shift, rec)
    if why is not None:
        raise ValueError(f"{why} ({rec['path']} vs {shift['dir']})")


def _identical_shift(a: dict, b: dict) -> bool:
    """Same shift matrix (NaN == NaN), same task order hash and same split hashes."""
    ma, mb = np.asarray(a["matrix"], dtype=float), np.asarray(b["matrix"], dtype=float)
    if ma.shape != mb.shape or not np.array_equal(ma, mb, equal_nan=True):
        return False
    keys = ("task_order_sha256", "class_order", "split_hashes")
    return all(a["meta"].get(k) == b["meta"].get(k) for k in keys)


def select_shift(candidates: list[dict], rec: dict) -> dict:
    """The shift result to pair with run ``rec`` (matching rule: :func:`shift_mismatch`).

    Exactly one match -> it. Several matches -> the one measured with the run's own seed
    (``seed<k>`` dir == run seed); else, if all matches are identical (same matrix, task order
    hash and split hashes, e.g. CLEAR-10 measured once per seed), the lowest-seed one; else
    ValueError (conflicting). No match -> ValueError listing why each candidate was rejected.
    """
    ok, rejected = [], []
    for sh in candidates:
        why = shift_mismatch(sh, rec)
        if why is None:
            ok.append(sh)
        else:
            rejected.append(f"{sh['dir']}: {why}")
    if len(ok) == 1:
        return ok[0]
    if len(ok) > 1:
        same = [sh for sh in ok if sh["seed"] == int(rec["seed"])]
        if len(same) == 1:
            return same[0]
        if all(_identical_shift(ok[0], sh) for sh in ok[1:]):
            return ok[0]  # e.g. CLEAR-10: fixed order, one measurement per seed, all identical
        raise ValueError(
            f"conflicting shift results for {rec['path']}: {[sh['dir'] for sh in ok]} all "
            "match its class order / split hashes / T, differ from each other and none was "
            "measured with the run's seed; keep only one of them"
        )
    raise ValueError(f"no shift result matches {rec['path']}: " + "; ".join(rejected))


# --------------------------------------------------------------------------- #
# Forgetting / alignment
# --------------------------------------------------------------------------- #
def _mapping(a: np.ndarray, test_exp_to_train_exp) -> np.ndarray:
    t_train, t_test = a.shape
    mapping = np.arange(t_test) if test_exp_to_train_exp is None else np.asarray(
        test_exp_to_train_exp, dtype=int
    )
    if mapping.shape[0] != t_test:
        raise ValueError(
            f"mapping length {mapping.shape[0]} != number of test experiences {t_test}"
        )
    if mapping.size and (mapping.min() < 0 or mapping.max() >= t_train):
        raise ValueError("mapping entries must lie in [0, T_train)")
    return mapping


def forgetting(a, test_exp_to_train_exp=None) -> tuple[np.ndarray, np.ndarray]:
    """CUMULATIVE forgetting per test experience (secondary; position-confounded).

    f_j = max_{m_j <= i <= T_train-2} a[i][j] - a[T_train-1][j]; NaN where
    m_j >= T_train - 1. Returns (f, mapping).
    """
    a = np.asarray(a, dtype=float)
    t_train, t_test = a.shape
    mapping = _mapping(a, test_exp_to_train_exp)
    f = np.full(t_test, np.nan)
    for j in range(t_test):
        m_j = int(mapping[j])
        if m_j < t_train - 1:
            f[j] = a[m_j : t_train - 1, j].max() - a[t_train - 1, j]
    return f, mapping


def one_step_forgetting(a, test_exp_to_train_exp=None) -> tuple[np.ndarray, np.ndarray]:
    """ONE-STEP forgetting per test experience (primary pairing).

    d_j = a[m_j][j] - a[m_j+1][j], the drop right after the next transition
    m_j -> m_j+1; NaN where m_j >= T_train - 1. Returns (d, mapping).
    """
    a = np.asarray(a, dtype=float)
    t_train, t_test = a.shape
    mapping = _mapping(a, test_exp_to_train_exp)
    d = np.full(t_test, np.nan)
    for j in range(t_test):
        m_j = int(mapping[j])
        if m_j < t_train - 1:
            d[j] = a[m_j, j] - a[m_j + 1, j]
    return d, mapping


def shift_transitions(shift_matrix) -> np.ndarray:
    """First off-diagonal of the shift matrix: M[k, k+1] for k = 0..T-2."""
    return np.diag(np.asarray(shift_matrix, dtype=float), k=1)


def align_forgetting_with_transitions(f, mapping, transitions) -> tuple[np.ndarray, np.ndarray]:
    """Pair forgetting of test exp j with transition index m_j (into train exp m_j+1).

    Returns (shift_vec, forget_vec) for all j with finite f_j and m_j < len(transitions).
    """
    f = np.asarray(f, dtype=float)
    mapping = np.asarray(mapping, dtype=int)
    transitions = np.asarray(transitions, dtype=float)
    keep = np.isfinite(f) & (mapping < len(transitions))
    return transitions[mapping[keep]], f[keep]


def _new_task_accuracy(a: np.ndarray, mapping: np.ndarray, k: int) -> float:
    """Plasticity a_tt of transition k: mean accuracy on the test experiences introduced at
    train experience k+1, right after training on it (NaN if there are none, e.g. CORe50
    batches without a first-appearance class group)."""
    cols = np.flatnonzero(mapping == k + 1)
    if cols.size == 0:
        return float("nan")
    return float(np.nanmean(a[k + 1, cols])) if np.isfinite(a[k + 1, cols]).any() else float("nan")


COLUMNS = ["k", "forget", "shift", "position", "a_tt", "t"]


def run_transition_table(rec: dict, transitions, mode: str = "one_step") -> pd.DataFrame:
    """Per-run table: one row per transition k with forgetting (mean over test exps j
    with m_j = k), the shift M[k, k+1], the position covariate T_train-1-k, the plasticity
    ``a_tt`` of the new task and ``t`` (= k).

    Rows with non-finite forgetting or shift are kept (NaN) so callers can count drops.
    """
    if mode not in FORGETTING_MODES:
        raise ValueError(f"mode must be one of {FORGETTING_MODES}, got {mode!r}")
    fn = one_step_forgetting if mode == "one_step" else forgetting
    a = np.asarray(rec["rows"], dtype=float)
    f, mp = fn(a, rec["test_exp_to_train_exp"])
    transitions = np.asarray(transitions, dtype=float)
    t_train = a.shape[0]
    rows = []
    for k in sorted(set(int(m) for m in mp if m < t_train - 1)):
        vals = f[mp == k]
        vals = vals[np.isfinite(vals)]
        rows.append(
            {
                "k": k,
                "forget": float(vals.mean()) if vals.size else float("nan"),
                "shift": float(transitions[k]) if k < len(transitions) else float("nan"),
                "position": t_train - 1 - k,
                "a_tt": _new_task_accuracy(a, mp, k),
                "t": k,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def all_pairs_table(rec: dict, shift_matrix) -> pd.DataFrame:
    """All (old test experience j, later transition t) pairs of a run (one-step drops).

    For j introduced at m_j and every t with m_j <= t <= T_train-2: ``forget`` = a[t][j] -
    a[t+1][j] (the drop caused by the step t -> t+1), ``shift`` = M[m_j, t+1] (distance between
    the task of j and the new task), ``k`` = ``t`` = t, ``a_tt`` = plasticity of the new task,
    ``position`` = T_train-1-t, plus ``j``. Includes the consecutive pairs (t = m_j).
    """
    a = np.asarray(rec["rows"], dtype=float)
    m = np.asarray(shift_matrix, dtype=float)
    mp = _mapping(a, rec["test_exp_to_train_exp"])
    t_train = a.shape[0]
    rows = []
    for j, m_j in enumerate(mp):
        for t in range(int(m_j), t_train - 1):
            rows.append(
                {
                    "j": j,
                    "k": t,
                    "forget": float(a[t, j] - a[t + 1, j]),
                    "shift": float(m[m_j, t + 1]) if max(m_j, t + 1) < m.shape[0] else np.nan,
                    "position": t_train - 1 - t,
                    "a_tt": _new_task_accuracy(a, mp, t),
                    "t": t,
                }
            )
    return pd.DataFrame(rows, columns=["j", *COLUMNS])


# --------------------------------------------------------------------------- #
# Transition identity
# --------------------------------------------------------------------------- #
def scenario_kind(rec: dict) -> str:
    """``"core50"`` (CORe50 NIC), ``"domain_il"`` (DomainNet, CLEAR-10) or ``"class_il"``.

    From the run's ``benchmark_meta`` (``protocol == "nic"`` / a ``run`` with a ``[run, batches]``
    task order) and the ``scenario`` recorded in ``accuracy_matrix.json``; a run that records no
    scenario counts as domain-IL when all its experiences share one class set, else class-IL.
    """
    bmeta = rec.get("benchmark_meta") or {}
    bench = str(rec.get("benchmark") or "")
    if bmeta.get("protocol") == "nic" or bench.startswith("core50"):
        return "core50"
    scen = rec.get("scenario") or bmeta.get("scenario")
    if scen in ("domain_il", "class_il"):
        return str(scen)
    tc = bmeta.get("train_classes")
    if tc and all(set(c) == set(tc[0]) for c in tc):
        return "domain_il"
    if not tc and bmeta.get("task_order") and bmeta.get("class_order") is None:
        return "domain_il"
    return "class_il"


def _classes_label(cs) -> str:
    return ",".join(str(int(c)) for c in sorted(cs))


def _task_labels(rec: dict) -> list[str] | None:
    """One label per training experience of the run's task sequence; None if not recorded.

    * domain-IL (DomainNet, CLEAR-10): the entry of ``task_order`` (domain name / bucket), even
      when a domain lacks a class (class sets would then wrongly separate or merge domains);
    * CORe50 NIC: ``run<r>:b<i>[<class set>]`` from ``task_order = [run, batch class sets]``
      (batch index i = position in the run's batch sequence), so batches of different sessions
      with the same class set never merge;
    * class-IL: the class set of the experience (``train_classes``).
    """
    bmeta = rec.get("benchmark_meta") or {}
    kind = scenario_kind(rec)
    order = bmeta.get("task_order")
    tc = bmeta.get("train_classes")
    if kind == "domain_il":
        return [str(o) for o in order] if order else None
    if kind == "core50":
        run = bmeta.get("run")
        batches = None
        if (isinstance(order, (list, tuple)) and len(order) == 2
                and isinstance(order[1], (list, tuple))):
            run = order[0] if run is None else run
            batches = order[1]
        if batches is None:
            batches = tc
        if not batches:
            return None
        return [f"run{run}:b{i}[{_classes_label(cs)}]" for i, cs in enumerate(batches)]
    if tc:
        return [_classes_label(cs) for cs in tc]
    return None


def transition_id(rec: dict, k: int) -> str:
    """Identity of transition k of a run: the pair of tasks ``<task k> -> <task k+1>``.

    Built from the run's recorded task sequence by scenario (:func:`_task_labels`): domain-IL
    from ``task_order`` entries, class-IL from class sets, CORe50 from run + batch index + class
    set. Two runs share transition k iff they train the same pair of tasks in that step (all
    methods and arms of one seed; different seeds only when their orders agree, as for
    CLEAR-10). Runs without a recorded task sequence (legacy) fall back to the class order (or
    ``"fixed"`` for domain-IL) plus the index k.
    """
    bmeta = rec.get("benchmark_meta") or {}
    labels = _task_labels(rec)
    if labels is not None and len(labels) >= k + 2:
        return f"{labels[k]}->{labels[k + 1]}"
    order = bmeta.get("class_order")
    tag = "fixed" if order is None else ",".join(str(c) for c in order)
    return f"{tag}|{k}"


def shift_is_full(shift: dict) -> bool:
    """True if the shift result holds the full upper triangle (``measure_stream --pairs full``).

    Uses ``shift_meta.json`` ``pairs`` when recorded; otherwise (legacy) checks that every
    upper-triangle entry beyond the first off-diagonal is finite.
    """
    pairs = (shift.get("meta") or {}).get("pairs")
    if pairs is not None:
        return str(pairs) == "full"
    m = np.asarray(shift["matrix"], dtype=float)
    iu = np.triu_indices(m.shape[0], k=2)
    return bool(np.isfinite(m[iu]).all())


def aggregate_transitions(table: pd.DataFrame) -> pd.DataFrame:
    """Collapse a run-level table (columns unit, forget, shift, position, run) to one
    row per transition unit: mean forgetting and mean shift over runs."""
    g = table.groupby("unit", sort=True)
    return pd.DataFrame(
        {
            "shift": g["shift"].mean(),
            "forget": g["forget"].mean(),
            "position": g["position"].first(),
            "n_runs": g["run"].nunique(),
        }
    ).reset_index()
