"""CLI: small CSV of benchmark properties + model capacity, the confounds of cross-benchmark claims.

Benchmarks differ in more than their shift (number and size of experiences, image size, classes
per experience), and arms differ in trainable capacity. This table puts those numbers next to
each other so the analysis/report can state them; nothing is re-computed from data -- every
value comes from the run directories written by ``clns.train``:

  * ``provenance.json``: ``benchmark_meta`` (``n_train_experiences``, ``train_sizes``,
    ``test_sizes``, ``train_classes``) and ``params`` (``trainable`` / ``total``);
  * ``config.yaml``: ``benchmark.name``, ``benchmark.image_size`` (after the model override),
    ``model.name``.

One row per (benchmark, model), taken from the first run in sorted path order; ``n_runs`` counts
the runs found and ``consistent`` is False if any run of the pair disagrees on a data/capacity
column (e.g. a truncated smoke run mixed in).

Usage:
    python -m clns.data.confounds --runs-root /workspace/outputs/runs --out confounds.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

COLUMNS = (
    "benchmark",
    "model",
    "n_experiences",
    "n_train_images",
    "n_test_images",
    "image_size",
    "mean_classes_per_experience",
    "trainable_params",
    "total_params",
    "n_runs",
    "consistent",
)
_COMPARED = COLUMNS[2:9]


def run_row(run_dir: str | Path) -> dict | None:
    """Confound columns of one run directory; None if provenance/config are incomplete."""
    from omegaconf import OmegaConf

    run_dir = Path(run_dir)
    prov_path, cfg_path = run_dir / "provenance.json", run_dir / "config.yaml"
    if not prov_path.is_file() or not cfg_path.is_file():
        return None
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    meta = prov.get("benchmark_meta") or {}
    params = prov.get("params") or {}
    if not meta:
        log.warning("%s: no benchmark_meta in provenance.json; skipped", run_dir)
        return None
    cfg = OmegaConf.load(cfg_path)
    train_classes = meta.get("train_classes") or []
    mean_cls = (
        sum(len(c) for c in train_classes) / len(train_classes) if train_classes else None
    )
    return {
        "benchmark": str(cfg.benchmark.name),
        "model": str(cfg.model.name),
        "n_experiences": meta.get("n_train_experiences"),
        "n_train_images": sum(meta.get("train_sizes") or []) or None,
        "n_test_images": sum(meta.get("test_sizes") or []) or None,
        "image_size": cfg.benchmark.get("image_size", None),
        "mean_classes_per_experience": mean_cls,
        "trainable_params": params.get("trainable"),
        "total_params": params.get("total"),
    }


def collect(runs_root: str | Path) -> list[dict]:
    """One row per (benchmark, model) over every ``provenance.json`` below ``runs_root``."""
    rows: dict[tuple[str, str], dict] = {}
    for prov in sorted(Path(runs_root).rglob("provenance.json")):
        r = run_row(prov.parent)
        if r is None:
            continue
        key = (r["benchmark"], r["model"])
        if key not in rows:
            rows[key] = {**r, "n_runs": 1, "consistent": True}
            continue
        agg = rows[key]
        agg["n_runs"] += 1
        if any(agg[c] != r[c] for c in _COMPARED):
            agg["consistent"] = False
            log.warning("%s: confound columns differ from the first run of %s/%s",
                        prov.parent, *key)
    return [rows[k] for k in sorted(rows)]


def write_csv(rows: list[dict], out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in COLUMNS})
    os.replace(tmp, out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs-root", required=True,
                    help="directory searched recursively for run dirs (e.g. <output_dir>/runs)")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args(argv)
    rows = collect(args.runs_root)
    if not rows:
        raise SystemExit(f"no run directories with provenance.json + config.yaml under "
                         f"{args.runs_root}")
    write_csv(rows, args.out)
    print(f"wrote {args.out} ({len(rows)} benchmark x model rows)")


if __name__ == "__main__":
    main()
