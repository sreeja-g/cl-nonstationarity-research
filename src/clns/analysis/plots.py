"""Plots for the analysis: accuracy-matrix heatmap and forgetting-vs-shift scatter.

Ported from the v1 ``evaluation/plots.py`` (only the accuracy-matrix heatmap was needed); the
interactive ``plt.show()`` is replaced by saving to a file so the plots run headless on the pod.
matplotlib is an OPTIONAL dependency (``pip install -e ".[plots]"``) and is imported lazily
with the Agg backend; without it the functions raise an ImportError that says how to install it.

CLI::

    python -m clns.analysis.plots heatmap --run-dir <run> --out acc.png
    python -m clns.analysis.plots scatter --runs-dir <out>/runs --shift-dir <shift> \\
        --benchmark clear10 --out forget_vs_shift.png [--pairs all]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FAMILY_COLORS = {"regularization": "tab:red", "replay": "tab:blue", "naive": "tab:gray"}


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "matplotlib is required for clns.analysis.plots; install the optional extra: "
            'pip install -e ".[plots]"'
        ) from exc
    return plt


def plot_accuracy_heatmap(rows, out_path, title: str = "Accuracy matrix") -> Path:
    """Heatmap of an accuracy matrix (rows = training experience, columns = test experience)."""
    plt = _pyplot()
    mat = np.asarray(rows, dtype=float) * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax).set_label("Accuracy (%)")
    ax.set_xlabel("Evaluated experience")
    ax.set_ylabel("Training experience")
    ax.set_title(title)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_xticklabels([f"{i + 1}" for i in range(mat.shape[1])])
    ax.set_yticklabels([f"{i + 1}" for i in range(mat.shape[0])])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]) and mat.shape[1] <= 12:
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] < 50 else "black")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_forgetting_vs_shift(table, out_path, title: str = "Forgetting vs shift") -> Path:
    """Scatter of one-step forgetting against shift, coloured by method family, with a
    least-squares line per family. ``table``: the frame from
    :func:`clns.analysis.hypotheses.load_table` (one benchmark). Descriptive only."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for fam, sub in table.groupby("family"):
        sub = sub[np.isfinite(sub["shift"]) & np.isfinite(sub["forget"])]
        if sub.empty:
            continue
        color = FAMILY_COLORS.get(fam, "tab:green")
        ax.scatter(sub["shift"], sub["forget"], s=12, alpha=0.4, color=color, label=fam)
        if len(sub) > 2 and np.ptp(sub["shift"]) > 0:
            slope, icpt = np.polyfit(sub["shift"], sub["forget"], 1)
            xs = np.array([sub["shift"].min(), sub["shift"].max()])
            ax.plot(xs, icpt + slope * xs, color=color)
    ax.set_xlabel("Shift (marginal MMD$^2$)")
    ax.set_ylabel("One-step forgetting")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analysis plots (needs matplotlib).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("heatmap", help="accuracy matrix of one finished run")
    h.add_argument("--run-dir", required=True)
    h.add_argument("--out", required=True)
    s = sub.add_parser("scatter", help="forgetting vs shift for one benchmark")
    s.add_argument("--runs-dir", required=True)
    s.add_argument("--shift-dir", required=True)
    s.add_argument("--benchmark", required=True)
    s.add_argument("--pairs", choices=("consecutive", "all"), default="consecutive")
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "heatmap":
        blob = json.loads((Path(args.run_dir) / "accuracy_matrix.json").read_text("utf-8"))
        plot_accuracy_heatmap(blob["rows"], args.out, title=Path(args.run_dir).as_posix())
    else:
        from clns.analysis.hypotheses import load_table

        table, _ = load_table(args.runs_dir, args.shift_dir, args.pairs)
        table = table[table["benchmark"] == args.benchmark]
        if table.empty:
            print(f"error: no paired runs for benchmark {args.benchmark}", file=sys.stderr)
            return 1
        plot_forgetting_vs_shift(table, args.out, title=f"{args.benchmark}: forgetting vs shift")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
