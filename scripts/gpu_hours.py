"""GPU-hour accounting for scripts/run_grid.sh (GPU_HOURS_CAP).

Usage: python scripts/gpu_hours.py <OUT>   -> prints the hours used (float) on stdout.

Sources, per run directory (counted once, the larger of the two):
  * <OUT>/logs/gpu_ledger.tsv, written by run_grid.sh around EVERY train.py call (columns:
    timestamp, run_dir, seconds, exit_code, status):
      - a ``start`` row (seconds / exit_code ``-``) right BEFORE train.py is launched;
      - an ``end`` row with the attempt's wall clock right after it returns, including failed,
        OOM, timed-out, killed and refused attempts; the driver's TERM/INT trap writes the end
        row of the running attempt before the driver exits. Rows without a status column
        (ledgers from before the start rows existed) are end rows.
    An end row closes the latest open start row of its run_dir. A start row that is never
    closed (driver SIGKILLed, pod crash) is charged from its timestamp up to the timestamp of
    the NEXT ledger row (any run), or up to NOW if it is the last row. This is deliberately
    conservative: if the pod was down after the crash, that downtime counts too. To correct it,
    append an end row for that run_dir by hand
    (``<iso time>\\t<run_dir>\\t<seconds>\\t<rc>\\tend``).
  * progress.json ``wall_clock_s_committed`` of every run under <OUT>/runs, <OUT>/tune (tuning
    sweeps) and <OUT>/smoke: covers runs started outside the grid driver. It only counts time up
    to each attempt's last committed checkpoint, so for those runs failed attempts are
    undercounted.

What the cap does NOT include: GPU time of clns.shift.measure_stream (shift / feature
extraction), idle pod time (setup, downloads, analysis, a pod left running after the grid),
storage. The budget (~$50) is POD hours ($0.59/h for the whole pod, GPU used or not), so keep
GPU_HOURS_CAP well below budget / hourly rate.

Any unreadable or malformed file is an ERROR (exit 1, message on stderr): the driver then aborts
(exit 7) instead of treating the budget as unused.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _key(path) -> str:
    return os.path.normcase(os.path.realpath(str(path)))


def _ts(text: str, where: str) -> datetime:
    try:
        t = datetime.fromisoformat(text.strip())
    except ValueError as exc:
        raise ValueError(f"{where}: bad timestamp {text!r}") from exc
    return t if t.tzinfo is not None else t.astimezone()  # naive -> local time


def ledger_seconds(lp: Path, now: datetime | None = None) -> dict[str, float]:
    """Seconds per run dir from the ledger (see module docstring for start/end matching)."""
    rows = []
    for n, line in enumerate(lp.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("timestamp\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(f"{lp}:{n}: expected timestamp<TAB>run_dir<TAB>seconds[...]")
        status = parts[4].strip() if len(parts) >= 5 else "end"
        if status not in ("start", "end"):
            raise ValueError(f"{lp}:{n}: unknown status {status!r} (start / end)")
        rows.append((n, parts[0], _key(parts[1]), parts[2], status))
    out: dict[str, float] = {}
    open_start: dict[str, int] = {}  # run key -> index into rows of its open start row
    unmatched: list[int] = []
    for i, (n, _, k, sec_s, status) in enumerate(rows):
        if status == "start":
            if k in open_start:
                unmatched.append(open_start[k])  # restarted without an end row
            open_start[k] = i
            continue
        sec = float(sec_s)
        if sec < 0 or sec != sec:
            raise ValueError(f"{lp}:{n}: invalid seconds {sec_s!r}")
        out[k] = out.get(k, 0.0) + sec
        open_start.pop(k, None)
    unmatched += open_start.values()
    now = now or datetime.now(timezone.utc)
    for i in unmatched:
        n, ts, k, _, _ = rows[i]
        start = _ts(ts, f"{lp}:{n}")
        if i + 1 < len(rows):
            stop = _ts(rows[i + 1][1], f"{lp}:{rows[i + 1][0]}")
        else:
            stop = now
        out[k] = out.get(k, 0.0) + max(0.0, (stop - start).total_seconds())
    return out


def hours_used(out, now: datetime | None = None) -> float:
    out = Path(out)
    per_run: dict[str, float] = {}
    for sub in ("runs", "tune", "smoke"):
        root = out / sub
        if not root.is_dir():
            continue
        for p in root.rglob("progress.json"):
            blob = json.loads(p.read_text(encoding="utf-8"))
            sec = float(blob.get("wall_clock_s_committed") or 0.0)
            if sec < 0 or sec != sec:
                raise ValueError(f"{p}: invalid wall_clock_s_committed {sec!r}")
            per_run[_key(p.parent)] = sec
    lp = out / "logs" / "gpu_ledger.tsv"
    ledger = ledger_seconds(lp, now) if lp.exists() else {}
    total = sum(max(per_run.get(k, 0.0), ledger.get(k, 0.0)) for k in set(per_run) | set(ledger))
    return total / 3600.0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python scripts/gpu_hours.py <OUT>", file=sys.stderr)
        return 2
    try:
        print(round(hours_used(argv[0]), 3))
    except Exception as exc:  # any failure must stop the grid, never count as 0 h
        print(f"gpu_hours.py: accounting failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
