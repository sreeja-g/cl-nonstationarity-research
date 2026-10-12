"""GPU_HOURS_CAP accounting of scripts/run_grid.sh (scripts/gpu_hours.py), no GPU needed."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gpu_hours.py"
GRID = REPO / "scripts" / "run_grid.sh"


def _mod():
    spec = importlib.util.spec_from_file_location("gpu_hours", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _progress(run_dir: Path, seconds: float):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "progress.json").write_text(json.dumps({"wall_clock_s_committed": seconds}))


def test_counts_every_attempt_and_tune_runs(tmp_path):
    out = tmp_path / "out"
    a = out / "runs" / "clear10" / "r18_scratch" / "ewc" / "seed0"
    b = out / "runs" / "clear10" / "r18_scratch" / "er" / "seed0"
    tune = out / "tune" / "runs" / "clear10" / "r18_scratch" / "ewc" / "hp_method.ewc_lambda=1"
    tune = tune / "seed0"
    _progress(a, 1800.0)    # committed 0.5 h, but the driver saw 1 h + a failed 0.5 h attempt
    _progress(b, 3600.0)    # committed 1 h, ledger knows only a 0.25 h check -> max = 1 h
    _progress(tune, 900.0)  # tuning sweep outside the driver: 0.25 h
    (out / "logs").mkdir(parents=True)
    failed_only = out / "runs" / "clear10" / "r18_scratch" / "lwf" / "seed0"  # crashed, no progress
    (out / "logs" / "gpu_ledger.tsv").write_text(
        "timestamp\trun_dir\tseconds\texit_code\n"
        f"t\t{a}\t1800\t1\n"
        f"t\t{a}\t3600\t0\n"
        f"t\t{b}\t900\t0\n"
        f"t\t{failed_only}\t720\t3\n"
    )
    assert _mod().hours_used(out) == pytest.approx(1.5 + 1.0 + 0.25 + 0.2)


def test_accounting_failure_is_an_error_not_zero(tmp_path, capsys):
    out = tmp_path / "out"
    run = out / "runs" / "b" / "m" / "x" / "seed0"
    run.mkdir(parents=True)
    (run / "progress.json").write_text("{not json")
    assert _mod().main([str(out)]) == 1
    assert "accounting failed" in capsys.readouterr().err
    (run / "progress.json").write_text(json.dumps({"wall_clock_s_committed": 7200}))
    (out / "logs").mkdir()
    (out / "logs" / "gpu_ledger.tsv").write_text("t\tonly-two-columns\n")
    assert _mod().main([str(out)]) == 1
    (out / "logs" / "gpu_ledger.tsv").unlink()
    assert _mod().main([str(out)]) == 0
    assert capsys.readouterr().out.strip() == "2.0"


def _grid_functions() -> str:
    """The driver's log / gpu_hours_used / check_budget functions, extracted from run_grid.sh."""
    text = GRID.read_text(encoding="utf-8")
    out = []
    for name in ("log", "gpu_hours_used", "check_budget"):
        start = text.index(f"\n{name}() ")
        if name == "log":
            out.append(text[start:text.index("\n", start + 1)])
            continue
        end = text.index("\n}\n", start) + 3
        out.append(text[start:end])
    return "\n".join(out)


@pytest.mark.parametrize("mode", ["fail", "garbage", "cap", "ok"])
def test_run_grid_check_budget_aborts_on_accounting_failure(tmp_path, mode):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    py = Path(sys.executable).as_posix()
    stub = {
        "fail": 'python() { echo "boom" >&2; return 1; }',
        "garbage": 'python() { echo "n/a"; }',
        "cap": f'python() {{ "{py}" "$@"; }}',
        "ok": f'python() {{ "{py}" "$@"; }}',
    }[mode]
    out = tmp_path / "out"
    _progress(out / "runs" / "b" / "m" / "x" / "seed0", 7200.0)  # 2 h used
    cap = "1" if mode == "cap" else "40"
    script = "\n".join([
        "set -euo pipefail",
        _grid_functions(),
        stub,
        f'REPO="{REPO.as_posix()}"; OUT="{out.as_posix()}"; GPU_HOURS_CAP={cap}; ABORT=""',
        'rc=0; check_budget "b/m/x/seed1" || rc=$?',
        'echo "rc=$rc USED=${USED:-} ABORT=$ABORT"',
    ])
    r = subprocess.run([bash, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    line = r.stdout.strip().splitlines()[-1]
    if mode in ("fail", "garbage"):
        assert line.startswith("rc=2") and "accounting failed" in line
    elif mode == "cap":
        assert line.startswith("rc=1") and "GPU_HOURS_CAP reached" in line and "USED=2.0" in line
    else:
        assert line.startswith("rc=0 USED=2.0") and line.endswith("ABORT=")


def test_run_grid_aborts_with_exit_7_on_accounting_failure():
    """Static check of the wiring: an accounting failure stops the loop and the driver exits 7."""
    text = GRID.read_text(encoding="utf-8")
    assert 'if ! check_budget "$b/$a/$m/seed$s"; then' in text
    assert 'if [ -n "$ABORT" ]; then exit 7; fi' in text
    assert "|| echo 0" not in text  # the old silent fallback
    loop = text[text.index("# Seed-outermost"):]
    assert loop.index('for s in "${SEED_A[@]}"') < loop.index('for b in "${BENCH_A[@]}"')


# --------------------------------------------------------------------------- #
# Review-05: start/end ledger rows, unmatched start rows, TERM trap
# --------------------------------------------------------------------------- #
def test_matched_start_end_rows_count_once_and_legacy_rows_still_work(tmp_path):
    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    a, b = out / "runs" / "x" / "seed0", out / "runs" / "y" / "seed0"
    (out / "logs" / "gpu_ledger.tsv").write_text(
        "timestamp\trun_dir\tseconds\texit_code\n"
        f"t\t{a}\t600\t1\n"                                   # legacy 4-column end row
        f"2026-10-12T10:00:00+00:00\t{a}\t-\t-\tstart\n"
        f"2026-10-12T11:00:00+00:00\t{a}\t3600\t0\tend\n"
        f"2026-10-12T11:00:01+00:00\t{b}\t-\t-\tstart\n"
        f"2026-10-12T11:30:01+00:00\t{b}\t1800\t143\tend\n"
    )
    assert _mod().hours_used(out) == pytest.approx((600 + 3600 + 1800) / 3600)


def test_unmatched_start_row_charged_up_to_next_row_or_now(tmp_path):
    from datetime import datetime, timezone

    out = tmp_path / "out"
    (out / "logs").mkdir(parents=True)
    a, b = out / "runs" / "x" / "seed0", out / "runs" / "y" / "seed0"
    lp = out / "logs" / "gpu_ledger.tsv"
    lp.write_text(
        "timestamp\trun_dir\tseconds\texit_code\tstatus\n"
        f"2026-10-12T10:00:00+00:00\t{a}\t-\t-\tstart\n"     # driver SIGKILLed: never closed
        f"2026-10-12T12:00:00+00:00\t{b}\t-\t-\tstart\n"     # next row 2 h later
        f"2026-10-12T12:30:00+00:00\t{b}\t1800\t0\tend\n"
    )
    m = _mod()
    now = datetime(2026, 10, 13, tzinfo=timezone.utc)
    assert m.hours_used(out, now=now) == pytest.approx(2.0 + 0.5)
    # the open start row is the LAST row -> charged up to now
    lp.write_text(lp.read_text() + f"2026-10-12T13:00:00+00:00\t{a}\t-\t-\tstart\n")
    now = datetime(2026, 10, 12, 14, 15, tzinfo=timezone.utc)
    assert m.hours_used(out, now=now) == pytest.approx(2.0 + 0.5 + 1.25)
    # a hand-written end row closes it (correction for pod downtime)
    lp.write_text(lp.read_text() + f"2026-10-12T13:10:00+00:00\t{a}\t600\t137\tend\n")
    assert m.hours_used(out, now=now) == pytest.approx(2.0 + 0.5 + 600 / 3600)
    # a progress.json larger than the ledger still wins for that run
    _progress(a, 5 * 3600.0)
    assert m.hours_used(out, now=now) == pytest.approx(5.0 + 0.5)
    lp.write_text(lp.read_text() + f"garbage-time\t{b}\t-\t-\tstart\n")
    assert m.main([str(out)]) == 1  # unparseable timestamp of an open start: accounting fails


def test_run_grid_trap_writes_end_row_when_driver_is_terminated(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    text = GRID.read_text(encoding="utf-8")
    start = text.index("\nledger_start() ")
    end = text.index("trap 'on_signal INT 130' INT", start)
    block = text[start:end] + "trap 'on_signal INT 130' INT\n"
    ledger = tmp_path / "gpu_ledger.tsv"
    script = "\n".join([
        "set -euo pipefail",
        _grid_functions().split("\ngpu_hours_used()")[0],  # log()
        f'LEDGER="{ledger.as_posix()}"',
        block,
        "set +e",
        'rd=/w/runs/b/m/x/seed0; t0=$SECONDS; ledger_start "$rd"',
        "sleep 30 & CHILD_PID=$!; CHILD_RD=$rd; CHILD_T0=$t0",
        'wait "$CHILD_PID"; rc=$?; CHILD_PID=""',
        'ledger_end "$rd" "$(( SECONDS - t0 ))" "$rc"',
        "echo NOT-REACHED",
    ])
    drv = tmp_path / "drv.sh"
    drv.write_bytes(script.encode("utf-8"))
    # SIGTERM sent by bash (Popen.terminate() is a hard TerminateProcess on Windows)
    outer = (f'bash "{drv.as_posix()}" & d=$!; '
             f'for i in $(seq 1 100); do grep -q start "{ledger.as_posix()}" 2>/dev/null && break; '
             'sleep 0.1; done; sleep 0.5; t=$SECONDS; kill -TERM $d; wait $d; '
             'echo "rc=$? waited=$(( SECONDS - t ))"')
    r = subprocess.run([bash, "-c", outer], capture_output=True, text=True, timeout=120)
    rows = [x.split("\t") for x in ledger.read_text().splitlines()]
    assert [x[4] for x in rows] == ["start", "end"], (r.stdout, r.stderr)
    assert rows[1][3] == "143" and "NOT-REACHED" not in r.stdout
    assert "rc=143" in r.stdout and "ledger end row written" in r.stdout
    waited = int(r.stdout.split("waited=")[1].split()[0])
    assert waited < 20  # the trap ran at once, it did not wait for sleep 30 to finish


def test_run_grid_writes_start_row_before_launch():
    text = GRID.read_text(encoding="utf-8")
    loop = text[text.index("# Seed-outermost"):]
    assert loop.index('ledger_start "$rd"') < loop.index("python -m clns.train") < loop.index(
        'ledger_end "$rd"')
    assert "trap 'on_signal TERM 143' TERM" in text
    for doc in ("measure_stream", "POD hours"):
        assert doc in text and doc in SCRIPT.read_text(encoding="utf-8")
