#!/usr/bin/env bash
# End-to-end smoke test of the framework.
# Usage: bash scripts/smoke_test.sh [--real] [--cpu]
#   (default) GPU allocation check, synthetic run (experiment=smoke, logger=wandb_offline, WANDB_MODE=offline)
#             into its OWN temp output dir (never the production $CLNS_OUTPUT_DIR), artefact assertions
#             (metrics.jsonl non-empty, summary.json keys, peak_gpu_mem_mb on GPU, 'W&B mode:' log line),
#             then, on GPU only, a checkpoint/resume run with the production-like DataLoader setting
#             (method=er, train.num_workers=2, CUDA): stop after 1 experience, resume to DONE, assert
#             a 2-row accuracy matrix; then pytest -q -m "not slow".
#   --real    additionally run experiment=smoke_clear10 (needs CLEAR-10: download_data.sh clear10) with
#             logger=wandb in its own W&B project/group ("...-smoke" / "smoke/..."), so production groups are
#             not polluted. Mode follows the normal precedence (WANDB_MODE, else online iff WANDB_API_KEY, else offline).
#   --cpu     skip the GPU requirement (local development only).
# Prints PASS/FAIL lines; exit 1 if anything failed.
set -uo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
_CALLER_WANDB_MODE="${WANDB_MODE-__unset__}"
if [ -f "$WORKSPACE/clns_env.sh" ]; then
  set +u
  # shellcheck disable=SC1091
  source "$WORKSPACE/clns_env.sh"
  set -u
fi
if [ "$_CALLER_WANDB_MODE" != "__unset__" ]; then export WANDB_MODE="$_CALLER_WANDB_MODE"; fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REAL=0
CPU=0
for a in "$@"; do
  case "$a" in
    --real) REAL=1 ;;
    --cpu) CPU=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

NFAIL=0
pass() { printf 'PASS: %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; NFAIL=$((NFAIL + 1)); }
check() {  # check <description> <command...>
  local d="$1"; shift
  if "$@" >/dev/null 2>&1; then pass "$d"; else fail "$d"; fi
}

cd "$REPO" || exit 1
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
# Everything this script writes lives under $TMP: smoke outputs (experiment=smoke* resolves to
# $CLNS_OUTPUT_DIR/smoke/...), W&B offline files, caches of the runs. Production dirs are not touched.
export WANDB_DIR="$TMP/wandb"; mkdir -p "$WANDB_DIR"

# --- GPU allocation check ----------------------------------------------------
ON_GPU=0
if [ "$CPU" -eq 0 ]; then
  check "nvidia-smi sees a GPU" nvidia-smi -L
  if python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    pass "torch.cuda.is_available()"
    ON_GPU=1
  else
    fail "torch.cuda.is_available() - no usable GPU (pod not allocated a GPU?). Use --cpu for a CPU-only dev check."
  fi
fi

# assert_run <run_dir> <label> <log file> <expected W&B mode regex>
assert_run() {
  local rd="$1" label="$2" lg="$3" modere="$4"
  check "$label: DONE marker" test -f "$rd/DONE"
  check "$label: accuracy_matrix.json" test -s "$rd/accuracy_matrix.json"
  check "$label: metrics.jsonl non-empty" test -s "$rd/metrics.jsonl"
  check "$label: summary.json has the expected keys" \
    python -c '
import json, sys
d = json.load(open(sys.argv[1]))
need = {"acc_final", "acc_avg_incremental", "bwt", "forgetting_max", "fwt", "peak_gpu_mem_mb",
        "train_time_s_total", "wall_clock_s_total", "acc_final_macro"}
missing = need - set(d)
sys.exit(0 if not missing else (print("missing:", sorted(missing)) or 1))' "$rd/summary.json"
  if [ "$ON_GPU" -eq 1 ]; then
    check "$label: summary.json has non-null peak_gpu_mem_mb" \
      python -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("peak_gpu_mem_mb") is not None else 1)' "$rd/summary.json"
  fi
  if grep -hE "W&B mode: ($modere)\b" "$lg" "$rd/log.txt" >/dev/null 2>&1; then
    pass "$label: 'W&B mode:' line present ($(grep -hoE 'W&B mode: [a-z]+' "$lg" "$rd/log.txt" 2>/dev/null | head -1))"
  else
    fail "$label: expected a 'W&B mode: ($modere)' line in the run log"
  fi
}

DEV_ARG=(device=cuda)
[ "$CPU" -eq 1 ] && DEV_ARG=(device=cpu)

# --- (a) synthetic smoke run ---------------------------------------------------
export CLNS_OUTPUT_DIR="$TMP/outputs"
LOG_A="$TMP/smoke.log"
if WANDB_MODE=offline python -m clns.train experiment=smoke logger=wandb_offline \
     logger.project=clns-nonstationarity-smoke logger.group=smoke/synthetic "${DEV_ARG[@]}" >"$LOG_A" 2>&1; then
  pass "synthetic smoke run exited 0"
else
  fail "synthetic smoke run (tail of log follows)"; tail -n 25 "$LOG_A"
fi
RD_A="$(find "$CLNS_OUTPUT_DIR/smoke/runs" -type d -name 'seed*' 2>/dev/null | head -1)"
if [ -n "$RD_A" ]; then assert_run "$RD_A" "smoke" "$LOG_A" "offline"; else fail "smoke: no run dir under $CLNS_OUTPUT_DIR/smoke/runs"; fi

# --- (a2) GPU only: num_workers>0 + CUDA + checkpoint/resume -------------------
if [ "$ON_GPU" -eq 1 ]; then
  export CLNS_OUTPUT_DIR="$TMP/outputs_resume"   # isolated from (a)
  LOG_R="$TMP/smoke_resume.log"
  RES_ARGS=(experiment=smoke method=er method.mem_size=20 logger=none device=cuda train.num_workers=2)
  if python -m clns.train "${RES_ARGS[@]}" train.stop_after_experiences=1 >"$LOG_R" 2>&1; then
    pass "resume smoke: run stopped after 1 experience (num_workers=2, CUDA) exited 0"
  else
    fail "resume smoke: stop_after_experiences=1 run (tail of log follows)"; tail -n 25 "$LOG_R"
  fi
  RD_R="$(find "$CLNS_OUTPUT_DIR/smoke/runs" -type d -name 'seed*' 2>/dev/null | head -1)"
  if [ -n "$RD_R" ]; then
    check "resume smoke: no DONE and a checkpoint after the stopped run"       bash -c '[ ! -e "$1/DONE" ] && [ -s "$1/checkpoint.pt" ]' _ "$RD_R"
    if python -m clns.train "${RES_ARGS[@]}" >>"$LOG_R" 2>&1; then
      pass "resume smoke: resumed run exited 0"
    else
      fail "resume smoke: resumed run (tail of log follows)"; tail -n 25 "$LOG_R"
    fi
    check "resume smoke: DONE after resume" test -f "$RD_R/DONE"
    check "resume smoke: accuracy_matrix.json has 2 rows"       python -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if len(d["rows"]) == 2 and d.get("random_init") is not None else 1)'       "$RD_R/accuracy_matrix.json"
  else
    fail "resume smoke: no run dir under $CLNS_OUTPUT_DIR/smoke/runs"
  fi
fi

# --- (b) real-data smoke (CLEAR-10) ------------------------------------------
if [ "$REAL" -eq 1 ]; then
  if [ -n "${WANDB_MODE:-}" ]; then MODERE="${WANDB_MODE}"
  elif [ -n "${WANDB_API_KEY:-}" ]; then MODERE="online|offline"   # online expected; offline only on init fallback
  else MODERE="offline"; echo "note: WANDB_API_KEY unset, logger=wandb resolves to offline"; fi
  export CLNS_OUTPUT_DIR="$TMP/outputs_real"
  LOG_B="$TMP/smoke_clear10.log"
  if python -m clns.train experiment=smoke_clear10 logger=wandb \
       logger.project=clns-nonstationarity-smoke logger.group=smoke/clear10 "logger.tags=[smoke]" \
       "${DEV_ARG[@]}" >"$LOG_B" 2>&1; then
    pass "smoke_clear10 run exited 0"
  else
    fail "smoke_clear10 run (tail of log follows)"; tail -n 25 "$LOG_B"
  fi
  RD_B="$(find "$CLNS_OUTPUT_DIR/smoke/runs" -type d -name 'seed*' 2>/dev/null | head -1)"
  if [ -n "$RD_B" ]; then assert_run "$RD_B" "smoke_clear10" "$LOG_B" "$MODERE"; else fail "smoke_clear10: no run dir under $CLNS_OUTPUT_DIR/smoke/runs"; fi
fi

# --- (c) unit tests (slow tests excluded) --------------------------------------
if python -m pytest -q -m "not slow" >"$TMP/pytest.log" 2>&1; then
  pass 'pytest -q -m "not slow"'
else
  fail 'pytest -q -m "not slow" (tail of log follows)'; tail -n 25 "$TMP/pytest.log"
fi

if [ "$NFAIL" -eq 0 ]; then echo "SMOKE TEST: ALL PASS"; exit 0; fi
echo "SMOKE TEST: $NFAIL FAILURE(S)"
exit 1
