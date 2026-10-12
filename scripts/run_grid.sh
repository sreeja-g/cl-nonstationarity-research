#!/usr/bin/env bash
# Run the study grid (benchmarks x model arms x methods x seeds) resumably, detached in tmux.
# Usage: bash scripts/run_grid.sh [--dry-run] [--inner] [--stop]
#                                 [--benchmarks "a b"] [--models "a b"] [--methods "a b"] [--seeds "0 1 2"]
#   --dry-run : print the commands only; launches nothing, needs no /workspace, tmux, flock or GPU.
#   --inner   : run in the foreground (used by the tmux relaunch; also handy for debugging).
#   --stop    : stop the grid cleanly: kills the tmux session, the driver and any `clns.train` processes
#               owned by this user, then releases the lock. Unfinished runs resume from checkpoint.pt.
# Arms (model configs, clns.models.ARM_SPECS) and where they may run -- invalid combinations are SKIPPED
#   (listed by --dry-run; train.py would refuse them with exit 2 anyway):
#     r18_scratch r18_places r50_scratch r50_places : imagenet100 clear10 domainnet core50_nic (+ synthetic)
#     r18_imagenet mbv2_imagenet                    : core50_nic only (ImageNet pretraining overlaps the others)
#     r18_places_all (optional H2 confound check)   : PILOT_BENCHMARK only (default clear10); add it with
#                                                     INCLUDE_PLACES_ALL=1 or MODELS="... r18_places_all"
#   Pilot = a subset of this grid, e.g.: BENCHMARKS=clear10 MODELS=r18_scratch bash scripts/run_grid.sh
# Env overrides: BENCHMARKS (default: imagenet100 clear10 domainnet core50_nic), MODELS (default: the six
#   non-optional arms), METHODS (default: naive ewc lwf er derpp), SEEDS (default: 0 1 2), INCLUDE_PLACES_ALL=1,
#   PILOT_BENCHMARK (default clear10), GPU_HOURS_CAP (see Budget), EXTRA_ARGS (extra hydra overrides, whitespace
#   separated, e.g. 'logger.tags=[a,b] train.epochs=2'; paths.*/run_dir/hydra dir/experiment overrides are
#   rejected, also with any leading '+'/'++'/'~'),
#   WORKSPACE (default /workspace), CLNS_OUTPUT_DIR, RUN_TIMEOUT (per run, default 72h),
#   MIN_FREE_GB (default 20, checked before each run), RETRY_BACKOFF (seconds before the one retry
#   of an exit-1 run, default 60).
# Order: SEED-OUTERMOST -- seed 0 of every benchmark x arm x method first, then seed 1, then seed 2, so a
#   budget cut (or an abort) loses seeds, not whole benchmarks/arms.
# Budget: GPU_HOURS_CAP (default 40 h = ~$24 at $0.59/h) caps the GPU wall clock spent under $OUT, computed by
#   scripts/gpu_hours.py: per run dir the larger of (a) the summed wall clock of EVERY train.py attempt the
#   driver made ($OUT/logs/gpu_ledger.tsv; failed, OOM, timed-out and killed attempts included) and (b)
#   progress.json wall_clock_s_committed (also scans $OUT/tune tuning sweeps and $OUT/smoke, i.e. runs started
#   outside the driver). The running total is logged before each run; once it reaches the cap the grid aborts
#   (driver exit 7). If the accounting itself fails (unreadable progress.json / ledger, python error) the grid
#   ALSO aborts with exit 7 -- the budget is never treated as 0 h. Finished runs are only re-checked (~0 h).
#   Ledger: a 'start' row is written BEFORE each train.py launch and an 'end' row (wall clock, exit code) after
#   it; on SIGTERM/SIGINT the driver forwards TERM to the running train.py, waits up to 60 s, writes its end row
#   and exits 143/130. A start row left open (driver SIGKILLed, pod crash) is charged up to the next ledger
#   row (or now), i.e. conservatively (see scripts/gpu_hours.py).
#   NOT included in the cap: clns.shift.measure_stream GPU time, idle pod time (setup, downloads, analysis, a
#   pod left running), storage. The ~$50 budget is POD hours ($0.59/h whether the GPU is busy or not).
# Re-running is safe: train.py is ALWAYS invoked; for a run dir with DONE it checks the config fingerprint and
#   exits 0 at once (status 'skipped', logged as already_done) or 4 on a mismatch (config_mismatch);
#   unfinished runs resume from checkpoint.pt.
# GPU: a preflight (nvidia-smi + torch.cuda.is_available) runs at start and before every run; if the GPU
#   is missing the whole grid aborts (driver exit 7) instead of silently running on CPU. Runs get device=cuda.
# Exit codes of clns.train -> queue.tsv status:
#   0 ok ('skipped' if DONE existed before the call; 'stopped' if no DONE after it) -> ok / skipped / stopped
#   1 error (failure.json): retried ONCE after RETRY_BACKOFF s                    -> failed
#   2 invalid model config (arm x benchmark, arm spec, missing Places365 weights); no run dir
#     written: not retried                                                        -> invalid_config
#   3 CUDA OOM: not retried                                                       -> oom
#   4 config fingerprint mismatch with an existing run dir: not retried           -> config_mismatch
#   5 run dir locked by another live process: not retried, not a failure          -> locked
#   124/137 per-run timeout hit: not retried (re-run resumes from checkpoint)     -> timeout
#     (GNU timeout sends TERM to the run, which writes failure.json and exits 143, but timeout itself
#     returns 124; 137 = still alive after the -k 60 grace period, KILLed)
#   143 the run got SIGTERM from someone else (failure.json written, W&B finished): not retried, grid
#     continues; a re-run resumes from checkpoint                                 -> stopped
#   129/130 SIGHUP/SIGINT: the grid stops                                         -> interrupted
#   (not started) the grid aborted before/instead of this run: GPU_HOURS_CAP reached, GPU accounting failed,
#     no GPU, low disk                                                            -> aborted
# Driver exit code: 0 no failures; 1 at least one failed/invalid_config/oom/config_mismatch/timeout/interrupted;
#   7 grid aborted (GPU_HOURS_CAP reached / GPU accounting failed / no GPU / low disk / lock held / flock
#   missing); 2 usage error.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
# Keep an explicit caller WANDB_MODE across the env file (which unsets it when a W&B key exists).
_CALLER_WANDB_MODE="${WANDB_MODE-__unset__}"
if [ -f "$WORKSPACE/clns_env.sh" ]; then
  set +u
  # shellcheck disable=SC1091
  source "$WORKSPACE/clns_env.sh"
  set -u
fi
if [ "$_CALLER_WANDB_MODE" != "__unset__" ]; then export WANDB_MODE="$_CALLER_WANDB_MODE"; fi

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${CLNS_OUTPUT_DIR:-$WORKSPACE/outputs}"
BENCHMARKS="${BENCHMARKS:-imagenet100 clear10 domainnet core50_nic}"
PILOT_BENCHMARK="${PILOT_BENCHMARK:-clear10}"
MODELS="${MODELS:-r18_scratch r18_places r50_scratch r50_places r18_imagenet mbv2_imagenet}"
# Optional H2 confound check (Places365 init, everything trainable); pilot benchmark only.
if [ "${INCLUDE_PLACES_ALL:-0}" = "1" ]; then MODELS="$MODELS r18_places_all"; fi
METHODS="${METHODS:-naive ewc lwf er derpp}"
SEEDS="${SEEDS:-0 1 2}"
GPU_HOURS_CAP="${GPU_HOURS_CAP:-40}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_TIMEOUT="${RUN_TIMEOUT:-72h}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
RETRY_BACKOFF="${RETRY_BACKOFF:-60}"
SESSION="clns_grid"
DRY=0
INNER=0
STOP=0

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { echo "error: $*" >&2; exit 2; }

ORIG_ARGS=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --inner) INNER=1 ;;
    --stop) STOP=1 ;;
    --benchmarks|--models|--methods|--seeds)
      [ $# -ge 2 ] || die "$1 needs a value"
      case "$1" in
        --benchmarks) BENCHMARKS="$2" ;;
        --models) MODELS="$2" ;;
        --methods) METHODS="$2" ;;
        --seeds) SEEDS="$2" ;;
      esac
      shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

# --- --stop ------------------------------------------------------------------
if [ "$STOP" -eq 1 ]; then
  uid="$(id -u)"
  mkdir -p "$OUT/logs" 2>/dev/null || true
  echo "STOP requested by $(id -un) at $(date -Is)" >> "$OUT/logs/grid_driver.log" 2>/dev/null || true
  log "stopping grid (tmux session, driver, clns.train processes of uid $uid)"
  if command -v tmux >/dev/null && tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION" || true
  fi
  pkill -TERM -u "$uid" -f 'run_grid\.sh .*--inner' 2>/dev/null || true   # driver first: no next run
  pkill -TERM -u "$uid" -f 'python.* -m clns\.train' 2>/dev/null || true
  for _ in $(seq 1 30); do
    pgrep -u "$uid" -f 'python.* -m clns\.train' >/dev/null 2>&1 || break
    sleep 1
  done
  if pgrep -u "$uid" -f 'python.* -m clns\.train' >/dev/null 2>&1; then
    log "clns.train still alive after 30 s; sending SIGKILL"
    pkill -KILL -u "$uid" -f 'python.* -m clns\.train' || true
  fi
  log "stopped. Unfinished runs resume from checkpoint.pt; re-run this script (or scripts/pod_start.sh) to continue."
  exit 0
fi

# --- validate inputs (before anything is launched) ---------------------------
tok_re='^[A-Za-z0-9_.-]+$'
set -f
read -r -a BENCH_A <<< "$BENCHMARKS"
read -r -a MODEL_A <<< "$MODELS"
read -r -a METH_A  <<< "$METHODS"
read -r -a SEED_A  <<< "$SEEDS"
read -r -a EXTRA_A <<< "$EXTRA_ARGS"
set +f
{ [ "${#BENCH_A[@]}" -gt 0 ] && [ "${#MODEL_A[@]}" -gt 0 ] && [ "${#METH_A[@]}" -gt 0 ] \
    && [ "${#SEED_A[@]}" -gt 0 ]; } || die "BENCHMARKS, MODELS, METHODS and SEEDS must each be non-empty"
for t in "${BENCH_A[@]}" "${MODEL_A[@]}" "${METH_A[@]}"; do [[ "$t" =~ $tok_re ]] || die "bad benchmark/method name '$t'"; done
for t in "${SEED_A[@]}"; do [[ "$t" =~ ^[0-9]+$ ]] || die "seeds must be non-negative integers, got '$t'"; done
[[ "$PILOT_BENCHMARK" =~ $tok_re ]] || die "bad PILOT_BENCHMARK '$PILOT_BENCHMARK'"
[[ "$GPU_HOURS_CAP" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "GPU_HOURS_CAP must be hours (number), got '$GPU_HOURS_CAP'"
[[ "$RUN_TIMEOUT" =~ ^[0-9]+[smhd]?$ ]] || die "RUN_TIMEOUT must look like 90m / 72h / 3d, got '$RUN_TIMEOUT'"
[[ "$MIN_FREE_GB" =~ ^[0-9]+$ ]] || die "MIN_FREE_GB must be an integer"
[[ "$RETRY_BACKOFF" =~ ^[0-9]+$ ]] || die "RETRY_BACKOFF must be an integer (seconds)"
for a in "${EXTRA_A[@]+"${EXTRA_A[@]}"}"; do
  k="$a"
  while [[ "$k" == [+~]* ]]; do k="${k#[+~]}"; done   # '+k=v', '++k=v', '~k' all target key k
  case "$k" in
    paths|paths.*|paths=*|run_dir*|hydra.run*|hydra.sweep*|hydra.job.chdir*|device|device=*|benchmark|benchmark=*|method|method=*|model|model=*|seed|seed=*|experiment|experiment=*|experiment.*)
      die "EXTRA_ARGS override '$a' is not allowed (the grid sets output location, benchmark, method, model, seed and device, and experiment=* configs redirect the output; use CLNS_OUTPUT_DIR / CLNS_DATA_DIR env vars for paths)" ;;
  esac
done

# Arm x benchmark rules; keep in sync with clns.models.ALLOWED (train.py refuses other combinations, exit 2).
MAIN_BENCHMARKS="imagenet100 clear10 domainnet core50_nic"
valid_combo() {  # valid_combo <arm> <benchmark> -> 0 if the grid should run it
  local arm="$1" bench="$2"
  case "$arm" in
    r18_scratch|r18_places|r50_scratch|r50_places) ;;
    r18_imagenet|mbv2_imagenet) [ "$bench" = "core50_nic" ] || [ "$bench" = "synthetic" ]; return ;;
    r18_places_all) [ "$bench" = "$PILOT_BENCHMARK" ] || [ "$bench" = "synthetic" ]; return ;;
    *) die "unknown model arm '$arm' (valid: r18_scratch r18_places r50_scratch r50_places r18_imagenet mbv2_imagenet r18_places_all)" ;;
  esac
  case " $MAIN_BENCHMARKS synthetic " in *" $bench "*) return 0 ;; *) return 1 ;; esac
}
for _m in "${MODEL_A[@]}"; do valid_combo "$_m" synthetic || true; done   # reject unknown arm names early

run_dir_of() {  # contract run dir: <out>/runs/<benchmark>/<model>/<method>/seed<seed>
  echo "$OUT/runs/$1/$2/$3/seed$4"
}

# --- dry run: no side effects at all -----------------------------------------
if [ "$DRY" -eq 1 ]; then
  n=0; skipped=0
  for b in "${BENCH_A[@]}"; do for a in "${MODEL_A[@]}"; do
    if ! valid_combo "$a" "$b"; then
      echo "# skip: model=$a is not run on benchmark=$b"
      skipped=$((skipped + 1))
    fi
  done; done
  echo "# order: seed-outermost (seed ${SEED_A[0]} of every benchmark x arm x method first)"
  for s in "${SEED_A[@]}"; do for b in "${BENCH_A[@]}"; do for a in "${MODEL_A[@]}"; do
    valid_combo "$a" "$b" || continue
    for m in "${METH_A[@]}"; do
      n=$((n + 1))
      echo "python -m clns.train benchmark=$b method=$m model=$a seed=$s device=cuda ${EXTRA_A[*]-}" \
           " # run_dir=$(run_dir_of "$b" "$a" "$m" "$s")  log=$OUT/logs/${b}_${a}_${m}_seed${s}.log"
    done
  done; done; done
  echo "[dry-run] $n commands ($skipped benchmark x arm combinations skipped); nothing launched."
  exit 0
fi

# --- relaunch detached (tmux, else setsid nohup) -----------------------------
if [ "$INNER" -eq 0 ] && [ -z "${TMUX:-}" ]; then
  mkdir -p "$OUT/logs"
  if command -v tmux >/dev/null; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION (stop: bash scripts/run_grid.sh --stop)" >&2
      exit 1
    fi
    # Forward the environment through a private 0600 file that the tmux shell sources and deletes:
    # a running tmux server does not pass the caller's env to new sessions, and `tmux new-session -e K=V`
    # would expose WANDB_API_KEY in the process list. Secret values are never printed.
    FWD="$(mktemp "$OUT/.fwd_env.XXXXXX")"; chmod 600 "$FWD"
    {
      printf 'export WORKSPACE=%q\n' "$WORKSPACE"
      for v in BENCHMARKS MODELS METHODS SEEDS EXTRA_ARGS RUN_TIMEOUT MIN_FREE_GB RETRY_BACKOFF PILOT_BENCHMARK GPU_HOURS_CAP; do
        printf 'export %s=%q\n' "$v" "${!v}"
      done
      printf 'export CLNS_OUTPUT_DIR=%q\n' "$OUT"
      while IFS= read -r v; do
        case "$v" in
          CLNS_OUTPUT_DIR) ;;
          WANDB_*|CLNS_*|HF_*|KAGGLE_*|TORCH_HOME|CUDA_VISIBLE_DEVICES|CUDA_DEVICE_ORDER)
            printf 'export %s=%q\n' "$v" "${!v}" ;;
        esac
      done < <(compgen -e)
    } > "$FWD"
    ARGSTR=""; [ ${#ORIG_ARGS[@]} -gt 0 ] && ARGSTR="$(printf ' %q' "${ORIG_ARGS[@]}")"
    tmux new-session -d -s "$SESSION" ". '$FWD'; rm -f '$FWD'; bash '$SELF' --inner$ARGSTR 2>&1 | tee -a '$OUT/logs/grid_driver.log'"
    log "started in tmux session '$SESSION' (attach: tmux attach -t $SESSION; detach: Ctrl-b d; stop: bash scripts/run_grid.sh --stop)"
  else
    export CLNS_OUTPUT_DIR="$OUT"
    setsid nohup bash "$SELF" --inner ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"} >> "$OUT/logs/grid_driver.log" 2>&1 < /dev/null &
    log "tmux not found; started with setsid nohup (follow: tail -f $OUT/logs/grid_driver.log)"
  fi
  exit 0
fi

# --- core loop ---------------------------------------------------------------
trap '' HUP
export CLNS_OUTPUT_DIR="$OUT"
mkdir -p "$OUT/logs"
command -v flock >/dev/null || { echo "flock is required (apt-get install util-linux)" >&2; exit 7; }
exec 9> "$OUT/.grid.lock"
flock -n 9 || { echo "another run_grid.sh holds $OUT/.grid.lock; exiting." >&2; exit 7; }

QUEUE="$OUT/logs/queue.tsv"
[ -f "$QUEUE" ] || printf 'timestamp\tbenchmark\tmodel\tmethod\tseed\tstatus\texit_code\trun_dir\n' > "$QUEUE"
cd "$REPO"

declare -A COUNT=([ok]=0 [stopped]=0 [skipped]=0 [failed]=0 [invalid_config]=0 [oom]=0 [config_mismatch]=0 [locked]=0 [timeout]=0 [interrupted]=0)
problems=()
total=0
for _b in "${BENCH_A[@]}"; do for _a in "${MODEL_A[@]}"; do
  valid_combo "$_a" "$_b" && total=$(( total + ${#METH_A[@]} * ${#SEED_A[@]} ))
done; done
record() {  # record <b> <a> <m> <s> <status> <rc> <rd>
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" "$4" "$5" "$6" "$7" >> "$QUEUE"
  COUNT[$5]=$(( ${COUNT[$5]:-0} + 1 ))
  case "$5" in ok|skipped|stopped) ;; *) problems+=("$1/$2/$3/seed$4=$5(exit $6)") ;; esac
}
LEDGER="$OUT/logs/gpu_ledger.tsv"
[ -f "$LEDGER" ] || printf 'timestamp\trun_dir\tseconds\texit_code\tstatus\n' > "$LEDGER"
ledger_start() {  # ledger_start <run_dir>: written BEFORE train.py is launched
  printf '%s\t%s\t-\t-\tstart\n' "$(date -Is)" "$1" >> "$LEDGER"
}
ledger_end() {  # ledger_end <run_dir> <seconds> <exit_code>: wall clock of one train.py attempt (any outcome)
  printf '%s\t%s\t%s\t%s\tend\n' "$(date -Is)" "$1" "$2" "$3" >> "$LEDGER"
}
# TERM/INT while train.py runs: forward TERM to it, wait (<= 60 s), write its ledger end row, exit.
# (--stop TERMs the driver first, then the clns.train processes, and SIGKILLs those still alive after 30 s.)
CHILD_PID=""; CHILD_RD=""; CHILD_T0=0
on_signal() {  # on_signal <name> <exit_code>
  trap '' TERM INT
  if [ -n "$CHILD_PID" ]; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$CHILD_PID" 2>/dev/null || break; sleep 1; done
    ledger_end "$CHILD_RD" "$(( SECONDS - CHILD_T0 ))" "$2"
    log "driver got SIG$1 while running $CHILD_RD; ledger end row written; exiting $2"
  else
    log "driver got SIG$1; exiting $2"
  fi
  exit "$2"
}
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT
gpu_hours_used() {  # GPU hours used under $OUT (scripts/gpu_hours.py); returns non-zero if accounting fails
  local out
  out="$(python "$REPO/scripts/gpu_hours.py" "$OUT")" || return 1
  [[ "$out" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "gpu_hours.py printed '$out', not a number" >&2; return 1; }
  echo "$out"
}
check_budget() {  # check_budget <label>: 0 ok, 1 cap reached, 2 accounting failed (sets USED / ABORT)
  if ! USED="$(gpu_hours_used)"; then
    ABORT="GPU-hour accounting failed before $1 (see the gpu_hours.py error above); refusing to continue without a budget"
    return 2
  fi
  log "GPU hours used so far: ${USED} h of GPU_HOURS_CAP=${GPU_HOURS_CAP} h (before $1)"
  if awk -v u="$USED" -v c="$GPU_HOURS_CAP" 'BEGIN{exit !(u >= c)}'; then
    ABORT="GPU_HOURS_CAP reached (${USED} h >= ${GPU_HOURS_CAP} h) before $1; raise GPU_HOURS_CAP to continue"
    return 1
  fi
  return 0
}

gpu_ok() {
  nvidia-smi -L >/dev/null 2>&1 \
    && python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1
}
free_gb() { df -Pk "$OUT" | awk 'NR==2{print int($4/1024/1024)}'; }

ABORT=""
summary() {
  log "SUMMARY (of $total): ok=${COUNT[ok]} stopped=${COUNT[stopped]} skipped(already_done)=${COUNT[skipped]} failed=${COUNT[failed]} invalid_config=${COUNT[invalid_config]} oom=${COUNT[oom]} config_mismatch=${COUNT[config_mismatch]} locked=${COUNT[locked]} timeout=${COUNT[timeout]} interrupted=${COUNT[interrupted]}"
  [ "${#problems[@]}" -eq 0 ] || log "not ok: ${problems[*]}"
  [ -z "$ABORT" ] || log "GRID ABORTED: $ABORT"
  log "queue: $QUEUE"
}

if ! gpu_ok; then
  ABORT="no usable GPU at start (nvidia-smi / torch.cuda.is_available failed) - refusing to run the grid on CPU"
  summary; exit 7
fi
log "GPU preflight OK: $(nvidia-smi -L | head -1)"

stop_grid=0
# Seed-outermost: a budget cut loses seeds, not benchmarks/arms.
for s in "${SEED_A[@]}"; do for b in "${BENCH_A[@]}"; do for a in "${MODEL_A[@]}"; do for m in "${METH_A[@]}"; do
  [ "$stop_grid" -eq 0 ] || break 4
  if ! valid_combo "$a" "$b"; then continue; fi   # e.g. core50-only arms on clear10: not part of the grid
  rd="$(run_dir_of "$b" "$a" "$m" "$s")"
  lg="$OUT/logs/${b}_${a}_${m}_seed${s}.log"
  # No script-side DONE skip: train.py checks the config fingerprint BEFORE its DONE check (exit 4 on a
  # mismatch) and exits 0 at once when DONE exists and the fingerprint matches.
  was_done=0; [ -f "$rd/DONE" ] && was_done=1
  if [ "$was_done" -eq 0 ]; then
    if ! check_budget "$b/$a/$m/seed$s"; then
      stop_grid=1; record "$b" "$a" "$m" "$s" aborted 0 "$rd"; break 4
    fi
  fi
  rc=0; status=failed
  for attempt in 1 2; do
    if ! gpu_ok; then
      ABORT="GPU not available before $b/$a/$m/seed$s (pod restarted without a GPU?)"; stop_grid=1; status=""; break
    fi
    free="$(free_gb)"
    if [ "$free" -lt "$MIN_FREE_GB" ]; then
      ABORT="only ${free} GiB free in $OUT (< MIN_FREE_GB=$MIN_FREE_GB) before $b/$a/$m/seed$s"; stop_grid=1; status=""; break
    fi
    if [ "$was_done" -eq 1 ]; then
      log "CHECK (DONE exists) $b/$a/$m/seed$s: train.py verifies the config fingerprint -> $lg"
    else
      log "RUN $b/$a/$m/seed$s (attempt $attempt, timeout $RUN_TIMEOUT, ${free} GiB free) -> $lg"
    fi
    set +e
    t0=$SECONDS
    ledger_start "$rd"   # before the launch: a driver killed mid-run still leaves a trace in the ledger
    # 9>&- : the child must not inherit the lock fd (an orphan would keep the lock after the driver dies).
    # Launched in the background + wait, so the TERM/INT trap runs at once (not after the child exits).
    timeout -k 60 "$RUN_TIMEOUT" python -m clns.train benchmark="$b" method="$m" model="$a" seed="$s" device=cuda \
      "${EXTRA_A[@]+"${EXTRA_A[@]}"}" >> "$lg" 2>&1 9>&- < /dev/null &
    CHILD_PID=$!; CHILD_RD="$rd"; CHILD_T0=$t0
    wait "$CHILD_PID"
    rc=$?
    CHILD_PID=""
    ledger_end "$rd" "$(( SECONDS - t0 ))" "$rc"   # every attempt counts towards GPU_HOURS_CAP
    set -e
    case "$rc" in
      0)   if [ "$was_done" -eq 1 ]; then log "SKIP (already_done) $b/$a/$m/seed$s"; status=skipped
           elif [ -f "$rd/DONE" ]; then status=ok; else status=stopped; fi; break ;;
      2)   log "INVALID CONFIG (exit 2) for $b/$a/$m/seed$s: model arm/benchmark/weights check failed; not retrying (see $lg)"; status=invalid_config; break ;;
      3)   log "OOM (exit 3) for $b/$a/$m/seed$s; not retrying"; status=oom; break ;;
      4)   log "CONFIG MISMATCH (exit 4) for $b/$a/$m/seed$s: existing run dir has a different config fingerprint; not retrying (see $lg)"; status=config_mismatch; break ;;
      5)   log "LOCKED (exit 5) for $b/$a/$m/seed$s: run dir is in use by another live process; not retrying"; status=locked; break ;;
      124|137) log "TIMEOUT (exit $rc, limit $RUN_TIMEOUT) for $b/$a/$m/seed$s; not retrying (re-run resumes from checkpoint)"; status=timeout; break ;;
      143) log "STOPPED (SIGTERM, exit 143) for $b/$a/$m/seed$s; not retrying (re-run resumes from checkpoint)"; status=stopped; break ;;
      129|130) log "interrupted (signal exit $rc) at $b/$a/$m/seed$s; stopping grid"; status=interrupted; stop_grid=1; break ;;
      *)   log "exit $rc for $b/$a/$m/seed$s"; status=failed
           if [ "$attempt" -eq 1 ]; then
             log "backing off ${RETRY_BACKOFF}s before the single retry"; sleep "$RETRY_BACKOFF"
           fi ;;
    esac
  done
  if [ -n "$status" ]; then record "$b" "$a" "$m" "$s" "$status" "$rc" "$rd"; else record "$b" "$a" "$m" "$s" aborted 0 "$rd"; fi
done; done; done; done

summary
if [ -n "$ABORT" ]; then exit 7; fi
if [ $(( COUNT[failed] + COUNT[invalid_config] + COUNT[oom] + COUNT[config_mismatch] + COUNT[timeout] + COUNT[interrupted] )) -gt 0 ]; then exit 1; fi
exit 0
