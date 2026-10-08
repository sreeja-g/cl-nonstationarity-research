#!/usr/bin/env bash
# Run the baseline grid (benchmarks x methods x seeds, ResNet-18) resumably, detached in tmux.
# Usage: bash scripts/run_all_baselines.sh [--dry-run] [--inner] [--stop]
#                                          [--benchmarks "a b"] [--methods "a b"] [--seeds "0 1 2"]
#   --dry-run : print the commands only; launches nothing, needs no /workspace, tmux, flock or GPU.
#   --inner   : run in the foreground (used by the tmux relaunch; also handy for debugging).
#   --stop    : stop the grid cleanly: kills the tmux session, the driver and any `clns.train` processes
#               owned by this user, then releases the lock. Unfinished runs resume from checkpoint.pt.
# Env overrides: BENCHMARKS, METHODS, SEEDS, MODEL, EXTRA_ARGS (extra hydra overrides, whitespace
#   separated, e.g. 'logger.tags=[a,b] train.epochs=2'; paths.*/run_dir/hydra dir/experiment overrides are
#   rejected, also with any leading '+'/'++'/'~'),
#   WORKSPACE (default /workspace), CLNS_OUTPUT_DIR, RUN_TIMEOUT (per run, default 72h),
#   MIN_FREE_GB (default 20, checked before each run), RETRY_BACKOFF (seconds before the one retry
#   of an exit-1 run, default 60).
# Re-running is safe: train.py is ALWAYS invoked; for a run dir with DONE it checks the config fingerprint and
#   exits 0 at once (status 'skipped', logged as already_done) or 4 on a mismatch (config_mismatch);
#   unfinished runs resume from checkpoint.pt.
# GPU: a preflight (nvidia-smi + torch.cuda.is_available) runs at start and before every run; if the GPU
#   is missing the whole grid aborts (driver exit 7) instead of silently running on CPU. Runs get device=cuda.
# Exit codes of clns.train -> queue.tsv status:
#   0 ok ('skipped' if DONE existed before the call; 'stopped' if no DONE after it) -> ok / skipped / stopped
#   1 error (failure.json): retried ONCE after RETRY_BACKOFF s                    -> failed
#   3 CUDA OOM: not retried                                                       -> oom
#   4 config fingerprint mismatch with an existing run dir: not retried           -> config_mismatch
#   5 run dir locked by another live process: not retried, not a failure          -> locked
#   124/137 per-run timeout hit: not retried (re-run resumes from checkpoint)     -> timeout
#     (GNU timeout sends TERM to the run, which writes failure.json and exits 143, but timeout itself
#     returns 124; 137 = still alive after the -k 60 grace period, KILLed)
#   143 the run got SIGTERM from someone else (failure.json written, W&B finished): not retried, grid
#     continues; a re-run resumes from checkpoint                                 -> stopped
#   129/130 SIGHUP/SIGINT: the grid stops                                         -> interrupted
# Driver exit code: 0 no failures, 1 at least one failed/oom/config_mismatch/timeout, 7 grid aborted
#   (no GPU / low disk / lock held / flock missing), 2 usage error.
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
BENCHMARKS="${BENCHMARKS:-imagenet100 clear10}"
METHODS="${METHODS:-naive ewc er}"
SEEDS="${SEEDS:-0 1 2}"
MODEL="${MODEL:-resnet18}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_TIMEOUT="${RUN_TIMEOUT:-72h}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
RETRY_BACKOFF="${RETRY_BACKOFF:-60}"
SESSION="clns_baselines"
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
    --benchmarks|--methods|--seeds)
      [ $# -ge 2 ] || die "$1 needs a value"
      case "$1" in
        --benchmarks) BENCHMARKS="$2" ;;
        --methods) METHODS="$2" ;;
        --seeds) SEEDS="$2" ;;
      esac
      shift ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

# --- --stop ------------------------------------------------------------------
if [ "$STOP" -eq 1 ]; then
  uid="$(id -u)"
  mkdir -p "$OUT/logs" 2>/dev/null || true
  echo "STOP requested by $(id -un) at $(date -Is)" >> "$OUT/logs/baselines_driver.log" 2>/dev/null || true
  log "stopping grid (tmux session, driver, clns.train processes of uid $uid)"
  if command -v tmux >/dev/null && tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION" || true
  fi
  pkill -TERM -u "$uid" -f 'run_all_baselines\.sh .*--inner' 2>/dev/null || true   # driver first: no next run
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
read -r -a METH_A  <<< "$METHODS"
read -r -a SEED_A  <<< "$SEEDS"
read -r -a EXTRA_A <<< "$EXTRA_ARGS"
set +f
{ [ "${#BENCH_A[@]}" -gt 0 ] && [ "${#METH_A[@]}" -gt 0 ] && [ "${#SEED_A[@]}" -gt 0 ]; } \
  || die "BENCHMARKS, METHODS and SEEDS must each be non-empty"
for t in "${BENCH_A[@]}" "${METH_A[@]}"; do [[ "$t" =~ $tok_re ]] || die "bad benchmark/method name '$t'"; done
for t in "${SEED_A[@]}"; do [[ "$t" =~ ^[0-9]+$ ]] || die "seeds must be non-negative integers, got '$t'"; done
[[ "$MODEL" =~ $tok_re ]] || die "bad MODEL '$MODEL'"
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

run_dir_of() {  # contract run dir: <out>/runs/<benchmark>/<model>/<method>/seed<seed>
  echo "$OUT/runs/$1/$MODEL/$2/seed$3"
}

# --- dry run: no side effects at all -----------------------------------------
if [ "$DRY" -eq 1 ]; then
  n=0
  for b in "${BENCH_A[@]}"; do for m in "${METH_A[@]}"; do for s in "${SEED_A[@]}"; do
    n=$((n + 1))
    echo "python -m clns.train benchmark=$b method=$m model=$MODEL seed=$s device=cuda ${EXTRA_A[*]-}" \
         " # run_dir=$(run_dir_of "$b" "$m" "$s")  log=$OUT/logs/${b}_${MODEL}_${m}_seed${s}.log"
  done; done; done
  echo "[dry-run] $n commands; nothing launched."
  exit 0
fi

# --- relaunch detached (tmux, else setsid nohup) -----------------------------
if [ "$INNER" -eq 0 ] && [ -z "${TMUX:-}" ]; then
  mkdir -p "$OUT/logs"
  if command -v tmux >/dev/null; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION (stop: bash scripts/run_all_baselines.sh --stop)" >&2
      exit 1
    fi
    # Forward the environment through a private 0600 file that the tmux shell sources and deletes:
    # a running tmux server does not pass the caller's env to new sessions, and `tmux new-session -e K=V`
    # would expose WANDB_API_KEY in the process list. Secret values are never printed.
    FWD="$(mktemp "$OUT/.fwd_env.XXXXXX")"; chmod 600 "$FWD"
    {
      printf 'export WORKSPACE=%q\n' "$WORKSPACE"
      for v in BENCHMARKS METHODS SEEDS MODEL EXTRA_ARGS RUN_TIMEOUT MIN_FREE_GB RETRY_BACKOFF; do
        printf 'export %s=%q\n' "$v" "${!v}"
      done
      printf 'export CLNS_OUTPUT_DIR=%q\n' "$OUT"
      while IFS= read -r v; do
        case "$v" in
          CLNS_OUTPUT_DIR) ;;
          WANDB_*|CLNS_*|HF_*|KAGGLE_*|TORCH_HOME|TIMM_CACHE_DIR|CUDA_VISIBLE_DEVICES|CUDA_DEVICE_ORDER)
            printf 'export %s=%q\n' "$v" "${!v}" ;;
        esac
      done < <(compgen -e)
    } > "$FWD"
    ARGSTR=""; [ ${#ORIG_ARGS[@]} -gt 0 ] && ARGSTR="$(printf ' %q' "${ORIG_ARGS[@]}")"
    tmux new-session -d -s "$SESSION" ". '$FWD'; rm -f '$FWD'; bash '$SELF' --inner$ARGSTR 2>&1 | tee -a '$OUT/logs/baselines_driver.log'"
    log "started in tmux session '$SESSION' (attach: tmux attach -t $SESSION; detach: Ctrl-b d; stop: bash scripts/run_all_baselines.sh --stop)"
  else
    export CLNS_OUTPUT_DIR="$OUT"
    setsid nohup bash "$SELF" --inner ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"} >> "$OUT/logs/baselines_driver.log" 2>&1 < /dev/null &
    log "tmux not found; started with setsid nohup (follow: tail -f $OUT/logs/baselines_driver.log)"
  fi
  exit 0
fi

# --- core loop ---------------------------------------------------------------
trap '' HUP
export CLNS_OUTPUT_DIR="$OUT"
mkdir -p "$OUT/logs"
command -v flock >/dev/null || { echo "flock is required (apt-get install util-linux)" >&2; exit 7; }
exec 9> "$OUT/.baselines.lock"
flock -n 9 || { echo "another run_all_baselines.sh holds $OUT/.baselines.lock; exiting." >&2; exit 7; }

QUEUE="$OUT/logs/queue.tsv"
[ -f "$QUEUE" ] || printf 'timestamp\tbenchmark\tmethod\tseed\tstatus\texit_code\trun_dir\n' > "$QUEUE"
cd "$REPO"

declare -A COUNT=([ok]=0 [stopped]=0 [skipped]=0 [failed]=0 [oom]=0 [config_mismatch]=0 [locked]=0 [timeout]=0 [interrupted]=0)
problems=()
total=$(( ${#BENCH_A[@]} * ${#METH_A[@]} * ${#SEED_A[@]} ))
record() {  # record <b> <m> <s> <status> <rc> <rd>
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" "$4" "$5" "$6" >> "$QUEUE"
  COUNT[$4]=$(( ${COUNT[$4]:-0} + 1 ))
  case "$4" in ok|skipped|stopped) ;; *) problems+=("$1/$2/seed$3=$4(exit $5)") ;; esac
}

gpu_ok() {
  nvidia-smi -L >/dev/null 2>&1 \
    && python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1
}
free_gb() { df -Pk "$OUT" | awk 'NR==2{print int($4/1024/1024)}'; }

ABORT=""
summary() {
  log "SUMMARY (of $total): ok=${COUNT[ok]} stopped=${COUNT[stopped]} skipped(already_done)=${COUNT[skipped]} failed=${COUNT[failed]} oom=${COUNT[oom]} config_mismatch=${COUNT[config_mismatch]} locked=${COUNT[locked]} timeout=${COUNT[timeout]} interrupted=${COUNT[interrupted]}"
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
for b in "${BENCH_A[@]}"; do for m in "${METH_A[@]}"; do for s in "${SEED_A[@]}"; do
  [ "$stop_grid" -eq 0 ] || break 3
  rd="$(run_dir_of "$b" "$m" "$s")"
  lg="$OUT/logs/${b}_${MODEL}_${m}_seed${s}.log"
  # No script-side DONE skip: train.py checks the config fingerprint BEFORE its DONE check (exit 4 on a
  # mismatch) and exits 0 at once when DONE exists and the fingerprint matches.
  was_done=0; [ -f "$rd/DONE" ] && was_done=1
  rc=0; status=failed
  for attempt in 1 2; do
    if ! gpu_ok; then
      ABORT="GPU not available before $b/$m/seed$s (pod restarted without a GPU?)"; stop_grid=1; status=""; break
    fi
    free="$(free_gb)"
    if [ "$free" -lt "$MIN_FREE_GB" ]; then
      ABORT="only ${free} GiB free in $OUT (< MIN_FREE_GB=$MIN_FREE_GB) before $b/$m/seed$s"; stop_grid=1; status=""; break
    fi
    if [ "$was_done" -eq 1 ]; then
      log "CHECK (DONE exists) $b/$m/seed$s: train.py verifies the config fingerprint -> $lg"
    else
      log "RUN $b/$m/seed$s (attempt $attempt, timeout $RUN_TIMEOUT, ${free} GiB free) -> $lg"
    fi
    set +e
    # 9>&- : the child must not inherit the lock fd (an orphan would keep the lock after the driver dies)
    timeout -k 60 "$RUN_TIMEOUT" python -m clns.train benchmark="$b" method="$m" model="$MODEL" seed="$s" device=cuda \
      "${EXTRA_A[@]+"${EXTRA_A[@]}"}" >> "$lg" 2>&1 9>&-
    rc=$?
    set -e
    case "$rc" in
      0)   if [ "$was_done" -eq 1 ]; then log "SKIP (already_done) $b/$m/seed$s"; status=skipped
           elif [ -f "$rd/DONE" ]; then status=ok; else status=stopped; fi; break ;;
      3)   log "OOM (exit 3) for $b/$m/seed$s; not retrying"; status=oom; break ;;
      4)   log "CONFIG MISMATCH (exit 4) for $b/$m/seed$s: existing run dir has a different config fingerprint; not retrying (see $lg)"; status=config_mismatch; break ;;
      5)   log "LOCKED (exit 5) for $b/$m/seed$s: run dir is in use by another live process; not retrying"; status=locked; break ;;
      124|137) log "TIMEOUT (exit $rc, limit $RUN_TIMEOUT) for $b/$m/seed$s; not retrying (re-run resumes from checkpoint)"; status=timeout; break ;;
      143) log "STOPPED (SIGTERM, exit 143) for $b/$m/seed$s; not retrying (re-run resumes from checkpoint)"; status=stopped; break ;;
      129|130) log "interrupted (signal exit $rc) at $b/$m/seed$s; stopping grid"; status=interrupted; stop_grid=1; break ;;
      *)   log "exit $rc for $b/$m/seed$s"; status=failed
           if [ "$attempt" -eq 1 ]; then
             log "backing off ${RETRY_BACKOFF}s before the single retry"; sleep "$RETRY_BACKOFF"
           fi ;;
    esac
  done
  if [ -n "$status" ]; then record "$b" "$m" "$s" "$status" "$rc" "$rd"; else record "$b" "$m" "$s" aborted 0 "$rd"; fi
done; done; done

summary
if [ -n "$ABORT" ]; then exit 7; fi
if [ $(( COUNT[failed] + COUNT[oom] + COUNT[config_mismatch] + COUNT[timeout] + COUNT[interrupted] )) -gt 0 ]; then exit 1; fi
exit 0
