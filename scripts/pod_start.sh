#!/usr/bin/env bash
# Idempotent pod (re)start: repair the environment, then relaunch the baseline grid.
# Usage: bash scripts/pod_start.sh [--foreground] [args passed to run_all_baselines.sh, e.g. --seeds "0 1"]
#
# Interactive (default): (1) scripts/runpod_setup.sh --quiet (re-creates the venv / ~/.bashrc hook / env file
# if the container disk was reset; no-ops when everything is in place), (2) scripts/run_all_baselines.sh,
# which starts the grid detached in tmux and RETURNS (exits 1 if a grid is already running in tmux, which is
# harmless). Finished runs are only fingerprint-checked by train.py, unfinished ones resume from checkpoint.
#
# --foreground (for the container start command): same setup, then the grid driver runs in the FOREGROUND
# (`run_all_baselines.sh --inner`, same env file / lock handling as the tmux path; output is appended to
# $CLNS_OUTPUT_DIR/logs/baselines_driver.log and echoed), and afterwards - whatever its exit code, and also
# if setup failed - the script ends in `sleep infinity` so the container stays up for SSH/inspection. A
# start command that only launches tmux and exits would stop the container. Append it to the template's
# own start command so sshd/jupyter still start, e.g. (verify for your template):
#     bash -c '/start.sh & cd /workspace/cl-nonstationarity-research && bash scripts/pod_start.sh --foreground'
#
# It does NOT download data (run scripts/download_data.sh once, by hand) and it needs the same secrets as
# always (WANDB_API_KEY as a RunPod secret; kaggle.json lives in /workspace/.secrets and is only needed for
# the ImageNet download). The grid aborts with exit 7 if the restarted pod has no GPU.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
export WORKSPACE
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log() { printf '[%s] pod_start: %s\n' "$(date +%H:%M:%S)" "$*"; }

FOREGROUND=0
PASS=()
for a in "$@"; do
  if [ "$a" = "--foreground" ]; then FOREGROUND=1; else PASS+=("$a"); fi
done

if [ "$FOREGROUND" -eq 1 ]; then
  keep_alive() { log "keeping the container alive (sleep infinity); stop the pod to end it"; exec sleep infinity; }
  log "environment repair (foreground mode)"
  if ! bash "$REPO/scripts/runpod_setup.sh" --quiet; then
    log "setup failed; not launching the grid"
    keep_alive
  fi
  # Same output dir resolution as run_all_baselines.sh (which sources $WORKSPACE/clns_env.sh itself).
  OUT="$(
    set +eu
    # shellcheck disable=SC1091
    [ -f "$WORKSPACE/clns_env.sh" ] && . "$WORKSPACE/clns_env.sh" >/dev/null 2>&1
    printf '%s' "${CLNS_OUTPUT_DIR:-$WORKSPACE/outputs}"
  )"
  mkdir -p "$OUT/logs"
  log "running the baseline grid in the foreground (log: $OUT/logs/baselines_driver.log)"
  set +e
  bash "$REPO/scripts/run_all_baselines.sh" --inner ${PASS[@]+"${PASS[@]}"} 2>&1 | tee -a "$OUT/logs/baselines_driver.log"
  rc="${PIPESTATUS[0]}"
  set -e
  log "run_all_baselines.sh --inner exited $rc (7 = aborted / another driver holds the lock; see queue.tsv)"
  keep_alive
fi

log "environment repair"
bash "$REPO/scripts/runpod_setup.sh" --quiet || { log "setup failed; not launching the grid"; exit 1; }

log "relaunching the baseline grid (finished runs are only re-checked, unfinished ones resume)"
# run_all_baselines.sh sources $WORKSPACE/clns_env.sh itself (it now exists), so W&B/CLNS_* vars apply.
bash "$REPO/scripts/run_all_baselines.sh" ${PASS[@]+"${PASS[@]}"} || {
  rc=$?
  log "run_all_baselines.sh exited $rc (1 may just mean a grid is already running: tmux attach -t clns_baselines)"
  exit "$rc"
}
log "launched; follow with: tmux attach -t clns_baselines  (stop: bash scripts/run_all_baselines.sh --stop)"
