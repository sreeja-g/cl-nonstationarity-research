#!/usr/bin/env bash
# Idempotent RunPod pod setup: system tools, venv, pinned torch (CUDA 12.1), clns, env file, secrets dir.
# Usage: bash scripts/runpod_setup.sh [--quiet]        (safe to re-run; also run by scripts/pod_start.sh)
#   --quiet : pip/apt output suppressed (used by pod_start.sh)
# Env: WORKSPACE (default /workspace; venv, data, outputs, caches, clns_env.sh and .secrets live under it).
# Pinned (via pyproject.toml / requirements.txt, installed by `pip install -e`): torch 2.3.1,
#   torchvision 0.18.1 (cu121), avalanche-lib 0.6.0, wandb==0.17.9.
# The env file is only ever sourced AFTER the venv exists, so a deleted venv is simply re-created.
set -euo pipefail

QUIET=0
for a in "$@"; do
  case "$a" in
    --quiet) QUIET=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

WORKSPACE="${WORKSPACE:-/workspace}"
ENV_FILE="$WORKSPACE/clns_env.sh"
SECRETS="$WORKSPACE/.secrets"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[%s] WARN: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
fail() { printf '[%s] FAIL: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }
PIPQ=(); [ "$QUIET" -eq 1 ] && PIPQ=(-q)

# Must match requirements.txt / pyproject.toml pins and the cu121 wheel index.
TORCH_VER="$(grep -E '^torch==' "$REPO/requirements.txt" | head -1 | sed -E 's/^torch==([^ #]+).*/\1/')"
TV_VER="$(grep -E '^torchvision==' "$REPO/requirements.txt" | head -1 | sed -E 's/^torchvision==([^ #]+).*/\1/')"
TORCH_INDEX="$(grep -E '^--extra-index-url' "$REPO/requirements.txt" | head -1 | awk '{print $2}')"
[ -n "$TORCH_VER" ] && [ -n "$TV_VER" ] && [ -n "$TORCH_INDEX" ] \
  || fail "could not read torch pins / index from $REPO/requirements.txt"

# --- 1. system packages ------------------------------------------------------
missing=()
command -v tmux    >/dev/null || missing+=(tmux)
command -v git     >/dev/null || missing+=(git)
command -v rsync   >/dev/null || missing+=(rsync)
command -v unzip   >/dev/null || missing+=(unzip)
command -v aria2c  >/dev/null || missing+=(aria2)
command -v flock   >/dev/null || missing+=(util-linux)
python3 -c 'import venv, ensurepip' >/dev/null 2>&1 || missing+=(python3-venv)
if [ "${#missing[@]}" -gt 0 ]; then
  log "apt-get installing: ${missing[*]}"
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  APTQ=(); [ "$QUIET" -eq 1 ] && APTQ=(-qq)
  $SUDO apt-get update -y "${APTQ[@]}"
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y "${APTQ[@]}" --no-install-recommends "${missing[@]}"
else
  log "system packages present"
fi

# --- 2. directories + venv ---------------------------------------------------
mkdir -p "$WORKSPACE"/{data,outputs/logs} "$WORKSPACE"/.cache/{pip,hf,torch,wandb} "$WORKSPACE"/weights
mkdir -p "$SECRETS"; chmod 700 "$SECRETS"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"
VENV="$WORKSPACE/venv"
if [ ! -x "$VENV/bin/python" ]; then
  log "creating venv at $VENV"
  rm -rf "$VENV"
  python3 -m venv "$VENV"
else
  log "venv exists at $VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,12), sys.version' \
  || fail "Python must be >=3.10,<3.12 (got $(python -V)); use a RunPod template with Python 3.10/3.11"
python -m pip install -q --upgrade pip

# --- 3. torch / torchvision (CUDA 12.1 wheels), skipped if already at the pin -
have_torch="$(pip show torch 2>/dev/null | awk '/^Version:/{print $2}')"
have_tv="$(pip show torchvision 2>/dev/null | awk '/^Version:/{print $2}')"
if [ "${have_torch%%+*}" = "$TORCH_VER" ] && [ "${have_tv%%+*}" = "$TV_VER" ]; then
  log "torch $have_torch / torchvision $have_tv already installed"
else
  log "installing torch==$TORCH_VER torchvision==$TV_VER from $TORCH_INDEX"
  pip install "${PIPQ[@]}" "torch==$TORCH_VER" "torchvision==$TV_VER" --index-url "$TORCH_INDEX"
fi

# --- 4. the package (editable); pulls wandb==0.17.9 from pyproject.toml ---------------------
log "pip install -e .[dev]"
pip install "${PIPQ[@]}" -e "$REPO[dev]" --extra-index-url "$TORCH_INDEX"
for pin in wandb==0.17.9; do
  pkg="${pin%%==*}"; want="${pin##*==}"
  have="$(pip show "$pkg" 2>/dev/null | awk '/^Version:/{print $2}')"
  [ "$have" = "$want" ] || warn "$pkg is $have, expected $want (pin in pyproject.toml; verify on first install)"
done

# --- 5. credentials that must survive pod restarts: keep them under $WORKSPACE/.secrets ---------
# Kaggle: KAGGLE_CONFIG_DIR points at $SECRETS, so put kaggle.json there (chmod 600). A kaggle.json
# found in the (ephemeral) home dir is COPIED here once; ~/.kaggle is not modified.
if [ ! -f "$SECRETS/kaggle.json" ] && [ -f "$HOME/.kaggle/kaggle.json" ]; then
  cp "$HOME/.kaggle/kaggle.json" "$SECRETS/kaggle.json"
  log "copied ~/.kaggle/kaggle.json to $SECRETS/kaggle.json"
fi
[ ! -f "$SECRETS/kaggle.json" ] || chmod 600 "$SECRETS/kaggle.json"

# --- 6. environment file (rewritten fully on every run; sourced only if the venv exists) ---------
cat > "$ENV_FILE" <<ENVEOF
# Generated by scripts/runpod_setup.sh -- do not edit (rewritten on every run).
# Caller-provided CLNS_DATA_DIR / CLNS_OUTPUT_DIR / CLNS_WEIGHTS_DIR win over these defaults.
export WORKSPACE="\${WORKSPACE:-$WORKSPACE}"
export CLNS_DATA_DIR="\${CLNS_DATA_DIR:-$WORKSPACE/data}"
export CLNS_OUTPUT_DIR="\${CLNS_OUTPUT_DIR:-$WORKSPACE/outputs}"
export CLNS_WEIGHTS_DIR="\${CLNS_WEIGHTS_DIR:-$WORKSPACE/weights}"
export PIP_CACHE_DIR="$WORKSPACE/.cache/pip"
export HF_HOME="$WORKSPACE/.cache/hf"
export TORCH_HOME="$WORKSPACE/.cache/torch"
export WANDB_DIR="$WORKSPACE/.cache/wandb"
export WANDB_CACHE_DIR="$WORKSPACE/.cache/wandb"
export KAGGLE_CONFIG_DIR="\${KAGGLE_CONFIG_DIR:-$SECRETS}"
export CLNS_REPO="$REPO"
# W&B mode: with a key, never stay sticky-offline; without one, default to offline (an explicit
# WANDB_MODE still wins in that case).
if [ -n "\${WANDB_API_KEY:-}" ]; then
  unset WANDB_MODE
else
  export WANDB_MODE="\${WANDB_MODE:-offline}"
fi
# shellcheck disable=SC1091
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
ENVEOF
# $HOME is wiped on pod restart; re-run (scripts/pod_start.sh does) to restore the ~/.bashrc hook.
if ! grep -qF "$ENV_FILE" "$HOME/.bashrc" 2>/dev/null; then
  echo "[ -f $ENV_FILE ] && source $ENV_FILE" >> "$HOME/.bashrc"
  log "added source line to ~/.bashrc"
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

# --- 7. credential presence (never printed) ----------------------------------
if [ -z "${WANDB_API_KEY:-}" ]; then
  warn "WANDB_API_KEY is not set (add it as a RunPod secret). W&B will run in offline mode; sync later with 'wandb sync'."
else
  log "WANDB_API_KEY is set (W&B online; logger=wandb mode auto)"
fi
if [ -n "${KAGGLE_USERNAME:-}${KAGGLE_KEY:-}" ] || [ -f "$KAGGLE_CONFIG_DIR/kaggle.json" ]; then
  log "Kaggle credentials found (needed only for ImageNet)"
else
  log "no Kaggle credentials (only needed for: download_data.sh imagenet100; put kaggle.json in $SECRETS, chmod 600)"
fi

# --- 8. verification ---------------------------------------------------------
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
else
  warn "nvidia-smi unavailable or no GPU visible"
fi
python - <<'PYEOF' || fail "environment verification failed"
import torch, avalanche, clns
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| avalanche", avalanche.__version__, "| clns", clns.__file__)
if not torch.cuda.is_available():
    print("WARN: CUDA not available")
PYEOF

if [ "$QUIET" -eq 0 ]; then
cat <<'MSG'

Setup complete. Next: open a new shell (or `source $WORKSPACE/clns_env.sh`), then
  bash scripts/smoke_test.sh
Private repo access (optional): add a read-only deploy key to the repo and keep the private key under
$WORKSPACE/.secrets/ (chmod 600; e.g. GIT_SSH_COMMAND="ssh -i $WORKSPACE/.secrets/deploy_key"). This
script never touches keys. Avoid `wandb login` (writes ~/.netrc, wiped on restart): use WANDB_API_KEY.
MSG
fi
