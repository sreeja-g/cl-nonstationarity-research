#!/usr/bin/env bash
# Download / prepare datasets under $CLNS_DATA_DIR (default $WORKSPACE/data). Resumable. The per-dataset
# `.done` marker (required by the loaders) is written ONLY after the data was verified complete.
# Usage: bash scripts/download_data.sh <clear10|core50|domainnet|imagenet100|all|places365|dino|weights> [--force]
# Weights (into $CLNS_WEIGHTS_DIR, default $WORKSPACE/weights; NOT part of `all`):
#   places365 : CSAIL resnet18_places365.pth.tar + resnet50_places365.pth.tar (model arms *_places*)
#   dino      : official DINO dino_resnet50_pretrain.pth (shift encoder dino:resnet50)
#   weights   : places365 + dino
#   Downloads go to <file>.part, are verified, then renamed. Each file is verified against SHA_<NAME>
#   (SHA_RESNET18_PLACES365 / SHA_RESNET50_PLACES365 default to the hashes pinned in
#   configs/model/r{18,50}_places*.yaml; an already-present file is re-verified and re-downloaded on
#   mismatch). SHA_DINO_RESNET50 has no default (not yet pinned): its sha256 is written to
#   <file>.sha256 and printed, to be pinned in clns.shift.measure_stream.DINO_RESNET50_SHA256.
# Disk: plan for a persistent volume of >= ~200 GiB. ImageNet needs ~156 GiB for the Kaggle zip plus ~25 GiB
#   while extracting the 100 classes (the zip is deleted afterwards; the kept data is ~15 GiB). The free-space
#   check runs only when something is actually downloaded, so re-runs on a full volume are fine.
#   clear10, core50 : fetched by Avalanche itself (one-off python call; Avalanche's own download=True, the
#                     same effect as benchmark.allow_download=true) into <data>/clear10 and <data>/core50,
#                     the exact roots clns.data expects; .done is written after a file-count check.
#   domainnet       : cleaned release zips + official lists -> <data>/domainnet/{<domain>/, splits/}
#   imagenet100     : Kaggle ILSVRC2012 (needs $KAGGLE_CONFIG_DIR/kaggle.json -- default $WORKSPACE/.secrets --
#                     or KAGGLE_USERNAME/KAGGLE_KEY, and accepted competition rules); only the 100 classes in
#                     data_splits/imagenet100/classes.txt are extracted -> <data>/ILSVRC2012/{train,val}/<wnid>/
#                     with exactly 50 val images per class (5000 in total).
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
if [ -f "$WORKSPACE/clns_env.sh" ]; then
  set +u
  # shellcheck disable=SC1091
  source "$WORKSPACE/clns_env.sh"
  set -u
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${CLNS_DATA_DIR:-$WORKSPACE/data}"
WEIGHTS="${CLNS_WEIGHTS_DIR:-$WORKSPACE/weights}"
# Pinned CSAIL Places365 hashes (verified against the real files 2026-10-11; also in configs/model/).
SHA_RESNET18_PLACES365="${SHA_RESNET18_PLACES365:-2f4759217d470da2b803f8f66cd4488a066406b555a5fb95ee9a4663f9f05588}"
SHA_RESNET50_PLACES365="${SHA_RESNET50_PLACES365:-46529c86902bd0cfb0ea562a30b2850c28d2620d96282b3db9c318e1d774f6c5}"
SHA_DINO_RESNET50="${SHA_DINO_RESNET50:-}"   # unknown yet: recorded + printed, not verified
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { printf '[%s] FAIL: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

FORCE=0
TARGET=""
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) TARGET="$a" ;;
  esac
done
[ -n "$TARGET" ] || fail "usage: $0 <clear10|core50|domainnet|imagenet100|all|places365|dino|weights> [--force]"
mkdir -p "$DATA"

need_space() {  # need_space <GiB> <path>
  local need_gib="$1" path="$2" avail_kib
  [ "$need_gib" -gt 0 ] || return 0
  avail_kib="$(df -Pk "$path" | awk 'NR==2{print $4}')"
  if [ "$avail_kib" -lt $((need_gib * 1024 * 1024)) ]; then
    fail "need ~${need_gib} GiB free in $path, have $((avail_kib / 1024 / 1024)) GiB"
  fi
}

fetch() {  # fetch <url> <dest-file>  (resumable)
  local url="$1" dest="$2"
  if command -v aria2c >/dev/null; then
    aria2c -c -x8 -s8 --console-log-level=warn -d "$(dirname "$dest")" -o "$(basename "$dest")" "$url"
  else
    wget -c -q --show-progress -O "$dest" "$url"
  fi
}

url_ok() {  # url_ok <url>: reachable?
  if command -v curl >/dev/null; then curl -sfIL --max-time 30 "$1" >/dev/null 2>&1
  else wget -q --spider --timeout=30 "$1" >/dev/null 2>&1; fi
}

count_files() { find "$1" -type f 2>/dev/null | head -n 1001 | wc -l; }   # capped at 1001

is_done() {  # --force drops the marker first, so a failed rebuild never leaves a stale .done
  if [ "$FORCE" -eq 1 ]; then rm -f "$DATA/$1/.done"; return 1; fi
  [ -f "$DATA/$1/.done" ]
}

avalanche_fetch() {  # avalanche_fetch <subdir> <python statements using `root`>
  local sub="$1" snippet="$2"
  mkdir -p "$DATA/$sub"
  DATA_ROOT="$DATA/$sub" SNIPPET="$snippet" python -I -c '
import os
from pathlib import Path
root = Path(os.environ["DATA_ROOT"])
exec(os.environ["SNIPPET"])
print("OK", root)
'
}

get_clear10() {
  if is_done clear10; then log "clear10 already done"; return; fi
  need_space 30 "$DATA"   # (verify) CLEAR-10 public release is roughly 10-15 GB incl. extraction
  log "CLEAR-10: letting Avalanche v0.6.0 download into $DATA/clear10"
  avalanche_fetch clear10 'from avalanche.benchmarks.classic.clear import CLEAR
CLEAR(data_name="clear10", evaluation_protocol="iid", seed=0, dataset_root=root)'
  [ "$(count_files "$DATA/clear10")" -gt 1000 ] || fail "CLEAR-10 download looks incomplete (<1000 files under $DATA/clear10); no .done written"
  touch "$DATA/clear10/.done"
}

get_core50() {
  if is_done core50; then log "core50 already done"; return; fi
  need_space 20 "$DATA"   # (verify) CORe50 128x128 is ~1.2 GB zipped; headroom for extraction + pickles
  log "CORe50: letting Avalanche v0.6.0 download into $DATA/core50"
  avalanche_fetch core50 'from avalanche.benchmarks.classic.core50 import CORe50
CORe50(scenario="nicv2_79", run=0, mini=False, object_lvl=True, dataset_root=root)'
  [ "$(count_files "$DATA/core50")" -gt 1000 ] || fail "CORe50 download looks incomplete (<1000 files under $DATA/core50); no .done written"
  touch "$DATA/core50/.done"
}

get_domainnet() {
  if is_done domainnet; then log "domainnet already done"; return; fi
  need_space 60 "$DATA"   # zips ~17.5 GB (sizes listed on ai.bu.edu/M3SDA) + extracted images
  local root="$DATA/domainnet" dom split f first
  local base="https://csr.bu.edu/ftp/visda/2019/multi-source"
  mkdir -p "$root/splits" "$root/_zips"
  url_ok "$base/real.zip" \
    || fail "DomainNet URL $base/real.zip is unreachable; verify the current links at http://ai.bu.edu/M3SDA/ and edit this script"
  # Zip URLs as listed on http://ai.bu.edu/M3SDA/ (cleaned release); no md5 published there.
  declare -A ZIP=(
    [clipart]="$base/groundtruth/clipart.zip"
    [infograph]="$base/infograph.zip"
    [painting]="$base/groundtruth/painting.zip"
    [quickdraw]="$base/quickdraw.zip"
    [real]="$base/real.zip"
    [sketch]="$base/sketch.zip"
  )
  for dom in clipart infograph painting quickdraw real sketch; do
    if [ ! -f "$root/_zips/$dom.extracted" ]; then
      log "DomainNet: $dom"
      fetch "${ZIP[$dom]}" "$root/_zips/$dom.zip"
      unzip -tq "$root/_zips/$dom.zip" >/dev/null \
        || { rm -f "$root/_zips/$dom.zip"; fail "$dom.zip failed the integrity test (deleted; re-run to download again)"; }
      unzip -q -o "$root/_zips/$dom.zip" -d "$root"
      touch "$root/_zips/$dom.extracted"
      rm -f "$root/_zips/$dom.zip"
    fi
    for split in train test; do
      [ -s "$root/splits/${dom}_${split}.txt" ] \
        || fetch "$base/domainnet/txt/${dom}_${split}.txt" "$root/splits/${dom}_${split}.txt"
    done
  done
  # sanity: lists non-empty and their first entry exists under the root (loader layout check)
  for dom in clipart infograph painting quickdraw real sketch; do
    for split in train test; do
      f="$root/splits/${dom}_${split}.txt"
      [ -s "$f" ] || fail "empty/missing $f"
      first="$(head -1 "$f")"
      [ -f "$root/${first%% *}" ] || fail "list $f points to missing file '${first%% *}' (layout mismatch)"
    done
  done
  touch "$root/.done"
}

get_imagenet100() {
  if is_done ILSVRC2012; then log "imagenet100 already done"; return; fi   # no space check / credentials needed
  local classes="$REPO/data_splits/imagenet100/classes.txt"
  [ -f "$classes" ] || fail "missing $classes"
  local dest="$DATA/ILSVRC2012" zipdir="$DATA/_imagenet_zip" tmp="$DATA/_imagenet_tmp"
  local zip="$zipdir/imagenet-object-localization-challenge.zip"
  mkdir -p "$dest" "$zipdir"
  if [ ! -f "$zipdir/.downloaded" ]; then
    if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
      [ -f "${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}/kaggle.json" ] || [ -f "$HOME/.kaggle/kaggle.json" ] \
        || fail "Kaggle credentials missing. Put kaggle.json in ${KAGGLE_CONFIG_DIR:-$WORKSPACE/.secrets} (chmod 600) or set KAGGLE_USERNAME/KAGGLE_KEY (RunPod secrets), and accept the rules at https://www.kaggle.com/c/imagenet-object-localization-challenge/rules"
    fi
    command -v kaggle >/dev/null || pip install -q kaggle
    # zip ~156 GiB + ~25 GiB extraction headroom; a partially downloaded zip already occupies space
    local have_gib=0
    [ -f "$zip" ] && have_gib=$(( $(stat -c %s "$zip") / 1024 / 1024 / 1024 ))
    need_space $(( 185 - have_gib )) "$DATA"
    log "Kaggle download (~156 GiB; resumable by re-running)"
    kaggle competitions download -c imagenet-object-localization-challenge -p "$zipdir"
    [ -f "$zip" ] || fail "expected $zip after the Kaggle download"
    log "verifying zip integrity (unzip -t)"
    unzip -tq "$zip" >/dev/null || { rm -f "$zip"; fail "zip failed the integrity test and was deleted; re-run to download again"; }
    touch "$zipdir/.downloaded"
  else
    need_space 30 "$DATA"   # extraction only
  fi
  local WNIDS
  mapfile -t WNIDS < <(grep -vE '^[[:space:]]*(#|$)' "$classes" | awk '{print $1}')
  [ "${#WNIDS[@]}" -eq 100 ] || fail "expected 100 wnids in $classes, got ${#WNIDS[@]}"
  local pats=("ILSVRC/Data/CLS-LOC/val/*" "LOC_val_solution.csv") w
  for w in "${WNIDS[@]}"; do pats+=("ILSVRC/Data/CLS-LOC/train/$w/*"); done
  rm -rf "$tmp"; mkdir -p "$tmp"
  log "extracting the 100 train classes + val from the zip"
  unzip -q -o "$zip" "${pats[@]}" -d "$tmp"
  mkdir -p "$dest/train" "$dest/val"
  declare -A WANT=()
  for w in "${WNIDS[@]}"; do
    WANT[$w]=1
    rm -rf "${dest:?}/train/$w" "${dest:?}/val/$w"
    mv "$tmp/ILSVRC/Data/CLS-LOC/train/$w" "$dest/train/$w"
    mkdir -p "$dest/val/$w"
  done
  # val: LOC_val_solution.csv rows are "ImageId,<wnid> x0 y0 x1 y1 ..." (possibly CRLF); keep our 100 classes
  local nval=0 img pred cr=$'\r'
  while IFS=, read -r img pred; do
    img="${img%"$cr"}"; pred="${pred%"$cr"}"
    [ "$img" = "ImageId" ] && continue
    [ -n "$img" ] || continue
    w="${pred%% *}"
    if [ -n "$w" ] && [ -n "${WANT[$w]:-}" ]; then
      mv "$tmp/ILSVRC/Data/CLS-LOC/val/$img.JPEG" "$dest/val/$w/"
      nval=$((nval + 1))
    fi
  done < "$tmp/LOC_val_solution.csv"
  [ "$nval" -eq 5000 ] || fail "expected 5000 val images for 100 classes, moved $nval; no .done written"
  # verification before the marker: exactly 50 val images per class, plausible train counts (ILSVRC min is 732)
  local n ntrain=0
  for w in "${WNIDS[@]}"; do
    n="$(find "$dest/val/$w" -maxdepth 1 -type f | wc -l)"
    [ "$n" -eq 50 ] || fail "val class $w has $n images, expected 50; no .done written"
    n="$(find "$dest/train/$w" -maxdepth 1 -type f | wc -l)"
    [ "$n" -ge 700 ] || fail "train class $w has only $n images (expected ~732-1300); no .done written"
    ntrain=$((ntrain + n))
  done
  log "verified: 100 classes, 5000 val images (50/class), $ntrain train images"
  touch "$dest/.done"
  rm -rf "$tmp"
  rm -f "$zip" "$zipdir/.downloaded"   # delete the zip only after the marker exists
  rmdir "$zipdir" 2>/dev/null || true
  log "ImageNet-100 ready at $dest"
}

sha_of() { sha256sum "$1" | awk '{print $1}'; }

get_weight() {  # get_weight <url> <file name> <name of the SHA_* variable>
  local url="$1" name="$2" shavar="$3" dest want have
  dest="$WEIGHTS/$name"
  want="${!shavar:-}"
  want="${want,,}"
  mkdir -p "$WEIGHTS"
  if [ "$FORCE" -eq 1 ]; then rm -f "$dest" "$dest.part" "$dest.sha256"; fi
  if [ -f "$dest" ]; then
    have="$(sha_of "$dest")"
    if [ -n "$want" ] && [ "$have" != "$want" ]; then
      # never silently accept a present file: re-download on mismatch
      log "$name present but sha256 $have != $shavar=$want; deleting and re-downloading"
      rm -f "$dest" "$dest.part" "$dest.sha256"
    else
      log "$name present"
    fi
  fi
  if [ ! -f "$dest" ]; then
    need_space 1 "$WEIGHTS"
    log "fetching $name"
    fetch "$url" "$dest.part"            # resumable; verified, then renamed
    [ -s "$dest.part" ] || fail "$dest.part is missing or empty after download"
    have="$(sha_of "$dest.part")"
    if [ -n "$want" ] && [ "$have" != "$want" ]; then
      rm -f "$dest.part"
      fail "$name sha256 $have != $shavar=$want (download deleted; re-run to download again)"
    fi
    mv -f "$dest.part" "$dest"
  fi
  if [ -n "$want" ]; then
    log "$name sha256 verified"
  else
    log "$name sha256=$have (UNPINNED: set $shavar to verify; pin it, see header)"
  fi
  printf '%s  %s\n' "$have" "$name" > "$dest.sha256"
}

get_places365() {
  # CSAIL Places365 release (http://places2.csail.mit.edu/models_places365/), torchvision ResNets.
  # CSAIL serves plain HTTP only (port 443 does not answer); integrity comes from the pinned
  # sha256 (SHA_RESNET{18,50}_PLACES365 above), checked before the file is accepted.
  local base="http://places2.csail.mit.edu/models_places365"
  get_weight "$base/resnet18_places365.pth.tar" resnet18_places365.pth.tar SHA_RESNET18_PLACES365
  get_weight "$base/resnet50_places365.pth.tar" resnet50_places365.pth.tar SHA_RESNET50_PLACES365
}

get_dino() {
  # Official DINO ResNet-50 backbone-only checkpoint (github.com/facebookresearch/dino); loaded by
  # clns.shift.measure_stream into torchvision resnet50 (no torch.hub).
  get_weight "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth" \
    dino_resnet50_pretrain.pth SHA_DINO_RESNET50
  log "pin: clns.shift.measure_stream.DINO_RESNET50_SHA256 (or pass --dino-sha256)"
}

case "$TARGET" in
  places365)   get_places365 ;;
  dino)        get_dino ;;
  weights)     get_places365; get_dino ;;
  clear10)     get_clear10 ;;
  core50)      get_core50 ;;
  domainnet)   get_domainnet ;;
  imagenet100) get_imagenet100 ;;
  all)         get_clear10; get_core50; get_domainnet; get_imagenet100 ;;
  *) fail "unknown dataset '$TARGET'" ;;
esac
log "done: $TARGET"
