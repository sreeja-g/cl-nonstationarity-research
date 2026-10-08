# clns: continual learning under nonstationarity

Code for a study of how distribution-shift patterns (class-incremental and domain-incremental streams)
affect continual-learning methods. Benchmarks: ImageNet-100, CLEAR-10, DomainNet, CORe50-NIC (plus a
synthetic benchmark for tests). Methods: Naive, EWC, LwF, ER, L2P (more planned). Built on
Avalanche 0.6.0, Hydra and (optionally) Weights & Biases.

This repository is code only. Research notes, the experiment matrix and the literature notes live
outside it, in the sibling folder `../research-hub/` (start at the dashboard `README.md`; then
`02-study-design/`, `03-experiments/` and `01-literature/paper-notes.md`). The older top-level folders `benchmarks/`, `evaluation/`, `experiments/`,
`models/`, `strategies/`, `tuning/`, `utils/` and `notebooks/` are legacy code kept for reference; the
framework is the `src/clns` package.

## Layout

```
pyproject.toml, requirements.txt
configs/            Hydra configs: train.yaml, benchmark/ method/ model/ experiment/ logger/ paths/
data_splits/        committed class lists / domain order (with sha256 checks)
src/clns/           data/ models/ strategies/ evaluation/ shift/ analysis/ utils/ train.py
scripts/            runpod_setup.sh  pod_start.sh  download_data.sh  run_all_baselines.sh  smoke_test.sh
tests/
```

## Install

Python 3.10 or 3.11 only. Pins: torch 2.3.1 / torchvision 0.18.1 (CUDA 12.1 wheels), avalanche-lib 0.6.0,
`timm==0.6.12` (L2P needs the 0.6.x helper API), `wandb==0.17.9` (Avalanche's `WandBLogger` uses a private
wandb call). The last two are installed by `pip install -e` from `pyproject.toml`; `runpod_setup.sh` warns if
the installed versions differ.

Local:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu121
```

### RunPod

Pod requirements: a GPU, Python 3.10/3.11 template, and a **persistent volume at `/workspace` of at least
~200 GiB** (the ImageNet Kaggle zip alone is ~156 GiB and ~25 GiB more is needed while extracting; the kept
data is small). Everything the scripts create lives under `$WORKSPACE` (default `/workspace`): `venv/`,
`data/`, `outputs/`, `.cache/`, `.secrets/`, `clns_env.sh`. Only `/workspace` survives a restart.

```bash
git clone <repo-url> /workspace/cl-nonstationarity-research && cd /workspace/cl-nonstationarity-research
git checkout refactor/nonstationarity-v2
bash scripts/runpod_setup.sh      # idempotent; tools, venv, pinned torch, clns, /workspace/clns_env.sh, /workspace/.secrets
source /workspace/clns_env.sh     # automatic in new shells (hook added to ~/.bashrc)
```

`runpod_setup.sh` can be re-run at any time (it re-creates a deleted venv; the env file is only sourced when the
venv exists). `clns_env.sh` uses `${CLNS_DATA_DIR:-...}` / `${CLNS_OUTPUT_DIR:-...}`, so exporting your own
`CLNS_DATA_DIR` / `CLNS_OUTPUT_DIR` before running any script is honoured.

## Credentials

Provide these as RunPod secrets or environment variables. Never commit them and never put them in configs
(`.env`, `kaggle.json` and `.secrets/` are git-ignored). The scripts never print secret values.

| Name | Needed for | Where / notes |
|---|---|---|
| `WANDB_API_KEY` | W&B online logging | RunPod secret / env var. Use this instead of `wandb login` (which writes `~/.netrc`, wiped on restart). Without it the run logs offline (`wandb sync` later). |
| Kaggle token | ImageNet download only | `kaggle.json` in `/workspace/.secrets/` (`chmod 600`); `KAGGLE_CONFIG_DIR` points there (set by `clns_env.sh`), so it survives restarts. Alternatively `KAGGLE_USERNAME` + `KAGGLE_KEY`. Accept the competition rules once on kaggle.com. A `~/.kaggle/kaggle.json` is copied to `.secrets` by setup. |
| SSH deploy key | cloning a private repo | Optional. Read-only deploy key on GitHub; keep the private key in `/workspace/.secrets/` (`chmod 600`) and use `GIT_SSH_COMMAND="ssh -i /workspace/.secrets/deploy_key"`. The scripts never touch keys. |
| `HF_TOKEN` | Hugging Face downloads | Optional; only if a model/dataset needs it. RunPod secret / env var. |

### W&B mode precedence

`logger=wandb` uses `mode: auto`: `$WANDB_MODE` if set, else **online** if `WANDB_API_KEY` is set, else
**offline** with a warning. Every run logs one line `W&B mode: <none|online|offline|disabled>`. If `wandb.init`
fails (network, auth, timeout; `WANDB_INIT_TIMEOUT`, default 120 s) the run retries once offline and records
`wandb_fallback_offline` in `progress.json`; if that fails too it continues without W&B (`metrics.jsonl` and
`log.txt` are always written). `clns_env.sh` unsets `WANDB_MODE` whenever `WANDB_API_KEY` is set (no sticky
offline mode); to force a mode after sourcing it, pass `WANDB_MODE=offline` on the command line or use
`logger.mode=offline`.

## Data

```bash
bash scripts/download_data.sh clear10|core50|domainnet|imagenet100|all [--force]
```

Data goes to `$CLNS_DATA_DIR` (default `/workspace/data`). Layouts match what `src/clns/data` reads:

| Dataset | Source | Layout |
|---|---|---|
| CLEAR-10 | downloaded by Avalanche 0.6.0 itself (one-off call), so the layout always matches the loader | `<data>/clear10/` |
| CORe50 | same, `nicv2_79` files | `<data>/core50/` |
| DomainNet | cleaned release zips (integrity-tested) and official train/test lists from ai.bu.edu/M3SDA (~17.5 GB zipped) | `<data>/domainnet/<domain>/`, `<data>/domainnet/splits/<domain>_{train,test}.txt` |
| ImageNet-100 | Kaggle `imagenet-object-localization-challenge` (~156 GiB zip, `unzip -t` verified); only the 100 classes in `data_splits/imagenet100/classes.txt` are kept, val re-sorted into wnid folders | `<data>/ILSVRC2012/{train,val}/<wnid>/` |

**Completion markers.** Each dataset root gets a `.done` file (`ILSVRC2012/.done`, `clear10/.done`,
`core50/.done`, `domainnet/.done`) written only after verification (ImageNet: exactly 50 val images per class,
5000 in total, plausible train counts). The loaders refuse a root without it. Escape hatches: create the file
by hand if you know the data is complete, or pass `benchmark.require_done_marker=false`; for CLEAR/CORe50,
`benchmark.allow_download=true` lets Avalanche fetch the data at run time instead (the marker is then not
required). `--force` drops the marker and rebuilds. The free-space check (ImageNet) runs only when a download
is actually needed, so re-running on a full volume is fine.

## Smoke test

```bash
bash scripts/smoke_test.sh            # GPU check, synthetic end-to-end run, artefact assertions, pytest -m "not slow"
bash scripts/smoke_test.sh --real     # also a tiny CLEAR-10 run (needs download_data.sh clear10)
bash scripts/smoke_test.sh --cpu      # local development without a GPU
```

It prints PASS/FAIL lines and exits 1 on any failure. It uses its own temporary output directory (never your
production `$CLNS_OUTPUT_DIR`; `experiment=smoke*` runs resolve to `<output_dir>/smoke/...` anyway). The
synthetic run uses `logger=wandb_offline` and asserts a non-empty `metrics.jsonl`, the `summary.json` keys, a
non-null `peak_gpu_mem_mb` on GPU and the `W&B mode:` log line. On GPU (not `--cpu`) it also checks
checkpoint/resume with the production DataLoader setting: `method=er train.num_workers=2
train.stop_after_experiences=1` (no `DONE`, a checkpoint), then the same command without the stop must resume
to `DONE` with a 2-row accuracy matrix (own temporary output dir). `--real` logs to the W&B project
`clns-nonstationarity-smoke`, group `smoke/clear10`, so production groups stay clean.

## Single run

```bash
python -m clns.train benchmark=imagenet100 method=er model=resnet18 seed=0 device=cuda    # or: clns-train ...
python -m clns.train experiment=smoke logger=wandb_offline
```

Re-running the same command resumes from `checkpoint.pt` and does nothing if `DONE` exists (after the config
fingerprint check, so a `DONE` run dir with a different config still exits 4). On resume the current
`train.eval_mb_size` replaces the value stored in the checkpoint (e.g. lower it after an evaluation OOM);
`train.mb_size` is part of the fingerprint. Exit codes:

| Code | Meaning | Grid driver action (`queue.tsv` status) |
|---|---|---|
| 0 | success, already `DONE`, or stopped via `train.stop_after_experiences` | `ok` / `skipped` (`DONE` existed before the call, logged `already_done`) / `stopped` (no `DONE`) |
| 1 | other error (`failure.json` written) | retried **once** after `RETRY_BACKOFF` s (default 60), then `failed` |
| 3 | CUDA out-of-memory | `oom`, not retried |
| 4 | run dir exists with a different config fingerprint (`fingerprint.json`) | `config_mismatch`, not retried. Use a new output dir or remove the old run dir deliberately. |
| 5 | run dir locked by another live process (`RUNNING.lock`) | `locked`, not retried, not counted as a failure of that run |
| 143 | SIGTERM (`failure.json` with `"interrupted": true`, W&B run marked preempting and finished, lock released; the last checkpoint is kept) | `stopped`, not retried, the grid continues. A per-run `timeout` also sends TERM, but the driver then sees `timeout`'s own 124 -> `timeout` |

Other train options worth knowing: `deterministic=strict` is the default (an op without a deterministic CUDA
kernel raises; rerun with `deterministic=warn` and note it in the results, or `off`). `deterministic` is not
part of the config fingerprint, so such a rerun resumes the same run dir; the mode of every attempt is recorded
(`provenance.json`: `deterministic_mode` and each `resumes[]` entry; `progress.json`: `deterministic_modes`); `train.eval_mask_unseen`
(default `true`) masks logits of classes not yet seen at evaluation (class-IL); `train.eval_valid` (default
`false`, needs `benchmark.val_fraction>0`) also evaluates the validation stream each experience and writes
`valid_matrix.json`. ImageNet-100's class order is fixed by `benchmark.class_order_seed` (1993), independent of
`seed`. EWC with more than 20 train experiences needs `method.mode=online method.decay_factor=<x>`.

## Baseline grid in tmux

```bash
bash scripts/run_all_baselines.sh --dry-run          # print the 18 commands, launch nothing
bash scripts/run_all_baselines.sh                    # starts tmux session "clns_baselines" and returns
tmux attach -t clns_baselines                        # detach with Ctrl-b d; survives SSH disconnects
bash scripts/run_all_baselines.sh --stop             # stop the grid cleanly (see below)
```

Defaults: `BENCHMARKS="imagenet100 clear10"`, `METHODS="naive ewc er"`, `SEEDS="0 1 2"` (override via env vars
or `--benchmarks/--methods/--seeds`), `MODEL=resnet18`. Other env: `EXTRA_ARGS` (extra Hydra overrides,
whitespace separated, e.g. `EXTRA_ARGS='logger.tags=[a,b] train.epochs=2'`; `paths.*`, `run_dir`, `hydra.run.*`,
`experiment=` and the grid's own keys are rejected, also with leading `+`/`++`/`~`; use `CLNS_OUTPUT_DIR` /
`CLNS_DATA_DIR` instead), `RUN_TIMEOUT` (per run,
default `72h`), `MIN_FREE_GB` (default 20, checked before each run), `RETRY_BACKOFF`, `WORKSPACE`.

* Runs get `device=cuda`. A GPU preflight (`nvidia-smi -L` and `torch.cuda.is_available()`) runs at start and
  before every run; if the GPU is missing the **whole grid aborts** (driver exit 7) rather than silently
  running on CPU. Low disk (< `MIN_FREE_GB`) aborts too. A run exceeding `RUN_TIMEOUT` is recorded as `timeout`
  and the grid continues; re-running resumes it from its checkpoint.
* `clns.train` is invoked for every run, also finished ones: with `DONE` and a matching config fingerprint it
  exits 0 at once (`skipped`), with a different config it exits 4 (`config_mismatch`) instead of silently
  reusing stale results. The exit codes above decide retries. A lock keeps two drivers from
  running at once; the lock is not inherited by training processes, so a dead driver never leaves it held.
* `WANDB_API_KEY`, `WANDB_*`, `CLNS_*`, `CUDA_VISIBLE_DEVICES`, `HF_*` and the options above are forwarded into
  the tmux session (through a private temporary file, not the process list).
* Logs: `$CLNS_OUTPUT_DIR/logs/` holds one log per run, `queue.tsv` (one row per run per launch with its
  status) and `baselines_driver.log`, which ends with a `SUMMARY` line (counts per status plus the list of runs
  that are not ok). Driver exit code: 0 no failures, 1 failed/oom/config_mismatch/timeout runs, 7 aborted.
* **Stopping**: `bash scripts/run_all_baselines.sh --stop` kills the tmux session, the driver and any
  `clns.train` process owned by your user (TERM, then KILL after 30 s) and releases the lock. Unfinished runs
  resume from their checkpoint next time.

## After a pod restart or preemption

Only `/workspace` persists; `~/.bashrc`, `~/.netrc`, `~/.kaggle`, apt packages and the tmux session are gone.

```bash
cd /workspace/cl-nonstationarity-research
bash scripts/pod_start.sh        # setup (quiet, repairs venv / env file / bashrc hook) then relaunch the grid
tmux attach -t clns_baselines
```

`pod_start.sh` is idempotent: finished runs are only re-checked (fingerprint), unfinished ones resume. Used
interactively it starts the grid in tmux and returns.

**Automatic restart.** A container start command must not return, or the container stops. Use
`pod_start.sh --foreground`: same quiet setup, then the grid driver runs in the foreground
(`run_all_baselines.sh --inner`, same env file and lock; output also appended to
`$CLNS_OUTPUT_DIR/logs/baselines_driver.log`), and afterwards (whatever the result, also when setup fails) it
ends in `sleep infinity` so the pod stays reachable. Append it to the template's existing start command so
sshd/jupyter still start, e.g. (verify for your template; RunPod templates commonly start services via
`/start.sh`):

```bash
bash -c '/start.sh & cd /workspace/cl-nonstationarity-research && bash scripts/pod_start.sh --foreground >> /workspace/pod_start.log 2>&1'
```

Keep `WANDB_API_KEY` as a RunPod secret. It does **not** download data (run `download_data.sh` once by hand)
and the grid aborts with exit 7 if the new pod has no GPU. `run_all_baselines.sh --stop` also stops a
foreground driver (the pod then idles in `sleep infinity`). Check `queue.tsv` / `baselines_driver.log`
afterwards; `config_mismatch` rows need a manual decision.

## Shift-forgetting analysis

`python -m clns.shift.measure_stream ... --benchmark-seed 0 --out <shift>` writes
`<shift>/<benchmark>/seed0/`. Class orders are fixed by `benchmark.class_order_seed`, not by the training seed,
so one measurement per benchmark is enough: `python -m clns.analysis.correlation_study --runs-dir
<out>/runs --shift-dir <shift> --out corr.csv` pairs each run with any `seed<k>` shift result of its benchmark
whose class order (none for domain-IL), split hashes and T match the run's `benchmark_meta` in
`provenance.json` (the run's own seed is preferred if several match; several other matches, or none, are an
error).

## Outputs

```
$CLNS_OUTPUT_DIR/runs/<benchmark>/<model>/<method>/seed<seed>/
  config.yaml  provenance.json  fingerprint.json  RUNNING.lock  progress.json  checkpoint.pt
  accuracy_matrix.json  per_class_matrix.json  valid_matrix.json (eval_valid)  metrics.jsonl
  summary.json  wandb_run_id.txt  failure.json (on error, removed on success)  DONE (success only)
```

`summary.json` keys include `acc_final`, `acc_avg_incremental`, `bwt`, `forgetting_max`, `fwt`,
`fwt_degenerate`, `peak_gpu_mem_mb` (+ `_per_exp`, `_eval`), `train_time_s_per_exp`, `train_time_s_total`,
`acc_final_macro`, `wall_clock_s_total` (plus `clear_*` for CLEAR-10). Copy `/workspace/outputs` off the pod (or
sync W&B) before terminating it. Only `/workspace` persists.
