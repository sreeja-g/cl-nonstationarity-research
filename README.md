# clns: continual learning under nonstationarity

Code for a study of how distribution shift between consecutive tasks (class-incremental and
domain-incremental streams) relates to forgetting in continual learning, and whether pretraining
changes that relation. Associational claims only (shift is not randomised).

* **H1**: the larger the shift between two tasks, the more regularisation methods (EWC, LwF) forget;
  replay methods (ER, DER++) are less affected.
* **H2**: a pretrained, partly frozen model is less affected by shift than the same network from scratch.

Benchmarks: ImageNet-100 (20 tasks x 5 classes), CLEAR-10, DomainNet, CORe50-NIC (plus a synthetic benchmark
for tests). Methods: `naive`, `ewc`, `lwf`, `er`, `derpp`. 3 seeds (0, 1, 2); the task order follows the
seed (ImageNet-100 class order, DomainNet domain order, CORe50 NIC run; CLEAR-10 keeps its time order).
Built on Avalanche 0.6.0, Hydra and (optionally) Weights & Biases.

This repository is code only. Research notes, the experiment matrix and the literature notes live
outside it, in the sibling folder `../research-hub/` (start at the dashboard `README.md`; then
`02-study-design/`, `03-experiments/` and `01-literature/paper-notes.md`). The v1 design (ResNet-18 vs
ViT/L2P) is archived on the branch `archive/v1-resnet-vit`; the framework is the `src/clns` package.

## Model arms

An arm is a model config (`configs/model/<arm>.yaml`; `model.name` = arm id, which is part of the run
directory). Arch / init / trainable are fixed per arm (`clns.models.ARM_SPECS`); `build_model` refuses other
arm x benchmark combinations.

| Arm | Architecture | Init | Trainable | Benchmarks |
|---|---|---|---|---|
| `r18_scratch` | ResNet-18 | random | everything | all four |
| `r18_places` | ResNet-18 | Places365 | `layer4` + `fc` | all four |
| `r50_scratch` | ResNet-50 | random | everything | all four |
| `r50_places` | ResNet-50 | Places365 | `layer4` + `fc` | all four |
| `r18_imagenet` | ResNet-18 | ImageNet-1k | `layer4` + `fc` | CORe50-NIC only |
| `mbv2_imagenet` | MobileNetV2 | ImageNet-1k | `features[17]`, `features[18]`, `classifier` | CORe50-NIC only |
| `r18_places_all` | ResNet-18 | Places365 | everything | optional H2 confound check, pilot benchmark only |

H2 compares the matched pairs `r18_scratch`/`r18_places` and `r50_scratch`/`r50_places`; `r18_places_all` checks
whether an effect comes from pretraining or from freezing. ImageNet-initialised arms overlap with ImageNet-100
(and, through ImageNet/DomainNet/CLEAR content, the others), hence CORe50-NIC only. Frozen parts use
`requires_grad=False` and `FrozenBatchNorm2d`; trainable / total parameter counts are logged to `provenance.json`.
All arms use the benchmark's image size and ImageNet normalisation.

**Pretrained weights.** Places365 (CSAIL `resnet{18,50}_places365.pth.tar`) lives in `$CLNS_WEIGHTS_DIR`
(default `/workspace/weights`); their sha256 are pinned in `configs/model/` (verified against the real files).
`bash scripts/download_data.sh places365|dino|weights` fetches and verifies them. ImageNet arms use pinned
torchvision enums (`IMAGENET1K_V1`). The DINO ResNet-50 (shift encoder `dino:resnet50`) has **no pinned sha256
yet**: it is recorded in `<weights>/dino_resnet50_pretrain.pth.sha256` on the first download on RunPod and must
be pinned (`DINO_RESNET50_SHA256` in `clns.shift.measure_stream`, or `--dino-sha256`) before shift is measured.
Pin any pretrained weights (`weights_sha256` in the arm config) before the first pretrained GPU run: the weights
hash is part of the run fingerprint.

## Layout

```
pyproject.toml, requirements.txt
configs/            Hydra configs: train.yaml, benchmark/ method/ model/ experiment/ logger/ paths/
data_splits/        committed class lists / domain order (with sha256 checks)
src/clns/           data/ models/ strategies/ evaluation/ shift/ analysis/ utils/ train.py
scripts/            runpod_setup.sh  pod_start.sh  download_data.sh  run_grid.sh  smoke_test.sh
tests/
```

## Install

Python 3.10 or 3.11 only. Pins: torch 2.3.1 / torchvision 0.18.1 (CUDA 12.1 wheels), avalanche-lib 0.6.0,
`wandb==0.17.9` (Avalanche's `WandBLogger` uses a private wandb call). Both are installed by `pip install -e` from
`pyproject.toml`; `runpod_setup.sh` warns if the installed versions differ. Optional extras: `[dev]` (pytest, ruff),
`[plots]` (matplotlib, only for `clns.analysis.plots`).

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
bash scripts/download_data.sh places365|dino|weights        # pretrained weights -> $CLNS_WEIGHTS_DIR (not part of `all`)
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
python -m clns.train benchmark=imagenet100 method=er model=r18_scratch seed=0 device=cuda    # or: clns-train ...
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
| 2 | invalid model config (arm x benchmark, arm spec, missing Places365 weights); **no run dir is written** | `invalid_config`, not retried |
| 3 | CUDA out-of-memory | `oom`, not retried |
| 4 | run dir exists with a different config fingerprint (`fingerprint.json`) | `config_mismatch`, not retried. Use a new output dir or remove the old run dir deliberately. |
| 5 | run dir locked by another live process (`RUNNING.lock`) | `locked`, not retried, not counted as a failure of that run |
| 143 | SIGTERM (`failure.json` with `"interrupted": true`, W&B run marked preempting and finished, lock released; the last checkpoint is kept) | `stopped`, not retried, the grid continues. A per-run `timeout` also sends TERM, but the driver then sees `timeout`'s own 124 -> `timeout` |

Other train options worth knowing: `deterministic=strict` is the default (an op without a deterministic CUDA
kernel raises; rerun with `deterministic=warn` and note it in the results, or `off`). `deterministic` is not
part of the config fingerprint, so such a rerun resumes the same run dir; the mode of every attempt is recorded
(`provenance.json`: `deterministic_mode` and each `resumes[]` entry; `progress.json`: `deterministic_modes`);
`train.eval_mask_unseen` (default `true`) masks logits of classes not yet seen at evaluation (class-IL);
`train.eval_valid` (default `false`, needs `benchmark.val_fraction>0`) also evaluates the validation stream each
experience and writes `valid_matrix.json`. `train.probe` (default on) dumps penultimate features of a fixed probe
set after every experience (`probe/`, below). EWC runs in `online` mode with `decay_factor` on every benchmark;
ER and DER++ share `mem_size`/`batch_size_mem`.

## Study grid in tmux

```bash
bash scripts/run_grid.sh --dry-run          # print the commands (and the skipped arm x benchmark combinations)
bash scripts/run_grid.sh                    # starts tmux session "clns_grid" and returns
tmux attach -t clns_grid                    # detach with Ctrl-b d; survives SSH disconnects
bash scripts/run_grid.sh --stop             # stop the grid cleanly (see below)
```

Defaults: `BENCHMARKS="imagenet100 clear10 domainnet core50_nic"`, `MODELS="r18_scratch r18_places r50_scratch
r50_places r18_imagenet mbv2_imagenet"`, `METHODS="naive ewc lwf er derpp"`, `SEEDS="0 1 2"` (override via env
vars or `--benchmarks/--models/--methods/--seeds`). Invalid arm x benchmark combinations are skipped (the
CORe50-only arms never run on the other benchmarks). `INCLUDE_PLACES_ALL=1` adds `r18_places_all`, which runs on
`PILOT_BENCHMARK` (default `clear10`) only. Other env: `EXTRA_ARGS` (extra Hydra overrides, whitespace separated,
e.g. `EXTRA_ARGS='logger.tags=[a,b] train.epochs=2'`; `paths.*`, `run_dir`, `hydra.run.*`, `experiment=` and the
grid's own keys are rejected, also with leading `+`/`++`/`~`; use `CLNS_OUTPUT_DIR` / `CLNS_DATA_DIR` instead),
`RUN_TIMEOUT` (per run, default `72h`), `MIN_FREE_GB` (default 20, checked before each run), `RETRY_BACKOFF`,
`GPU_HOURS_CAP` (default 40 h, about $24 at $0.59/h), `WORKSPACE`.

**Pilot** = a subset of the grid, e.g. the cheapest arm on the cheapest benchmark:

```bash
BENCHMARKS=clear10 MODELS=r18_scratch bash scripts/run_grid.sh
```

**Order of work: tune -> freeze -> pilot -> grid.** (1) Tune (next section); (2) write the chosen values into
`configs/method/*.yaml`; (3) run the pilot; (4) run the grid. Method hyper-parameters are part of the config
fingerprint: **changing `configs/method/*.yaml` after runs exist makes every existing run dir of that method
exit 4 (`config_mismatch`)** on the next launch, pilot runs included. Freeze the method configs before the pilot;
if they must change later, use a new `CLNS_OUTPUT_DIR` (or move the old run dirs away deliberately).

* **Order.** The grid runs **seed-outermost**: seed 0 of every benchmark x arm x method first, then seed 1, then
  seed 2, so a budget cut (or any abort) loses seeds, not whole benchmarks or arms.
* **Budget.** `scripts/gpu_hours.py` computes the GPU hours spent under `$CLNS_OUTPUT_DIR`: per run dir the larger
  of (a) the summed wall clock of **every** `train.py` attempt the driver made (`logs/gpu_ledger.tsv`; failed,
  OOM, timed-out and killed attempts included) and (b) `progress.json` `wall_clock_s_committed` (also for the
  `tune/` sweeps and `smoke/` runs started outside the driver; that field only counts time up to the last
  checkpoint, so failed attempts of runs started by hand are undercounted). The total is logged before every new
  run; once it reaches `GPU_HOURS_CAP` the grid aborts (driver exit 7). If the accounting itself fails
  (unreadable `progress.json` or ledger) the grid also aborts with exit 7 and a message: the budget is never
  treated as 0 h. Finished runs are only re-checked and cost ~nothing; raise the cap to continue.
* **Ledger robustness.** The driver writes a `start` row to `gpu_ledger.tsv` *before* launching `train.py` and an
  `end` row (wall clock, exit code) after it returns (columns `timestamp, run_dir, seconds, exit_code, status`;
  older 4-column rows count as end rows). On SIGTERM/SIGINT the driver forwards TERM to the running `train.py`,
  waits up to 60 s, writes that attempt's end row and exits 143/130 (`--stop` behaves as before). A start row that
  is never closed (driver SIGKILLed, pod crash) is charged from its timestamp to the next ledger row, or to *now*
  if it is the last row -- conservative: pod downtime after a crash is counted. To correct it, append an end row
  for that run dir by hand (`<ISO time>\t<run_dir>\t<seconds>\t<rc>\tend`).
* **What the cap does NOT include**: `clns.shift.measure_stream` GPU time, idle pod time (setup, downloads,
  analysis, a pod left running after the grid) and storage. The ~$50 budget is **pod hours** ($0.59/h whether the
  GPU is busy or not), so keep `GPU_HOURS_CAP` well below budget / hourly rate.
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
  status), `gpu_ledger.tsv` (start / end row of every attempt) and `grid_driver.log`, which ends with a `SUMMARY`
  line (counts per status plus the list of runs that are not ok). Besides the statuses in the exit-code table,
  `queue.tsv` has `interrupted` (SIGHUP/SIGINT exit 129/130: the grid stops) and `aborted` (the run was not
  started because the grid aborted: `GPU_HOURS_CAP` reached, accounting failed, no GPU, low disk). Driver exit
  code: 0 no failures; 1 at least one `failed` / `invalid_config` / `oom` / `config_mismatch` / `timeout` /
  `interrupted` run; 7 aborted (`GPU_HOURS_CAP` reached, GPU accounting failed, no GPU, low disk, lock held,
  `flock` missing); 2 usage error.
* **Stopping**: `bash scripts/run_grid.sh --stop` kills the tmux session, the driver and any
  `clns.train` process owned by your user (TERM, then KILL after 30 s) and releases the lock. Unfinished runs
  resume from their checkpoint next time.

## Hyper-parameter tuning

`configs/experiment/tune.yaml` runs on the pilot benchmark (CLEAR-10), the first 3 experiences (transitions 0
and 1), with a class-balanced validation stream carved from TRAIN (`valid_matrix.json` is the score; test is
never used) and its own output root `$CLNS_OUTPUT_DIR/tune`. Unlike production runs, a tuning run dir encodes
the overridden hyper-parameters (Hydra `job.override_dirname`), e.g.
`$CLNS_OUTPUT_DIR/tune/runs/clear10/r18_scratch/ewc/hp_method.ewc_lambda=100/seed0`, so every value of a sweep
gets its own dir (without it every job after the first would exit 4). Volatile keys (`device`, `deterministic`,
`train.num_workers`, `train.eval_mb_size`, `train.stop_after_experiences`, `logger` and every `logger.*` key:
`name, project, entity, group, tags, mode`; `paths` and every `paths.*` key: `data_dir, output_dir, weights_dir`)
are not part of that name, so a resume finds its dir. Hydra matches these keys exactly, so a key added to
`configs/logger/` or `configs/paths/` must be added to `exclude_keys` in `tune.yaml` (a test checks this).
Override values become a directory name: use only letters, digits and `. _ - = ,` (no `/ \ : * ? " < > |`, no
spaces), and keep the `hp_` segment short (at most 255 bytes per path component on Linux; Windows limits paths
to 260 characters unless long paths are enabled): sweep few keys per job with short values (`1e3`, not long
lists); no hashing is applied. One small grid per method with Hydra multirun:

```bash
python -m clns.train experiment=tune method=ewc method.ewc_lambda=100,1000,10000 seed=0 --multirun
python -m clns.train experiment=tune method=lwf method.alpha=0.5,1,2 seed=0 --multirun
python -m clns.train experiment=tune method=er method.mem_size=500,2000 seed=0 --multirun
python -m clns.train experiment=tune method=derpp method.alpha=0.1,0.3 method.beta=0.5,1.0 seed=0 --multirun
```

Then write the chosen values into `configs/method/*.yaml` **before the pilot and the grid** (order: tune ->
`configs/method/*.yaml` -> pilot -> grid); they are reused on every benchmark and arm. Changing a method config
after pilot/grid runs exist makes those runs exit 4 (config fingerprint) on the next launch. The tuned
transitions are flagged in the analysis (`--tuned-transitions clear10:0,clear10:1`). Tuning runs count towards
`GPU_HOURS_CAP` when the grid uses the same `CLNS_OUTPUT_DIR`.

## After a pod restart or preemption

Only `/workspace` persists; `~/.bashrc`, `~/.netrc`, `~/.kaggle`, apt packages and the tmux session are gone.

```bash
cd /workspace/cl-nonstationarity-research
bash scripts/pod_start.sh        # setup (quiet, repairs venv / env file / bashrc hook) then relaunch the grid
tmux attach -t clns_grid
```

`pod_start.sh` is idempotent: finished runs are only re-checked (fingerprint), unfinished ones resume. Used
interactively it starts the grid in tmux and returns.

**Automatic restart.** A container start command must not return, or the container stops. Use
`pod_start.sh --foreground`: same quiet setup, then the grid driver runs in the foreground
(`run_grid.sh --inner`, same env file and lock; output also appended to
`$CLNS_OUTPUT_DIR/logs/grid_driver.log`), and afterwards (whatever the result, also when setup fails) it
ends in `sleep infinity` so the pod stays reachable. Append it to the template's existing start command so
sshd/jupyter still start, e.g. (verify for your template; RunPod templates commonly start services via
`/start.sh`):

```bash
bash -c '/start.sh & cd /workspace/cl-nonstationarity-research && bash scripts/pod_start.sh --foreground >> /workspace/pod_start.log 2>&1'
```

Keep `WANDB_API_KEY` as a RunPod secret. It does **not** download data (run `download_data.sh` once by hand)
and the grid aborts with exit 7 if the new pod has no GPU. `run_grid.sh --stop` also stops a
foreground driver (the pod then idles in `sleep infinity`). Check `queue.tsv` / `grid_driver.log`
afterwards; `config_mismatch` rows need a manual decision.

## Shift, probe and analysis

All commands below read finished runs (`DONE`) under `$CLNS_OUTPUT_DIR/runs`.

**1. Shift: measure once per run seed.** The task order follows the seed, so a run is only paired with a shift
matrix measured on its own order (`task_order_sha256`); a run without a recorded task order (pre-v2) never pairs
with a shift matrix that has one. CLEAR-10's order is seed-independent (one seed suffices there, more is harmless).

```bash
for k in 0 1 2; do
  python -m clns.shift.measure_stream --benchmark-config configs/benchmark/imagenet100.yaml \
      --encoder tv:resnet50 --seed $k --out $CLNS_OUTPUT_DIR/shift
done
```

Add `--pairs full` to measure the full upper triangle of the shift matrix (every task pair), which the
`all_pairs` analysis (step 4, `--pairs all|both`) needs; the default `--pairs consecutive` only measures
`M[k, k+1]`. Writes `<shift>/<benchmark>/seed<k>/{shift_matrix.npz, shift_pairs.csv, shift_meta.json}`
(`shift_meta.json` records `pairs`). Encoders: `tv:resnet50`
(ImageNet) and `dino:resnet50` (needs the pinned DINO weights, see above). MMD is computed on L2-normalised
features with one fixed bandwidth set per encoder and a fixed number of images per experience; features are
cached per image, so extra seeds are cheap. Shift is a marginal MMD^2 and is **not comparable across benchmarks**.

**2. Probe features / CKA.** With `train.probe.enabled` (default) the run dir gets `probe/meta.npz` and
`probe/row_XXXX.npz` (penultimate features of a fixed probe set after each experience; the probe set does not
depend on the training seed). Read them with `clns.evaluation.probe.load_probe(run_dir / "probe")`; linear CKA
is `clns.shift.metrics.linear_cka`. Representation drift per run and transition:

```bash
python -m clns.analysis.cka --runs-dir $CLNS_OUTPUT_DIR/runs --out $CLNS_OUTPUT_DIR/analysis
```

writes `cka.csv` (`benchmark, model, method, seed, k, cka_all, cka_old, n_probe, n_old, old_def, reason`):
debiased linear CKA (float32 features) between probe row k and row k+1, over all probe images (`cka_all`) and
over the probe images of **old tasks** (`cka_old`): images whose test experience maps, through the run's
`test_exp_to_train_exp` (`accuracy_matrix.json`), to a training experience <= k. Domain-IL: old = domains 0..k
(so `cka_old` differs from `cka_all`); class-IL: old = the classes of experiences 0..k; CORe50 NIC: the
first-appearance test groups introduced in experiences 0..k (`old_def` names the rule; without a mapping the
identity is assumed and `old_def` says so). A value needs at least **25 images**; below that it is NaN and
`reason` says why. Debiased CKA **can be negative**, and for independent representations its noise sd is roughly
1/sqrt(n) (n = `n_old` or `n_probe`), so values of that size are indistinguishable from 0. Runs without `DONE` or
without probe features are skipped with a warning.

**3. Confounds table.** `python -m clns.data.confounds --runs-root $CLNS_OUTPUT_DIR/runs --out confounds.csv`
(benchmark properties and trainable parameter counts per benchmark x arm).

**4. Hypothesis tests (H1/H2, primary).**

```bash
python -m clns.analysis.hypotheses --runs-dir $CLNS_OUTPUT_DIR/runs --shift-dir $CLNS_OUTPUT_DIR/shift \
    --out $CLNS_OUTPUT_DIR/analysis --pairs both --tuned-transitions clear10:0,clear10:1
```

One-step forgetting vs within-benchmark z-scored shift (`shift_z`). H1: regularisation (EWC, LwF) minus replay
(ER, DER++) shift slope; H2: scratch minus pretrained on the matched pairs (r18_scratch/r18_places,
r50_scratch/r50_places). `delta > 0` always means "as predicted". `a_tt` = accuracy on the new task right after
training on it (plasticity), `t` = stream index of the transition (the coefficient is called `t`, not
`position`; the table column `position` = T-1-k). Every analysis is repeated without `a_tt` (`variant=no_a_tt`,
it may be post-treatment); **CORe50** is fitted without `a_tt` (undefined for batches without a new class). The
`analysis` column says which analysis a row belongs to:

* **`analysis=two_stage` -- primary, confirmatory.** A run = (arm, method, seed). Stage 1: per run, OLS
  `forget ~ shift_z + a_tt + t` (CORe50: no `a_tt`) on its own rows -> run slope. Runs with < 4 rows or a rank
  deficient design are skipped and counted (`n_runs_skipped`). Stage 2 per benchmark: **H1** per arm (primary)
  and `all_arms`: **term `delta` is the primary test**: OLS of the run slopes on `1 + replay + seed FE` (+ arm FE
  for `all_arms`), `delta = -coef(replay)` = regularisation minus replay slope adjusted for the seed
  (`ci_method=ols_seed_fe(run_slopes)`; classical OLS SE, t CI with the residual df = n_runs - columns, e.g.
  12 runs / 3 seeds -> df 8; `estimate, se, df, ci_lo, ci_hi, p_value`). The seed fixes the task order
  (ImageNet-100, DomainNet, CORe50), so runs of one seed share its transitions; the seed FE remove that shared
  slope noise (for CLEAR-10 the order is fixed and they absorb init-level shifts). **Term `delta_welch` is
  secondary**: a Welch two-sample t-test of the same slopes ignoring seeds (`ci_method=welch_t(run_slopes)`),
  conservative when seeds differ in their transitions. `n_runs_ref` = regularisation runs, `n_runs_other` =
  replay runs. `slope_regularization` / `slope_replay` and naive (`H1_naive`) are group means with a one-sample
  t CI (`one_sample_t(run_slopes)`). **H2** per pair (`pair=r18`, `pair=r50`) and
  `pooled_pairs`: runs paired by (method, seed), mean of `b_scratch - b_pretrained` with a paired t CI
  (`paired_t(run_slopes)`, `n_pairs`); per family (`pair=r18|family=regularization` / `replay`) as secondary
  scopes; runs without a partner are dropped (noted). Inference generalises over runs (seeds, methods) *given the
  benchmark's transitions*, not over tasks. Review-05 simulation (19 transitions x 3 seeds, 12 runs, 400
  replicates; 95% CIs):

  | world | `delta` (seed FE, primary) | `delta_welch` (secondary) |
  |---|---|---|
  | null, per-run slope sd 0 | 0.043 | 0.003 |
  | null, per-run slope sd 0.05 | 0.048 | 0.010 |
  | null, per-run slope sd 0.10 | 0.052 | 0.018 |
  | planted 0.05, sd 0.05 | 0.338 | 0.113 |
  | planted 0.10, sd 0.05 | 0.792 | 0.472 |
  | planted 0.10, sd 0.10 | 0.372 | 0.242 |
  | null with a transition x family shock (sd 0.3) shared by a family's runs | 0.285 | 0.125 |

  The last row is the known limit of "inference given the transitions": a shock shared by all runs of one
  family on the same transitions is indistinguishable from a family slope difference on these tasks.
* **`analysis=pooled_transition` -- secondary, always `descriptive_only=True`.** The pooled OLS
  `forget ~ shift_z * family + a_tt + t + method FE` (H1; within-family method dummies, + arm FE for `all_arms`)
  and `forget ~ shift_z * pretrained + a_tt + t + method FE` (H2; + pair FE for `pooled_pairs`) with 95% wild
  cluster bootstrap-t CIs clustered by transition (`wild_cluster_bootstrap_t(webb)`, `n_clusters`, no CI below 5
  clusters). It ignores per-run slope heterogeneity, which made its CIs far too narrow in simulation (FPR
  0.35-0.50 at a per-run slope sd of 0.05-0.10), hence descriptive only.
* **`analysis=accounting`.** Per benchmark one row `term=n_runs_shift_mismatch` (`estimate` = runs skipped,
  `n_runs_loaded`), written even if the benchmark yields no fit; with `--pairs both`, one row
  `term=all_pairs_skipped` (the only rows with `all_pairs_skipped=True`) per benchmark whose shift was not
  measured with `--pairs full`.

Scopes: `pairs=consecutive` (primary): the transition k -> k+1 with shift `M[k,k+1]`; `pairs=all_pairs`
(`--pairs all|both`): every (old task j, later transition) pair with shift `M[m_j, t+1]`, always
`descriptive_only=True` (rows sharing the old task j are dependent, not modelled); it **needs shift measured
with `measure_stream --pairs full`**: `--pairs all` stops with an error otherwise, `--pairs both` skips all_pairs
for that benchmark with a WARNING and the accounting row, never a silent copy of `consecutive`. Near-chance
transitions (`a_tt` below chance + margin; chance = 1 / classes the model can predict: classes seen so far in
class-IL with eval-time masking, all classes in domain-IL) and tuned transitions are flagged and counted
(`n_near_chance`, `n_tuned`; `--exclude-flagged` drops them before stage 1). **Shift mismatches**: a run whose
shift result does not match its task order / split hashes is skipped, a `WARNING: benchmark X: n of N runs
skipped` summary is printed and the accounting row written; **without `--allow-partial` the CLI writes the
outputs and then exits 3**, because confirmatory output must cover every run (`--allow-partial` accepts the
partial result, exit 0; `--strict` raises at the first mismatch). Several matching shift results that are
identical (same matrix, task order hash and split hashes, e.g. CLEAR-10 measured for every seed) are accepted.
Output: `hypotheses.csv` and `hypotheses.json` (`per_pairs`: rows used per benchmark).

**5. Descriptive Spearman study.** `python -m clns.analysis.correlation_study --runs-dir <runs> --shift-dir
<shift> --out corr.csv` (one-step forgetting vs shift per benchmark x arm x method; unit = transition, permutation
p-value). A run is paired with any `seed<k>` shift result whose class order, split hashes (including
`task_order_sha256`) and T match its `benchmark_meta`; the run's own seed is preferred if several match, else
the lowest seed if all matches are identical; a run with several differing matches, or none, is skipped with a
warning and counted (`n_runs_shift_mismatch`; `--strict` raises).

**6. Plots** (optional, `pip install -e ".[plots]"`): `python -m clns.analysis.plots heatmap --run-dir <run> --out
acc.png`, `python -m clns.analysis.plots scatter --runs-dir <runs> --shift-dir <shift> --benchmark clear10 --out
s.png`.

## Outputs

```
$CLNS_OUTPUT_DIR/runs/<benchmark>/<model>/<method>/seed<seed>/
  config.yaml  provenance.json  fingerprint.json  RUNNING.lock  progress.json  checkpoint.pt (deleted after DONE)
  accuracy_matrix.json  per_class_matrix.json  valid_matrix.json (eval_valid)  metrics.jsonl
  probe/meta.npz  probe/row_XXXX.npz (train.probe)
  summary.json  wandb_run_id.txt  failure.json (on error, removed on success)  DONE (success only)
```

`provenance.json` holds the benchmark meta (including `task_order` / `task_order_sha256`), trainable and total
parameter counts and, for pretrained arms, the loaded weights file and its sha256. `summary.json` keys include
`acc_final`, `acc_avg_incremental`, `bwt`, `forgetting_max`, `fwt`, `fwt_degenerate`, `peak_gpu_mem_mb` (+ `_per_exp`,
`_eval`), `train_time_s_per_exp`, `train_time_s_total`, `acc_final_macro`, `wall_clock_s_total` (plus `clear_*` for
CLEAR-10). Copy `/workspace/outputs` off the pod (or sync W&B) before terminating it. Only `/workspace` persists.
