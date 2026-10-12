"""experiment=tune: every swept hyper-parameter value gets its own run directory.

Without the hyper-parameters in the run dir, all jobs of a tuning sweep share one directory and
every job after the first is refused with exit 4 (config fingerprint mismatch).
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("hydra")

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs"


def _resolve(*overrides):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="train", overrides=list(overrides), return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    try:
        return str(cfg.run_dir), str(cfg.hydra.sweep.subdir)
    finally:
        HydraConfig.instance().cfg = None


def test_tune_run_dir_encodes_swept_hyper_parameters():
    a, sub_a = _resolve("experiment=tune", "method=ewc", "method.ewc_lambda=1", "seed=0")
    b, sub_b = _resolve("experiment=tune", "method=ewc", "method.ewc_lambda=2", "seed=0")
    assert a != b and sub_a != sub_b
    assert a.endswith("/tune/runs/clear10/r18_scratch/ewc/hp_method.ewc_lambda=1/seed0")
    assert sub_a == "clear10/r18_scratch/ewc/hp_method.ewc_lambda=1/seed0"


def test_tune_run_dir_ignores_volatile_and_path_keys():
    """A resume with another device / deterministic mode / output dir keeps the hp segment."""
    a, _ = _resolve("experiment=tune", "method=ewc", "method.ewc_lambda=1")
    b, _ = _resolve("experiment=tune", "method=ewc", "method.ewc_lambda=1", "device=cpu",
                    "deterministic=warn", "train.num_workers=0", "logger=none",
                    "train.stop_after_experiences=1", "seed=0")
    assert a == b


def test_production_run_dir_contract_unchanged():
    run_dir, sub = _resolve("benchmark=clear10", "method=ewc", "method.ewc_lambda=5", "seed=2")
    assert run_dir.endswith("/runs/clear10/r18_scratch/ewc/seed2")
    assert sub == "clear10/r18_scratch/ewc/seed2"


def test_tune_run_dir_ignores_logger_and_paths_overrides(tmp_path):
    """Review-05: logger.* / paths.* overrides (also with +/++ prefixes) keep the hp segment."""
    a, sub_a = _resolve("experiment=tune", "method=ewc", "method.ewc_lambda=1", "seed=0")
    b, sub_b = _resolve("experiment=tune", "method=ewc", "method.ewc_lambda=1", "seed=0",
                        "logger=wandb_offline", "logger.tags=[a,b]", "logger.project=p",
                        "logger.group=g", "logger.mode=disabled", "logger.entity=e",
                        f"paths.data_dir={tmp_path.as_posix()}",
                        "paths.weights_dir=/tmp/w", "++logger.name=wandb")
    assert sub_a == sub_b
    assert a.split("/runs/", 1)[1] == b.split("/runs/", 1)[1]


def test_tune_exclude_keys_cover_every_logger_and_paths_key():
    import yaml

    tune = yaml.safe_load((CONFIG_DIR / "experiment" / "tune.yaml").read_text(encoding="utf-8"))
    excl = set(tune["hydra"]["job"]["config"]["override_dirname"]["exclude_keys"])
    for group in ("logger", "paths"):
        assert group in excl
        for f in (CONFIG_DIR / group).glob("*.yaml"):
            keys = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
            missing = {f"{group}.{k}" for k in keys} - excl
            assert not missing, (f.name, missing)


@pytest.mark.slow
def test_tune_multirun_two_values_both_done(tmp_path):
    """The reviewer's collision, end to end on CPU (synthetic, smoke-sized)."""
    pytest.importorskip("avalanche")
    cmd = [sys.executable, "-m", "clns.train", "experiment=tune", "benchmark=synthetic",
           "benchmark.train_per_class=8", "benchmark.test_per_class=4", "model.cifar_stem=true",
           "train.mb_size=8", "train.eval_mb_size=16", "train.num_workers=0", "device=cpu",
           "deterministic=warn", "method=ewc", "method.ewc_lambda=1,2",
           f"paths.output_dir={tmp_path.as_posix()}", "--multirun"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    done = sorted(tmp_path.glob("runs/synthetic/r18_scratch/ewc/hp_*/seed0/DONE"))
    assert len(done) == 2 and done[0].parent != done[1].parent
