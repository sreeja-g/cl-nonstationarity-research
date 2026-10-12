"""End-to-end smoke tests of clns.train on the synthetic benchmark (CPU, logger=none).

Per method (naive, ewc, lwf, er, derpp):
  run 1: stop_after_experiences=1 -> no DONE, checkpoint with exp_counter 1
  run 2: resume -> DONE, 2x2 accuracy matrix, summary.json, no re-training of exp 0
  run 3: DONE present -> returns 0 without building anything (checkpoint.pt deleted after DONE)
plus: an uninterrupted run with the same seed gives the same matrix (resume == no resume);
config-fingerprint mismatch -> exit 4; crash during experience 0 -> metrics.jsonl truncated;
invalid model config -> exit 2 without a run dir; a Places365 arm (fake checkpoint) stop/resume.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("avalanche")
pytest.importorskip("hydra")

from hydra import compose, initialize_config_dir  # noqa: E402

import clns.train as train_mod  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
B2_KEYS = {
    "acc_final",
    "acc_avg_incremental",
    "bwt",
    "forgetting_max",
    "fwt",
    "fwt_degenerate",
    "peak_gpu_mem_mb",
    "peak_gpu_mem_mb_per_exp",
    "train_time_s_per_exp",
    "train_time_s_total",
}
SUMMARY_KEYS = B2_KEYS | {"acc_final_macro", "peak_gpu_mem_mb_eval", "wall_clock_s_total"}
METHOD_OVERRIDES = {
    "naive": [],
    "ewc": ["method.ewc_lambda=10"],
    "lwf": [],
    "er": ["method.mem_size=20", "method.batch_size_mem=4"],
    "derpp": ["method.mem_size=20", "method.batch_size_mem=4"],
}


def _cfg(out_dir: Path, *extra: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[
                "experiment=smoke",
                "logger=none",
                "device=cpu",
                "train.num_workers=0",
                f"paths.output_dir={out_dir.as_posix()}",
                f"paths.data_dir={(out_dir / 'data').as_posix()}",
                *extra,
            ],
        )


def _run_dir(out_dir: Path, method: str = "naive", model: str = "r18_scratch") -> Path:
    return out_dir / "runs" / "synthetic" / model / method / "seed0"


def _jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


@pytest.mark.parametrize("method", list(METHOD_OVERRIDES))
def test_run_resume_done(tmp_path, method):
    out = tmp_path / "a"
    rd = _run_dir(out, method)
    extra = [f"method={method}", *METHOD_OVERRIDES[method]]

    # --- run 1: stop after the first training experience ---
    assert train_mod.run(_cfg(out, *extra, "train.stop_after_experiences=1")) == 0
    assert not (rd / "DONE").exists()
    assert (rd / "checkpoint.pt").exists() and (rd / "fingerprint.json").exists()
    assert (rd / "config.yaml").exists() and (rd / "provenance.json").exists()
    from types import SimpleNamespace

    from clns.utils.checkpoint import load_checkpoint

    _, counter = load_checkpoint(SimpleNamespace(), rd / "checkpoint.pt", "cpu")
    assert counter == 1
    m1 = json.loads((rd / "accuracy_matrix.json").read_text())
    assert len(m1["rows"]) == 1 and m1["random_init"] is not None
    prog1 = json.loads((rd / "progress.json").read_text())
    assert prog1["n_done"] == 1 and len(prog1["train_time_s_per_exp"]) == 1
    import numpy as np

    from clns.evaluation.probe import load_probe

    z = load_probe(rd / "probe")
    assert z["features"].shape == (1, 16, 512)  # 2 test exps x 8 images, r18 penultimate
    assert z["features"].dtype == np.float16 and z["exp_index"].tolist() == [0]

    # --- run 2: resume and finish ---
    assert train_mod.run(_cfg(out, *extra)) == 0
    assert (rd / "DONE").exists() and not (rd / "failure.json").exists()
    assert not (rd / "checkpoint.pt").exists()  # deleted once DONE is written
    matrix = json.loads((rd / "accuracy_matrix.json").read_text())
    assert len(matrix["rows"]) == 2 and all(len(r) == 2 for r in matrix["rows"])
    assert matrix["test_exp_to_train_exp"] == [0, 1]
    assert matrix["scenario"] == "class_il"
    assert matrix["rows"][0] == m1["rows"][0]  # row 0 preserved across the resume
    z = load_probe(rd / "probe")
    assert z["exp_index"].tolist() == [0, 1]  # one row per completed experience
    # eval-time class mask: after exp 0, the unseen test experience 1 scores exactly 0
    assert matrix["rows"][0][1] == 0.0

    per_class = json.loads((rd / "per_class_matrix.json").read_text())
    assert len(per_class["rows"]) == 2
    assert sum(per_class["rows"][1][0]["total"]) == 8  # 2 classes x test_per_class 4

    summary = json.loads((rd / "summary.json").read_text())
    assert set(summary) == SUMMARY_KEYS
    assert summary["fwt_degenerate"] is True
    assert summary["peak_gpu_mem_mb"] is None and summary["peak_gpu_mem_mb_eval"] is None  # CPU
    assert len(summary["train_time_s_per_exp"]) == 2
    assert all(t is not None for t in summary["train_time_s_per_exp"])  # survives the resume
    assert summary["wall_clock_s_total"] >= summary["train_time_s_total"] > 0
    assert summary["acc_final_macro"] is not None
    prog2 = json.loads((rd / "progress.json").read_text())
    assert prog2["attempts"] == 2 and prog2["wall_clock_s_committed"] > 0

    # metrics.jsonl: each experience trained exactly once (epochs=1 -> one Loss_Epoch line each)
    recs = _jsonl(rd / "metrics.jsonl")
    epoch_loss = [r for r in recs if r["phase"] == "train" and r["name"].startswith("Loss_Epoch/")]
    assert sorted(r["n_trained"] for r in epoch_loss) == [0, 1]
    assert any(r["name"].startswith("Top1_ClassAcc_Stream/") for r in recs)
    prov = json.loads((rd / "provenance.json").read_text())
    assert len(prov["resumes"]) == 1
    assert prov["params"]["trainable"] == prov["params"]["total"] > 0  # r18_scratch: all
    assert prov["weights"] is None  # scratch arm

    # --- run 3: DONE -> immediate return, nothing is built ---
    def _boom(*a, **k):
        raise AssertionError("build_benchmark must not be called when DONE exists")

    orig = train_mod.build_benchmark
    train_mod.build_benchmark = _boom
    try:
        assert train_mod.run(_cfg(out, *extra)) == 0
    finally:
        train_mod.build_benchmark = orig

    if method != "naive":
        return
    # --- uninterrupted run with the same seed: identical matrix (B4) ---
    out_b = tmp_path / "b"
    assert train_mod.run(_cfg(out_b, *extra)) == 0
    matrix_b = json.loads((_run_dir(out_b, method) / "accuracy_matrix.json").read_text())
    assert matrix_b["random_init"] == pytest.approx(matrix["random_init"], abs=1e-6)
    for ra, rb in zip(matrix["rows"], matrix_b["rows"]):
        assert ra == pytest.approx(rb, abs=1e-6)


def test_fingerprint_mismatch_exit_4(tmp_path):
    out = tmp_path / "fp"
    rd = _run_dir(out)
    assert train_mod.run(_cfg(out, "train.stop_after_experiences=1")) == 0
    # volatile key changes are fine (still stopped after 1 -> exit 0, no DONE)
    assert train_mod.run(_cfg(out, "train.stop_after_experiences=1", "train.eval_mb_size=4")) == 0
    # a hyper-parameter change in the same run dir is refused, before anything is built
    assert train_mod.run(_cfg(out, "train.lr=0.5")) == train_mod.EXIT_CONFIG_MISMATCH == 4
    assert not (rd / "DONE").exists()
    # also refused once DONE exists (checked before the DONE-skip)
    assert train_mod.run(_cfg(out)) == 0 and (rd / "DONE").exists()
    assert train_mod.run(_cfg(out, "train.lr=0.5")) == 4


def test_exp0_crash_truncates_metrics(tmp_path, monkeypatch):
    out = tmp_path / "crash"
    rd = _run_dir(out)
    orig_save = train_mod.save_checkpoint_atomic

    def _crash(*a, **k):
        raise RuntimeError("simulated crash before the first checkpoint")

    monkeypatch.setattr(train_mod, "save_checkpoint_atomic", _crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        train_mod.run(_cfg(out))
    assert (rd / "failure.json").exists() and not (rd / "checkpoint.pt").exists()
    assert any(r["phase"] == "train" for r in _jsonl(rd / "metrics.jsonl"))

    monkeypatch.setattr(train_mod, "save_checkpoint_atomic", orig_save)
    assert train_mod.run(_cfg(out)) == 0
    assert (rd / "DONE").exists() and not (rd / "failure.json").exists()
    recs = _jsonl(rd / "metrics.jsonl")
    epoch_loss = [r for r in recs if r["phase"] == "train" and r["name"].startswith("Loss_Epoch/")]
    assert sorted(r["n_trained"] for r in epoch_loss) == [0, 1]  # exp 0 counted once
    init_stream = [r for r in recs if r["name"].startswith("Top1_Acc_Stream/")
                   and r["n_trained"] == 0]
    assert len(init_stream) == 1  # random-init eval logged once

    from clns.evaluation.probe import load_probe

    # the stale row of the crashed attempt was dropped
    assert load_probe(rd / "probe")["exp_index"].tolist() == [0, 1]


def test_eval_valid_writes_valid_matrix(tmp_path):
    out = tmp_path / "valid"
    rd = _run_dir(out)
    assert train_mod.run(_cfg(out, "benchmark.val_fraction=0.25", "train.eval_valid=true")) == 0
    valid = json.loads((rd / "valid_matrix.json").read_text())
    assert valid["stream"] == "valid" and len(valid["rows"]) == 2
    test = json.loads((rd / "accuracy_matrix.json").read_text())
    assert test["stream"] == "test" and len(test["rows"]) == 2


@pytest.mark.parametrize(
    "extra",
    [
        ["model=r18_scratch", "model.init=imagenet"],  # not the canonical arm spec
        ["benchmark=clear10", "model=r18_imagenet"],  # arm x benchmark not allowed
        ["model=r18_places"],  # Places365 weights file missing
    ],
)
def test_invalid_model_config_exit_2_no_run_dir(tmp_path, extra):
    out = tmp_path / "bad"
    weights = tmp_path / "no_weights"
    cfg = _cfg(out, *extra, f"paths.weights_dir={weights.as_posix()}")
    assert train_mod.run(cfg) == train_mod.EXIT_INVALID_CONFIG == 2
    assert not out.exists() and not weights.exists()  # no run dir, provenance, lock


def test_pretrained_arm_smoke_resume(tmp_path, fake_places):
    """experiment=smoke with a Places365 arm: cifar_stem reset, weights hash pinned and recorded,
    FrozenBatchNorm2d survives stop/resume."""
    out = tmp_path / "places"
    wdir = tmp_path / "weights"
    wdir.mkdir()
    sha = fake_places(wdir / "resnet18_places365.pth.tar")
    extra = ["model=r18_places", f"paths.weights_dir={wdir.as_posix()}",
             f"model.weights_sha256={sha}"]
    rd = _run_dir(out, model="r18_places")
    assert train_mod.run(_cfg(out, *extra, "train.stop_after_experiences=1")) == 0
    assert (rd / "checkpoint.pt").exists() and not (rd / "DONE").exists()
    assert train_mod.run(_cfg(out, *extra)) == 0
    assert (rd / "DONE").exists() and not (rd / "failure.json").exists()
    matrix = json.loads((rd / "accuracy_matrix.json").read_text())
    assert len(matrix["rows"]) == 2
    prov = json.loads((rd / "provenance.json").read_text())
    assert 0 < prov["params"]["trainable"] < prov["params"]["total"]
    assert prov["weights"]["sha256"] == sha
    assert prov["weights"]["file"].endswith("resnet18_places365.pth.tar")
    assert len(prov["resumes"]) == 1
    from omegaconf import OmegaConf

    saved = OmegaConf.load(rd / "config.yaml")
    assert saved["model"]["cifar_stem"] is False  # smoke's cifar_stem=true ignored
    assert saved["benchmark"]["image_size"] == 32  # benchmark resolution kept


def test_seed_follows_task_order(tmp_path):
    """seed=1 trains on its own synthetic class order (recorded in provenance benchmark_meta)."""
    from clns.data.synthetic import synthetic_class_order

    out = tmp_path / "o"
    assert train_mod.run(_cfg(out, "seed=1", "train.stop_after_experiences=1")) == 0
    rd = out / "runs" / "synthetic" / "r18_scratch" / "naive" / "seed1"
    meta = json.loads((rd / "provenance.json").read_text(encoding="utf-8"))["benchmark_meta"]
    order1 = synthetic_class_order(1993, 1)
    assert order1 != synthetic_class_order(1993, 0)
    assert meta["task_order"] == order1 and meta["seed"] == 1
    assert meta["train_classes"] == [sorted(order1[:2]), sorted(order1[2:4])]
