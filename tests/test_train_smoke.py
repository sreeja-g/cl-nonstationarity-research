"""End-to-end smoke tests of clns.train on the synthetic benchmark (CPU, logger=none).

Per method (naive, ewc, lwf, er):
  run 1: stop_after_experiences=1 -> no DONE, checkpoint with exp_counter 1
  run 2: resume -> DONE, 2x2 accuracy matrix, summary.json, no re-training of exp 0
  run 3: DONE present -> returns 0 without building anything
plus: an uninterrupted run with the same seed gives the same matrix (resume == no resume);
config-fingerprint mismatch -> exit 4; crash during experience 0 -> metrics.jsonl truncated;
L2P (pretrained=false, slow) trains one iteration and evaluates.
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
    "er": ["method.mem_size=20"],
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


def _run_dir(out_dir: Path, method: str = "naive", model: str = "resnet18") -> Path:
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

    # --- run 2: resume and finish ---
    assert train_mod.run(_cfg(out, *extra)) == 0
    assert (rd / "DONE").exists() and not (rd / "failure.json").exists()
    matrix = json.loads((rd / "accuracy_matrix.json").read_text())
    assert len(matrix["rows"]) == 2 and all(len(r) == 2 for r in matrix["rows"])
    assert matrix["test_exp_to_train_exp"] == [0, 1]
    assert matrix["scenario"] == "class_il"
    assert matrix["rows"][0] == m1["rows"][0]  # row 0 preserved across the resume
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


def test_eval_valid_writes_valid_matrix(tmp_path):
    out = tmp_path / "valid"
    rd = _run_dir(out)
    assert train_mod.run(_cfg(out, "benchmark.val_fraction=0.25", "train.eval_valid=true")) == 0
    valid = json.loads((rd / "valid_matrix.json").read_text())
    assert valid["stream"] == "valid" and len(valid["rows"]) == 2
    test = json.loads((rd / "accuracy_matrix.json").read_text())
    assert test["stream"] == "test" and len(test["rows"]) == 2


@pytest.mark.slow
def test_l2p_smoke(tmp_path):
    """L2P (ViT-B/16, random init, CPU): random-init eval, 1 training iteration, eval."""
    pytest.importorskip("timm")
    out = tmp_path / "l2p"
    cfg = _cfg(
        out,
        "method=l2p",
        "model=vit_b16_l2p",
        "model.pretrained=false",
        "method.pool_size=4",
        "method.prompt_length=2",
        "method.top_k=2",
        "benchmark.train_per_class=4",  # 2 classes x 4 = 8 = one minibatch
        "benchmark.test_per_class=2",
        "train.stop_after_experiences=1",
    )
    assert train_mod.run(cfg) == 0
    rd = _run_dir(out, "l2p", "vit_b16_l2p")
    m = json.loads((rd / "accuracy_matrix.json").read_text())
    assert len(m["rows"]) == 1 and m["random_init"] is not None
    resolved = (rd / "config.yaml").read_text()
    assert "image_size: 224" in resolved and "normalization: vit" in resolved
    recs = _jsonl(rd / "metrics.jsonl")
    losses = [r["value"] for r in recs if r["name"].startswith("Loss_Epoch/")]
    assert losses and all(v == v and abs(v) < 1e6 for v in losses)  # finite (mask not all -inf)
