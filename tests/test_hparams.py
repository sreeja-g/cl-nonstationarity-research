"""Tuned-values files (clns.utils.hparams) and their use by clns.train / the analysis."""

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("avalanche")
pytest.importorskip("hydra")

from hydra import compose, initialize_config_dir  # noqa: E402

import clns.train as train_mod  # noqa: E402
from clns.utils.fingerprint import config_fingerprint  # noqa: E402
from clns.utils.hparams import (  # noqa: E402
    HparamsError,
    apply_tuned_hparams,
    default_hparams_dir,
    file_sha256,
    tuned_transitions_from_dir,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _cfg(out: Path, *extra: str):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="train", overrides=[
            "experiment=smoke", "logger=none", "device=cpu", "train.num_workers=0",
            f"paths.output_dir={out.as_posix()}", f"paths.data_dir={(out / 'data').as_posix()}",
            *extra])


def _write_hp(directory: Path, lr: float = 0.123, method: str = "naive") -> Path:
    path = directory / "synthetic" / "r18_scratch.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "tuned_transitions": [0, 1],
        "methods": {method: {"values": {"train.lr": lr}, "provenance": {"n_trials": 2}}}}),
        encoding="utf-8")
    return path


def _rd(out: Path) -> Path:
    return out / "runs" / "synthetic" / "r18_scratch" / "naive" / "seed0"


def test_default_dir_and_repo_has_hparams_dir():
    assert default_hparams_dir() == CONFIG_DIR / "hparams"


def test_apply_merges_values_and_cli_wins(tmp_path):
    hp = tmp_path / "hp"
    path = _write_hp(hp)
    cfg = _cfg(tmp_path / "o", "hparams.mode=required", f"hparams.dir={hp.as_posix()}")
    info = apply_tuned_hparams(cfg, [])
    assert cfg.train.lr == 0.123 and dict(cfg.hparams.applied) == {"train.lr": 0.123}
    assert info["status"] == "applied" and info["sha256"] == file_sha256(path)
    assert info["path"] == str(path) and info["cli_wins"] == {}

    cfg = _cfg(tmp_path / "o", "hparams.mode=required", f"hparams.dir={hp.as_posix()}",
               "train.lr=0.5")
    info = apply_tuned_hparams(cfg, ["+train.lr=0.5", "seed=0"])
    assert cfg.train.lr == 0.5 and dict(cfg.hparams.applied) == {}
    assert set(info["cli_wins"]) == {"train.lr"} and info["status"] == "applied"


def test_apply_errors_and_modes(tmp_path):
    hp = tmp_path / "hp"
    with pytest.raises(HparamsError, match="not found"):
        apply_tuned_hparams(_cfg(tmp_path, "hparams.mode=required", f"hparams.dir={hp.as_posix()}"))
    _write_hp(hp, method="ewc")  # file exists, no naive entry
    with pytest.raises(HparamsError, match="no tuned values for method 'naive'"):
        apply_tuned_hparams(_cfg(tmp_path, "hparams.mode=required", f"hparams.dir={hp.as_posix()}"))
    cfg = _cfg(tmp_path, "hparams.mode=untuned", f"hparams.dir={hp.as_posix()}")
    assert apply_tuned_hparams(cfg)["status"] == "untuned" and cfg.train.lr == 0.01
    assert apply_tuned_hparams(_cfg(tmp_path))["status"] == "off"  # smoke: mode off
    bad = hp / "synthetic" / "r18_scratch.yaml"
    bad.write_text(yaml.safe_dump({"methods": {"naive": {"values": {"seed": 1}}}}))
    with pytest.raises(HparamsError, match="outside"):
        apply_tuned_hparams(_cfg(tmp_path, "hparams.mode=required", f"hparams.dir={hp.as_posix()}"))
    bad.write_text(yaml.safe_dump({"methods": {"naive": {"values": {"train.nope": 1}}}}))
    with pytest.raises(HparamsError, match="does not exist"):
        apply_tuned_hparams(_cfg(tmp_path, "hparams.mode=required", f"hparams.dir={hp.as_posix()}"))


def test_fingerprint_ignores_dir_and_mode_but_not_applied_values(tmp_path):
    hp1, hp2 = tmp_path / "hp1", tmp_path / "hp2"
    _write_hp(hp1)
    _write_hp(hp2)
    _write_hp(tmp_path / "hp3", lr=0.2)

    def fp(d):
        c = _cfg(tmp_path, "hparams.mode=required", f"hparams.dir={d.as_posix()}")
        apply_tuned_hparams(c)
        return config_fingerprint(c)[0]

    assert fp(hp1) == fp(hp2) != fp(tmp_path / "hp3")


def test_production_run_applies_file_and_records_provenance(tmp_path):
    out, hp = tmp_path / "o", tmp_path / "hp"
    path = _write_hp(hp)
    cfg = _cfg(out, "hparams.mode=required", f"hparams.dir={hp.as_posix()}")
    assert train_mod.run(cfg, []) == 0
    rd = _rd(out)
    prov = json.loads((rd / "provenance.json").read_text())
    assert prov["hparams"]["status"] == "applied" and prov["hparams"]["sha256"] == file_sha256(path)
    assert prov["hparams"]["applied"] == {"train.lr": 0.123}
    assert yaml.safe_load((rd / "config.yaml").read_text())["train"]["lr"] == 0.123

    # command line wins (separate output dir): recorded in cli_wins, lr 0.5 is what runs
    out2 = tmp_path / "o2"
    cfg = _cfg(out2, "hparams.mode=required", f"hparams.dir={hp.as_posix()}", "train.lr=0.5")
    assert train_mod.run(cfg, ["train.lr=0.5"]) == 0
    prov = json.loads((_rd(out2) / "provenance.json").read_text())
    assert prov["hparams"]["applied"] == {} and "train.lr" in prov["hparams"]["cli_wins"]
    assert yaml.safe_load((_rd(out2) / "config.yaml").read_text())["train"]["lr"] == 0.5


def test_missing_file_exits_2_without_run_dir_and_untuned_exits_0(tmp_path):
    out, hp = tmp_path / "o", tmp_path / "empty"
    cfg = _cfg(out, "hparams.mode=required", f"hparams.dir={hp.as_posix()}")
    assert train_mod.run(cfg, []) == train_mod.EXIT_INVALID_CONFIG == 2
    assert not _rd(out).exists()
    cfg = _cfg(out, "hparams.mode=untuned", f"hparams.dir={hp.as_posix()}")
    assert train_mod.run(cfg, []) == 0
    prov = json.loads((_rd(out) / "provenance.json").read_text())
    assert prov["hparams"]["status"] == "untuned" and prov["hparams"]["applied"] == {}


def _hp_file(hp: Path, bench: str, ks, seed=None, sha=None):
    f = hp / bench / "r18_scratch.yaml"
    f.parent.mkdir(parents=True, exist_ok=True)
    blob = {"tuned_transitions": ks, "methods": {}}
    if seed is not None:
        blob["tuning_seed"] = seed
    if sha is not None:
        blob["task_order_sha256"] = sha
    f.write_text(yaml.safe_dump(blob))


def test_tuned_transitions_from_dir_and_hypotheses_cli(tmp_path):
    import pandas as pd

    from clns.analysis.hypotheses import main as hyp_main
    from test_hypotheses import _build_world, _slopes

    assert tuned_transitions_from_dir(tmp_path / "none") == set()
    hp = tmp_path / "hp"
    _hp_file(hp, "bm", [0], seed=0, sha="order-sha")
    _hp_file(hp, "other", [0, 1])
    assert tuned_transitions_from_dir(hp) == {
        ("bm", 0, 0, "order-sha"), ("other", 0, None, None), ("other", 1, None, None)}

    arms = ["r18_scratch", "r18_places"]
    runs, shifts = _build_world(tmp_path, _slopes(arms, reg=0.10, rep=0.02, pre=0.5), arms)

    def n_tuned(tag, *args):
        out = tmp_path / f"res_{tag}"
        assert hyp_main(["--runs-dir", str(runs), "--shift-dir", str(shifts), "--out", str(out),
                         "--n-boot", "20", *args]) == 0
        return int(pd.read_csv(out / "hypotheses.csv")["n_tuned"].fillna(0).sum())

    none = n_tuned("none", "--hparams-dir", str(tmp_path / "none"))
    same_order = n_tuned("dir", "--hparams-dir", str(hp))   # hash equal -> every seed (0, 1, 2)
    cli = n_tuned("cli", "--hparams-dir", str(tmp_path / "none"), "--tuned-transitions", "bm:0")
    assert none == 0 and same_order > 0 and cli == same_order

    other_order = tmp_path / "hp_other"                      # another order hash: not flagged
    _hp_file(other_order, "bm", [0], seed=0, sha="another-sha")
    assert n_tuned("other", "--hparams-dir", str(other_order)) == 0
    seed_only = tmp_path / "hp_seed"                         # no hash known: tuning seed only
    _hp_file(seed_only, "bm", [0], seed=0)
    n_seed = n_tuned("seed", "--hparams-dir", str(seed_only))
    assert 0 < n_seed < same_order
