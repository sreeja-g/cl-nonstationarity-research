"""clns.data.confounds: CSV of benchmark properties + params from run directories."""

import csv
import json

import pytest

pytest.importorskip("omegaconf")

from clns.data import confounds  # noqa: E402


def _run(root, bench, model, method, seed, n_train=(20, 20), params=(10, 100), image_size=32):
    d = root / bench / model / method / f"seed{seed}"
    d.mkdir(parents=True)
    meta = {
        "n_train_experiences": len(n_train),
        "train_sizes": list(n_train),
        "test_sizes": [8] * len(n_train),
        "train_classes": [[0, 1], [2, 3, 4]][: len(n_train)],
    }
    prov = {"benchmark_meta": meta, "params": {"trainable": params[0], "total": params[1]}}
    (d / "provenance.json").write_text(json.dumps(prov), encoding="utf-8")
    (d / "config.yaml").write_text(
        f"benchmark:\n  name: {bench}\n  image_size: {image_size}\nmodel:\n  name: {model}\n",
        encoding="utf-8",
    )
    return d


def test_confounds_csv(tmp_path):
    runs = tmp_path / "runs"
    _run(runs, "synthetic", "r18_scratch", "naive", 0)
    _run(runs, "synthetic", "r18_scratch", "ewc", 1)
    _run(runs, "synthetic", "r18_places", "naive", 0, params=(5, 100), image_size=224)
    _run(runs, "clear10", "r18_scratch", "naive", 0, n_train=(7, 7))  # differs only in data
    (runs / "incomplete").mkdir()
    (runs / "incomplete" / "provenance.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "c" / "confounds.csv"
    confounds.main(["--runs-root", str(runs), "--out", str(out)])
    with open(out, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == list(confounds.COLUMNS)
    by = {(r["benchmark"], r["model"]): r for r in rows}
    assert set(by) == {("synthetic", "r18_scratch"), ("synthetic", "r18_places"),
                       ("clear10", "r18_scratch")}
    r = by[("synthetic", "r18_scratch")]
    assert r["n_runs"] == "2" and r["consistent"] == "True"
    assert r["n_experiences"] == "2" and r["n_train_images"] == "40"
    assert r["n_test_images"] == "16" and float(r["mean_classes_per_experience"]) == 2.5
    assert by[("synthetic", "r18_places")]["trainable_params"] == "5"
    assert by[("synthetic", "r18_places")]["image_size"] == "224"
    assert by[("clear10", "r18_scratch")]["n_train_images"] == "14"


def test_inconsistent_runs_flagged(tmp_path):
    runs = tmp_path / "runs"
    _run(runs, "synthetic", "r18_scratch", "naive", 0)
    _run(runs, "synthetic", "r18_scratch", "naive", 1, n_train=(5, 5))
    rows = confounds.collect(runs)
    assert len(rows) == 1 and rows[0]["consistent"] is False and rows[0]["n_runs"] == 2


def test_no_runs_raises(tmp_path):
    with pytest.raises(SystemExit):
        confounds.main(["--runs-root", str(tmp_path), "--out", str(tmp_path / "x.csv")])
