"""clns.evaluation.probe: probe set, penultimate extraction, row files/truncate, resume."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("avalanche")

from clns.data import build_benchmark  # noqa: E402
from clns.evaluation import probe as pr  # noqa: E402


def _bundle(load_benchmark_cfg, **kw):
    return build_benchmark(load_benchmark_cfg("synthetic", **kw), "unused", 0)


def _stream(groups, per=50):
    """Test stream of first-appearance-like groups; every image has distinct content."""
    from torch.utils.data import TensorDataset

    from clns.data.common import make_cls_dataset

    class _Exp:
        def __init__(self, ds):
            self.dataset = ds

    out = []
    for classes in groups:
        y = torch.tensor([c for c in classes for _ in range(per)])
        x = (y.float() * 1000 + torch.arange(len(y)) % per).view(-1, 1, 1, 1).expand(-1, 3, 2, 2)
        out.append(_Exp(make_cls_dataset(TensorDataset(x.clone(), y), y.tolist(), group="eval")))
    return out


def test_probe_set_per_class_quota(load_benchmark_cfg):
    b = _bundle(load_benchmark_cfg)  # 5 test experiences x (2 classes x 10 images)
    p = pr.build_probe_set(b.benchmark.test_stream, 23, probe_seed=7)
    assert len(p) == 23 and len(p.image_id) == 23
    per_class = np.bincount(p.labels, minlength=10)
    assert sorted(per_class.tolist()) == [2] * 7 + [3] * 3  # 23 = 10 x 2 + 3
    # labels / test_exp stored match the images actually returned
    test = b.benchmark.test_stream
    for k in (0, 7, 22):
        assert int(p.dataset[k][1]) == int(p.labels[k])
        assert int(p.labels[k]) in test[int(p.test_exp[k])].classes_in_this_experience
    # fixed by probe_seed, independent of anything else
    q = pr.build_probe_set(test, 23, probe_seed=7)
    assert np.array_equal(p.index, q.index) and np.array_equal(p.image_id, q.image_id)
    r = pr.build_probe_set(test, 23, probe_seed=8)
    assert not np.array_equal(p.image_id, r.image_id)


def test_probe_uneven_groups_balanced_per_class():
    """CORe50-like: one 10-class group + forty 1-class groups -> every class ~ its quota."""
    stream = _stream([list(range(10))] + [[c] for c in range(10, 50)])
    p = pr.build_probe_set(stream, 500, probe_seed=12345)
    counts = np.bincount(p.labels, minlength=50)
    assert len(p) == 500 and np.all(np.abs(counts - 10) <= 1)
    q = pr.build_probe_set(stream, 503, probe_seed=12345)
    assert len(q) == 503
    assert np.all(np.abs(np.bincount(q.labels, minlength=50) - 503 / 50) <= 1)


def test_probe_images_independent_of_grouping():
    """Same test images split into different groups / orders -> the same probe images."""
    a = _stream([list(range(10))] + [[c] for c in range(10, 20)], per=20)
    b = _stream([[c] for c in range(19, 9, -1)] + [list(range(5, 10)), list(range(5))], per=20)
    pa = pr.build_probe_set(a, 60, probe_seed=3)
    pb = pr.build_probe_set(b, 60, probe_seed=3)
    assert sorted(pa.image_id.tolist()) == sorted(pb.image_id.tolist())
    assert not np.array_equal(pa.test_exp, pb.test_exp)  # positions differ, images do not


def test_single_test_experience_gets_full_budget(load_benchmark_cfg):
    b = _bundle(load_benchmark_cfg, max_experiences=1)  # one test exp, 2 classes x 10
    assert len(b.benchmark.test_stream) == 1
    p = pr.build_probe_set(b.benchmark.test_stream, 12, probe_seed=0)
    assert len(p) == 12 and list(np.unique(p.labels, return_counts=True)[1]) == [6, 6]
    p_all = pr.build_probe_set(b.benchmark.test_stream, 500, probe_seed=0)
    assert len(p_all) == 20  # all available images


@pytest.mark.parametrize("arch,dim", [("resnet18", 512), ("mobilenetv2", 1280)])
def test_extract_penultimate_restores_mode(load_benchmark_cfg, arch, dim):
    pytest.importorskip("torchvision")
    from clns.models.backbones import build_backbone

    b = _bundle(load_benchmark_cfg)
    p = pr.build_probe_set(b.benchmark.test_stream, 10, probe_seed=0)
    model = build_backbone(arch, num_classes=10)
    model.train()
    # one submodule in eval mode: its flag must survive extraction too
    (model.layer1 if hasattr(model, "layer1") else model.features[0]).eval()
    before = {n: m.training for n, m in model.named_modules()}
    assert isinstance(pr.penultimate_layer(model), torch.nn.Linear)
    f = pr.extract_probe(model, p, torch.device("cpu"), batch_size=4)
    assert f.shape == (10, dim) and np.isfinite(f).all()
    assert {n: m.training for n, m in model.named_modules()} == before
    f2 = pr.extract_probe(model, p, torch.device("cpu"), batch_size=3)
    np.testing.assert_allclose(f, f2, rtol=1e-4, atol=1e-5)  # eval mode: deterministic


def test_rows_write_truncate_load(tmp_path, load_benchmark_cfg):
    b = _bundle(load_benchmark_cfg)
    p = pr.build_probe_set(b.benchmark.test_stream, 6, probe_seed=0)
    pdir = tmp_path / pr.PROBE_DIR
    for r in range(3):
        f = pr.write_probe_row(pdir, r, np.full((6, 4), r, dtype=np.float32), p)
        assert f.name == f"row_{r:04d}.npz"
    d = pr.load_probe(pdir)
    assert d["features"].shape == (3, 6, 4) and d["features"].dtype == np.float16
    assert d["exp_index"].tolist() == [0, 1, 2]
    assert int(d["probe_seed"]) == 0 and np.array_equal(d["labels"], p.labels)
    assert np.array_equal(d["image_id"], p.image_id)
    pr.truncate_probe(pdir, 2)
    d = pr.load_probe(pdir)
    assert d["features"].shape[0] == 2 and float(d["features"][1, 0, 0]) == 1.0
    assert not list(pdir.glob("*.tmp"))
    other = pr.build_probe_set(b.benchmark.test_stream, 6, probe_seed=1)
    with pytest.raises(ValueError, match="differs"):
        pr.write_probe_row(pdir, 2, np.zeros((6, 4)), other)
    # a gap (row 3 without row 2) is an error, not a warning
    pr.write_probe_row(pdir, 3, np.zeros((6, 4)), p)
    with pytest.raises(ValueError, match="not exactly"):
        pr.load_probe(pdir)
    pr.truncate_probe(pdir, 2)
    assert pr.load_probe(pdir)["features"].shape[0] == 2
    pr.truncate_probe(pdir, 0)
    assert not list(pdir.iterdir())
    with pytest.raises(FileNotFoundError):
        pr.load_probe(pdir)
    pr.truncate_probe(tmp_path / "missing", 3)  # missing directory: no-op


def test_resume_after_crash_between_probe_and_checkpoint(tmp_path, monkeypatch):
    """Crash after the probe row of experience 1 but before its checkpoint: resume -> 2 rows."""
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    import clns.train as train_mod

    out = tmp_path / "o"

    def cfg(*extra):
        with initialize_config_dir(
            config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.3"
        ):
            return compose(config_name="train", overrides=[
                "experiment=smoke", "logger=none", "device=cpu", "train.num_workers=0",
                f"paths.output_dir={out.as_posix()}", *extra,
            ])

    rd = out / "runs" / "synthetic" / "r18_scratch" / "naive" / "seed0"
    orig_save = train_mod.save_checkpoint_atomic
    n_saves = {"n": 0}

    def crash_on_second(*a, **k):
        n_saves["n"] += 1
        if n_saves["n"] == 2:
            raise RuntimeError("simulated crash before checkpoint 2")
        return orig_save(*a, **k)

    monkeypatch.setattr(train_mod, "save_checkpoint_atomic", crash_on_second)
    with pytest.raises(RuntimeError, match="simulated crash"):
        train_mod.run(cfg())
    assert pr.load_probe(rd / pr.PROBE_DIR)["features"].shape[0] == 2  # stale extra row
    monkeypatch.setattr(train_mod, "save_checkpoint_atomic", orig_save)
    assert train_mod.run(cfg()) == 0 and (rd / "DONE").exists()
    d = pr.load_probe(rd / pr.PROBE_DIR)
    assert d["features"].shape[0] == 2 and d["exp_index"].tolist() == [0, 1]


def test_probe_disabled_writes_nothing(tmp_path):
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    import clns.train as train_mod

    out = tmp_path / "o"
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "configs"), version_base="1.3"
    ):
        c = compose(config_name="train", overrides=[
            "experiment=smoke", "logger=none", "device=cpu", "train.num_workers=0",
            f"paths.output_dir={out.as_posix()}", "train.probe.enabled=false",
        ])
    assert train_mod.run(c) == 0
    rd = out / "runs" / "synthetic" / "r18_scratch" / "naive" / "seed0"
    assert (rd / "DONE").exists() and not (rd / pr.PROBE_DIR).exists()
