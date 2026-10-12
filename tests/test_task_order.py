"""Task order follows the seed (ImageNet-100, DomainNet, CORe50 run); CLEAR-10 stays fixed."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from clns.data.common import read_list_file, record_task_order, splits_dir


def test_imagenet100_order_by_seed():
    from clns.data.imagenet100 import imagenet100_class_order

    s0, s1, s2 = (imagenet100_class_order(1993, s) for s in (0, 1, 2))
    # seed 0 = committed UCIR/PODNet order
    assert s0 == [int(c) for c in np.random.RandomState(1993).permutation(100)]
    assert s1 == [int(c) for c in np.random.default_rng([1993, 1]).permutation(100)]
    assert len({tuple(s0), tuple(s1), tuple(s2)}) == 3
    assert all(sorted(o) == list(range(100)) for o in (s0, s1, s2))
    assert s1 == imagenet100_class_order(1993, 1)
    with pytest.raises(ValueError):
        imagenet100_class_order(1993, -1)


def test_imagenet100_default_is_20_tasks(load_benchmark_cfg):
    assert int(load_benchmark_cfg("imagenet100").n_experiences) == 20


def test_domainnet_order_by_seed():
    from clns.data.domainnet import domainnet_domain_order

    committed = read_list_file(splits_dir() / "domainnet" / "domain_order.txt")
    assert domainnet_domain_order(committed, 0) == committed
    orders = [domainnet_domain_order(committed, s) for s in (1, 2)]
    assert all(sorted(o) == sorted(committed) for o in orders)
    assert committed not in orders and orders[0] != orders[1]
    assert domainnet_domain_order(committed, 1, order_seed=5) != orders[0]


def test_core50_run_follows_seed():
    from clns.data.core50 import resolve_core50_run

    cfg = OmegaConf.create({"run": None})
    assert [resolve_core50_run(cfg, s) for s in (0, 1, 2)] == [0, 1, 2]
    assert resolve_core50_run(OmegaConf.create({"run": 4}), 2) == 4
    with pytest.raises(ValueError, match="0..9"):
        resolve_core50_run(cfg, 10)


def test_core50_config_run_null(load_benchmark_cfg):
    assert load_benchmark_cfg("core50_nic").run is None


def test_record_task_order_hash():
    a, b = {}, {}
    record_task_order(a, [3, 1, 2])
    record_task_order(b, [1, 3, 2])
    assert a["task_order"] == [3, 1, 2] and len(a["task_order_sha256"]) == 64
    assert a["task_order_sha256"] != b["task_order_sha256"]
    c = {}
    record_task_order(c, ((1, 2), (3,)))
    assert c["task_order"] == [[1, 2], [3]]


def test_clear10_order_fixed_across_seeds(monkeypatch, load_benchmark_cfg, tmp_path):
    """CLEAR's buckets keep their time order for every seed (stubbed Avalanche CLEAR)."""
    pytest.importorskip("avalanche")
    import avalanche.benchmarks.classic.clear as clear_mod
    import torch
    from torch.utils.data import TensorDataset

    from clns.data import clear10
    from clns.data.common import make_cls_dataset

    class _Exp:
        def __init__(self, ds):
            self.dataset = ds

    def fake_clear(**kw):
        def stream(off):
            out = []
            for b in range(10):
                x = torch.full((2, 3, 4, 4), float(b + off))
                out.append(_Exp(make_cls_dataset(TensorDataset(x, torch.tensor([0, 1])), [0, 1])))
            return out

        bm = type("BM", (), {})()
        bm.train_stream, bm.test_stream = stream(0), stream(100)
        return bm

    monkeypatch.setattr(clear_mod, "CLEAR", fake_clear)
    monkeypatch.setattr(clear10, "_check_clear_disjoint", lambda *a, **k: None)
    root = tmp_path / "clear10"
    root.mkdir()
    (root / ".done").write_text("")
    cfg = load_benchmark_cfg("clear10", train_transform="none", eval_transform="none")
    metas = [clear10.build_clear10(cfg, str(tmp_path), s).meta for s in (0, 1, 2)]
    assert metas[0]["task_order"] == list(range(10))
    assert len({m["task_order_sha256"] for m in metas}) == 1


def _fake_core50(monkeypatch, tmp_path):
    """Stub Avalanche CORe50 (3 batches with the SAME class sets in every run) + real filelists."""
    pytest.importorskip("avalanche")
    import avalanche.benchmarks.classic.core50 as core50_mod
    import torch
    from torch.utils.data import TensorDataset

    from clns.data.common import make_cls_dataset

    class _Exp:
        def __init__(self, ds):
            self.dataset = ds

    def ds(labels, off):
        x = torch.arange(len(labels), dtype=torch.float32).view(-1, 1, 1, 1) + off
        return make_cls_dataset(TensorDataset(x.expand(-1, 3, 2, 2).clone(),
                                              torch.tensor(labels)), labels)

    batches = [[0, 0, 1, 1], [2, 2], [1, 3, 3]]

    def fake_core50(**kw):
        bm = type("BM", (), {})()
        bm.train_stream = [_Exp(ds(b, 10 * k)) for k, b in enumerate(batches)]
        bm.test_stream = [_Exp(ds([0, 0, 1, 1, 2, 2, 3, 3], 100))]
        return bm

    monkeypatch.setattr(core50_mod, "CORe50", fake_core50)
    monkeypatch.setitem(core50_mod.nbatch, "nicv2_79", len(batches))
    root = tmp_path / "core50"
    for run in range(3):
        d = root / "NIC_v2_79" / f"run{run}"
        d.mkdir(parents=True)
        for k, b in enumerate(batches):
            (d / f"train_batch_{k:02d}_filelist.txt").write_text(
                "".join(f"s{run}/tr_{k}_{i}.png {y}\n" for i, y in enumerate(b)))
        (d / "test_filelist.txt").write_text(
            "".join(f"s3/te_{run}_{i}.png {i // 2}\n" for i in range(8)))
    (root / ".done").write_text("")


def test_core50_order_hash_includes_run_and_test_filelist(monkeypatch, tmp_path,
                                                          load_benchmark_cfg):
    from clns.data.core50 import build_core50
    from clns.shift.measure_stream import split_hashes

    _fake_core50(monkeypatch, tmp_path)
    cfg = load_benchmark_cfg("core50_nic", train_transform="none", eval_transform="none")
    m0, m1 = (build_core50(cfg, str(tmp_path), s).meta for s in (0, 1))
    assert m0["run"] == 0 and m1["run"] == 1
    assert m0["task_order"] == [0, m0["train_stream_classes"]]
    # identical batch class sets, different runs -> different task-order hashes
    assert m0["train_stream_classes"] == m1["train_stream_classes"]
    assert m0["task_order_sha256"] != m1["task_order_sha256"]
    assert len(m0["test_filelist_sha256"]) == 64
    assert m0["test_filelist_sha256"] != m1["test_filelist_sha256"]
    assert split_hashes(m0)["test_filelist_sha256"] == m0["test_filelist_sha256"]
    assert m0["test_exp_to_train_exp"] == [0, 1, 2]


def test_shift_mismatch_on_domain_il_task_order(tmp_path, load_benchmark_cfg):
    """A DomainNet shift result measured on another seed's domain order never pairs with a run."""
    pytest.importorskip("avalanche")
    from clns.analysis.correlation_study import shift_mismatch
    from clns.data import build_benchmark
    from clns.shift.measure_stream import split_hashes
    from test_loaders_fake import _fake_domainnet

    _fake_domainnet(tmp_path / "domainnet")
    cfg = load_benchmark_cfg("domainnet", num_classes=2, image_size=16)
    m0, m1 = (build_benchmark(cfg, str(tmp_path), s).meta for s in (0, 1))

    def shift(m):
        meta = {"class_order": None, "split_hashes": split_hashes(m), "T": 6,
                "task_order_sha256": m["task_order_sha256"]}
        return {"meta": meta, "matrix": np.zeros((6, 6))}

    run = {"rows": np.zeros((6, 6)), "benchmark_meta": m0}
    assert shift_mismatch(shift(m0), run) is None
    assert "task_order_sha256" in shift_mismatch(shift(m1), run)
