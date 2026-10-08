"""Committed split files, leakage guards and the CORe50 first-appearance partition."""

import re

import pytest

from clns.data.common import (
    assert_disjoint,
    check_manifest,
    read_list_file,
    sha256_file,
)
from clns.data.core50 import partition_test_by_first_appearance


def test_imagenet100_classes(repo_root, load_benchmark_cfg):
    path = repo_root / "data_splits" / "imagenet100" / "classes.txt"
    wnids = read_list_file(path)
    assert len(wnids) == 100 and len(set(wnids)) == 100
    assert wnids == sorted(wnids)
    assert all(re.fullmatch(r"n\d{8}", w) for w in wnids)
    cfg = load_benchmark_cfg("imagenet100")
    assert sha256_file(path) == cfg.classes_sha256


def test_domainnet_order(repo_root, load_benchmark_cfg):
    path = repo_root / "data_splits" / "domainnet" / "domain_order.txt"
    domains = read_list_file(path)
    assert domains == ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    cfg = load_benchmark_cfg("domainnet")
    assert len(domains) == cfg.n_experiences
    assert sha256_file(path) == cfg.domain_order_sha256


def test_check_manifest_mismatch(repo_root):
    path = repo_root / "data_splits" / "domainnet" / "domain_order.txt"
    assert check_manifest(path, None) == sha256_file(path)
    with pytest.raises(ValueError):
        check_manifest(path, "0" * 64)


def test_assert_disjoint():
    assert_disjoint(["a", "b"], ["c"])
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint(["a", "b"], ["b", "c"], "unit")


def test_clear10_streaming_rejected(load_benchmark_cfg):
    from clns.data import build_benchmark

    cfg = load_benchmark_cfg("clear10", protocol="streaming")
    with pytest.raises(ValueError, match="streaming"):
        build_benchmark(cfg, "/nonexistent", 0)


def test_missing_data_root_raises(load_benchmark_cfg, tmp_path):
    from clns.data import build_benchmark

    with pytest.raises(FileNotFoundError):
        build_benchmark(load_benchmark_cfg("clear10"), str(tmp_path), 0)
    with pytest.raises(FileNotFoundError):
        build_benchmark(load_benchmark_cfg("core50_nic"), str(tmp_path), 0)


def test_core50_first_appearance_partition():
    train = [[0, 1], [1, 2], [2], [3, 4]]
    test = {0: [0, 1], 1: [2], 2: [3, 4], 3: [5], 4: [6]}
    exp_map, groups, idx = partition_test_by_first_appearance(train, test)
    assert exp_map == [0, 1, 3]  # exp 2 introduces no new class -> dropped
    assert groups == [[0, 1], [2], [3, 4]]
    assert idx == [[0, 1, 2], [3, 4], [5, 6]]
    # every test index appears exactly once
    flat = sorted(i for g in idx for i in g)
    assert flat == sorted(i for v in test.values() for i in v)


def test_core50_partition_truncation_and_unseen():
    train = [[0, 1], [2], [3]]
    test = {0: [0], 1: [1], 2: [2], 3: [3]}
    exp_map, groups, idx = partition_test_by_first_appearance(train, test, max_experiences=2)
    assert exp_map == [0, 1] and groups == [[0, 1], [2]] and idx == [[0, 1], [2]]
    with pytest.raises(ValueError, match="never seen"):
        partition_test_by_first_appearance([[0]], {0: [0], 9: [1]})


def test_assert_disjoint_basename_key():
    from clns.data.common import basename_key

    train = ["train/n01/a.JPEG", "train/n02/b.JPEG"]
    assert_disjoint(train, ["val/n01/c.JPEG"], key=basename_key)
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint(train, ["val/n01/a.JPEG"], "unit", key=basename_key)
    assert basename_key(r"x\y\z.png") == "z.png"


def test_require_data_root_done_marker(tmp_path):
    from omegaconf import OmegaConf

    from clns.data.common import DONE_MARKER, require_data_root

    root = tmp_path / "bench"
    root.mkdir()
    (root / "some_file").write_text("x")
    cfg = OmegaConf.create({})
    with pytest.raises(FileNotFoundError, match=r"download_data\.sh mybench"):
        require_data_root(root, cfg, "Bench", "mybench")
    # opt-outs
    require_data_root(root, OmegaConf.create({"require_done_marker": False}), "Bench", "mybench")
    require_data_root(tmp_path / "missing", OmegaConf.create({"allow_download": True}), "B", "b")
    (root / DONE_MARKER).write_text("")
    assert require_data_root(root, cfg, "Bench", "mybench") == root
    with pytest.raises(FileNotFoundError):
        require_data_root(tmp_path / "missing", cfg, "Bench", "mybench")


def test_core50_val_fraction_rejected(load_benchmark_cfg, tmp_path):
    from clns.data import build_benchmark

    with pytest.raises(ValueError, match="session hold-out"):
        build_benchmark(load_benchmark_cfg("core50_nic", val_fraction=0.1), str(tmp_path), 0)


def test_imagenet100_class_order_fixed():
    from clns.data.imagenet100 import imagenet100_class_order

    a = imagenet100_class_order(1993)
    assert sorted(a) == list(range(100))
    assert a == imagenet100_class_order(1993)
    assert a != imagenet100_class_order(0)
