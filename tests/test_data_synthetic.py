"""Synthetic benchmark + shared finalize() rules (needs torch + avalanche-lib)."""

import pytest

from clns.data import BenchmarkBundle, build_benchmark


def _stream(bm, name):
    return [e.dataset for e in getattr(bm, name)]


def test_basic_structure(load_benchmark_cfg):
    bundle = build_benchmark(load_benchmark_cfg("synthetic"), data_dir="unused", seed=0)
    assert isinstance(bundle, BenchmarkBundle)
    assert bundle.scenario == "class_il" and bundle.num_classes == 10
    assert sorted(bundle.class_order) == list(range(10))
    assert len(_stream(bundle.benchmark, "train_stream")) == 5
    assert len(_stream(bundle.benchmark, "test_stream")) == 5
    assert bundle.meta["test_exp_to_train_exp"] == [0, 1, 2, 3, 4]
    assert bundle.meta["n_train_experiences"] == 5
    assert bundle.meta["train_sizes"] == [40] * 5  # 2 classes x 20


def test_task_labels_all_zero(load_benchmark_cfg):
    bm = build_benchmark(load_benchmark_cfg("synthetic"), "unused", 0).benchmark
    for name in ("train_stream", "test_stream"):
        for ds in _stream(bm, name):
            assert set(ds.targets_task_labels.uniques) == {0}


def test_class_incremental_and_order(load_benchmark_cfg):
    bundle = build_benchmark(load_benchmark_cfg("synthetic"), "unused", 3)
    for k, ds in enumerate(_stream(bundle.benchmark, "train_stream")):
        assert set(ds.targets.uniques) == set(bundle.class_order[2 * k : 2 * k + 2])
    for k, ds in enumerate(_stream(bundle.benchmark, "test_stream")):
        assert set(ds.targets.uniques) == set(bundle.class_order[2 * k : 2 * k + 2])


def test_deterministic_and_seed_dependent(load_benchmark_cfg):
    cfg = load_benchmark_cfg("synthetic")
    a = build_benchmark(cfg, "unused", 1)
    b = build_benchmark(cfg, "unused", 1)
    c = build_benchmark(cfg, "unused", 2)
    assert a.class_order == b.class_order
    xa = a.benchmark.train_stream[0].dataset[0][0]
    xb = b.benchmark.train_stream[0].dataset[0][0]
    xc = c.benchmark.train_stream[0].dataset[0][0]
    assert bool((xa == xb).all())
    assert not bool((xa == xc).all())  # data depends on the training seed ...
    assert a.class_order != c.class_order  # ... and so does the class order (like ImageNet-100)
    from clns.data.synthetic import synthetic_class_order

    assert a.class_order == synthetic_class_order(1993, 1)
    seed0 = build_benchmark(cfg, "unused", 0).class_order
    import numpy as np

    assert seed0 == [int(k) for k in np.random.RandomState(1993).permutation(10)]  # unchanged


def test_class_order_seed(load_benchmark_cfg):
    a = build_benchmark(load_benchmark_cfg("synthetic"), "unused", 0)
    b = build_benchmark(load_benchmark_cfg("synthetic", class_order_seed=7), "unused", 0)
    assert a.meta["class_order_seed"] == 1993
    assert a.class_order != b.class_order


def test_classes_timeline_on_every_stream(load_benchmark_cfg):
    bundle = build_benchmark(load_benchmark_cfg("synthetic", val_fraction=0.25), "unused", 0)
    bm = bundle.benchmark
    for name in ("train_stream", "valid_stream", "test_stream"):
        for k, exp in enumerate(getattr(bm, name)):
            cls = exp.classes_in_this_experience
            assert isinstance(cls, list)  # sorted list, not set (deterministic)
            assert cls == sorted(bundle.class_order[2 * k : 2 * k + 2])
    assert bundle.meta["train_classes"][0] == sorted(bundle.class_order[:2])


def test_image_size(load_benchmark_cfg):
    bundle = build_benchmark(load_benchmark_cfg("synthetic", image_size=48), "unused", 0)
    assert tuple(bundle.benchmark.train_stream[0].dataset[0][0].shape) == (3, 48, 48)


def test_max_experiences_truncates_train_and_test(load_benchmark_cfg):
    bundle = build_benchmark(load_benchmark_cfg("synthetic", max_experiences=2), "unused", 0)
    assert len(_stream(bundle.benchmark, "train_stream")) == 2
    assert len(_stream(bundle.benchmark, "test_stream")) == 2
    assert bundle.num_classes == 10  # output head size unchanged
    assert bundle.meta["test_exp_to_train_exp"] == [0, 1]


def test_subsample_per_class(load_benchmark_cfg):
    bundle = build_benchmark(load_benchmark_cfg("synthetic", subsample_per_class=3), "unused", 0)
    assert bundle.meta["train_sizes"] == [6] * 5
    assert bundle.meta["test_sizes"] == [6] * 5


def test_test_subsample_independent_of_training_seed(load_benchmark_cfg):
    """finalize(): test subsets come from benchmark.subsample_seed, not the training seed."""
    import torch
    from torch.utils.data import TensorDataset

    from clns.data.common import finalize, make_cls_dataset

    x = torch.arange(40, dtype=torch.float32).view(40, 1)
    y = torch.tensor([0] * 20 + [1] * 20)
    cfg = load_benchmark_cfg("synthetic", subsample_per_class=3)

    def test_values(seed, cfg=cfg):
        tr = make_cls_dataset(TensorDataset(x, y), y.tolist())
        te = make_cls_dataset(TensorDataset(x, y), y.tolist())
        bm = finalize([tr], [te], cfg, seed, {})
        ds = bm.test_stream[0].dataset
        return sorted(float(ds[i][0]) for i in range(len(ds)))

    assert test_values(0) == test_values(5)
    other = load_benchmark_cfg("synthetic", subsample_per_class=3, subsample_seed=1)
    assert test_values(0, other) != test_values(0)


def test_validation_carved_from_train_only(load_benchmark_cfg):
    cfg = load_benchmark_cfg("synthetic", val_fraction=0.25)
    bundle = build_benchmark(cfg, "unused", 0)
    assert hasattr(bundle.benchmark, "valid_stream")
    assert bundle.meta["train_sizes"] == [30] * 5  # 20 -> 15 per class
    assert bundle.meta["valid_sizes"] == [10] * 5
    assert bundle.meta["test_sizes"] == [20] * 5  # test untouched


def test_no_valid_stream_by_default(load_benchmark_cfg):
    bundle = build_benchmark(load_benchmark_cfg("synthetic"), "unused", 0)
    assert not hasattr(bundle.benchmark, "valid_stream")


def test_bad_n_experiences(load_benchmark_cfg):
    with pytest.raises(ValueError):
        build_benchmark(load_benchmark_cfg("synthetic", n_experiences=3), "unused", 0)


def test_unknown_benchmark(load_benchmark_cfg):
    with pytest.raises(ValueError):
        build_benchmark(load_benchmark_cfg("synthetic", name="nope"), "unused", 0)


def test_train_classes_meta_after_truncation(load_benchmark_cfg):
    bundle = build_benchmark(
        load_benchmark_cfg("synthetic", max_experiences=2, subsample_per_class=3), "unused", 0
    )
    tc = bundle.meta["train_classes"]
    assert len(tc) == 2 == bundle.meta["n_train_experiences"]
    for k, exp in enumerate(bundle.benchmark.train_stream):
        assert tc[k] == sorted(int(c) for c in exp.dataset.targets.uniques)


def test_build_evaluator_with_synthetic_bundle(load_benchmark_cfg, tmp_path):
    """train.py passes bundle.meta["train_classes"] to build_evaluator (class-IL eval mask)."""
    pytest.importorskip("avalanche")
    from omegaconf import OmegaConf

    from clns.evaluation.evaluator import build_evaluator
    from clns.evaluation.plugins import EvalClassMaskPlugin

    bundle = build_benchmark(load_benchmark_cfg("synthetic"), "unused", 0)
    cfg = OmegaConf.create({"train": {"eval_mask_unseen": True, "eval_every": 0}})
    n_test = len(bundle.benchmark.test_stream)
    _, plugins = build_evaluator(
        cfg, [], tmp_path, scenario=bundle.scenario,
        test_exp_to_train_exp=bundle.meta["test_exp_to_train_exp"], n_test=n_test,
        train_classes=bundle.meta.get("train_classes"), num_classes=bundle.num_classes,
    )
    assert isinstance(plugins[0], EvalClassMaskPlugin)
