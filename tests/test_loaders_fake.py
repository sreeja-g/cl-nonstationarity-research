"""ImageNet-100 and DomainNet loaders on tiny fake directory trees (PIL JPEGs in tmp_path)."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("avalanche")
PIL = pytest.importorskip("PIL.Image")

from clns.data import build_benchmark  # noqa: E402
from clns.data.common import DONE_MARKER, read_list_file, splits_dir  # noqa: E402


def _jpeg(path: Path, color) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL.new("RGB", (8, 8), color=color).save(path, format="JPEG")


# --------------------------------------------------------------------------- #
# ImageNet-100
# --------------------------------------------------------------------------- #
def _fake_imagenet(root: Path, n_train=2, n_val=2, marker=True) -> list[str]:
    wnids = read_list_file(splits_dir() / "imagenet100" / "classes.txt")
    for c, w in enumerate(wnids):
        for k in range(n_train):
            _jpeg(root / "train" / w / f"{w}_{k}.JPEG", (c, k, 0))
        for k in range(n_val):
            _jpeg(root / "val" / w / f"ILSVRC2012_val_{c:05d}{k}.JPEG", (c, k, 9))
    if marker:
        (root / DONE_MARKER).write_text("")
    return wnids


def _in_cfg(load_benchmark_cfg, **kw):
    base = dict(val_images_per_class=2, image_size=16, n_experiences=10)
    base.update(kw)
    return load_benchmark_cfg("imagenet100", **base)


def test_imagenet100_fake(tmp_path, load_benchmark_cfg):
    root = tmp_path / "ILSVRC2012"
    _fake_imagenet(root)
    bundle = build_benchmark(_in_cfg(load_benchmark_cfg), str(tmp_path), seed=3)
    assert bundle.scenario == "class_il" and bundle.num_classes == 100
    bm = bundle.benchmark
    assert len(bm.train_stream) == 10 and len(bm.test_stream) == 10
    assert bundle.meta["train_sizes"] == [20] * 10 and bundle.meta["test_sizes"] == [20] * 10
    assert bundle.meta["class_order_seed"] == 1993
    # class order follows the training seed (seed 0 = committed UCIR/PODNet order)
    other = build_benchmark(_in_cfg(load_benchmark_cfg), str(tmp_path), seed=4)
    assert other.class_order != bundle.class_order
    assert bundle.meta["task_order"] == bundle.class_order
    assert bundle.meta["task_order_sha256"] != other.meta["task_order_sha256"]
    for k, exp in enumerate(bm.train_stream):
        assert exp.classes_in_this_experience == sorted(bundle.class_order[10 * k : 10 * k + 10])
    x, y, t = bm.test_stream[0].dataset[0]
    assert tuple(x.shape) == (3, 16, 16) and t == 0


def test_imagenet100_requires_marker(tmp_path, load_benchmark_cfg):
    _fake_imagenet(tmp_path / "ILSVRC2012", n_train=1, n_val=2, marker=False)
    with pytest.raises(FileNotFoundError, match="download_data.sh imagenet100"):
        build_benchmark(_in_cfg(load_benchmark_cfg), str(tmp_path), 0)


def test_imagenet100_val_count_checked(tmp_path, load_benchmark_cfg):
    _fake_imagenet(tmp_path / "ILSVRC2012", n_train=1, n_val=2)
    with pytest.raises(ValueError, match="expected 3 images per class"):
        build_benchmark(_in_cfg(load_benchmark_cfg, val_images_per_class=3), str(tmp_path), 0)


def test_imagenet100_leak_by_basename(tmp_path, load_benchmark_cfg):
    root = tmp_path / "ILSVRC2012"
    wnids = _fake_imagenet(root, n_train=1, n_val=1)
    # replace one val image by a copy of a train image (same file name, other folder)
    w = wnids[5]
    for p in (root / "val" / w).iterdir():
        p.unlink()
    _jpeg(root / "val" / w / f"{w}_0.JPEG", (1, 2, 3))
    with pytest.raises(ValueError, match="leakage"):
        build_benchmark(_in_cfg(load_benchmark_cfg, val_images_per_class=1), str(tmp_path), 0)


# --------------------------------------------------------------------------- #
# DomainNet
# --------------------------------------------------------------------------- #
def _fake_domainnet(root: Path, leak_domain: str | None = None) -> list[str]:
    domains = read_list_file(splits_dir() / "domainnet" / "domain_order.txt")
    (root / "splits").mkdir(parents=True, exist_ok=True)
    for d in domains:
        lines = {"train": [], "test": []}
        for split, n in (("train", 3), ("test", 2)):
            for c in range(2):
                for k in range(n):
                    rel = f"{d}/class{c}/{split}_{k}.jpg"
                    _jpeg(root / rel, (c, k, 1))
                    lines[split].append(f"{rel} {c}")
        if d == leak_domain:
            lines["test"].append(lines["train"][0])
        for split in ("train", "test"):
            (root / "splits" / f"{d}_{split}.txt").write_text("\n".join(lines[split]) + "\n")
    (root / DONE_MARKER).write_text("")
    return domains


def test_domainnet_fake(tmp_path, load_benchmark_cfg):
    domains = _fake_domainnet(tmp_path / "domainnet")
    cfg = load_benchmark_cfg("domainnet", num_classes=2, image_size=16)
    bundle = build_benchmark(cfg, str(tmp_path), 0)
    assert bundle.scenario == "domain_il" and bundle.num_classes == 2
    assert bundle.meta["domain_order"] == domains
    assert len(bundle.benchmark.train_stream) == 6
    assert bundle.meta["train_sizes"] == [6] * 6 and bundle.meta["test_sizes"] == [4] * 6
    assert bundle.meta["test_exp_to_train_exp"] == list(range(6))
    x, y, t = bundle.benchmark.train_stream[0].dataset[0]
    assert tuple(x.shape) == (3, 16, 16) and t == 0


def test_domainnet_leak_detected(tmp_path, load_benchmark_cfg):
    _fake_domainnet(tmp_path / "domainnet", leak_domain="painting")
    cfg = load_benchmark_cfg("domainnet", num_classes=2, image_size=16)
    with pytest.raises(ValueError, match="leakage in DomainNet/painting"):
        build_benchmark(cfg, str(tmp_path), 0)


def _test_image_set(bundle, root) -> set:
    from clns.shift.feature_cache import sample_ids

    return {i for e in bundle.benchmark.test_stream for i in sample_ids(e.dataset, str(root))}


def test_imagenet100_test_subsample_same_for_all_seeds(tmp_path, load_benchmark_cfg):
    """subsample_per_class: every training seed (= another class order) sees the same test
    images; the training subsample still follows the seed."""
    _fake_imagenet(tmp_path / "ILSVRC2012", n_train=3, n_val=4)
    cfg = _in_cfg(load_benchmark_cfg, val_images_per_class=4, subsample_per_class=2)
    b0, b1 = (build_benchmark(cfg, str(tmp_path), s) for s in (0, 1))
    assert b0.class_order != b1.class_order
    t0, t1 = _test_image_set(b0, tmp_path), _test_image_set(b1, tmp_path)
    assert len(t0) == 200 and t0 == t1


def test_domainnet_seed_order_and_test_subsample(tmp_path, load_benchmark_cfg):
    from clns.data.domainnet import domainnet_domain_order
    from clns.shift.feature_cache import sample_ids

    committed = _fake_domainnet(tmp_path / "domainnet")
    cfg = load_benchmark_cfg("domainnet", num_classes=2, image_size=16, subsample_per_class=1)
    b0, b2 = (build_benchmark(cfg, str(tmp_path), s) for s in (0, 2))
    expected = domainnet_domain_order(committed, 2)
    assert expected != committed and b2.meta["domain_order"] == expected
    assert b2.meta["task_order"] == expected
    assert b2.meta["task_order_sha256"] != b0.meta["task_order_sha256"]
    # the train stream really follows the seed's domain order (ids are <domain>/class/...)
    for k, exp in enumerate(b2.benchmark.train_stream):
        doms = {i.split(":", 1)[1].split("/")[1] for i in sample_ids(exp.dataset, str(tmp_path))}
        assert doms == {expected[k]}
    # the test subsample of a domain does not depend on its position in the order
    assert _test_image_set(b0, tmp_path) == _test_image_set(b2, tmp_path)
