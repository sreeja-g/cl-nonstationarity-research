"""DomainNet domain-incremental benchmark (cleaned release, official per-domain lists).

Experience k = domain ``domain_order[k]`` (order fixed in data_splits/domainnet/domain_order.txt);
train data from ``<domain>_train.txt``, test data from ``<domain>_test.txt`` (official lists, lines
``relative/path label``). All 345 classes are present in every domain.
Requires ``<root>/.done`` (``scripts/download_data.sh domainnet``; see ``require_data_root``).
Train/test disjointness is checked per domain on the root-relative paths of the official lists.
"""

from __future__ import annotations

from pathlib import Path

from avalanche.benchmarks.utils.datasets_from_filelists import FilelistDataset

from .common import (
    BenchmarkBundle,
    assert_disjoint,
    build_transforms,
    check_manifest,
    dataset_classes,
    finalize,
    make_cls_dataset,
    read_list_file,
    require_data_root,
    splits_dir,
)


def build_domainnet(cfg, data_dir: str, seed: int) -> BenchmarkBundle:
    order_file = splits_dir() / "domainnet" / str(cfg.get("domain_order_file", "domain_order.txt"))
    order_sha = check_manifest(order_file, cfg.get("domain_order_sha256"))
    domains = read_list_file(order_file)
    if len(set(domains)) != len(domains) or len(domains) != int(cfg.n_experiences):
        raise ValueError(
            f"{order_file}: expected {cfg.n_experiences} unique domains, got {domains}"
        )

    root = Path(data_dir) / cfg.get("data_subdir", "domainnet")
    require_data_root(root, cfg, "DomainNet", "domainnet")
    flist_dir = root / cfg.get("filelist_dir", "splits")
    train_tf, eval_tf = build_transforms(cfg)

    train_list, test_list = [], []
    for dom in domains:
        for split in ("train", "test"):
            f = flist_dir / f"{dom}_{split}.txt"
            if not f.is_file():
                raise FileNotFoundError(f"Missing DomainNet list file '{f}'")
        tr = FilelistDataset(root, flist_dir / f"{dom}_train.txt")
        te = FilelistDataset(root, flist_dir / f"{dom}_test.txt")
        assert_disjoint([p for p, _ in tr.imgs], [p for p, _ in te.imgs], f"DomainNet/{dom}")
        train_list.append(
            make_cls_dataset(tr, tr.targets, train_transform=train_tf, eval_transform=eval_tf)
        )
        test_list.append(
            make_cls_dataset(
                te, te.targets, train_transform=train_tf, eval_transform=eval_tf, group="eval"
            )
        )

    classes = sorted({c for d in train_list for c in dataset_classes(d)})
    num_classes = int(cfg.get("num_classes", 345))
    if max(classes) + 1 != num_classes:
        raise ValueError(
            f"DomainNet: expected {num_classes} classes, found labels up to {max(classes)}"
        )

    meta = {
        "protocol": "domain_incremental",
        "domain_order": domains,
        "domain_order_file": str(order_file),
        "domain_order_sha256": order_sha,
        "n_experiences_full": len(domains),
        "dataset_root": str(root),
    }
    benchmark = finalize(train_list, test_list, cfg, seed, meta)
    return BenchmarkBundle(
        benchmark=benchmark,
        scenario="domain_il",
        num_classes=num_classes,
        class_order=None,
        meta=meta,
    )
