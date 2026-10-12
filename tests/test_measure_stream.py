"""clns.shift.measure_stream / feature_cache with a stub encoder (no weights, no network)."""

import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from clns.shift import feature_cache as fc  # noqa: E402
from clns.shift import measure_stream as ms  # noqa: E402

STUB = "stub:enc"


class StubTransform:
    """PIL image -> float CHW tensor in [0, 1]; ``tag`` only changes the repr."""

    def __init__(self, tag: str = "a"):
        self.tag = tag

    def __call__(self, img):
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def __repr__(self):
        return f"StubTransform({self.tag})"


class StubEncoder(torch.nn.Module):
    """(B, 3, H, W) -> (B, 6): per-channel mean and std. Counts embedded images."""

    calls = 0

    def forward(self, x):
        StubEncoder.calls += int(x.shape[0])
        return torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3))], dim=1)


def _stub_factory(tag):
    def build(name, device, **kwargs):
        ms.check_encoder_name(name)
        return StubEncoder().to(device).eval(), StubTransform(tag)

    return build


@pytest.fixture
def stub(monkeypatch):
    pytest.importorskip("avalanche")
    pytest.importorskip("omegaconf")
    pytest.importorskip("torchvision")
    monkeypatch.setitem(ms.ENCODER_BANDWIDTHS, STUB, (0.5, 1.0, 2.0))
    monkeypatch.setattr(ms, "build_encoder", _stub_factory("a"))
    StubEncoder.calls = 0
    return monkeypatch


def _args(repo_root, tmp_path, *extra, config="synthetic.yaml"):
    return [
        "--benchmark-config", str(repo_root / "configs" / "benchmark" / config),
        "--data-dir", str(tmp_path / "data"),
        "--seed", "1",
        "--sample-seed", "0",
        "--encoder", STUB,
        "--n-per-experience", "15",
        "--perm", "20",
        "--bootstrap", "10",
        "--device", "cpu",
        "--out", str(tmp_path / "out"),
        *extra,
    ]


def test_cli_smoke_outputs(repo_root, tmp_path, stub):
    ms.main(_args(repo_root, tmp_path))

    out = tmp_path / "out" / "synthetic" / "seed1"
    z = np.load(out / "shift_matrix.npz")
    mat = z["matrix"]
    assert mat.shape == (5, 5)
    assert np.isfinite(np.diag(mat, k=1)).all()
    assert np.isnan(mat[0, 2])  # consecutive pairs only by default
    assert np.isfinite(np.diag(z["null95"], k=1)).all()
    assert (np.diag(z["ci_lo"], k=1) <= np.diag(z["ci_hi"], k=1)).all()
    assert list(z["bandwidths"]) == [0.5, 1.0, 2.0]
    assert list(z["n"]) == [15] * 5

    with open(out / "shift_pairs.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert {"i", "j", "mmd2", "ci_lo", "ci_hi", "null95", "p"} <= set(rows[0])
    assert rows[0]["measure"] == "marginal_mmd"
    assert rows[0]["cross_benchmark_comparable"] == "False"

    meta = json.loads((out / "shift_meta.json").read_text(encoding="utf-8"))
    assert meta["measure"] == "marginal_mmd"
    assert meta["cross_benchmark_comparable"] is False
    assert meta["seed"] == 1 and meta["sample_seed"] == 0
    assert meta["T"] == 5
    assert sorted(meta["class_order"]) == list(range(10))
    assert meta["task_order"] == meta["class_order"]
    assert meta["task_order_sha256"] == meta["split_hashes"]["task_order_sha256"]
    assert meta["encoder_transform"] == "StubTransform(a)"
    assert meta["bandwidths"] == [0.5, 1.0, 2.0] and meta["l2_normalized"] is True
    assert meta["n_per_experience"] == 15 and meta["pairs"] == "consecutive"
    assert 0.0 < meta["median_sq_dist"] <= 2.0  # non-negative unit vectors


def test_n_capped_at_smallest_experience(repo_root, tmp_path, stub):
    ms.main(_args(repo_root, tmp_path, "--n-per-experience", "1000", "--perm", "0",
                  "--bootstrap", "0"))
    meta = json.loads((tmp_path / "out" / "synthetic" / "seed1" / "shift_meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["n_per_experience"] == min(meta["experience_sizes"]) == 40  # 2 classes x 20


def test_bandwidths_fixed_per_encoder_not_per_benchmark():
    assert ms.encoder_bandwidths("tv:resnet50") == ms.encoder_bandwidths("dino:resnet50")
    with pytest.raises(ValueError, match="no fixed bandwidth"):
        ms.encoder_bandwidths("tv:resnet18")
    assert "median" not in ms.parse_args(["--benchmark-config", "x", "--out", "y"]).__dict__


def test_cache_reused_and_tracks_transform(repo_root, tmp_path, stub):
    a = _args(repo_root, tmp_path, "--pairs", "full", "--perm", "0", "--bootstrap", "0")
    ms.main(a)
    first = StubEncoder.calls
    assert first == 5 * 15
    mat = np.load(tmp_path / "out" / "synthetic" / "seed1" / "shift_matrix.npz")["matrix"]
    assert np.isfinite(mat).all() and np.allclose(mat, mat.T)

    ms.main(a)  # every image cached -> nothing embedded
    assert StubEncoder.calls == first

    stub.setattr(ms, "build_encoder", _stub_factory("b"))  # new transform repr -> new cache
    ms.main(a)
    assert StubEncoder.calls == 2 * first
    assert len(list((tmp_path / "out" / "feature_cache").iterdir())) == 2


def _fake_domainnet_cfg(repo_root, tmp_path):
    """Tiny DomainNet tree (2 classes, 4 train images per domain) + a matching config file."""
    PIL = pytest.importorskip("PIL.Image")
    from omegaconf import OmegaConf

    from clns.data.common import DONE_MARKER, read_list_file, splits_dir

    root = tmp_path / "data" / "domainnet"
    (root / "splits").mkdir(parents=True)
    for d_i, d in enumerate(read_list_file(splits_dir() / "domainnet" / "domain_order.txt")):
        for split, n in (("train", 2), ("test", 1)):
            lines = []
            for c in range(2):
                for k in range(n):
                    rel = f"{d}/class{c}/{split}_{k}.jpg"
                    (root / rel).parent.mkdir(parents=True, exist_ok=True)
                    PIL.new("RGB", (8, 8), color=(40 * d_i, 100 * c + 20 * k, 7)).save(root / rel)
                    lines.append(f"{rel} {c}")
            (root / "splits" / f"{d}_{split}.txt").write_text("\n".join(lines) + "\n")
    (root / DONE_MARKER).write_text("")
    cfg = OmegaConf.merge(OmegaConf.load(repo_root / "configs" / "benchmark" / "domainnet.yaml"),
                          {"num_classes": 2, "image_size": 8})
    path = tmp_path / "domainnet_fake.yaml"
    OmegaConf.save(cfg, path)
    return path


def test_cache_hits_under_permuted_order(repo_root, tmp_path, stub):
    """Seed 1 permutes the DomainNet domains: no image is embedded again, and every domain pair
    gets the same MMD as under seed 0 (same bandwidths, same per-image selection)."""
    cfg = _fake_domainnet_cfg(repo_root, tmp_path)

    def run(seed):
        args = _args(repo_root, tmp_path, "--pairs", "full", "--perm", "0", "--bootstrap", "0")
        args[1] = str(cfg)
        args[args.index("--seed") + 1] = str(seed)
        ms.main(args)
        out = tmp_path / "out" / "domainnet" / f"seed{seed}"
        meta = json.loads((out / "shift_meta.json").read_text(encoding="utf-8"))
        return meta, np.load(out / "shift_matrix.npz")["matrix"]

    m0, mat0 = run(0)
    calls = StubEncoder.calls
    assert calls == 6 * 4 and m0["n_per_experience"] == 4
    m1, mat1 = run(1)
    assert StubEncoder.calls == calls  # every feature came from the per-image cache
    o0, o1 = m0["task_order"], m1["task_order"]
    assert o0 != o1 and sorted(o0) == sorted(o1)
    assert m0["task_order_sha256"] != m1["task_order_sha256"]
    assert m0["bandwidths"] == m1["bandwidths"]
    for a in range(6):
        for b in range(6):
            a1, b1 = o1.index(o0[a]), o1.index(o0[b])
            assert mat0[a, b] == pytest.approx(mat1[a1, b1], abs=1e-9)


def test_select_ids_order_independent():
    ids = [f"path:x/{i}.jpg" for i in range(50)]
    a = [ids[k] for k in fc.select_ids(ids, 10, 3)]
    rev = ids[::-1]
    b = [rev[k] for k in fc.select_ids(rev, 10, 3)]
    assert sorted(a) == sorted(b) and len(set(a)) == 10
    assert sorted(a) != sorted(ids[k] for k in fc.select_ids(ids, 10, 4))


def test_sample_ids_paths_and_content(tmp_path):
    pytest.importorskip("avalanche")
    from avalanche.benchmarks.utils.datasets_from_filelists import PathsDataset
    from torch.utils.data import TensorDataset

    from clns.data.common import make_cls_dataset

    root = tmp_path / "data" / "bench"
    files = [(f"cls{i % 2}/{i}.jpg", i % 2) for i in range(6)]
    ds = make_cls_dataset(PathsDataset(root, files), targets=[y for _, y in files])
    sub = ds.subset([4, 1, 3])
    ids = fc.sample_ids(sub, tmp_path / "data")
    assert ids == ["path:bench/cls0/4.jpg", "path:bench/cls1/1.jpg", "path:bench/cls1/3.jpg"]
    cat = ds.subset([0, 1]).concat(ds.subset([5]))
    assert fc.sample_ids(cat, tmp_path / "data")[2] == "path:bench/cls1/5.jpg"

    g = torch.Generator().manual_seed(0)
    t1 = TensorDataset(torch.randn(4, 3, 2, 2, generator=g), torch.zeros(4, dtype=torch.long))
    t2 = TensorDataset(torch.randn(4, 3, 2, 2, generator=g), torch.zeros(4, dtype=torch.long))
    d1 = make_cls_dataset(t1, [0] * 4)
    d2 = make_cls_dataset(t2, [0] * 4)
    ids1, ids2 = fc.sample_ids(d1), fc.sample_ids(d2)
    assert all(s.startswith("sha:") for s in ids1)
    assert not set(ids1) & set(ids2)  # different data (e.g. another seed) -> different ids
    assert fc.sample_ids(d1.subset([2, 0])) == [ids1[2], ids1[0]]


def test_feature_cache_append_only(tmp_path):
    c = fc.FeatureCache(tmp_path, "enc", "T", "p1")
    assert c.missing(["a", "b", "a"]) == ["a", "b"]
    c.add(["a", "b"], np.ones((2, 3)))
    c.add(["b", "c"], np.full((2, 3), 2.0))  # "b" kept from the first shard
    c2 = fc.FeatureCache(tmp_path, "enc", "T", "p1")
    assert len(c2) == 3
    np.testing.assert_array_equal(c2.get(["b", "c"]), [[1, 1, 1], [2, 2, 2]])
    assert len(list(c2.dir.glob("shard_*.npz"))) == 2
    assert not list(c2.dir.glob("*.tmp"))
    assert fc.FeatureCache(tmp_path, "enc", "T2", "p1").dir != c2.dir


def test_features_are_l2_normalised(repo_root, tmp_path, stub, monkeypatch):
    seen = []
    real = ms.pairwise_shift

    def spy(feats, *a, **k):
        seen.extend(feats)
        return real(feats, *a, **k)

    monkeypatch.setattr(ms, "pairwise_shift", spy)
    ms.main(_args(repo_root, tmp_path, "--perm", "0", "--bootstrap", "0"))
    norms = torch.cat([f.norm(dim=1) for f in seen])
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_clip_encoder_rejected(tmp_path):
    for name in ("dino:clip_resnet50", "tv:CLIP"):
        with pytest.raises(ValueError, match="CLIP"):
            ms.main(["--benchmark-config", "unused.yaml", "--encoder", name,
                     "--out", str(tmp_path)])
        with pytest.raises(ValueError, match="CLIP"):
            ms.build_encoder(name, torch.device("cpu"))


def test_pair_seed_and_pairs():
    seeds = {ms.pair_seed(0, 0, 1, 0), ms.pair_seed(0, 1, 2, 0), ms.pair_seed(1, 0, 1, 0),
             ms.pair_seed(0, 0, 1, 1)}
    assert len(seeds) == 4
    assert ms.pair_list(4, "consecutive") == [(0, 1), (1, 2), (2, 3)]
    assert len(ms.pair_list(4, "full")) == 6


def test_raw_item_guard_and_to_pil():
    pytest.importorskip("torchvision")
    with pytest.raises(RuntimeError, match="still tensors"):
        ms._check_raw_item(torch.zeros(3, 4, 4), benchmark_has_transform=True)
    ms._check_raw_item(torch.zeros(3, 4, 4), benchmark_has_transform=False)
    img = ms.to_pil(torch.full((3, 4, 4), 2.0))  # clamped to 1.0 -> 255
    assert img.mode == "RGB" and img.size == (4, 4)
    assert np.asarray(img).max() == 255
    with pytest.raises(ValueError):
        ms.to_pil(torch.zeros(5, 4, 4))


# --------------------------------------------------------------------------- #
# DINO loader (fake checkpoints, no network)
# --------------------------------------------------------------------------- #
def _fake_dino(path, prefix="", drop=None, extra=None):
    tv = pytest.importorskip("torchvision")
    state = {k: v for k, v in tv.models.resnet50(weights=None).state_dict().items()
             if not k.startswith("fc.")}
    if drop:
        state.pop(drop)
    if extra:
        state[extra] = torch.zeros(1)
    torch.save({prefix + k: v for k, v in state.items()}, path)
    return path


def test_dino_fake_checkpoint_loads(tmp_path):
    from clns.models.pretrained import sha256_binary

    p = _fake_dino(tmp_path / ms.DINO_RESNET50_FILE, prefix="module.")
    sha = sha256_binary(p)
    info = {}
    model, transform = ms.build_encoder("dino:resnet50", torch.device("cpu"),
                                        weights_dir=tmp_path, dino_sha256=sha, info=info)
    assert isinstance(model.fc, torch.nn.Identity) and not model.training
    assert all(not q.requires_grad for q in model.parameters())
    assert info["encoder_weights_sha256"] == sha
    out = model(torch.zeros(2, 3, 64, 64))
    assert tuple(out.shape) == (2, 2048)
    assert "CenterCrop" in repr(transform)


def test_dino_sha_mismatch_and_bad_keys(tmp_path):
    p = _fake_dino(tmp_path / ms.DINO_RESNET50_FILE)
    with pytest.raises(ValueError, match="sha256"):
        ms.load_dino_resnet50(p, "0" * 64)
    bad = _fake_dino(tmp_path / "missing.pth", drop="layer4.2.bn3.weight")
    with pytest.raises(ValueError, match="not a DINO ResNet-50"):
        ms.load_dino_resnet50(bad, None)
    bad2 = _fake_dino(tmp_path / "extra.pth", extra="head.weight")
    with pytest.raises(ValueError, match="not a DINO ResNet-50"):
        ms.load_dino_resnet50(bad2, None)
    with pytest.raises(FileNotFoundError, match="download_data.sh dino"):
        ms.load_dino_resnet50(tmp_path / "nope.pth", None)


def test_select_ids_per_class():
    ids = [f"path:x/c{c}/{i}.jpg" for c in range(3) for i in range(20)]
    labels = [c for c in range(3) for _ in range(20)]
    pos = fc.select_ids(ids, 10, 0, labels=labels)
    counts = np.bincount([labels[k] for k in pos], minlength=3)
    assert len(pos) == 10 and sorted(counts.tolist()) == [3, 3, 4]  # ceil(10/3)=4, cut to 10
    # order-independent
    perm = np.random.default_rng(0).permutation(len(ids))
    pos_p = fc.select_ids([ids[k] for k in perm], 10, 0, labels=[labels[k] for k in perm])
    assert sorted(ids[perm[k]] for k in pos_p) == sorted(ids[k] for k in pos)
    # a class keeps the same images whatever other classes share its experience (same k, n)
    other = ids[:20] + [f"path:y/c{c}/{i}.jpg" for c in (7, 8) for i in range(20)]
    lab_o = [0] * 20 + [c for c in (7, 8) for _ in range(20)]
    full = fc.select_ids(ids, 12, 5, labels=labels)
    full_o = fc.select_ids(other, 12, 5, labels=lab_o)
    assert {ids[k] for k in full if labels[k] == 0} == {other[k] for k in full_o if lab_o[k] == 0}
    # short classes are topped up by the others
    pos_s = fc.select_ids(ids[:22], 10, 0, labels=labels[:22])
    assert len(pos_s) == 10 and sum(labels[k] == 1 for k in pos_s) == 2


def test_feature_cache_key_tracks_weights(tmp_path):
    a = fc.FeatureCache(tmp_path, "enc", "T", "p1", weights_sha256="aa", weights="w.pth")
    assert fc.FeatureCache(tmp_path, "enc", "T", "p1", weights_sha256="aa", weights="w.pth").dir \
        == a.dir
    assert fc.FeatureCache(tmp_path, "enc", "T", "p1", weights_sha256="bb", weights="w.pth").dir \
        != a.dir
    assert fc.FeatureCache(tmp_path, "enc", "T", "p1").dir != a.dir


def test_weights_hash_change_invalidates_cache(repo_root, tmp_path, stub):
    def factory(sha):
        def build(name, device, info=None, **kwargs):
            if info is not None:
                info["encoder_weights"] = "/w/enc.pth"
                info["encoder_weights_sha256"] = sha
            return StubEncoder().to(device).eval(), StubTransform("a")

        return build

    a = _args(repo_root, tmp_path, "--perm", "0", "--bootstrap", "0")
    stub.setattr(ms, "build_encoder", factory("1" * 64))
    ms.main(a)
    first = StubEncoder.calls
    ms.main(a)
    assert StubEncoder.calls == first  # same weights: all cached
    stub.setattr(ms, "build_encoder", factory("2" * 64))
    ms.main(a)
    assert StubEncoder.calls == 2 * first  # other weights: everything re-embedded
    meta = json.loads((tmp_path / "out" / "synthetic" / "seed1" / "shift_meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["encoder_weights_sha256"] == "2" * 64


def test_median_sq_dist():
    f = [torch.nn.functional.normalize(torch.rand(30, 8), dim=1) for _ in range(3)]
    m = ms.median_sq_dist(f)
    assert 0.0 < m <= 2.0
    sub = ms.median_sq_dist(f, max_points=50, seed=1)
    assert sub == ms.median_sq_dist(f, max_points=50, seed=1) and 0.0 < sub <= 2.0
