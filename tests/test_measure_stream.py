"""CLI smoke tests for clns.shift.measure_stream with a stub encoder (no weight download)."""

import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from clns.shift import measure_stream as ms  # noqa: E402


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
    """(B, 3, H, W) -> (B, 6): per-channel mean and std."""

    def forward(self, x):
        return torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3))], dim=1)


def _stub_factory(tag):
    def build(name, device):
        ms.check_encoder_name(name)
        return StubEncoder().to(device).eval(), StubTransform(tag)

    return build


def _args(repo_root, tmp_path, *extra):
    return [
        "--benchmark-config", str(repo_root / "configs" / "benchmark" / "synthetic.yaml"),
        "--data-dir", str(tmp_path / "data"),
        "--benchmark-seed", "1",
        "--sample-seed", "0",
        "--encoder", "stub:enc",
        "--max-samples", "20",
        "--perm", "20",
        "--bootstrap", "10",
        "--device", "cpu",
        "--out", str(tmp_path / "out"),
        *extra,
    ]


def test_cli_smoke_outputs(repo_root, tmp_path, monkeypatch):
    pytest.importorskip("avalanche")
    pytest.importorskip("omegaconf")
    pytest.importorskip("torchvision")
    monkeypatch.setattr(ms, "build_encoder", _stub_factory("a"))
    ms.main(_args(repo_root, tmp_path))

    out = tmp_path / "out" / "synthetic" / "seed1"
    z = np.load(out / "shift_matrix.npz")
    mat = z["matrix"]
    assert mat.shape == (5, 5)
    assert np.isfinite(np.diag(mat, k=1)).all()
    assert np.isnan(mat[0, 2])  # consecutive pairs only by default
    assert np.isfinite(np.diag(z["null95"], k=1)).all()
    assert (np.diag(z["ci_lo"], k=1) <= np.diag(z["ci_hi"], k=1)).all()

    with open(out / "shift_pairs.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert {"i", "j", "mmd2", "ci_lo", "ci_hi", "null95", "p"} <= set(rows[0])
    assert rows[0]["measure"] == "marginal_mmd"
    assert rows[0]["cross_benchmark_comparable"] == "False"

    meta = json.loads((out / "shift_meta.json").read_text(encoding="utf-8"))
    assert meta["measure"] == "marginal_mmd"
    assert meta["cross_benchmark_comparable"] is False
    assert meta["benchmark_seed"] == 1 and meta["sample_seed"] == 0
    assert meta["T"] == 5
    assert sorted(meta["class_order"]) == list(range(10))
    assert meta["encoder_transform"] == "StubTransform(a)"
    assert "split_hashes" in meta and meta["pairs"] == "consecutive"


def test_cli_full_pairs_and_cache_key_tracks_transform(repo_root, tmp_path, monkeypatch):
    pytest.importorskip("avalanche")
    pytest.importorskip("omegaconf")
    pytest.importorskip("torchvision")
    cache = tmp_path / "out" / "synthetic" / "feature_cache"

    monkeypatch.setattr(ms, "build_encoder", _stub_factory("a"))
    ms.main(_args(repo_root, tmp_path, "--pairs", "full", "--perm", "0", "--bootstrap", "0"))
    assert len(list(cache.glob("*.pt"))) == 5
    mat = np.load(tmp_path / "out" / "synthetic" / "seed1" / "shift_matrix.npz")["matrix"]
    assert np.isfinite(mat).all()
    assert np.allclose(mat, mat.T, equal_nan=True)

    ms.main(_args(repo_root, tmp_path, "--perm", "0", "--bootstrap", "0"))  # cache hit
    assert len(list(cache.glob("*.pt"))) == 5

    monkeypatch.setattr(ms, "build_encoder", _stub_factory("b"))  # new transform repr
    ms.main(_args(repo_root, tmp_path, "--perm", "0", "--bootstrap", "0"))
    assert len(list(cache.glob("*.pt"))) == 10


def test_clip_encoder_rejected(tmp_path):
    for name in ("timm:vit_base_patch16_clip_224", "tv:CLIP"):
        with pytest.raises(ValueError, match="CLIP"):
            ms.main(["--benchmark-config", "unused.yaml", "--encoder", name,
                     "--out", str(tmp_path)])
        with pytest.raises(ValueError, match="CLIP"):
            ms.build_encoder(name, torch.device("cpu"))


def test_cache_key_and_pair_seed():
    base = dict(bench_name="b", meta={"class_order": [1, 0]}, exp=0, benchmark_seed=0,
                sample_seed=0, max_samples=10, encoder="tv:resnet50")
    k1 = ms.cache_key(**base, transform_repr="T1")
    assert k1 == ms.cache_key(**base, transform_repr="T1")
    assert k1 != ms.cache_key(**base, transform_repr="T2")
    assert k1 != ms.cache_key(**{**base, "benchmark_seed": 1}, transform_repr="T1")
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
