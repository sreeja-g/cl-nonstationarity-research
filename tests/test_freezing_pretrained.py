"""Model arms: pretrained loading, partial freezing, arm x benchmark rules, LwF teacher mode.

No network: Places365 checkpoints are fake ``module.``-prefixed CSAIL-style files written on the
fly (``fake_places`` fixture, conftest.py: legacy torch.save format, no num_batches_tracked, like
the real release); ImageNet paths monkeypatch ``clns.models.pretrained.imagenet_state_dict``.
Inputs are 32 px (build_model is called directly).
"""
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("avalanche")
from omegaconf import OmegaConf  # noqa: E402
from torchvision.ops.misc import FrozenBatchNorm2d  # noqa: E402

import clns.models.pretrained as pretrained_mod  # noqa: E402
from clns.models import (  # noqa: E402
    ALLOWED,
    ARM_SPECS,
    build_model,
    count_params,
    validate_model_config,
)
from clns.models.backbones import GlobalMeanPool2d, MBV2Wrapper  # noqa: E402
from clns.strategies import build_strategy  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN = OmegaConf.create(
    {"epochs": 1, "mb_size": 8, "eval_mb_size": 8, "lr": 0.05, "optimizer": "sgd",
     "momentum": 0.9, "weight_decay": 0.0005, "num_workers": 0}
)
METHODS = {
    "naive": {"name": "naive"},
    "ewc": {"name": "ewc", "ewc_lambda": 10, "mode": "online", "decay_factor": 0.9},
    "lwf": {"name": "lwf", "alpha": 1.0, "temperature": 2.0},
    "er": {"name": "er", "mem_size": 20, "storage": "class_balanced", "batch_size_mem": 4},
    "derpp": {"name": "derpp", "mem_size": 20, "batch_size_mem": 4, "alpha": 0.1, "beta": 0.5},
}


def _arm(name: str) -> OmegaConf:
    return OmegaConf.load(REPO_ROOT / "configs" / "model" / f"{name}.yaml")


@pytest.fixture
def places_dir(tmp_path, fake_places):
    sha = fake_places(tmp_path / "resnet18_places365.pth.tar")
    return tmp_path, sha


def _places_cfg(sha, name="r18_places"):
    return OmegaConf.merge(_arm(name), {"weights_sha256": sha})


# --------------------------------------------------------------------------- #
# Places365 loading
# --------------------------------------------------------------------------- #
def test_places_strict_load_head_and_freezing(places_dir):
    wdir, sha = places_dir
    info: dict = {}
    m = build_model(_places_cfg(sha), 10, "clear10", weights_dir=wdir, info=info)
    ckpt = torch.load(wdir / "resnet18_places365.pth.tar", weights_only=True)["state_dict"]
    assert torch.equal(m.conv1.weight, ckpt["module.conv1.weight"])
    assert torch.equal(m.layer4[1].conv2.weight, ckpt["module.layer4.1.conv2.weight"])
    # frozen BN keeps the checkpoint's statistics as buffers
    assert isinstance(m.bn1, FrozenBatchNorm2d) and isinstance(m.layer3[1].bn2, FrozenBatchNorm2d)
    assert torch.equal(m.layer1[0].bn1.running_var, ckpt["module.layer1.0.bn1.running_var"])
    assert isinstance(m.layer4[0].bn1, torch.nn.BatchNorm2d)  # trainable block stays BN
    assert isinstance(m.avgpool, GlobalMeanPool2d)
    assert m.fc.out_features == 10
    expected = sum(p.numel() for p in m.layer4.parameters()) + sum(
        p.numel() for p in m.fc.parameters()
    )
    assert count_params(m) == (expected, info["params"]["total"])
    assert info["params"]["trainable"] == expected < info["params"]["total"]
    assert info["weights"]["sha256"] == sha
    assert all(not p.requires_grad for n, p in m.named_parameters()
               if not n.startswith(("layer4.", "fc.")))
    assert m(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


@pytest.mark.parametrize("legacy", [True, False])
def test_places_csail_format_loads(tmp_path, fake_places, legacy):
    """The real CSAIL file: legacy (non-zip) torch.save, extra top-level keys, ``module.`` keys
    and NO num_batches_tracked buffers (pre-0.4.1); the modern format loads as well."""
    path = tmp_path / "resnet18_places365.pth.tar"
    sha = fake_places(path, legacy=legacy)
    with open(path, "rb") as f:
        assert (f.read(2) == b"PK") is not legacy  # zip magic only for the new format
    raw = torch.load(path, weights_only=True)
    assert set(raw) == {"epoch", "arch", "state_dict", "best_prec1"}
    assert any(k.endswith("num_batches_tracked") for k in raw["state_dict"]) is not legacy
    info: dict = {}
    m = build_model(_places_cfg(sha), 10, "clear10", weights_dir=tmp_path, info=info)
    sd = raw["state_dict"]
    assert torch.equal(m.conv1.weight, sd["module.conv1.weight"])
    assert torch.equal(m.layer4[0].bn1.running_var, sd["module.layer4.0.bn1.running_var"])
    assert int(m.layer4[0].bn1.num_batches_tracked) == 0  # filled by the strict load
    assert info["weights"]["sha256"] == sha


def test_places_all_trainable(places_dir):
    wdir, sha = places_dir
    m = build_model(_places_cfg(sha, "r18_places_all"), 5, "imagenet100", weights_dir=wdir)
    trainable, total = count_params(m)
    assert trainable == total
    assert not any(isinstance(x, FrozenBatchNorm2d) for x in m.modules())


def test_param_total_same_for_all_and_last_block(places_dir):
    wdir, sha = places_dir
    i_all: dict = {}
    i_last: dict = {}
    i_scratch: dict = {}
    build_model(_places_cfg(sha, "r18_places_all"), 10, "clear10", weights_dir=wdir, info=i_all)
    build_model(_places_cfg(sha, "r18_places"), 10, "clear10", weights_dir=wdir, info=i_last)
    build_model(_arm("r18_scratch"), 10, "clear10", info=i_scratch)
    assert i_all["params"]["total"] == i_last["params"]["total"] == i_scratch["params"]["total"]
    assert i_last["params"]["trainable"] < i_last["params"]["total"]


def test_places_errors(tmp_path, places_dir, fake_places):
    wdir, sha = places_dir
    with pytest.raises(ValueError, match="sha256"):
        build_model(_places_cfg("0" * 64), 10, "clear10", weights_dir=wdir)
    with pytest.raises(FileNotFoundError, match="download_data.sh.*BEFORE the first GPU run"):
        build_model(_places_cfg(sha), 10, "clear10", weights_dir=tmp_path / "nowhere")
    with pytest.raises(ValueError, match="weights_dir"):
        build_model(_places_cfg(sha), 10, "clear10", weights_dir=None)
    bad = tmp_path / "bad"
    bad.mkdir()
    # genuinely missing weights are still refused (legacy and modern format)
    for legacy in (True, False):
        bad_sha = fake_places(bad / "resnet18_places365.pth.tar",
                              drop="layer2.0.bn1.running_mean", legacy=legacy)
        with pytest.raises(ValueError, match="tensors.*layer2.0.bn1.running_mean"):
            build_model(_places_cfg(bad_sha), 10, "clear10", weights_dir=bad)


def test_places_unpinned_sha_warns_and_records(places_dir, caplog):
    wdir, sha = places_dir
    info: dict = {}
    with caplog.at_level("WARNING"):
        build_model(_places_cfg(None), 10, "clear10", weights_dir=wdir, info=info)
    assert info["weights"]["sha256"] == sha
    assert any("No weights_sha256 pinned" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# ImageNet arms (monkeypatched loader) + MobileNetV2 last block
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_imagenet(tmp_path, monkeypatch):
    from clns.models.backbones import build_backbone

    def _fake(arch, weights_dir, sha256):
        net = build_backbone(arch, num_classes=1000)
        for p in net.parameters():
            torch.nn.init.constant_(p, 0.01)
        path = Path(weights_dir) / f"fake_{arch}.pth"
        path.write_bytes(arch.encode())
        return net.state_dict(), path, pretrained_mod.check_sha256(path, sha256)

    monkeypatch.setattr(pretrained_mod, "imagenet_state_dict", _fake)
    return tmp_path


def test_mbv2_imagenet_last_block(fake_imagenet):
    info: dict = {}
    m = build_model(_arm("mbv2_imagenet"), 6, "core50_nic", weights_dir=fake_imagenet, info=info)
    assert isinstance(m, MBV2Wrapper) and isinstance(m.pool, GlobalMeanPool2d)
    assert torch.all(m.features[0][0].weight == 0.01)  # loaded
    assert m.classifier[1].out_features == 6
    for name, p in m.named_parameters():
        trainable = name.startswith(("features.17.", "features.18.", "classifier."))
        assert p.requires_grad == trainable, name
    assert isinstance(m.features[16].conv[1][1], FrozenBatchNorm2d)
    assert isinstance(m.features[18][1], torch.nn.BatchNorm2d)  # intended: stays trainable
    assert info["params"]["trainable"] < info["params"]["total"]
    assert m(torch.zeros(2, 3, 32, 32)).shape == (2, 6)


def test_r18_imagenet_loads(fake_imagenet):
    m = build_model(_arm("r18_imagenet"), 4, "core50_nic", weights_dir=fake_imagenet)
    assert torch.all(m.conv1.weight == 0.01)
    assert not m.layer3[0].conv1.weight.requires_grad and m.layer4[0].conv1.weight.requires_grad


def test_imagenet_weights_file_must_match_url(fake_imagenet):
    assert _arm("r18_imagenet").weights_file == pretrained_mod.imagenet_weights_file("resnet18")
    assert _arm("mbv2_imagenet").weights_file == pretrained_mod.imagenet_weights_file(
        "mobilenetv2")
    bad = OmegaConf.merge(_arm("r18_imagenet"), {"weights_file": "resnet18-other.pth"})
    with pytest.raises(ValueError, match="weights_file"):
        build_model(bad, 4, "core50_nic", weights_dir=fake_imagenet)


def test_imagenet_cached_file_sha_checked_before_load(tmp_path, monkeypatch):
    """A cached torchvision file is hashed on every load, before torchvision unpickles it."""
    import torchvision.models as tvm

    def _no_load(*a, **k):
        raise AssertionError("get_state_dict must not run when the cached file's hash differs")

    monkeypatch.setattr(type(tvm.ResNet18_Weights.IMAGENET1K_V1), "get_state_dict", _no_load)
    cached = tmp_path / pretrained_mod.imagenet_weights_file("resnet18")
    cached.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256"):
        pretrained_mod.imagenet_state_dict("resnet18", tmp_path, "0" * 64)


# --------------------------------------------------------------------------- #
# Arm x benchmark rules; scratch arms never touch the filesystem
# --------------------------------------------------------------------------- #
def test_arm_configs_match_allowed():
    names = sorted(p.stem for p in (REPO_ROOT / "configs" / "model").glob("*.yaml"))
    assert names == sorted(ALLOWED)
    for n in names:
        cfg = _arm(n)
        assert cfg.name == n and "weights_sha256" in cfg


def test_arm_benchmark_validation(fake_imagenet):
    for arm in ("r18_imagenet", "mbv2_imagenet"):
        with pytest.raises(ValueError, match="core50_nic"):
            build_model(_arm(arm), 10, "clear10", weights_dir=fake_imagenet)
    with pytest.raises(ValueError, match="valid arms"):
        build_model(OmegaConf.create({"name": "vit_like", "arch": "resnet18"}), 10, "clear10")
    for bm in ("imagenet100", "clear10", "domainnet", "core50_nic", "synthetic"):
        assert build_model(_arm("r18_scratch"), 10, bm, weights_dir=None) is not None
    assert "imagenet100" in ALLOWED["r18_places_all"]


def test_arm_specs_canonical(tmp_path):
    for n in ALLOWED:
        cfg = _arm(n)
        assert (cfg.arch, cfg.init, cfg.trainable) == ARM_SPECS[n]
    for override in ({"init": "imagenet"}, {"trainable": "last_block"}, {"arch": "resnet50"}):
        bad = OmegaConf.merge(_arm("r18_scratch"), override)
        key = next(iter(override))
        with pytest.raises(ValueError, match=f"model.{key}=.*requires"):
            build_model(bad, 10, "synthetic", weights_dir=tmp_path)
        with pytest.raises(ValueError, match="ARM_SPECS"):
            validate_model_config(bad, "synthetic", tmp_path)
    with pytest.raises(ValueError, match="trainable"):
        validate_model_config(OmegaConf.merge(_arm("r18_places"), {"trainable": "all"}),
                              "clear10", tmp_path)


def test_validate_model_config(tmp_path, places_dir):
    wdir, _ = places_dir
    validate_model_config(_arm("r18_scratch"), "clear10", None)  # scratch: no weights_dir
    validate_model_config(_arm("r18_places"), "clear10", wdir)
    validate_model_config(_arm("r18_imagenet"), "core50_nic", tmp_path / "x")  # fetched later
    with pytest.raises(FileNotFoundError, match="download_data.sh"):
        validate_model_config(_arm("r50_places"), "clear10", wdir)  # r50 file absent
    with pytest.raises(ValueError, match="core50_nic"):
        validate_model_config(_arm("mbv2_imagenet"), "domainnet", tmp_path)
    assert not (tmp_path / "x").exists()


def test_scratch_arms_ignore_weights_dir(tmp_path):
    missing = tmp_path / "does_not_exist"
    for arm in ("r18_scratch", "r50_scratch"):
        info: dict = {}
        m = build_model(_arm(arm), 3, "synthetic", weights_dir=missing, info=info)
        assert info["params"]["trainable"] == info["params"]["total"]
        assert info["weights"] is None
        assert m(torch.zeros(1, 3, 64, 64)).shape == (1, 3)
    assert not missing.exists()
    with pytest.raises(ValueError, match="scratch only"):
        build_model(OmegaConf.merge(_arm("r18_places"), {"cifar_stem": True}), 3, "synthetic",
                    weights_dir=tmp_path)


# --------------------------------------------------------------------------- #
# Freezing holds through Avalanche training, for every method
# --------------------------------------------------------------------------- #
def _bench(load_benchmark_cfg):
    from clns.data import build_benchmark

    cfg = load_benchmark_cfg("synthetic", max_experiences=2, train_per_class=8, test_per_class=4)
    return build_benchmark(cfg, "unused", 0)


@pytest.mark.parametrize("method", list(METHODS))
def test_frozen_part_unchanged_after_training(method, places_dir, load_benchmark_cfg):
    from avalanche.training.plugins import EvaluationPlugin, LwFPlugin

    wdir, sha = places_dir
    bundle = _bench(load_benchmark_cfg)
    model = build_model(_places_cfg(sha), bundle.num_classes, "synthetic", weights_dir=wdir)
    frozen = {n: t.detach().clone() for n, t in model.state_dict().items()
              if not n.startswith(("layer4.", "fc."))}
    trainable_before = {n: p.detach().clone() for n, p in model.named_parameters()
                        if p.requires_grad}
    assert len(frozen) > 0 and all(n.startswith(("layer4.", "fc.")) for n in trainable_before)

    strat = build_strategy(
        OmegaConf.create(METHODS[method]), _arm("r18_places"), TRAIN, model, bundle.num_classes,
        EvaluationPlugin(loggers=[]), [], "cpu", scenario=bundle.scenario,
        n_train_experiences=2,
    )
    for exp in bundle.benchmark.train_stream:
        strat.train(exp, eval_streams=[])
        if method == "lwf":
            lwf = next(p for p in strat.plugins if isinstance(p, LwFPlugin))
            teacher = lwf.lwf.prev_model
            assert teacher is not None and teacher.training is False
            assert all(not m.training for m in teacher.modules())
            assert all(not p.requires_grad for p in teacher.parameters())

    after = strat.model.state_dict()
    for n, t in frozen.items():  # frozen params AND FrozenBatchNorm2d buffers
        assert torch.equal(after[n], t), n
    changed = [n for n, p in strat.model.named_parameters()
               if p.requires_grad and not torch.equal(p, trainable_before[n])]
    assert changed  # the last block did train


def test_lwf_teacher_eval_after_resume(load_benchmark_cfg):
    """A teacher restored in train mode (e.g. from a checkpoint) is put back in eval mode
    before it is used in the next experience."""
    from avalanche.core import SupervisedPlugin
    from avalanche.training.plugins import EvaluationPlugin, LwFPlugin

    bundle = _bench(load_benchmark_cfg)
    model = build_model(OmegaConf.merge(_arm("r18_scratch"), {"cifar_stem": True}),
                        bundle.num_classes, "synthetic")
    seen: list[bool] = []

    class Probe(SupervisedPlugin):
        def before_training_iteration(self, strategy, **kwargs):
            lwf = next(p for p in strategy.plugins if isinstance(p, LwFPlugin))
            if lwf.lwf.prev_model is not None:
                seen.append(lwf.lwf.prev_model.training)

    strat = build_strategy(
        OmegaConf.create(METHODS["lwf"]), _arm("r18_scratch"), TRAIN, model,
        bundle.num_classes, EvaluationPlugin(loggers=[]), [Probe()], "cpu",
    )
    exp0, exp1 = bundle.benchmark.train_stream
    strat.train(exp0, eval_streams=[])
    lwf = next(p for p in strat.plugins if isinstance(p, LwFPlugin))
    lwf.lwf.prev_model.train()  # simulate an unpickled train-mode teacher
    strat.train(exp1, eval_streams=[])
    assert seen and not any(seen)


# --------------------------------------------------------------------------- #
# ViT / L2P / timm are gone
# --------------------------------------------------------------------------- #
LEGACY_RE = re.compile(r"\bvit\b|vit_|l2p|timm", re.IGNORECASE)
# Only this file (it holds the pattern itself) is exempt.
LEGACY_EXCLUDE = {
    Path("tests/test_freezing_pretrained.py"),
}


def test_no_vit_l2p_timm_references():
    hits = []
    roots = [REPO_ROOT / "src", REPO_ROOT / "configs", REPO_ROOT / "tests"]
    files = [p for r in roots for p in r.rglob("*")
             if p.suffix in {".py", ".yaml", ".toml", ".txt"}]
    files += [REPO_ROOT / "pyproject.toml", REPO_ROOT / "requirements.txt"]
    for p in files:
        rel = p.relative_to(REPO_ROOT)
        if rel in LEGACY_EXCLUDE or "egg-info" in str(rel) or "__pycache__" in rel.parts:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if LEGACY_RE.search(line):
                hits.append(f"{rel}:{i}: {line.strip()}")
    assert not hits, "\n".join(hits)
