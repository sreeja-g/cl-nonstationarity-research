from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def load_benchmark_cfg():
    """Load configs/benchmark/<name>.yaml with dotlist-style overrides."""
    from omegaconf import OmegaConf

    def _load(name: str, /, **overrides):  # positional-only: allows a `name=` override
        cfg = OmegaConf.load(REPO_ROOT / "configs" / "benchmark" / f"{name}.yaml")
        return OmegaConf.merge(cfg, OmegaConf.create(overrides))

    return _load


@pytest.fixture(scope="session")
def fake_places():
    """Writer of fake CSAIL Places365 checkpoints (no network).

    ``fake_places(path, arch="resnet18", drop=None, legacy=True) -> sha256`` writes a dict with
    ``epoch``/``arch``/``state_dict``/``best_prec1``, ``module.``-prefixed keys, a 365-way fc and
    non-trivial BN statistics. ``legacy=True`` reproduces the real 2017 release: legacy (non-zip)
    ``torch.save`` format and no ``num_batches_tracked`` buffers. ``drop``: a key (without
    ``module.``) to remove, e.g. to simulate genuinely missing weights.
    """
    import torch
    import torchvision.models as tvm

    from clns.models.pretrained import sha256_binary

    def _write(path, arch="resnet18", drop=None, legacy=True):
        g = torch.Generator().manual_seed(123)
        net = getattr(tvm, arch)(weights=None, num_classes=365)
        for name, buf in net.named_buffers():
            if name.endswith("running_mean"):
                buf.copy_(torch.randn(buf.shape, generator=g) * 0.1)
            elif name.endswith("running_var"):
                buf.copy_(torch.rand(buf.shape, generator=g) + 0.5)
        state = {f"module.{k}": v for k, v in net.state_dict().items()
                 if not (legacy and k.endswith("num_batches_tracked"))}
        if drop is not None:
            state.pop(f"module.{drop}")
        ckpt = {"epoch": 90, "arch": arch, "state_dict": state, "best_prec1": 50.0}
        torch.save(ckpt, path, _use_new_zipfile_serialization=not legacy)
        return sha256_binary(path)

    return _write
