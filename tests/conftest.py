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

    def _load(name: str, **overrides):
        cfg = OmegaConf.load(REPO_ROOT / "configs" / "benchmark" / f"{name}.yaml")
        return OmegaConf.merge(cfg, OmegaConf.create(overrides))

    return _load
