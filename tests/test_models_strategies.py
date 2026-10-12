import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("avalanche")
from omegaconf import OmegaConf  # noqa: E402

from clns.models import build_model  # noqa: E402
from clns.strategies import build_strategy  # noqa: E402

TRAIN = OmegaConf.create(
    {
        "epochs": 1,
        "mb_size": 8,
        "eval_mb_size": 8,
        "lr": 0.01,
        "optimizer": "sgd",
        "momentum": 0.9,
        "weight_decay": 0.0,
        "num_workers": 0,
    }
)
METHODS = {
    "naive": {"name": "naive"},
    "ewc": {"name": "ewc", "ewc_lambda": 10, "mode": "separate", "decay_factor": None},
    "lwf": {"name": "lwf", "alpha": 1.0, "temperature": 2.0},
    "er": {"name": "er", "mem_size": 20, "storage": "reservoir", "batch_size_mem": None},
    "derpp": {"name": "derpp", "mem_size": 20, "batch_size_mem": 4, "alpha": 0.1, "beta": 0.5},
}
RESNET = OmegaConf.create(
    {"name": "r18_scratch", "arch": "resnet18", "init": "scratch", "trainable": "all",
     "cifar_stem": False}
)


def _evaluator():
    from avalanche.training.plugins import EvaluationPlugin

    return EvaluationPlugin(loggers=[])


def _build(method_cfg, model_cfg=RESNET, model=None, num_classes=10, **kw):
    if model is None:
        model = build_model(model_cfg, num_classes, "synthetic")
    return build_strategy(
        OmegaConf.create(method_cfg), model_cfg, TRAIN, model, num_classes, _evaluator(), [], "cpu",
        **kw,
    )


@pytest.mark.parametrize("name", list(METHODS))
def test_build_methods(name):
    assert _build(METHODS[name]) is not None


def test_resnet18_shapes():
    m = build_model(RESNET, 7, "synthetic")
    assert m(torch.zeros(2, 3, 64, 64)).shape == (2, 7)
    m2 = build_model(OmegaConf.merge(RESNET, {"cifar_stem": True}), 5, "synthetic")
    assert isinstance(m2.maxpool, torch.nn.Identity)
    assert m2(torch.zeros(2, 3, 32, 32)).shape == (2, 5)


def test_unknown_model():
    with pytest.raises(ValueError, match="r18_scratch"):
        build_model(OmegaConf.create({"name": "foo", "arch": "resnet18"}), 10, "synthetic")


def test_invalid_combos():
    with pytest.raises(ValueError, match="Unknown method"):
        _build({"name": "packnet"})
    with pytest.raises(ValueError, match="Unknown method"):
        _build({"name": "bogus"})
    with pytest.raises(ValueError, match="decay_factor"):
        _build({"name": "ewc", "ewc_lambda": 1, "mode": "online", "decay_factor": None})


def test_unknown_optimizer():
    bad = TRAIN.copy()
    bad.optimizer = "rmsprop"
    model = build_model(RESNET, 10, "synthetic")
    with pytest.raises(ValueError, match="optimizer"):
        build_strategy(OmegaConf.create(METHODS["naive"]), RESNET, bad, model, 10, _evaluator(), [],
                       "cpu")


def test_er_buffer_bounded():
    from avalanche.benchmarks import tensors_benchmark
    from avalanche.benchmarks.utils import AvalancheDataset
    from avalanche.training.plugins import ReplayPlugin

    g = torch.Generator().manual_seed(0)

    def exp(lo, hi):
        x = torch.randn(60, 8, generator=g)
        y = torch.randint(lo, hi, (60,), generator=g)
        return [x, y]

    bm = tensors_benchmark(
        train_tensors=[exp(0, 2), exp(2, 4)],
        test_tensors=[exp(0, 2), exp(2, 4)],
        task_labels=[0, 0],
    )
    model = torch.nn.Linear(8, 4)
    strat = build_strategy(
        OmegaConf.create(METHODS["er"]), RESNET, TRAIN, model, 4, _evaluator(), [], "cpu"
    )
    replay = next(p for p in strat.plugins if isinstance(p, ReplayPlugin))
    for e in bm.train_stream:
        strat.train(e)
        buf = replay.storage_policy.buffer
        assert isinstance(buf, AvalancheDataset)
        assert len(buf) <= 20
    assert len(replay.storage_policy.buffer) > 0


def test_resnet18_deterministic_pooling():
    from clns.models import GlobalMeanPool2d

    m = build_model(RESNET, 3, "synthetic")
    assert isinstance(m.avgpool, GlobalMeanPool2d)
    x = torch.randn(2, 512, 4, 4)
    ref = torch.nn.AdaptiveAvgPool2d((1, 1))(x)
    assert torch.allclose(m.avgpool(x), ref, atol=1e-6)


def test_ewc_separate_guard():
    cfg = dict(METHODS["ewc"], max_separate_experiences=20)
    assert _build(cfg, n_train_experiences=20) is not None
    with pytest.raises(ValueError, match="mode=online"):
        _build(cfg, n_train_experiences=79)
    online = {"name": "ewc", "ewc_lambda": 10, "mode": "online", "decay_factor": 0.9}
    assert _build(online, n_train_experiences=79) is not None


def test_ewc_config_has_no_keep_importance_data(repo_root):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(repo_root / "configs" / "method" / "ewc.yaml")
    assert "keep_importance_data" not in cfg
    assert cfg.max_separate_experiences == 20
    # spec: EWC online + decay_factor on every benchmark (default config)
    assert cfg.mode == "online" and cfg.decay_factor is not None


def test_deterministic_modes():
    from clns.utils.seed import resolve_deterministic, seed_everything

    assert resolve_deterministic(True) == "strict"
    assert resolve_deterministic(False) == "off"
    assert resolve_deterministic("warn") == "warn"
    with pytest.raises(ValueError):
        resolve_deterministic("sometimes")
    try:
        assert seed_everything(0, "strict") == "strict"
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert seed_everything(0, "warn") == "warn"
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        seed_everything(0, "off")
