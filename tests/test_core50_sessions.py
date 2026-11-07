"""CORe50 whole-session hold-out validation (fake filelists, no images), CLEAR split_seed guard
and the ``finalize`` empty-validation guard."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("avalanche")
PILImage = pytest.importorskip("PIL.Image")

from torch.utils.data import TensorDataset  # noqa: E402

from clns.data import build_benchmark  # noqa: E402
from clns.data.common import DONE_MARKER, finalize, make_cls_dataset  # noqa: E402
from clns.data.core50 import core50_session, holdout_sessions  # noqa: E402


# --------------------------------------------------------------------------- #
# session parsing
# --------------------------------------------------------------------------- #
def test_core50_session_parsing():
    assert core50_session("s1/o3/C_01_03_000.png") == 1
    assert core50_session("s11/o50/C_11_50_199.png") == 11
    assert core50_session("s4\\o2\\C_04_02_010.png") == 4  # backslashes normalised


@pytest.mark.parametrize(
    "bad", ["s1/o3/C_02_03_000.png", "o3/C_01_03_000.png", "s1/o3/img.png", "s1/o3/D_01_03_0.png"]
)
def test_core50_session_mismatch_or_format_raises(bad):
    with pytest.raises(ValueError, match="CORe50"):
        core50_session(bad)


# --------------------------------------------------------------------------- #
# holdout_sessions on fake datasets
# --------------------------------------------------------------------------- #
def _ds(ids, labels):
    return make_cls_dataset(
        TensorDataset(torch.tensor(ids, dtype=torch.float32).view(-1, 1)), targets=labels
    )


def _ids(ds):
    return [int(ds[i][0].item()) for i in range(len(ds))]


def _p(s, o, f):
    return f"s{s}/o{o}/C_{s:02d}_{o:02d}_{f:03d}.png"


def test_holdout_sessions_splits_frames():
    paths = [
        [_p(1, 1, 0), _p(2, 1, 0), _p(1, 2, 0), _p(4, 2, 0)],
        [_p(2, 3, 0), _p(1, 3, 1)],
    ]
    train = [_ds([0, 1, 2, 3], [0, 0, 1, 1]), _ds([10, 11], [2, 2])]
    kept, held = holdout_sessions(train, paths, [1])
    assert [_ids(d) for d in kept] == [[1, 3], [10]]
    assert [_ids(d) for d in held] == [[0, 2], [11]]
    # several sessions, and a held-out batch that is empty is returned as an empty dataset
    kept, held = holdout_sessions(train, paths, [2, 4])
    assert [_ids(d) for d in kept] == [[0, 2], [11]]
    assert [_ids(d) for d in held] == [[1, 3], [10]]


def test_holdout_sessions_errors():
    paths = [[_p(1, 1, 0), _p(2, 1, 0)], [_p(1, 2, 0)]]
    train = [_ds([0, 1], [0, 0]), _ds([5], [1])]
    with pytest.raises(ValueError, match="not in the training sessions"):
        holdout_sessions(train, paths, [3])  # a test session
    with pytest.raises(ValueError, match="no frame of these sessions"):
        holdout_sessions(train, paths, [11])  # nothing held out
    with pytest.raises(ValueError, match="without frames"):
        holdout_sessions(train, paths, [1])  # batch 1 only has session 1
    with pytest.raises(ValueError, match="filelist entries"):
        holdout_sessions(train, [paths[0], []], [1])
    # batch 1 empties, but only batch 0 is used: allowed, stream length is kept
    kept, held = holdout_sessions(train, paths, [1], n_used=1)
    assert len(kept) == 2 and len(kept[1]) == 0 and _ids(held[1]) == [5]


# --------------------------------------------------------------------------- #
# end to end with fake filelists (images never read)
# --------------------------------------------------------------------------- #
def _write_core50(root, objs=((0, 1), (1, 2), (2, 3))):
    """3 train batches (classes ``objs``); sessions 1/2/4, 1/2/4, 1/4 (test: 3, 7).

    Every frame is also written as a 1x1 PNG whose pixel is (session, object, frame): the
    identity of any sample can be read back from the loaded image.
    """
    base = root / "NIC_v2_79" / "run0"
    base.mkdir(parents=True)
    img_root = root / "core50_128x128"

    def lines(sessions, objs, n=2):
        out = []
        for s in sessions:
            for o in objs:
                for f in range(n):
                    rel = _p(s, o + 1, f)
                    (img_root / rel).parent.mkdir(parents=True, exist_ok=True)
                    PILImage.new("RGB", (1, 1), (s, o, f)).save(img_root / rel)
                    out.append(f"{rel} {o}")
        return out

    batches = [
        lines([1, 2, 4], objs[0]),
        lines([1, 2, 4], objs[1]),
        lines([1, 4], objs[2]),
    ]
    for b, ls in enumerate(batches):
        (base / f"train_batch_{b:02d}_filelist.txt").write_text("\n".join(ls) + "\n")
    test = lines([3, 7], [0, 1, 2, 3])
    (base / "test_filelist.txt").write_text("\n".join(test) + "\n")
    (root / DONE_MARKER).write_text("")
    return batches


@pytest.fixture
def fake_core50_factory(tmp_path, monkeypatch):
    import avalanche.benchmarks.classic.core50 as c50

    class _FakeData:
        def __init__(self, root, mini=False):
            self.root = root

    monkeypatch.setattr(c50, "CORe50Dataset", _FakeData)
    monkeypatch.setitem(c50.nbatch, "nicv2_79", 3)

    def make(objs=((0, 1), (1, 2), (2, 3))):
        root = tmp_path / "core50"
        root.mkdir()
        return tmp_path, _write_core50(root, objs)

    return make


@pytest.fixture
def fake_core50(fake_core50_factory):
    return fake_core50_factory()


def _cfg(load_benchmark_cfg, **kw):
    base = dict(image_size=16, train_transform="none", eval_transform="none")
    base.update(kw)
    return load_benchmark_cfg("core50_nic", **base)


def test_core50_session_holdout_end_to_end(fake_core50, load_benchmark_cfg):
    data_dir, batches = fake_core50
    bundle = build_benchmark(_cfg(load_benchmark_cfg, val_sessions=[1]), str(data_dir), seed=0)
    bm, meta = bundle.benchmark, bundle.meta
    n_s1 = [sum(x.startswith("s1/") for x in b) for b in batches]
    assert meta["val_sessions"] == [1]
    assert meta["heldout_frames_per_batch"] == n_s1 == [4, 4, 4]
    assert meta["train_sizes"] == [len(b) - k for b, k in zip(batches, n_s1)] == [8, 8, 4]
    # validation: first-appearance groups {0,1}, {2}, {3}; frames come from all batches
    assert meta["valid_exp_to_train_exp"] == [0, 1, 2]
    assert meta["test_exp_to_train_exp"] == [0, 1, 2]
    assert len(bm.valid_stream) == 3 and all(len(e.dataset) > 0 for e in bm.valid_stream)
    assert meta["valid_sizes"] == [6, 4, 2]  # classes {0,1} / {2} / {3}, 2 frames per object
    assert sum(meta["valid_sizes"]) == sum(n_s1)
    # session 1 contributes all 4 classes in valid; test stays sessions 3/7
    v_classes = {c for e in bm.valid_stream for c in e.classes_in_this_experience}
    assert sorted(v_classes) == [0, 1, 2, 3]
    assert sum(meta["test_sizes"]) == 2 * 2 * 4
    # frame-level disjointness of valid from the kept training frames
    from clns.data.common import assert_disjoint
    from clns.data.core50 import _read_train_paths

    paths = _read_train_paths(data_dir / "core50", "nicv2_79", 0)
    kept = [p for b in paths for p in b if not p.startswith("s1/")]
    held = [p for b in paths for p in b if p.startswith("s1/")]
    assert_disjoint(kept, held)
    assert len(held) == sum(meta["valid_sizes"])
    # sample identities: pixel (session, object, frame) -- valid = session 1 only, train never
    for e in bm.valid_stream:
        assert {e.dataset[i][0].getpixel((0, 0))[0] for i in range(len(e.dataset))} == {1}
    for e in bm.train_stream:
        assert 1 not in {e.dataset[i][0].getpixel((0, 0))[0] for i in range(len(e.dataset))}
    for e in bm.test_stream:
        assert {e.dataset[i][0].getpixel((0, 0))[0] for i in range(len(e.dataset))} == {3, 7}


def test_core50_batch_without_new_class_gives_non_identity_map(
    fake_core50_factory, load_benchmark_cfg
):
    data_dir, _ = fake_core50_factory(objs=((0, 1), (0, 1), (2, 3)))
    bundle = build_benchmark(_cfg(load_benchmark_cfg, val_sessions=[1]), str(data_dir), seed=0)
    assert bundle.meta["test_exp_to_train_exp"] == [0, 2]
    assert bundle.meta["valid_exp_to_train_exp"] == [0, 2]
    assert len(bundle.benchmark.valid_stream) == 2
    from omegaconf import OmegaConf

    from clns.evaluation.evaluator import build_evaluator

    cfg = OmegaConf.create({"train": {"eval_mask_unseen": False, "eval_every": 0}})
    _, plugins = build_evaluator(
        cfg, [], data_dir / "run", scenario="class_il", n_test=2,
        test_exp_to_train_exp=bundle.meta["test_exp_to_train_exp"], n_valid=2,
        valid_exp_to_train_exp=bundle.meta["valid_exp_to_train_exp"],
    )
    assert len(plugins) == 2  # test + valid matrix plugins


def test_build_evaluator_valid_mapping_length_mismatch(tmp_path):
    from omegaconf import OmegaConf

    from clns.evaluation.evaluator import build_evaluator

    cfg = OmegaConf.create({"train": {"eval_mask_unseen": False, "eval_every": 0}})
    with pytest.raises(ValueError, match="valid_exp_to_train_exp"):
        build_evaluator(
            cfg, [], tmp_path, scenario="domain_il", n_test=2, n_valid=3,
            valid_exp_to_train_exp=[0, 1],
        )


@pytest.mark.parametrize("bad", [1, "1", [True], [1.5], ["1"], []])
def test_core50_val_sessions_input_validation(load_benchmark_cfg, bad):
    with pytest.raises(ValueError, match="val_sessions"):
        build_benchmark(_cfg(load_benchmark_cfg, val_sessions=bad), "/nonexistent", 0)


def test_core50_session_check_and_cli(fake_core50, capsys):
    from clns.data.core50 import _read_train_paths, core50_session_check, main

    data_dir, _ = fake_core50
    paths = _read_train_paths(data_dir / "core50", "nicv2_79", 0)
    rows = core50_session_check(paths, [1], n_used=2)
    assert [r["batch"] for r in rows] == [0, 1]
    assert rows[0]["sessions"] == {1: 4, 2: 4, 4: 4} and rows[0]["n_heldout"] == 4
    assert not any(r["empties_batch"] for r in rows)
    assert core50_session_check(paths, [1, 2, 4])[0]["empties_batch"]
    argv = ["--check-sessions", "--root", str(data_dir / "core50"), "--n-used", "3"]
    assert main(argv + ["--val-sessions", "1"]) == 0
    assert "batches emptied: none" in capsys.readouterr().out
    assert main(argv + ["--val-sessions", "1", "2", "4"]) == 1


def test_core50_without_val_sessions_has_no_valid_stream(fake_core50, load_benchmark_cfg):
    data_dir, batches = fake_core50
    bundle = build_benchmark(_cfg(load_benchmark_cfg), str(data_dir), seed=0)
    assert not hasattr(bundle.benchmark, "valid_stream") or bundle.benchmark.valid_stream is None
    assert "val_sessions" not in bundle.meta and "valid_exp_to_train_exp" not in bundle.meta
    assert bundle.meta["train_sizes"] == [len(b) for b in batches]


def test_core50_val_sessions_with_max_experiences(fake_core50, load_benchmark_cfg):
    data_dir, _ = fake_core50
    bundle = build_benchmark(
        _cfg(load_benchmark_cfg, val_sessions=[1], max_experiences=2), str(data_dir), seed=0
    )
    assert bundle.meta["n_train_experiences"] == 2
    assert bundle.meta["valid_exp_to_train_exp"] == [0, 1]


def test_core50_val_fraction_and_bad_sessions_raise(fake_core50, load_benchmark_cfg):
    data_dir, _ = fake_core50
    with pytest.raises(ValueError, match="val_sessions"):
        build_benchmark(_cfg(load_benchmark_cfg, val_fraction=0.2), str(data_dir), 0)
    with pytest.raises(ValueError, match="training sessions"):
        build_benchmark(_cfg(load_benchmark_cfg, val_sessions=[3]), str(data_dir), 0)
    with pytest.raises(ValueError, match="empty"):
        build_benchmark(_cfg(load_benchmark_cfg, val_sessions=[]), str(data_dir), 0)


# --------------------------------------------------------------------------- #
# CLEAR split_seed guard, finalize guards
# --------------------------------------------------------------------------- #
def test_clear10_split_seed_nonzero_raises_before_data(load_benchmark_cfg):
    with pytest.raises(ValueError, match="no effect"):
        build_benchmark(load_benchmark_cfg("clear10", split_seed=1), "/nonexistent", 0)


def _fin_cfg(**kw):
    from omegaconf import OmegaConf

    return OmegaConf.create(dict(max_experiences=None, subsample_per_class=None, **kw))


def test_finalize_valid_list_and_guards():
    tr = [_ds([0, 1, 2, 3], [0, 0, 1, 1])]
    te = [_ds([4, 5], [0, 1])]
    va = [_ds([6, 7], [0, 1])]
    meta: dict = {}
    sc = finalize(tr, te, _fin_cfg(), 0, meta, valid_list=va)
    assert meta["valid_sizes"] == [2] and meta["valid_exp_to_train_exp"] == [0]
    assert len(sc.valid_stream) == 1
    with pytest.raises(ValueError, match="val_fraction"):
        finalize(tr, te, _fin_cfg(val_fraction=0.2), 0, {}, valid_list=va)
    with pytest.raises(ValueError, match="empty"):
        finalize(tr, te, _fin_cfg(), 0, {}, valid_list=[va[0].subset([])])
    # val_fraction carve that leaves an experience without validation samples
    with pytest.raises(ValueError, match="without validation samples"):
        finalize(tr, te, _fin_cfg(val_fraction=0.01), 0, {})


# --------------------------------------------------------------------------- #
# CLEAR-10: Avalanche 0.6.0 gives every bucket the images of ALL buckets; we build our own
# --------------------------------------------------------------------------- #
def _fake_clear(root, n_buckets=3, n_classes=2):
    """Fake released CLEAR-10 folders: bucket b (key b+1) has b+1 images per class (train) and
    1 image per class (test)."""
    import json

    names = [f"c{k}" for k in range(n_classes)]
    for split in ("train", "test"):
        folder = root / split
        labeled = {}
        for b in range(n_buckets):
            labeled[str(b + 1)] = {}
            for k, name in enumerate(names):
                n = (b + 1) if split == "train" else 1
                meta = {}
                for i in range(n):
                    rel = f"images/{b + 1}/{name}/{split}_{i}.png"
                    (folder / rel).parent.mkdir(parents=True, exist_ok=True)
                    PILImage.new("RGB", (2, 2), (b, k, i)).save(folder / rel)
                    meta[str(i)] = {"IMG_PATH": rel}
                mpath = f"labeled_metadata/{b + 1}_{name}.json"
                (folder / mpath).parent.mkdir(parents=True, exist_ok=True)
                (folder / mpath).write_text(json.dumps(meta))
                labeled[str(b + 1)][name] = mpath
        (folder / "labeled_metadata.json").write_text(json.dumps(labeled))
        (folder / "class_names.txt").write_text("\n".join(names))
    (root / DONE_MARKER).write_text("")


def test_clear10_buckets_are_disjoint_and_not_shared(tmp_path, load_benchmark_cfg):
    root = tmp_path / "clear10"
    _fake_clear(root)
    cfg = load_benchmark_cfg(
        "clear10", n_experiences=3, train_transform="none", eval_transform="none"
    )
    bundle = build_benchmark(cfg, str(tmp_path), seed=0)
    assert bundle.meta["bucket_ids"] == [1, 2, 3]
    assert bundle.meta["train_sizes"] == [2, 4, 6]  # differ per bucket (Avalanche: all 12)
    assert bundle.meta["test_sizes"] == [2, 2, 2]
    for b, e in enumerate(bundle.benchmark.train_stream):
        pix = {e.dataset[i][0].getpixel((0, 0))[0] for i in range(len(e.dataset))}
        assert pix == {b}  # only this bucket's images
    for b, e in enumerate(bundle.benchmark.test_stream):
        pix = {e.dataset[i][0].getpixel((0, 0))[0] for i in range(len(e.dataset))}
        assert pix == {b}


def test_clear10_bucket_checks_raise(tmp_path):
    from clns.data.clear10 import check_clear_buckets, read_clear_buckets

    root = tmp_path / "clear10"
    _fake_clear(root)
    _, _, tr = read_clear_buckets(root, "train")
    _, _, te = read_clear_buckets(root, "test")
    with pytest.raises(ValueError, match="expected 4"):
        check_clear_buckets(root, tr, te, 4)
    with pytest.raises(ValueError, match="leakage"):  # bucket reused as test
        check_clear_buckets(root, tr, [tr[0], te[1], te[2]], 3)
    shared = [tr[0] + tr[1], tr[1], tr[2]]  # overlapping buckets (the Avalanche bug's shape)
    with pytest.raises(ValueError, match="leakage|twice|overlap"):
        check_clear_buckets(root, shared, te, 3)
    check_clear_buckets(root, tr, te, 3)
