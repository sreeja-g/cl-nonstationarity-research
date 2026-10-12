"""CLI: measure inter-experience shift (marginal MMD^2_u) of a benchmark's train stream.

Every experience is embedded with ONE frozen reference encoder (non-CLIP; RESEARCH_LOG 7.3 /
7.4 Q10) and unbiased RBF-MMD^2 is computed on L2-NORMALISED features with a FIXED set of
bandwidths per encoder (:data:`ENCODER_BANDWIDTHS`, kernels summed). The same bandwidths are used
for every benchmark and seed, so the kernel does not adapt to the data it measures (a pooled
median heuristic would rescale each benchmark/order differently). Both encoders output
post-ReLU global-average-pooled activations (all entries >= 0), so after L2 normalisation every
cosine is >= 0 and squared distances ``2 - 2 cos`` lie in [0, 2]; sigmas 0.5 / 1 / 2 cover that
range. As a sanity check the median pairwise squared distance of the pooled selected features is
logged and recorded per benchmark/seed (``median_sq_dist`` in shift_meta.json).

Encoders:
  * ``tv:resnet50``   -- torchvision ``ResNet50_Weights.IMAGENET1K_V2`` (supervised ImageNet),
    its own ``weights.transforms()``; fetched by torchvision into ``$TORCH_HOME`` (the file's
    sha256 is recorded as ``encoder_weights_sha256`` when the file is found there).
  * ``dino:resnet50`` -- official self-supervised DINO ResNet-50 backbone
    ``<weights-dir>/dino_resnet50_pretrain.pth`` (``scripts/download_data.sh dino``; no torch.hub),
    sha256-checked against ``--dino-sha256`` / :data:`DINO_RESNET50_SHA256`, loaded with
    ``torch.load(weights_only=True)`` into torchvision ``resnet50(weights=None)``; ``fc`` is
    replaced by Identity. Preprocessing: Resize(256) + CenterCrop(224) + ImageNet normalisation.
CLIP encoders are rejected (CLEAR labels were built with CLIP).

Preprocessing (RESEARCH_LOG 7.3, "same encoder for all benchmarks"): the benchmark's own
transforms are REMOVED and the RAW items are passed through the ENCODER's eval transform.

Sampling: a fixed ``n = min(--n-per-experience, smallest train experience)`` images per
experience, selected PER CLASS by :func:`clns.shift.feature_cache.select_ids` (``ceil(n/k)`` per
class of the experience by an order-independent hash of a stable per-image id, then truncated to
``n``), so a class is represented by the same images in every seed's order whenever ``n`` and
its experience's class count ``k`` are unchanged (ImageNet-100 tasks, DomainNet / CLEAR domains;
CORe50 NIC batches change with the run, so there only part of the images recur). Features are
cached PER IMAGE (:class:`~clns.shift.feature_cache.FeatureCache`, default
``<out>/feature_cache``; keyed by encoder, encoder weights hash, transform and pipeline version),
so measuring another seed only embeds images not embedded before.

Measure once per training seed: run ``--seed k`` for every k in the run seeds (e.g. 0, 1, 2);
``--seed k`` builds the benchmark exactly as ``clns.train seed=k`` does, because the task order
follows the seed (ImageNet-100 class order, DomainNet domain order, CORe50 run). CLEAR-10's order
is seed-independent, so one seed suffices there (measuring every seed is harmless: identical
task_order_sha256, and the cache makes it cheap). Outputs go to ``<out>/<benchmark>/seed<k>/``:
``shift_matrix.npz``,
``shift_pairs.csv`` (long format) and ``shift_meta.json`` (seed, task_order, task_order_sha256,
class order, split hashes, T, encoder, bandwidths, n; read by
``clns.analysis.correlation_study``, which pairs a run only with a result whose class order,
split hashes -- including ``task_order_sha256`` -- and T match the run's ``benchmark_meta``).

The measure is MARGINAL MMD on P(X) and is NOT comparable across benchmarks (RESEARCH_LOG 7.3).
TODO(RESEARCH_LOG 7.3): class-conditional MMD is deferred.

Example:
    python -m clns.shift.measure_stream --benchmark-config configs/benchmark/synthetic.yaml \
        --encoder tv:resnet50 --n-per-experience 200 --seed 0 --out /tmp/shift
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from clns.shift.feature_cache import FeatureCache, sample_ids, select_ids
from clns.shift.metrics import mmd2_unbiased, mmd_permutation_test, mmd_subsample_ci

log = logging.getLogger(__name__)

MEASURE = "marginal_mmd"
MEASURE_NOTE = (
    "Unbiased RBF MMD^2 on the MARGINAL P(X) of L2-normalised frozen-encoder features, fixed "
    "bandwidth set per encoder (kernels summed; same for every benchmark and seed). Not "
    "comparable across benchmarks: it measures a label-block change in class-IL benchmarks but "
    "a P(X|Y) change in domain-IL benchmarks (RESEARCH_LOG 7.3). Class-conditional MMD is not "
    "yet implemented. CI = m-out-of-n subsampling without replacement (conservative, not "
    "rescaled); null95 = 95th percentile of the permutation null."
)
# Bumped whenever the feature pipeline changes, so stale caches are never reused.
_PIPELINE_VERSION = "raw_items+encoder_transform/per_image/v3"

# One fixed bandwidth set per encoder, applied to L2-normalised features (see module docstring).
ENCODER_BANDWIDTHS: dict[str, tuple[float, ...]] = {
    "tv:resnet50": (0.5, 1.0, 2.0),
    "dino:resnet50": (0.5, 1.0, 2.0),
}
DINO_RESNET50_FILE = "dino_resnet50_pretrain.pth"
# TODO(before real runs): pin the sha256 printed by `bash scripts/download_data.sh dino`
# (written to <weights_dir>/dino_resnet50_pretrain.pth.sha256). While None, the actual hash is
# only logged and recorded in shift_meta.json (encoder_weights_sha256).
DINO_RESNET50_SHA256: str | None = None
DINO_PREFIXES = ("module.", "backbone.")


def check_encoder_name(name: str) -> None:
    """Reject CLIP-family encoders (CLEAR labels were built with CLIP; RESEARCH_LOG 7.4 Q10)."""
    if "clip" in name.lower():
        raise ValueError("CLIP encoders are excluded (CLEAR labels were built with CLIP; Q10).")


def encoder_bandwidths(name: str) -> tuple[float, ...]:
    if name not in ENCODER_BANDWIDTHS:
        raise ValueError(
            f"no fixed bandwidth set for encoder {name!r}; known: {sorted(ENCODER_BANDWIDTHS)}"
        )
    return tuple(float(s) for s in ENCODER_BANDWIDTHS[name])


def _imagenet_eval_transform():
    from torchvision import transforms as T

    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_dino_resnet50(path: str | Path, sha256: str | None) -> tuple[torch.nn.Module, str]:
    """torchvision resnet50 with the DINO backbone weights and ``fc = Identity``.

    The checkpoint holds the backbone only: loading must leave exactly ``fc.weight`` /
    ``fc.bias`` missing and nothing unexpected (anything else means a wrong file). Returns
    (model, actual sha256 of the file).
    """
    import torchvision

    from clns.models.pretrained import _torch_load_checkpoint, check_sha256

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"DINO ResNet-50 weights not found at '{path}'. Run `bash scripts/download_data.sh "
            f"dino` (or pass --weights-dir / set CLNS_WEIGHTS_DIR)."
        )
    actual = check_sha256(path, sha256)
    state = _torch_load_checkpoint(path, verified=sha256 is not None)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clean = {}
    for k, v in state.items():
        for pre in DINO_PREFIXES:
            if k.startswith(pre):
                k = k[len(pre):]
        clean[k] = v
    model = torchvision.models.resnet50(weights=None)
    res = model.load_state_dict(clean, strict=False)
    if set(res.missing_keys) != {"fc.weight", "fc.bias"} or res.unexpected_keys:
        raise ValueError(
            f"{path} is not a DINO ResNet-50 backbone checkpoint: missing "
            f"{sorted(res.missing_keys)[:5]}, unexpected {sorted(res.unexpected_keys)[:5]} "
            "(expected exactly fc.weight/fc.bias missing)"
        )
    model.fc = torch.nn.Identity()
    return model, actual


def build_encoder(
    name: str,
    device: torch.device,
    weights_dir: str | Path | None = None,
    dino_sha256: str | None = None,
    info: dict | None = None,
):
    """Return (frozen eval-mode encoder -> (B, D) features, encoder eval transform).

    The transform expects PIL images. ``info`` (optional dict) receives
    ``encoder_weights`` / ``encoder_weights_sha256`` for file-based encoders.
    """
    check_encoder_name(name)
    if name == "tv:resnet50":
        import torchvision

        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        transform = weights.transforms()
        if info is not None:
            info["encoder_weights"] = "torchvision:ResNet50_Weights.IMAGENET1K_V2"
            info["encoder_weights_sha256"] = _torchvision_weights_sha256(weights)
    elif name == "dino:resnet50":
        wdir = Path(weights_dir if weights_dir is not None else _default_weights_dir())
        model, actual = load_dino_resnet50(wdir / DINO_RESNET50_FILE, dino_sha256)
        transform = _imagenet_eval_transform()
        if info is not None:
            info["encoder_weights"] = str(wdir / DINO_RESNET50_FILE)
            info["encoder_weights_sha256"] = actual
    else:
        raise ValueError(f"encoder must be one of {sorted(ENCODER_BANDWIDTHS)}, got {name!r}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, transform


def _default_weights_dir() -> str:
    return os.environ.get("CLNS_WEIGHTS_DIR", "/workspace/weights")


def _torchvision_weights_sha256(weights) -> str | None:
    """sha256 of a torchvision weights file in ``$TORCH_HOME/hub/checkpoints`` (None if absent)."""
    from clns.models.pretrained import sha256_binary

    path = Path(torch.hub.get_dir()) / "checkpoints" / os.path.basename(str(weights.url))
    if not path.is_file():
        log.warning("torchvision weights file %s not found; encoder_weights_sha256 = None", path)
        return None
    return sha256_binary(path)


def median_sq_dist(feats, max_points: int = 2000, seed: int = 0) -> float:
    """Median pairwise squared Euclidean distance of the pooled rows of ``feats`` (a list of
    (n_i, D) tensors), on at most ``max_points`` rows drawn with ``seed`` (sanity check of the
    fixed bandwidths: unit vectors with non-negative entries give values in [0, 2])."""
    pooled = torch.cat([f.float().cpu() for f in feats], dim=0)
    if pooled.shape[0] > max_points:
        g = torch.Generator().manual_seed(int(seed))
        pooled = pooled[torch.randperm(pooled.shape[0], generator=g)[:max_points]]
    d2 = torch.cdist(pooled, pooled).pow(2)
    iu = torch.triu_indices(d2.shape[0], d2.shape[0], offset=1)
    return float(d2[iu[0], iu[1]].median())


def split_hashes(meta: dict) -> dict:
    """All ``*sha256`` entries of the benchmark meta (manifest / split / task-order hashes)."""
    return {k: meta[k] for k in sorted(meta) if str(k).endswith("sha256")}


def raw_view(dataset):
    """Dataset view WITHOUT the benchmark's transforms (Avalanche v0.6.0 API).

    ``AvalancheDataset.remove_current_transform_group()`` (avalanche/benchmarks/utils/
    data.py) sets the CURRENT transform group to None, recursively through nested
    datasets. It is applied to the dataset in its native current group (the group
    ``__getitem__`` then uses), deliberately without ``with_transforms``: that call
    only switches the top level, so nested datasets would keep a different group.
    Frozen transform groups are NOT removed by Avalanche; ``_check_raw_item``
    catches that case. Plain torch datasets are returned unchanged.
    """
    if hasattr(dataset, "remove_current_transform_group"):
        return dataset.remove_current_transform_group()
    return dataset


def _check_raw_item(x, benchmark_has_transform: bool) -> None:
    """Fail loudly if a benchmark transform evidently survived ``raw_view``."""
    if benchmark_has_transform and torch.is_tensor(x):
        raise RuntimeError(
            "Benchmark items are still tensors after removing the benchmark transform group "
            "(frozen transform groups?). Refusing to embed: the encoder's own transform must "
            "be applied to raw images (RESEARCH_LOG 7.3, same encoder for all benchmarks)."
        )


def to_pil(x):
    """Raw item -> RGB PIL image for the encoder transform.

    PIL images are converted to RGB. Tensor items (only the synthetic benchmark has
    no image files) must be CHW with 1 or 3 channels: uint8 is used as-is, float is
    ASSUMED to be on a [0, 1] intensity scale and clamped to it, then converted with
    torchvision ``to_pil_image``. Clamping discards out-of-range values (the synthetic
    data are N(offset, 0.5^2)), which is acceptable because the synthetic benchmark is
    for smoke tests only, never for research results.
    """
    if hasattr(x, "convert"):
        return x.convert("RGB")
    if torch.is_tensor(x):
        from torchvision.transforms.functional import to_pil_image

        if x.ndim != 3 or x.shape[0] not in (1, 3):
            raise ValueError(f"tensor items must be CHW with 1 or 3 channels, got {tuple(x.shape)}")
        if x.dtype != torch.uint8:
            x = x.float().clamp(0.0, 1.0)
        return to_pil_image(x.cpu()).convert("RGB")
    raise TypeError(f"unsupported item type {type(x).__name__}; expected PIL image or tensor")


@torch.no_grad()
def extract_features(
    dataset,
    encoder,
    transform,
    indices,
    device,
    batch_size: int = 128,
    benchmark_has_transform: bool = True,
):
    """Encoder features (n, D) for raw ``dataset[i]``, i in ``indices``.

    ``dataset`` must already be a raw view (``raw_view``); every item goes through
    ``to_pil`` and then the ENCODER's transform.
    """
    feats = []
    for start in range(0, len(indices), batch_size):
        xs = []
        for i in indices[start : start + batch_size]:
            item = dataset[int(i)]
            x = item[0] if isinstance(item, (tuple, list)) else item
            _check_raw_item(x, benchmark_has_transform)
            xs.append(transform(to_pil(x)))
        batch = torch.stack(xs).to(device)
        out = encoder(batch)
        if out.ndim != 2:
            raise RuntimeError(f"encoder must return (B, D) features, got {tuple(out.shape)}")
        feats.append(out.float().cpu())
    return torch.cat(feats, dim=0)


def pair_seed(sample_seed: int, i: int, j: int, stream: int) -> int:
    """Independent, reproducible seed per (sample_seed, pair, purpose)."""
    ss = np.random.SeedSequence([int(sample_seed), int(i), int(j), int(stream)])
    return int(ss.generate_state(1)[0])


def pair_list(t: int, mode: str) -> list[tuple[int, int]]:
    if mode == "consecutive":
        return [(i, i + 1) for i in range(t - 1)]
    if mode == "full":
        return [(i, j) for i in range(t) for j in range(i + 1, t)]
    raise ValueError(f"pairs must be 'consecutive' or 'full', got {mode!r}")


def pairwise_shift(
    feats,
    sigma,
    pairs,
    block_size: int,
    n_perm: int,
    n_sub: int,
    subsample_size: int | None,
    sample_seed: int,
    device,
    max_pooled: int = 8000,
) -> list[dict]:
    """One record per pair: mmd2, subsample CI, permutation null95 and p (NaN when off).

    ``sigma``: one bandwidth or a list of bandwidths (kernels summed).
    """
    rows = []
    for i, j in pairs:
        fi = feats[i].to(device)
        fj = feats[j].to(device)
        rec = {
            "i": i, "j": j,
            "mmd2": mmd2_unbiased(fi, fj, bandwidth=sigma, block_size=block_size),
            "ci_lo": float("nan"), "ci_hi": float("nan"),
            "null95": float("nan"), "p": float("nan"),
            "n_i": int(fi.shape[0]), "n_j": int(fj.shape[0]),
        }
        if n_perm > 0:
            _, p, null = mmd_permutation_test(
                fi, fj, n_permutations=n_perm, seed=pair_seed(sample_seed, i, j, 0),
                bandwidth=sigma, max_pooled=max_pooled, return_null=True,
            )
            rec["p"] = p
            rec["null95"] = float(torch.quantile(null, 0.95))
        if n_sub > 0:
            rec["ci_lo"], rec["ci_hi"] = mmd_subsample_ci(
                fi, fj, n_subsamples=n_sub, subsample_size=subsample_size,
                seed=pair_seed(sample_seed, i, j, 1), bandwidth=sigma, block_size=block_size,
            )
        rows.append(rec)
    return rows


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--benchmark-config", required=True)
    ap.add_argument("--data-dir", default=os.environ.get("CLNS_DATA_DIR", "/workspace/data"))
    ap.add_argument("--seed", "--benchmark-seed", dest="seed", type=int, default=0,
                    help="training seed whose benchmark instance (task order, subsampling) is "
                    "measured; names the output dir seed<k>. Run once per run seed k (CLEAR-10: "
                    "order is seed-independent, one seed suffices; per-seed is harmless)")
    ap.add_argument("--sample-seed", type=int, default=0,
                    help="seed of the per-experience image selection and perm/CI draws")
    ap.add_argument("--encoder", default="tv:resnet50",
                    help=f"one of {sorted(ENCODER_BANDWIDTHS)}")
    ap.add_argument("--n-per-experience", type=int, default=500,
                    help="images per experience (capped at the smallest experience; the same "
                    "n for every experience)")
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--pairs", choices=("consecutive", "full"), default="consecutive",
                    help="consecutive (i, i+1) pairs only (default) or the full upper triangle")
    ap.add_argument("--perm", type=int, default=0, help="permutation-test resamples (0=off)")
    ap.add_argument("--max-pooled", type=int, default=8000,
                    help="max n_i + n_j for the permutation test (memory ~2*(n_i+n_j)^2 floats)")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="m-out-of-n subsamples WITHOUT replacement for the CI (0=off)")
    ap.add_argument("--subsample-size", type=int, default=None,
                    help="m per set for the CI (default n//2 of each experience)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", required=True, help="root; writes <out>/<benchmark>/seed<k>/")
    ap.add_argument("--cache-dir", default=None,
                    help="per-image feature cache root (default <out>/feature_cache; shared by "
                    "all benchmarks and seeds)")
    ap.add_argument("--weights-dir", default=None,
                    help="dino:resnet50 weights dir (default $CLNS_WEIGHTS_DIR or "
                    "/workspace/weights)")
    ap.add_argument("--dino-sha256", default=None,
                    help="expected sha256 of dino_resnet50_pretrain.pth (default "
                    "DINO_RESNET50_SHA256)")
    ap.add_argument("--device", default="auto")
    return ap.parse_args(argv)


def _benchmark_has_transform(cfg_b) -> bool:
    for key in ("train_transform", "eval_transform"):
        v = cfg_b.get(key)
        if v is not None and str(v).lower() != "none":
            return True
    return False


def _atomic_write(path: Path, write_fn, mode: str = "w") -> None:
    tmp = path.with_name(path.name + ".tmp")
    kw = {"encoding": "utf-8", "newline": ""} if "b" not in mode else {}
    with open(tmp, mode, **kw) as fh:
        write_fn(fh)
    os.replace(tmp, path)


def main(argv=None) -> None:
    args = parse_args(argv)
    check_encoder_name(args.encoder)  # before any heavy work
    bandwidths = list(encoder_bandwidths(args.encoder))
    from omegaconf import OmegaConf

    from clns.data import build_benchmark  # lazy: heavy dependency

    cfg = OmegaConf.load(args.benchmark_config)
    cfg_b = cfg.benchmark if "benchmark" in cfg and "name" not in cfg else cfg
    bench_name = str(cfg_b.name)
    bundle = build_benchmark(cfg_b, args.data_dir, args.seed)
    meta = dict(bundle.meta)
    has_tf = _benchmark_has_transform(cfg_b)

    dev_name = args.device
    if dev_name == "auto":
        dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_name)
    enc_info: dict = {}
    dino_sha = args.dino_sha256 if args.dino_sha256 is not None else DINO_RESNET50_SHA256
    encoder, transform = build_encoder(
        args.encoder, device, weights_dir=args.weights_dir, dino_sha256=dino_sha, info=enc_info
    )
    transform_repr = repr(transform)

    out = Path(args.out) / bench_name / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir) if args.cache_dir else Path(args.out) / "feature_cache"
    cache = FeatureCache(
        cache_root, args.encoder, transform_repr, _PIPELINE_VERSION,
        weights_sha256=enc_info.get("encoder_weights_sha256"),
        # file NAME, not path: moving the weights dir keeps the cache (the hash pins the file)
        weights=os.path.basename(str(enc_info.get("encoder_weights", ""))) or None,
    )

    stream = bundle.benchmark.train_stream
    t = len(stream)
    if t < 2:
        raise ValueError(f"need at least 2 train experiences, got {t}")
    exp_ids = [sample_ids(exp.dataset, args.data_dir) for exp in stream]
    n = min([int(args.n_per_experience)] + [len(ids) for ids in exp_ids])
    if n < 2:
        raise ValueError(f"need at least 2 images per experience, got n={n}")

    feats, n_new = [], 0
    for i, exp in enumerate(stream):
        labels = [int(y) for y in exp.dataset.targets]
        pos = select_ids(exp_ids[i], n, args.sample_seed, labels=labels)
        ids = [exp_ids[i][k] for k in pos]
        miss = set(cache.missing(ids))
        if miss:
            todo = [k for k in pos if exp_ids[i][k] in miss]
            # one position per missing id (duplicate ids would otherwise be embedded twice)
            first: dict[str, int] = {}
            for k in todo:
                first.setdefault(exp_ids[i][k], k)
            todo = sorted(first.values())
            f = extract_features(
                raw_view(exp.dataset), encoder, transform, todo, device, args.batch_size, has_tf
            )
            cache.add([exp_ids[i][k] for k in todo], f.numpy())
            n_new += len(todo)
        feats.append(F.normalize(torch.from_numpy(cache.get(ids)), dim=1))
    log.info("feature cache %s: %d new images embedded, %d cached in total",
             cache.dir, n_new, len(cache))
    med = median_sq_dist(feats, seed=args.sample_seed)
    log.info("%s seed %d: median pairwise squared distance of the L2-normalised features = %.4f "
             "(bandwidths %s)", bench_name, args.seed, med, bandwidths)

    pairs = pair_list(t, args.pairs)
    rows = pairwise_shift(
        feats, bandwidths, pairs, args.block_size, args.perm, args.bootstrap,
        args.subsample_size, args.sample_seed, device, args.max_pooled,
    )
    mat = np.full((t, t), np.nan)
    np.fill_diagonal(mat, 0.0)
    pv, lo, hi, n95 = (np.full((t, t), np.nan) for _ in range(4))
    for r in rows:
        for a, b in ((r["i"], r["j"]), (r["j"], r["i"])):
            mat[a, b], pv[a, b] = r["mmd2"], r["p"]
            lo[a, b], hi[a, b], n95[a, b] = r["ci_lo"], r["ci_hi"], r["null95"]
    _atomic_write(
        out / "shift_matrix.npz",
        lambda fh: np.savez(
            fh, matrix=mat, pvals=pv, ci_lo=lo, ci_hi=hi, null95=n95,
            n=np.full(t, n), bandwidths=np.asarray(bandwidths),
        ),
        mode="wb",
    )
    fields = ["i", "j", "mmd2", "ci_lo", "ci_hi", "null95", "p", "n_i", "n_j",
              "measure", "cross_benchmark_comparable"]

    def _csv(fh):
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "measure": MEASURE, "cross_benchmark_comparable": False})

    _atomic_write(out / "shift_pairs.csv", _csv)
    class_order = bundle.class_order if bundle.class_order is not None else meta.get("class_order")
    shift_meta = {
        "benchmark": bench_name,
        "scenario": bundle.scenario,
        "measure": MEASURE,
        "cross_benchmark_comparable": False,
        "note": MEASURE_NOTE,
        "pipeline": _PIPELINE_VERSION,
        "seed": args.seed,
        "benchmark_seed": args.seed,  # legacy alias of ``seed``
        "sample_seed": args.sample_seed,
        "task_order": meta.get("task_order"),
        "task_order_sha256": meta.get("task_order_sha256"),
        "class_order": class_order,
        "split_hashes": split_hashes(meta),
        "T": t,
        "test_exp_to_train_exp": meta.get("test_exp_to_train_exp"),
        "encoder": args.encoder,
        "encoder_transform": transform_repr,
        **enc_info,
        "l2_normalized": True,
        "bandwidths": bandwidths,
        "median_sq_dist": med,
        "kernel": "sum of RBF kernels over bandwidths",
        "pairs": args.pairs,
        "n_per_experience": n,
        "n_per_experience_requested": args.n_per_experience,
        "experience_sizes": [len(ids) for ids in exp_ids],
        "permutations": args.perm,
        "subsamples": args.bootstrap,
        "subsample_size": args.subsample_size or "n//2 per experience",
        "feature_cache": str(cache.dir),
        "benchmark_meta": meta,
    }
    _atomic_write(
        out / "shift_meta.json", lambda fh: json.dump(shift_meta, fh, indent=2, default=str)
    )
    print(f"wrote {out} ({t} experiences, {len(pairs)} pairs, n={n}), bandwidths={bandwidths}")


if __name__ == "__main__":
    main()
