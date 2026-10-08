"""CLI: measure inter-experience shift (marginal MMD^2_u) of a benchmark's train stream.

Every experience is embedded with ONE frozen reference encoder (non-CLIP;
RESEARCH_LOG 7.3 / 7.4 Q10, provisional) and unbiased RBF-MMD^2 is computed with a
single bandwidth (median heuristic over the pooled subsample of all experiences).

Preprocessing (RESEARCH_LOG 7.3, "same encoder for all benchmarks"): the benchmark's
own transforms are REMOVED and the RAW items are passed through the ENCODER's eval
transform (torchvision ``weights.transforms()`` / timm ``resolve_data_config`` +
``create_transform``), so every benchmark is embedded identically.

Outputs go to ``<out>/<benchmark>/seed<benchmark-seed>/``: ``shift_matrix.npz``,
``shift_pairs.csv`` (labelled long format) and ``shift_meta.json`` (records benchmark seed,
class order, split hashes, T, encoder and encoder transform; read by
``clns.analysis.correlation_study``). Class-IL class orders are fixed by
``benchmark.class_order_seed`` and do NOT depend on the benchmark/training seed, so one
measurement per benchmark (e.g. ``--benchmark-seed 0``) serves the runs of every training seed:
correlation_study pairs a run with any ``seed<k>`` result whose class order (none for
domain-IL), split hashes and T match the run's ``benchmark_meta`` (the run's own seed is
preferred when several match).

The measure is MARGINAL MMD on P(X) and is NOT comparable across benchmarks
(RESEARCH_LOG 7.3). TODO(RESEARCH_LOG 7.3): class-conditional MMD (and the CORe50
cumulative-reference variant, CLEAR BACKGROUND row) is deferred.

Example:
    python -m clns.shift.measure_stream --benchmark-config configs/benchmark/synthetic.yaml \
        --encoder tv:resnet50 --max-samples 200 --benchmark-seed 0 --out /tmp/shift
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from clns.shift.metrics import (
    median_bandwidth,
    mmd2_unbiased,
    mmd_permutation_test,
    mmd_subsample_ci,
)

MEASURE = "marginal_mmd"
MEASURE_NOTE = (
    "Unbiased RBF MMD^2 on the MARGINAL P(X) of frozen-encoder features, single pooled "
    "median bandwidth per benchmark and benchmark seed. Not comparable across benchmarks: "
    "it measures a label-block change in class-IL benchmarks but a P(X|Y) change in "
    "domain-IL benchmarks (RESEARCH_LOG 7.3). Class-conditional MMD is not yet implemented. "
    "CI = m-out-of-n subsampling without replacement (conservative, not rescaled); "
    "null95 = 95th percentile of the permutation null. Encoder provisional (7.4 Q10)."
)
# Bumped whenever the feature pipeline changes, so stale caches are never reused.
_PIPELINE_VERSION = "raw_items+encoder_transform/v2"


def check_encoder_name(name: str) -> None:
    """Reject CLIP-family encoders (CLEAR labels were built with CLIP; RESEARCH_LOG 7.4 Q10)."""
    if "clip" in name.lower():
        raise ValueError("CLIP encoders are excluded (CLEAR labels were built with CLIP; Q10).")


def build_encoder(name: str, device: torch.device):
    """Return (frozen eval-mode encoder -> (B, D) features, encoder eval transform).

    The transform is the encoder's OWN eval preprocessing (it expects PIL images):
    torchvision ``ResNet50_Weights.IMAGENET1K_V2.transforms()`` or, for timm,
    ``create_transform(**resolve_data_config({}, model=model))`` (timm 0.9.16 API).
    """
    check_encoder_name(name)
    kind, _, arch = name.partition(":")
    if kind == "tv" and arch == "resnet50":
        import torchvision

        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        transform = weights.transforms()
    elif kind == "timm" and arch:
        import timm
        from timm.data import create_transform, resolve_data_config

        model = timm.create_model(arch, pretrained=True, num_classes=0)
        cfg = resolve_data_config({}, model=model)
        transform = create_transform(**cfg)  # is_training defaults to False (eval transform)
    else:
        raise ValueError(f"encoder must be 'tv:resnet50' or 'timm:<name>', got {name!r}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, transform


def split_hashes(meta: dict) -> dict:
    """All ``*sha256`` entries of the benchmark meta (manifest / split hashes)."""
    return {k: meta[k] for k in sorted(meta) if str(k).endswith("sha256")}


def cache_key(
    bench_name: str,
    meta: dict,
    exp: int,
    benchmark_seed: int,
    sample_seed: int,
    max_samples: int,
    encoder: str,
    transform_repr: str,
) -> str:
    """sha256 over everything that changes the cached features (incl. the encoder transform)."""
    payload = json.dumps(
        {
            "pipeline": _PIPELINE_VERSION,
            "benchmark": bench_name,
            "meta": meta,
            "exp": int(exp),
            "benchmark_seed": int(benchmark_seed),
            "sample_seed": int(sample_seed),
            "max_samples": int(max_samples),
            "encoder": encoder,
            "encoder_transform": transform_repr,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


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
    sigma: float,
    pairs,
    block_size: int,
    n_perm: int,
    n_sub: int,
    subsample_size: int | None,
    sample_seed: int,
    device,
    max_pooled: int = 8000,
) -> list[dict]:
    """One record per pair: mmd2, subsample CI, permutation null95 and p (NaN when off)."""
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
    ap.add_argument("--data-dir", default="/workspace/data")
    ap.add_argument("--benchmark-seed", type=int, default=0,
                    help="seed passed to build_benchmark (training-seed-dependent subsampling / "
                    "validation split; the class order is fixed by benchmark.class_order_seed). "
                    "Names the output dir seed<k>; correlation_study matches runs of any seed "
                    "by class order / split hashes / T")
    ap.add_argument("--sample-seed", type=int, default=0,
                    help="seed for per-experience subsampling, bandwidth subsample, perm/CI draws")
    ap.add_argument("--encoder", default="tv:resnet50")
    ap.add_argument("--max-samples", type=int, default=2000, help="per experience")
    ap.add_argument("--bandwidth", default="median", help="'median' or a positive float")
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
    ap.add_argument("--cache-dir", default=None, help="default <out>/<benchmark>/feature_cache")
    ap.add_argument("--device", default="auto")
    return ap.parse_args(argv)


def _benchmark_has_transform(cfg_b) -> bool:
    for key in ("train_transform", "eval_transform"):
        v = cfg_b.get(key)
        if v is not None and str(v).lower() != "none":
            return True
    return False


def main(argv=None) -> None:
    args = parse_args(argv)
    check_encoder_name(args.encoder)  # before any heavy work
    from omegaconf import OmegaConf

    from clns.data import build_benchmark  # lazy: heavy dependency

    cfg = OmegaConf.load(args.benchmark_config)
    cfg_b = cfg.benchmark if "benchmark" in cfg and "name" not in cfg else cfg
    bench_name = str(cfg_b.name)
    bundle = build_benchmark(cfg_b, args.data_dir, args.benchmark_seed)
    meta = dict(bundle.meta)
    has_tf = _benchmark_has_transform(cfg_b)

    dev_name = args.device
    if dev_name == "auto":
        dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev_name)
    encoder, transform = build_encoder(args.encoder, device)
    transform_repr = repr(transform)

    out = Path(args.out) / bench_name / f"seed{args.benchmark_seed}"
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        Path(args.cache_dir) if args.cache_dir else Path(args.out) / bench_name / "feature_cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    feats, sizes = [], []
    for i, exp in enumerate(bundle.benchmark.train_stream):
        key = cache_key(bench_name, meta, i, args.benchmark_seed, args.sample_seed,
                        args.max_samples, args.encoder, transform_repr)
        path = cache_dir / f"{key}.pt"
        if path.exists():
            f = torch.load(path, map_location="cpu")
        else:
            ds = raw_view(exp.dataset)
            rng = np.random.default_rng([int(args.sample_seed), i])
            n = len(ds)
            idx = np.sort(rng.choice(n, size=min(args.max_samples, n), replace=False))
            f = extract_features(ds, encoder, transform, idx, device, args.batch_size, has_tf)
            torch.save(f, path)
        feats.append(f)
        sizes.append(int(f.shape[0]))
    t = len(feats)
    if t < 2:
        raise ValueError(f"need at least 2 train experiences, got {t}")

    if args.bandwidth == "median":
        sigma = median_bandwidth(torch.cat(feats, dim=0), seed=args.sample_seed)
    else:
        sigma = float(args.bandwidth)
        if not (np.isfinite(sigma) and sigma > 0):
            raise ValueError(
                f"--bandwidth must be 'median' or a positive float, got {args.bandwidth}"
            )

    pairs = pair_list(t, args.pairs)
    rows = pairwise_shift(
        feats, sigma, pairs, args.block_size, args.perm, args.bootstrap, args.subsample_size,
        args.sample_seed, device, args.max_pooled,
    )
    mat = np.full((t, t), np.nan)
    np.fill_diagonal(mat, 0.0)
    pv, lo, hi, n95 = (np.full((t, t), np.nan) for _ in range(4))
    for r in rows:
        for a, b in ((r["i"], r["j"]), (r["j"], r["i"])):
            mat[a, b], pv[a, b] = r["mmd2"], r["p"]
            lo[a, b], hi[a, b], n95[a, b] = r["ci_lo"], r["ci_hi"], r["null95"]
    np.savez(
        out / "shift_matrix.npz",
        matrix=mat, pvals=pv, ci_lo=lo, ci_hi=hi, null95=n95,
        n=np.asarray(sizes), bandwidth=np.asarray(sigma),
    )
    fields = ["i", "j", "mmd2", "ci_lo", "ci_hi", "null95", "p", "n_i", "n_j",
              "measure", "cross_benchmark_comparable"]
    with open(out / "shift_pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "measure": MEASURE, "cross_benchmark_comparable": False})
    class_order = bundle.class_order if bundle.class_order is not None else meta.get("class_order")
    with open(out / "shift_meta.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "benchmark": bench_name,
                "scenario": bundle.scenario,
                "measure": MEASURE,
                "cross_benchmark_comparable": False,
                "note": MEASURE_NOTE,
                "benchmark_seed": args.benchmark_seed,
                "sample_seed": args.sample_seed,
                "class_order": class_order,
                "split_hashes": split_hashes(meta),
                "T": t,
                "test_exp_to_train_exp": meta.get("test_exp_to_train_exp"),
                "encoder": args.encoder,
                "encoder_transform": transform_repr,
                "pairs": args.pairs,
                "max_samples": args.max_samples,
                "bandwidth": sigma,
                "n_per_experience": sizes,
                "permutations": args.perm,
                "subsamples": args.bootstrap,
                "subsample_size": args.subsample_size or "n//2 per experience",
                "benchmark_meta": meta,
            },
            fh,
            indent=2,
            default=str,
        )
    print(f"wrote {out} ({t} experiences, {len(pairs)} pairs), sigma={sigma:.4g}")


if __name__ == "__main__":
    main()
