"""Per-image encoder feature cache for the shift measurement.

Why per image: the task order (and, for CORe50, the partition into experiences) depends on the
training seed. Caching features per EXPERIENCE would recompute every image for every order;
caching per IMAGE lets any order / partition reuse what was already embedded.

* :func:`sample_ids` gives every sample of an Avalanche dataset a stable identity that does not
  depend on its position in a stream:
    - leaves with an ``.imgs`` list (Avalanche ``PathsDataset`` / ``FilelistDataset``: ImageNet-100,
      DomainNet, CORe50, CLEAR-10) -> ``path:<path>``, made relative to ``data_root`` when the file
      lies under it (moving the data volume keeps the cache valid);
    - other leaves (the in-memory synthetic benchmark) -> ``sha:<sha256 of the raw item tensor>``.
      The content hash is order-independent and changes with the data, hence with the seed that
      generated it.
  Resolution walks Avalanche's PRIVATE flat-data structure (``AvalancheDataset._flat_data``,
  ``FlatData._datasets`` / ``FlatData._get_idx``; avalanche-lib 0.6.0, pinned and covered by
  tests/test_measure_stream.py).
* :func:`select_ids` ranks ids by ``sha256(f"{seed}:{id}")`` (order-independent). With ``labels``
  it selects PER CLASS: the ``ceil(n/k)`` best-ranked images of each of the ``k`` classes, then
  truncates to ``n`` by (rank within class, hash). A class is therefore always represented by a
  prefix of ONE fixed per-class ranking, whatever experience, position or seed it appears in:
  identical images when ``n`` and ``k`` are unchanged (e.g. the equal-size ImageNet-100 tasks;
  a DomainNet / CLEAR domain), nested prefixes otherwise (CORe50 NIC batches, whose class count
  and frames change with the run, so reuse there is only partial).
* :class:`FeatureCache` stores features of one (encoder, encoder weights, encoder transform,
  pipeline version) in
  append-only ``shard_XXXXX.npz`` files (``ids`` unicode array + ``feats`` float32), each written
  to a temporary file and ``os.replace``-d, so an interrupted run never leaves a corrupt shard.
  Features are stored RAW (not normalised); the measurement L2-normalises them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Sample identities
# --------------------------------------------------------------------------- #
def _resolve_leaf(ds, i: int):
    """(leaf dataset, index in it) of sample ``i`` of an (Avalanche) dataset tree."""
    from avalanche.benchmarks.utils.data import AvalancheDataset
    from avalanche.benchmarks.utils.flat_data import FlatData
    from torch.utils.data import Subset

    i = int(i)
    for _ in range(1000):  # guard against cycles; real trees are a few levels deep
        if isinstance(ds, AvalancheDataset):
            ds = ds._flat_data
        elif isinstance(ds, FlatData):
            d_idx, i = ds._get_idx(i)
            ds = ds._datasets[d_idx]
        elif isinstance(ds, Subset):
            i = int(ds.indices[i])
            ds = ds.dataset
        else:
            return ds, i
    raise RuntimeError("dataset tree too deep while resolving a sample id")


def _path_id(leaf, i: int, data_root: Path | None) -> str:
    entry = leaf.imgs[i]
    p = Path(str(entry[0] if isinstance(entry, (tuple, list)) else entry))
    root = getattr(leaf, "root", None)
    if root is not None and not p.is_absolute():
        p = Path(root) / p
    if data_root is not None:
        try:
            p = Path(os.path.abspath(p)).relative_to(data_root)
        except ValueError:
            pass  # outside the data root: keep the full path
    return "path:" + p.as_posix()


def _content_id(leaf, i: int) -> str:
    import torch

    item = leaf[i]
    x = item[0] if isinstance(item, (tuple, list)) else item
    if torch.is_tensor(x):
        arr = x.detach().cpu().contiguous().numpy()
    elif isinstance(x, np.ndarray):
        arr = np.ascontiguousarray(x)
    else:
        raise TypeError(
            f"cannot build a sample id for leaf {type(leaf).__name__} item of type "
            f"{type(x).__name__}: expected an .imgs path list or tensor items"
        )
    h = hashlib.sha256()
    h.update(f"{arr.dtype}|{arr.shape}|".encode())
    h.update(arr.tobytes())
    return "sha:" + h.hexdigest()


def sample_ids(ds, data_root: str | Path | None = None) -> list[str]:
    """Stable, order-independent id of every sample of ``ds`` (see the module docstring)."""
    # abspath (lexical) rather than resolve(): no file-system call per image.
    root = Path(os.path.abspath(data_root)) if data_root is not None else None
    ids = []
    for i in range(len(ds)):
        leaf, j = _resolve_leaf(ds, i)
        if hasattr(leaf, "imgs"):
            ids.append(_path_id(leaf, j, root))
        else:
            ids.append(_content_id(leaf, j))
    return ids


def select_ids(
    ids: Sequence[str], n: int, sample_seed: int, labels: Sequence[int] | None = None
) -> list[int]:
    """Positions (sorted) of ``n`` ids chosen by ``sha256(f"{seed}:{id}")`` rank.

    Without ``labels``: the ``n`` smallest hashes. With ``labels`` (class of every id): per
    class, rank its ids by hash; keep the ``ceil(n / k)`` first of each of the ``k`` classes and
    truncate to ``n`` by (rank within class, hash) -- classes with too few images are topped up
    by the others' next ranks. Depends only on the SET of (id, label) pairs and the seed, never
    on their order, so an experience that moves in the task order selects the same images.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    hashes = [hashlib.sha256(f"{int(sample_seed)}:{s}".encode()).hexdigest() for s in ids]
    if labels is None:
        keys = sorted((h, k) for k, h in enumerate(hashes))
        return sorted(k for _, k in keys[:n])
    if len(labels) != len(ids):
        raise ValueError(f"{len(labels)} labels for {len(ids)} ids")
    by_class: dict[int, list[tuple[str, int]]] = {}
    for k, (h, y) in enumerate(zip(hashes, labels)):
        by_class.setdefault(int(y), []).append((h, k))
    ranked = []
    for items in by_class.values():
        items.sort()
        ranked.extend((rank, h, k) for rank, (h, k) in enumerate(items))
    ranked.sort()  # ranks 0..ceil(n/k)-1 of every class come first: ceil(n/k) per class, then cut
    return sorted(k for _, _, k in ranked[:n])


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
class FeatureCache:
    """Append-only per-image feature store for one (encoder, weights, transform, pipeline).

    Layout: ``<root>/<key>/meta.json`` + ``shard_XXXXX.npz``, ``key`` = first 16 hex digits of
    sha256 over the encoder name, the encoder weights identity (``weights``: e.g. the
    torchvision weights enum or the weights file path; ``weights_sha256``: the actual file hash),
    the encoder transform repr and ``pipeline``. Other weights -> another cache. Not safe for
    concurrent writers (measurement runs are sequential).
    """

    def __init__(
        self,
        root: str | Path,
        encoder: str,
        transform_repr: str,
        pipeline: str = "",
        weights_sha256: str | None = None,
        weights: str | None = None,
    ):
        payload = json.dumps(
            {
                "encoder": encoder, "transform": transform_repr, "pipeline": pipeline,
                "encoder_weights": weights, "encoder_weights_sha256": weights_sha256,
            },
            sort_keys=True,
        )
        self.key = hashlib.sha256(payload.encode()).hexdigest()[:16]
        self.dir = Path(root) / self.key
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / "meta.json"
        if not meta.exists():
            tmp = meta.with_name(meta.name + ".tmp")
            tmp.write_text(payload + "\n", encoding="utf-8")
            os.replace(tmp, meta)
        self._feats: dict[str, np.ndarray] = {}
        self._n_shards = 0
        for shard in sorted(self.dir.glob("shard_*.npz")):
            with np.load(shard, allow_pickle=False) as z:
                for sid, f in zip(z["ids"].tolist(), z["feats"]):
                    self._feats[str(sid)] = f
            self._n_shards += 1

    def __len__(self) -> int:
        return len(self._feats)

    def __contains__(self, sid: str) -> bool:
        return sid in self._feats

    def missing(self, ids: Sequence[str]) -> list[str]:
        """Ids (unique, in input order) not in the cache."""
        seen, out = set(), []
        for s in ids:
            if s not in self._feats and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def get(self, ids: Sequence[str]) -> np.ndarray:
        """(len(ids), D) float32 features; KeyError if any id is missing."""
        if not ids:
            raise ValueError("get() needs at least one id")
        return np.stack([self._feats[s] for s in ids]).astype(np.float32, copy=False)

    def add(self, ids: Sequence[str], feats) -> None:
        """Store ``feats[k]`` for ``ids[k]`` as a new shard (ids already cached are skipped)."""
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[0] != len(ids):
            raise ValueError(f"feats must be (len(ids), D), got {feats.shape} for {len(ids)} ids")
        keep = [k for k, s in enumerate(ids) if s not in self._feats]
        if not keep:
            return
        new_ids = np.asarray([str(ids[k]) for k in keep])
        new_feats = feats[keep]
        path = self.dir / f"shard_{self._n_shards:05d}.npz"
        while path.exists():  # never overwrite an existing shard
            self._n_shards += 1
            path = self.dir / f"shard_{self._n_shards:05d}.npz"
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, ids=new_ids, feats=new_feats)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        self._n_shards += 1
        for s, f in zip(new_ids.tolist(), new_feats):
            self._feats[str(s)] = f
        log.debug("feature cache %s: +%d (total %d)", self.dir, len(keep), len(self._feats))


__all__ = ["FeatureCache", "sample_ids", "select_ids"]
