"""Embedding-collision metrics for the isomorphism benchmark.

A "collision" is a pair of distinct molecules whose (L2-normalized) embeddings lie
within a cosine distance `radius` of each other. The collision rate = collisions /
pairs measures how often a model maps distinct molecules to (near-)identical points
-- lower is better.

All backends count the same relation; they differ only in implementation. Each
computes the expensive step (distance matrix / index search) ONCE and then bins the
result across every radius, so evaluating a list of radii costs little more than one.

The unit-vector identity used throughout: for L2-normalized a, b,
    ||a - b||^2 = 2 * (1 - cos(a, b)) = 2 * cosine_distance(a, b)
so a cosine-distance threshold `r` corresponds to squared-L2 `2r` (faiss IndexFlatL2)
or euclidean `sqrt(2r)` (sklearn BallTree).
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pyarrow.parquet as pq

Radii = Union[float, List[float]]
Result = Dict[float, Tuple[float, float]]  # radius -> (collisions, pairs)

DEFAULT_RADII: List[float] = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def _as_list(radii: Radii) -> List[float]:
    return [float(radii)] if isinstance(radii, (int, float)) else [float(r) for r in radii]


def remove_nans(embeddings: np.ndarray) -> np.ndarray:
    """Drop rows that are NaN (failed molecules) or exactly zero (degenerate)."""
    nan_mask = np.isnan(embeddings).any(axis=1)
    zero_mask = np.linalg.norm(embeddings, axis=1) == 0
    return embeddings[~(nan_mask | zero_mask)]


def _tally(distances_cos: np.ndarray, n1: int, n2: int, same_datasets: bool, radii: List[float]) -> Result:
    """Count collisions at each radius from precomputed cosine distances.

    `distances_cos` is either the full n1 x n2 matrix (brute/torch) or a flat array
    of the distances below the largest radius (faiss/sklearn) -- either way `.sum()`
    over a `< r` mask gives the raw count.
    """
    total_pairs = n1 * n2
    results: Result = {}
    for r in radii:
        collisions = float((distances_cos < r).sum())
        pairs = float(total_pairs)
        if same_datasets:
            # each point matches itself (dist 0 < r); remove the n1 self-matches,
            # then halve to turn ordered (i, j)/(j, i) counts into unordered pairs
            collisions = (collisions - n1) / 2
            pairs = (total_pairs - n1) / 2
        results[r] = (collisions, pairs)
    return results


def count_collisions_brute_force(e1: np.ndarray, e2: np.ndarray, same_datasets: bool, radii: Radii = 1e-3) -> Result:
    radii = _as_list(radii)
    e1, e2 = remove_nans(e1), remove_nans(e2)
    n1, n2 = len(e1), len(e2)
    if n1 == 0 or n2 == 0:
        return {r: (0.0, 0.0) for r in radii}

    a = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
    b = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
    distances = 1.0 - a @ b.T  # expensive step, once
    return _tally(distances, n1, n2, same_datasets, radii)


def count_collisions_faiss(e1: np.ndarray, e2: np.ndarray, same_datasets: bool, radii: Radii = 1e-3) -> Result:
    import faiss

    radii = _as_list(radii)
    e1, e2 = remove_nans(e1), remove_nans(e2)
    n1, n2 = len(e1), len(e2)
    if n1 == 0 or n2 == 0:
        return {r: (0.0, 0.0) for r in radii}

    a = np.ascontiguousarray(e1, dtype=np.float32)
    b = np.ascontiguousarray(e2, dtype=np.float32)
    faiss.normalize_L2(a)
    faiss.normalize_L2(b)

    index = faiss.IndexFlatL2(b.shape[1])
    index.add(b)
    # one range search at the largest radius (squared-L2 = 2 * cosine distance);
    # smaller radii are subsets we bin in _tally
    _, sq_l2, _ = index.range_search(a, 2.0 * max(radii))
    return _tally(sq_l2 / 2.0, n1, n2, same_datasets, radii)


def count_collisions_sklearn(e1: np.ndarray, e2: np.ndarray, same_datasets: bool, radii: Radii = 1e-3) -> Result:
    from sklearn.neighbors import BallTree
    from sklearn.preprocessing import normalize

    radii = _as_list(radii)
    e1, e2 = remove_nans(e1), remove_nans(e2)
    n1, n2 = len(e1), len(e2)
    if n1 == 0 or n2 == 0:
        return {r: (0.0, 0.0) for r in radii}

    a = normalize(e1, norm="l2")
    b = normalize(e2, norm="l2")
    tree = BallTree(b, metric="euclidean")  # expensive build, once
    _, dist = tree.query_radius(a, r=np.sqrt(2.0 * max(radii)), return_distance=True)
    all_d = np.concatenate(dist) if len(dist) else np.empty(0)
    return _tally((all_d ** 2) / 2.0, n1, n2, same_datasets, radii)


def count_collisions_torch(e1: np.ndarray, e2: np.ndarray, same_datasets: bool, radii: Radii = 1e-3) -> Result:
    import torch

    radii = _as_list(radii)
    e1, e2 = remove_nans(e1), remove_nans(e2)
    n1, n2 = len(e1), len(e2)
    if n1 == 0 or n2 == 0:
        return {r: (0.0, 0.0) for r in radii}

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        return count_collisions_brute_force(e1, e2, same_datasets, radii)

    a = torch.tensor(e1, dtype=torch.float32, device=device)
    b = torch.tensor(e2, dtype=torch.float32, device=device)
    a = a / torch.linalg.norm(a, dim=1, keepdim=True)
    b = b / torch.linalg.norm(b, dim=1, keepdim=True)
    distances = 1.0 - a @ b.T  # expensive step, once

    total_pairs = n1 * n2
    results: Result = {}
    for r in radii:
        collisions = float((distances < r).sum().item())
        pairs = float(total_pairs)
        if same_datasets:
            collisions = (collisions - n1) / 2
            pairs = (total_pairs - n1) / 2
        results[r] = (collisions, pairs)
    return results


BACKENDS = {
    "faiss": count_collisions_faiss,
    "brute": count_collisions_brute_force,
    "sklearn": count_collisions_sklearn,
    "torch": count_collisions_torch,
}


def _load_shard(path: str) -> np.ndarray:
    return np.stack(pq.ParquetFile(path).read().to_pandas()["embedding"].to_numpy())


def count_dataset_collisions(dataset_dir: Union[str, Path], radii: Radii = DEFAULT_RADII, backend: str = "faiss") -> Result:
    """Total (collisions, pairs) per radius over a whole sharded dataset.

    Iterates every shard pair (combinations_with_replacement): within-shard pairs
    (deduplicated) plus cross-shard pairs together cover every unordered molecule
    pair exactly once, so the accumulated totals equal the full-dataset computation
    without ever materializing the N x N matrix.
    """
    radii = _as_list(radii)
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(BACKENDS)}")
    count_fn = BACKENDS[backend]

    shards = pq.ParquetDataset(str(dataset_dir)).files
    totals: Result = {r: (0.0, 0.0) for r in radii}
    for shard_1, shard_2 in combinations_with_replacement(shards, 2):
        e1 = _load_shard(shard_1)
        e2 = _load_shard(shard_2)
        res = count_fn(e1, e2, shard_1 == shard_2, radii)
        for r in radii:
            c, p = res[r]
            tc, tp = totals[r]
            totals[r] = (tc + c, tp + p)
    return totals
