from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score


def fisher_ratio(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    overall = x.mean(axis=0)
    between = 0.0
    within = 0.0
    for label in np.unique(y):
        group = x[y == label]
        center = group.mean(axis=0)
        between += len(group) * float(np.sum((center - overall) ** 2))
        within += float(np.sum((group - center) ** 2))
    return float(between / (within + 1e-12))


def representation_diagnostics(
    full: np.ndarray,
    reduced: np.ndarray,
    y: np.ndarray,
    seed: int,
    distance_pairs: int = 20_000,
    k: int = 10,
) -> dict[str, float]:
    full = np.asarray(full, dtype=np.float32)
    reduced = np.asarray(reduced, dtype=np.float32)
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)
    left = rng.integers(0, n, size=distance_pairs)
    right = rng.integers(0, n, size=distance_pairs)
    valid = left != right
    left, right = left[valid], right[valid]
    d_full = np.linalg.norm(full[left] - full[right], axis=1)
    d_reduced = np.linalg.norm(reduced[left] - reduced[right], axis=1)
    rho = float(spearmanr(d_full, d_reduced).statistic)
    return {
        "distance_spearman": rho,
        "trustworthiness_k10": float(trustworthiness(full, reduced, n_neighbors=k)),
        "fisher_ratio": fisher_ratio(reduced, y),
        "silhouette": float(silhouette_score(reduced, y, metric="euclidean")),
    }

