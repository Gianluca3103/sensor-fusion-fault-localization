"""Radar-only, post-hoc alternatives to ellipsoid support rasterization.

These functions are comparison tools. Neither uses clean LiDAR or modifies a
Stage-I checkpoint. Scores are model scores, not calibrated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .config import VoxelGrid


@dataclass(frozen=True)
class InterpolatedField:
    coordinates_xyz: np.ndarray  # fine-grid integer XYZ, [N, 3]
    scores: np.ndarray  # [N], in [0, 1]
    source_seeds: int
    accepted_edges: int = 0

    def centers_xyz(self, grid: VoxelGrid) -> np.ndarray:
        origin = np.asarray(grid.minimum_xyz, dtype=np.float64)
        step = np.asarray(grid.size_xyz, dtype=np.float64)
        return (origin + (self.coordinates_xyz.astype(np.float64) + .5) * step).astype(np.float32)


def _validate_seeds(xyz: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or scores.shape != (len(xyz),):
        raise ValueError("Expected seed XYZ [N,3] and one score per seed")
    if not np.isfinite(xyz).all() or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Seed positions and scores must be finite; scores must be in [0,1]")
    return xyz, scores


def trilinear_seed_field(xyz: np.ndarray, scores: np.ndarray, grid: VoxelGrid, *,
                         lattice_stride: int = 2, seed_threshold: float = 0.25) -> InterpolatedField:
    """Splat sparse surface proposals with a separable 3D linear hat basis.

    The hat is the trilinear grid basis. It has finite box support of one
    lattice spacing per axis, with no learned or fixed ellipsoid. Contributions
    from different proposals combine by maximum, so repeated correlated seeds
    do not manufacture confidence by summation. An isolated seed makes only a
    small local patch; there is no interpolation across unsupported gaps.
    """
    xyz, scores = _validate_seeds(xyz, scores)
    if lattice_stride < 1 or not 0 <= seed_threshold <= 1:
        raise ValueError("Invalid lattice stride or seed threshold")
    chosen = scores > seed_threshold
    xyz, scores = xyz[chosen], scores[chosen]
    shape = np.asarray(grid.shape_zyx[::-1], dtype=np.int64)
    if not len(xyz):
        return InterpolatedField(np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.float32), 0)
    origin = np.asarray(grid.minimum_xyz, dtype=np.float64)
    step = np.asarray(grid.size_xyz, dtype=np.float64)
    pitch = step * lattice_stride
    local = (xyz - origin) / step - .5
    keys: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for center, score in zip(local, scores):
        lo = np.maximum(0, np.floor(center - lattice_stride).astype(np.int64) + 1)
        hi = np.minimum(shape - 1, np.ceil(center + lattice_stride).astype(np.int64) - 1)
        if np.any(lo > hi):
            continue
        cells = np.stack(np.meshgrid(*(np.arange(a, b + 1) for a, b in zip(lo, hi)),
                                     indexing="ij"), axis=-1).reshape(-1, 3)
        relative = np.abs((cells - center) * step / pitch)
        weight = np.prod(np.maximum(0., 1. - relative), axis=1)
        positive = weight > 0
        if not positive.any():
            continue
        cells, weight = cells[positive], weight[positive]
        # Preserve the proposal score at its nearest fine voxel center.
        weight = weight / weight.max()
        keys.append(np.ravel_multi_index(cells.T, shape))
        values.append((score * weight).astype(np.float32))
    if not keys:
        return InterpolatedField(np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.float32), len(xyz))
    all_keys = np.concatenate(keys)
    all_values = np.concatenate(values)
    order = np.argsort(all_keys)
    sorted_keys = all_keys[order]
    first = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    unique = sorted_keys[first]
    maximum = np.maximum.reduceat(all_values[order], first)
    cells = np.stack(np.unravel_index(unique, tuple(shape)), axis=1).astype(np.int32)
    return InterpolatedField(cells, maximum, len(xyz))


def edge_aware_seed_field(xyz: np.ndarray, scores: np.ndarray, features: np.ndarray,
                          grid: VoxelGrid, *, groups: np.ndarray | None = None,
                          lattice_stride: int = 2,
                          seed_threshold: float = 0.25, max_link_m: float = 2.0,
                          feature_sigma: float = 0.25, score_sigma: float = 0.20,
                          min_affinity: float = 0.20, neighbors: int = 8) -> InterpolatedField:
    """Smooth on a learned-feature graph; interpolate only accepted seed pairs.

    The search distance limits computation, but does not activate a volume.
    Edges form only between nearby proposals with compatible radar features
    and scores. No clean-LiDAR features enter this inference-time field.
    """
    xyz, scores = _validate_seeds(xyz, scores)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) != len(xyz) or not np.isfinite(features).all():
        raise ValueError("Expected one finite learned radar feature per proposal")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (len(xyz),):
            raise ValueError("Expected one anchor group per proposal")
    if max_link_m <= 0 or feature_sigma <= 0 or score_sigma <= 0 or neighbors < 1 or not 0 < min_affinity < 1:
        raise ValueError("Invalid edge-aware interpolation settings")
    chosen = scores > seed_threshold
    xyz, scores, features = xyz[chosen], scores[chosen], features[chosen]
    if groups is not None:
        groups = groups[chosen]
    if len(xyz) < 2:
        return trilinear_seed_field(xyz, scores, grid, lattice_stride=lattice_stride,
                                    seed_threshold=seed_threshold)
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norm, 1e-12)
    tree = cKDTree(xyz)
    distance, neighbor_index = tree.query(xyz, k=min(neighbors + 1, len(xyz)),
                                          distance_upper_bound=max_link_m)
    distance = np.atleast_2d(distance)
    neighbor_index = np.atleast_2d(neighbor_index)
    edges: dict[tuple[int, int], float] = {}
    for i in range(len(xyz)):
        for d, j in zip(distance[i, 1:], neighbor_index[i, 1:]):
            if j >= len(xyz) or i == j:
                continue
            j = int(j)
            if groups is not None and groups[i] == groups[j]:
                # Two proposal slots from one anchor are alternative hypotheses,
                # not independent evidence that a surface spans between them.
                continue
            cosine = float(np.clip(features[i] @ features[j], -1, 1))
            # Similarity and score changes act as discontinuity guides.
            affinity = np.exp(-(1 - cosine) / feature_sigma
                              - .5 * ((scores[i] - scores[j]) / score_sigma) ** 2
                              - .5 * (d / max_link_m) ** 2)
            if affinity >= min_affinity:
                edge = (min(i, j), max(i, j))
                edges[edge] = max(edges.get(edge, 0.), float(affinity))
    smoothed = scores.copy()
    if edges:
        sums = scores.copy()
        weights = np.ones(len(scores), dtype=np.float64)
        for (i, j), affinity in edges.items():
            sums[i] += affinity * scores[j]
            sums[j] += affinity * scores[i]
            weights[i] += affinity
            weights[j] += affinity
        smoothed = sums / weights
    # Keep the same compact trilinear basis for both methods. The only extra
    # support in this method is along graph edges between compatible seeds.
    all_xyz = [xyz]
    all_scores = [smoothed]
    step = min(grid.size_xyz)
    for (i, j), affinity in edges.items():
        distance_ij = np.linalg.norm(xyz[j] - xyz[i])
        steps = max(1, int(np.ceil(distance_ij / step)))
        fraction = np.arange(1, steps, dtype=np.float64) / steps
        if len(fraction):
            all_xyz.append(xyz[i] + fraction[:, None] * (xyz[j] - xyz[i]))
            all_scores.append((smoothed[i] * (1 - fraction) + smoothed[j] * fraction) * affinity)
    field = trilinear_seed_field(np.concatenate(all_xyz), np.concatenate(all_scores), grid,
                                lattice_stride=lattice_stride, seed_threshold=0.)
    return InterpolatedField(field.coordinates_xyz, field.scores, len(xyz), len(edges))
