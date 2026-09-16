"""Soft-occupancy 2.5D geometry supervision and hard-point evaluation.

No threshold appears in training. NN assignments use a detached KD-tree;
selected distances are recomputed in torch, retaining XYZ/height gradients.
Coverage is expected truncated nearest distance under Bernoulli occupancy.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
import torch.nn.functional as F
from Fault_Localization_Model.bev_utils import HEIGHT_RANGE_M, metric_to_grid


@dataclass(frozen=True)
class GeometricLossConfig:
    enabled: bool = False
    lambda_coverage: float = 1.0
    lambda_accuracy: float = .5
    legacy_weight: float = 0.0
    distance_type: str = 'huber'
    huber_delta_m: float = .10
    max_distance_m: float = .50
    repair_only: bool = True
    halo_m: float = .50
    chunk_size: int = 1024

    def validate(self):
        if self.distance_type not in {'huber', 'linear'}:
            raise ValueError('distance_type must be huber or linear')
        values = (self.lambda_coverage, self.lambda_accuracy, self.legacy_weight,
                  self.halo_m, self.huber_delta_m, self.max_distance_m)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError('Geometric distances/weights must be finite and nonnegative')
        if self.max_distance_m <= 0 or self.huber_delta_m <= 0 or self.chunk_size < 1:
            raise ValueError('Distance cap, Huber delta and chunk_size must be positive')


def region_masks(mask, geometry, halo_m, repair_only=True):
    core = mask > .5 if repair_only else torch.ones_like(mask, dtype=torch.bool)
    rx, ry = math.ceil(halo_m/geometry.pillar_size_x), math.ceil(halo_m/geometry.pillar_size_y)
    if rx or ry:
        x = torch.arange(-rx, rx+1, device=mask.device).float()*geometry.pillar_size_x
        y = torch.arange(-ry, ry+1, device=mask.device).float()*geometry.pillar_size_y
        kernel = (x[:, None].square()+y[None, :].square() <= halo_m**2+1e-8).float()[None, None]
        match = F.conv2d(core.float(), kernel, padding=(rx, ry)) > 0
    else:
        match = core
    return core, match


def expected_xyz(bev, geometry):
    """One cell-center XY and differentiable P90 height per BEV cell."""
    height, width = bev.shape[-2:]
    rows = torch.arange(height, device=bev.device, dtype=torch.float32)
    cols = torch.arange(width, device=bev.device, dtype=torch.float32)
    x, y = torch.meshgrid(geometry.x_max-(rows+.5)*geometry.pillar_size_x,
                         geometry.y_min+(cols+.5)*geometry.pillar_size_y, indexing='ij')
    z = HEIGHT_RANGE_M[0] + bev[2].float() * (HEIGHT_RANGE_M[1]-HEIGHT_RANGE_M[0])
    return torch.stack([x, y, z], dim=-1).reshape(-1, 3)


def select_reference(points, mask, geometry):
    array = points.detach().cpu().numpy() if torch.is_tensor(points) else np.asarray(points)
    _, rows, cols, valid, _, _ = metric_to_grid(array[:, :3],
        (geometry.x_min, geometry.x_max), (geometry.y_min, geometry.y_max), geometry.pillar_size_x)
    selected = np.flatnonzero(valid)
    keep = mask.detach().cpu().numpy().reshape(geometry.height, geometry.width)[rows, cols]
    return selected[keep.astype(bool)]


def nearest_distances(query, target, chunk_size=1024):
    """Exact detached KD-tree selection, locally differentiable distances."""
    if len(target) == 0:
        return query.new_full((len(query),), float('inf'))
    tree = cKDTree(target.detach().cpu().numpy())
    result = []
    for q in query.split(chunk_size):
        _, indices = tree.query(q.detach().cpu().numpy(), k=1)
        selected = target[torch.as_tensor(indices, device=target.device)]
        result.append(torch.linalg.vector_norm(q-selected, dim=1))
    return torch.cat(result) if result else query.new_empty((0,))


class SoftGeometricReconstructionLoss(nn.Module):
    def __init__(self, config, geometry):
        super().__init__()
        self.config, self.geometry = config, geometry
        config.validate()

    def robust(self, distances):
        d = distances.clamp(max=self.config.max_distance_m)
        if self.config.distance_type == 'linear':
            return d
        delta = self.config.huber_delta_m
        # Smooth-L1 in metres, not squared metres.
        return torch.where(d < delta, .5*d.square()/delta, d-.5*delta)

    def forward(self, prediction, references, repair_mask):
        geometry, config = self.geometry, self.config
        core, match = region_masks(repair_mask, geometry, config.halo_m, config.repair_only)
        coverage, accuracy = [], []
        for index, points in enumerate(references):
            points = points.to(prediction.device, dtype=torch.float32).detach()
            xyz = expected_xyz(prediction[index], geometry)
            probabilities = prediction[index, 0].float().flatten().clamp(0, 1)
            # Halo supplies matching evidence but is not an optimization region.
            probabilities = torch.where(core[index].flatten(), probabilities, probabilities.detach())
            xyz = torch.where(core[index].flatten()[:, None], xyz, xyz.detach())
            target_query = points[select_reference(points, core[index], geometry)]
            target_match = points[select_reference(points, match[index], geometry)]
            pred_indices = torch.nonzero(core[index].flatten(), as_tuple=False).flatten()
            q = xyz[pred_indices]
            p = probabilities[pred_indices]
            zero = prediction[index].float().sum()*0
            if not len(target_match):
                accuracy.append(p.mean()*self.robust(p.new_tensor(config.max_distance_m)) if len(p) else zero)
            elif len(q):
                distances = nearest_distances(q, target_match, config.chunk_size)
                accuracy.append((p*self.robust(distances)).sum()/p.sum().clamp_min(1e-6))
            else:
                accuracy.append(zero)
            if not len(target_query):
                coverage.append(zero)
                continue
            # Enumerate only XY cells within the truncated-distance search box.
            rx = math.ceil(config.max_distance_m/geometry.pillar_size_x)+1
            ry = math.ceil(config.max_distance_m/geometry.pillar_size_y)+1
            dr, dc = torch.meshgrid(torch.arange(-rx, rx+1, device=prediction.device),
                                   torch.arange(-ry, ry+1, device=prediction.device), indexing='ij')
            dr, dc = dr.flatten(), dc.flatten()
            sums = zero
            for reference in target_query.split(config.chunk_size):
                rows = torch.floor((geometry.x_max-reference[:, 0])/geometry.pillar_size_x).long()
                cols = torch.floor((reference[:, 1]-geometry.y_min)/geometry.pillar_size_y).long()
                r, c = rows[:, None]+dr, cols[:, None]+dc
                valid = (r >= 0) & (r < geometry.height) & (c >= 0) & (c < geometry.width)
                ids = r.clamp(0, geometry.height-1)*geometry.width+c.clamp(0, geometry.width-1)
                valid = valid & match[index].flatten()[ids]
                occupancy = probabilities[ids]*valid
                distances = torch.linalg.vector_norm(xyz[ids]-reference[:, None], dim=-1)
                distances, order = distances.sort(dim=1)
                occupancy = occupancy.gather(1, order)
                survival = torch.cumprod(1-occupancy, dim=1)
                prior = torch.cat([torch.ones_like(survival[:, :1]), survival[:, :-1]], dim=1)
                costs = (prior*occupancy*self.robust(distances)).sum(1)
                costs = costs + survival[:, -1]*self.robust(costs.new_tensor(config.max_distance_m))
                sums = sums+costs.sum()
            coverage.append(sums/len(target_query))
        cover = torch.stack(coverage).mean()
        accurate = torch.stack(accuracy).mean()
        return {'geometric_loss': config.lambda_coverage*cover+config.lambda_accuracy*accurate,
                'geometric_coverage_loss': cover, 'geometric_accuracy_loss': accurate}


def point_metrics(pred_query, ref_query, pred_match=None, ref_match=None,
                  tolerances=(.1, .2, .5), observable=None):
    """Euclidean, unsquared geometry metrics. Empty one-sided distances are inf.

    Observable is a caller-supplied Boolean per prediction, never fabricated.
    Without it, far returns are unknown rather than proven hallucinations.
    """
    pred_query, ref_query = np.asarray(pred_query).reshape(-1, 3), np.asarray(ref_query).reshape(-1, 3)
    pred_match = pred_query if pred_match is None else np.asarray(pred_match).reshape(-1, 3)
    ref_match = ref_query if ref_match is None else np.asarray(ref_match).reshape(-1, 3)
    def distances(a, b):
        return cKDTree(b).query(a)[0] if len(a) and len(b) else np.full(len(a), np.inf)
    a, b = distances(pred_query, ref_match), distances(ref_query, pred_match)
    both_empty = not len(pred_query) and not len(ref_query)
    mean_a = float(a.mean()) if len(a) else 0.
    mean_b = float(b.mean()) if len(b) else 0.
    combined = np.concatenate([a, b])
    result = {'prediction_points': len(a), 'reference_points': len(b),
              'pred_to_ref_mean_m': mean_a, 'ref_to_pred_mean_m': mean_b,
              'chamfer_mean_m': (mean_a+mean_b)/2,
              'geometric_error_p95_m': float(np.percentile(combined, 95)) if len(combined) and np.isfinite(combined).all() else (0. if both_empty else float('inf')),
              'far_prediction_rate_0_2m': float(np.mean(a > .2)) if len(a) else 0.}
    for threshold in tolerances:
        precision = float(np.mean(a <= threshold)) if len(a) else float(both_empty)
        recall = float(np.mean(b <= threshold)) if len(b) else float(both_empty)
        label = f'{threshold:g}m'
        result.update({f'precision_{label}': precision, f'recall_{label}': recall,
                       f'f1_{label}': 2*precision*recall/(precision+recall) if precision+recall else 0.})
    supported = a <= .2
    known = np.zeros(len(a), dtype=bool) if observable is None else np.asarray(observable, dtype=bool)
    if known.shape != a.shape:
        raise ValueError('Observable-region flags must match predicted query points')
    result.update({'supported_predictions': int(supported.sum()),
                   'unsupported_predictions': int((~supported & known).sum()),
                   'unknown_predictions': int((~supported & ~known).sum()),
                   'hallucination_rate_0_2m': None if observable is None else float((~supported & known).sum()/max(1, known.sum()))})
    return result


def bev_geometry_metrics(bev, reference, repair_mask, geometry, config, threshold=.5):
    core, match = region_masks(repair_mask[None] if repair_mask.ndim == 3 else repair_mask,
                               geometry, config.halo_m, config.repair_only)
    xyz = expected_xyz(bev, geometry).detach().cpu().numpy()
    occupied = bev[0].detach().cpu().numpy().flatten() >= threshold
    ref = reference.detach().cpu().numpy() if torch.is_tensor(reference) else reference
    return point_metrics(xyz[occupied & core.flatten().cpu().numpy()],
        ref[select_reference(ref, core, geometry)],
        xyz[occupied & match.flatten().cpu().numpy()], ref[select_reference(ref, match, geometry)])


def configured_geometry_loss(payload, dataset):
    config = GeometricLossConfig(**payload.get('geometric_loss', {}))
    config.validate()
    if not config.enabled:
        return None
    if not payload.get('geometric_reference', {}).get('enabled', False):
        raise ValueError('geometric_loss requires enabled geometric_reference')
    return SoftGeometricReconstructionLoss(config, dataset.grid_geometry)


def merge_geometric_config(payload, path):
    """Small optional overlay avoids duplicating complete model configs."""
    if path is None:
        return payload
    import json
    from pathlib import Path
    overlay = json.loads(Path(path).read_text())
    allowed = {'geometric_reference', 'geometric_loss', 'geometric_metrics'}
    if set(overlay)-allowed:
        raise ValueError('Geometric overlay may only contain reference/loss/metrics sections')
    return {**payload, **overlay}
