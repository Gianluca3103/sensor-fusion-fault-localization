"""Dataset-level counts for occupancy and metric XYZ reconstruction."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
import torch

from models.radar_lidar_stage1.sparse import encode_keys, voxelize
from .reconstruction_model import Stage2Output
from .voxel_target import VoxelTargets, decode_centroids


def _prf(tp: int, predicted: int, actual: int) -> dict[str, float]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / actual if actual else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0}


def clean_points_in_domain(output: Stage2Output, clean: torch.Tensor,
                           valid: torch.Tensor) -> np.ndarray:
    coords, inverse, selected, _ = voxelize(clean, valid, output.domain.grid)
    if not len(coords) or not len(output.candidate_coordinates):
        return np.empty((0, 3), np.float32)
    domain_keys = encode_keys(output.candidate_coordinates, output.domain.grid.shape_zyx)
    query_keys = encode_keys(coords, output.domain.grid.shape_zyx)
    where = torch.searchsorted(domain_keys, query_keys)
    present = (where < len(domain_keys)) & (domain_keys[where.clamp(max=len(domain_keys)-1)] == query_keys)
    return selected[present[inverse], :3].detach().cpu().numpy()


class Stage2MetricAccumulator:
    def __init__(self):
        self.tp = self.fp = self.fn = 0
        self.offset_abs_sum = self.offset_sq_sum = self.offset_count = 0.0
        self.point_counts = {t: {"pred": 0, "actual": 0, "pred_match": 0, "actual_match": 0}
                             for t in (0.1, 0.2, 0.5)}
        self.candidate_sites = self.predicted_sites = self.clean_in_grid = self.clean_in_candidates = 0
        self.positive_sites = self.known_free_sites = self.unknown_sites = 0
        self.frames = 0
        self.confidence_bins = {f"{lo:.1f}-{hi:.1f}": {"tp": 0, "fp": 0, "fn": 0,
                                  "predicted": 0, "true": 0, "pred_match": 0, "true_match": 0}
                                for lo, hi in ((.0,.5),(.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.000001))}

    def add(self, output: Stage2Output, target: VoxelTargets,
            clean: torch.Tensor, valid: torch.Tensor, *,
            occupancy_threshold: float = 0.5) -> None:
        self.frames += clean.shape[0]
        self.candidate_sites += len(output.candidate_coordinates)
        predicted = output.occupancy_probability > occupancy_threshold
        self.predicted_sites += int(predicted.sum())
        supervised = target.occupied | target.known_free
        self.positive_sites += int(target.occupied.sum())
        self.known_free_sites += int(target.known_free.sum())
        self.unknown_sites += int((~supervised).sum())
        self.tp += int((predicted & target.occupied).sum())
        self.fp += int((predicted & target.known_free).sum())
        self.fn += int((~predicted & target.occupied).sum())
        self.clean_in_grid += target.clean_points_in_grid
        self.clean_in_candidates += target.clean_points_in_candidates
        if bool(target.occupied.any()):
            size = output.predicted_offsets.new_tensor(output.domain.grid.size_xyz)
            error = (output.predicted_offsets[target.occupied] - target.offsets_normalized[target.occupied]) * size
            self.offset_abs_sum += float(error.abs().sum())
            self.offset_sq_sum += float(error.square().sum())
            self.offset_count += error.numel()
        all_pred = decode_centroids(output.domain, output.predicted_offsets)
        clean_xyz = clean_points_in_domain(output, clean, valid)
        # Match within each frame; neither point-set distances nor occupancy
        # metrics are allowed to pair different batch elements.
        batches = output.candidate_coordinates[:, 0].detach().cpu().numpy()
        clean_coords, inverse, _, inside = voxelize(clean, valid, output.domain.grid)
        clean_batches = clean_coords[inverse, 0].detach().cpu().numpy() if len(inverse) else np.empty(0, np.int64)
        candidate_keys = encode_keys(output.candidate_coordinates, output.domain.grid.shape_zyx)
        clean_keys = encode_keys(clean_coords, output.domain.grid.shape_zyx)
        if len(clean_keys) and len(candidate_keys):
            positions = torch.searchsorted(candidate_keys, clean_keys)
            present = (positions < len(candidate_keys)) & (candidate_keys[positions.clamp(max=len(candidate_keys)-1)] == clean_keys)
            clean_batches = clean_batches[present[inverse].detach().cpu().numpy()]
        else:
            clean_batches = np.empty(0, np.int64)
        all_pred_np = all_pred.detach().cpu().numpy()
        pred_np = predicted.detach().cpu().numpy()
        conf_np = output.confidence.detach().cpu().numpy()
        positive_np = target.occupied.detach().cpu().numpy()
        free_np = target.known_free.detach().cpu().numpy()
        oracle_np = decode_centroids(output.domain, target.offsets_normalized).detach().cpu().numpy()
        for batch in range(clean.shape[0]):
            p = all_pred_np[pred_np & (batches == batch)]
            g = clean_xyz[clean_batches == batch]
            dp = cKDTree(g).query(p, k=1, workers=1)[0] if len(p) and len(g) else np.full(len(p), np.inf)
            dg = cKDTree(p).query(g, k=1, workers=1)[0] if len(g) and len(p) else np.full(len(g), np.inf)
            for tolerance, counts in self.point_counts.items():
                counts["pred"] += len(p)
                counts["actual"] += len(g)
                counts["pred_match"] += int(np.count_nonzero(dp <= tolerance))
                counts["actual_match"] += int(np.count_nonzero(dg <= tolerance))
            for label, counts in self.confidence_bins.items():
                lo, hi = (float(v) for v in label.split("-"))
                if hi == 1.0:
                    hi = 1.000001
                in_bin = (batches == batch) & (conf_np >= lo) & (conf_np < hi)
                selected = in_bin & pred_np
                truth = in_bin & positive_np
                counts["tp"] += int(np.count_nonzero(selected & positive_np))
                counts["fp"] += int(np.count_nonzero(selected & free_np))
                counts["fn"] += int(np.count_nonzero(truth & ~pred_np))
                candidate_pred = all_pred_np[selected]
                candidate_true = oracle_np[truth]
                counts["predicted"] += len(candidate_pred)
                counts["true"] += len(candidate_true)
                if len(candidate_pred) and len(candidate_true):
                    counts["pred_match"] += int(np.count_nonzero(cKDTree(candidate_true).query(candidate_pred)[0] <= .2))
                    counts["true_match"] += int(np.count_nonzero(cKDTree(candidate_pred).query(candidate_true)[0] <= .2))

    def summary(self) -> dict:
        occ = _prf(self.tp, self.tp+self.fp, self.tp+self.fn)
        occ["iou"] = self.tp / (self.tp+self.fp+self.fn) if self.tp+self.fp+self.fn else 0.0
        point = {}
        for tolerance, counts in self.point_counts.items():
            precision = counts["pred_match"] / counts["pred"] if counts["pred"] else 0.0
            recall = counts["actual_match"] / counts["actual"] if counts["actual"] else 0.0
            point[f"{tolerance:.1f}m"] = {"precision": precision, "recall": recall,
                                         "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
                                         **counts}
        bins = {}
        for label, row in self.confidence_bins.items():
            occupancy = _prf(row["tp"], row["tp"]+row["fp"], row["tp"]+row["fn"])
            precision = row["pred_match"]/row["predicted"] if row["predicted"] else 0.0
            recall = row["true_match"]/row["true"] if row["true"] else 0.0
            bins[label] = {"reconstructed_points": row["predicted"],
                           "occupancy_precision": occupancy["precision"],
                           "centroid_precision_0.2m": precision, "centroid_recall_0.2m": recall,
                           "centroid_f1_0.2m": 2*precision*recall/(precision+recall) if precision+recall else 0.0}
        return {"frames": self.frames, "occupancy": occ, "geometry": point,
                "confidence_bins": bins,
                "offset_mae_m": self.offset_abs_sum/self.offset_count if self.offset_count else None,
                "offset_rmse_m": (self.offset_sq_sum/self.offset_count)**0.5 if self.offset_count else None,
                "candidate_sites": self.candidate_sites, "predicted_sites": self.predicted_sites,
                "positive_sites": self.positive_sites, "known_free_sites": self.known_free_sites,
                "unknown_sites": self.unknown_sites,
                "all_occupied_baseline_iou": self.positive_sites/(self.positive_sites+self.known_free_sites)
                if self.positive_sites+self.known_free_sites else 0.0,
                "candidate_point_coverage": self.clean_in_candidates/self.clean_in_grid if self.clean_in_grid else 0.0,
                "clean_points_in_grid": self.clean_in_grid,
                "clean_points_in_candidates": self.clean_in_candidates}
