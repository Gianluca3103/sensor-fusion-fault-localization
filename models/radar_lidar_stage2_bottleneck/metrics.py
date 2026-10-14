"""Observed-geometry and global-missing validation for completed point clouds."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
import torch

from models.radar_lidar_stage2.voxel_target import decode_centroids
from models.radar_lidar_stage2_residual.train import ResidualMetrics


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


class ReconstructionMetrics:
    def __init__(self):
        self.legacy = ResidualMetrics()
        self.counts = defaultdict(float)
        self.tolerance = {distance: defaultdict(float) for distance in (0.2, 0.5)}
        self.by_fault = defaultdict(lambda: defaultdict(float))
        self.error_frames = 0

    def add(self, output, target, clean: torch.Tensor, clean_valid: torch.Tensor,
            faulty: torch.Tensor, faulty_valid: torch.Tensor,
            fault_kinds: list[str], threshold: float) -> None:
        self.legacy.add(output, target, threshold, clean, clean_valid, faulty, faulty_valid)
        selected = (output.addition_probability > threshold) & output.coverage.may_add
        batch_ids = output.domain.coordinates[:, 0]
        predicted_xyz = decode_centroids(output.domain, output.predicted_offsets)
        true_xyz = target.clean.clean_centroid_xyz
        if len(fault_kinds) != len(clean):
            raise ValueError("Fault labels must match the validation batch")
        for batch, kind in enumerate(fault_kinds):
            frame = batch_ids == batch
            predicted = selected & frame
            positive = target.addition & frame
            free = target.known_free & frame
            unknown = ~(target.supervised | ~output.coverage.may_add) & frame
            row = self.by_fault[kind]
            tp = int((predicted & positive).sum())
            fp = int((predicted & free).sum())
            fn = int((~selected & positive).sum())
            row["tp"] += tp
            row["fp_known_free"] += fp
            row["fn"] += fn
            row["frames"] += 1
            self.counts["tp"] += tp
            self.counts["fp_known_free"] += fp
            self.counts["fn"] += fn
            self.counts["unknown_predicted"] += int((predicted & unknown).sum())
            self.counts["outside_repair_additions"] += int((predicted & ~output.coverage.in_fault_region).sum())
            self.counts["faulty_preserved_points"] += int(faulty_valid[batch].sum())
            self.counts["predicted_points"] += int(predicted.sum())
            self.counts["target_points"] += int(positive.sum())
            self.counts["frames"] += 1
            p = predicted_xyz[predicted].detach().cpu().numpy()
            g = true_xyz[positive].detach().cpu().numpy()
            if len(p) and len(g):
                predicted_distance = cKDTree(g).query(p, workers=1)[0]
                reference_distance = cKDTree(p).query(g, workers=1)[0]
                self.counts["predicted_distance_sum_m"] += float(predicted_distance.sum())
                self.counts["reference_distance_sum_m"] += float(reference_distance.sum())
                self.counts["predicted_distance_count"] += len(predicted_distance)
                self.counts["reference_distance_count"] += len(reference_distance)
                self.counts["symmetric_chamfer_sum_m"] += (
                    float(predicted_distance.mean()) + float(reference_distance.mean()))
                self.error_frames += 1
                for distance, numbers in self.tolerance.items():
                    numbers["predicted_matched"] += int((predicted_distance <= distance).sum())
                    numbers["reference_matched"] += int((reference_distance <= distance).sum())
            for numbers in self.tolerance.values():
                numbers["predicted"] += len(p)
                numbers["reference"] += len(g)

    def summary(self) -> dict:
        report = self.legacy.summary()
        counts = dict(self.counts)
        tp, fp, fn = (counts.get("tp", 0), counts.get("fp_known_free", 0), counts.get("fn", 0))
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        report["voxel_repair"] = {
            "precision_on_observed": precision,
            "recall_on_eligible_clean": recall,
            "f1_on_observed": _ratio(2 * precision * recall, precision + recall),
            "iou_on_observed": _ratio(tp, tp + fp + fn),
            "false_positive_known_free": int(fp),
            "unverified_predicted_sites": int(counts.get("unknown_predicted", 0)),
        }
        report["point_geometry"] = {
            "symmetric_chamfer_m": (_ratio(counts.get("symmetric_chamfer_sum_m", 0), self.error_frames)
                                    if self.error_frames else None),
            "predicted_to_reference_mean_m": (
                _ratio(counts.get("predicted_distance_sum_m", 0), counts.get("predicted_distance_count", 0))
                if counts.get("predicted_distance_count", 0) else None),
            "reference_to_predicted_mean_m": (
                _ratio(counts.get("reference_distance_sum_m", 0), counts.get("reference_distance_count", 0))
                if counts.get("reference_distance_count", 0) else None),
            "distance_frames_with_both_clouds": self.error_frames,
        }
        for distance, values in self.tolerance.items():
            p = _ratio(values["predicted_matched"], values["predicted"])
            r = _ratio(values["reference_matched"], values["reference"])
            report["point_geometry"][f"{distance:.1f}m"] = {
                "precision": p, "recall": r, "f1": _ratio(2 * p * r, p + r),
                "predicted_within_tolerance_percent": 100 * p,
                "reference_within_tolerance_percent": 100 * r,
            }
        report["preservation"] = {
            "outside_repair_additions": int(counts.get("outside_repair_additions", 0)),
            "faulty_points_kept_for_merge": int(counts.get("faulty_preserved_points", 0)),
        }
        report["by_fault"] = {}
        for kind, values in self.by_fault.items():
            local_tp, local_fp, local_fn = values["tp"], values["fp_known_free"], values["fn"]
            p = _ratio(local_tp, local_tp + local_fp)
            r = _ratio(local_tp, local_tp + local_fn)
            report["by_fault"][kind] = {"frames": int(values["frames"]),
                                        "voxel_precision_on_observed": p,
                                        "voxel_recall_on_eligible_clean": r,
                                        "voxel_f1_on_observed": _ratio(2 * p * r, p + r)}
        return report
