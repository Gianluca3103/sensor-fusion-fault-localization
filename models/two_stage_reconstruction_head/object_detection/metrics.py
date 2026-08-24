"""Class-wise rotated-IoU detection and reconstruction-recovery metrics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .annotations import RotatedBEVBox
from .geometry import rotated_box_iou


def match_frame(
    predictions: list[RotatedBEVBox],
    ground_truth: list[RotatedBEVBox],
    iou_threshold: float,
) -> dict:
    matches: list[tuple[int, int, float]] = []
    unmatched_gt = set(range(len(ground_truth)))
    ordered = sorted(enumerate(predictions), key=lambda item: item[1].confidence, reverse=True)
    prediction_status: dict[int, tuple[bool, int | None, float]] = {}
    for prediction_index, prediction in ordered:
        candidates = [
            (gt_index, rotated_box_iou(prediction, ground_truth[gt_index]))
            for gt_index in unmatched_gt
            if ground_truth[gt_index].class_name == prediction.class_name
        ]
        gt_index, iou = max(candidates, key=lambda item: item[1], default=(None, 0.0))
        if gt_index is not None and iou >= iou_threshold:
            unmatched_gt.remove(gt_index)
            matches.append((prediction_index, gt_index, iou))
            prediction_status[prediction_index] = (True, gt_index, iou)
        else:
            prediction_status[prediction_index] = (False, None, iou)
    return {
        "matches": matches,
        "prediction_status": prediction_status,
        "detected_gt": {gt_index for _, gt_index, _ in matches},
        "tp": len(matches),
        "fp": len(predictions) - len(matches),
        "fn": len(ground_truth) - len(matches),
    }


def _average_precision(events: list[tuple[float, bool]], gt_count: int) -> float:
    if gt_count == 0:
        return float("nan")
    events = sorted(events, key=lambda event: event[0], reverse=True)
    tp = np.cumsum([int(is_tp) for _, is_tp in events], dtype=np.float64)
    fp = np.cumsum([int(not is_tp) for _, is_tp in events], dtype=np.float64)
    recall = tp / gt_count
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_detection_conditions(
    frame_records: list[dict],
    class_names: tuple[str, ...],
    iou_threshold: float,
) -> tuple[dict, list[dict], list[dict]]:
    """Evaluate conditions and return summary, per-frame rows, per-object rows.

    Each record contains ``frame_id``, ``ground_truth``, and a ``predictions``
    mapping. GT indices are stable across conditions, which makes clean-loss
    and coarse/fine recovery measurements unambiguous.
    """

    if not frame_records:
        raise ValueError("frame_records cannot be empty")
    conditions = tuple(frame_records[0]["predictions"])
    required = {"clean", "faulty"}
    missing = sorted(required - set(conditions))
    if missing:
        raise ValueError("Missing required conditions: " + ", ".join(missing))
    aggregates = {
        condition: {
            class_name: {"tp": 0, "fp": 0, "fn": 0, "ious": [], "events": [], "gt": 0}
            for class_name in class_names
        }
        for condition in conditions
    }
    frame_rows: list[dict] = []
    object_rows: list[dict] = []
    recovery = defaultdict(int)
    recovery_by_class = {class_name: defaultdict(int) for class_name in class_names}

    for record in frame_records:
        gt = record["ground_truth"]
        matches = {}
        for condition in conditions:
            predictions = record["predictions"][condition]
            result = match_frame(predictions, gt, iou_threshold)
            matches[condition] = result
            row = {"frame_id": record["frame_id"], "condition": condition}
            for class_name in class_names:
                class_gt = sum(box.class_name == class_name for box in gt)
                class_predictions = [box for box in predictions if box.class_name == class_name]
                class_tp = sum(
                    gt[gt_index].class_name == class_name
                    for _, gt_index, _ in result["matches"]
                )
                class_fp = len(class_predictions) - class_tp
                class_fn = class_gt - class_tp
                bucket = aggregates[condition][class_name]
                bucket["tp"] += class_tp
                bucket["fp"] += class_fp
                bucket["fn"] += class_fn
                bucket["gt"] += class_gt
                bucket["ious"].extend(
                    iou for _, gt_index, iou in result["matches"] if gt[gt_index].class_name == class_name
                )
                for prediction_index, prediction in enumerate(predictions):
                    if prediction.class_name == class_name:
                        bucket["events"].append(
                            (prediction.confidence, result["prediction_status"][prediction_index][0])
                        )
                row[f"{class_name}_tp"] = class_tp
                row[f"{class_name}_fp"] = class_fp
                row[f"{class_name}_fn"] = class_fn
            precision = _safe_ratio(result["tp"], result["tp"] + result["fp"])
            recall = _safe_ratio(result["tp"], result["tp"] + result["fn"])
            row.update(
                {
                    "tp": result["tp"], "fp": result["fp"], "fn": result["fn"],
                    "precision": precision, "recall": recall,
                    "f1": _safe_ratio(2 * precision * recall, precision + recall),
                    "mean_matched_iou": float(np.mean([iou for _, _, iou in result["matches"]]))
                    if result["matches"] else 0.0,
                }
            )
            frame_rows.append(row)

        for gt_index, box in enumerate(gt):
            detected = {
                condition: gt_index in matches[condition]["detected_gt"]
                for condition in conditions
            }
            matched_iou = {
                condition: next(
                    (
                        iou
                        for _prediction_index, matched_gt_index, iou
                        in matches[condition]["matches"]
                        if matched_gt_index == gt_index
                    ),
                    0.0,
                )
                for condition in conditions
            }
            lost = detected["clean"] and not detected["faulty"]
            recovered_coarse = lost and detected.get("coarse", False)
            recovered_fine = lost and detected.get("fine", False)
            additional_fine = recovered_fine and not detected.get("coarse", False)
            recovery["clean_detected"] += int(detected["clean"])
            recovery["lost_after_fault"] += int(lost)
            recovery["coarse_recovered"] += int(recovered_coarse)
            recovery["fine_recovered"] += int(recovered_fine)
            recovery["additional_fine"] += int(additional_fine)
            class_recovery = recovery_by_class[box.class_name]
            class_recovery["clean_detected"] += int(detected["clean"])
            class_recovery["lost_after_fault"] += int(lost)
            class_recovery["coarse_recovered"] += int(recovered_coarse)
            class_recovery["fine_recovered"] += int(recovered_fine)
            class_recovery["additional_fine"] += int(additional_fine)
            object_rows.append(
                {
                    "frame_id": record["frame_id"],
                    "gt_index": gt_index,
                    **box.to_dict(),
                    **{f"{condition}_detected": detected[condition] for condition in conditions},
                    **{f"{condition}_iou": matched_iou[condition] for condition in conditions},
                    "lost_after_fault": lost,
                    "recovered_by_coarse": recovered_coarse,
                    "recovered_by_fine": recovered_fine,
                    "additional_fine_over_coarse": additional_fine,
                }
            )

    summary: dict = {"iou_threshold": iou_threshold, "conditions": {}}
    for condition in conditions:
        per_class = {}
        total_tp = total_fp = total_fn = 0
        aps = []
        all_ious = []
        for class_name in class_names:
            bucket = aggregates[condition][class_name]
            precision = _safe_ratio(bucket["tp"], bucket["tp"] + bucket["fp"])
            recall = _safe_ratio(bucket["tp"], bucket["tp"] + bucket["fn"])
            ap = _average_precision(bucket["events"], bucket["gt"])
            if np.isfinite(ap):
                aps.append(ap)
            per_class[class_name] = {
                "ap": ap if np.isfinite(ap) else None,
                "precision": precision,
                "recall": recall,
                "f1": _safe_ratio(2 * precision * recall, precision + recall),
                "mean_matched_iou": float(np.mean(bucket["ious"])) if bucket["ious"] else 0.0,
                "tp": bucket["tp"],
                "fp": bucket["fp"],
                "fn": bucket["fn"],
            }
            total_tp += bucket["tp"]
            total_fp += bucket["fp"]
            total_fn += bucket["fn"]
            all_ious.extend(bucket["ious"])
        precision = _safe_ratio(total_tp, total_tp + total_fp)
        recall = _safe_ratio(total_tp, total_tp + total_fn)
        summary["conditions"][condition] = {
            "map": float(np.mean(aps)) if aps else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": _safe_ratio(2 * precision * recall, precision + recall),
            "mean_matched_iou": float(np.mean(all_ious)) if all_ious else 0.0,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "per_class": per_class,
        }
    maps = {condition: summary["conditions"][condition]["map"] for condition in conditions}
    summary["map_improvement"] = {}
    if "coarse" in maps:
        summary["map_improvement"]["coarse_minus_faulty"] = (
            maps["coarse"] - maps["faulty"]
        )
    if "fine" in maps:
        summary["map_improvement"]["fine_minus_faulty"] = (
            maps["fine"] - maps["faulty"]
        )
        if "coarse" in maps:
            summary["map_improvement"]["fine_minus_coarse"] = (
                maps["fine"] - maps["coarse"]
            )
    lost = recovery["lost_after_fault"]
    summary["object_recovery"] = {
        **dict(recovery),
        "object_loss_rate": _safe_ratio(recovery["lost_after_fault"], recovery["clean_detected"]),
        "coarse_recovery_rate": _safe_ratio(recovery["coarse_recovered"], lost),
        "fine_recovery_rate": _safe_ratio(recovery["fine_recovered"], lost),
        "additional_fine_over_coarse_rate": _safe_ratio(recovery["additional_fine"], lost),
    }
    summary["object_recovery"]["per_class"] = {}
    for class_name, counts in recovery_by_class.items():
        class_lost = counts["lost_after_fault"]
        summary["object_recovery"]["per_class"][class_name] = {
            **dict(counts),
            "object_loss_rate": _safe_ratio(counts["lost_after_fault"], counts["clean_detected"]),
            "coarse_recovery_rate": _safe_ratio(counts["coarse_recovered"], class_lost),
            "fine_recovery_rate": _safe_ratio(counts["fine_recovered"], class_lost),
            "additional_fine_over_coarse_rate": _safe_ratio(counts["additional_fine"], class_lost),
        }
    return summary, frame_rows, object_rows
