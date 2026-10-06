"""Time a Stage 1 batch without modifying the training run.

Example (run after the active trainer has released the GPU):
python -m scripts.profile_stage1_batch --vod-root PATH --device cuda --batch-size 4
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import torch

from models.radar_lidar_stage1 import model as model_module
from models.radar_lidar_stage1 import sparse as sparse_module
from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from models.radar_lidar_stage1.model import RadarLidarStage1
from models.radar_lidar_stage1.train import config_from_dict
from models.radar_lidar_stage1.metrics import Stage1MetricAccumulator
from models.radar_lidar_stage1.evaluate import evaluate_object_instances
from Fault_Localization_Model.vod_dataset.vod_io import resolve_vod_public_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/radar_lidar_stage1_small.json"))
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0,
                        help="First training frame to profile; choose a typical frame rather than the first sparse frames")
    parser.add_argument("--lidar-point-limit", type=int, default=0,
                        help="Diagnostic CPU run only: cap LiDAR points per frame; 0 uses all points")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile-validation", action="store_true")
    parser.add_argument("--reference-sparse", action="store_true",
                        help="Use the original offset-wise sparse convolution for an A/B comparison")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.batch_size < 1 or args.lidar_point_limit < 0 or args.start_index < 0:
        parser.error("batch size must be positive, point limit and start index nonnegative")

    stats = defaultdict(lambda: [0.0, 0])

    def sync() -> None:
        if args.device == "cuda":
            torch.cuda.synchronize()

    def timed(label, fn):
        def wrapped(*a, **kw):
            sync()
            start = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                sync()
                stats[label][0] += time.perf_counter() - start
                stats[label][1] += 1
        return wrapped

    # Module wrappers are inclusive; subtract nested values when reading the report.
    sparse_module.PointVoxelEncoder.forward = timed("point_voxel_encoding", sparse_module.PointVoxelEncoder.forward)
    convolution = (sparse_module.SparseConv3d.forward_reference if args.reference_sparse
                   else sparse_module.SparseConv3d.forward)
    sparse_module.SparseConv3d.forward = timed("sparse_conv_3d", convolution)
    sparse_module.axial_connectivity = timed("axial_connectivity", sparse_module.axial_connectivity)
    model_module.local_neighbors = timed("kd_tree_correspondence_search", model_module.local_neighbors)
    model_module.LocalCorrespondence.forward = timed("cross_attention_and_contrastive", model_module.LocalCorrespondence.forward)
    model_module.probe_target = timed("probe_target", model_module.probe_target)

    start = time.perf_counter()
    data = VoDStage1Dataset(args.vod_root, "train", radar_variant=args.radar_variant)
    stats["dataset_discovery"] = [time.perf_counter() - start, 1]
    start = time.perf_counter()
    if args.start_index + args.batch_size > len(data):
        parser.error("Requested batch extends beyond the training split")
    rows = [data[i] for i in range(args.start_index, args.start_index + args.batch_size)]
    original_counts = [{"radar": len(row["radar"]), "lidar": len(row["clean_lidar"])} for row in rows]
    if args.lidar_point_limit:
        for row in rows:
            points = row["clean_lidar"]
            if len(points) > args.lidar_point_limit:
                indices = torch.linspace(0, len(points) - 1, args.lidar_point_limit).long()
                row["clean_lidar"] = points[indices]
    batch = collate_stage1(rows)
    stats["data_load_and_collate"] = [time.perf_counter() - start, 1]

    config = config_from_dict(json.loads(args.config.read_text()))
    start = time.perf_counter()
    model = RadarLidarStage1(config).to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    stats["model_and_optimizer_setup"] = [time.perf_counter() - start, 1]
    start = time.perf_counter()
    batch = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    sync()
    stats["host_to_device"] = [time.perf_counter() - start, 1]
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    losses, diagnostics = model.forward_train(batch["radar"], batch["radar_valid"],
                                              batch["clean_lidar"], batch["clean_lidar_valid"])
    sync()
    forward_s = time.perf_counter() - start
    start = time.perf_counter()
    losses["loss/total"].backward()
    sync()
    backward_s = time.perf_counter() - start
    start = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    sync()
    optimizer_s = time.perf_counter() - start

    report = {
        "device": args.device,
        "sparse_backend": "reference" if args.reference_sparse else "kernel_map",
        "batch_size": args.batch_size,
        "start_index": args.start_index,
        "lidar_point_limit": args.lidar_point_limit,
        "original_point_counts": original_counts,
        "used_point_counts": [{"radar": int(x), "lidar": int(y)} for x, y in zip(batch["radar_valid"].sum(-1), batch["clean_lidar_valid"].sum(-1))],
        "forward_s": forward_s, "backward_s": backward_s, "optimizer_s": optimizer_s,
        "total_step_s": forward_s + backward_s + optimizer_s,
        "components": {key: {"seconds": value[0], "calls": value[1]} for key, value in stats.items()},
        "active_sites": {key: {"radar": row["radar_sites"], "lidar": row["lidar_sites"]}
                         for key, row in diagnostics["levels"].items()},
    }
    if args.device == "cuda":
        report["cuda_memory_mib"] = {
            "peak_allocated": round(torch.cuda.max_memory_allocated() / 1048576, 1),
            "peak_reserved": round(torch.cuda.max_memory_reserved() / 1048576, 1),
        }
    if args.profile_validation:
        val = VoDStage1Dataset(args.vod_root, "val", radar_variant=args.radar_variant)
        val_row = val[100]
        val_batch = collate_stage1([val_row])
        val_batch = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value
                     for key, value in val_batch.items()}
        inputs = (val_batch["radar"], val_batch["radar_valid"],
                  val_batch["clean_lidar"], val_batch["clean_lidar_valid"])
        model.eval()
        validation = {"frame_id": val_row["frame_id"], "radar_points": len(val_row["radar"]),
                      "lidar_points": len(val_row["clean_lidar"])}
        with torch.no_grad():
            sync(); start = time.perf_counter()
            model.forward_train(*inputs)
            sync(); validation["loss_forward_s"] = time.perf_counter() - start
            accumulator = Stage1MetricAccumulator(config)
            sync(); start = time.perf_counter()
            accumulator.update(model, *inputs)
            sync(); validation["metrics_update_s"] = time.perf_counter() - start
            frame = val.frames[100]
            label = resolve_vod_public_root(args.vod_root) / "lidar" / "training" / "label_2" / f"{frame.frame_id}.txt"
            if label.is_file():
                sync(); start = time.perf_counter()
                evaluate_object_instances(model, *inputs, label, frame.lidar_calibration_path)
                sync(); validation["object_instances_s"] = time.perf_counter() - start
        validation["total_compute_s"] = sum(value for key, value in validation.items() if key.endswith("_s"))
        with torch.no_grad():
            sync(); start = time.perf_counter()
            _, diagnostics = model.forward_train(*inputs,return_intermediates=True,defer_diagnostics=True)
            sync(); validation["reused_loss_forward_s"] = time.perf_counter() - start
            cached = diagnostics["intermediates"]
            reused_accumulator = Stage1MetricAccumulator(config)
            sync(); start = time.perf_counter()
            reused_accumulator.update(model,*inputs,intermediates=cached)
            sync(); validation["reused_metrics_update_s"] = time.perf_counter() - start
            if label.is_file():
                sync(); start = time.perf_counter()
                evaluate_object_instances(model,*inputs,label,frame.lidar_calibration_path,intermediates=cached)
                sync(); validation["reused_object_instances_s"] = time.perf_counter() - start
        validation["reused_total_compute_s"] = sum(value for key,value in validation.items() if key.startswith("reused_") and key.endswith("_s"))
        report["validation"] = validation
    rendered = json.dumps(report, indent=2)
    print(rendered, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
