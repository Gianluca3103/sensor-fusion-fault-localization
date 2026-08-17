"""Train deterministic direct-BEV coarse LiDAR reconstruction independently."""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fault_Localization_Model.io_utils import (
    atomic_torch_save,
    atomic_write_json,
    write_csv_rows,
)
from models.Fault_Localization.training_utils import _split_paths, resolve_device, seed_everything
from models.two_stage_reconstruction_head import (
    CoarseReconstructionDataset,
    CoarseReconstructionModel,
    MaskedBEVReconstructionLoss,
    build_augmentation_config,
    build_configs,
    coarse_reconstruction_collate,
    coarse_reconstruction_metrics,
    load_config,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--radar-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "coarse_reconstruction_vod.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-train-samples", type=int)
    parser.add_argument("--limit-val-samples", type=int)
    parser.add_argument(
        "--disable-radar",
        action="store_true",
        help=(
            "Run a radar-input ablation by replacing radar BEVs with zeros "
            "during both training and validation."
        ),
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Write live training metrics for TensorBoard.",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        help="TensorBoard directory (default: <output-root>/tensorboard).",
    )
    return parser.parse_args()


def _create_tensorboard_writer(enabled: bool, log_dir: Path):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard tracking requires the 'tensorboard' package. "
            "Install the project requirements or run: "
            "python -m pip install tensorboard"
        ) from exc
    writer = SummaryWriter(log_dir=str(log_dir))
    atexit.register(writer.close)
    return writer


def _write_tensorboard_epoch(writer, row: dict, optimizer, epoch: int) -> None:
    if writer is None:
        return
    for name, value in row.items():
        if name == "epoch" or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            writer.add_scalar(name, value, epoch)
    for index, group in enumerate(optimizer.param_groups):
        writer.add_scalar(
            f"optimizer/learning_rate_group_{index}",
            float(group["lr"]),
            epoch,
        )
    writer.flush()


def _move_batch(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "faulty_bev",
        "radar_bev",
        "reconstruction_mask",
        "healthy_context_mask",
        "halo_mask",
        "clean_bev",
    )
    moved = {key: batch[key].to(device, non_blocking=True) for key in keys}
    for key in ("faulty_lidar_points", "radar_points"):
        if key in batch:
            moved[key] = tuple(
                points.to(device, non_blocking=True) for points in batch[key]
            )
    if "observability_confidence" in batch:
        moved["observability_confidence"] = batch[
            "observability_confidence"
        ].to(device, non_blocking=True)
    mask_dtype = moved["faulty_bev"].dtype
    for key in ("reconstruction_mask", "healthy_context_mask", "halo_mask"):
        moved[key] = moved[key].to(dtype=mask_dtype)
    return moved


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _shape_log(inputs: dict, outputs: dict) -> dict:
    names = (
        "erased_lidar_bev",
        "replacement_raw",
        "replacement_bev",
        "occupancy_logits",
        "predicted_density",
        "predicted_height",
        "coarse_lidar_bev",
        "reconstruction_mask",
        "healthy_context_mask",
        "halo_mask",
        "local_input",
        "lidar_pillar_bev",
        "radar_pillar_bev",
        "hrnet_stage_1_branch_0",
        "hrnet_stage_2_branch_0",
        "hrnet_stage_2_branch_1",
        "hrnet_stage_3_branch_0",
        "hrnet_stage_3_branch_1",
        "hrnet_stage_3_branch_2",
        "hrnet_stage_4_branch_0",
        "hrnet_stage_4_branch_1",
        "hrnet_stage_4_branch_2",
        "hrnet_stage_4_branch_3",
        "hrnet_final_concatenated",
        "hrnet_final_features",
    )
    result = {
        key: (
            list(value.shape)
            if isinstance(value, torch.Tensor)
            else [list(points.shape) for points in value]
        )
        for key, value in inputs.items()
    }
    result.update(
        {key: list(outputs[key].shape) for key in names if key in outputs}
    )
    for sensor in ("lidar", "radar"):
        statistics = outputs.get(f"{sensor}_pillar_statistics")
        if statistics:
            result[f"{sensor}_pillar_statistics"] = {
                key: value.detach().cpu().tolist()
                for key, value in statistics.items()
            }
    return result


def _summarize_active_fractions(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty active-fraction collection")
    fractions = torch.tensor(values, dtype=torch.float64)
    return {
        "median": float(torch.quantile(fractions, 0.5)),
        "p90": float(torch.quantile(fractions, 0.9)),
        "maximum": float(fractions.max()),
    }


def _active_fraction_recommendation(summary: dict[str, float]) -> str:
    if summary["median"] >= 0.25:
        return "Keep dense HRNet: typical active coverage is at least 25%."
    if summary["p90"] <= 0.15:
        return (
            "Test cropped dense processing first: at least 90% of samples use "
            "no more than 15% of the BEV. Consider cropping only if "
            "cropping remains too expensive."
        )
    return (
        "Keep dense HRNet processing and profile cropping only if needed: "
        "active coverage is neither consistently sparse nor typically above 25%."
    )


def _save_conditioning(
    output_root: Path,
    epoch: int,
    batch: dict,
    outputs: dict,
    max_samples: int,
) -> None:
    if max_samples <= 0:
        return
    destination = output_root / "conditioning" / f"epoch_{epoch:03d}"
    keys = (
        "coarse_lidar_bev",
        "replacement_raw",
        "replacement_bev",
        "occupancy_logits",
        "reconstruction_mask",
    )
    count = min(max_samples, outputs["coarse_lidar_bev"].shape[0])
    for index in range(count):
        payload = {key: outputs[key][index].detach().cpu() for key in keys}
        payload["sample_path"] = batch["sample_path"][index]
        atomic_torch_save(payload, destination / f"sample_{index:03d}.pt")


def _run_epoch(
    model,
    loader,
    loss_fn,
    device,
    *,
    optimizer=None,
    scaler=None,
    grad_clip=0.0,
    use_amp=False,
    conditioning_callback=None,
    active_fraction_samples=None,
    radar_enabled=True,
    profile_first_batch=False,
):
    training = optimizer is not None
    model.train(training)
    loss_sums = {}
    metric_sums = {}
    samples = 0
    metric_samples = 0
    logged_shapes = None
    for batch_index, batch in enumerate(loader):
        if active_fraction_samples is not None:
            active_mask = torch.maximum(
                batch["reconstruction_mask"], batch["halo_mask"]
            )
            fractions = active_mask.flatten(1).float().mean(dim=1)
            active_fraction_samples.extend(fractions.tolist())
        inputs = _move_batch(batch, device)
        pointpillars_enabled = bool(
            getattr(
                getattr(getattr(model, "config", None), "pointpillars", None),
                "enabled",
                False,
            )
        )
        if not radar_enabled:
            inputs["radar_bev"] = torch.zeros_like(inputs["radar_bev"])
        if training:
            optimizer.zero_grad(set_to_none=True)
        capture_debug = logged_shapes is None
        measure_runtime = capture_debug and profile_first_batch
        if measure_runtime:
            _synchronize_device(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            forward_started = time.perf_counter()
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                model_options = {"radar_enabled": radar_enabled}
                if pointpillars_enabled:
                    model_options.update(
                        {
                            "faulty_lidar_points": inputs.get(
                                "faulty_lidar_points"
                            ),
                            "radar_points": inputs.get("radar_points"),
                        }
                    )
                outputs = model(
                    inputs["faulty_bev"],
                    inputs["radar_bev"],
                    inputs["reconstruction_mask"],
                    inputs["healthy_context_mask"],
                    inputs["halo_mask"],
                    **model_options,
                )
                if measure_runtime:
                    _synchronize_device(device)
                    forward_time_ms = 1000.0 * (
                        time.perf_counter() - forward_started
                    )
                losses = loss_fn(
                    outputs,
                    inputs["clean_bev"],
                    inputs.get("observability_confidence"),
                )
            if training:
                if measure_runtime:
                    _synchronize_device(device)
                    backward_started = time.perf_counter()
                scaler.scale(losses["loss"]).backward()
                if measure_runtime:
                    _synchronize_device(device)
                    backward_time_ms = 1000.0 * (
                        time.perf_counter() - backward_started
                    )
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        with torch.no_grad():
            metrics = coarse_reconstruction_metrics(
                outputs,
                inputs["faulty_bev"],
                inputs["clean_bev"],
                loss_fn.config.epsilon,
                inputs.get("observability_confidence"),
                include_tolerant=True,
                resolution_m=loss_fn.bev_resolution_m,
                tolerance_m=loss_fn.config.occupancy.tolerance_radius_m,
            )
        batch_size = inputs["faulty_bev"].shape[0]
        samples += batch_size
        valid_metric_samples = int(
            (inputs["reconstruction_mask"].flatten(1) > 0)
            .any(dim=1)
            .sum()
            .item()
        )
        metric_samples += valid_metric_samples
        for key, value in losses.items():
            loss_sums[key] = (
                loss_sums.get(key, 0.0)
                + float(value.detach()) * batch_size
            )
        for key, value in metrics.items():
            metric_sums[key] = (
                metric_sums.get(key, 0.0)
                + float(value.detach()) * valid_metric_samples
            )
        if logged_shapes is None:
            logged_shapes = _shape_log(inputs, outputs)
            if measure_runtime:
                logged_shapes["runtime"] = {
                    "forward_time_ms": forward_time_ms,
                }
                if training:
                    logged_shapes["runtime"]["backward_pass_ms"] = (
                        backward_time_ms
                    )
                if device.type == "cuda":
                    logged_shapes["runtime"].update(
                        {
                            "memory_allocated_bytes": torch.cuda.memory_allocated(
                                device
                            ),
                            "memory_reserved_bytes": torch.cuda.memory_reserved(
                                device
                            ),
                            "peak_memory_allocated_bytes": (
                                torch.cuda.max_memory_allocated(device)
                            ),
                        }
                    )
        if conditioning_callback is not None and batch_index == 0:
            conditioning_callback(batch, outputs)
    statistics = {
        key: value / max(samples, 1) for key, value in loss_sums.items()
    }
    statistics.update(
        {
            key: value / max(metric_samples, 1)
            for key, value in metric_sums.items()
        }
    )
    statistics["metric_samples"] = metric_samples
    statistics["excluded_empty_mask_samples"] = samples - metric_samples
    return statistics, logged_shapes


def main():
    args = _parse_args()
    radar_enabled = not args.disable_radar
    payload = load_config(args.config)
    model_config, loss_config, selector_config = build_configs(payload)
    augmentation_config = build_augmentation_config(payload)
    training = dict(payload.get("training", {}))
    epochs = args.epochs or int(training.get("epochs", 50))
    batch_size = args.batch_size or int(training.get("batch_size", 8))
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(training.get("num_workers", 4))
    )
    if epochs < 1 or batch_size < 1 or num_workers < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers non-negative")
    seed = int(training.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir = Path(
        args.tensorboard_log_dir or output_root / "tensorboard"
    )
    tensorboard_writer = _create_tensorboard_writer(
        args.tensorboard,
        tensorboard_log_dir,
    )
    data_root = Path(args.data_root)
    radar_root = Path(args.radar_root)
    train_paths = _split_paths(
        data_root,
        "train",
        args.limit_train_samples,
        seed,
    )
    val_paths = _split_paths(
        data_root,
        "val",
        args.limit_val_samples,
        seed,
    )
    dataset_options = {
        "radar_root": radar_root,
        "data_root": data_root,
        "selector_config": selector_config,
        "use_pointpillars": model_config.pointpillars.enabled,
    }
    train_dataset = CoarseReconstructionDataset(
        train_paths,
        augmentation_config=augmentation_config,
        **dataset_options,
    )
    val_dataset = CoarseReconstructionDataset(val_paths, **dataset_options)
    if train_dataset.grid_geometry != val_dataset.grid_geometry:
        raise ValueError("Training and validation BEV geometry must match")
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "collate_fn": coarse_reconstruction_collate,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    model = CoarseReconstructionModel(
        model_config,
        grid_geometry=(
            train_dataset.grid_geometry
            if model_config.pointpillars.enabled
            else None
        ),
    ).to(device)
    geometry = train_dataset.grid_geometry
    if not math.isclose(
        geometry.pillar_size_x,
        geometry.pillar_size_y,
        rel_tol=1.0e-6,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            "Tolerance-aware occupancy requires square BEV cells; got "
            f"{geometry.pillar_size_x:.6f}m x "
            f"{geometry.pillar_size_y:.6f}m"
        )
    loss_fn = MaskedBEVReconstructionLoss(
        loss_config,
        bev_resolution_m=geometry.pillar_size_x,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training.get("learning_rate", 2.0e-4)),
        weight_decay=float(training.get("weight_decay", 1.0e-3)),
    )
    use_amp = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip = float(training.get("grad_clip", 1.0))
    save_conditioning_samples = int(training.get("save_conditioning_samples", 4))
    atomic_write_json(
        output_root / "resolved_config.json",
        {
            "model": model_config.to_dict(),
            "loss": asdict(loss_config),
            "selector": selector_config.__dict__,
            "augmentation": augmentation_config.to_dict(),
            "training": training,
            "args": vars(args),
            "grid_geometry": train_dataset.grid_geometry.to_dict(),
        },
    )
    print(f"Training samples: {len(train_dataset)}; validation: {len(val_dataset)}")
    print(f"Device: {device}; AMP: {use_amp}")
    print(f"Radar enabled: {radar_enabled}")
    print(f"Training augmentation enabled: {augmentation_config.enabled}")
    print(
        "Sensor representation: "
        + ("PointPillars" if model_config.pointpillars.enabled else "handcrafted BEV")
    )
    print("Reconstruction backbone: HRNet")
    print(f"Occupancy loss: {loss_config.occupancy.type}")
    if loss_config.occupancy.type == "tolerance_aware":
        print(
            "Tolerance-aware occupancy: "
            f"exact={loss_config.occupancy.exact_weight:g}, "
            "tolerant_recall="
            f"{loss_config.occupancy.tolerant_recall_weight:g}, "
            f"far_fp={loss_config.occupancy.far_fp_weight:g}, "
            f"requested_radius={loss_config.occupancy.tolerance_radius_m:.3f}m, "
            f"cell_radius={loss_fn.tolerance_radius_cells}, "
            "effective_axis_radius="
            f"{loss_fn.tolerance_radius_cells * loss_fn.bev_resolution_m:.3f}m"
        )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Trainable parameters: {trainable_parameters:,}")
    if model_config.pointpillars.enabled:
        geometry = train_dataset.grid_geometry
        print(
            "PointPillars voxelization: batched native PyTorch "
            f"on {device.type.upper()}"
        )
        print(
            "PointPillars grid: "
            f"{geometry.height}x{geometry.width}; "
            f"pillar={geometry.pillar_size_x:.3f}m x "
            f"{geometry.pillar_size_y:.3f}m"
        )
    history = []
    best_validation = float("inf")
    best_tolerant_iou = float("-inf")
    active_fraction_profile = None
    for epoch in range(1, epochs + 1):
        train_active_fractions = [] if epoch == 1 else None
        val_active_fractions = [] if epoch == 1 else None
        _synchronize_device(device)
        train_started = time.perf_counter()
        train_stats, train_shapes = _run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=grad_clip,
            use_amp=use_amp,
            active_fraction_samples=train_active_fractions,
            radar_enabled=radar_enabled,
            profile_first_batch=epoch == 1,
        )
        _synchronize_device(device)
        train_seconds = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        with torch.no_grad():
            val_stats, val_shapes = _run_epoch(
                model,
                val_loader,
                loss_fn,
                device,
                use_amp=use_amp,
                conditioning_callback=lambda batch, outputs: _save_conditioning(
                    output_root,
                    epoch,
                    batch,
                    outputs,
                    save_conditioning_samples,
                ),
                active_fraction_samples=val_active_fractions,
                radar_enabled=radar_enabled,
                profile_first_batch=epoch == 1,
            )
        _synchronize_device(device)
        validation_seconds = time.perf_counter() - validation_started
        epoch_seconds = train_seconds + validation_seconds
        if epoch == 1:
            train_active_summary = _summarize_active_fractions(
                train_active_fractions
            )
            val_active_summary = _summarize_active_fractions(val_active_fractions)
            combined_active_summary = _summarize_active_fractions(
                train_active_fractions + val_active_fractions
            )
            active_fraction_profile = {
                "definition": (
                    "Per-sample mean of reconstruction_mask OR halo_mask over "
                    "the complete BEV grid"
                ),
                "train": train_active_summary,
                "validation": val_active_summary,
                "combined": combined_active_summary,
                "recommendation": _active_fraction_recommendation(
                    combined_active_summary
                ),
            }
            atomic_write_json(
                output_root / "active_fraction_profile.json",
                active_fraction_profile,
            )
            atomic_write_json(
                output_root / "debug_batch_shapes.json",
                {"train": train_shapes, "validation": val_shapes},
            )
            print("Debug tensor shapes:", json.dumps(val_shapes, indent=2))
            print(
                "Active-mask coverage: "
                f"median={combined_active_summary['median']:.2%}, "
                f"p90={combined_active_summary['p90']:.2%}, "
                f"max={combined_active_summary['maximum']:.2%}"
            )
            print(
                "Architecture recommendation: "
                f"{active_fraction_profile['recommendation']}"
            )
        row = {
            "epoch": epoch,
            "runtime/train_seconds": train_seconds,
            "runtime/validation_seconds": validation_seconds,
            "runtime/epoch_seconds": epoch_seconds,
        }
        row.update({f"train/{key}": value for key, value in train_stats.items()})
        row.update({f"val/{key}": value for key, value in val_stats.items()})
        if epoch == 1:
            for split, summary in (
                ("train", train_active_summary),
                ("val", val_active_summary),
                ("combined", combined_active_summary),
            ):
                row.update(
                    {
                        f"active_fraction/{split}_{key}": value
                        for key, value in summary.items()
                    }
                )
        history.append(row)
        _write_tensorboard_epoch(
            tensorboard_writer,
            row,
            optimizer,
            epoch,
        )
        summary_lines = [
            "",
            (
                f"Epoch {epoch:03d}/{epochs:03d}  |  "
                f"{epoch_seconds:.1f}s "
                f"(train {train_seconds:.1f}s, val {validation_seconds:.1f}s)"
            ),
            (
                f"  Train  loss={train_stats['loss']:.6f}  "
                "exact_IoU="
                f"{train_stats['coarse_occupancy_exact_iou']:.3%}"
            ),
            "  Validation loss      total       occupancy   density     height",
            (
                f"                      {val_stats['loss']:10.6f}  "
                f"{val_stats['loss_occupancy']:10.6f}  "
                f"{val_stats['loss_density']:10.6f}  "
                f"{val_stats['loss_height']:10.6f}"
            ),
        ]
        if loss_config.occupancy.type == "tolerance_aware":
            summary_lines.extend(
                [
                    "  Occupancy terms         exact   tolerant-recall      far-FP",
                    (
                        "                      "
                        f"{val_stats['loss_occupancy_exact']:10.6f}  "
                        f"{val_stats['loss_occupancy_tolerant_recall']:16.6f}  "
                        f"{val_stats['loss_occupancy_far_fp']:10.6f}"
                    ),
                ]
            )
        summary_lines.extend(
            [
                "  Exact occupancy           precision     recall         F1        IoU",
                (
                    "                      "
                    f"{val_stats['coarse_occupancy_exact_precision']:10.3%}  "
                    f"{val_stats['coarse_occupancy_exact_recall']:9.3%}  "
                    f"{val_stats['coarse_occupancy_exact_f1']:9.3%}  "
                    f"{val_stats['coarse_occupancy_exact_iou']:9.3%}"
                ),
                "  Tolerant occupancy        precision     recall         F1        IoU",
                (
                    "                      "
                    f"{val_stats['coarse_occupancy_tolerant_precision']:10.3%}  "
                    f"{val_stats['coarse_occupancy_tolerant_recall']:9.3%}  "
                    f"{val_stats['coarse_occupancy_tolerant_f1']:9.3%}  "
                    f"{val_stats['coarse_occupancy_tolerant_iou']:9.3%}"
                ),
                (
                    "  Validation quality  "
                    f"hallucination={val_stats['coarse_occupancy_hallucination_rate']:.3%}  "
                    f"height_MAE={val_stats['coarse_height_mae_m']:.3f}m  "
                    f"outside_change={val_stats['outside_mask_max_change']:.3e}"
                ),
            ]
        )
        if loss_config.observability_weighting.enabled:
            summary_lines.append(
                "  Observability       "
                f"empty={val_stats['mean_empty_observability_repair']:.3f}  "
                f"empty_weight={val_stats['mean_empty_occupancy_weight']:.3f}  "
                "high-observability hallucination="
                f"{val_stats['hallucination_rate_high_observability']:.3%}"
            )
        print("\n".join(summary_lines), flush=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "model_config": model_config.to_dict(),
            "loss_config": asdict(loss_config),
            "augmentation_config": augmentation_config.to_dict(),
            "active_fraction_profile": active_fraction_profile,
            "radar_enabled": radar_enabled,
            "grid_geometry": train_dataset.grid_geometry.to_dict(),
            "history": history,
        }
        atomic_torch_save(checkpoint, output_root / "last_checkpoint.pt")
        if val_stats["loss"] < best_validation:
            best_validation = val_stats["loss"]
            atomic_torch_save(checkpoint, output_root / "best_model.pt")
        validation_tolerant_iou = val_stats["coarse_occupancy_tolerant_iou"]
        if validation_tolerant_iou > best_tolerant_iou:
            best_tolerant_iou = validation_tolerant_iou
            atomic_torch_save(
                checkpoint,
                output_root / "best_tolerant_iou.pt",
            )
        write_csv_rows(
            output_root / "history.csv",
            history,
            fieldnames=list(history[0]),
        )
    if tensorboard_writer is not None:
        tensorboard_writer.close()


if __name__ == "__main__":
    main()
