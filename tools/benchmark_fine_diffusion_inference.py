"""Benchmark end-to-end Fine Diffusion inference on identical VoD samples.

Model/checkpoint construction is intentionally excluded.  The primary latency starts
when a collated faulty sample is requested from the loader and ends only after the
final reconstructed BEV has finished on the accelerator.  A model-ready breakdown
also reports loader wait, host-to-device transfer, frozen coarse reconstruction, and
Fine Diffusion refinement.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Fault_Localization_Model.io_utils import atomic_write_json
from models.Fault_Localization.training_utils import (
    _split_paths,
    resolve_device,
    seed_everything,
)
from models.two_stage_reconstruction_head import (
    BEVChannelNormalization,
    CoarseReconstructionDataset,
    FineDiffusionRefiner,
    FrozenCoarseFineDiffusionPipeline,
    ResidualChannelNormalization,
    coarse_reconstruction_collate,
    load_frozen_coarse_model,
    validate_fine_diffusion_checkpoint_compatibility,
)
from models.two_stage_reconstruction_head.coarse_reconstruction.evaluate_coarse_by_fault import (
    _load_selector_config,
    _move_batch,
)
from models.two_stage_reconstruction_head.diffusion_process.evaluate_fine_diffusion_by_fault import (
    _diffusion_config_from_checkpoint,
    _normalizer_from_fine_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--coarse-checkpoint", type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--radar-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fine-config", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--benchmark-batches", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument(
        "--inference-bucket-multiple",
        type=int,
        default=32,
        help="Round local crop padding to this fixed shape multiple; 0 disables.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the Fine backbone with torch.compile for inference.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument(
        "--profile-one-batch",
        action="store_true",
        help="Export a torch.profiler Chrome trace after warm-up.",
    )
    parser.add_argument(
        "--require-fine-pointpillars-conditioning",
        action="store_true",
        help=(
            "Reject checkpoints without Fine-stage PointPillars conditioning. "
            "Use this guard when comparing U-Net and Transformer fairly."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty timing sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "std_ms": statistics.pstdev(values),
        "min_ms": min(values),
        "p50_ms": _percentile(values, 50.0),
        "p90_ms": _percentile(values, 90.0),
        "p95_ms": _percentile(values, 95.0),
        "p99_ms": _percentile(values, 99.0),
        "max_ms": max(values),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_event(device: torch.device) -> torch.cuda.Event | None:
    if device.type != "cuda":
        return None
    return torch.cuda.Event(enable_timing=True)


def _elapsed_ms(
    start: torch.cuda.Event | None,
    end: torch.cuda.Event | None,
    fallback_seconds: float,
) -> float:
    if start is None or end is None:
        return fallback_seconds * 1000.0
    return float(start.elapsed_time(end))


def _load_pipeline(args: argparse.Namespace, device: torch.device):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "diffusion_state_dict" not in checkpoint or "diffusion_config" not in checkpoint:
        raise ValueError("Checkpoint does not contain a Fine Diffusion model")
    diffusion_config = _diffusion_config_from_checkpoint(
        checkpoint["diffusion_config"],
        checkpoint.get("fine_diffusion_architecture"),
    )
    validate_fine_diffusion_checkpoint_compatibility(checkpoint, diffusion_config)

    bev_normalizer = _normalizer_from_fine_config(args.fine_config, diffusion_config)
    bev_metadata = checkpoint.get("bev_normalization")
    if bev_metadata is not None:
        bev_normalizer = BEVChannelNormalization(
            means=bev_metadata["means"],
            stds=bev_metadata["stds"],
            epsilon=float(bev_metadata["epsilon"]),
            source=bev_metadata.get("source", "fine_diffusion_checkpoint"),
        )
    residual_metadata = checkpoint.get("residual_normalization")
    if residual_metadata is None:
        raise ValueError("Fine checkpoint has no residual normalization metadata")
    residual_normalizer = ResidualChannelNormalization(
        residual_metadata.get(
            "raw_channel_stds", residual_metadata.get("channel_stds")
        ),
        minimum_std=float(
            residual_metadata.get(
                "minimum_std", diffusion_config.minimum_residual_std
            )
        ),
        source=residual_metadata.get("source", "fine_diffusion_checkpoint"),
    )

    if diffusion_config.bypass_coarse_reconstruction:
        coarse = None
        use_pointpillars = False
        coarse_checkpoint_path = None
    else:
        recorded_coarse = checkpoint.get("coarse_checkpoint")
        if args.coarse_checkpoint is None and not recorded_coarse:
            raise ValueError("Fine checkpoint does not identify its coarse model")
        coarse_checkpoint_path = args.coarse_checkpoint or Path(recorded_coarse)
        coarse, _ = load_frozen_coarse_model(
            coarse_checkpoint_path,
            device,
            allow_pointpillars=True,
        )
        use_pointpillars = coarse.config.pointpillars_enabled

    diffusion = FineDiffusionRefiner(
        diffusion_config,
        bev_normalizer,
        residual_normalizer,
    ).to(device)
    diffusion.load_state_dict(checkpoint["diffusion_state_dict"], strict=True)
    pipeline = FrozenCoarseFineDiffusionPipeline(coarse, diffusion).to(device).eval()
    return checkpoint, diffusion_config, pipeline, use_pointpillars, coarse_checkpoint_path


def _run_inference(
    pipeline,
    inputs: dict[str, torch.Tensor],
    *,
    sampling_steps: int,
    device: torch.device,
    use_amp: bool,
):
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=use_amp,
    ):
        coarse_bev, coarse_outputs = pipeline.coarse_forward(
            inputs["faulty_bev"],
            inputs["radar_bev"],
            inputs["reconstruction_mask"],
            inputs["healthy_context_mask"],
            inputs["halo_mask"],
            faulty_lidar_points=inputs.get("faulty_lidar_points"),
            radar_points=inputs.get("radar_points"),
        )
        sampled = pipeline.sample(
            inputs["faulty_bev"],
            inputs["radar_bev"],
            inputs["reconstruction_mask"],
            inputs["healthy_context_mask"],
            inputs["halo_mask"],
            coarse_lidar_bev=coarse_bev,
            coarse_output=coarse_outputs,
            faulty_lidar_points=inputs.get("faulty_lidar_points"),
            radar_points=inputs.get("radar_points"),
            sampling_steps=sampling_steps,
        )
    return sampled["final_lidar_bev"]


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    if args.warmup_batches < 0 or args.benchmark_batches < 1:
        raise ValueError("warmup must be non-negative and benchmark batches positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    checkpoint, diffusion_config, pipeline, use_pointpillars, coarse_path = (
        _load_pipeline(args, device)
    )
    if (
        args.require_fine_pointpillars_conditioning
        and not diffusion_config.use_pointpillars_conditioning
    ):
        raise ValueError(
            "Checkpoint has Fine PointPillars conditioning disabled; it is not "
            "comparable to PointPillars-conditioned checkpoints"
        )
    bucket_multiple = pipeline.diffusion.configure_inference_bucket(
        args.inference_bucket_multiple
    )
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch build does not provide torch.compile")
        if diffusion_config.fine_backbone == "transformer":
            pipeline.diffusion.transformer = torch.compile(
                pipeline.diffusion.transformer,
                mode=args.compile_mode,
                dynamic=False,
            )
        else:
            pipeline.diffusion.unet = torch.compile(
                pipeline.diffusion.unet,
                mode=args.compile_mode,
                dynamic=False,
            )
    sampling_steps = args.sampling_steps or diffusion_config.sampling_steps
    requested_samples = (
        args.warmup_batches
        + args.benchmark_batches
        + int(args.profile_one_batch)
    ) * args.batch_size
    sample_paths = _split_paths(
        args.data_root,
        args.split,
        requested_samples,
        args.seed,
    )
    if len(sample_paths) < requested_samples:
        raise ValueError(
            "Dataset is too small for the requested warm-up, profile, and "
            "benchmark batches"
        )
    measured_start = (
        args.warmup_batches + int(args.profile_one_batch)
    ) * args.batch_size
    measured_paths = sample_paths[
        measured_start : measured_start
        + args.benchmark_batches * args.batch_size
    ]
    sample_fingerprint = hashlib.sha256(
        "\n".join(str(path) for path in measured_paths).encode("utf-8")
    ).hexdigest()
    dataset = CoarseReconstructionDataset(
        sample_paths,
        args.radar_root,
        data_root=args.data_root,
        selector_config=_load_selector_config(args.config),
        use_pointpillars=use_pointpillars,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=coarse_reconstruction_collate,
    )
    iterator = iter(loader)
    use_amp = device.type == "cuda" and not args.no_amp

    print(
        f"Benchmarking {diffusion_config.fine_backbone} on {device}; "
        f"batch={args.batch_size}; steps={sampling_steps}; "
        f"warm-up={args.warmup_batches}; measured={args.benchmark_batches}; "
        f"bucket={bucket_multiple}; compiled={args.compile}",
        flush=True,
    )
    with torch.inference_mode():
        for _ in range(args.warmup_batches):
            batch = next(iterator)
            inputs = _move_batch(batch, device)
            _run_inference(
                pipeline,
                inputs,
                sampling_steps=sampling_steps,
                device=device,
                use_amp=use_amp,
            )
            _synchronize(device)

        if args.profile_one_batch:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            batch = next(iterator)
            inputs = _move_batch(batch, device)
            args.output_root.mkdir(parents=True, exist_ok=True)
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
            ) as profiler:
                _run_inference(
                    pipeline,
                    inputs,
                    sampling_steps=sampling_steps,
                    device=device,
                    use_amp=use_amp,
                )
                _synchronize(device)
            profiler.export_chrome_trace(
                str(args.output_root / "inference_profile.json")
            )
            sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
            print(
                profiler.key_averages().table(
                    sort_by=sort_by,
                    row_limit=25,
                ),
                flush=True,
            )

        timings = {
            "end_to_end": [],
            "loader_wait": [],
            "model_ready_pipeline": [],
            "host_to_device": [],
            "coarse": [],
            "fine": [],
        }
        samples_measured = 0
        for batch_index in range(args.benchmark_batches):
            _synchronize(device)
            end_to_end_start = time.perf_counter()
            loader_start = end_to_end_start
            batch = next(iterator)
            loader_end = time.perf_counter()

            transfer_start_event = _cuda_event(device)
            transfer_end_event = _cuda_event(device)
            coarse_end_event = _cuda_event(device)
            fine_end_event = _cuda_event(device)
            if transfer_start_event is not None:
                transfer_start_event.record()
            transfer_wall_start = time.perf_counter()
            inputs = _move_batch(batch, device)
            transfer_wall_end = time.perf_counter()
            if transfer_end_event is not None:
                transfer_end_event.record()

            coarse_wall_start = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                coarse_bev, coarse_outputs = pipeline.coarse_forward(
                    inputs["faulty_bev"],
                    inputs["radar_bev"],
                    inputs["reconstruction_mask"],
                    inputs["healthy_context_mask"],
                    inputs["halo_mask"],
                    faulty_lidar_points=inputs.get("faulty_lidar_points"),
                    radar_points=inputs.get("radar_points"),
                )
            coarse_wall_end = time.perf_counter()
            if coarse_end_event is not None:
                coarse_end_event.record()

            fine_wall_start = time.perf_counter()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                sampled = pipeline.sample(
                    inputs["faulty_bev"],
                    inputs["radar_bev"],
                    inputs["reconstruction_mask"],
                    inputs["healthy_context_mask"],
                    inputs["halo_mask"],
                    coarse_lidar_bev=coarse_bev,
                    coarse_output=coarse_outputs,
                    faulty_lidar_points=inputs.get("faulty_lidar_points"),
                    radar_points=inputs.get("radar_points"),
                    sampling_steps=sampling_steps,
                )
            _ = sampled["final_lidar_bev"]
            fine_wall_end = time.perf_counter()
            if fine_end_event is not None:
                fine_end_event.record()
            _synchronize(device)
            inference_end = time.perf_counter()

            current_batch_size = int(inputs["faulty_bev"].shape[0])
            samples_measured += current_batch_size
            divisor = float(current_batch_size)
            loader_ms = (loader_end - loader_start) * 1000.0 / divisor
            pipeline_ms = (inference_end - loader_end) * 1000.0 / divisor
            timings["loader_wait"].append(loader_ms)
            timings["model_ready_pipeline"].append(pipeline_ms)
            timings["end_to_end"].append(
                (inference_end - end_to_end_start) * 1000.0 / divisor
            )
            timings["host_to_device"].append(
                _elapsed_ms(
                    transfer_start_event,
                    transfer_end_event,
                    transfer_wall_end - transfer_wall_start,
                )
                / divisor
            )
            timings["coarse"].append(
                _elapsed_ms(
                    transfer_end_event,
                    coarse_end_event,
                    coarse_wall_end - coarse_wall_start,
                )
                / divisor
            )
            timings["fine"].append(
                _elapsed_ms(
                    coarse_end_event,
                    fine_end_event,
                    fine_wall_end - fine_wall_start,
                )
                / divisor
            )
            if (batch_index + 1) % 10 == 0 or batch_index + 1 == args.benchmark_batches:
                print(
                    f"Measured {batch_index + 1}/{args.benchmark_batches} batches",
                    end="\r" if batch_index + 1 < args.benchmark_batches else "\n",
                    flush=True,
                )

    distributions = {name: _distribution(values) for name, values in timings.items()}
    mean_end_to_end = distributions["end_to_end"]["mean_ms"]
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "coarse_checkpoint": str(coarse_path) if coarse_path is not None else None,
        "split": args.split,
        "seed": args.seed,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "amp": use_amp,
        "fine_backbone": diffusion_config.fine_backbone,
        "fine_pointpillars_conditioning": (
            diffusion_config.use_pointpillars_conditioning
        ),
        "transformer_inference_optimizations": (
            {
                "single_adaptive_normalization": True,
                "cached_window_layouts": True,
                "cached_radar_windows": True,
                "cached_radar_key_values": True,
                "fused_self_attention_qkv": True,
            }
            if diffusion_config.fine_backbone == "transformer"
            else None
        ),
        "architecture": checkpoint.get("fine_diffusion_architecture", {}),
        "sampling_steps": sampling_steps,
        "inference_bucket_multiple": bucket_multiple,
        "compiled": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "benchmark_batches": args.benchmark_batches,
        "samples_measured": samples_measured,
        "measured_sample_fingerprint_sha256": sample_fingerprint,
        "comparison_signature": {
            "data_root": str(args.data_root.resolve()),
            "radar_root": str(args.radar_root.resolve()),
            "coarse_checkpoint": (
                str(Path(coarse_path).resolve())
                if coarse_path is not None
                else None
            ),
            "split": args.split,
            "seed": args.seed,
            "sample_fingerprint_sha256": sample_fingerprint,
            "batch_size": args.batch_size,
            "sampling_steps": sampling_steps,
            "amp": use_amp,
            "inference_bucket_multiple": bucket_multiple,
            "fine_pointpillars_conditioning": (
                diffusion_config.use_pointpillars_conditioning
            ),
            "compiled": args.compile,
            "compile_mode": args.compile_mode if args.compile else None,
        },
        "parameters": {
            "fine": sum(parameter.numel() for parameter in pipeline.diffusion.parameters()),
            "coarse": (
                sum(parameter.numel() for parameter in pipeline.coarse_model.parameters())
                if pipeline.coarse_model is not None
                else 0
            ),
        },
        "latency_per_sample": distributions,
        "end_to_end_throughput_samples_per_second": 1000.0 / mean_end_to_end,
        "scope": {
            "included": [
                "dataset loader wait and collation",
                "host-to-device transfer",
                "frozen coarse reconstruction",
                "Fine Diffusion conditioning, crop extraction, all refinement steps",
                "final BEV assembly",
            ],
            "excluded": ["Python import", "checkpoint loading", "model construction"],
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "inference_timing.json", result)

    print()
    print("END-TO-END INFERENCE LATENCY PER SAMPLE")
    print(f"Backbone: {diffusion_config.fine_backbone}")
    print(f"Samples:  {samples_measured}")
    print(f"Steps:    {sampling_steps}")
    print()
    print(f"{'Stage':<24} {'Mean':>10} {'P50':>10} {'P95':>10} {'P99':>10}")
    print("-" * 68)
    for name in (
        "end_to_end",
        "loader_wait",
        "model_ready_pipeline",
        "host_to_device",
        "coarse",
        "fine",
    ):
        values = distributions[name]
        print(
            f"{name:<24} {values['mean_ms']:9.2f}ms "
            f"{values['p50_ms']:9.2f}ms {values['p95_ms']:9.2f}ms "
            f"{values['p99_ms']:9.2f}ms"
        )
    print()
    print(
        "End-to-end throughput: "
        f"{result['end_to_end_throughput_samples_per_second']:.2f} samples/s"
    )
    print(f"Saved: {args.output_root / 'inference_timing.json'}")


if __name__ == "__main__":
    main()
