from pathlib import Path
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from Fault_Localization_Model.sample_utils import (
    load_sample_metadata,
    sample_frame_identity,
)
from Fault_Localization_Model.visualization_utils import (
    add_label_above,
    add_reliability_colorbar,
    blue_red_reliability,
    draw_cell_boundaries,
    localization_match_overlay,
    make_grid_like,
    save_image,
    side_by_side,
)


def seed_everything(seed):
    if int(seed) < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    """Restore a checkpoint RNG snapshot; tolerate legacy checkpoints."""
    if not state:
        return False
    required = {"python", "numpy", "torch_cpu"}
    missing = required - set(state)
    if missing:
        raise ValueError(
            "Checkpoint RNG state is incomplete; missing: "
            + ", ".join(sorted(missing))
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # map_location="cuda" moves RNG snapshots onto CUDA too, but PyTorch's
    # RNG restoration APIs require CPU ByteTensors.
    torch_cpu_state = torch.as_tensor(
        state["torch_cpu"],
        dtype=torch.uint8,
        device="cpu",
    )
    torch.set_rng_state(torch_cpu_state)
    if "torch_cuda" in state and torch.cuda.is_available():
        torch_cuda_states = [
            torch.as_tensor(value, dtype=torch.uint8, device="cpu")
            for value in state["torch_cuda"]
        ]
        torch.cuda.set_rng_state_all(torch_cuda_states)
    return True


def resolve_device(requested):
    requested = str(requested)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested!r} was requested, but torch.cuda.is_available() "
            "is False. Install a driver-compatible CUDA build or use --device cpu."
        )
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid torch device {requested!r}") from exc
    if device.type == "cuda":
        device_index = (
            torch.cuda.current_device()
            if device.index is None
            else device.index
        )
        device_count = torch.cuda.device_count()
        if device_index >= device_count:
            raise ValueError(
                f"CUDA device index {device_index} is unavailable; "
                f"this process sees {device_count} CUDA device(s)."
            )
    return device


def require_checkpoint_semantics(checkpoint, expected_version, pipeline_name):
    """Reject exact resume when a checkpoint used different training behavior."""
    saved_version = checkpoint.get("training_semantics_version")
    if saved_version is None:
        raise ValueError(
            f"{pipeline_name} checkpoint predates training-semantics versioning "
            "and cannot be resumed safely after objective fixes. Start a fresh "
            "run or use a supported weights-only initialization."
        )
    try:
        saved_version = int(saved_version)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{pipeline_name} checkpoint has an invalid training-semantics "
            f"version: {saved_version!r}"
        ) from exc
    if saved_version != int(expected_version):
        raise ValueError(
            f"{pipeline_name} checkpoint uses training semantics version "
            f"{saved_version}, but this code requires version {expected_version}. "
            "Do not mix objectives in an exact resume."
        )


def require_checkpoint_args_match(saved_args, current_args, keys):
    """Reject an inexact resume when behavior-defining arguments changed."""
    for key in keys:
        if key not in saved_args:
            continue
        saved = saved_args[key]
        current = getattr(current_args, key)
        if isinstance(saved, (list, tuple)) or isinstance(current, (list, tuple)):
            saved_value = tuple(sorted(saved or []))
            current_value = tuple(sorted(current or []))
            matches = saved_value == current_value
        elif (
            isinstance(saved, (int, float))
            and not isinstance(saved, bool)
            and isinstance(current, (int, float))
            and not isinstance(current, bool)
        ):
            matches = math.isclose(
                float(saved),
                float(current),
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        else:
            matches = saved == current
        if not matches:
            raise ValueError(
                f"Checkpoint {key}={saved!r} does not match requested "
                f"{key}={current!r}. Use the original value for --resume; "
                "use a weights-only initialization when intentionally changing "
                "training behavior."
            )


def split_paths(paths, val_ratio, seed):
    """Split complete source-frame groups so augmented variants cannot leak."""
    paths = list(paths)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be strictly between 0 and 1, got {val_ratio}")
    groups = {}
    for path in paths:
        identity = sample_frame_identity(load_sample_metadata(path))
        groups.setdefault(identity, []).append(path)
    if len(groups) < 2:
        raise ValueError(
            "At least two distinct physical source frames are required for a "
            "leakage-free training/validation split"
        )

    identities = list(groups)
    random.Random(seed).shuffle(identities)
    val_group_count = min(
        len(identities) - 1,
        max(1, int(round(len(identities) * val_ratio))),
    )
    val_identities = set(identities[:val_group_count])
    train_paths = [
        path
        for identity, group_paths in groups.items()
        if identity not in val_identities
        for path in group_paths
    ]
    val_paths = [
        path
        for identity, group_paths in groups.items()
        if identity in val_identities
        for path in group_paths
    ]
    return train_paths, val_paths


def _split_paths(data_root, split, limit, seed):
    """Select samples from an existing dataset split."""
    split_root = Path(data_root) / split
    if not split_root.is_dir():
        raise FileNotFoundError(
            f"Required dataset split is missing: {split_root}"
        )

    paths = sorted(split_root.rglob("*.npz"))
    if limit is not None:
        if limit < 1:
            raise ValueError("Split sample limits must be positive")
        paths = random.Random(seed).sample(paths, k=min(limit, len(paths)))
    if not paths:
        raise FileNotFoundError(f"No samples were found under {split_root}")
    return paths


def _dice_loss(logits, target, eps=1e-6):
    prediction = torch.sigmoid(logits)
    intersection = torch.sum(prediction * target, dim=(1, 2, 3))
    union = torch.sum(prediction, dim=(1, 2, 3)) + torch.sum(
        target, dim=(1, 2, 3)
    )
    return 1.0 - torch.mean((2.0 * intersection + eps) / (union + eps))


def original_reliability_loss(logits, target, grid_size=100):
    prediction = torch.sigmoid(logits)
    weight = 1.0 + 5.0 * target
    prediction_grid = F.adaptive_avg_pool2d(
        prediction, output_size=(grid_size, grid_size)
    )
    target_grid = F.adaptive_avg_pool2d(
        target, output_size=(grid_size, grid_size)
    )
    return (
        F.binary_cross_entropy_with_logits(logits, target, weight=weight)
        + 0.75 * F.l1_loss(prediction, target)
        + 0.25 * F.mse_loss(prediction, target)
        + 0.20 * _dice_loss(logits, target)
        + 1.25 * F.l1_loss(prediction_grid, target_grid)
        + 0.50 * F.mse_loss(prediction_grid, target_grid)
    )


def save_curve(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(history["epoch"], history["train_loss"], label="train")
    axis.plot(history["epoch"], history["val_loss"], label="validation")
    axis.set(xlabel="Epoch", ylabel="Loss", title="Reliability map training")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_predictions(
    model,
    loader,
    output_root: Path,
    device,
    max_images,
    visual_grid_size=100,
    localization_threshold=0.5,
    localization_tolerance_m=0.20,
    target_fault_threshold=0.0,
):
    prediction_dir = output_root / "val_predictions"
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["x"].to(device)
            targets = batch["y"].cpu().numpy()
            predictions = torch.sigmoid(model(inputs)).cpu().numpy()
            for index in range(inputs.shape[0]):
                metadata = batch["metadata"][index]
                target = make_grid_like(
                    targets[index, 0],
                    grid_size=visual_grid_size,
                )
                prediction = make_grid_like(
                    predictions[index, 0],
                    grid_size=visual_grid_size,
                )
                target_rgb = draw_cell_boundaries(
                    blue_red_reliability(target),
                    grid_size=visual_grid_size,
                )
                prediction_rgb = draw_cell_boundaries(
                    blue_red_reliability(prediction),
                    grid_size=visual_grid_size,
                )
                match_rgb = draw_cell_boundaries(
                    localization_match_overlay(
                        target,
                        prediction,
                        metadata,
                        prediction_threshold=localization_threshold,
                        target_fault_threshold=target_fault_threshold,
                        tolerance_m=localization_tolerance_m,
                    ),
                    grid_size=visual_grid_size,
                )

                input_rgb = batch["rgb"][index]
                if input_rgb.shape[:2] != target_rgb.shape[:2]:
                    input_rgb = np.asarray(
                        Image.fromarray(input_rgb, mode="RGB").resize(
                            (target_rgb.shape[1], target_rgb.shape[0]),
                            Image.Resampling.BILINEAR,
                        )
                    )
                fault = metadata["fault"]
                severity = metadata["severity"]
                timestamp = metadata["timestamp"]
                tolerance_cm = int(round(localization_tolerance_m * 100.0))
                label = f"{fault} S{severity}"
                stem = f"{len(rows):04d}_{fault}_s{severity}_{timestamp}"

                comparison = side_by_side(
                    [
                        add_label_above(input_rgb, f"faulty BEV input: {label}"),
                        add_reliability_colorbar(
                            add_label_above(target_rgb, f"ideal reliability: {label}")
                        ),
                        add_reliability_colorbar(
                            add_label_above(
                                prediction_rgb,
                                f"learned reliability: {label}",
                            )
                        ),
                        add_label_above(
                            match_rgb,
                            f"{tolerance_cm}cm match: white=both cyan=pred ok "
                            "green=GT ok red=miss yellow=false",
                        ),
                    ]
                )
                heatmaps = add_reliability_colorbar(
                    side_by_side(
                        [
                            add_label_above(target_rgb, f"ideal heatmap: {label}"),
                            add_label_above(
                                prediction_rgb,
                                f"learned heatmap: {label}",
                            ),
                        ]
                    )
                )
                save_image(prediction_dir / f"{stem}_comparison.png", comparison)
                save_image(
                    prediction_dir / f"{stem}_ideal_vs_learned_heatmaps.png",
                    heatmaps,
                )
                save_image(
                    prediction_dir
                    / f"{stem}_localization_{tolerance_cm:03d}cm_overlay.png",
                    match_rgb,
                )
                save_image(
                    prediction_dir / f"{stem}_target_reliability.png",
                    target_rgb,
                )
                save_image(
                    prediction_dir / f"{stem}_pred_reliability.png",
                    prediction_rgb,
                )
                rows.append(
                    {
                        "fault": fault,
                        "severity": severity,
                        "timestamp": timestamp,
                        "mae": float(np.mean(np.abs(prediction - target))),
                        "mean_pred_unreliability": float(np.mean(prediction)),
                        "mean_target_unreliability": float(np.mean(target)),
                    }
                )
                if len(rows) >= max_images:
                    return rows
    return rows
