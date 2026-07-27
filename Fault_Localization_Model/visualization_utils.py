from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from Fault_Localization_Model.heatmap_metrics import one_to_one_match_masks


def blue_red_reliability(unreliability):
    reliability = 1.0 - np.clip(unreliability, 0.0, 1.0)
    rgb = np.zeros((*reliability.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip((1.0 - reliability) * 255, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(reliability * 255, 0, 255).astype(np.uint8)
    return rgb


def make_grid_like(values, grid_size=100):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError(f"values must be a non-empty 2D map, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("values contains non-finite entries")
    if (
        not isinstance(grid_size, (int, np.integer))
        or grid_size < 1
        or grid_size > min(values.shape)
    ):
        raise ValueError(
            f"grid_size must be between 1 and {min(values.shape)}, got {grid_size}"
        )
    tensor = torch.from_numpy(values)[None, None]
    pooled = F.adaptive_avg_pool2d(tensor, output_size=(grid_size, grid_size))
    return F.interpolate(pooled, size=values.shape, mode="nearest")[0, 0].numpy()


def draw_cell_boundaries(rgb, grid_size=100):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape [H,W,3], got {rgb.shape}")
    output = rgb.copy()
    height, width = output.shape[:2]
    if (
        not isinstance(grid_size, (int, np.integer))
        or grid_size < 1
        or grid_size > min(height, width)
    ):
        raise ValueError(
            f"grid_size must be between 1 and {min(height, width)}, got {grid_size}"
        )
    if grid_size >= min(height, width):
        return output
    line_color = np.array([18, 18, 18], dtype=np.uint8)
    row_boundaries = np.ceil(
        np.arange(1, grid_size, dtype=np.float64) * height / grid_size
    ).astype(np.int64)
    col_boundaries = np.ceil(
        np.arange(1, grid_size, dtype=np.float64) * width / grid_size
    ).astype(np.int64)
    output[row_boundaries, :] = line_color
    output[:, col_boundaries] = line_color
    return output


def _font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def add_label_above(rgb, label):
    pad = 28
    canvas = np.zeros((rgb.shape[0] + pad, rgb.shape[1], 3), dtype=np.uint8)
    canvas[:pad] = np.array([18, 18, 18], dtype=np.uint8)
    canvas[pad:] = rgb
    image = Image.fromarray(canvas, mode="RGB")
    ImageDraw.Draw(image).text((8, 7), label, fill=(255, 255, 255), font=_font(14))
    return np.asarray(image)


def add_reliability_colorbar(rgb):
    bar_width = 34
    label_width = 104
    pad = 8
    height, width = rgb.shape[:2]
    canvas = np.zeros((height, width + bar_width + label_width + pad, 3), dtype=np.uint8)
    canvas[:, :width] = rgb
    x0 = width + pad

    values = np.linspace(1.0, 0.0, height, dtype=np.float32)
    bar = np.zeros((height, bar_width, 3), dtype=np.uint8)
    bar[..., 0] = np.clip((1.0 - values[:, None]) * 255, 0, 255).astype(np.uint8)
    bar[..., 2] = np.clip(values[:, None] * 255, 0, 255).astype(np.uint8)
    canvas[:, x0 : x0 + bar_width] = bar

    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = _font(13)
    text_x = x0 + bar_width + 8
    draw.text((text_x, 8), "Reliable", fill=(80, 160, 255), font=font)
    draw.text((text_x, 26), "1.0", fill=(80, 160, 255), font=font)
    draw.text((text_x, height // 2 - 9), "0.5", fill=(210, 120, 255), font=font)
    draw.text((text_x, height - 42), "0.0", fill=(255, 90, 90), font=font)
    draw.text((text_x, height - 24), "Unreliable", fill=(255, 90, 90), font=font)
    return np.asarray(image)


def localization_match_overlay(
    target,
    prediction,
    metadata,
    prediction_threshold=0.5,
    target_fault_threshold=0.0,
    tolerance_m=0.20,
):
    target_mask = target > target_fault_threshold
    prediction_mask = prediction >= prediction_threshold
    x_range = metadata.get("x_range", [0.0, float(target.shape[0])])
    y_range = metadata.get("y_range", [0.0, float(target.shape[1])])
    x_cell_size_m = (float(x_range[1]) - float(x_range[0])) / target.shape[0]
    y_cell_size_m = (float(y_range[1]) - float(y_range[0])) / target.shape[1]

    prediction_matched, target_matched = one_to_one_match_masks(
        prediction_mask,
        target_mask,
        x_cell_size_m,
        y_cell_size_m,
        tolerance_m,
    )

    rgb = np.zeros((*target.shape, 3), dtype=np.uint8)
    rgb[:] = np.array([0, 0, 55], dtype=np.uint8)
    rgb[target_mask & ~target_matched] = np.array([255, 0, 0], dtype=np.uint8)
    rgb[prediction_mask & ~prediction_matched] = np.array([255, 230, 0], dtype=np.uint8)
    rgb[target_matched] = np.array([0, 255, 90], dtype=np.uint8)
    rgb[prediction_matched] = np.array([0, 210, 255], dtype=np.uint8)
    rgb[target_matched & prediction_matched] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb


def side_by_side(images):
    max_height = max(image.shape[0] for image in images)
    padded = []
    for image in images:
        if image.shape[0] == max_height:
            padded.append(image)
            continue
        canvas = np.zeros((max_height, image.shape[1], 3), dtype=np.uint8)
        canvas[: image.shape[0]] = image
        padded.append(canvas)
    return np.concatenate(padded, axis=1)


def save_image(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)
