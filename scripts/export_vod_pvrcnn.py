"""Export matched VoD clean/faulty/reconstructed scans for OpenPCDet PV-RCNN.

Run on Linux with the VoD range-view cache and an installed OpenPCDet checkout.
All three conditions use identical frame IDs, labels, and forward-FOV policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from Fault_Localization_Model.vod_dataset.vod_io import resolve_vod_public_root
from models.two_stage_reconstruction_head.voxelization.inputs import (
    load_clean_lidar_from_metadata, read_sample_metadata, radar_cache_path,
)


CLASSES = {"Car": "Car", "Pedestrian": "Pedestrian", "Cyclist": "Cyclist",
           "bicycle": "Cyclist"}
CONDITIONS = ("clean", "faulty", "reconstructed")


def normalized_labels(source: Path) -> str:
    """Drop VoD's optional track ID and map bicycle to KITTI Cyclist."""
    foreground, ignored = [], []
    for line in source.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 15:
            raise ValueError(f"Malformed VoD label in {source}: {line!r}")
        fields[0] = CLASSES.get(fields[0], "DontCare")
        (ignored if fields[0] == "DontCare" else foreground).append(" ".join(fields[:15]))
    lines = foreground + ignored  # OpenPCDet assumes ignored boxes are last.
    return "\n".join(lines) + ("\n" if lines else "")


def detector_points(lidar: np.ndarray, radar: np.ndarray | None, *,
                    forward_only: bool) -> np.ndarray:
    """Return KITTI xyzi points; radar uses RCS as the fourth feature."""
    lidar = np.asarray(lidar, dtype=np.float32)
    if lidar.ndim != 2 or lidar.shape[1] != 4 or not np.isfinite(lidar).all():
        raise ValueError("Expected finite LiDAR [N,4] xyzi")
    if forward_only:
        lidar = lidar[lidar[:, 0] >= 0]
    if radar is None:
        result = np.ascontiguousarray(lidar)
        return result if len(result) else np.array([[0.01, 0.0, -2.9, 0.0]], dtype=np.float32)
    radar = np.asarray(radar, dtype=np.float32)
    if radar.ndim != 2 or radar.shape[1] != 5:
        raise ValueError("Expected aligned radar [N,5] xyz,RCS,velocity")
    radar = radar[np.isfinite(radar[:, :4]).all(axis=1)]
    if forward_only:
        radar = radar[radar[:, 0] >= 0]
    # This is deliberately simple early fusion, not a native radar backbone.
    # The fourth feature is RCS for radar and reflectivity for LiDAR.
    result = np.ascontiguousarray(np.concatenate((lidar, radar[:, :4]), axis=0))
    return result if len(result) else np.array([[0.01, 0.0, -2.9, 0.0]], dtype=np.float32)


def _sample_index(root: Path, split: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted((root / split).glob("*.npz")):
        meta = read_sample_metadata(path)
        if meta.get("dataset") != "View-of-Delft" or not meta.get("range_view_full_scan"):
            raise ValueError(f"Not a full-scan VoD range sample: {path}")
        frame_id = str(meta["frame_id"])
        if frame_id in result:
            raise ValueError(f"Multiple faults for {split}/{frame_id}; select one per frame")
        result[frame_id] = path
    return result


def _checkpoint(args: argparse.Namespace):
    if args.checkpoint is None:
        return None
    import torch
    from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
    from models.two_stage_reconstruction_head.range_view.merge import MergeConfig
    from models.two_stage_reconstruction_head.range_view.model import RangeModelConfig, RangeViewReconstructor

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("representation") != "range_view":
        raise ValueError("Expected range-view reconstruction checkpoint")
    geometry = RangeGeometry(**checkpoint["geometry"])
    merge_config = MergeConfig(**checkpoint["merge_config"])
    if merge_config.allow_original_deletion:
        raise ValueError("This benchmark expects conservative, original-preserving reconstruction")
    model_config = RangeModelConfig(**checkpoint["model_config"])
    if model_config.use_fault_map_conditioning:
        raise ValueError("Fault-map conditioned checkpoints are not supported by this exporter")
    model = RangeViewReconstructor(model_config).to(args.device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    return (model, geometry, merge_config, int(checkpoint["epoch"]),
            float(checkpoint.get("radar_floor_band_m", 0.0)),
            bool(checkpoint.get("filter_radar_by_lidar_min", False)))


def _reconstruct(path: Path, radar_root: Path, loaded, device: str) -> np.ndarray:
    import torch
    from models.two_stage_reconstruction_head.range_view.data import load_range_sample
    from models.two_stage_reconstruction_head.range_view.merge import merge_reconstruction

    model, geometry, merge_config, _, radar_floor_band_m, filter_radar_by_lidar_min = loaded
    sample = load_range_sample(path, radar_root, geometry,
                               forward_only=merge_config.forward_only,
                               radar_floor_band_m=radar_floor_band_m,
                               filter_radar_by_lidar_min=filter_radar_by_lidar_min)
    with torch.inference_mode():
        output = model(torch.from_numpy(sample.features)[None].to(device))
    merged = merge_reconstruction(
        sample.faulty_points, sample.faulty_projection, geometry,
        output["add_probability"][0].cpu().numpy(),
        output["add_range_m"][0].cpu().numpy(),
        output["delete_probability"][0].cpu().numpy(),
        config=merge_config, radar_support=sample.radar_features[0],
    )
    return merged.points


def _write_bin(path: Path, points: np.ndarray) -> None:
    if path.exists():
        existing = np.fromfile(path, dtype="<f4")
        desired = np.asarray(points, dtype="<f4").reshape(-1)
        if np.array_equal(existing, desired):
            return
        raise FileExistsError(f"Existing detector cloud differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(points, dtype="<f4").tofile(path)


def _image_placeholder(source_root: Path, destination: Path, frame_id: str) -> None:
    """OpenPCDet's LiDAR KITTI loader requires .png to read image dimensions.

    Pixel values are never used (`GET_ITEM_LIST: ['points']`), so a small blank
    PNG with the original VoD JPG dimensions avoids recompressing camera data.
    """
    if destination.is_file():
        return
    from PIL import Image

    original = source_root / "image_2" / f"{frame_id}.jpg"
    if not original.is_file():
        raise FileNotFoundError(original)
    with Image.open(original) as image:
        size = image.size
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=0).save(destination, format="PNG", optimize=True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        help="Range-view checkpoint; omit to export clean/faulty only")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--with-radar", action="store_true",
                        help="Also export an early-fusion LiDAR+radar detector experiment")
    parser.add_argument("--limit", type=int, help="Smoke-test frames per split")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    public = resolve_vod_public_root(args.vod_root)
    loaded = _checkpoint(args)
    forward_only = loaded[2].forward_only if loaded else True
    modes = ("lidar", "lidar_radar") if args.with_radar else ("lidar",)
    selected: dict[str, dict[str, Path]] = {}
    for split in ("train", "val"):
        selected[split] = _sample_index(args.samples_root, split)
        if not selected[split]:
            raise FileNotFoundError(f"No {split} artifacts in {args.samples_root}")
        official = set((public / "lidar" / "ImageSets" / f"{split}.txt").read_text().split())
        if not set(selected[split]).issubset(official):
            raise ValueError(f"{split} cache contains frames outside official split")
        if args.limit:
            selected[split] = dict(list(sorted(selected[split].items()))[:args.limit])
    split_ids = {split: list(sorted(frames)) for split, frames in selected.items()}
    if set(split_ids["train"]) & set(split_ids["val"]):
        raise ValueError("Train/val frame overlap")

    for mode in modes:
        for condition in CONDITIONS if loaded else CONDITIONS[:2]:
            root = args.output_root / mode / condition
            (root / "ImageSets").mkdir(parents=True, exist_ok=True)
            for split, ids in split_ids.items():
                partition = "training"  # Official VoD train/val are in this tree.
                source = public / "lidar" / partition
                for frame_id in ids:
                    sample_path = selected[split][frame_id]
                    meta = read_sample_metadata(sample_path)
                    clean = load_clean_lidar_from_metadata(meta)
                    with np.load(sample_path, allow_pickle=False) as archive:
                        faulty = np.asarray(archive["faulty_lidar_points"], dtype=np.float32)
                    calibration = source / "calib" / f"{frame_id}.txt"
                    if not calibration.is_file():
                        raise FileNotFoundError(f"Missing calibration for {frame_id}")
                    calibration_dest = root / partition / "calib" / calibration.name
                    calibration_dest.parent.mkdir(parents=True, exist_ok=True)
                    if calibration_dest.exists():
                        if calibration_dest.read_bytes() != calibration.read_bytes():
                            raise ValueError(f"Existing calibration differs: {calibration_dest}")
                    else:
                        calibration_dest.write_bytes(calibration.read_bytes())
                    _image_placeholder(source, root / partition / "image_2" / f"{frame_id}.png",
                                       frame_id)
                    label_source = source / "label_2" / f"{frame_id}.txt"
                    if not label_source.is_file():
                        raise FileNotFoundError(f"Missing labels for {frame_id}")
                    label_dest = root / partition / "label_2" / f"{frame_id}.txt"
                    label_dest.parent.mkdir(parents=True, exist_ok=True)
                    label_text = normalized_labels(label_source)
                    if label_dest.exists():
                        if label_dest.read_text(encoding="utf-8") != label_text:
                            raise ValueError(f"Existing label differs: {label_dest}")
                    else:
                        label_dest.write_text(label_text, encoding="utf-8")
                    radar = None
                    if mode == "lidar_radar":
                        with np.load(radar_cache_path(args.radar_root, meta), allow_pickle=False) as archive:
                            radar = np.asarray(archive["radar_points"], dtype=np.float32)
                    if condition == "clean" or split == "train":
                        points = clean
                    elif condition == "faulty":
                        points = faulty
                    else:
                        points = _reconstruct(sample_path, args.radar_root, loaded, args.device)
                    dest = root / partition / "velodyne" / f"{frame_id}.bin"
                    _write_bin(dest, detector_points(points, radar, forward_only=forward_only))
                    if condition == "clean":
                        print(f"{mode} {split}: {frame_id}", flush=True)
            for split, ids in split_ids.items():
                (root / "ImageSets" / f"{split}.txt").write_text(
                    "\n".join(ids) + "\n", encoding="utf-8")
            (root / "ImageSets" / "trainval.txt").write_text(
                "\n".join(split_ids["train"] + split_ids["val"]) + "\n", encoding="utf-8")
            (root / "ImageSets" / "test.txt").write_text("", encoding="utf-8")
            manifest = {"mode": mode, "condition": condition, "splits": {
                split: len(ids) for split, ids in split_ids.items()},
                "forward_only": forward_only, "checkpoint": str(args.checkpoint) if loaded else None,
                "checkpoint_epoch": loaded[3] if loaded else None,
                "reconstruction_radar_floor_band_m": loaded[4] if loaded else 0.0,
                "reconstruction_filter_radar_by_lidar_min": loaded[5] if loaded else False,
                "radar_policy": "aligned_xyz_plus_rcs_as_intensity" if mode == "lidar_radar" else None,
                "empty_cloud_sentinel": [0.01, 0.0, -2.9, 0.0]}
            (root / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "splits": {
        key: len(value) for key, value in split_ids.items()}, "modes": modes}, indent=2))


if __name__ == "__main__":
    main()
