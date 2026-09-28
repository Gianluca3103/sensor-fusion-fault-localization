"""Inspect checkpoint reconstructions in rotatable 3D and matched XYZ projections.

The viewer runs on CPU by default so it can inspect a checkpoint while a GPU
training job continues. Existing training runs retain only their last checkpoint;
saved PNGs from earlier validation epochs do not contain recoverable XYZ points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.two_stage_reconstruction_head.range_view.data import load_range_sample
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry
from models.two_stage_reconstruction_head.range_view.merge import MergeConfig, merge_reconstruction
from models.two_stage_reconstruction_head.range_view.model import RangeModelConfig, RangeViewReconstructor


COLORS = {"faulty": "#bb3434", "clean": "#2b8f58",
          "original": "#8051a7", "generated": "#167bbb"}


def _display_points(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points[:, :3]
    return points[np.linspace(0, len(points) - 1, maximum, dtype=np.int64), :3]


def _shared_bounds(*clouds: np.ndarray) -> tuple[tuple[float, float], ...]:
    occupied = [cloud[:, :3] for cloud in clouds if len(cloud)]
    if not occupied:
        return ((-1.0, 1.0),) * 3
    xyz = np.concatenate(occupied)
    lower = xyz.min(axis=0).astype(np.float64)
    upper = xyz.max(axis=0).astype(np.float64)
    span = upper - lower
    padding = np.maximum(span * 0.02, 0.25)
    return tuple((float(lo - pad), float(hi + pad))
                 for lo, hi, pad in zip(lower, upper, padding))


def _save_ply(path: Path, points: np.ndarray) -> None:
    """Write the full XYZ cloud as a portable binary PLY, not the plot subsample."""
    xyz = np.ascontiguousarray(points[:, :3], dtype="<f4")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\nend_header\n")
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(xyz.tobytes())


def _render_comparison(output_root: Path, *, faulty: np.ndarray, clean: np.ndarray,
                       original: np.ndarray, generated: np.ndarray,
                       sample_name: str, fault: str, epoch: int,
                       max_plot_points: int, show: bool) -> None:
    import matplotlib.pyplot as plt

    if show and plt.get_backend().lower().endswith("agg"):
        raise RuntimeError(
            "No interactive Matplotlib display is available. Use --no-show to "
            "export PNG/PLY, then open the PLY files in a point-cloud viewer."
        )

    faulty_plot = _display_points(faulty, max_plot_points)
    clean_plot = _display_points(clean, max_plot_points)
    original_plot = _display_points(original, max_plot_points)
    generated_plot = _display_points(generated, max_plot_points)
    bounds = _shared_bounds(faulty, clean, original, generated)
    panels = (
        ((faulty_plot, COLORS["faulty"]),),
        ((clean_plot, COLORS["clean"]),),
        ((original_plot, COLORS["original"]),
         (generated_plot, COLORS["generated"])),
    )
    names = (f"Faulty LiDAR ({len(faulty):,})", f"Clean LiDAR ({len(clean):,})",
             f"Reconstructed ({len(original) + len(generated):,}; +{len(generated):,})")
    title = f"{sample_name} | {fault} | checkpoint epoch {epoch}"

    figure = plt.figure(figsize=(16, 6), constrained_layout=True)
    axes = [figure.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    for axis, name, layers in zip(axes, names, panels):
        for points, color in layers:
            if len(points):
                axis.scatter(points[:, 0], points[:, 1], points[:, 2],
                             s=0.35, c=color, depthshade=False, rasterized=True)
        axis.set_title(name)
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_zlabel("Z (m)")
        axis.set_xlim(*bounds[0])
        axis.set_ylim(*bounds[1])
        axis.set_zlim(*bounds[2])
        axis.set_box_aspect(tuple(hi - lo for lo, hi in bounds), zoom=0.82)
        axis.view_init(elev=20, azim=-65)
    figure.suptitle(title)

    def synchronize_view(event) -> None:
        if event.inaxes not in axes:
            return
        source = event.inaxes
        for axis in axes:
            if axis is not source:
                axis.view_init(elev=source.elev, azim=source.azim)
                axis.set_xlim(source.get_xlim())
                axis.set_ylim(source.get_ylim())
                axis.set_zlim(source.get_zlim())
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("button_release_event", synchronize_view)
    figure.savefig(output_root / f"{sample_name}_rotatable_3d.png", dpi=150)

    projections = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    projection_figure, projection_axes = plt.subplots(
        3, 3, figsize=(16, 11), constrained_layout=True)
    for row, (horizontal, vertical, label) in enumerate(projections):
        for col, (name, layers) in enumerate(zip(names, panels)):
            axis = projection_axes[row, col]
            for points, color in layers:
                if len(points):
                    axis.scatter(points[:, horizontal], points[:, vertical],
                                 s=0.35, c=color, alpha=0.75, rasterized=True)
            axis.set_xlim(*bounds[horizontal])
            axis.set_ylim(*bounds[vertical])
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(f"{name} | {label}")
            axis.set_xlabel(f"{label[0]} (m)")
            axis.set_ylabel(f"{label[1]} (m)")
            axis.grid(alpha=0.15)
    projection_figure.suptitle(title)
    projection_figure.savefig(output_root / f"{sample_name}_xy_xz_yz.png", dpi=150)

    if show:
        print("Drag a 3D panel to rotate; its view synchronizes to the other panels. "
              "Close both windows to advance to the next sample.", flush=True)
        plt.show()
    plt.close(figure)
    plt.close(projection_figure)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--radar-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--sample-indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-plot-points", type=int, default=10000)
    parser.add_argument("--device", default="cpu", help="CPU avoids competing with GPU training")
    parser.add_argument("--fault-map-root", type=Path)
    parser.add_argument("--no-show", action="store_true", help="Save PNG and PLY without opening GUI windows")
    args = parser.parse_args()
    if args.max_plot_points < 1 or any(index < 0 for index in args.sample_indices):
        parser.error("sample indices and max plot points must be nonnegative/positive")
    return args


def main() -> None:
    args = _arguments()
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("representation") != "range_view":
        raise ValueError("Expected a range-view reconstruction checkpoint")
    geometry = RangeGeometry(**checkpoint["geometry"])
    model_config = RangeModelConfig(**checkpoint["model_config"])
    merge_config = MergeConfig(**checkpoint["merge_config"])
    if model_config.use_fault_map_conditioning and args.fault_map_root is None:
        raise ValueError("Checkpoint uses fault-map conditioning; supply --fault-map-root")
    model = RangeViewReconstructor(model_config).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    paths = sorted((args.data_root / args.split).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No samples in {args.data_root / args.split}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for index in args.sample_indices:
        if index >= len(paths):
            raise IndexError(f"sample index {index} is outside 0..{len(paths) - 1}")
        path = paths[index]
        sample = load_range_sample(
            path, args.radar_root, geometry, fault_map_root=args.fault_map_root,
            forward_only=merge_config.forward_only)
        with torch.inference_mode():
            prediction = model(torch.from_numpy(sample.features)[None].to(args.device))
        merged = merge_reconstruction(
            sample.faulty_points, sample.faulty_projection, geometry,
            prediction["add_probability"][0].cpu().numpy(),
            prediction["add_range_m"][0].cpu().numpy(),
            prediction["delete_probability"][0].cpu().numpy(),
            config=merge_config, radar_support=sample.radar_features[0],
        )
        destination = args.output_root / path.stem
        destination.mkdir(parents=True, exist_ok=True)
        for label, points in (
            ("faulty", sample.faulty_points), ("clean", sample.clean_points),
            ("generated", merged.generated_points), ("reconstructed", merged.points),
        ):
            _save_ply(destination / f"{label}.ply", points)
        metadata = {
            "sample": str(path), "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "fault": sample.metadata.get("fault", "unknown"),
            "faulty_points": len(sample.faulty_points),
            "clean_points": len(sample.clean_points),
            "generated_points": len(merged.generated_points),
            "reconstructed_points": len(merged.points),
            "deleted_original_points": len(merged.deleted_original_indices),
            "max_plot_points_per_cloud": args.max_plot_points,
        }
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _render_comparison(
            destination, faulty=sample.faulty_points, clean=sample.clean_points,
            original=sample.faulty_points[merged.retained_original_indices],
            generated=merged.generated_points, sample_name=path.stem,
            fault=str(metadata["fault"]), epoch=int(checkpoint["epoch"]),
            max_plot_points=args.max_plot_points, show=not args.no_show,
        )
        print(f"{path.name}: {metadata['fault']} | faulty {metadata['faulty_points']:,} | "
              f"clean {metadata['clean_points']:,} | reconstructed "
              f"{metadata['reconstructed_points']:,} | {destination}", flush=True)


if __name__ == "__main__":
    main()
