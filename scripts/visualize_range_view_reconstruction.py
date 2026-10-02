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
    """Write the full cloud, including measured or predicted intensity when present."""
    columns = 4 if points.shape[1] >= 4 else 3
    cloud = np.ascontiguousarray(points[:, :columns], dtype="<f4")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(cloud)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              + ("property float intensity\n" if columns == 4 else "")
              + "end_header\n")
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(cloud.tobytes())


def _save_interactive_html(path: Path, *, faulty: np.ndarray, clean: np.ndarray,
                           original: np.ndarray, generated: np.ndarray,
                           sample_name: str, fault: str, epoch: int,
                           max_plot_points: int) -> None:
    """Create a self-contained browser viewer with one synchronized 3D camera."""
    bounds = _shared_bounds(faulty, clean, original, generated)

    def xyz(points: np.ndarray) -> list[list[float]]:
        return np.round(_display_points(points, max_plot_points), 2).tolist()

    payload = {
        "sample": sample_name, "fault": fault, "epoch": epoch,
        "bounds": bounds,
        "panels": [
            {"name": "Faulty LiDAR", "count": len(faulty),
             "layers": [{"color": COLORS["faulty"], "points": xyz(faulty)}]},
            {"name": "Clean LiDAR", "count": len(clean),
             "layers": [{"color": COLORS["clean"], "points": xyz(clean)}]},
            {"name": "Reconstructed LiDAR", "count": len(original) + len(generated),
             "layers": [{"color": COLORS["original"], "points": xyz(original)},
                        {"color": COLORS["generated"], "points": xyz(generated)}]},
        ],
    }
    # Escaping '<' prevents a sample name from ending the JSON script element.
    data = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    page = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LiDAR reconstruction comparison</title>
<style>
body{margin:0;padding:20px;font:15px system-ui,sans-serif;background:#10151b;color:#eaf0f6}
header{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:12px}
h1{font-size:19px;font-weight:600;margin:0}p{margin:3px 0;color:#b8c5d0}
button{font:inherit;padding:7px 12px;color:#eaf0f6;background:#263340;border:1px solid #5a6d80;border-radius:6px}
.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
section{min-width:0;background:#19222c;border:1px solid #364756;border-radius:7px;padding:10px}
h2{font-size:16px;font-weight:600;margin:0 0 8px}canvas{display:block;width:100%;height:440px;touch-action:none;cursor:grab;background:#111a22}
canvas:active{cursor:grabbing}.note{margin-top:12px}
@media(max-width:900px){.panels{grid-template-columns:1fr}canvas{height:380px}}
</style>
<header><div><h1 id="title"></h1><p>Drag to rotate · wheel to zoom · Shift-drag to pan. All panels share one camera and scale.</p></div><button id="reset" type="button">Reset view</button></header>
<div id="panels" class="panels"></div><p class="note">Purple = retained LiDAR; blue = generated additions. Counts are full clouds; display points are capped for speed.</p>
<script id="cloud-data" type="application/json">__DATA__</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById('cloud-data').textContent);
  document.getElementById('title').textContent = `${data.sample} · ${data.fault} · checkpoint epoch ${data.epoch}`;
  const root = document.getElementById('panels');
  const canvases = data.panels.map((panel) => {
    const section = document.createElement('section');
    const heading = document.createElement('h2');
    heading.textContent = `${panel.name} (${panel.count.toLocaleString()} points)`;
    const canvas = document.createElement('canvas');
    canvas.setAttribute('aria-label', `${panel.name} rotatable point cloud`);
    section.append(heading, canvas); root.append(section);
    return canvas;
  });
  const bounds = data.bounds;
  const center = bounds.map(pair => (pair[0] + pair[1]) / 2);
  const spans = bounds.map(pair => pair[1] - pair[0]);
  const radius = Math.max(Math.hypot(...spans) / 2, 1);
  const initial = {yaw:-0.85,pitch:0.35,zoom:1,panX:0,panY:0};
  const view = {...initial};
  function project(point, width, height, scale) {
    const x=point[0]-center[0], y=point[1]-center[1], z=point[2]-center[2];
    const cy=Math.cos(view.yaw), sy=Math.sin(view.yaw);
    const cp=Math.cos(view.pitch), sp=Math.sin(view.pitch);
    const u=cy*x-sy*y, v=sy*x+cy*y;
    return [width/2+view.panX+u*scale,height/2+view.panY-(cp*z-sp*v)*scale];
  }
  function draw(canvas, panel) {
    const dpi=window.devicePixelRatio||1, width=canvas.clientWidth, height=canvas.clientHeight;
    const pixelWidth=Math.max(1,Math.round(width*dpi)), pixelHeight=Math.max(1,Math.round(height*dpi));
    if(canvas.width!==pixelWidth||canvas.height!==pixelHeight){canvas.width=pixelWidth;canvas.height=pixelHeight;}
    const ctx=canvas.getContext('2d'); ctx.setTransform(dpi,0,0,dpi,0,0);
    ctx.clearRect(0,0,width,height);
    const scale=0.43*Math.min(width,height)*view.zoom/radius;
    ctx.lineWidth=1;
    const axes=[[radius,0,0,'#e08181','X'],[0,radius,0,'#9bd3a9','Y'],[0,0,radius,'#9ebfea','Z']];
    const origin=project(center,width,height,scale);
    for(const [dx,dy,dz,color,label] of axes){
      const end=project([center[0]+dx,center[1]+dy,center[2]+dz],width,height,scale);
      ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(...end);ctx.stroke();
      ctx.fillStyle=color;ctx.fillText(label,end[0]+3,end[1]-3);
    }
    for(const layer of panel.layers){
      ctx.fillStyle=layer.color;
      for(const point of layer.points){
        const [px,py]=project(point,width,height,scale);
        if(px>=0&&px<width&&py>=0&&py<height) ctx.fillRect(px,py,1.7,1.7);
      }
    }
  }
  function drawAll(){canvases.forEach((canvas,index)=>draw(canvas,data.panels[index]));}
  let pending=false;
  function scheduleDraw(){
    if(pending)return;
    pending=true;
    requestAnimationFrame(()=>{pending=false;drawAll();});
  }
  const active=new Map();
  for(const canvas of canvases){
    canvas.addEventListener('pointerdown',event=>{
      canvas.setPointerCapture(event.pointerId);
      active.set(event.pointerId,{x:event.clientX,y:event.clientY});
    });
    canvas.addEventListener('pointermove',event=>{
      const prior=active.get(event.pointerId);if(!prior)return;
      const dx=event.clientX-prior.x,dy=event.clientY-prior.y;
      prior.x=event.clientX;prior.y=event.clientY;
      if(event.shiftKey){view.panX+=dx;view.panY+=dy;}
      else{view.yaw+=dx*0.008;view.pitch=Math.max(-1.55,Math.min(1.55,view.pitch+dy*0.008));}
      scheduleDraw();
    });
    canvas.addEventListener('pointerup',event=>active.delete(event.pointerId));
    canvas.addEventListener('pointercancel',event=>active.delete(event.pointerId));
    canvas.addEventListener('wheel',event=>{
      event.preventDefault();view.zoom=Math.max(0.15,Math.min(15,view.zoom*Math.exp(-event.deltaY*0.001)));
      scheduleDraw();
    },{passive:false});
  }
  document.getElementById('reset').addEventListener('click',()=>{Object.assign(view,initial);scheduleDraw();});
  window.addEventListener('resize',scheduleDraw); drawAll();
})();
</script></html>
"""
    path.write_text(page.replace("__DATA__", data), encoding="utf-8")


def _render_comparison(output_root: Path, *, faulty: np.ndarray, clean: np.ndarray,
                       original: np.ndarray, generated: np.ndarray,
                       sample_name: str, fault: str, epoch: int,
                       max_plot_points: int, show: bool) -> None:
    import matplotlib.pyplot as plt

    if show and plt.get_backend().lower().endswith("agg"):
        print("No interactive Matplotlib display; open the saved _interactive.html "
              "file in a browser to rotate the clouds.", flush=True)
        show = False

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
    parser.add_argument("--radar-floor-band-m", type=float,
                        help="Override checkpoint radar floor band; 0 disables it")
    parser.add_argument("--no-show", action="store_true", help="Save HTML, PNG and PLY without GUI windows")
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
    radar_floor_band_m = (checkpoint.get("radar_floor_band_m", 0.0)
                          if args.radar_floor_band_m is None else args.radar_floor_band_m)
    if not np.isfinite(radar_floor_band_m) or radar_floor_band_m < 0:
        raise ValueError("radar floor band must be finite and nonnegative")
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
            forward_only=merge_config.forward_only,
            radar_floor_band_m=radar_floor_band_m,
            filter_radar_by_lidar_min=bool(checkpoint.get("filter_radar_by_lidar_min", False)))
        with torch.inference_mode():
            prediction = model(torch.from_numpy(sample.features)[None].to(args.device))
        merged = merge_reconstruction(
            sample.faulty_points, sample.faulty_projection, geometry,
            prediction["add_probability"][0].cpu().numpy(),
            prediction["add_range_m"][0].cpu().numpy(),
            prediction["delete_probability"][0].cpu().numpy(),
            config=merge_config, radar_support=sample.radar_features[0],
            add_intensity=(prediction["add_intensity"][0].cpu().numpy()
                           if "add_intensity" in prediction else None),
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
            "radar_floor_band_m": radar_floor_band_m,
            "radar_floor_removed_points": sample.metadata["radar_floor_removed_points"],
            "radar_lidar_min_z_m": sample.metadata["radar_lidar_min_z_m"],
            "radar_below_lidar_removed_points": sample.metadata["radar_below_lidar_removed_points"],
            "max_plot_points_per_cloud": args.max_plot_points,
        }
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _save_interactive_html(
            destination / f"{path.stem}_interactive.html",
            faulty=sample.faulty_points, clean=sample.clean_points,
            original=sample.faulty_points[merged.retained_original_indices],
            generated=merged.generated_points, sample_name=path.stem,
            fault=str(metadata["fault"]), epoch=int(checkpoint["epoch"]),
            max_plot_points=args.max_plot_points,
        )
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
