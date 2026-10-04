"""Inspect checkpoint reconstructions in rotatable 3D and matched XYZ projections.

The viewer runs on CPU by default so it can inspect a checkpoint while a GPU
training job continues. Existing training runs retain only their last checkpoint;
saved PNGs from earlier validation epochs do not contain recoverable XYZ points.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from Fault_Localization_Model.vod_dataset.vod_io import load_vod_lidar_to_camera
from models.two_stage_reconstruction_head.range_view.data import load_range_sample
from models.two_stage_reconstruction_head.range_view.geometry import RangeGeometry, angular_indices
from models.two_stage_reconstruction_head.range_view.merge import MergeConfig, merge_reconstruction
from models.two_stage_reconstruction_head.range_view.model import RangeModelConfig, RangeViewReconstructor
from models.two_stage_reconstruction_head.range_view.object_targets import vod_label_paths


COLORS = {"faulty": "#bb3434", "clean": "#2b8f58",
          "original": "#8051a7", "generated": "#167bbb",
          "radar": "#ffbf47"}
BOX_COLORS = {"Car": "#4ce0ed", "Pedestrian": "#ff77be", "Cyclist": "#d6eb62"}
BOX_EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))


def _load_annotated_boxes(metadata: dict) -> list[dict]:
    """Read VoD ground-truth boxes and transform their corners into LiDAR XYZ."""
    if str(metadata.get("dataset", "")).strip().lower() not in {
        "view-of-delft", "view of delft", "vod"
    }:
        return []
    labels_path, calibration_path = vod_label_paths(metadata)
    lidar_from_camera = np.linalg.inv(load_vod_lidar_to_camera(calibration_path))
    boxes = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        name = "Cyclist" if fields[0] == "bicycle" else fields[0]
        if name not in BOX_COLORS:
            continue
        if len(fields) < 15:
            raise ValueError(f"Malformed VoD box in {labels_path}: {line!r}")
        height, width, length, x, y, z, yaw = map(float, fields[8:15])
        values = np.asarray((height, width, length, x, y, z, yaw))
        if not np.isfinite(values).all() or min(height, width, length) <= 0:
            raise ValueError(f"Invalid VoD box in {labels_path}: {line!r}")
        # VoD/KITTI boxes use camera Y as the bottom and yaw around camera Y.
        local = np.asarray([(sx * width / 2, sy * height, sz * length / 2)
                            for sx in (-1, 1) for sy in (-1, 0) for sz in (-1, 1)])
        camera = np.empty_like(local)
        camera[:, 0] = x + np.cos(yaw) * local[:, 0] + np.sin(yaw) * local[:, 2]
        camera[:, 1] = y + local[:, 1]
        camera[:, 2] = z - np.sin(yaw) * local[:, 0] + np.cos(yaw) * local[:, 2]
        lidar = camera @ lidar_from_camera[:3, :3].T + lidar_from_camera[:3, 3]
        if np.max(lidar[:, 0]) < 0:
            continue  # This viewer displays forward LiDAR and radar only.
        boxes.append({"name": name, "color": BOX_COLORS[name], "corners": lidar})
    return boxes


def _radar_box_stats(radar_points: np.ndarray, boxes: list[dict]) -> dict:
    """Count radar returns in the union of oriented ground-truth 3D boxes."""
    inside = np.zeros(len(radar_points), dtype=bool)
    for box in boxes:
        corners = np.asarray(box["corners"], dtype=np.float64)
        origin = corners[0]
        axes = corners[[4, 2, 1]] - origin
        squared_lengths = np.sum(axes * axes, axis=1)
        coordinates = (radar_points[:, :3] - origin) @ axes.T / squared_lengths
        inside |= np.all((coordinates >= -1e-6) & (coordinates <= 1 + 1e-6), axis=1)
    total = len(radar_points)
    inside_count = int(inside.sum())
    outside_count = total - inside_count
    return {
        "radar_returns": total, "inside_boxes": inside_count,
        "outside_boxes": outside_count,
        "inside_percent": 100 * inside_count / total if total else 0.0,
        "outside_percent": 100 * outside_count / total if total else 0.0,
        "box_count": len(boxes),
    }


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
                           max_plot_points: int,
                           radar: np.ndarray | None = None,
                           boxes: list[dict] | None = None,
                           radar_stats: dict | None = None) -> None:
    """Create a self-contained browser viewer with one synchronized 3D camera."""
    bounds = _shared_bounds(faulty, clean, original, generated)
    radar = np.empty((0, 3), dtype=np.float32) if radar is None else radar

    def xyz(points: np.ndarray) -> list[list[float]]:
        return np.round(_display_points(points, max_plot_points), 2).tolist()

    payload = {
        "sample": sample_name, "fault": fault, "epoch": epoch,
        "bounds": bounds,
        "boxes": [{"name": box["name"], "color": box["color"],
                   "corners": np.round(box["corners"], 3).tolist()}
                  for box in (boxes or [])],
        "radar_stats": radar_stats,
        "radar": {"color": COLORS["radar"], "count": len(radar),
                  "points": xyz(radar)},
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
button{font:inherit;padding:8px 13px;color:#eaf0f6;background:#263340;border:1px solid #5a6d80;border-radius:6px;cursor:pointer}
button:focus-visible,input:focus-visible{outline:2px solid #ffbf47;outline-offset:2px}
.controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.controls label{cursor:pointer;white-space:nowrap;margin-right:8px}
.zoom{font-size:20px;font-weight:700;line-height:1;min-width:42px;min-height:42px}
.view-tabs{display:none}
.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
section{min-width:0;background:#19222c;border:1px solid #364756;border-radius:7px;padding:10px}
h2{font-size:16px;font-weight:600;margin:0 0 8px}canvas{display:block;width:100%;height:440px;touch-action:none;cursor:grab;background:#111a22}
canvas:active{cursor:grabbing}.note{margin-top:12px}
@media(max-width:900px){
  body{padding:8px}header{gap:8px}h1{font-size:16px}
  .view-tabs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px;margin-bottom:8px}
  .view-tabs button{padding:10px 3px;font-size:13px;min-height:44px}
  .view-tabs button[aria-selected="true"]{background:#31536b;border-color:#71b5e1}
  .panels{display:block}section{display:none;padding:6px}section.active{display:block}
  canvas{height:min(70dvh,700px)}.note{font-size:13px}
}
</style>
<header><div><h1 id="title"></h1><p>Drag to rotate · wheel or pinch to zoom · Shift-drag or two fingers to pan. All views share one camera and scale.</p><p id="radar-coverage"></p></div><div class="controls"><label><input id="show-radar" type="checkbox" checked> Radar (<span id="radar-count"></span>)</label><label><input id="show-boxes" type="checkbox" checked> Ground-truth boxes (<span id="box-count"></span>)</label><button id="zoom-out" class="zoom" type="button" aria-label="Zoom out">−</button><button id="zoom-in" class="zoom" type="button" aria-label="Zoom in">+</button><button id="reset" type="button">Reset view</button></div></header>
<div id="view-tabs" class="view-tabs" role="tablist" aria-label="LiDAR condition"></div>
<div id="panels" class="panels"></div><p class="note">Purple = retained LiDAR; blue = generated additions; amber = aligned radar. Box outlines: cyan Car, pink Pedestrian, lime Cyclist. Counts are full clouds; display points are capped for speed.</p>
<script id="cloud-data" type="application/json">__DATA__</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById('cloud-data').textContent);
  document.getElementById('title').textContent = `${data.sample} · ${data.fault} · checkpoint epoch ${data.epoch}`;
  document.getElementById('radar-count').textContent = data.radar.count.toLocaleString();
  document.getElementById('box-count').textContent = data.boxes.length.toLocaleString();
  if (data.radar_stats) {
    const stats = data.radar_stats;
    document.getElementById('radar-coverage').textContent = stats.radar_returns
      ? `Radar returns in any GT 3D box: ${stats.inside_boxes.toLocaleString()} (${stats.inside_percent.toFixed(2)}%); outside: ${stats.outside_boxes.toLocaleString()} (${stats.outside_percent.toFixed(2)}%).`
      : 'No radar returns after viewer filtering.';
  }
  const radarToggle = document.getElementById('show-radar');
  const boxToggle = document.getElementById('show-boxes');
  const root = document.getElementById('panels');
  const tabs = document.getElementById('view-tabs');
  const sections = [];
  const canvases = data.panels.map((panel,index) => {
    const section = document.createElement('section');
    section.classList.toggle('active',index===2);
    sections.push(section);
    const heading = document.createElement('h2');
    heading.textContent = `${panel.name} (${panel.count.toLocaleString()} points)`;
    const canvas = document.createElement('canvas');
    canvas.setAttribute('aria-label', `${panel.name} rotatable point cloud`);
    section.append(heading, canvas); root.append(section);
    const tab = document.createElement('button');
    tab.type='button'; tab.setAttribute('role','tab');
    tab.textContent=panel.name.replace(' LiDAR','');
    tab.setAttribute('aria-selected',String(index===2));
    tab.addEventListener('click',()=>{
      sections.forEach((item,i)=>item.classList.toggle('active',i===index));
      [...tabs.children].forEach((item,i)=>item.setAttribute('aria-selected',String(i===index)));
      scheduleDraw();
    });
    tabs.append(tab);
    return canvas;
  });
  const bounds = data.bounds;
  const center = bounds.map(pair => (pair[0] + pair[1]) / 2);
  const spans = bounds.map(pair => pair[1] - pair[0]);
  const radius = Math.max(Math.hypot(...spans) / 2, 1);
  const initial = {yaw:-0.85,pitch:0.35,zoom:1,panX:0,panY:0};
  const view = {...initial};
  function zoomAt(multiplier,x,y,width,height){
    const next=Math.max(0.1,Math.min(80,view.zoom*multiplier));
    const ratio=next/view.zoom;
    view.panX=x-width/2-(x-width/2-view.panX)*ratio;
    view.panY=y-height/2-(y-height/2-view.panY)*ratio;
    view.zoom=next;
  }
  function project(point, width, height, scale) {
    const x=point[0]-center[0], y=point[1]-center[1], z=point[2]-center[2];
    const cy=Math.cos(view.yaw), sy=Math.sin(view.yaw);
    const cp=Math.cos(view.pitch), sp=Math.sin(view.pitch);
    const u=cy*x-sy*y, v=sy*x+cy*y;
    return [width/2+view.panX+u*scale,height/2+view.panY-(cp*z-sp*v)*scale];
  }
  function draw(canvas, panel) {
    const dpi=window.devicePixelRatio||1, width=canvas.clientWidth, height=canvas.clientHeight;
    if(width===0||height===0)return;
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
        if(px>=0&&px<width&&py>=0&&py<height) ctx.fillRect(px-1,py-1,2.2,2.2);
      }
    }
    if(radarToggle.checked){
      ctx.fillStyle=data.radar.color;
      for(const point of data.radar.points){
        const [px,py]=project(point,width,height,scale);
        if(px>=0&&px<width&&py>=0&&py<height) ctx.fillRect(px-1.6,py-1.6,3.4,3.4);
      }
    }
    if(boxToggle.checked){
      ctx.lineWidth=2;
      ctx.font='12px system-ui,sans-serif';
      for(const box of data.boxes){
        const corners=box.corners.map(point=>project(point,width,height,scale));
        ctx.strokeStyle=box.color;
        for(const [start,end] of __BOX_EDGES__){
          ctx.beginPath();ctx.moveTo(...corners[start]);ctx.lineTo(...corners[end]);ctx.stroke();
        }
        const label=corners.reduce((best,point)=>point[1]<best[1]?point:best);
        ctx.fillStyle=box.color;ctx.fillText(box.name,label[0]+4,label[1]-4);
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
      active.set(event.pointerId,{x:event.clientX,y:event.clientY,canvas});
    });
    canvas.addEventListener('pointermove',event=>{
      const prior=active.get(event.pointerId);if(!prior)return;
      const peers=[...active.values()].filter(pointer=>pointer.canvas===canvas);
      const previous=peers.map(pointer=>({x:pointer.x,y:pointer.y}));
      const dx=event.clientX-prior.x,dy=event.clientY-prior.y;
      prior.x=event.clientX;prior.y=event.clientY;
      if(peers.length===2){
        const oldMid={x:(previous[0].x+previous[1].x)/2,y:(previous[0].y+previous[1].y)/2};
        const newMid={x:(peers[0].x+peers[1].x)/2,y:(peers[0].y+peers[1].y)/2};
        const oldDistance=Math.hypot(previous[0].x-previous[1].x,previous[0].y-previous[1].y);
        const newDistance=Math.hypot(peers[0].x-peers[1].x,peers[0].y-peers[1].y);
        const bounds=canvas.getBoundingClientRect();
        if(oldDistance>0)zoomAt(newDistance/oldDistance,oldMid.x-bounds.left,oldMid.y-bounds.top,canvas.clientWidth,canvas.clientHeight);
        view.panX+=newMid.x-oldMid.x;view.panY+=newMid.y-oldMid.y;
      }else if(event.shiftKey){view.panX+=dx;view.panY+=dy;}
      else{view.yaw+=dx*0.008;view.pitch=Math.max(-1.55,Math.min(1.55,view.pitch+dy*0.008));}
      scheduleDraw();
    });
    canvas.addEventListener('pointerup',event=>active.delete(event.pointerId));
    canvas.addEventListener('pointercancel',event=>active.delete(event.pointerId));
    canvas.addEventListener('wheel',event=>{
      event.preventDefault();zoomAt(Math.exp(-event.deltaY*0.001),event.offsetX,event.offsetY,canvas.clientWidth,canvas.clientHeight);
      scheduleDraw();
    },{passive:false});
  }
  for(const [id,multiplier] of [['zoom-in',1.6],['zoom-out',1/1.6]]){
    document.getElementById(id).addEventListener('click',()=>{
      const canvas=canvases.find(item=>item.clientWidth>0);
      if(!canvas)return;
      zoomAt(multiplier,canvas.clientWidth/2,canvas.clientHeight/2,canvas.clientWidth,canvas.clientHeight);
      scheduleDraw();
    });
  }
  document.getElementById('reset').addEventListener('click',()=>{Object.assign(view,initial);scheduleDraw();});
  radarToggle.addEventListener('change',scheduleDraw);
  boxToggle.addEventListener('change',scheduleDraw);
  window.addEventListener('resize',scheduleDraw); drawAll();
})();
</script></html>
"""
    path.write_text(page.replace("__DATA__", data).replace("__BOX_EDGES__", json.dumps(BOX_EDGES)),
                    encoding="utf-8")


def _render_comparison(output_root: Path, *, faulty: np.ndarray, clean: np.ndarray,
                       original: np.ndarray, generated: np.ndarray,
                       sample_name: str, fault: str, epoch: int,
                       max_plot_points: int, show: bool,
                       radar: np.ndarray | None = None,
                       boxes: list[dict] | None = None) -> None:
    import matplotlib.pyplot as plt

    if show and plt.get_backend().lower().endswith("agg"):
        print("No interactive Matplotlib display; open the saved _interactive.html "
              "file in a browser to rotate the clouds.", flush=True)
        show = False

    faulty_plot = _display_points(faulty, max_plot_points)
    clean_plot = _display_points(clean, max_plot_points)
    original_plot = _display_points(original, max_plot_points)
    generated_plot = _display_points(generated, max_plot_points)
    radar_plot = (_display_points(radar, max_plot_points)
                  if radar is not None else np.empty((0, 3), dtype=np.float32))
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
        if len(radar_plot):
            axis.scatter(radar_plot[:, 0], radar_plot[:, 1], radar_plot[:, 2],
                         s=2.5, c=COLORS["radar"], depthshade=False, rasterized=True)
        for box in boxes or []:
            corners = box["corners"]
            for start, end in BOX_EDGES:
                axis.plot(corners[[start, end], 0], corners[[start, end], 1],
                          corners[[start, end], 2], c=box["color"], linewidth=0.8)
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
            if len(radar_plot):
                axis.scatter(radar_plot[:, horizontal], radar_plot[:, vertical],
                             s=2.5, c=COLORS["radar"], alpha=0.9, rasterized=True)
            for box in boxes or []:
                corners = box["corners"]
                for start, end in BOX_EDGES:
                    axis.plot(corners[[start, end], horizontal],
                              corners[[start, end], vertical],
                              c=box["color"], linewidth=0.7)
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
    parser.add_argument("--enforce-first-return", action="store_true",
                        help="Reject generated points in cells containing a retained LiDAR return")
    parser.add_argument("--radar-anchor-radius-m", type=float,
                        help="Retain generated points only within this 3D distance of aligned radar returns")
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
    if args.enforce_first_return:
        merge_config = replace(merge_config, enforce_single_return_per_cell=True)
    if args.radar_anchor_radius_m is not None:
        merge_config = replace(merge_config, radar_anchor_radius_m=args.radar_anchor_radius_m)
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
    coverage_rows = []
    for index in args.sample_indices:
        if index >= len(paths):
            raise IndexError(f"sample index {index} is outside 0..{len(paths) - 1}")
        path = paths[index]
        sample = load_range_sample(
            path, args.radar_root, geometry, fault_map_root=args.fault_map_root,
            forward_only=merge_config.forward_only,
            radar_floor_band_m=radar_floor_band_m,
            filter_radar_by_lidar_min=bool(checkpoint.get("filter_radar_by_lidar_min", False)),
            use_ray_encoding=model_config.use_ray_encoding)
        _, _, _, radar_valid = angular_indices(
            sample.radar_points, geometry, require_beam_match=False)
        radar_points = sample.radar_points[radar_valid]
        boxes = _load_annotated_boxes(sample.metadata)
        is_vod = str(sample.metadata.get("dataset", "")).strip().lower() in {
            "view-of-delft", "view of delft", "vod"
        }
        radar_stats = _radar_box_stats(radar_points, boxes) if is_vod else None
        with torch.inference_mode():
            prediction = model(torch.from_numpy(sample.features)[None].to(args.device))
        merged = merge_reconstruction(
            sample.faulty_points, sample.faulty_projection, geometry,
            prediction["add_probability"][0].cpu().numpy(),
            prediction["add_range_m"][0].cpu().numpy(),
            prediction["delete_probability"][0].cpu().numpy(),
            config=merge_config, radar_support=sample.radar_features[0],
            radar_points=sample.radar_points,
            add_intensity=(prediction["add_intensity"][0].cpu().numpy()
                           if "add_intensity" in prediction else None),
        )
        destination = args.output_root / path.stem
        destination.mkdir(parents=True, exist_ok=True)
        for label, points in (
            ("faulty", sample.faulty_points), ("clean", sample.clean_points),
            ("generated", merged.generated_points), ("reconstructed", merged.points),
            ("radar", radar_points[:, :3]),
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
            "radar_points": len(radar_points),
            "ground_truth_boxes": len(boxes),
            "radar_box_coverage": radar_stats,
            "radar_coverage_basis": "aligned radar after reconstruction input filters and range-view selection",
            "deleted_original_points": len(merged.deleted_original_indices),
            "first_return_filter": merge_config.enforce_single_return_per_cell,
            "blocked_generated_occupied_cells": merged.blocked_generated_occupied_cells,
            "radar_anchor_radius_m": merge_config.radar_anchor_radius_m,
            "blocked_generated_radar_distance": merged.blocked_generated_radar_distance,
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
            radar=radar_points, boxes=boxes, radar_stats=radar_stats,
        )
        _render_comparison(
            destination, faulty=sample.faulty_points, clean=sample.clean_points,
            original=sample.faulty_points[merged.retained_original_indices],
            generated=merged.generated_points, sample_name=path.stem,
            fault=str(metadata["fault"]), epoch=int(checkpoint["epoch"]),
            max_plot_points=args.max_plot_points, show=not args.no_show,
            radar=radar_points, boxes=boxes,
        )
        print(f"{path.name}: {metadata['fault']} | faulty {metadata['faulty_points']:,} | "
              f"clean {metadata['clean_points']:,} | reconstructed "
              f"{metadata['reconstructed_points']:,} | {destination}", flush=True)
        if radar_stats is not None:
            coverage_rows.append({"sample": path.stem, "fault": str(metadata["fault"]),
                                  **radar_stats})
            print(f"  radar inside GT boxes: {radar_stats['inside_boxes']:,}/"
                  f"{radar_stats['radar_returns']:,} ({radar_stats['inside_percent']:.2f}%); "
                  f"outside: {radar_stats['outside_boxes']:,} "
                  f"({radar_stats['outside_percent']:.2f}%)", flush=True)
    if coverage_rows:
        total = sum(row["radar_returns"] for row in coverage_rows)
        inside = sum(row["inside_boxes"] for row in coverage_rows)
        pooled = {
            "sample": "ALL_SELECTED", "fault": "all",
            "radar_returns": total, "inside_boxes": inside,
            "outside_boxes": total - inside,
            "inside_percent": 100 * inside / total if total else 0.0,
            "outside_percent": 100 * (total - inside) / total if total else 0.0,
            "box_count": sum(row["box_count"] for row in coverage_rows),
        }
        output_csv = args.output_root / "radar_box_coverage.csv"
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0]))
            writer.writeheader()
            writer.writerows(coverage_rows)
            writer.writerow(pooled)
        print(f"Radar box coverage: {output_csv}", flush=True)


if __name__ == "__main__":
    main()
