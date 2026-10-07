"""Compare aligned VoD radar, clean LiDAR, and Stage-I confidence in 3D.

The confidence cloud consists of radar-derived S1 voxel centers. It is not a
reconstructed LiDAR cloud, and clean LiDAR is loaded only for visualization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.radar_lidar_stage1.data import VoDStage1Dataset
from models.radar_lidar_stage1.model import RadarOnlyEncoder
from models.radar_lidar_stage1.train import config_from_dict


def load_encoder(path: Path, device: str) -> tuple[RadarOnlyEncoder, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = RadarOnlyEncoder(config_from_dict(saved["config"]))
    if "radar_only" in saved:
        model.load_state_dict(saved["radar_only"])
    elif "model" in saved:
        prefix = "radar_only."
        weights = {key[len(prefix):]: value for key, value in saved["model"].items()
                   if key.startswith(prefix)}
        if not weights:
            raise ValueError("Checkpoint has no radar-only encoder weights")
        model.load_state_dict(weights)
    else:
        raise ValueError("Expected a Stage-I full checkpoint or radar_only.pth")
    model.confidence_trained = bool(saved.get("confidence_trained", False))
    model.confidence_calibrated = bool(saved.get("confidence_calibrated", False))
    return model.to(device).eval(), saved


def write_ply(path: Path, xyz: np.ndarray, confidence: np.ndarray | None = None) -> None:
    xyz = np.asarray(xyz, dtype="<f4")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("PLY cloud must have XYZ columns")
    fields = [(name, "<f4") for name in ("x", "y", "z")]
    if confidence is not None:
        confidence = np.asarray(confidence, dtype="<f4")
        if confidence.shape != (len(xyz),):
            raise ValueError("One confidence value is required per voxel center")
        fields.append(("confidence", "<f4"))
    records = np.empty(len(xyz), dtype=fields)
    for index, name in enumerate(("x", "y", "z")):
        records[name] = xyz[:, index]
    if confidence is not None:
        records["confidence"] = confidence
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              + ("property float confidence\n" if confidence is not None else "")
              + "end_header\n")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(records.tobytes())


def display_subset(xyz: np.ndarray, maximum: int, limits: tuple[float, ...],
                   values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, int]:
    xmin, xmax, ymin, ymax, zmin, zmax = limits
    mask = ((xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax)
            & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)
            & (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax))
    indices = np.flatnonzero(mask)
    in_view = len(indices)
    if len(indices) > maximum:
        indices = indices[np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)]
    return xyz[indices], None if values is None else values[indices], in_view


def save_viewer(path: Path, *, frame_id: str, epoch: int | None, radar: np.ndarray,
                lidar: np.ndarray, sites: np.ndarray, confidence: np.ndarray,
                limits: tuple[float, ...], max_points: int, trained: bool,
                calibrated: bool) -> dict:
    clouds = []
    for name, xyz, values in (("Radar", radar, None), ("Clean LiDAR", lidar, None),
                              ("Stage-I confidence", sites, confidence)):
        shown, selected_values, in_view = display_subset(xyz, max_points, limits, values)
        clouds.append({"name": name, "total": len(xyz), "in_view": in_view,
                       "shown": len(shown), "xyz": np.round(shown, 3).tolist(),
                       "confidence": None if selected_values is None
                       else np.round(selected_values, 4).tolist()})
    payload = {"frame_id": frame_id, "epoch": epoch, "clouds": clouds,
               "limits": limits, "trained": trained, "calibrated": calibrated}
    packed = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
    html = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage-I radar, LiDAR and confidence</title>
<style>
body{margin:0;padding:14px;background:#10161d;color:#edf3f8;font:14px system-ui,sans-serif}
h1{font-size:20px;margin:0 0 4px}p{margin:4px 0 12px;color:#b9c6d3}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:10px 0}
input[type=range]{width:190px}button{background:#263747;color:#edf3f8;border:1px solid #60768a;border-radius:5px;padding:6px 10px;cursor:pointer}
.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
section{min-width:0;background:#19232e;border:1px solid #3c5264;border-radius:6px;padding:8px}
h2{font-size:15px;margin:0 0 5px}canvas{width:100%;height:70vh;max-height:780px;min-height:420px;background:#0d151e;touch-action:none;cursor:grab}
.legend{display:flex;align-items:center;gap:5px;color:#b9c6d3;margin-top:7px}.ramp{height:12px;width:130px;background:linear-gradient(to right,#2862be,#51cad1,#ffe173,#f56868)}
@media(max-width:950px){.panels{display:block}section{margin-bottom:12px}canvas{height:60vh;min-height:330px}}
</style>
<h1 id="title"></h1>
<p>Drag to rotate · wheel or pinch to zoom · Shift-drag or two fingers to pan. All three panels share one camera. Confidence markers are radar-derived S1 voxel centers, not reconstructed LiDAR points.</p>
<div class="controls"><label>Minimum confidence <input id="threshold" type="range" min="0" max="1" step="0.01" value="0"><output id="threshold-value">0.00</output></label>
<label><input id="overlay-radar" type="checkbox" checked> Overlay radar on confidence</label>
<label><input id="overlay-lidar" type="checkbox"> Overlay clean LiDAR on confidence</label>
<button id="zoom-in">Zoom in</button><button id="zoom-out">Zoom out</button><button id="reset">Reset view</button></div>
<div id="panels" class="panels"></div><div class="legend">Confidence 0 <span class="ramp"></span> 1 · colors show model confidence, not measured surface accuracy</div>
<p id="notice"></p><p>Display crop: forward 0–80 m, lateral ±40 m, height −5–7 m by default. PLY files retain the full clouds. Display point counts are capped for speed.</p>
<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
const d=JSON.parse(document.getElementById('data').textContent), panels=document.getElementById('panels');
document.getElementById('title').textContent=`Frame ${d.frame_id} · Stage-I ${d.epoch===null?'radar-only export':'epoch '+d.epoch}`;
document.getElementById('notice').textContent=d.trained
  ? (d.calibrated?'Checkpoint marks confidence calibrated.':'Confidence was trained but is not calibrated as a probability of a correct LiDAR surface.')
  : 'Checkpoint does not mark confidence as trained; colors are not interpretable.';
const canvases=d.clouds.map((cloud,i)=>{const section=document.createElement('section'),h=document.createElement('h2'),c=document.createElement('canvas');
  h.textContent=`${cloud.name}: ${cloud.total.toLocaleString()} full, ${cloud.in_view.toLocaleString()} in crop, ${cloud.shown.toLocaleString()} displayed`;
  section.append(h,c);panels.append(section);return c;});
const [xmin,xmax,ymin,ymax,zmin,zmax]=d.limits;
const center=[(xmin+xmax)/2,(ymin+ymax)/2,(zmin+zmax)/2],radius=Math.hypot(xmax-xmin,ymax-ymin,zmax-zmin)/2;
const initial={yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0},view={...initial};
const slider=document.getElementById('threshold'),overlayRadar=document.getElementById('overlay-radar'),overlayLidar=document.getElementById('overlay-lidar');
function color(v){const stops=[[40,98,190],[81,202,209],[255,225,115],[245,104,104]],a=Math.max(0,Math.min(.9999,v))*3,i=Math.floor(a),t=a-i;
  return `rgb(${stops[i].map((n,k)=>Math.round(n*(1-t)+stops[i+1][k]*t)).join(',')})`;}
function project(p,w,h,scale){const x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch);
  const u=cy*x-sy*y,v=sy*x+cy*y;return [w/2+view.panX+u*scale,h/2+view.panY-(cp*z-sp*v)*scale];}
function layer(ctx,cloud,w,h,scale,kind,alpha){ctx.globalAlpha=alpha;const limit=Number(slider.value);
  for(let i=0;i<cloud.xyz.length;i++){if(kind==='confidence'&&cloud.confidence[i]<limit)continue;
    const [x,y]=project(cloud.xyz[i],w,h,scale);if(x<0||x>=w||y<0||y>=h)continue;
    ctx.fillStyle=kind==='radar'?'#ffab40':kind==='lidar'?'#45b975':color(cloud.confidence[i]);
    const r=kind==='radar'?1.7:kind==='lidar'?1.1:2.4;ctx.fillRect(x-r/2,y-r/2,r,r);}
  ctx.globalAlpha=1;}
function draw(c,index){const ratio=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;if(!w||!h)return;
  const pw=Math.round(w*ratio),ph=Math.round(h*ratio);if(c.width!==pw||c.height!==ph){c.width=pw;c.height=ph;}
  const ctx=c.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);
  const scale=.47*Math.min(w,h)*view.zoom/radius;
  if(index===0)layer(ctx,d.clouds[0],w,h,scale,'radar',.8);
  else if(index===1)layer(ctx,d.clouds[1],w,h,scale,'lidar',.6);
  else{if(overlayLidar.checked)layer(ctx,d.clouds[1],w,h,scale,'lidar',.19);
    if(overlayRadar.checked)layer(ctx,d.clouds[0],w,h,scale,'radar',.25);
    layer(ctx,d.clouds[2],w,h,scale,'confidence',.95);}
  const o=project([0,0,0],w,h,scale);for(const [axis,color,p] of [['X','#eb7777',[10,0,0]],['Y','#78d18b',[0,10,0]],['Z','#9eabf0',[0,0,3]]]){
    const e=project(p,w,h,scale);ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(...o);ctx.lineTo(...e);ctx.stroke();ctx.fillStyle=color;ctx.fillText(axis,e[0],e[1]);}}
let pending=false;function redraw(){if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;canvases.forEach(draw);});}
function zoom(mult,x,y,w,h){const next=Math.max(.1,Math.min(80,view.zoom*mult)),k=next/view.zoom;
  view.panX=x-w/2-(x-w/2-view.panX)*k;view.panY=y-h/2-(y-h/2-view.panY)*k;view.zoom=next;}
const active=new Map();for(const canvas of canvases){canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);active.set(e.pointerId,{x:e.clientX,y:e.clientY,canvas});});
  canvas.addEventListener('pointermove',e=>{const p=active.get(e.pointerId);if(!p)return;const peers=[...active.values()].filter(q=>q.canvas===canvas),old=peers.map(q=>({x:q.x,y:q.y}));
    const dx=e.clientX-p.x,dy=e.clientY-p.y;p.x=e.clientX;p.y=e.clientY;
    if(peers.length===2){const oldDist=Math.hypot(old[0].x-old[1].x,old[0].y-old[1].y),newDist=Math.hypot(peers[0].x-peers[1].x,peers[0].y-peers[1].y);
      if(oldDist>0)zoom(newDist/oldDist,canvas.clientWidth/2,canvas.clientHeight/2,canvas.clientWidth,canvas.clientHeight);
      view.panX+=(peers[0].x+peers[1].x-old[0].x-old[1].x)/2;view.panY+=(peers[0].y+peers[1].y-old[0].y-old[1].y)/2;}
    else if(e.shiftKey){view.panX+=dx;view.panY+=dy;}else{view.yaw+=dx*.008;view.pitch=Math.max(-1.55,Math.min(1.55,view.pitch+dy*.008));}redraw();});
  for(const event of ['pointerup','pointercancel'])canvas.addEventListener(event,e=>active.delete(e.pointerId));
  canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.001),e.offsetX,e.offsetY,canvas.clientWidth,canvas.clientHeight);redraw();},{passive:false});}
slider.addEventListener('input',()=>{document.getElementById('threshold-value').textContent=Number(slider.value).toFixed(2);redraw();});
overlayRadar.addEventListener('change',redraw);overlayLidar.addEventListener('change',redraw);
document.getElementById('zoom-in').onclick=()=>{zoom(1.6,canvases[0].clientWidth/2,canvases[0].clientHeight/2,canvases[0].clientWidth,canvases[0].clientHeight);redraw();};
document.getElementById('zoom-out').onclick=()=>{zoom(1/1.6,canvases[0].clientWidth/2,canvases[0].clientHeight/2,canvases[0].clientWidth,canvases[0].clientHeight);redraw();};
document.getElementById('reset').onclick=()=>{Object.assign(view,initial);redraw();};
window.addEventListener('resize',redraw);redraw();
})();
</script></html>'''.replace("__DATA__", packed)
    path.write_text(html, encoding="utf-8")
    return {cloud["name"]: {key: cloud[key] for key in ("total", "in_view", "shown")}
            for cloud in clouds}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frame-id", nargs="+", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant", help="Override checkpoint radar variant")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-plot-points", type=int, default=30000)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=80.0)
    parser.add_argument("--y-min", type=float, default=-40.0)
    parser.add_argument("--y-max", type=float, default=40.0)
    parser.add_argument("--z-min", type=float, default=-5.0)
    parser.add_argument("--z-max", type=float, default=7.0)
    args = parser.parse_args()
    if args.max_plot_points < 1:
        parser.error("--max-plot-points must be positive")
    limits = (args.x_min, args.x_max, args.y_min, args.y_max, args.z_min, args.z_max)
    if any(a >= b for a, b in zip(limits[::2], limits[1::2])):
        parser.error("Each display crop minimum must be smaller than its maximum")
    if args.split == "test":
        parser.error("Official VoD test labels do not provide a clean LiDAR comparison here; use train or val")
    model, saved = load_encoder(args.checkpoint, args.device)
    variant = args.radar_variant or saved.get("data", {}).get("radar_variant", "radar_20frames_verified_doppler_radial")
    dataset = VoDStage1Dataset(args.vod_root, args.split, frame_ids=args.frame_id,
                               radar_variant=variant)
    found = {frame.frame_id for frame in dataset.frames}
    missing = set(args.frame_id) - found
    if missing:
        raise ValueError(f"Frame IDs not found in {args.split}: {sorted(missing)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for sample in dataset:
        radar = sample["radar"].numpy()
        lidar = sample["clean_lidar"].numpy()
        points = sample["radar"].unsqueeze(0).to(args.device)
        valid = torch.ones(points.shape[:2], dtype=torch.bool, device=args.device)
        output = model(points, valid)
        sites = output.confidence.centers_xyz(model.config.grid).cpu().numpy()
        confidence = output.confidence.features[:, 0].cpu().numpy()
        if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("Confidence values must be finite and between zero and one")
        prefix = args.output_root / f"{sample['frame_id']}_stage1"
        write_ply(prefix.with_name(prefix.name + "_radar.ply"), radar[:, :3])
        write_ply(prefix.with_name(prefix.name + "_clean_lidar.ply"), lidar[:, :3])
        write_ply(prefix.with_name(prefix.name + "_confidence.ply"), sites, confidence)
        counts = save_viewer(prefix.with_suffix(".html"), frame_id=sample["frame_id"],
                             epoch=saved.get("epoch"), radar=radar[:, :3],
                             lidar=lidar[:, :3], sites=sites, confidence=confidence,
                             limits=limits, max_points=args.max_plot_points,
                             trained=model.confidence_trained,
                             calibrated=model.confidence_calibrated)
        prefix.with_suffix(".json").write_text(json.dumps({
            "frame_id": sample["frame_id"], "split": args.split,
            "checkpoint": str(args.checkpoint.resolve()), "epoch": saved.get("epoch"),
            "radar_variant": variant, "confidence_trained": model.confidence_trained,
            "confidence_calibrated": model.confidence_calibrated,
            "confidence_mean": float(confidence.mean()) if len(confidence) else None,
            "confidence_min": float(confidence.min()) if len(confidence) else None,
            "confidence_max": float(confidence.max()) if len(confidence) else None,
            "counts": counts, "display_limits_xyz_m": limits,
            "note": "Confidence is on radar-derived S1 voxel centers; it is not a reconstructed LiDAR cloud."
        }, indent=2), encoding="utf-8")
        print(prefix.with_suffix(".html"), flush=True)


if __name__ == "__main__":
    main()
