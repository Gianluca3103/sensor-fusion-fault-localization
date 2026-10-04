"""Build an offline, mouse-rotatable 3D comparison of VoD radar stacks.

The same current-time ground-truth boxes and optional clean LiDAR reference are
shown in all four panels. The HTML embeds its data and needs no CDN or server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from Fault_Localization_Model.vod_dataset.vod_io import (
    _named_transform, align_radar_to_lidar, load_vod_lidar, load_vod_radar,
    load_vod_radar_to_lidar, load_vod_split_ids, resolve_vod_public_root,
    vod_partition_for_split,
)
from models.two_stage_reconstruction_head.cross_modal_data import (
    observed_lidar_height_mask,
)
from scripts.compare_vod_doppler_radar import VARIANTS, _target_box_masks


BOX_EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))
BOX_COLORS = {"Car": "#c58aff", "Pedestrian": "#ff69b4", "Cyclist": "#a8ed66"}
DISPLAY_NAMES = {
    "ego": "Ego alignment",
    "old_gate": "Old velocity-age deletion",
    "radial": "Doppler radial shift",
    "radial_window": "Radial shift + age window",
    "radial_height": "Radial shift + faulty-LiDAR height",
}
DEFAULT_PREVIOUS = ("train:00544", "train:00564", "val:05072")
ROI_LOWER = np.asarray((0.0, -25.0, -5.0))
ROI_UPPER = np.asarray((50.0, 25.0, 8.0))


def _rounded_rows(points: np.ndarray, ages: np.ndarray | None = None) -> list[list[float]]:
    xyz = np.round(points[:, :3], 2)
    if ages is None:
        return xyz.tolist()
    return np.column_stack((xyz, ages.astype(np.int16))).tolist()


def _inside_roi(points: np.ndarray) -> np.ndarray:
    return np.all((points[:, :3] >= ROI_LOWER) & (points[:, :3] <= ROI_UPPER), axis=1)


def _calibrated_boxes(public: Path, partition: str, frame_id: str) -> list[dict]:
    camera_from_radar = _named_transform(
        public / "radar" / partition / "calib" / f"{frame_id}.txt"
    )
    radar_from_camera = np.linalg.inv(camera_from_radar)
    label_path = public / "lidar" / partition / "label_2" / f"{frame_id}.txt"
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        name = "Cyclist" if fields[0] in {"Cyclist", "bicycle", "rider"} else fields[0]
        if name not in BOX_COLORS:
            continue
        height, width, length, x, y, z, yaw = map(float, fields[8:15])
        if min(height, width, length) <= 0:
            raise ValueError(f"Invalid 3D box in {label_path}: {line!r}")
        local = np.asarray([(sx * width / 2, sy * height, sz * length / 2)
                            for sx in (-1, 1) for sy in (-1, 0) for sz in (-1, 1)])
        camera = np.empty_like(local)
        camera[:, 0] = x + np.cos(yaw) * local[:, 0] + np.sin(yaw) * local[:, 2]
        camera[:, 1] = y + local[:, 1]
        camera[:, 2] = z - np.sin(yaw) * local[:, 0] + np.cos(yaw) * local[:, 2]
        radar = camera @ radar_from_camera[:3, :3].T + radar_from_camera[:3, 3]
        boxes.append({
            "name": name, "color": BOX_COLORS[name],
            "corners": np.round(radar, 2).tolist(),
        })
    return boxes


def _clean_lidar_in_radar(public: Path, partition: str, frame_id: str,
                          maximum: int) -> dict:
    lidar = load_vod_lidar(
        public / "lidar" / partition / "velodyne" / f"{frame_id}.bin"
    )
    lidar_from_radar = load_vod_radar_to_lidar(
        public / "lidar" / partition / "calib" / f"{frame_id}.txt",
        public / "radar" / partition / "calib" / f"{frame_id}.txt",
    )
    radar_from_lidar = np.linalg.inv(lidar_from_radar)
    xyz = lidar[:, :3] @ radar_from_lidar[:3, :3].T + radar_from_lidar[:3, 3]
    xyz = xyz[_inside_roi(xyz)]
    count = len(xyz)
    if count > maximum:
        xyz = xyz[np.linspace(0, count - 1, maximum, dtype=np.int64)]
    return {"count_in_view": count, "displayed": len(xyz),
            "points": _rounded_rows(xyz)}


def _radar_panel(public: Path, partition: str, frame_id: str,
                 condition: str, variant: str,
                 points_override: np.ndarray | None = None) -> dict:
    path = public / variant / partition / "velodyne" / f"{frame_id}.bin"
    points = load_vod_radar(path) if points_override is None else points_override
    age = -np.rint(points[:, 6]).astype(np.int16)
    in_view = _inside_roi(points)
    current = (age == 0) & in_view
    fast_old = (age > 0) & (np.abs(points[:, 5]) >= 1.0) & in_view
    slow_old = (age > 0) & (np.abs(points[:, 5]) < 1.0) & in_view
    counts = _target_box_masks(public, partition, frame_id, points)
    return {
        "key": condition, "name": DISPLAY_NAMES[condition],
        "total": len(points), "in_view": int(in_view.sum()),
        "box_counts": {name: int(mask.sum()) for name, mask in counts.items()},
        "layers": {
            "current": _rounded_rows(points[current], age[current]),
            "old_slow": _rounded_rows(points[slow_old], age[slow_old]),
            "old_fast": _rounded_rows(points[fast_old], age[fast_old]),
        },
    }


def build_frame(public: Path, split: str, frame_id: str, *, source: str,
                max_lidar_points: int,
                fault_samples_root: Path | None = None) -> dict:
    if frame_id not in set(load_vod_split_ids(public, split)):
        raise ValueError(f"{frame_id} is not in the official {split} split")
    partition = vod_partition_for_split(public, split, [frame_id])
    panels = [_radar_panel(public, partition, frame_id, condition, variant)
              for condition, variant in VARIANTS.items()
              if condition != "radial_window" or fault_samples_root is None]
    height_gate = None
    if fault_samples_root is not None:
        matches = list((fault_samples_root / split).glob(f"{frame_id}_*.npz"))
        if len(matches) != 1:
            raise ValueError(f"Expected one faulty LiDAR sample for {split}:{frame_id}: {matches}")
        with np.load(matches[0], allow_pickle=False) as archive:
            observed = np.asarray(archive["faulty_lidar_points"][:, :4],
                                  dtype=np.float32)
        raw_radial = load_vod_radar(
            public / VARIANTS["radial"] / partition / "velodyne" / f"{frame_id}.bin"
        )
        lidar_from_radar = load_vod_radar_to_lidar(
            public / "lidar" / partition / "calib" / f"{frame_id}.txt",
            public / "radar" / partition / "calib" / f"{frame_id}.txt",
        )
        aligned = align_radar_to_lidar(raw_radial, lidar_from_radar)
        keep = observed_lidar_height_mask(aligned, observed)
        panels.append(_radar_panel(
            public, partition, frame_id, "radial_height", VARIANTS["radial"],
            points_override=raw_radial[keep],
        ))
        height_gate = {
            "observed_lidar_points": len(observed),
            "lowest_z": round(float(observed[:, 2].min()), 3) if len(observed) else None,
            "highest_z": round(float(observed[:, 2].max()), 3) if len(observed) else None,
            "removed_radar": int((~keep).sum()),
            "sample": matches[0].name,
        }
    return {
        "id": frame_id, "split": split, "source": source,
        "boxes": _calibrated_boxes(public, partition, frame_id),
        "lidar": _clean_lidar_in_radar(public, partition, frame_id,
                                       max_lidar_points),
        "panels": panels, "height_gate": height_gate,
    }


def _html(payload: dict) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    page = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VoD radar motion alignment | 10 interactive 3D frames</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;padding:16px;background:#0b1118;color:#eaf2f8;font:14px system-ui,sans-serif}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;flex-wrap:wrap;margin-bottom:10px}
h1{font-size:21px;margin:0 0 4px}p{margin:3px 0;color:#b7c8d5}.controls{display:flex;align-items:center;gap:7px;flex-wrap:wrap}
button,select{font:inherit;color:#edf7ff;background:#233647;border:1px solid #58738a;border-radius:6px;padding:8px 11px;cursor:pointer}
button:hover{background:#31516b}button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid #ffc268;outline-offset:2px}
label{white-space:nowrap;cursor:pointer;margin-right:7px}.toolbar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;padding:9px;background:#15232f;border:1px solid #355066;border-radius:7px;margin-bottom:9px}
.tabs{display:none}.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px}section{min-width:0;background:#15222e;border:1px solid #355066;border-radius:7px;padding:7px}
section h2{font-size:15px;margin:0 0 4px}section p{font-size:12px;margin:0 0 7px}canvas{display:block;width:100%;height:min(68vh,680px);min-height:420px;background:#08141d;touch-action:none;cursor:grab;border-radius:4px}
canvas:active{cursor:grabbing}.legend{margin-top:9px}.legend b{font-weight:600}.note{font-size:12px;margin-top:8px}
@media(max-width:1000px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:650px){body{padding:8px}.tabs{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:4px;margin:8px 0}.tabs button{font-size:11px;padding:9px 3px}.tabs button[aria-selected=true]{background:#326381}.panels{display:block}section{display:none}section.active{display:block}canvas{height:min(68dvh,640px)}}
</style>
<header><div><h1>VoD radar motion alignment · 10 frames</h1><p>Drag to rotate · wheel or pinch to zoom · Shift-drag or two fingers to pan. Every panel shares one view.</p><p id="frame-status"></p></div>
<div class="controls"><button id="previous" type="button">← Previous</button><select id="frame-select" aria-label="Select frame"></select><button id="next" type="button">Next →</button></div></header>
<div class="toolbar"><label><input id="show-lidar" type="checkbox"> Clean LiDAR</label><label><input id="show-current" type="checkbox" checked> Current radar</label>
<label><input id="show-old-slow" type="checkbox" checked> Older slow radar</label><label><input id="show-old-fast" type="checkbox" checked> Older fast radar</label>
<label><input id="show-boxes" type="checkbox" checked> 3D boxes</label><label>History <input id="history" type="range" min="0" max="19" value="19"> <span id="history-value">19</span> scans</label>
<button id="top-view" type="button">Top</button><button id="side-view" type="button">Side</button><button id="zoom-out" type="button">−</button><button id="zoom-in" type="button">+</button><button id="reset" type="button">Reset view</button></div>
<div id="tabs" class="tabs" role="tablist" aria-label="Radar stack variants"></div><div id="panels" class="panels"></div>
<p class="legend"><b>Colors:</b> cyan current radar · gray older slow radar · orange older fast radar (|compensated Doppler| ≥ 1 m/s) · green clean LiDAR · purple Car boxes · pink Pedestrian boxes · lime Cyclist boxes.</p>
<p class="note">Boxes and clean LiDAR are for inspection only. The height gate uses each frame's observed faulty LiDAR, not the clean reference. Box counts include the full radar stack; the viewer displays a 0–50 m forward ROI, and clean LiDAR is capped for speed. Historical points are repeated across overlapping target frames.</p>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(() => {
const data=JSON.parse(document.getElementById('payload').textContent);
const $=id=>document.getElementById(id);const root=$('panels'),tabs=$('tabs'),selector=$('frame-select');
const toggles={lidar:$('show-lidar'),current:$('show-current'),old_slow:$('show-old-slow'),old_fast:$('show-old-fast'),boxes:$('show-boxes')};
const colors={current:'#49c6ff',old_slow:'#8295a5',old_fast:'#ffa348',lidar:'#48b975'};
const sections=[],canvases=[],headings=[],subtitles=[];
let frameIndex=0,activePanel=0;
for(let i=0;i<4;i++){
  const section=document.createElement('section');section.classList.toggle('active',i===0);sections.push(section);
  const h=document.createElement('h2'),p=document.createElement('p'),canvas=document.createElement('canvas');
  canvas.setAttribute('aria-label',`Rotatable 3D radar comparison panel ${i+1}`);
  section.append(h,p,canvas);root.append(section);headings.push(h);subtitles.push(p);canvases.push(canvas);
  const tab=document.createElement('button');tab.type='button';tab.textContent=data.frames[0].panels[i].name.replace('Doppler ','').replace('Radial shift + faulty-LiDAR height','Height gate');
  tab.setAttribute('role','tab');tab.setAttribute('aria-selected',String(i===0));
  tab.addEventListener('click',()=>{activePanel=i;sections.forEach((v,j)=>v.classList.toggle('active',j===i));[...tabs.children].forEach((v,j)=>v.setAttribute('aria-selected',String(j===i)));scheduleDraw();});tabs.append(tab);
}
data.frames.forEach((frame,i)=>{const opt=document.createElement('option');opt.value=String(i);opt.textContent=`${String(i+1).padStart(2,'0')}/10 · ${frame.id} · ${frame.source}`;selector.append(opt);});
const initial={yaw:-0.75,pitch:0.43,zoom:1,panX:0,panY:0},view={...initial};
const center=[25,0,1.5],radius=37;
function project(point,w,h,scale){const x=point[0]-center[0],y=point[1]-center[1],z=point[2]-center[2];const cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch);const u=cy*x-sy*y,v=sy*x+cy*y;return [w/2+view.panX+u*scale,h/2+view.panY-(cp*z-sp*v)*scale];}
function zoomAt(multiplier,x,y,w,h){const next=Math.max(0.25,Math.min(45,view.zoom*multiplier));const ratio=next/view.zoom;view.panX=x-w/2-(x-w/2-view.panX)*ratio;view.panY=y-h/2-(y-h/2-view.panY)*ratio;view.zoom=next;}
function dots(ctx,rows,color,size,limit,w,h,scale){ctx.fillStyle=color;for(const row of rows){if(row.length>3&&row[3]>limit)continue;const p=project(row,w,h,scale);if(p[0]>=0&&p[0]<w&&p[1]>=0&&p[1]<h)ctx.fillRect(p[0]-size/2,p[1]-size/2,size,size);}}
function draw(canvas,panel,frame){const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;const pw=Math.round(w*dpr),ph=Math.round(h*dpr);if(canvas.width!==pw||canvas.height!==ph){canvas.width=pw;canvas.height=ph;}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);const scale=0.55*Math.min(w,h)*view.zoom/radius;
  ctx.lineWidth=1;const origin=project([0,0,0],w,h,scale);for(const [endpoint,color,label] of [[[10,0,0],'#e88787','X'],[[0,10,0],'#9ce3aa','Y'],[[0,0,5],'#9ebfea','Z']]){const end=project(endpoint,w,h,scale);ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(...end);ctx.stroke();ctx.fillStyle=color;ctx.fillText(label,end[0]+3,end[1]-3);}
  if(toggles.lidar.checked)dots(ctx,frame.lidar.points,colors.lidar,1.2,19,w,h,scale);
  const limit=Number($('history').value);for(const key of ['old_slow','old_fast','current'])if(toggles[key].checked)dots(ctx,panel.layers[key],colors[key],key==='old_fast'?2.8:2.2,limit,w,h,scale);
  if(toggles.boxes.checked){ctx.lineWidth=1.5;ctx.font='11px system-ui,sans-serif';for(const box of frame.boxes){const corner=box.corners.map(p=>project(p,w,h,scale));ctx.strokeStyle=box.color;for(const [a,b] of data.box_edges){ctx.beginPath();ctx.moveTo(...corner[a]);ctx.lineTo(...corner[b]);ctx.stroke();}const label=corner.reduce((best,p)=>p[1]<best[1]?p:best);ctx.fillStyle=box.color;ctx.fillText(box.name,label[0]+3,label[1]-3);}}
}
function showFrame(index){frameIndex=(index+data.frames.length)%data.frames.length;selector.value=String(frameIndex);const frame=data.frames[frameIndex];let status=`${frame.split} frame ${frame.id} · ${frame.source} · ${frame.boxes.length} annotated boxes · clean LiDAR ${frame.lidar.displayed.toLocaleString()}/${frame.lidar.count_in_view.toLocaleString()} displayed`;if(frame.height_gate){const g=frame.height_gate;status+=g.observed_lidar_points>=2?` · faulty LiDAR Z ${g.lowest_z} to ${g.highest_z} m · radar removed ${g.removed_radar.toLocaleString()}`:' · faulty LiDAR too sparse: height gate skipped';}$('frame-status').textContent=status;frame.panels.forEach((panel,i)=>{headings[i].textContent=`${panel.name} · ${panel.total.toLocaleString()} radar returns`;const b=panel.box_counts;subtitles[i].textContent=`Inside current boxes: Cyclist ${b.Cyclist}, Car ${b.Car}, Pedestrian ${b.Pedestrian}`;});scheduleDraw();}
function drawAll(){const frame=data.frames[frameIndex];canvases.forEach((canvas,i)=>draw(canvas,frame.panels[i],frame));}
let pending=false;function scheduleDraw(){if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;drawAll();});}
const active=new Map();for(const canvas of canvases){canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);active.set(e.pointerId,{x:e.clientX,y:e.clientY,canvas});});canvas.addEventListener('pointermove',e=>{const prior=active.get(e.pointerId);if(!prior)return;const peers=[...active.values()].filter(p=>p.canvas===canvas),before=peers.map(p=>({x:p.x,y:p.y})),dx=e.clientX-prior.x,dy=e.clientY-prior.y;prior.x=e.clientX;prior.y=e.clientY;if(peers.length===2){const oldMid={x:(before[0].x+before[1].x)/2,y:(before[0].y+before[1].y)/2},newMid={x:(peers[0].x+peers[1].x)/2,y:(peers[0].y+peers[1].y)/2},oldDist=Math.hypot(before[0].x-before[1].x,before[0].y-before[1].y),newDist=Math.hypot(peers[0].x-peers[1].x,peers[0].y-peers[1].y),box=canvas.getBoundingClientRect();if(oldDist>0)zoomAt(newDist/oldDist,oldMid.x-box.left,oldMid.y-box.top,canvas.clientWidth,canvas.clientHeight);view.panX+=newMid.x-oldMid.x;view.panY+=newMid.y-oldMid.y;}else if(e.shiftKey){view.panX+=dx;view.panY+=dy;}else{view.yaw+=dx*.008;view.pitch=Math.max(-1.55,Math.min(1.55,view.pitch+dy*.008));}scheduleDraw();});canvas.addEventListener('pointerup',e=>active.delete(e.pointerId));canvas.addEventListener('pointercancel',e=>active.delete(e.pointerId));canvas.addEventListener('wheel',e=>{e.preventDefault();zoomAt(Math.exp(-e.deltaY*.001),e.offsetX,e.offsetY,canvas.clientWidth,canvas.clientHeight);scheduleDraw();},{passive:false});}
selector.addEventListener('change',()=>showFrame(Number(selector.value)));$('previous').addEventListener('click',()=>showFrame(frameIndex-1));$('next').addEventListener('click',()=>showFrame(frameIndex+1));
for(const input of Object.values(toggles))input.addEventListener('change',scheduleDraw);$('history').addEventListener('input',()=>{$('history-value').textContent=$('history').value;scheduleDraw();});
for(const [id,factor] of [['zoom-in',1.6],['zoom-out',1/1.6]])$(id).addEventListener('click',()=>{const c=canvases[activePanel];zoomAt(factor,c.clientWidth/2,c.clientHeight/2,c.clientWidth,c.clientHeight);scheduleDraw();});
$('top-view').addEventListener('click',()=>{view.yaw=0;view.pitch=Math.PI/2;scheduleDraw();});$('side-view').addEventListener('click',()=>{view.yaw=-Math.PI/2;view.pitch=0;scheduleDraw();});$('reset').addEventListener('click',()=>{Object.assign(view,initial);scheduleDraw();});window.addEventListener('resize',scheduleDraw);document.addEventListener('keydown',e=>{if(e.target===selector)return;if(e.key==='ArrowRight')showFrame(frameIndex+1);if(e.key==='ArrowLeft')showFrame(frameIndex-1);});showFrame(0);
})();
</script></html>'''
    return page.replace("__PAYLOAD__", encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--previous", nargs="+", default=DEFAULT_PREVIOUS,
                        help="Previously inspected split:ID frames")
    parser.add_argument("--random-count", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-lidar-points", type=int, default=15000)
    parser.add_argument("--fault-samples-root", type=Path,
                        help="Add a fourth panel gated by each frame's observed faulty LiDAR Z range")
    args = parser.parse_args()
    if args.random_count < 0 or args.max_lidar_points < 1:
        parser.error("Random count must be non-negative and LiDAR cap positive")
    public = resolve_vod_public_root(args.vod_root)
    previous = []
    for item in args.previous:
        split, separator, frame_id = item.partition(":")
        if not separator or split not in {"train", "val"} or not frame_id.isdigit():
            parser.error(f"Invalid --previous frame {item!r}; expected train:00564")
        previous.append((split, frame_id.zfill(5)))
    if len(set(previous)) != len(previous):
        parser.error("Previously inspected frames must be distinct")
    validation = [frame_id for frame_id in load_vod_split_ids(public, "val")
                  if ("val", frame_id) not in previous]
    if args.random_count > len(validation):
        parser.error("Requested more random frames than the validation split has")
    random_ids = random.Random(args.seed).sample(validation, args.random_count)
    selected = [(split, frame_id, "previously inspected")
                for split, frame_id in previous]
    selected += [("val", frame_id, f"random seed {args.seed}")
                 for frame_id in random_ids]
    if len(selected) != 10:
        parser.error("This viewer must contain exactly 10 frames")
    frames = []
    for split, frame_id, source in selected:
        frames.append(build_frame(public, split, frame_id, source=source,
                                  max_lidar_points=args.max_lidar_points,
                                  fault_samples_root=args.fault_samples_root))
        print(f"Prepared {len(frames)}/10: {split} {frame_id} ({source})", flush=True)
    payload = {"frames": frames, "box_edges": BOX_EDGES,
               "random_seed": args.seed, "random_validation_ids": random_ids,
               "previous": args.previous, "roi": [ROI_LOWER.tolist(), ROI_UPPER.tolist()]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_html(payload), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
