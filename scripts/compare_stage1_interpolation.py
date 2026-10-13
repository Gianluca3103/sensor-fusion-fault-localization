"""Compare edge-aware Stage-II support with other interpolation methods.

This is an inference-only audit. It never trains a checkpoint or passes clean
LiDAR to any interpolation method; clean LiDAR is used only for evaluation.
The old ellipsoid is retained solely as a historical comparison.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np
from scipy.spatial import cKDTree
import torch

from models.radar_lidar_stage1.data import VoDStage1Dataset
from models.radar_lidar_stage1.interpolation import (
    edge_aware_seed_field, gaussian_seed_field, inverse_distance_seed_field,
    local_signed_distance_field, nearest_seed_field, trilinear_seed_field,
)
from models.radar_lidar_stage2.candidate_domain import make_candidates, proposal_radar_features
from scripts.visualize_stage1_confidence_cloud import load_encoder, write_ply


METHOD_LABELS = {"ellipsoid": "Old ellipsoid baseline (inactive)",
                 "nearest": "Nearest neighbor",
                 "trilinear": "Trilinear",
                 "inverse_distance": "Inverse-distance weighting",
                 "gaussian": "Gaussian weighting",
                 "edge_aware": "Edge-aware",
                 "local_sdf": "Local signed-distance field"}
POINTCLOUD_LABELS = {"radar": "Raw stacked radar", "clean_lidar": "Clean LiDAR reference",
                     "seeds": "Raw Stage I proposals", **METHOD_LABELS}


def _pointcloud_path(prefix: Path, name: str) -> Path:
    return prefix.with_name(f"{prefix.name}_{name}.ply")


def _bundle_pointclouds(prefix: Path) -> Path:
    """Bundle the full-resolution PLY exports without resampling or point caps."""
    bundle = prefix.with_name(f"{prefix.name}_pointclouds.zip")
    with ZipFile(bundle, "w", compression=ZIP_STORED) as archive:
        for name in POINTCLOUD_LABELS:
            cloud = _pointcloud_path(prefix, name)
            archive.write(cloud, arcname=cloud.name)
        sdf = prefix.with_name(f"{prefix.name}_local_sdf.npz")
        archive.write(sdf, arcname=sdf.name)
        manifest = {"frame": prefix.name.split("_", 1)[0],
                    "pointclouds": {name: {"label": label,
                                           "file": _pointcloud_path(prefix, name).name}
                                    for name, label in POINTCLOUD_LABELS.items()},
                    "local_sdf_data": sdf.name,
                    "note": "Raw radar and raw Stage I proposals are distinct. Clean LiDAR is a reference only."
                    }
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
    return bundle


def _crop(xyz: np.ndarray, limits: tuple[float, ...]) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = limits
    return ((xyz[:, 0] >= x0) & (xyz[:, 0] <= x1) &
            (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1) &
            (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1))


def _rates(sites: np.ndarray, clean: np.ndarray, radar: np.ndarray,
           radar_near_clean: np.ndarray, tolerance: float) -> dict[str, float | int]:
    if not len(sites):
        return {"sites": 0, "site_precision": 0., "clean_coverage": 0.,
                "radar_near_clean_coverage": 0., "near_radar_fraction_0.5m": 0.}
    clean_distance = cKDTree(clean).query(sites, workers=-1)[0] if len(clean) else np.full(len(sites), np.inf)
    surface_distance = cKDTree(sites).query(clean, workers=-1)[0] if len(clean) else np.empty(0)
    radar_distance = cKDTree(radar).query(sites, workers=-1)[0] if len(radar) else np.full(len(sites), np.inf)
    return {"sites": len(sites),
            "site_precision": float(np.mean(clean_distance <= tolerance)),
            "clean_coverage": float(np.mean(surface_distance <= tolerance)) if len(clean) else 0.,
            "radar_near_clean_coverage": float(np.mean(surface_distance[radar_near_clean] <= tolerance))
            if radar_near_clean.any() else 0.,
            "near_radar_fraction_0.5m": float(np.mean(radar_distance <= .5))}


def _write_summary(output_root: Path, reports: list[dict]) -> None:
    """Aggregate measured-frame metrics without using them to choose regions."""
    methods = list(reports[0]["methods"])
    summary = {"checkpoint": reports[0]["checkpoint"],
               "checkpoint_epoch": reports[0]["checkpoint_epoch"],
               "frames": [report["frame"] for report in reports],
               "tolerance_m": .2, "score_threshold": .25,
               "methods": {}}
    for method in methods:
        rows = [(report, report["methods"][method]["threshold_sweep_20cm"]["0.25"])
                for report in reports]
        matched = [(report, report["methods"][method]["matched_site_count_20cm"])
                   for report in reports]
        sites = sum(row["sites"] for _, row in rows)
        matched_sites = sum(row["sites"] for _, row in matched)
        clean = sum(report["clean_lidar_points_in_crop"] for report, _ in rows)
        near_clean = sum(report["radar_near_clean_points_in_crop"] for report, _ in rows)
        summary["methods"][method] = {
            "active_sites": sites,
            "site_precision_20cm": (sum(row["site_precision"] * row["sites"]
                                        for _, row in rows) / sites if sites else 0.),
            "clean_coverage_20cm": (sum(row["clean_coverage"] * report["clean_lidar_points_in_crop"]
                                         for report, row in rows) / clean if clean else 0.),
            "radar_near_clean_coverage_20cm": (
                sum(row["radar_near_clean_coverage"] * report["radar_near_clean_points_in_crop"]
                    for report, row in rows) / near_clean if near_clean else 0.),
            "matched_budget_complete_frames": sum(
                report["methods"][method]["matched_budget_complete"] for report in reports),
            "matched_sites": matched_sites,
            "matched_site_precision_20cm": (
                sum(row["site_precision"] * row["sites"] for _, row in matched) / matched_sites
                if matched_sites else 0.),
            "matched_radar_near_clean_coverage_20cm": (
                sum(row["radar_near_clean_coverage"] * report["radar_near_clean_points_in_crop"]
                    for report, row in matched) / near_clean if near_clean else 0.),
        }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    body = []
    for method, row in summary["methods"].items():
        body.append("<tr><td>" + escape(METHOD_LABELS[method]) + "</td>" +
                    f"<td>{row['active_sites']:,}</td>" +
                    f"<td>{row['site_precision_20cm']:.1%}</td>" +
                    f"<td>{row['clean_coverage_20cm']:.1%}</td>" +
                    f"<td>{row['radar_near_clean_coverage_20cm']:.1%}</td></tr>")
    matched_body = []
    for method, row in summary["methods"].items():
        complete = row["matched_budget_complete_frames"] == len(reports)
        precision = f"{row['matched_site_precision_20cm']:.1%}" if complete else "n/a"
        coverage = f"{row['matched_radar_near_clean_coverage_20cm']:.1%}" if complete else "n/a"
        matched_body.append("<tr><td>" + escape(METHOD_LABELS[method]) +
                            f"</td><td>{row['matched_sites']:,}</td><td>{precision}</td>"
                            f"<td>{coverage}</td></tr>")
    links = " ".join(f'<a href="{escape(report["frame"], quote=True)}_interpolation.html">'
                     f'{escape(report["frame"])}</a>' for report in reports)
    downloads = []
    for report in reports:
        prefix = f'{report["frame"]}_interpolation'
        files = " ".join(
            f'<a href="{escape(prefix + "_" + name + ".ply", quote=True)}">{escape(label)}</a>'
            for name, label in POINTCLOUD_LABELS.items())
        files += (f' <a href="{escape(prefix + "_local_sdf.npz", quote=True)}">'
                  'SDF scores and signed distances</a>')
        downloads.append(
            f'<li><b>Frame {escape(report["frame"])}</b>: '
            f'<a href="{escape(prefix + "_pointclouds.zip", quote=True)}">Download all point clouds</a>'
            f'<div class="files">{files}</div></li>')
    page = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
            "<title>Stage I interpolation comparison</title>"
            "<style>body{background:#10161d;color:#e8eff5;font:15px system-ui,sans-serif;"
            "margin:24px;max-width:1100px}a{color:#83c9ed;margin-right:14px}"
            "table{border-collapse:collapse;margin:18px 0}td,th{border:1px solid #456075;"
            "padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}"
            "li{margin:12px 0}.files{margin:5px 0;line-height:1.8}</style>"
            f"<h1>Stage I interpolation · epoch {summary['checkpoint_epoch']}</h1>"
            "<p>Radar-only candidate regions. Clean LiDAR is used only to measure the rows below. "
            "The methods have different active-cell counts, so read precision together with coverage. "
            "This is a support-region audit, not reconstructed point-cloud accuracy. "
            "The metric rows use a fixed 0–80 m by ±40 m by −5–7 m audit area; "
            "the interactive views and PLY files contain every generated point.</p>"
            f"<p>Frame views: {links}</p>"
            "<h2>Full point clouds</h2><p>Each bundle contains uncapped PLY files for raw stacked "
            "radar, raw Stage I proposals, every interpolation, and clean LiDAR as a reference. "
            "The local SDF also includes an NPZ with scores and signed distances.</p><ul>"
            + "".join(downloads) + "</ul>"
            "<table><tr><th>Method</th><th>Active cells</th><th>Cell precision at 20 cm</th>"
            "<th>Clean coverage at 20 cm</th><th>Radar-near clean coverage at 20 cm</th></tr>"
            + "".join(body) + "</table><p>Audit rows use a 0.25 score threshold; the viewer shows all points. "
            "Radar-near clean means at least three stacked radar returns within 1.5 m; "
            "this subset is used only for evaluation.</p>"
            "<h2>Matched active-cell budget</h2><p>The budget comes from the ellipsoid, "
            "trilinear, inverse-distance, Gaussian and edge-aware methods. Nearest and the local "
            "SDF are only comparable on frames where they supply enough active cells; otherwise "
            "their matched scores are marked n/a.</p>"
            "<table><tr><th>Method</th><th>Matched cells</th><th>Cell precision at 20 cm</th>"
            "<th>Radar-near clean coverage at 20 cm</th></tr>" + "".join(matched_body) + "</table>"
            "</html>")
    (output_root / "index.html").write_text(page, encoding="utf-8")


def _viewer(path: Path, frame: str, epoch: int | None, clouds: list[dict],
            limits: tuple[float, ...]) -> None:
    """Synchronized metric cameras; no cloud is truncated."""
    data = {"frame": frame, "epoch": epoch, "clouds": clouds, "limits": limits}
    packed = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    html = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage I interpolation comparison</title><style>
body{background:#10161d;color:#e8eff5;font:13px system-ui,sans-serif;margin:12px}
h1{font-size:20px;margin:0 0 6px}p{color:#acbdca;margin:5px 0 10px}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:10px 0}
input[type=range]{width:180px}button{background:#2b4050;border:1px solid #637d90;border-radius:5px;color:white;padding:5px 9px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:7px}
section{min-width:0;background:#19242e;border:1px solid #3b5265;padding:5px;border-radius:5px}
h2{font-size:12px;margin:0 0 4px;min-height:32px}
.download{display:inline-block;color:#83c9ed;margin:0 0 5px}
canvas{width:100%;height:68vh;min-height:400px;touch-action:none;background:#0b151e;cursor:grab}
@media(max-width:700px){canvas{height:45vh;min-height:300px}}
</style><h1 id="title"></h1><p>Radar-only support methods. Raw Stage I proposals are the model output before interpolation; raw stacked radar is the sensor input. Clean LiDAR is shown only as a reference. Every generated point is present in each panel, with no crop, score cutoff, or point cap. All panels share the same metric camera. <a id="bundle" class="download">Download every point cloud for this frame</a></p>
<div class="controls">
<label><input type="checkbox" id="overlay"> Overlay raw radar on support panels</label><label><input type="checkbox" id="sdfcolor"> Color local SDF by signed distance</label><button id="reset">Reset view</button></div>
<div id="grid" class="grid"></div><p>Blue → cyan → yellow → red indicates a support score, not calibrated accuracy. Edge-aware interpolation is the current Stage II candidate rule; the ellipsoid remains here as a historical comparison. Every interpolation starts from the same raw Stage I proposals; no interpolation uses clean LiDAR. Nearest is an unsmoothed baseline; Gaussian and inverse-distance use compact local kernels. The local SDF estimates oriented planes from nearby predicted seeds. With signed-distance coloring, blue is behind a patch, white is near zero, and red faces the LiDAR sensor; behind is unknown, not occupied.</p>
<script id="data" type="application/json">__DATA__</script><script>
(()=>{const d=JSON.parse(document.getElementById('data').textContent),root=document.getElementById('grid');
document.getElementById('title').textContent=`Frame ${d.frame} · Stage I epoch ${d.epoch??'unknown'}`;
const bundle=document.getElementById('bundle');bundle.href=`${d.frame}_interpolation_pointclouds.zip`;bundle.download=bundle.href;
const cs=d.clouds.map((cloud)=>{let s=document.createElement('section'),h=document.createElement('h2'),a=document.createElement('a'),c=document.createElement('canvas');a.className='download';a.textContent='Download full PLY';a.href=cloud.file;a.download=cloud.file;s.append(h,a,c);root.append(s);return {h,c,cloud};});
const overlay=document.getElementById('overlay'),sdfcolor=document.getElementById('sdfcolor');
const [x0,x1,y0,y1,z0,z1]=d.limits,center=[(x0+x1)/2,(y0+y1)/2,(z0+z1)/2],radius=Math.hypot(x1-x0,y1-y0,z1-z0)/2;
const initial={yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0},v={...initial};
function color(s){const a=[[40,98,190],[81,202,209],[255,225,115],[245,104,104]],q=Math.max(0,Math.min(.9999,s))*3,i=Math.floor(q),t=q-i;return `rgb(${a[i].map((n,k)=>Math.round(n*(1-t)+a[i+1][k]*t)).join(',')})`;}
function distanceColor(s){const t=Math.min(1,Math.abs(s)/.25),base=s>=0?[242,100,84]:[76,148,238];return `rgb(${base.map(v=>Math.round(235*(1-t)+v*t)).join(',')})`;}
function project(p,w,h,scale){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],cy=Math.cos(v.yaw),sy=Math.sin(v.yaw),cp=Math.cos(v.pitch),sp=Math.sin(v.pitch),u=cy*x-sy*y,depth=sy*x+cy*y;return [w/2+v.panX+u*scale,h/2+v.panY-(cp*z-sp*depth)*scale];}
function layer(ctx,cloud,w,h,scale,kind,alpha){ctx.globalAlpha=alpha;for(let i=0;i<cloud.xyz.length;i++){let [x,y]=project(cloud.xyz[i],w,h,scale);if(x<0||x>w||y<0||y>h)continue;ctx.fillStyle=kind==='radar'?'#ffab40':kind==='lidar'?'#45b975':cloud.signed_distance_m&&sdfcolor.checked?distanceColor(cloud.signed_distance_m[i]):color(cloud.score[i]);const size=kind==='score'?2.2:1.2;ctx.fillRect(x-size/2,y-size/2,size,size);}ctx.globalAlpha=1;}
function draw(){for(let index=0;index<cs.length;index++){let {h,c,cloud}=cs[index];h.textContent=`${cloud.name}: ${cloud.xyz.length.toLocaleString()} total points`;let w=c.clientWidth,height=c.clientHeight,ratio=devicePixelRatio||1;c.width=Math.round(w*ratio);c.height=Math.round(height*ratio);let ctx=c.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);let scale=.47*Math.min(w,height)*v.zoom/radius;if(index>1&&overlay.checked)layer(ctx,d.clouds[0],w,height,scale,'radar',.3);layer(ctx,cloud,w,height,scale,cloud.kind,.9);}}
let pending=false;function redraw(){if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;draw();});}
const active=new Map();for(const {c} of cs){c.addEventListener('pointerdown',e=>{c.setPointerCapture(e.pointerId);active.set(e.pointerId,[e.clientX,e.clientY]);});c.addEventListener('pointermove',e=>{let old=active.get(e.pointerId);if(!old)return;let dx=e.clientX-old[0],dy=e.clientY-old[1];active.set(e.pointerId,[e.clientX,e.clientY]);if(e.shiftKey){v.panX+=dx;v.panY+=dy;}else{v.yaw+=dx*.008;v.pitch=Math.max(-1.55,Math.min(1.55,v.pitch+dy*.008));}redraw();});for(let ev of ['pointerup','pointercancel'])c.addEventListener(ev,e=>active.delete(e.pointerId));c.addEventListener('wheel',e=>{e.preventDefault();v.zoom=Math.max(.1,Math.min(80,v.zoom*Math.exp(-e.deltaY*.001)));redraw();},{passive:false});}
overlay.addEventListener('change',redraw);sdfcolor.addEventListener('change',redraw);document.getElementById('reset').onclick=()=>{Object.assign(v,initial);redraw();};window.addEventListener('resize',redraw);redraw();})();
</script></html>'''.replace("__DATA__", packed)
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frame-id", nargs="+", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed-threshold", type=float, default=.25)
    parser.add_argument("--display-threshold", type=float, default=.25)
    parser.add_argument("--lattice-stride", type=int, default=2)
    parser.add_argument("--gaussian-sigma", type=float, default=.6)
    parser.add_argument("--inverse-distance-power", type=float, default=2.)
    parser.add_argument("--sdf-normal-radius-m", type=float, default=1.5)
    parser.add_argument("--sdf-surface-band-m", type=float, default=.2)
    parser.add_argument("--sdf-min-normal-quality", type=float, default=.1)
    parser.add_argument("--max-link-m", type=float, default=2.0)
    parser.add_argument("--feature-sigma", type=float, default=.25)
    parser.add_argument("--score-sigma", type=float, default=.20)
    parser.add_argument("--min-affinity", type=float, default=.20)
    parser.add_argument("--edge-neighbors", type=int, default=8)
    parser.add_argument("--radar-variant")
    args = parser.parse_args()
    if not 0 <= args.seed_threshold <= 1 or not 0 <= args.display_threshold <= 1:
        parser.error("Thresholds must be in [0,1]")
    if (args.lattice_stride < 1 or args.gaussian_sigma <= 0 or args.inverse_distance_power <= 0
            or args.sdf_normal_radius_m <= 0 or args.sdf_surface_band_m <= 0
            or not 0 <= args.sdf_min_normal_quality <= 1
            or args.max_link_m <= 0 or args.feature_sigma <= 0
            or args.score_sigma <= 0 or not 0 < args.min_affinity < 1
            or args.edge_neighbors < 1):
        parser.error("Invalid interpolation parameters")
    limits = (0., 80., -40., 40., -5., 7.)
    model, saved = load_encoder(args.checkpoint, args.device)
    variant = args.radar_variant or saved.get("data", {}).get("radar_variant", "radar_20frames_verified_doppler_radial")
    dataset = VoDStage1Dataset(args.vod_root, args.split, radar_variant=variant, frame_ids=args.frame_id)
    found = {frame.frame_id for frame in dataset.frames}
    if found != set(args.frame_id):
        raise ValueError(f"Missing frame IDs: {sorted(set(args.frame_id) - found)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for sample in dataset:
        frame = sample["frame_id"]
        with torch.no_grad():
            points = sample["radar"].unsqueeze(0).to(args.device)
            valid = torch.ones(points.shape[:2], dtype=torch.bool, device=args.device)
            output = model(points, valid)
        if output.surface is None or output.surface.radii_xyz is None or output.surface.anchor_xyz is None:
            raise ValueError("This comparison needs a Stage-I surface-region checkpoint")
        grid = model.config.grid
        seed_xyz = output.surface.xyz.reshape(-1, 3).detach().cpu().numpy()
        seed_score = output.surface.score.reshape(-1).detach().cpu().numpy()
        fine = output.features["s1"]
        learned_guide = proposal_radar_features(output, grid)
        fields = {
            "nearest": nearest_seed_field(seed_xyz, seed_score, grid,
                                          seed_threshold=args.seed_threshold),
            "trilinear": trilinear_seed_field(seed_xyz, seed_score, grid,
                                              lattice_stride=args.lattice_stride,
                                              seed_threshold=args.seed_threshold),
            "inverse_distance": inverse_distance_seed_field(
                seed_xyz, seed_score, grid, lattice_stride=args.lattice_stride,
                seed_threshold=args.seed_threshold, power=args.inverse_distance_power),
            "gaussian": gaussian_seed_field(seed_xyz, seed_score, grid,
                                            lattice_stride=args.lattice_stride,
                                            seed_threshold=args.seed_threshold,
                                            sigma=args.gaussian_sigma),
            "edge_aware": edge_aware_seed_field(
                seed_xyz, seed_score, learned_guide, grid,
                groups=np.repeat(np.arange(len(output.surface.xyz)), output.surface.xyz.shape[1]),
                lattice_stride=args.lattice_stride,
                seed_threshold=args.seed_threshold,
                max_link_m=args.max_link_m, feature_sigma=args.feature_sigma,
                score_sigma=args.score_sigma, min_affinity=args.min_affinity,
                neighbors=args.edge_neighbors),
            "local_sdf": local_signed_distance_field(
                seed_xyz, seed_score, grid,
                groups=np.repeat(np.arange(len(output.surface.xyz)), output.surface.xyz.shape[1]),
                lattice_stride=args.lattice_stride,
                seed_threshold=args.seed_threshold,
                normal_radius_m=args.sdf_normal_radius_m,
                surface_band_m=args.sdf_surface_band_m,
                min_normal_quality=args.sdf_min_normal_quality),
        }
        baseline = make_candidates(output, grid, confidence_threshold=args.seed_threshold,
                                   method="ellipsoid")
        clouds_xyz = {"ellipsoid": baseline.centers_xyz.detach().cpu().numpy()}
        clouds_xyz.update({name: field.centers_xyz(grid) for name, field in fields.items()})
        clouds_score = {"ellipsoid": baseline.confidence.detach().cpu().numpy()}
        clouds_score.update({name: field.scores for name, field in fields.items()})
        radar = sample["radar"].numpy()[:, :3]
        clean = sample["clean_lidar"].numpy()[:, :3]
        cropped_radar, cropped_clean = radar[_crop(radar, limits)], clean[_crop(clean, limits)]
        # A radar-only audit subset, never an inference gate: at least three
        # stacked returns within 1.5 m of the measured clean surface point.
        radar_near_clean = (np.isfinite(cKDTree(cropped_radar).query(
            cropped_clean, k=3, distance_upper_bound=1.5, workers=-1)[0][:, 2])
                            if len(cropped_radar) >= 3 else np.zeros(len(cropped_clean), dtype=bool))
        prefix = args.output_root / f"{frame}_interpolation"
        write_ply(_pointcloud_path(prefix, "radar"), radar)
        write_ply(_pointcloud_path(prefix, "clean_lidar"), clean)
        write_ply(_pointcloud_path(prefix, "seeds"), seed_xyz, seed_score)
        results = {}
        html_clouds = [
            {"name": "Raw stacked radar", "kind": "radar", "file": _pointcloud_path(prefix, "radar").name,
             "xyz": np.round(radar, 3).tolist()},
            {"name": "Clean LiDAR reference", "kind": "lidar", "file": _pointcloud_path(prefix, "clean_lidar").name,
             "xyz": np.round(clean, 3).tolist()},
            {"name": "Raw Stage I proposals (no interpolation)", "kind": "score",
             "file": _pointcloud_path(prefix, "seeds").name,
             "xyz": np.round(seed_xyz, 3).tolist(),
             "score": np.round(seed_score, 4).tolist()},
        ]
        for key in clouds_xyz:
            xyz, score = clouds_xyz[key], clouds_score[key]
            write_ply(_pointcloud_path(prefix, key), xyz, score)
            crop = _crop(xyz, limits)
            html_clouds.append({"name": METHOD_LABELS[key], "kind": "score",
                                "file": _pointcloud_path(prefix, key).name,
                                "xyz": np.round(xyz, 3).tolist(),
                                "score": np.round(score, 4).tolist()})
            if key == "local_sdf":
                signed = fields[key].signed_distance_m
                assert signed is not None
                html_clouds[-1]["signed_distance_m"] = np.round(signed, 4).tolist()
                np.savez_compressed(prefix.with_name(prefix.name + "_local_sdf.npz"),
                                    xyz=clouds_xyz[key], score=clouds_score[key],
                                    signed_distance_m=signed)
            xyz, score = xyz[crop], score[crop]
            sweep = {}
            for threshold in (.10, .20, .25, .30, .40, .50):
                sweep[f"{threshold:.2f}"] = _rates(xyz[score >= threshold], cropped_clean,
                                                     cropped_radar, radar_near_clean, .2)
            results[key] = {"full_sites": len(clouds_xyz[key]), "crop_sites": len(xyz),
                            "score_mean": float(score.mean()) if len(score) else None,
                            "threshold_sweep_20cm": sweep}
        # Nearest cannot create more than one cell per seed. SDF may abstain
        # when the proposals have no coherent local surface. Neither sets the
        # shared support budget, but both join when they can fill it.
        budget_methods = [key for key in results if key not in ("nearest", "local_sdf")]
        selected_count = min(int(np.count_nonzero(clouds_score[key][_crop(clouds_xyz[key], limits)]
                                                      >= args.display_threshold))
                             for key in budget_methods)
        for key in results:
            xyz = clouds_xyz[key]
            score = clouds_score[key]
            crop = _crop(xyz, limits)
            xyz, score = xyz[crop], score[crop]
            count = min(selected_count, int(np.count_nonzero(score >= args.display_threshold)))
            top = np.argsort(score)[-count:] if count else np.empty(0, dtype=np.int64)
            results[key]["matched_site_count_20cm"] = _rates(xyz[top], cropped_clean,
                                                                cropped_radar, radar_near_clean, .2)
            results[key]["matched_budget_complete"] = count == selected_count
        report = {"frame": frame, "checkpoint": str(args.checkpoint.resolve()),
                  "checkpoint_epoch": saved.get("epoch"), "radar_variant": variant,
                  "input_boundary": "radar only; clean LiDAR used only for metrics and display",
                  "seed_count": len(seed_xyz), "seed_threshold": args.seed_threshold,
                  "display_threshold": args.display_threshold,
                  "lattice_stride": args.lattice_stride,
                  "gaussian_sigma": args.gaussian_sigma,
                  "inverse_distance_power": args.inverse_distance_power,
                  "sdf_normal_radius_m": args.sdf_normal_radius_m,
                  "sdf_surface_band_m": args.sdf_surface_band_m,
                  "sdf_min_normal_quality": args.sdf_min_normal_quality,
                  "sdf_oriented_seeds": fields["local_sdf"].source_seeds,
                  "max_link_m": args.max_link_m,
                  "feature_sigma": args.feature_sigma, "score_sigma": args.score_sigma,
                  "min_affinity": args.min_affinity, "edge_neighbors": args.edge_neighbors,
                  "edge_graph_links": fields["edge_aware"].accepted_edges,
                  "matched_active_sites": selected_count,
                  "crop_xyz_m": limits, "clean_lidar_points_in_crop": len(cropped_clean),
                  "radar_near_clean_audit_definition": "At least 3 stacked radar returns within 1.5 m; evaluation denominator only",
                  "radar_near_clean_points_in_crop": int(radar_near_clean.sum()),
                  "methods": results}
        prefix.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports.append(report)
        all_xyz = [radar, clean, seed_xyz, *clouds_xyz.values()]
        occupied = [points for points in all_xyz if len(points)]
        low = np.min([points.min(axis=0) for points in occupied], axis=0)
        high = np.max([points.max(axis=0) for points in occupied], axis=0)
        padding = max(.5, float(np.max(high - low)) * .02)
        view_limits = tuple(value for lower, upper in zip(low, high)
                            for value in (float(lower - padding), float(upper + padding)))
        _viewer(prefix.with_suffix(".html"), frame, saved.get("epoch"), html_clouds,
                view_limits)
        _bundle_pointclouds(prefix)
        print(f"{frame}: links={fields['edge_aware'].accepted_edges}, matched={selected_count}; " +
              " | ".join(f"{name}={int(np.count_nonzero(clouds_score[name][_crop(clouds_xyz[name], limits)] >= args.display_threshold))}"
                         for name in results) + f" | {prefix.with_suffix('.html')}", flush=True)
    _write_summary(args.output_root, reports)
    print(f"Summary: {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
