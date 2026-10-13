"""Compare local Stage-I support interpolation methods against the ellipsoid.

This is an inference-only audit. It never trains a checkpoint or passes clean
LiDAR to any interpolation method; clean LiDAR is used only for evaluation.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from models.radar_lidar_stage1.data import VoDStage1Dataset
from models.radar_lidar_stage1.interpolation import (
    edge_aware_seed_field, gaussian_seed_field, inverse_distance_seed_field,
    nearest_seed_field, trilinear_seed_field,
)
from models.radar_lidar_stage2.candidate_domain import make_candidates
from scripts.visualize_stage1_confidence_cloud import load_encoder, write_ply


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
        body.append("<tr><td>" + escape(method.replace("_", " ").title()) + "</td>" +
                    f"<td>{row['active_sites']:,}</td>" +
                    f"<td>{row['site_precision_20cm']:.1%}</td>" +
                    f"<td>{row['clean_coverage_20cm']:.1%}</td>" +
                    f"<td>{row['radar_near_clean_coverage_20cm']:.1%}</td></tr>")
    matched_body = []
    for method, row in summary["methods"].items():
        complete = row["matched_budget_complete_frames"] == len(reports)
        precision = f"{row['matched_site_precision_20cm']:.1%}" if complete else "n/a"
        coverage = f"{row['matched_radar_near_clean_coverage_20cm']:.1%}" if complete else "n/a"
        matched_body.append("<tr><td>" + escape(method.replace("_", " ").title()) +
                            f"</td><td>{row['matched_sites']:,}</td><td>{precision}</td>"
                            f"<td>{coverage}</td></tr>")
    links = " ".join(f'<a href="{escape(report["frame"])}_interpolation.html">'
                     f'{escape(report["frame"])}</a>' for report in reports)
    page = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
            "<title>Stage I interpolation comparison</title>"
            "<style>body{background:#10161d;color:#e8eff5;font:15px system-ui,sans-serif;"
            "margin:24px;max-width:1100px}a{color:#83c9ed;margin-right:14px}"
            "table{border-collapse:collapse;margin:18px 0}td,th{border:1px solid #456075;"
            "padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}</style>"
            f"<h1>Stage I interpolation · epoch {summary['checkpoint_epoch']}</h1>"
            "<p>Radar-only candidate regions. Clean LiDAR is used only to measure the rows below. "
            "The methods have different active-cell counts, so read precision together with coverage. "
            "This is a support-region audit, not reconstructed point-cloud accuracy.</p>"
            f"<p>Frame views: {links}</p>"
            "<table><tr><th>Method</th><th>Active cells</th><th>Cell precision at 20 cm</th>"
            "<th>Clean coverage at 20 cm</th><th>Radar-near clean coverage at 20 cm</th></tr>"
            + "".join(body) + "</table><p>Scores use a 0.25 threshold. "
            "Radar-near clean means at least three stacked radar returns within 1.5 m; "
            "this subset is used only for evaluation.</p>"
            "<h2>Matched active-cell budget</h2><p>Within each frame, all methods except Nearest "
            "are compared using the same number of highest-scoring cells. Nearest cannot fill "
            "that budget, so its matched scores are not comparable.</p>"
            "<table><tr><th>Method</th><th>Matched cells</th><th>Cell precision at 20 cm</th>"
            "<th>Radar-near clean coverage at 20 cm</th></tr>" + "".join(matched_body) + "</table>"
            "</html>")
    (output_root / "index.html").write_text(page, encoding="utf-8")


def _viewer(path: Path, frame: str, epoch: int | None, clouds: list[dict],
            limits: tuple[float, ...], threshold: float) -> None:
    """Synchronized metric cameras; no cloud is truncated."""
    data = {"frame": frame, "epoch": epoch, "clouds": clouds, "limits": limits,
            "threshold": threshold}
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
canvas{width:100%;height:68vh;min-height:400px;touch-action:none;background:#0b151e;cursor:grab}
@media(max-width:700px){canvas{height:45vh;min-height:300px}}
</style><h1 id="title"></h1><p>Radar-only support methods. Clean LiDAR is shown for comparison and is never used to create an interpolated field. All panels share the same metric camera; no cloud is capped.</p>
<div class="controls"><label>Minimum score <input type="range" id="threshold" min="0" max="1" step="0.01"><output id="value"></output></label>
<label><input type="checkbox" id="overlay"> Overlay raw radar on support panels</label><button id="reset">Reset view</button></div>
<div id="grid" class="grid"></div><p>Blue → cyan → yellow → red indicates a support score, not calibrated accuracy. The ellipsoid panel is the current Stage II candidate rule. Every other method starts from the same predicted surface seeds and ignores the learned ellipsoid radii. Nearest is an unsmoothed baseline; Gaussian and inverse-distance use compact local kernels.</p>
<script id="data" type="application/json">__DATA__</script><script>
(()=>{const d=JSON.parse(document.getElementById('data').textContent),root=document.getElementById('grid');
document.getElementById('title').textContent=`Frame ${d.frame} · Stage I epoch ${d.epoch??'unknown'}`;
const cs=d.clouds.map((cloud)=>{let s=document.createElement('section'),h=document.createElement('h2'),c=document.createElement('canvas');s.append(h,c);root.append(s);return {h,c,cloud};});
const slider=document.getElementById('threshold'),label=document.getElementById('value'),overlay=document.getElementById('overlay');slider.value=d.threshold;
const [x0,x1,y0,y1,z0,z1]=d.limits,center=[(x0+x1)/2,(y0+y1)/2,(z0+z1)/2],radius=Math.hypot(x1-x0,y1-y0,z1-z0)/2;
const initial={yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0},v={...initial};
function color(s){const a=[[40,98,190],[81,202,209],[255,225,115],[245,104,104]],q=Math.max(0,Math.min(.9999,s))*3,i=Math.floor(q),t=q-i;return `rgb(${a[i].map((n,k)=>Math.round(n*(1-t)+a[i+1][k]*t)).join(',')})`;}
function project(p,w,h,scale){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],cy=Math.cos(v.yaw),sy=Math.sin(v.yaw),cp=Math.cos(v.pitch),sp=Math.sin(v.pitch),u=cy*x-sy*y,depth=sy*x+cy*y;return [w/2+v.panX+u*scale,h/2+v.panY-(cp*z-sp*depth)*scale];}
function layer(ctx,cloud,w,h,scale,kind,alpha){ctx.globalAlpha=alpha;for(let i=0;i<cloud.xyz.length;i++){if(kind==='score'&&cloud.score[i]<Number(slider.value))continue;let [x,y]=project(cloud.xyz[i],w,h,scale);if(x<0||x>w||y<0||y>h)continue;ctx.fillStyle=kind==='radar'?'#ffab40':kind==='lidar'?'#45b975':color(cloud.score[i]);const size=kind==='score'?2.2:1.2;ctx.fillRect(x-size/2,y-size/2,size,size);}ctx.globalAlpha=1;}
function draw(){label.textContent=Number(slider.value).toFixed(2);for(let index=0;index<cs.length;index++){let {h,c,cloud}=cs[index],n=cloud.score?cloud.score.filter(q=>q>=Number(slider.value)).length:cloud.xyz.length;h.textContent=`${cloud.name}: ${n.toLocaleString()} shown / ${cloud.xyz.length.toLocaleString()} in crop`;let w=c.clientWidth,height=c.clientHeight,ratio=devicePixelRatio||1;c.width=Math.round(w*ratio);c.height=Math.round(height*ratio);let ctx=c.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);let scale=.47*Math.min(w,height)*v.zoom/radius;if(index>1&&overlay.checked)layer(ctx,d.clouds[0],w,height,scale,'radar',.3);layer(ctx,cloud,w,height,scale,index===0?'radar':index===1?'lidar':'score',.9);}}
let pending=false;function redraw(){if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;draw();});}
const active=new Map();for(const {c} of cs){c.addEventListener('pointerdown',e=>{c.setPointerCapture(e.pointerId);active.set(e.pointerId,[e.clientX,e.clientY]);});c.addEventListener('pointermove',e=>{let old=active.get(e.pointerId);if(!old)return;let dx=e.clientX-old[0],dy=e.clientY-old[1];active.set(e.pointerId,[e.clientX,e.clientY]);if(e.shiftKey){v.panX+=dx;v.panY+=dy;}else{v.yaw+=dx*.008;v.pitch=Math.max(-1.55,Math.min(1.55,v.pitch+dy*.008));}redraw();});for(let ev of ['pointerup','pointercancel'])c.addEventListener(ev,e=>active.delete(e.pointerId));c.addEventListener('wheel',e=>{e.preventDefault();v.zoom=Math.max(.1,Math.min(80,v.zoom*Math.exp(-e.deltaY*.001)));redraw();},{passive:false});}
slider.addEventListener('input',redraw);overlay.addEventListener('change',redraw);document.getElementById('reset').onclick=()=>{Object.assign(v,initial);redraw();};window.addEventListener('resize',redraw);redraw();})();
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
        anchor = output.surface.anchor_xyz.detach().cpu().numpy()
        source = fine.centers_xyz(grid).detach().cpu().numpy()
        nearest = cKDTree(source).query(anchor, workers=-1)[1]
        learned_guide = np.repeat(fine.features.detach().cpu().numpy()[nearest],
                                  output.surface.xyz.shape[1], axis=0)
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
                groups=np.repeat(np.arange(len(anchor)), output.surface.xyz.shape[1]),
                lattice_stride=args.lattice_stride,
                seed_threshold=args.seed_threshold,
                max_link_m=args.max_link_m, feature_sigma=args.feature_sigma,
                score_sigma=args.score_sigma, min_affinity=args.min_affinity,
                neighbors=args.edge_neighbors),
        }
        baseline = make_candidates(output, grid, confidence_threshold=args.seed_threshold)
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
        write_ply(prefix.with_name(prefix.name + "_radar.ply"), radar)
        write_ply(prefix.with_name(prefix.name + "_clean_lidar.ply"), clean)
        write_ply(prefix.with_name(prefix.name + "_seeds.ply"), seed_xyz, seed_score)
        results = {}
        html_clouds = [{"name": "Radar", "xyz": np.round(cropped_radar, 3).tolist()},
                       {"name": "Clean LiDAR", "xyz": np.round(cropped_clean, 3).tolist()}]
        for key in clouds_xyz:
            xyz, score = clouds_xyz[key], clouds_score[key]
            write_ply(prefix.with_name(prefix.name + f"_{key}.ply"), xyz, score)
            crop = _crop(xyz, limits)
            xyz, score = xyz[crop], score[crop]
            html_clouds.append({"name": key.replace("_", " ").title(),
                                "xyz": np.round(xyz, 3).tolist(),
                                "score": np.round(score, 4).tolist()})
            sweep = {}
            for threshold in (.10, .20, .25, .30, .40, .50):
                sweep[f"{threshold:.2f}"] = _rates(xyz[score >= threshold], cropped_clean,
                                                     cropped_radar, radar_near_clean, .2)
            results[key] = {"full_sites": len(clouds_xyz[key]), "crop_sites": len(xyz),
                            "score_mean": float(score.mean()) if len(score) else None,
                            "threshold_sweep_20cm": sweep}
        # Nearest deliberately cannot create more than one cell per seed. It
        # stays in the raw comparison, but cannot set the shared support budget.
        budget_methods = [key for key in results if key != "nearest"]
        selected_count = min(int(np.count_nonzero(clouds_score[key][_crop(clouds_xyz[key], limits)]
                                                      >= args.display_threshold))
                             for key in budget_methods)
        for key in results:
            xyz = clouds_xyz[key]
            score = clouds_score[key]
            crop = _crop(xyz, limits)
            xyz, score = xyz[crop], score[crop]
            count = min(selected_count, len(xyz))
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
        _viewer(prefix.with_suffix(".html"), frame, saved.get("epoch"), html_clouds,
                limits, args.display_threshold)
        print(f"{frame}: links={fields['edge_aware'].accepted_edges}, matched={selected_count}; " +
              " | ".join(f"{name}={int(np.count_nonzero(clouds_score[name][_crop(clouds_xyz[name], limits)] >= args.display_threshold))}"
                         for name in results) + f" | {prefix.with_suffix('.html')}", flush=True)
    _write_summary(args.output_root, reports)
    print(f"Summary: {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
