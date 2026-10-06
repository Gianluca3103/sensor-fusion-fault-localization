"""Physical XYZ Stage-1 sensor, sparse-scale and local-attention diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .correspondence import local_neighbors


@torch.no_grad()
def save_diagnostics(model, batch: dict, output_prefix: str | Path, *, max_points: int = 30000) -> tuple[Path,Path]:
    """Save one PNG and JSON record; inference confidence is labeled uncalibrated."""
    prefix=Path(output_prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    device=next(model.parameters()).device
    radar=batch["radar"].to(device)
    clean=batch["clean_lidar"].to(device)
    rv=batch["radar_valid"].to(device)
    lv=batch["clean_lidar_valid"].to(device)
    if radar.shape[0]!=1:
        raise ValueError("Visualize one synchronized frame at a time")
    model.eval()
    output=model.forward_radar(radar,rv)
    teacher,_=model.lidar_teacher(clean,lv)
    raw_r=radar[0,rv[0],:3].cpu().numpy()
    raw_l=clean[0,lv[0],:3].cpu().numpy()
    fig,axes=plt.subplots(2,2,figsize=(15,11))
    rng=np.random.default_rng(0)
    if len(raw_l)>max_points:
        raw_l=raw_l[rng.choice(len(raw_l),max_points,replace=False)]
    axes[0,0].scatter(raw_l[:,0],raw_l[:,1],s=.2,c="#238b45",alpha=.45,label="clean LiDAR")
    axes[0,0].scatter(raw_r[:,0],raw_r[:,1],s=2,c="#ff9f1c",alpha=.7,label="radar")
    axes[0,0].set(title="Aligned raw sensors: XY",xlabel="x [m]",ylabel="y [m]",xlim=(0,80),ylim=(-40,40))
    axes[0,0].legend(markerscale=4)
    axes[0,1].scatter(raw_l[:,0],raw_l[:,2],s=.2,c="#238b45",alpha=.4)
    axes[0,1].scatter(raw_r[:,0],raw_r[:,2],s=2,c="#ff9f1c",alpha=.7)
    axes[0,1].set(title="Aligned raw sensors: XZ",xlabel="x [m]",ylabel="z [m]",xlim=(0,80),ylim=(-5,7))
    palette=("#1f77b4","#d62728","#9467bd","#2ca02c")
    records=[]
    selected_xy=[]
    for i,(name,li) in enumerate(zip(output.features,teacher)):
        r=output.features[name]
        xyz=r.centers_xyz(model.config.grid).cpu().numpy()
        if len(xyz)>max_points:
            xyz=xyz[rng.choice(len(xyz),max_points,replace=False)]
        axes[1,0].scatter(xyz[:,0],xyz[:,1],s=1.5,c=palette[i],alpha=.55,label=f"{name} {len(r.coords):,} sites")
        neighbors=local_neighbors(r,li,model.config.grid,model.config.attention_radii_m[i],model.config.max_neighbors)
        attention=model.correspondence[i](r,li,model.config.grid,neighbors,
            model.config.attention_radii_m[i],model.config.positive_radii_m[i],
            model.config.temperature,model.config.negative_strategy,model.config.num_negatives)
        covered=torch.where(neighbors.valid.any(-1))[0]
        if not len(covered):
            records.append({"scale":name,"candidate_count":0})
            continue
        # A stable query near the forward center of the displayed scene.
        q_xyz=r.centers_xyz(model.config.grid)
        candidate=covered[torch.linalg.vector_norm(q_xyz[covered]-q_xyz.new_tensor((25.0,0.0,-1.0)),dim=-1).argmin()]
        valid=neighbors.valid[candidate]
        li_idx=neighbors.indices[candidate,valid]
        weights=attention["weights"][candidate,valid]
        max_local=weights.argmax()
        matched=li.centers_xyz(model.config.grid)[li_idx[max_local]]
        query_xyz=q_xyz[candidate]
        record={"scale":name,"radar_xyz_m":query_xyz.cpu().tolist(),"matched_lidar_xyz_m":matched.cpu().tolist(),"correspondence_distance_m":float(torch.linalg.vector_norm(matched-query_xyz)),"correspondence_score":float(weights[max_local]),"confidence":float(output.confidence.features[candidate]) if i==0 and candidate<len(output.confidence.features) else None,"candidates":[{"lidar_xyz_m":p,"weight":float(w),"distance_m":float(d)} for p,w,d in zip(li.centers_xyz(model.config.grid)[li_idx].cpu().tolist(),weights.cpu().tolist(),neighbors.distances_m[candidate,valid].cpu().tolist())]}
        records.append(record)
        selected_xy.append(query_xyz[:2].cpu().numpy())
        axes[1,1].scatter(float(query_xyz[0]),float(query_xyz[1]),s=45,c=palette[i],marker="x",label=f"{name} query")
        locs=li.centers_xyz(model.config.grid)[li_idx].cpu().numpy()
        axes[1,1].scatter(locs[:,0],locs[:,1],s=10+weights.cpu().numpy()*100,c=palette[i],alpha=.4)
        axes[1,1].plot([float(query_xyz[0]),float(matched[0])],[float(query_xyz[1]),float(matched[1])],c=palette[i],linewidth=1)
    axes[1,0].set(title="Radar active sites by scale: XY",xlabel="x [m]",ylabel="y [m]",xlim=(0,80),ylim=(-40,40))
    axes[1,1].set(title="Selected local radar → LiDAR neighborhoods: XY",xlabel="x [m]",ylabel="y [m]")
    if selected_xy:
        focus=np.mean(selected_xy,axis=0)
        span=max(4.0,2*max(model.config.attention_radii_m[:len(selected_xy)]))
        axes[1,1].set(xlim=(focus[0]-span,focus[0]+span),ylim=(focus[1]-span,focus[1]+span))
    for ax in axes.flat:
        ax.grid(alpha=.15)
    axes[1,0].legend(markerscale=3)
    axes[1,1].legend()
    fig.suptitle(f"Stage 1 sample {batch['frame_id'][0]} | confidence supervised={output.metadata['confidence_trained']}, calibration unverified")
    fig.tight_layout()
    png=prefix.with_suffix(".png")
    meta=prefix.with_suffix(".json")
    fig.savefig(png,dpi=150)
    plt.close(fig)
    meta.write_text(json.dumps({"sample":batch["frame_id"][0],"split":batch["split"][0],"radar_points":int(rv.sum()),"clean_lidar_points":int(lv.sum()),"scales":output.metadata["scales"],"queries":records},indent=2))
    return png,meta
