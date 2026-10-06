"""Stage-1 correspondence, light-probe and radar corruption evaluation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from Fault_Localization_Model.vod_dataset.vod_io import load_vod_lidar_to_camera
from .correspondence import local_neighbors
from .metrics import Stage1MetricAccumulator


def _summary(values: np.ndarray) -> dict:
    if not len(values):
        return {"count":0}
    return {"count":len(values),"mean_m":float(np.mean(values)),"median_m":float(np.median(values)),"p90_m":float(np.percentile(values,90)),**{f"within_{x:g}m":float(np.mean(values<=x)) for x in (.1,.2,.5,1.0)}}


@torch.no_grad()
def evaluate_pair(model, radar, radar_valid, clean, clean_valid) -> dict:
    """Physical-radius correspondence and radar-only probe metrics for one batch."""
    was_training=model.training
    model.eval()
    accumulator=Stage1MetricAccumulator(model.config)
    accumulator.update(model,radar,radar_valid,clean,clean_valid)
    full=accumulator.finish()
    result={name:{**row,"matched_count":row["valid_corr_query_count"],
        "candidate_coverage":1-row["no_corr_fraction"] if row["no_corr_fraction"] is not None else None,
        **{f"recall_at_{k}":row[f"corr_r{k}"] for k in (1,5,10)}} for name,row in full["scales"].items()}
    result["geometry"]=full["geometry"]
    result["confidence"]=full["confidence"]
    if was_training:
        model.train()
    return result


def _inside_boxes(xyz_lidar: np.ndarray, box_lines: list[str], camera_from_lidar: np.ndarray) -> tuple[np.ndarray,list[str]]:
    """KITTI bottom-center camera boxes; IDs are line indices in the label file."""
    xyz_camera=np.column_stack((xyz_lidar,np.ones(len(xyz_lidar)))) @ camera_from_lidar.T
    xyz_camera=xyz_camera[:,:3]
    ids=np.full(len(xyz_lidar),-1,dtype=np.int32)
    classes=[]
    for index,line in enumerate(box_lines):
        parts=line.split()
        if len(parts)<15:
            raise ValueError(f"Malformed KITTI object annotation: {line}")
        cls=parts[0]
        classes.append(cls)
        h,w,length=map(float,parts[8:11]); x,y,z=map(float,parts[11:14]); yaw=float(parts[14])
        dx=xyz_camera[:,0]-x; dy=xyz_camera[:,1]-y; dz=xyz_camera[:,2]-z
        local_x=np.cos(yaw)*dx-np.sin(yaw)*dz
        local_z=np.sin(yaw)*dx+np.cos(yaw)*dz
        inside=(np.abs(local_x)<=w/2)&(np.abs(local_z)<=length/2)&(dy>=-h)&(dy<=0)
        ids[(ids<0)&inside]=index
    return ids,classes


@torch.no_grad()
def evaluate_object_instances(model, radar, radar_valid, clean, clean_valid, label_path: str | Path, lidar_calibration_path: str | Path,
                              *, intermediates=None) -> dict:
    """Held-out, label-only analysis; boxes never enter the model."""
    if radar.shape[0]!=1:
        raise ValueError("Object-instance evaluation expects one frame")
    model.eval()
    output=(intermediates["output"] if intermediates is not None
            else model.forward_radar(radar,radar_valid))
    lidar_levels=(intermediates["lidar_levels"] if intermediates is not None
                  else model.lidar_teacher(clean,clean_valid)[0])
    lines=Path(label_path).read_text().splitlines()
    transform=load_vod_lidar_to_camera(lidar_calibration_path)
    report={}
    for i,(name,li) in enumerate(zip(output.features,lidar_levels)):
        r=output.features[name]
        if not len(r.coords) or not len(li.coords):
            report[name]={"count":0}; continue
        rid,classes=_inside_boxes(r.centers_xyz(model.config.grid).cpu().numpy(),lines,transform)
        lid,_=_inside_boxes(li.centers_xyz(model.config.grid).cpu().numpy(),lines,transform)
        neighbors=(intermediates["neighbors"][name] if intermediates is not None
                   else local_neighbors(r,li,model.config.grid,model.config.attention_radii_m[i],model.config.max_neighbors))
        attention=(intermediates["attention"].get(name) if intermediates is not None else None)
        if attention is None:
            attention=model.correspondence[i](r,li,model.config.grid,neighbors,
                model.config.attention_radii_m[i],model.config.positive_radii_m[i],
                model.config.temperature,model.config.negative_strategy,model.config.num_negatives)
        logits=attention["logits"]
        per_class=defaultdict(lambda: {"count":0,"hit1":0,"hit3":0,"hit5":0,"same_class_count":0,"same_class_hit1":0,"same_class_hit3":0,"same_class_hit5":0})
        for query,instance in enumerate(rid):
            if instance<0 or not np.any(lid==instance):
                continue
            cls=classes[instance]
            values=per_class[cls]
            values["count"]+=1
            local_idx=neighbors.indices[query][neighbors.valid[query]].cpu().numpy()
            local_scores=logits[query][neighbors.valid[query]].cpu().numpy()
            top=local_idx[np.argsort(-local_scores)[:5]]
            for n in (1,3,5):
                values[f"hit{n}"]+=int(np.any(lid[top[:n]]==instance))
            same_class=np.array([j>=0 and classes[j]==cls for j in lid])
            if np.any(same_class[local_idx] & (lid[local_idx]!=instance)):
                values["same_class_count"]+=1
                eligible=same_class[local_idx]
                top_local=local_idx[eligible][np.argsort(-local_scores[eligible])[:5]]
                for n in (1,3,5):
                    values[f"same_class_hit{n}"]+=int(np.any(lid[top_local[:n]]==instance))
        report[name]={cls:{**v,**{f"recall_at_{n}":v[f"hit{n}"]/max(v["count"],1) for n in (1,3,5)},**{f"same_class_recall_at_{n}":v[f"same_class_hit{n}"]/max(v["same_class_count"],1) for n in (1,3,5)}} for cls,v in per_class.items()}
    return report


@torch.no_grad()
def evaluate_corruptions(model, radar, radar_valid, clean, clean_valid, *, mismatched_radar=None, mismatched_valid=None, shifts_m=(.5,1.0,2.0,5.0)) -> dict:
    """Compare correct pairing with mismatched and physically shifted radar.

    Fixed clean-LiDAR anchors are taken from the aligned input, so shifting
    radar cannot silently redefine the nearest-surface target.
    """
    if radar.shape[0]!=1 or clean.shape[0]!=1:
        raise ValueError("Corruption audit expects a single synchronized frame")
    original_radar=radar[0,radar_valid[0],:3].detach().cpu().numpy()
    clean_xyz=clean[0,clean_valid[0],:3].detach().cpu().numpy()
    if not len(original_radar) or not len(clean_xyz):
        raise ValueError("Corruption audit requires nonempty radar and clean LiDAR")
    nearest_distance,nearest_index=cKDTree(clean_xyz).query(original_radar)
    eligible=nearest_distance<=1.0
    anchors=clean_xyz[nearest_index[eligible]]

    def anchored_geometry(condition_radar,condition_valid):
        out=model.forward_radar(condition_radar,condition_valid)
        sites=out.features["s1"]
        if not len(sites.coords) or not len(anchors):
            return {"anchor_count":len(anchors),"predicted_sites":len(sites.coords),"coverage_within_1m":0.0}
        pred=model.probe(out)["s1"]
        emitted=torch.sigmoid(pred[:,0])>=.5
        predicted_xyz=(sites.centers_xyz(model.config.grid)[emitted]+pred[emitted,1:4]*model.config.attention_radii_m[0]).detach().cpu().numpy()
        if not len(predicted_xyz):
            return {"anchor_count":len(anchors),"predicted_sites":0,"coverage_within_1m":0.0}
        error,_=cKDTree(predicted_xyz).query(anchors)
        return {"anchor_count":len(anchors),"predicted_sites":len(predicted_xyz),"coverage_within_1m":float(np.mean(error<=1.0)),"distance_to_original_clean_anchor":_summary(error)}

    baseline=evaluate_pair(model,radar,radar_valid,clean,clean_valid)
    baseline["anchor_geometry"]=anchored_geometry(radar,radar_valid)
    result={"aligned":baseline}
    if mismatched_radar is not None:
        if mismatched_valid is None:
            raise ValueError("A mask is required with mismatched radar")
        result["mismatched_scene"]=evaluate_pair(model,mismatched_radar,mismatched_valid,clean,clean_valid)
        result["mismatched_scene"]["anchor_geometry"]=anchored_geometry(mismatched_radar,mismatched_valid)
    for shift in shifts_m:
        altered=radar.clone()
        altered[...,0][radar_valid]+=shift
        result[f"x_shift_{shift:g}m"]=evaluate_pair(model,altered,radar_valid,clean,clean_valid)
        result[f"x_shift_{shift:g}m"]["anchor_geometry"]=anchored_geometry(altered,radar_valid)
    for scale in baseline:
        if scale=="anchor_geometry":
            continue
        aligned=baseline[scale].get("recall_at_1")
        if aligned is None:
            continue
        for condition,scores in result.items():
            if condition=="aligned": continue
            degraded=scores[scale].get("recall_at_1")
            if degraded is not None and degraded>=aligned:
                scores[scale]["warning"]="Corruption did not reduce proxy Recall@1; inspect learned alignment and metric limitations"
    aligned_anchor=baseline["anchor_geometry"].get("coverage_within_1m",0)
    for condition,scores in result.items():
        if condition=="aligned": continue
        if scores["anchor_geometry"].get("coverage_within_1m",0)>=aligned_anchor:
            scores["anchor_geometry"]["warning"]="Corrupted radar retained or improved fixed-anchor coverage; no alignment benefit established"
    return result
