"""Held-out Stage-1 retrieval/probe audit; optional boxes and radar corruption."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from Fault_Localization_Model.vod_dataset.vod_io import resolve_vod_public_root
from models.radar_lidar_stage1 import RadarLidarStage1
from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from models.radar_lidar_stage1.evaluate import evaluate_corruptions, evaluate_object_instances, evaluate_pair
from models.radar_lidar_stage1.metrics import Stage1MetricAccumulator
from models.radar_lidar_stage1.train import config_from_dict


def _tensor_batch(dataset, index, device):
    batch=collate_stage1([dataset[index]])
    return {k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--vod-root",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True,help="Full Stage-1 training checkpoint")
    parser.add_argument("--radar-variant",help="Override the checkpoint's radar stack variant")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--split",default="val")
    parser.add_argument("--limit",type=int)
    parser.add_argument("--corruptions",action="store_true")
    parser.add_argument("--object-instances",action="store_true")
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    model=RadarLidarStage1(config_from_dict(checkpoint["config"])).to(args.device).eval()
    model.load_state_dict(checkpoint["model"])
    variant=args.radar_variant or checkpoint.get("data",{}).get("radar_variant","radar_20frames_verified_doppler_radial")
    dataset=VoDStage1Dataset(args.vod_root,args.split,radar_variant=variant)
    count=min(args.limit or len(dataset),len(dataset))
    frames=[]
    metrics=Stage1MetricAccumulator(model.config)
    for i in range(count):
        batch=_tensor_batch(dataset,i,args.device)
        inputs=(batch["radar"],batch["radar_valid"],batch["clean_lidar"],batch["clean_lidar_valid"])
        metrics.update(model,*inputs)
        record={"frame_id":batch["frame_id"][0],"spatial":evaluate_pair(model,*inputs)}
        if args.corruptions:
            other=_tensor_batch(dataset,(i+1)%len(dataset),args.device)
            record["corruptions"]=evaluate_corruptions(model,*inputs,mismatched_radar=other["radar"],mismatched_valid=other["radar_valid"])
        if args.object_instances:
            frame=dataset.frames[i]
            label=resolve_vod_public_root(args.vod_root)/"lidar"/"training"/"label_2"/f"{frame.frame_id}.txt"
            if label.is_file():
                record["object_instances"]=evaluate_object_instances(model,*inputs,label,frame.lidar_calibration_path)
                metrics.add_instances(record["object_instances"])
        frames.append(record)
        print(f"evaluated {i+1}/{count}: {record['frame_id']}",flush=True)
    report={"checkpoint":str(args.checkpoint),"split":args.split,"frames":frames,"summary":metrics.finish(),"caveat":"Geometric proximity is a proxy for correspondence, not proof that radar and LiDAR measured the same reflector. Confidence calibration must be judged from held-out reliability bins."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(args.output)


if __name__=="__main__":
    main()
