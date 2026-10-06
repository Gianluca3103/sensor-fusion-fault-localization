"""Render Stage-1 local correspondence diagnostics for one VoD frame."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch

from models.radar_lidar_stage1 import RadarLidarStage1, Stage1Config
from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from models.radar_lidar_stage1.train import config_from_dict
from models.radar_lidar_stage1.visualize import save_diagnostics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--vod-root",type=Path,required=True)
    parser.add_argument("--frame-id",required=True)
    parser.add_argument("--split",default="val")
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--radar-variant",help="Override the checkpoint's radar stack variant")
    parser.add_argument("--output-prefix",type=Path,required=True)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False) if args.checkpoint else None
    config=config_from_dict(checkpoint["config"]) if checkpoint else Stage1Config()
    model=RadarLidarStage1(config).to(args.device).eval()
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    variant=args.radar_variant or (checkpoint or {}).get("data",{}).get("radar_variant","radar_20frames_verified_doppler_radial")
    dataset=VoDStage1Dataset(args.vod_root,args.split,frame_ids=[args.frame_id],radar_variant=variant)
    batch=collate_stage1([dataset[0]])
    paths=save_diagnostics(model,batch,args.output_prefix)
    print(*paths,sep="\n")


if __name__=="__main__":
    main()
