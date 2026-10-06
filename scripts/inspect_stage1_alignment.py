"""Physical XYZ overlays for the Stage-1 synchronized sensor audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Fault_Localization_Model.vod_dataset.vod_io import (
    align_radar_to_lidar, discover_vod_frames, load_vod_lidar,
    load_vod_radar, load_vod_radar_to_lidar,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    args = parser.parse_args()
    frame = discover_vod_frames(args.vod_root,args.split,radar_variant=args.radar_variant,frame_ids=[args.frame_id])[0]
    radar = align_radar_to_lidar(load_vod_radar(frame.radar_path),load_vod_radar_to_lidar(frame.lidar_calibration_path,frame.radar_calibration_path))
    lidar = load_vod_lidar(frame.lidar_path)
    # The same physical XYZ bounds are used on both overlays.
    region_r = (radar[:,0]>=0)&(radar[:,0]<60)&(np.abs(radar[:,1])<30)&(radar[:,2]>-4)&(radar[:,2]<5)
    region_l = (lidar[:,0]>=0)&(lidar[:,0]<60)&(np.abs(lidar[:,1])<30)&(lidar[:,2]>-4)&(lidar[:,2]<5)
    r,l = radar[region_r], lidar[region_l]
    if not len(r) or not len(l):
        raise ValueError("No overlapping sensor points in inspection region")
    fig, axes = plt.subplots(2,2,figsize=(15,9),sharex=True)
    for ax,inds,ylim,title in ((axes[0,0],(0,1),(-30,30),"XY clean LiDAR"),(axes[0,1],(0,1),(-30,30),"XY radar over clean LiDAR"),(axes[1,0],(0,2),(-4,5),"XZ clean LiDAR"),(axes[1,1],(0,2),(-4,5),"XZ radar over clean LiDAR")):
        step=max(1,len(l)//50000)
        ax.scatter(l[::step,inds[0]],l[::step,inds[1]],s=.15,c="#238b45",alpha=.4,rasterized=True)
        if "over" in title:
            ax.scatter(r[:,inds[0]],r[:,inds[1]],s=4,c="#fb8500",alpha=.65,rasterized=True)
        ax.set(xlim=(0,60),ylim=ylim,title=title,xlabel="x forward [m]",ylabel=("y left [m]" if inds[1]==1 else "z up [m]"))
        ax.grid(alpha=.2)
    fig.suptitle(f"VoD {args.split} {frame.frame_id}: LiDAR {len(lidar):,} points, aligned radar {len(radar):,} returns")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output,dpi=160)
    print(args.output)


if __name__ == "__main__":
    main()
