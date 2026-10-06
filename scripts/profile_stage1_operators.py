"""Profile Stage I voxel encoding and backward operators on one real VoD batch.

Run with --device cuda only after the training GPU is free. This creates no
checkpoint and does not change the saved training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch.profiler import ProfilerActivity, profile

from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from models.radar_lidar_stage1.model import RadarLidarStage1
from models.radar_lidar_stage1.train import config_from_dict


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=Path("configs/radar_lidar_stage1_small.json"))
    parser.add_argument("--radar-variant",default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--start-index",type=int,default=1000)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.batch_size<1 or args.start_index<0:
        parser.error("Batch size must be positive and start index nonnegative")
    if args.device=="cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    dataset=VoDStage1Dataset(args.vod_root,"train",radar_variant=args.radar_variant)
    if args.start_index+args.batch_size>len(dataset):
        parser.error("Requested batch extends past the training split")
    batch=collate_stage1([dataset[i] for i in range(args.start_index,args.start_index+args.batch_size)])
    batch={key:value.to(args.device) if isinstance(value,torch.Tensor) else value for key,value in batch.items()}
    config=config_from_dict(json.loads(args.config.read_text()))
    model=RadarLidarStage1(config).to(args.device).train()
    activities=[ProfilerActivity.CPU]
    if args.device=="cuda":
        activities.append(ProfilerActivity.CUDA)

    def sync() -> None:
        if args.device=="cuda":
            torch.cuda.synchronize()

    def measure(fn):
        sync(); start=time.perf_counter()
        with profile(activities=activities,record_shapes=False,profile_memory=False,with_stack=False) as prof:
            fn()
        sync()
        rows=[]
        for event in prof.key_averages():
            rows.append({"name":event.key,"calls":event.count,
                         "self_cpu_ms":event.self_cpu_time_total/1000,
                         "self_device_ms":getattr(event,"self_device_time_total",0)/1000})
        sort_key="self_device_ms" if args.device=="cuda" else "self_cpu_ms"
        rows.sort(key=lambda row:row[sort_key],reverse=True)
        return {"elapsed_s":time.perf_counter()-start,"top_operators":rows[:30]}

    with torch.no_grad():
        voxel=measure(lambda: (
            model.radar_only.point(batch["radar"],batch["radar_valid"],defer_diagnostics=True),
            model.lidar_teacher.point(batch["clean_lidar"],batch["clean_lidar_valid"],defer_diagnostics=True),
        ))
    model.zero_grad(set_to_none=True)
    losses,_=model.forward_train(batch["radar"],batch["radar_valid"],
                                 batch["clean_lidar"],batch["clean_lidar_valid"],
                                 defer_diagnostics=True)
    backward=measure(lambda: losses["loss/total"].backward())
    result={"device":args.device,"batch_size":args.batch_size,"start_index":args.start_index,
            "point_counts":[{"radar":int(r),"lidar":int(l)} for r,l in zip(
                batch["radar_valid"].sum(-1),batch["clean_lidar_valid"].sum(-1))],
            "voxel_encoding":voxel,"backward":backward}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    for name in ("voxel_encoding","backward"):
        print(f"{name}: {result[name]['elapsed_s']:.3f}s")
        for row in result[name]["top_operators"][:10]:
            print(f"  {row['name'][:55]:55s} {row['self_device_ms' if args.device=='cuda' else 'self_cpu_ms']:.2f} ms ({row['calls']} calls)")
    print(args.output)


if __name__=="__main__":
    main()
