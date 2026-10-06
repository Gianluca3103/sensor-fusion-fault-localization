"""Train/evaluate Stage 1 on official VoD pairs; export an isolated radar-only model."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Stage1Config, VoxelGrid
from .data import VoDStage1Dataset, collate_stage1
from .evaluate import evaluate_corruptions, evaluate_object_instances
from .metrics import Stage1MetricAccumulator
from .model import RadarLidarStage1, RadarOnlyEncoder
from Fault_Localization_Model.vod_dataset.vod_io import resolve_vod_public_root


def config_from_dict(values: dict) -> Stage1Config:
    values=dict(values)
    grid=values.pop("grid",{})
    if isinstance(grid,dict):
        values["grid"]=VoxelGrid(**{k:tuple(v) if isinstance(v,list) else v for k,v in grid.items()})
    for key in ("channels","attention_radii_m","growth_scales","positive_radii_m","corr_scale_weights"):
        if key in values:
            values[key]=tuple(values[key])
    allowed={field.name for field in fields(Stage1Config)}
    unknown=set(values)-allowed
    if unknown:
        raise ValueError(f"Unknown Stage-1 config keys: {sorted(unknown)}")
    return Stage1Config(**values)


def save_checkpoint(path: Path, model: RadarLidarStage1, optimizer, scheduler, epoch: int, step: int, validation: dict, data: dict | None = None, best_scores: dict | None = None) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"config":model.config.as_dict(),"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"epoch":epoch,"step":step,"validation":validation,"data":data or {},"best_scores":best_scores or {},"confidence_calibrated":model.radar_only.confidence_calibrated,"confidence_trained":model.radar_only.confidence_trained},path)


def export_radar_only(path: Path, model: RadarLidarStage1, data: dict | None = None) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    state=model.radar_only_state_dict()
    state["data"]=data or {}
    torch.save(state,path)


def load_radar_only(path: str | Path, device: str = "cpu") -> RadarOnlyEncoder:
    state=torch.load(path,map_location=device,weights_only=False)
    config=config_from_dict(state["config"])
    model=RadarOnlyEncoder(config)
    model.load_state_dict(state["radar_only"])
    model.confidence_calibrated=bool(state.get("confidence_calibrated",False))
    model.confidence_trained=bool(state.get("confidence_trained",False))
    return model.to(device).eval()


def _move(batch: dict, device: str) -> dict:
    return {k:(v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}


def _accumulate_scalar(bucket: dict, key: str, value, weight: int = 1) -> None:
    """Accumulate diagnostics on their device instead of synchronizing each batch."""
    contribution=value.detach().to(dtype=torch.float64)*weight if isinstance(value,torch.Tensor) else value*weight
    bucket[key]=bucket.get(key,0)+contribution


def _python_scalar(value) -> float:
    return float(value)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def _selection_value(metrics: dict, name: str):
    corr=metrics["correspondence"]
    if name=="corr_r1":return corr["corr_r1"]
    if name=="corr_within_0.2m":return corr["corr_within_0.2m"]
    if name=="geom_f1_0.2m":return metrics["geometry"]["0.2m"]["f1"]
    if name=="geom_f1_0.5m":return metrics["geometry"]["0.5m"]["f1"]
    raise ValueError(name)


def _flat_validation(losses: dict, metrics: dict) -> dict:
    flat={f"val/{key}":value for key,value in losses.items()}
    corr=metrics["correspondence"]
    for key,value in corr.items():flat[f"val/{key}"]=value
    for name,scale in metrics["scales"].items():
        for key,value in scale.items():flat[f"val/{name}/{key}"]=value
    for tolerance,row in metrics["geometry"].items():
        for key in ("precision","recall","f1"):
            flat[f"val/geom_{key}_{tolerance}"]=row[key]
    flat["val/conf_mean"]=metrics["confidence"]["mean"]
    flat["val/conf_ece"]=metrics["confidence"]["ece"]
    half=next(row for row in metrics["confidence"]["threshold_sweep"] if row["threshold"]==0.5)
    for key in ("coverage","target_coverage","precision","recall","f1"):
        flat[f"val/conf_threshold_0.5/{key}"]=half[key]
    for key,value in metrics["instances"].get("s1",{}).get("all",{}).items():
        if key in ("r1","r3","r5","same_class_r1","same_class_r3","same_class_r5"):
            flat[f"val/{key.replace('same_class_', 'same_class_instance_') if key.startswith('same_class_') else 'instance_'+key}"]=value
    return flat


def _tensorboard(writer,epoch: int,train: dict,flat: dict,metrics: dict) -> None:
    if writer is None:return
    for key,value in train.items():
        if key.startswith("loss/"):writer.add_scalar("Loss/train_"+key.removeprefix("loss/"),value,epoch)
    for key,value in flat.items():
        if value is None or not isinstance(value,(int,float)):continue
        field=key.removeprefix("val/")
        if field.startswith("loss/"):tag="Loss/val_"+field.removeprefix("loss/")
        elif field.startswith("corr_"):tag="Correspondence/"+field
        elif field.startswith("geom_"):tag="Geometry/"+field
        elif field.startswith("conf_"):tag="Confidence/"+field
        elif field.startswith(("instance_","same_class_instance_")) or field in ("query_count","valid_corr_query_count","no_corr_query_count","no_corr_fraction"):
            tag="Diagnostics/"+field
        elif field.startswith("s") and "/" in field:
            scale,metric=field.split("/",1);tag=f"Scales/{scale.upper()}/{metric}"
        else:continue
        writer.add_scalar(tag,value,epoch)
    for row in metrics["confidence"]["threshold_sweep"]:
        if row["precision"] is not None:writer.add_scalar(f"Confidence/precision_at_threshold_{row['threshold']:.1f}",row["precision"],epoch)
        if row["coverage"] is not None:writer.add_scalar(f"Confidence/coverage_at_threshold_{row['threshold']:.1f}",row["coverage"],epoch)
    writer.flush()


def _corruption_summary(frames: list[dict]) -> dict:
    summary={}
    for condition in ("aligned","x_shift_0.5m","x_shift_1m","x_shift_2m","x_shift_5m","mismatched_scene"):
        rows=[frame["conditions"][condition] for frame in frames if condition in frame["conditions"]]
        valid=sum(row["s1"]["valid_corr_query_count"] for row in rows)
        hits=sum((row["s1"]["corr_r1"] or 0)*row["s1"]["valid_corr_query_count"] for row in rows)
        error_sum=sum((row["s1"]["corr_error_mean_m"] or 0)*row["s1"]["valid_corr_query_count"] for row in rows)
        geo=[row["geometry"]["0.2m"] for row in rows]
        tp=sum(row["tp"] for row in geo); predictions=sum(row["predictions"] for row in geo);targets=sum(row["targets"] for row in geo)
        precision=tp/predictions if predictions else None;recall=tp/targets if targets else None
        f1=2*tp/(predictions+targets) if targets else None
        anchors=sum(row["anchor_geometry"]["anchor_count"] for row in rows)
        anchor_hits=sum(row["anchor_geometry"]["coverage_within_1m"]*row["anchor_geometry"]["anchor_count"] for row in rows)
        summary[condition]={"frames":len(rows),"s1_corr_r1":hits/valid if valid else None,
            "s1_corr_error_mean_m":error_sum/valid if valid else None,
            "geom_precision_0.2m":precision,"geom_recall_0.2m":recall,"geom_f1_0.2m":f1,
            "fixed_original_anchor_coverage_1m":anchor_hits/anchors if anchors else None}
    return summary


def _pct(value):return "n/a" if value is None else f"{value*100:.1f}%"


def _console(epoch,epochs,train,validation):
    print(f"Epoch {epoch:03d}/{epochs:03d} | train total {train['loss/total']:.4f} corr {train['loss/corr']:.4f} geom {train['loss/geom']:.4f} conf {train['loss/conf']:.4f}",flush=True)
    if not validation:return
    m=validation["metrics"];c=m["correspondence"];g=m["geometry"];q=m["confidence"]
    half=next(row for row in q["threshold_sweep"] if row["threshold"]==.5)
    median="n/a" if c["corr_error_median_m"] is None else f"{c['corr_error_median_m']:.3f} m"
    print(f"  val loss {validation['loss']['loss/total']:.4f} | corr R@1/5/10 {_pct(c['corr_r1'])}/{_pct(c['corr_r5'])}/{_pct(c['corr_r10'])} | median {median} | within .2 m {_pct(c['corr_within_0.2m'])}",flush=True)
    fine=m["scales"].get("s1",{})
    if fine:
        print(f"  S1 corr R@1 {_pct(fine['corr_r1'])} | within .2 m {_pct(fine['corr_within_0.2m'])} | no corr {fine['no_corr_query_count']}/{fine['query_count']}",flush=True)
    print(f"  geom .2 m P/R/F1 {_pct(g['0.2m']['precision'])}/{_pct(g['0.2m']['recall'])}/{_pct(g['0.2m']['f1'])} | .5 m F1 {_pct(g['0.5m']['f1'])}",flush=True)
    objects=m["instances"].get("s1",{}).get("all")
    if objects:
        print(f"  instance R@1 {_pct(objects['r1'])} | same-class instance R@1 {_pct(objects['same_class_r1'])} ({objects['same_class_count']} queries)",flush=True)
    print(f"  conf mean {q['mean'] if q['mean'] is not None else float('nan'):.3f} ECE {q['ece'] if q['ece'] is not None else float('nan'):.3f} | tau .5 coverage {_pct(half['coverage'])} precision {_pct(half['precision'])} | valid/no corr {c['valid_corr_query_count']}/{c['no_corr_query_count']}",flush=True)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--vod-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--radar-variant",default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--batch-size",type=int,default=1)
    parser.add_argument("--grad-accum-steps",type=int,default=1)
    parser.add_argument("--learning-rate",type=float,default=2e-4)
    parser.add_argument("--num-workers",type=int,default=0)
    parser.add_argument("--validate-every",type=int,default=1)
    parser.add_argument("--train-limit",type=int)
    parser.add_argument("--val-limit",type=int)
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--selection-metric",choices=("corr_r1","corr_within_0.2m","geom_f1_0.2m","geom_f1_0.5m"),default="corr_r1")
    parser.add_argument("--object-instances",action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument("--corruption-every",type=int,default=0,help="Run shifted/mismatched radar diagnostic every N validation epochs; 0 disables")
    parser.add_argument("--corruption-limit",type=int,default=8)
    parser.add_argument("--tensorboard",action="store_true",help="Write grouped TensorBoard scalars (requires tensorboard package)")
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()
    if min(args.epochs,args.batch_size,args.grad_accum_steps,args.validate_every)<1 or args.corruption_every<0 or args.corruption_limit<1:
        parser.error("epochs, batch size, accumulation and validation interval must be positive")
    config=config_from_dict(json.loads(args.config.read_text())) if args.config else Stage1Config()
    args.output_root.mkdir(parents=True,exist_ok=True)
    (args.output_root/"config.json").write_text(json.dumps(config.as_dict(),indent=2))
    writer=None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError("--tensorboard requires: python -m pip install tensorboard") from exc
        writer=SummaryWriter(log_dir=str(args.output_root/"tensorboard"))
    train=VoDStage1Dataset(args.vod_root,"train",radar_variant=args.radar_variant)
    val=VoDStage1Dataset(args.vod_root,"val",radar_variant=args.radar_variant)
    mismatch_dataset=VoDStage1Dataset(args.vod_root,"val",radar_variant=args.radar_variant) if args.corruption_every else None
    if args.train_limit: train.frames=train.frames[:args.train_limit]
    if args.val_limit: val.frames=val.frames[:args.val_limit]
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,collate_fn=collate_stage1,pin_memory=args.device.startswith("cuda"))
    val_loader=DataLoader(val,batch_size=1,shuffle=False,num_workers=args.num_workers,collate_fn=collate_stage1,pin_memory=args.device.startswith("cuda"))
    model=RadarLidarStage1(config).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    start_epoch,step=1,0
    best_scores={}
    if args.resume:
        checkpoint=torch.load(args.resume,map_location=args.device,weights_only=False)
        if config_from_dict(checkpoint["config"])!=config:
            raise ValueError("Resume config differs from current model")
        old_variant=checkpoint.get("data",{}).get("radar_variant")
        if old_variant and old_variant!=args.radar_variant:
            raise ValueError("Resume radar variant differs from original training data")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch=checkpoint["epoch"]+1
        step=checkpoint["step"]
        best_scores=checkpoint.get("best_scores",{})
        model.radar_only.confidence_trained=bool(checkpoint.get("confidence_trained",False))
    interactive_progress=sys.stderr.isatty()
    for epoch in range(start_epoch,args.epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals={};train_count=0
        diagnostic_totals={}
        scale_geometry={}
        train_progress=tqdm(total=len(train_loader),desc=f"Train {epoch:03d}/{args.epochs:03d}",unit="batch",
                            dynamic_ncols=True,mininterval=0.5,leave=False,disable=not interactive_progress)
        average_loss=float("nan")
        for batch_index,raw in enumerate(train_loader):
            batch=_move(raw,args.device)
            losses,diagnostics=model.forward_train(batch["radar"],batch["radar_valid"],batch["clean_lidar"],batch["clean_lidar_valid"],
                                                  defer_diagnostics=True,trusted_inputs=True)
            group_start=batch_index-(batch_index%args.grad_accum_steps)
            group_size=min(args.grad_accum_steps,len(train_loader)-group_start)
            (losses["loss/total"]/group_size).backward()
            batch_size=len(batch["frame_id"]);train_count+=batch_size
            for key,value in losses.items():
                _accumulate_scalar(totals,key,value,batch_size)
            for name,stats in diagnostics["levels"].items():
                source=diagnostics["radar"]["scales"][name]
                teacher=diagnostics["teacher"]["scales"][int(name[1:])-1]
                values={"radar_sites":stats["radar_sites"],"lidar_sites":stats["lidar_sites"],"candidate_coverage":stats["local_candidate_coverage"],"valid_corr_queries":stats["valid_corr_queries"],"no_valid_correspondence":stats["no_valid_correspondence"],"radar_isolated_fraction":source["isolated_fraction"],"lidar_isolated_fraction":teacher["isolated_fraction"],"radar_after_downsample":source["post_down"],"radar_after_growth":source["post_growth"]}
                if name=="s1" and "confidence_target_mean" in stats:
                    values.update({key:stats[key] for key in ("confidence_target_mean","confidence_target_nonzero_fraction","confidence_target_p90")})
                bucket=diagnostic_totals.setdefault(name,{})
                for key,value in values.items():
                    _accumulate_scalar(bucket,key,value)
                scale_geometry[name]={key:source[key] for key in ("shape_zyx","stride","channels","representation_channels","voxel_spacing_xyz_m","receptive_field_fine_voxels","receptive_field_xyz_m")}
            if (batch_index+1)%args.grad_accum_steps==0 or batch_index+1==len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                step+=1
            if (batch_index+1)%20==0 or batch_index+1==len(train_loader):
                average_loss=_python_scalar(totals["loss/total"])/train_count
                train_progress.set_postfix_str(f"avg loss={average_loss:.4f}",refresh=False)
            train_progress.update(1)
            if not interactive_progress and (batch_index+1)%100==0:
                print(f"epoch {epoch} batch {batch_index+1}/{len(train_loader)} loss {average_loss:.4f}",flush=True)
        train_progress.close()
        scheduler.step()
        model.radar_only.confidence_trained=bool(config.confidence_weight)
        averaged={key:_python_scalar(value)/max(train_count,1) for key,value in totals.items()}
        diagnostic_mean={name:{"geometry":scale_geometry[name],"mean":{key:_python_scalar(value)/max(len(train_loader),1) for key,value in stats.items()}} for name,stats in diagnostic_totals.items()}
        validation={}
        if epoch%args.validate_every==0 or epoch==args.epochs:
            model.eval();metric_acc=Stage1MetricAccumulator(config);val_totals={};val_count=0
            corruption=[]
            val_progress=tqdm(total=len(val_loader),desc=f"Val   {epoch:03d}/{args.epochs:03d}",unit="batch",
                              dynamic_ncols=True,mininterval=0.5,leave=False,disable=not interactive_progress)
            with torch.no_grad():
                for index,raw in enumerate(val_loader):
                    batch=_move(raw,args.device)
                    inputs=(batch["radar"],batch["radar_valid"],batch["clean_lidar"],batch["clean_lidar_valid"])
                    val_losses,val_diagnostics=model.forward_train(*inputs,return_intermediates=True,defer_diagnostics=True,
                                                                   trusted_inputs=True)
                    intermediates=val_diagnostics["intermediates"]
                    for key,value in val_losses.items():_accumulate_scalar(val_totals,key,value)
                    val_count+=1
                    metric_acc.update(model,*inputs,intermediates=intermediates)
                    if args.object_instances:
                        frame=val.frames[index]
                        label=resolve_vod_public_root(args.vod_root)/"lidar"/"training"/"label_2"/f"{frame.frame_id}.txt"
                        if label.is_file():
                            metric_acc.add_instances(evaluate_object_instances(model,*inputs,label,frame.lidar_calibration_path,
                                                                              intermediates=intermediates))
                    if args.corruption_every and epoch%args.corruption_every==0 and index<args.corruption_limit:
                        other_index=(index+1)%len(mismatch_dataset)
                        if mismatch_dataset.frames[other_index].frame_id==batch["frame_id"][0]:
                            raise ValueError("Misalignment diagnostic requires a different validation frame")
                        other=_move(collate_stage1([mismatch_dataset[other_index]]),args.device)
                        corruption.append({"frame_id":batch["frame_id"][0],"conditions":evaluate_corruptions(model,*inputs,
                            mismatched_radar=other["radar"],mismatched_valid=other["radar_valid"])})
                    if val_count%20==0 or val_count==len(val_loader):
                        val_progress.set_postfix_str(f"avg loss={_python_scalar(val_totals['loss/total'])/val_count:.4f}",refresh=False)
                    val_progress.update(1)
            val_progress.close()
            val_loss={key:_python_scalar(value)/max(val_count,1) for key,value in val_totals.items()}
            metrics=metric_acc.finish()
            validation={"loss":val_loss,"metrics":metrics,"flat":_flat_validation(val_loss,metrics),
                "corruption_summary":_corruption_summary(corruption) if corruption else {},"corruption_frames":corruption}
            (args.output_root/f"validation_epoch_{epoch:03d}.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
            _write_csv(args.output_root/f"confidence_bins_epoch_{epoch:03d}.csv",metrics["confidence"]["bins"])
            _write_csv(args.output_root/f"confidence_sweep_epoch_{epoch:03d}.csv",metrics["confidence"]["threshold_sweep"])
            _tensorboard(writer,epoch,averaged,validation["flat"],metrics)
            if writer is not None and "s1" in diagnostic_mean:
                for key in ("confidence_target_mean","confidence_target_nonzero_fraction","confidence_target_p90"):
                    value=diagnostic_mean["s1"]["mean"].get(key)
                    if value is not None:writer.add_scalar("Confidence/train_"+key,value,epoch)
        record={"epoch":epoch,"step":step,"train":averaged,"sparse_diagnostics":diagnostic_mean,"validation":validation}
        with (args.output_root/"metrics.jsonl").open("a",encoding="utf-8") as file:
            file.write(json.dumps(record)+"\n")
        provenance={"vod_root":str(args.vod_root.resolve()),"radar_variant":args.radar_variant,"train_frames":len(train),"val_frames":len(val)}
        if validation:
            for label,metric_name in (("corr","corr_r1"),("geom","geom_f1_0.2m"),("selected",args.selection_metric)):
                score=_selection_value(validation["metrics"],metric_name)
                if score is not None and score>best_scores.get(label,float("-inf")):
                    best_scores[label]=score
                    save_checkpoint(args.output_root/f"best_{label}.ckpt",model,optimizer,scheduler,epoch,step,validation,provenance,best_scores)
        save_checkpoint(args.output_root/"last.ckpt",model,optimizer,scheduler,epoch,step,validation,provenance,best_scores)
        export_radar_only(args.output_root/"radar_only.pth",model,provenance)
        _console(epoch,args.epochs,averaged,validation)
    if writer is not None:writer.close()


if __name__=="__main__":
    main()
