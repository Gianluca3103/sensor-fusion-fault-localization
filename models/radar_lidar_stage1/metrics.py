"""Dataset-level, physically defined Stage-1 validation statistics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .correspondence import local_neighbors
from .model import probe_target


DISTANCES = (0.1, 0.2, 0.5, 1.0)
GEOM_DISTANCES = (0.1, 0.2, 0.5)
THRESHOLDS = tuple(round(i / 10, 1) for i in range(1, 10))


def _fraction(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def _prf(tp: int, predictions: int, targets: int) -> dict:
    precision=_fraction(tp,predictions)
    recall=_fraction(tp,targets)
    if targets:
        f1=2*tp/(predictions+targets)
    else:
        f1=None
    return {"precision":precision,"recall":recall,"f1":f1,"tp":tp,"predictions":predictions,"targets":targets}


class Stage1MetricAccumulator:
    """Pool integer counts and physical distances before computing rates/quantiles."""

    def __init__(self, config):
        self.config=config
        self.scales={f"s{i+1}":{"query_count":0,"valid_corr_query_count":0,"corr_hits":defaultdict(int),"knn_hits":0,"corr_errors":[]} for i in range(len(config.channels))}
        self.geometry={t:{"tp":0,"predictions":0,"targets":0} for t in GEOM_DISTANCES}
        self.conf_count=np.zeros(10,dtype=np.int64)
        self.conf_sum=np.zeros(10,dtype=np.float64)
        self.conf_quality=np.zeros(10,dtype=np.float64)
        self.conf_success=np.zeros(10,dtype=np.int64)
        self.conf_total=0
        self.conf_all_sum=0.0
        self.sweep={t:{"sites":0,"targets_retained":0,"tp":0,"predictions":0} for t in THRESHOLDS}
        self.geom_target_total=0
        self.instance=defaultdict(lambda:defaultdict(lambda:defaultdict(int)))

    @torch.no_grad()
    def update(self,model,radar,radar_valid,clean,clean_valid,*,intermediates=None):
        output=(intermediates["output"] if intermediates is not None
                else model.forward_radar(radar,radar_valid))
        lidar_levels=(intermediates["lidar_levels"] if intermediates is not None
                      else model.lidar_teacher(clean,clean_valid)[0])
        probe=model.probe(output)
        for i,(name,li) in enumerate(zip(output.features,lidar_levels)):
            r=output.features[name]
            bucket=self.scales[name]
            count=len(r.coords)
            bucket["query_count"]+=count
            neighbors=(intermediates["neighbors"][name] if intermediates is not None
                       else local_neighbors(r,li,self.config.grid,self.config.attention_radii_m[i],self.config.max_neighbors))
            if count and len(li.coords):
                positive=neighbors.valid & (neighbors.distances_m<=self.config.positive_radii_m[i])
                valid=positive.any(-1)
                bucket["valid_corr_query_count"]+=int(valid.sum())
                if bool(valid.any()):
                    closest=neighbors.distances_m[valid].argmin(-1)
                    bucket["knn_hits"]+=int(positive[valid].gather(1,closest[:,None]).sum())
                    result=(intermediates["attention"].get(name) if intermediates is not None else None)
                    if result is None:
                        result=model.correspondence[i](r,li,self.config.grid,neighbors,
                            self.config.attention_radii_m[i],self.config.positive_radii_m[i],
                            self.config.temperature,self.config.negative_strategy,self.config.num_negatives)
                    ranking=result["logits"].argsort(-1,descending=True)[valid]
                    ranked_positive=positive[valid].gather(1,ranking)
                    for k in (1,5,10):
                        bucket["corr_hits"][k]+=int(ranked_positive[:,:k].any(-1).sum())
                    top_distance=neighbors.distances_m[valid].gather(1,ranking[:,:1]).squeeze(1)
                    bucket["corr_errors"].append(top_distance.cpu().numpy())
            if name!="s1" or not count:
                continue
            predicted=probe[name]
            center=r.centers_xyz(self.config.grid)
            hit,target=probe_target(r,li,neighbors,self.config.grid,self.config.attention_radii_m[i])
            predicted_xyz=center+predicted[:,1:4]*self.config.attention_radii_m[i]
            error=torch.linalg.vector_norm(predicted_xyz-target,dim=-1)
            emitted=torch.sigmoid(predicted[:,0])>=0.5
            confidence=output.confidence.features.squeeze(1)
            quality=(hit & emitted).float()*torch.exp(-error/self.config.confidence_sigma_m)
            success=hit & emitted & (error<=self.config.geometry_eval_tolerance_m)
            self.geom_target_total+=int(hit.sum())
            for tolerance in GEOM_DISTANCES:
                row=self.geometry[tolerance]
                row["tp"]+=int((hit & emitted & (error<=tolerance)).sum())
                row["predictions"]+=int(emitted.sum())
                row["targets"]+=int(hit.sum())
            c=confidence.cpu().numpy(); q=quality.cpu().numpy(); s=success.cpu().numpy()
            h=hit.cpu().numpy(); p=emitted.cpu().numpy(); e=error.cpu().numpy()
            bins=np.clip((c*10).astype(np.int64),0,9)
            self.conf_count+=np.bincount(bins,minlength=10)
            self.conf_sum+=np.bincount(bins,weights=c,minlength=10)
            self.conf_quality+=np.bincount(bins,weights=q,minlength=10)
            self.conf_success+=np.bincount(bins,weights=s.astype(np.float64),minlength=10).astype(np.int64)
            self.conf_total+=len(c)
            self.conf_all_sum+=float(c.sum())
            for threshold in THRESHOLDS:
                retained=c>threshold
                row=self.sweep[threshold]
                row["sites"]+=int(retained.sum())
                row["targets_retained"]+=int((retained & h).sum())
                selected=retained & p
                row["predictions"]+=int(selected.sum())
                row["tp"]+=int((selected & h & (e<=self.config.geometry_eval_tolerance_m)).sum())

    def add_instances(self,report: dict):
        for scale,classes in report.items():
            for cls,values in classes.items():
                if not isinstance(values,dict):
                    continue
                for key,value in values.items():
                    if key in ("count","hit1","hit3","hit5","same_class_count","same_class_hit1","same_class_hit3","same_class_hit5"):
                        self.instance[scale][cls][key]+=int(value)

    def finish(self) -> dict:
        report={"scales":{},"geometry":{},"confidence":{},"instances":{}}
        total_valid=0; total_queries=0; total_knn=0; total_hits={k:0 for k in (1,5,10)}; arrays=[]
        for name,row in self.scales.items():
            valid=row["valid_corr_query_count"]; query=row["query_count"]
            errors=np.concatenate(row["corr_errors"]) if row["corr_errors"] else np.empty(0)
            scale={"query_count":query,"valid_corr_query_count":valid,"no_corr_query_count":query-valid,
                "no_corr_fraction":_fraction(query-valid,query),
                "knn_r1":_fraction(row["knn_hits"],valid),
                **{f"corr_r{k}":_fraction(row["corr_hits"][k],valid) for k in (1,5,10)},
                "corr_error_mean_m":float(errors.mean()) if len(errors) else None,
                "corr_error_median_m":float(np.median(errors)) if len(errors) else None,
                "corr_error_p90_m":float(np.percentile(errors,90)) if len(errors) else None,
                **{f"corr_within_{t:.1f}m":float(np.mean(errors<=t)) if len(errors) else None for t in DISTANCES}}
            report["scales"][name]=scale
            total_valid+=valid;total_queries+=query
            total_knn+=row["knn_hits"]
            for k in total_hits:total_hits[k]+=row["corr_hits"][k]
            if len(errors):arrays.append(errors)
        errors=np.concatenate(arrays) if arrays else np.empty(0)
        report["correspondence"]={"query_count":total_queries,"valid_corr_query_count":total_valid,
            "no_corr_query_count":total_queries-total_valid,"no_corr_fraction":_fraction(total_queries-total_valid,total_queries),
            "knn_r1":_fraction(total_knn,total_valid),
            **{f"corr_r{k}":_fraction(total_hits[k],total_valid) for k in (1,5,10)},
            "corr_error_mean_m":float(errors.mean()) if len(errors) else None,
            "corr_error_median_m":float(np.median(errors)) if len(errors) else None,
            "corr_error_p90_m":float(np.percentile(errors,90)) if len(errors) else None,
            **{f"corr_within_{t:.1f}m":float(np.mean(errors<=t)) if len(errors) else None for t in DISTANCES}}
        for t,row in self.geometry.items():report["geometry"][f"{t:.1f}m"]=_prf(**row)
        bins=[];ece=0.0;binary_ece=0.0
        for index,count in enumerate(self.conf_count):
            mean_conf=float(self.conf_sum[index]/count) if count else None
            mean_quality=float(self.conf_quality[index]/count) if count else None
            actual=float(self.conf_success[index]/count) if count else None
            fraction=float(count/self.conf_total) if self.conf_total else 0.0
            bins.append({"lower":index/10,"upper":(index+1)/10,"count":int(count),"fraction":fraction,
                "mean_confidence":mean_conf,"actual_quality":mean_quality,"actual_geom_success":actual})
            if count:
                ece+=fraction*abs(mean_conf-mean_quality)
                binary_ece+=fraction*abs(mean_conf-actual)
        sweep=[]
        for threshold,row in self.sweep.items():
            rates=_prf(row["tp"],row["predictions"],self.geom_target_total)
            sweep.append({"threshold":threshold,"coverage":_fraction(row["sites"],self.conf_total),
                "target_coverage":_fraction(row["targets_retained"],self.geom_target_total),**rates})
        report["confidence"]={"mean":_fraction(self.conf_all_sum,self.conf_total),"ece":ece if self.conf_total else None,
            "binary_success_ece":binary_ece if self.conf_total else None,"bins":bins,"threshold_sweep":sweep,
            "quality_definition":"emitted clean-nearby probe surface * exp(-localization_error_m/sigma_m); no target and abstentions have zero quality",
            "success_tolerance_m":self.config.geometry_eval_tolerance_m}
        for scale,classes in self.instance.items():
            report["instances"][scale]={}
            all_values=defaultdict(int)
            for cls,values in classes.items():
                report["instances"][scale][cls]={**values,**{f"r{k}":_fraction(values[f"hit{k}"],values["count"]) for k in (1,3,5)},
                    **{f"same_class_r{k}":_fraction(values[f"same_class_hit{k}"],values["same_class_count"]) for k in (1,3,5)}}
                for key,value in values.items():all_values[key]+=value
            report["instances"][scale]["all"]={**all_values,
                **{f"r{k}":_fraction(all_values[f"hit{k}"],all_values["count"]) for k in (1,3,5)},
                **{f"same_class_r{k}":_fraction(all_values[f"same_class_hit{k}"],all_values["same_class_count"]) for k in (1,3,5)}}
        return report
