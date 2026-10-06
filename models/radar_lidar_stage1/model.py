"""Stage-1 radar-only sparse features and training-only LiDAR supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import Stage1Config
from .correspondence import LocalCorrespondence, local_neighbors
from .sparse import PointVoxelEncoder, SparseBackbone, SparseSites, receptive_fields


@dataclass
class Stage1Output:
    features: dict[str,SparseSites]
    confidence: SparseSites  # S1 sites; values are in [0,1]
    coordinates: dict[str,torch.Tensor]
    metadata: dict


class RadarOnlyEncoder(nn.Module):
    """The only computation required by the deployed model."""

    def __init__(self, config: Stage1Config):
        super().__init__()
        self.config=config
        self.point=PointVoxelEncoder(config.grid,"radar",config.channels[0])
        self.backbone=SparseBackbone(config.channels,config.growth_scales)
        self.projections=nn.ModuleList(nn.Linear(ch,ch) for ch in config.channels)
        self.confidence_head=nn.Linear(config.channels[0],1)
        self.confidence_calibrated=False
        self.confidence_trained=False

    def forward(self, radar_points: torch.Tensor, radar_valid: torch.Tensor, *, defer_diagnostics: bool = False,
                check_finite: bool = True) -> Stage1Output:
        sites,voxel_stats=self.point(radar_points,radar_valid,defer_diagnostics=defer_diagnostics,check_finite=check_finite)
        levels,operation_stats=self.backbone(sites,defer_diagnostics=defer_diagnostics)
        features={f"s{i+1}":level.replace_features(F.normalize(proj(level.features),dim=-1)) for i,(level,proj) in enumerate(zip(levels,self.projections))}
        fine=features["s1"]
        conf=fine.replace_features(torch.sigmoid(self.confidence_head(fine.features)))
        rf=receptive_fields(len(levels),self.config.growth_scales)
        scale_stats={}
        for i,level in enumerate(levels):
            scale_stats[f"s{i+1}"]={**operation_stats[i],"representation_channels":features[f"s{i+1}"].features.shape[-1],"shape_zyx":level.shape_zyx,"voxel_spacing_xyz_m":tuple(v*level.stride for v in self.config.grid.size_xyz),"receptive_field_fine_voxels":rf[i],"receptive_field_xyz_m":tuple(v*rf[i] for v in self.config.grid.size_xyz)}
        return Stage1Output(features,conf,{name:site.coords for name,site in features.items()}, {"radar_voxels":voxel_stats,"scales":scale_stats,"confidence_calibrated":self.confidence_calibrated,"confidence_trained":self.confidence_trained})


class LidarTeacher(nn.Module):
    def __init__(self,config: Stage1Config):
        super().__init__()
        self.point=PointVoxelEncoder(config.grid,"lidar",config.channels[0])
        # Supervision sites must descend from measured clean returns. Growing
        # teacher support would turn unmeasured context cells into positives.
        self.backbone=SparseBackbone(config.channels,())

    def forward(self,points: torch.Tensor,valid: torch.Tensor, *, defer_diagnostics: bool = False,
                check_finite: bool = True):
        sites,stats=self.point(points,valid,defer_diagnostics=defer_diagnostics,check_finite=check_finite)
        levels,ops=self.backbone(sites,defer_diagnostics=defer_diagnostics)
        return levels,{"voxel":stats,"scales":ops}


class GeometricProbe(nn.Module):
    """One linear prediction per radar site; incapable of standalone completion."""

    def __init__(self,config: Stage1Config):
        super().__init__()
        self.heads=nn.ModuleList(nn.Linear(ch,4) for ch in config.channels)

    def forward(self,output: Stage1Output) -> dict[str,torch.Tensor]:
        return {name:self.heads[i](output.features[name].features) for i,name in enumerate(output.features)}


def probe_target(radar: SparseSites, lidar: SparseSites, neighbors, grid, radius_m: float):
    """Nearest *observed* clean voxel in local radius; no target behind a return is invented."""
    center=radar.centers_xyz(grid)
    nearest,which=neighbors.distances_m.min(-1)
    hit=torch.isfinite(nearest) & (nearest <= radius_m)
    target=center.clone()
    if len(lidar.coords):
        idx=neighbors.indices.gather(1,which[:,None]).squeeze(1)[hit]
        target[hit]=lidar.centers_xyz(grid)[idx]
    return hit,target


class RadarLidarStage1(nn.Module):
    def __init__(self,config: Stage1Config = Stage1Config()):
        super().__init__()
        self.config=config
        self.radar_only=RadarOnlyEncoder(config)
        self.lidar_teacher=LidarTeacher(config)
        self.correspondence=nn.ModuleList(LocalCorrespondence(ch,ch,config.attention_dim) for ch in config.channels)
        self.probe=GeometricProbe(config)

    def forward_radar(self,radar_points: torch.Tensor,radar_valid: torch.Tensor,**kwargs) -> Stage1Output:
        if kwargs:
            raise TypeError("forward_radar accepts radar points and radar mask only")
        return self.radar_only(radar_points,radar_valid)

    def forward_train(self,radar_points: torch.Tensor,radar_valid: torch.Tensor,clean_lidar: torch.Tensor,clean_valid: torch.Tensor,
                      *, return_intermediates: bool = False, defer_diagnostics: bool = False,
                      trusted_inputs: bool = False) -> tuple[dict,dict]:
        output=self.radar_only(radar_points,radar_valid,defer_diagnostics=defer_diagnostics,check_finite=not trusted_inputs)
        lidar_levels,teacher_stats=self.lidar_teacher(clean_lidar,clean_valid,defer_diagnostics=defer_diagnostics,check_finite=not trusted_inputs)
        zero=self.radar_only.confidence_head.weight.sum()*0
        corr_loss,geom_loss,conf_loss=zero,zero,zero
        diagnostics={"radar":output.metadata,"teacher":teacher_stats,"levels":{}}
        intermediates={"output":output,"lidar_levels":lidar_levels,"neighbors":{},"attention":{}} if return_intermediates else None
        probe_predictions=self.probe(output) if self.config.geometric_weight or self.config.confidence_weight else {}
        losses={}
        for i,(name,li) in enumerate(zip(output.features,lidar_levels)):
            radar=output.features[name]
            radius=self.config.attention_radii_m[i]
            neighbors=local_neighbors(radar,li,self.config.grid,radius,self.config.max_neighbors)
            if intermediates is not None:
                intermediates["neighbors"][name]=neighbors
            if self.config.correspondence_weight and self.config.corr_scale_weights[i]:
                attention=self.correspondence[i](radar,li,self.config.grid,neighbors,radius,
                    self.config.positive_radii_m[i],self.config.temperature,
                    self.config.negative_strategy,self.config.num_negatives,
                    defer_diagnostics=defer_diagnostics)
                if intermediates is not None:
                    intermediates["attention"][name]=attention
                scale_corr=attention["loss"]
                corr_loss=corr_loss+self.config.corr_scale_weights[i]*scale_corr
                positives=attention["positive_queries"]
                contrastive=attention["contrastive_queries"]
            else:
                scale_corr=zero
                positives=((neighbors.distances_m<=self.config.positive_radii_m[i]) & neighbors.valid).any(-1).sum()
                if not defer_diagnostics:
                    positives=int(positives)
                contrastive=0
            losses[f"loss/corr_{name}"]=scale_corr
            coverage=neighbors.valid.any(-1).float().mean() if len(radar.coords) else zero.detach()
            if not defer_diagnostics:
                coverage=float(coverage)
            record={"radar_sites":len(radar.coords),"lidar_sites":len(li.coords),"local_candidate_coverage":coverage,"valid_corr_queries":positives,"no_valid_correspondence":len(radar.coords)-positives,"contrastive_queries":contrastive}
            if self.config.geometric_weight or self.config.confidence_weight:
                predicted=probe_predictions[name]
                center=radar.centers_xyz(self.config.grid)
                hit,target_xyz=probe_target(radar,li,neighbors,self.config.grid,radius)
                if len(predicted):
                    presence=F.binary_cross_entropy_with_logits(predicted[:,0],hit.float())
                    predicted_xyz=center+predicted[:,1:4]*radius
                    error=torch.linalg.vector_norm(predicted_xyz-target_xyz,dim=-1)
                    per_coordinate=F.smooth_l1_loss(predicted_xyz,target_xyz,beta=0.2,reduction="none")
                    local=(per_coordinate*hit[:,None]).sum()/(hit.sum().clamp_min(1)*3)
                    geom_loss=geom_loss+self.config.occupancy_weight*presence+self.config.local_geometry_weight*local
                    hit_rate=hit.float().mean().detach()
                    if defer_diagnostics:
                        localization=(error*hit).sum().detach()/hit.sum().clamp_min(1)
                    else:
                        localization=float(error[hit].mean().detach()) if bool(hit.any()) else None
                        hit_rate=float(hit_rate)
                    record.update({"probe_hit_rate":hit_rate,"probe_localization_m":localization})
                    if name=="s1" and self.config.confidence_weight:
                        # Quality is correctness of the *predicted* local point,
                        # not radar occupancy or clean-LiDAR presence alone.
                        predicted_hit=(torch.sigmoid(predicted[:,0])>=0.5)
                        quality=(hit & predicted_hit).float()*torch.exp(-error/self.config.confidence_sigma_m)
                        quality=quality.detach()
                        confidence=output.confidence.features.squeeze(-1)
                        conf_loss=F.binary_cross_entropy(confidence.clamp(1e-6,1-1e-6),quality)
                        quality_mean=quality.mean()
                        nonzero_fraction=(quality>0).float().mean()
                        quality_p90=torch.quantile(quality,0.9)
                        if not defer_diagnostics:
                            quality_mean,nonzero_fraction,quality_p90=float(quality_mean),float(nonzero_fraction),float(quality_p90)
                        record.update({"confidence_target_mean":quality_mean,"confidence_target_nonzero_fraction":nonzero_fraction,"confidence_target_p90":quality_p90})
            diagnostics["levels"][name]=record
        scales=len(output.features)
        losses.update({"loss/corr":corr_loss,"loss/geom":geom_loss/scales,"loss/conf":conf_loss})
        losses["loss/corr_weighted"]=self.config.correspondence_weight*losses["loss/corr"]
        losses["loss/geom_weighted"]=self.config.geometric_weight*losses["loss/geom"]
        losses["loss/conf_weighted"]=self.config.confidence_weight*losses["loss/conf"]
        losses["loss/total"]=losses["loss/corr_weighted"]+losses["loss/geom_weighted"]+losses["loss/conf_weighted"]
        if intermediates is not None:
            diagnostics["intermediates"]=intermediates
        return losses,diagnostics

    def radar_only_state_dict(self) -> dict:
        return {"config":self.config.as_dict(),"radar_only":self.radar_only.state_dict(),"confidence_calibrated":self.radar_only.confidence_calibrated,"confidence_trained":self.radar_only.confidence_trained}
