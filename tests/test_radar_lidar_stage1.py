"""Causal, physical and checkpoint tests for sparse radar-only Stage 1."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from Fault_Localization_Model.vod_dataset.vod_io import (
    align_radar_to_lidar, discover_vod_frames, load_vod_lidar,
    load_vod_radar, load_vod_radar_to_lidar,
)
from models.radar_lidar_stage1 import RadarLidarStage1, Stage1Config, VoxelGrid
from models.radar_lidar_stage1.correspondence import local_neighbors
from models.radar_lidar_stage1.evaluate import _inside_boxes, evaluate_corruptions, evaluate_object_instances
from models.radar_lidar_stage1.metrics import Stage1MetricAccumulator
from models.radar_lidar_stage1.sparse import (
    PointVoxelEncoder, SparseBackbone, SparseConv3d, SparseSites, axial_connectivity, decode_keys, encode_keys,
    receptive_fields, voxelize,
)
from models.radar_lidar_stage1.train import export_radar_only, load_radar_only, save_checkpoint


GRID=VoxelGrid((0.,0.,0.),(3.2,3.2,4.),(.2,.2,.25))


def config(**changes):
    return replace(Stage1Config(grid=GRID,channels=(8,16,32,64),attention_radii_m=(.5,1.,2.,4.),attention_dim=8,max_neighbors=6,growth_scales=(3,4)),**changes)


def inputs():
    radar=torch.tensor([[[.10,.10,.10,3.,2.,1.,0.],[.15,.15,.12,4.,1.,1.,-1.],[.60,.10,.10,1.,0.,0.,0.],[1.20,1.20,.35,5.,1.,0.,-2.]]])
    lidar=torch.tensor([[[.10,.10,.10,.3],[.22,.10,.10,.5],[.60,.10,.10,.7],[1.20,1.20,.35,.2]]])
    valid=torch.ones((1,4),dtype=torch.bool)
    return radar,lidar,valid


def test_deterministic_physical_voxels_and_modality_coordinates():
    r,l,v=inputs()
    rc,ri,_,rm=voxelize(r,v,GRID)
    lc,li,_,lm=voxelize(l,v,GRID)
    assert torch.equal(rc[0],torch.tensor([0,0,0,0]))
    assert ri[0]==ri[1]  # two radar returns in one fine voxel
    assert bool(rm.all() and lm.all())
    assert bool((encode_keys(rc,GRID.shape_zyx)[1:]>encode_keys(rc,GRID.shape_zyx)[:-1]).all())
    assert torch.equal(rc[0],lc[0])
    assert torch.allclose(GRID.centers_xyz(rc)[0],torch.tensor([.1,.1,.125]))
    assert ri.shape[0]==4 and li.shape[0]==4


def test_voxelization_still_rejects_nonfinite_external_inputs():
    radar,_,valid=inputs()
    radar[0,0,0]=float("nan")
    with pytest.raises(ValueError,match="finite"):
        voxelize(radar,valid,GRID)


def test_sparse_encoder_strides_channels_receptive_fields_and_gradients():
    r,_,v=inputs()
    point=PointVoxelEncoder(GRID,"radar",8)
    fine,stats=point(r,v)
    backbone=SparseBackbone((8,16,32,64),(3,4))
    levels,ops=backbone(fine)
    assert [x.stride for x in levels]==[1,2,4,8]
    assert [x.features.shape[-1] for x in levels]==[8,16,32,64]
    assert [x.shape_zyx for x in levels]==[GRID.scale_shape(s) for s in (1,2,4,8)]
    assert receptive_fields(4,(3,4))==(7,17,45,101)
    assert receptive_fields(4,())==(7,17,37,77)
    assert stats["active"]==len(fine.coords)
    assert ops[2]["post_growth"]>=ops[2]["post_down"]
    sum(x.features.sum() for x in levels).backward()
    assert point.point[0].weight.grad is not None
    assert backbone.stem.kernel.grad is not None


def test_sparse_sites_reject_invalid_or_duplicate_coordinates():
    with pytest.raises(ValueError):
        SparseSites(torch.tensor([[0,0,0,0],[0,0,0,0]]),torch.zeros(2,4),GRID.shape_zyx)
    with pytest.raises(ValueError):
        SparseSites(torch.tensor([[0,100,0,0]]),torch.zeros(1,4),GRID.shape_zyx)


def test_internal_sparse_sites_reuse_validated_coordinate_keys(monkeypatch):
    import models.radar_lidar_stage1.sparse as sparse
    coords=torch.tensor([[0,0,0,0],[0,0,0,1]])
    sites=SparseSites(coords,torch.randn(2,4),GRID.shape_zyx)
    expected=encode_keys(coords,GRID.shape_zyx)

    def unexpected_validation(*_args,**_kwargs):
        raise AssertionError("validated internal coordinates were checked again")

    monkeypatch.setattr(sparse,"encode_keys",unexpected_validation)
    replaced=sites.replace_features(sites.features+1)
    assert torch.equal(replaced.keys,expected)
    output=SparseConv3d(4,4)(replaced)
    assert torch.equal(output.keys,expected)


@pytest.mark.parametrize("stride,grow",[(1,False),(1,True),(2,False)])
@pytest.mark.parametrize("device",["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_neighbor_map_sparse_convolution_matches_reference_and_gradients(stride,grow,device):
    torch.manual_seed(19)
    # Two batches, boundaries, and disconnected cells exercise coordinate keys.
    coords=torch.tensor([[0,0,0,0],[0,0,0,1],[0,1,1,1],[0,4,4,5],
                         [1,0,0,0],[1,2,2,2],[1,15,15,15]],device=device)
    if stride==2:
        grow=False
    conv=SparseConv3d(5,7,stride=stride,grow=grow).to(device)
    fast_input=torch.randn(len(coords),5,device=device,requires_grad=True)
    ref_input=fast_input.detach().clone().requires_grad_()
    fast=conv(SparseSites(coords,fast_input,GRID.shape_zyx))
    fast.features.square().sum().backward()
    fast_input_grad=fast_input.grad.detach().clone()
    fast_kernel_grad=conv.kernel.grad.detach().clone()
    fast_bias_grad=conv.bias.grad.detach().clone()
    conv.zero_grad(set_to_none=True)
    ref=conv.forward_reference(SparseSites(coords,ref_input,GRID.shape_zyx))
    ref.features.square().sum().backward()
    assert torch.equal(fast.coords,ref.coords)
    assert torch.allclose(fast.features,ref.features,rtol=1e-5,atol=1e-6)
    assert torch.allclose(fast_input_grad,ref_input.grad,rtol=1e-5,atol=1e-5)
    assert torch.allclose(fast_kernel_grad,conv.kernel.grad,rtol=1e-5,atol=1e-5)
    assert torch.allclose(fast_bias_grad,conv.bias.grad,rtol=1e-5,atol=1e-5)


def test_submanifold_neighbor_map_is_reused_between_layers(monkeypatch):
    import models.radar_lidar_stage1.sparse as sparse
    builds=[]
    original=sparse._kernel_map
    def counted(*args,**kwargs):
        builds.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(sparse,"_kernel_map",counted)
    coords=torch.tensor([[0,0,0,0],[0,0,0,1],[0,0,1,0]])
    sites=SparseSites(coords,torch.randn(3,4),GRID.shape_zyx)
    first=SparseConv3d(4,4)(sites)
    second=SparseConv3d(4,4)(first.replace_features(first.features))
    assert len(builds)==1
    assert first._neighbor_cache is sites._neighbor_cache is second._neighbor_cache


def test_neighbor_map_chunk_boundary_and_empty_sites():
    shape=(64,64,64)
    coords=decode_keys(torch.arange(32775),shape)
    sites=SparseSites(coords,torch.randn(len(coords),2),shape)
    conv=SparseConv3d(2,2)
    assert torch.allclose(conv(sites).features,conv.forward_reference(sites).features,rtol=1e-5,atol=1e-6)
    empty=SparseSites(torch.empty((0,4),dtype=torch.long),torch.empty((0,2)),shape)
    for stride,grow in ((1,False),(1,True),(2,False)):
        result=SparseConv3d(2,3,stride=stride,grow=grow)(empty)
        assert result.coords.shape==(0,4)
        assert result.features.shape==(0,3)


def test_axial_connectivity_matches_explicit_neighbor_count():
    coords=torch.tensor([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,15,15,15],[1,0,0,0]])
    sites=SparseSites(coords,torch.ones(len(coords),2),GRID.shape_zyx)
    stats=axial_connectivity(sites)
    existing={tuple(row) for row in coords.tolist()}
    counts=[]
    for row in coords.tolist():
        counts.append(sum(tuple(row[:axis]+[row[axis]+sign]+row[axis+1:]) in existing
                          for axis in (1,2,3) for sign in (-1,1)))
    assert stats["mean_axial_neighbors"]==pytest.approx(sum(counts)/len(counts))
    assert stats["isolated_fraction"]==pytest.approx(sum(count==0 for count in counts)/len(counts))


def test_full_stage1_loss_and_gradients_match_reference_sparse_backend(monkeypatch):
    torch.manual_seed(23)
    fast=RadarLidarStage1(config())
    reference=RadarLidarStage1(config())
    reference.load_state_dict(fast.state_dict())
    radar,lidar,valid=inputs()
    fast_losses,_=fast.forward_train(radar,valid,lidar,valid)
    fast_losses["loss/total"].backward()
    with monkeypatch.context() as patch:
        patch.setattr(SparseConv3d,"forward",SparseConv3d.forward_reference)
        ref_losses,_=reference.forward_train(radar,valid,lidar,valid)
        ref_losses["loss/total"].backward()
    assert torch.allclose(fast_losses["loss/total"],ref_losses["loss/total"],rtol=1e-5,atol=1e-6)
    for (name,fast_param),(other_name,ref_param) in zip(fast.named_parameters(),reference.named_parameters()):
        assert name==other_name
        assert (fast_param.grad is None)==(ref_param.grad is None)
        if fast_param.grad is not None:
            assert torch.allclose(fast_param.grad,ref_param.grad,rtol=1e-4,atol=1e-5),name


def test_local_attention_radius_is_physical_and_batch_separated():
    coords=torch.tensor([[0,0,0,0],[0,0,0,2],[1,0,0,0]],dtype=torch.long)
    radar=SparseSites(coords,torch.ones(3,4),GRID.shape_zyx)
    lidar=SparseSites(coords.clone(),torch.ones(3,4),GRID.shape_zyx)
    near=local_neighbors(radar,lidar,GRID,.21,3)
    assert near.valid.sum(-1).tolist()==[1,1,1]
    assert near.indices[0,0]==0 and near.indices[2,0]==2
    far=local_neighbors(radar,lidar,GRID,.39,3)
    assert not bool((far.indices[0]==1).any())  # 0.4 m apart
    assert torch.isinf(far.distances_m[~far.valid]).all()


def test_radar_only_has_no_teacher_dependency_and_confidence_is_bounded():
    model=RadarLidarStage1(config()).eval()
    r,l,v=inputs()
    first=model.forward_radar(r,v)
    assert [site.features.shape[-1] for site in first.features.values()]==[8,16,32,64]
    model.lidar_teacher.forward=lambda *a,**kw: (_ for _ in ()).throw(AssertionError("teacher called"))
    model.correspondence[0].forward=lambda *a,**kw: (_ for _ in ()).throw(AssertionError("attention called"))
    second=model.forward_radar(r,v)
    for name in first.features:
        assert torch.equal(first.features[name].coords,second.features[name].coords)
        assert torch.allclose(first.features[name].features,second.features[name].features)
    assert bool(((first.confidence.features>=0)&(first.confidence.features<=1)).all())
    assert first.metadata["confidence_calibrated"] is False
    with pytest.raises(TypeError):
        model.forward_radar(r,v,clean_lidar=l)


def test_same_radar_different_clean_lidar_does_not_change_deployed_output():
    model=RadarLidarStage1(config()).eval()
    r,l,v=inputs()
    a=model.forward_radar(r,v)
    altered=l.clone(); altered[:,:,3]*=100
    # Training supervision sees both contradictory clean examples. No update is
    # applied, and the deployed path must remain a pure function of radar.
    model.forward_train(r,v,l,v)
    model.forward_train(r,v,altered,v)
    b=model.forward_radar(r,v)
    assert all(torch.equal(a.features[k].features,b.features[k].features) for k in a.features)
    assert torch.equal(a.confidence.features,b.confidence.features)
    assert not torch.equal(l,altered)


def test_train_loss_weights_independent_and_confidence_target_is_detached():
    r,l,v=inputs()
    model=RadarLidarStage1(config(correspondence_weight=0,geometric_weight=1,confidence_weight=0))
    losses,diagnostics=model.forward_train(r,v,l,v)
    assert torch.allclose(losses["loss/total"],losses["loss/geom"])
    assert diagnostics["levels"]["s1"]["radar_sites"]>0
    assert "loss/corr_s4" in losses
    model=RadarLidarStage1(config(correspondence_weight=0,geometric_weight=0,confidence_weight=1))
    losses,_=model.forward_train(r,v,l,v)
    assert torch.allclose(losses["loss/total"],losses["loss/conf"])
    losses["loss/total"].backward()
    assert model.radar_only.confidence_head.weight.grad is not None
    grad=model.probe.heads[0].weight.grad
    assert grad is None or torch.count_nonzero(grad)==0


def test_one_scale_probe_only_ablation_never_calls_attention():
    r,l,v=inputs()
    one=replace(config(),channels=(8,),attention_radii_m=(.5,),growth_scales=(),correspondence_weight=0.,geometric_weight=1.,confidence_weight=0.)
    model=RadarLidarStage1(one)
    model.correspondence[0].forward=lambda *a,**kw: (_ for _ in ()).throw(AssertionError("attention called"))
    losses,_=model.forward_train(r,v,l,v)
    assert torch.allclose(losses["loss/total"],losses["loss/geom"])
    assert list(model.forward_radar(r,v).features)==["s1"]


def test_physical_positive_mask_and_dataset_level_denominators():
    model=RadarLidarStage1(config()).eval()
    r,l,v=inputs()
    _,teacher_stats=model.lidar_teacher(l,v)
    assert teacher_stats["scales"][3]["post_growth"]==teacher_stats["scales"][3]["post_down"]
    acc=Stage1MetricAccumulator(model.config)
    acc.update(model,r,v,l,v)
    first=acc.finish()
    acc.update(model,r,v,l,v)
    doubled=acc.finish()
    for scale in first["scales"]:
        a,b=first["scales"][scale],doubled["scales"][scale]
        assert b["query_count"]==2*a["query_count"]
        assert b["valid_corr_query_count"]==2*a["valid_corr_query_count"]
        assert b["no_corr_query_count"]==2*a["no_corr_query_count"]
        assert b["corr_r1"]==a["corr_r1"]
    assert len(doubled["confidence"]["bins"])==10
    sweep=doubled["confidence"]["threshold_sweep"]
    assert len(sweep)==9
    assert all(a["coverage"]>=b["coverage"] for a,b in zip(sweep,sweep[1:]))
    assert first["correspondence"]["knn_r1"]==1.0


def test_validation_reuses_encoded_features_and_preserves_metrics(tmp_path,monkeypatch):
    import models.radar_lidar_stage1.evaluate as evaluation
    model=RadarLidarStage1(config()).eval()
    radar,lidar,valid=inputs()
    label=tmp_path/"labels.txt"
    label.write_text("Car 0 0 0 0 0 0 0 1 2 2 0.6 0.1 0.1 0\n")
    monkeypatch.setattr(evaluation,"load_vod_lidar_to_camera",lambda path:np.eye(4))
    baseline=Stage1MetricAccumulator(model.config)
    baseline.update(model,radar,valid,lidar,valid)
    baseline_report=baseline.finish()
    baseline_objects=evaluate_object_instances(model,radar,valid,lidar,valid,label,label)
    with torch.no_grad():
        _,diagnostics=model.forward_train(radar,valid,lidar,valid,return_intermediates=True)
    cached=diagnostics["intermediates"]
    monkeypatch.setattr(model,"forward_radar",lambda *a,**kw: (_ for _ in ()).throw(AssertionError("radar re-encoded")))
    monkeypatch.setattr(model.lidar_teacher,"forward",lambda *a,**kw: (_ for _ in ()).throw(AssertionError("LiDAR re-encoded")))
    for correspondence in model.correspondence:
        monkeypatch.setattr(correspondence,"forward",lambda *a,**kw: (_ for _ in ()).throw(AssertionError("attention recomputed")))
    reused=Stage1MetricAccumulator(model.config)
    reused.update(model,radar,valid,lidar,valid,intermediates=cached)
    assert reused.finish()==baseline_report
    assert evaluate_object_instances(model,radar,valid,lidar,valid,label,label,intermediates=cached)==baseline_objects


def test_deferred_diagnostics_preserve_training_losses_and_epoch_values():
    from models.radar_lidar_stage1.train import _accumulate_scalar, _python_scalar
    model=RadarLidarStage1(config()).eval()
    radar,lidar,valid=inputs()
    direct,direct_stats=model.forward_train(radar,valid,lidar,valid)
    deferred,stats=model.forward_train(radar,valid,lidar,valid,defer_diagnostics=True,trusted_inputs=True)
    for key in direct:
        assert torch.allclose(direct[key],deferred[key],rtol=1e-5,atol=1e-6),key
    assert isinstance(stats["levels"]["s1"]["local_candidate_coverage"],torch.Tensor)
    assert _python_scalar(stats["levels"]["s1"]["local_candidate_coverage"])==pytest.approx(
        direct_stats["levels"]["s1"]["local_candidate_coverage"])
    bucket={}
    for _ in range(3):
        _accumulate_scalar(bucket,"loss",deferred["loss/total"],weight=2)
    assert _python_scalar(bucket["loss"])/6==pytest.approx(float(direct["loss/total"].detach()))


def test_no_local_positive_has_finite_zero_correspondence_gradient():
    model=RadarLidarStage1(config())
    radar,lidar,valid=inputs()
    lidar[:,:,:3]=torch.tensor([2.9,2.9,3.9])
    losses,_=model.forward_train(radar,valid,lidar,valid,defer_diagnostics=True)
    assert torch.isfinite(losses["loss/total"])
    assert float(losses["loss/corr"].detach())==pytest.approx(0.0,abs=1e-7)
    losses["loss/total"].backward()
    assert model.radar_only.backbone.stem.kernel.grad is not None
    assert torch.isfinite(model.radar_only.backbone.stem.kernel.grad).all()


def test_no_valid_correspondence_is_not_a_retrieval_failure():
    model=RadarLidarStage1(config()).eval()
    r,l,v=inputs()
    far=l.clone();far[:,:,:3]=torch.tensor([2.9,2.9,3.9])
    acc=Stage1MetricAccumulator(model.config)
    acc.update(model,r,v,far,v)
    row=acc.finish()["scales"]["s1"]
    assert row["valid_corr_query_count"]==0
    assert row["no_corr_query_count"]==row["query_count"]
    assert row["corr_r1"] is None
    mixed=Stage1MetricAccumulator(model.config)
    mixed.update(model,r,v,l,v)
    valid_row=mixed.finish()["scales"]["s1"]
    mixed.update(model,r,v,far,v)
    mixed_row=mixed.finish()["scales"]["s1"]
    assert mixed_row["corr_r1"]==valid_row["corr_r1"]
    assert mixed_row["no_corr_query_count"]>valid_row["no_corr_query_count"]


def test_corruption_pipeline_and_checkpoint_roundtrip(tmp_path):
    model=RadarLidarStage1(config()).eval()
    r,l,v=inputs()
    scores=evaluate_corruptions(model,r,v,l,v,mismatched_radar=r.flip(1),mismatched_valid=v,shifts_m=(.5,1,2,5))
    assert set(scores)=={"aligned","mismatched_scene","x_shift_0.5m","x_shift_1m","x_shift_2m","x_shift_5m"}
    assert scores["aligned"]["anchor_geometry"]["anchor_count"]>0
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=1)
    path=tmp_path/"full.pth"
    save_checkpoint(path,model,opt,scheduler,1,0,{"s1":{}})
    full=torch.load(path,weights_only=False)
    assert full["epoch"]==1 and "optimizer" in full and "lidar_teacher.point.point.0.weight" in full["model"]
    radar_only_path=tmp_path/"radar.pth"
    export_radar_only(radar_only_path,model)
    raw=torch.load(radar_only_path,weights_only=False)
    assert all("lidar" not in key and "correspondence" not in key and "probe" not in key for key in raw["radar_only"])
    deployed=load_radar_only(radar_only_path)
    original=model.forward_radar(r,v)
    restored=deployed(r,v)
    assert all(torch.allclose(original.features[k].features,restored.features[k].features) for k in original.features)


def test_camera_box_membership_for_object_analysis_only():
    # Bottom-centered camera box at (1,1,1), dimensions h=2,w=2,l=2.
    line="Car 0 0 0 0 0 0 0 2 2 2 1 1 1 0"
    ids,classes=_inside_boxes(np.array([[1.,0.,1.],[1.,3.,1.]]),[line],np.eye(4))
    assert ids.tolist()==[0,-1] and classes==["Car"]


def test_real_vod_calibration_alignment_when_dataset_present():
    root=Path(r"C:\Users\gianl\Desktop\Thesis\View-Of-Delft dataset\view_of_delft_PUBLIC")
    if not root.exists():
        pytest.skip("Local VoD data unavailable")
    frame=discover_vod_frames(root,"train",frame_ids=["00644"],radar_variant="radar_20frames_verified_doppler_radial")[0]
    radar=align_radar_to_lidar(load_vod_radar(frame.radar_path),load_vod_radar_to_lidar(frame.lidar_calibration_path,frame.radar_calibration_path))
    lidar=load_vod_lidar(frame.lidar_path)
    assert radar.shape[1]==7 and lidar.shape[1]==4
    from scipy.spatial import cKDTree
    distances=cKDTree(lidar[:,:3]).query(radar[:,:3])[0]
    assert np.median(distances)<1.0
