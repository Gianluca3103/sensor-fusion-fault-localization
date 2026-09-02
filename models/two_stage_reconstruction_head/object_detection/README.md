# Frozen BEV detector evaluation

This package measures whether reconstruction restores downstream object
detections. The detector is a lightweight CenterNet-style model with rotated
metric boxes. It is trained once on clean View-of-Delft training BEVs, selected
using only clean validation BEVs, and then frozen.

The default classes are the three official VoD benchmark labels: `Car`,
`Pedestrian`, and `Cyclist`. Lower-case auxiliary labels such as `bicycle_rack`
are intentionally ignored unless an explicit alias policy is added.

## Train the detector

```bash
python -u -m models.two_stage_reconstruction_head.object_detection.train_bev_detector \
  --data-root /workspace/reconstruction_vod_radar3_unique_multibox_selector \
  --vod-root /workspace/view_of_delft_PUBLIC \
  --output-root /workspace/vod_clean_bev_detector \
  --device cuda --epochs 50 --batch-size 16 --num-workers 8
```

## Evaluate reconstruction conditions

```bash
python -u -m models.two_stage_reconstruction_head.object_detection.evaluate_reconstruction_detection \
  --detector-checkpoint /workspace/vod_clean_bev_detector/best_model.pt \
  --coarse-checkpoint /workspace/COARSE_RUN/best_model.pt \
  --fine-checkpoint /workspace/FINE_RUN/best_model.pt \
  --selector-config configs/coarse_reconstruction_vod_pointpillars_hrnet_halo_recall_weighted_unlimited_pillars_dropout010.json \
  --data-root /workspace/reconstruction_vod_radar3_unique_multibox_selector \
  --radar-root /workspace/radar20_pointpillars_cache \
  --vod-root /workspace/view_of_delft_PUBLIC \
  --output-root /workspace/vod_object_detection_evaluation \
  --split val --device cuda --batch-size 4 --num-workers 8 \
  --visualize-samples 50
```

Outputs include `summary.json`, `frame_metrics.csv/json`,
`predictions.csv/json`, `object_recovery.csv/json`, and side-by-side images.

The public VoD release contains GT labels for the 5,139 train and 1,296
validation frames, but not the 2,247 official test frames. Consequently,
`--split test` intentionally fails unless `--label-root` points to a separate,
authorized test annotation set. Missing test labels are never interpreted as
empty scenes.

## Direct fault-robust LiDAR/radar fusion detector

`PointPillarsHRNetFusionDetector` is a separate experiment that does not
reconstruct LiDAR. It independently pillarizes the faulty LiDAR and accumulated
radar point clouds on the dataset's shared BEV grid, concatenates the two dense
feature maps, fuses them with HRNet, and predicts object centres and rotated 3D
boxes with an anchor-free center head. It does not load a fault selector, repair
mask, coarse model, fine model, pseudo-LiDAR, or clean LiDAR as an input.

```bash
python -u -m models.two_stage_reconstruction_head.object_detection.train_fusion_detector \
  --data-root /workspace/reconstruction_vod_radar3_unique_multibox_selector \
  --radar-root /workspace/radar20_pointpillars_cache \
  --vod-root /workspace/view_of_delft_PUBLIC \
  --output-root /workspace/fault_robust_pointpillars_hrnet_center \
  --config configs/fault_robust_pointpillars_hrnet_center.json \
  --device cuda --epochs 80 --batch-size 8 \
  --validation-batch-size 8 --num-workers 8
```

Evaluate the frozen best-validation-mAP checkpoint and optionally compare it
with the same detector after zeroing its radar feature map:

```bash
python -u -m models.two_stage_reconstruction_head.object_detection.evaluate_fusion_detector \
  --checkpoint /workspace/fault_robust_pointpillars_hrnet_center/best_model.pt \
  --data-root /workspace/reconstruction_vod_radar3_unique_multibox_selector \
  --radar-root /workspace/radar20_pointpillars_cache \
  --vod-root /workspace/view_of_delft_PUBLIC \
  --output-root /workspace/fault_robust_pointpillars_hrnet_center_val \
  --split val --device cuda --batch-size 8 --num-workers 8 \
  --include-lidar-only-ablation
```
