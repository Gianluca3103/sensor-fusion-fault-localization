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
