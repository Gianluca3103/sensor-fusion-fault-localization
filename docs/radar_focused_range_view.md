# Radar-focused VoD range-view experiment

This experiment keeps the existing range-view reconstructor. VoD 3D labels
identify *visible clean LiDAR first returns* from cars, pedestrians and
cyclists. The labels and clean scan are training targets only. A dilated
angular region around aligned radar returns limits training and generated
points to radar-supported directions. Empty pixels inside a projected 3D box
are **not** called positive returns. The box outlines in the preview are for
inspection, not masks used by the model.

The focused loss weights missing object returns (Car 2x, Pedestrian 4x,
Cyclist 4x), applies ADD/DELETE loss in the radar region, and adds a horizontal
scanline consistency term where adjacent clean ranges belong to the same
surface. The 3D nearest-radar gate is optional; leave it unset on the first
run so training and validation use the same angular region. The first-return
filter prevents an addition to an occupied virtual ray.

On the professor machine, from `sensor-fusion-fault-localization`:

```bash
BASE=/mnt/3D10B36523559581/Gianluca
REPO="$BASE/sensor-fusion-fault-localization"
PY="$BASE/svefusion-clean-cu118/bin/python"
CACHE="$BASE/sensor_fusion_outputs/vod_range5_full_cache"
GEOM="$BASE/sensor_fusion_outputs/vod_virtual_128x2048_geometry.json"
OBJECT_CACHE="$BASE/sensor_fusion_outputs/vod_virtual_128x2048_object_inputs"
RUN="$BASE/sensor_fusion_outputs/vod_radar_focused_range_view_80ep"
cd "$REPO"

"$PY" -u -m scripts.visualize_vod_range_boxes \
  --sample "$CACHE/samples/train/00544_fog_sim_s5.npz" \
  --radar-root "$CACHE/radar" --geometry "$GEOM" \
  --output "$BASE/sensor_fusion_outputs/radar_focused_boxes_00544.png"

"$PY" -u -m scripts.cache_range_view_inputs \
  --data-root "$CACHE/samples" --radar-root "$CACHE/radar" \
  --geometry "$GEOM" --output-root "$OBJECT_CACHE" \
  --require-lidar-intensity --object-targets --workers 4

"$PY" -u -m scripts.train_range_view_reconstruction \
  --data-root "$CACHE/samples" --radar-root "$CACHE/radar" \
  --geometry "$GEOM" --input-cache-root "$OBJECT_CACHE" \
  --output-root "$RUN" --epochs 80 --batch-size 8 --hidden-channels 32 \
  --predict-intensity --use-ray-encoding --radar-focused-objective \
  --val-limit 1296 --validate-every 80 --chamfer-every 80 \
  --num-workers 4 --device cuda
```

This starts a fresh model. The final checkpoint is
`$RUN/checkpoint_epoch_80.pt`. Inspect the preview and final detector AP
against the same clean/faulty matched validation IDs and official detector
checkpoint before deciding whether to redesign the network around sparse 3D
features. The range-view grid is a fitted virtual ray grid because the VoD
point files do not include physical LiDAR ring IDs.
