# VoD PV-RCNN detector benchmark

This is a 3D object-detection benchmark, separate from the range-view
reconstruction model. Train **PV-RCNN on clean LiDAR only**, then run the same
checkpoint on the same official VoD validation IDs with clean, faulty and
reconstructed clouds. The optional `lidar_radar` mode repeats that protocol
with a separate clean-LiDAR-plus-radar trained detector. Never compare a
LiDAR-only checkpoint against radar-concatenated inputs without retraining.

The detector uses the [PV-RCNN implementation in OpenPCDet](https://github.com/open-mmlab/OpenPCDet),
not the README-only [PV-RCNN paper repository](https://github.com/sshaoshuai/PV-RCNN).
Install OpenPCDet in a CUDA-capable Linux Python environment according to its
[installation guide](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md)
before training. Installation and GPU training are **not** performed by these
repository scripts. The export environment needs NumPy, PyTorch and Pillow;
configuration generation also needs PyYAML. The info/evaluation environment
needs an importable `pcdet` package.

## Data policy

- Official VoD `train` and `val` frame IDs are checked; no frame crosses splits.
- All three validation conditions use the **same IDs, KITTI labels and camera
  calibration**. Only the point cloud changes.
- The detector reads XYZ plus one scalar. LiDAR uses reflectivity. Optional
  five-frame aligned radar contributes XYZ plus RCS in that scalar slot. This
  is simple early fusion, not a learned radar-specific branch, and it must be
  trained as a separate detector.
- The checkpoint's forward-only rule is applied to every condition. PV-RCNN
  also uses its standard `[0,-40,-3,70.4,40,1]` point-cloud range; points
  outside that detector range are excluded uniformly.
- Generated reconstruction points have scalar feature 0; original points keep
  their reflectivity. Conservative merge must preserve originals.
- A completely empty detector input (for example LiDAR-only `total_loss`) gets
  one near-origin `[0.01,0,-2.9,0]` sentinel point. The source fault remains
  empty; the sentinel exists only to keep OpenPCDet's voxel/keypoint path
  defined. Any resulting detections are still evaluated as usual.
- `bicycle` labels are mapped to KITTI `Cyclist`; `Car`, `Pedestrian`, and
  `Cyclist` are preserved. Other VoD labels become `DontCare`, placed last as
  OpenPCDet requires. The optional VoD tracking ID is discarded.
- OpenPCDet requires a `.png` filename to read image size. The exporter writes
  tiny blank PNGs with the same dimensions as VoD's JPGs. The detector config
  requests points only; camera pixels are never detector inputs.
- The `train` cloud in every export condition is clean. `faulty` and
  `reconstructed` modifications occur **only in validation**; therefore only
  the `clean` config should be used for training.

## Prepare on the Ubuntu GPU machine

The following uses the existing VoD cache paths. Set `RECON` to a range-view
checkpoint from the run you intend to assess. `PCDET` is a **separate cloned
and installed** OpenPCDet checkout. `PYTHON` must run both the reconstruction
and OpenPCDet packages; set `PYTHONPATH` as shown if they live in separate
checkouts. Use a fresh `EXPORT` directory. For a smoke test, append
`--limit 10` to the exporter and do not report its AP as a full-set result.

```bash
REPO=/mnt/3D10B36523559581/Gianluca/sensor-fusion-fault-localization
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
CACHE=/mnt/3D10B36523559581/Gianluca/sensor_fusion_outputs/vod_range5_full_cache
PCDET=/mnt/3D10B36523559581/Gianluca/OpenPCDet
RECON="$CACHE/training_e100_b24_h32_val5_cd10_gHPyX6/last_checkpoint.pt"
EXPORT=/mnt/3D10B36523559581/Gianluca/sensor_fusion_outputs/vod_pvrcnn_benchmark
PYTHON=/mnt/3D10B36523559581/Gianluca/Sensor-Fusion/.venv_model_v2/bin/python
export PYTHONPATH="$REPO:$PCDET${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO"
"$PYTHON" -m scripts.export_vod_pvrcnn \
  --vod-root "$VOD" --samples-root "$CACHE/samples" --radar-root "$CACHE/radar" \
  --checkpoint "$RECON" --output-root "$EXPORT" --device cpu --with-radar
"$PYTHON" -m scripts.configure_vod_pvrcnn \
  --openpcdet-root "$PCDET" --export-root "$EXPORT" --with-radar
"$PYTHON" -m scripts.prepare_vod_pvrcnn_infos \
  --openpcdet-root "$PCDET" --export-root "$EXPORT" --with-radar --workers 4
```

The export can take considerable time: it reconstructs every validation
sample. The clean train points are exported without running reconstruction.
Do not run the exporter while GPU training if its `--device` is `cuda`.
If the export is interrupted, rerunning accepts identical point files and
refuses to overwrite different ones.

## Train and compare

These commands assume OpenPCDet is installed in `PYTHON` and that the export
and info generation above completed. Start with batch size 1; PV-RCNN is much
heavier than the range-view model. A larger batch size is only appropriate
after confirming GPU memory headroom. Choose your own epoch count; 80 is the
upstream PV-RCNN default.

```bash
cd "$PCDET/tools"
"$PYTHON" train.py --cfg_file cfgs/kitti_models/vod_pvrcnn_lidar_clean.yaml \
  --batch_size 1 --workers 2 --extra_tag vod_clean_lidar
```

Find the clean detector checkpoint under
`$PCDET/output/kitti_models/vod_pvrcnn_lidar_clean/vod_clean_lidar/ckpt/`.
The following invokes OpenPCDet's KITTI 3D AP evaluation three times with the
same checkpoint; its log and `result.pkl` remain in the corresponding
OpenPCDet `output/.../eval/` directory, and `--save_to_file` writes per-frame
KITTI-format predictions.

```bash
DETECTOR_CKPT="$PCDET/output/kitti_models/vod_pvrcnn_lidar_clean/vod_clean_lidar/ckpt/checkpoint_epoch_80.pth"
bash "$REPO/scripts/eval_vod_pvrcnn.sh" "$PCDET" "$EXPORT" lidar "$DETECTOR_CKPT" "$PYTHON"
```

For a compact frame-checked comparison of KITTI 3D AP_R40, pass the three
`result.pkl` files to the summarizer. For the example epoch-80 checkpoint:

```bash
BASE="$PCDET/output/kitti_models"
"$PYTHON" "$REPO/scripts/summarize_vod_pvrcnn.py" \
  --clean-infos "$EXPORT/lidar/clean/kitti_infos_val.pkl" \
  --clean-results "$BASE/vod_pvrcnn_lidar_clean/vod_clean_trained/eval/epoch_80/val/clean/result.pkl" \
  --faulty-results "$BASE/vod_pvrcnn_lidar_faulty/vod_clean_trained/eval/epoch_80/val/faulty/result.pkl" \
  --reconstructed-results "$BASE/vod_pvrcnn_lidar_reconstructed/vod_clean_trained/eval/epoch_80/val/reconstructed/result.pkl" \
  --output "$EXPORT/lidar_3d_ap_r40.csv"
```

For the optional radar ablation, train a **new** checkpoint with
`cfgs/kitti_models/vod_pvrcnn_lidar_radar_clean.yaml`, then call the same
evaluation script with `lidar_radar` and that new checkpoint. Compare each
detector's clean/faulty/reconstructed AP within its own modality, and compare
LiDAR-only versus radar using the same validation IDs and IoU definition.

The official VoD `test` split is intentionally not evaluated here: labels may
not be available on the machine, and selecting a reconstruction checkpoint
or detector hyperparameters by test performance would leak evaluation data.
