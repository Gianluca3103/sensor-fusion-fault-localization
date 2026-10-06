# Radar → LiDAR Stage 1

This folder learns a **radar-only, sparse 3D geometric representation** with a synchronized clean-LiDAR teacher. It contains no reconstruction, diffusion, BEV, detector or faulty-LiDAR input. The repository/data audit and assumptions are in [`docs/radar_lidar_stage1_audit.md`](../../docs/radar_lidar_stage1_audit.md).

## Interface

`RadarLidarStage1.forward_radar(radar, radar_valid)` returns sparse S1–S4 radar features, sparse S1 confidence values, coordinates and physical geometry metadata. `RadarOnlyEncoder`, exported separately, contains no teacher, attention or probe weights. Confidence is supervised during training, but it is **not automatically calibrated**; inspect held-out reliability bins before choosing a threshold.

`RadarLidarStage1.forward_train(radar, radar_valid, clean_lidar, clean_valid)` uses clean LiDAR only to train local feature correspondence and a linear geometric probe. The deployed output is computed before the teacher is called. A test explicitly checks that different clean scans leave radar-only output identical.

## Representation and losses

- Canonical coordinates are current-frame LiDAR XYZ. Radar data are transformed with the per-frame VoD radar/camera/LiDAR calibration in `vod_io.py`. Raw radar attributes are RCS, raw radial velocity, compensated radial velocity and ordinal scan age. LiDAR has reflectivity. Each modality has its own point MLP and learned voxel mean/max/count aggregation.
- Sparse sites have sorted unique `[batch,z,y,x]` integer coordinates and `[N,C]` features. The native PyTorch backend applies learned 3×3×3 kernels only to active input/output sites. S4 grows axial support by default for context; S3 growth is an optional ablation. Active counts and isolated-site fractions are logged. There is no dense 3D allocation.
- Each radar query sees at most `max_neighbors` clean-LiDAR sites within the scale's **physical search radius**. Sites inside the independent `positive_radii_m` threshold are positives; local sites outside it are negatives. Multi-positive InfoNCE uses normalized **learned feature similarity only** for ranking, with configurable temperature, scale weights, and nearest or random local negatives. The positive radii default to approximately one voxel diagonal at each stride: 0.4, 0.8, 1.6 and 3.2 m. These are geometric proxies, not verified shared radar/LiDAR reflectors. Queries without a positive are counted and excluded from retrieval denominators.
- The lightweight probe is one linear head per scale. It predicts local clean-surface presence and an XYZ offset from each radar sparse site. `L_geom` is binary cross entropy for local presence plus Smooth L1 for the positive site's XYZ in **meters** (`beta=0.2 m`). The target is the nearest *observed* clean voxel within the scale's search radius. This is site-local geometry, not full-scene reconstruction; no Chamfer is used. The occupancy and offset components have configurable weights within `L_geom`.
- At S1, the detached confidence target is `1[probe emits a point AND a clean target exists] × exp(-XYZ error / confidence_sigma_m)`. Thus a confident, inaccurate point or an abstention receives zero quality; clean occupancy alone cannot give a high target. `L_conf` is BCE against that soft target. `L_total = λcorr Lcorr + λgeom Lgeom + λconf Lconf`; all three weights can be zero independently, and all raw/weighted losses are logged. Confidence-only training from a randomly initialized probe gives an unreliable target; keep geometry supervision active or use a separately trained probe.

The default four-scale encoder and deployed feature widths are `[32,64,128,256]`; local correspondence projects these to a separate 64-D comparison space. [`configs/radar_lidar_stage1_small.json`](../../configs/radar_lidar_stage1_small.json) uses `[16,32,64,128]`. Edit a copied JSON config to test one scale, four scales, point resolution, channels or independently disable a loss. For a geometry-only ablation set `correspondence_weight=0`, `confidence_weight=0`, `geometric_weight>0`. The model never reads object labels for training.

## Run

From the repository root, with a CUDA PyTorch environment:

```bash
python -m models.radar_lidar_stage1.train \
  --vod-root /path/to/view_of_delft_PUBLIC \
  --output-root /path/to/stage1_run \
  --config configs/radar_lidar_stage1_small.json \
  --epochs 10 --batch-size 1 --grad-accum-steps 4 \
  --validate-every 5 --num-workers 2 --device cuda \
  --selection-metric corr_r1 --corruption-every 5 --tensorboard
```

Start with `--train-limit 1 --val-limit 1 --epochs 1` to verify the machine. Training reads official VoD train/val ImageSets and the selected 20-frame Doppler-shifted radar files directly; it does not need the fault cache. `last.ckpt` holds the full state; `best_corr.ckpt`, `best_geom.ckpt` and `best_selected.ckpt` are selected on held-out validation metrics, never training loss. `radar_only.pth` holds only the deployed model. `metrics.jsonl`, `validation_epoch_XXX.json`, and confidence-bin/threshold CSV files hold the full dataset-level metrics. `--tensorboard` writes the same key curves into grouped TensorBoard series and requires the `tensorboard` Python package. Object annotation evaluation is on by default for validation and can be disabled with `--no-object-instances`; the shifted/mismatched-radar audit is opt-in with `--corruption-every N`.

On this Windows workspace, the corresponding PowerShell command is:

```powershell
$VOD = 'C:\Users\gianl\Desktop\Thesis\View-Of-Delft dataset\view_of_delft_PUBLIC'
python -m models.radar_lidar_stage1.train `
  --vod-root "$VOD" --output-root outputs_local/stage1_run `
  --config configs/radar_lidar_stage1_small.json `
  --epochs 10 --batch-size 1 --grad-accum-steps 4 `
  --validate-every 5 --num-workers 0 --device cuda
```

```bash
python -m scripts.evaluate_radar_lidar_stage1 \
  --vod-root /path/to/view_of_delft_PUBLIC \
  --checkpoint /path/to/stage1_run/last.ckpt \
  --output /path/to/stage1_eval.json --split val --limit 50 \
  --corruptions --object-instances --device cuda

python -m scripts.visualize_radar_lidar_stage1 \
  --vod-root /path/to/view_of_delft_PUBLIC \
  --checkpoint /path/to/stage1_run/last.ckpt \
  --split val --frame-id 00000 \
  --output-prefix /path/to/stage1_00000 --device cuda
```

Validation reports per-scale and pooled local Recall@1/5/10 against **any** clean voxel inside the configured physical positive radius; XYZ localization distributions and thresholds; S1 probe precision/recall/F1 at 0.1/0.2/0.5 m; 10 confidence reliability bins and thresholds 0.1–0.9; no-correspondence counts; and object-instance retrieval when annotations exist. Counts are pooled across frames before rates are calculated. Geometry recall is over radar S1 sites with a clean target within the probe radius, not every clean point in the scene. ECE compares confidence to the *soft quality target*; binary geometric success at `geometry_eval_tolerance_m` is also shown in each bin. This is not full-cloud reconstruction precision/recall.

**Interpretation limits:** Because positives are *defined* as a clean voxel near a radar query, nearest-neighbor retrieval has Recall@1 of 100% on eligible queries. The required learned-feature Recall@1 is useful for tracking optimization, but cannot alone prove semantic correspondence. Compare it with the logged nearest-neighbor baseline, held-out object-instance retrieval, radar-only probe geometry, confidence reliability, and the shifted-radar audit. Shifted-radar retrieval uses the shifted positions to define new eligible queries and can stay high even for a bad model; fixed original clean-LiDAR anchor coverage is logged separately. S4's large physical voxels can dominate pooled metrics, so inspect S1 and other scales separately, especially for 0.2 m thresholds. A one-frame smoke checkpoint has no meaningful quality claim. Confidence targets are derived from an in-sample probe and can be optimistic; judge calibration on held-out validation and consider out-of-fold targets if a substantial train/validation gap appears.

## Backend and current verification

The local Windows PyTorch 2.8 CUDA installation lacks spconv/MinkowskiEngine/TorchSparse. The portable coordinate-sparse convolution backend was tested on a real VoD frame at base widths; one forward/backward pass with S3/S4 growth used approximately 3.3 GB GPU memory on the local RTX 4090. Full-run throughput must still be measured, especially with multiple data-loader workers. Sparse support growth can produce more S4 active sites than S1; read the logged site counts and isolated fractions before interpreting that level.
