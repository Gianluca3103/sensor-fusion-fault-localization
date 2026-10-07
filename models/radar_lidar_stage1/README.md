# Radar → LiDAR Stage 1

This folder learns a **radar-only, sparse 3D geometric representation** with a synchronized clean-LiDAR teacher. It contains no reconstruction, diffusion, BEV, detector or faulty-LiDAR input. The repository/data audit and assumptions are in [`docs/radar_lidar_stage1_audit.md`](../../docs/radar_lidar_stage1_audit.md).

## Interface

`RadarLidarStage1.forward_radar(radar, radar_valid)` returns sparse S1–S4 radar features, sparse S1 confidence values, coordinates and physical geometry metadata. `RadarOnlyEncoder`, exported separately, contains no teacher, attention or probe weights. Confidence is supervised during training, but it is **not automatically calibrated**; inspect held-out reliability bins before choosing a threshold.

`RadarLidarStage1.forward_train(radar, radar_valid, clean_lidar, clean_valid)` uses clean LiDAR only to train local feature correspondence and a linear geometric probe. The deployed output is computed before the teacher is called. A test explicitly checks that different clean scans leave radar-only output identical.

### Radar-conditioned surface proposals

[`configs/radar_lidar_stage1_surface.json`](../../configs/radar_lidar_stage1_surface.json) enables a new deployed output: four candidate LiDAR surface positions **and a score for each** per occupied fine radar voxel. The head attends to nearby radar features at all four scales, including coarser scene context. Its positions can move away from radar voxel centers within a configured 1.5 m local radius. Clean LiDAR teaches the proposed positions through a local bidirectional set-distance loss; scores learn whether each proposed position is near an observed clean surface. Sites with no nearby observed clean surface receive no geometry target and train the score toward abstention. The clean encoder and training-only radar–LiDAR correspondence loss remain separate from the radar-only deployed path.

Stage II now seeds its candidate voxels from these predicted LiDAR locations when a surface-proposal checkpoint is loaded. It uses the original radar-site confidence positions for old checkpoints, so previous checkpoints retain their previous behavior. This is a new interface and needs **new Stage-I training**, followed by new Stage-II training; old weights cannot be converted into learned surface proposals. The local proposal radius limits the initial search and should be audited against validation coverage. A proposal near a clean point is a geometric proxy, not proof that radar and LiDAR reflected from the same physical surface.

Start with the new config and select the best checkpoint on held-out proposed-surface F1:

```bash
python -u -m models.radar_lidar_stage1.train \
  --vod-root /path/to/view_of_delft_PUBLIC \
  --radar-variant radar_20frames_verified_doppler_radial \
  --output-root /path/to/stage1_surface_run \
  --config configs/radar_lidar_stage1_surface.json \
  --epochs 50 --batch-size 8 --validate-every 5 \
  --selection-metric surface_f1_0.2m --num-workers 2 --device cuda
```

Validation logs proposal precision against all measured clean voxels and recall over clean voxels within the configured radar support radius, both at 0.2 m and 0.5 m. Candidate scores are thresholded at 0.25 to match the default Stage-II candidate selector. These proximity metrics do not establish reflector identity or point-cloud correctness; inspect held-out clouds and downstream results before using the proposal interface.

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

To inspect the learned confidence in a rotatable, synchronized three-panel cloud, run:

```bash
python -m scripts.visualize_stage1_confidence_cloud \
  --vod-root /path/to/view_of_delft_PUBLIC \
  --checkpoint /path/to/stage1_run/best_selected.ckpt \
  --split val --frame-id 08433 05001 \
  --output-root /path/to/stage1_confidence_views --device cuda
```

Open an output `*_stage1.html` in a browser. The three panels share one 3D scale. Every clean LiDAR point **inside the common display crop** is rendered without a point-count cap. Raw radar is hidden on the third panel by default; its optional overlay is independent of the score slider. For a surface-proposal checkpoint, `*_surface_proposals.ply` contains the predicted candidate XYZ and score. For a legacy checkpoint, `*_confidence.ply` contains radar-derived S1 voxel centers and confidence; those centers are **not** reconstructed LiDAR points. All PLY files retain the full clouds, including points outside the HTML crop. The score is not a calibrated probability unless separately validated.

Previously exported PLYs can be revisualized without rerunning the model or using CUDA:

```bash
python -m scripts.rerender_stage1_clouds \
  --input-root /path/to/old_stage1_views \
  --output-root /path/to/uncapped_views \
  --frame-id 03605 08469 00180 04696 04834
```

**Interpretation limits:** Because positives are *defined* as a clean voxel near a radar query, nearest-neighbor retrieval has Recall@1 of 100% on eligible queries. The required learned-feature Recall@1 is useful for tracking optimization, but cannot alone prove semantic correspondence. Compare it with the logged nearest-neighbor baseline, held-out object-instance retrieval, radar-only probe geometry, confidence reliability, and the shifted-radar audit. Shifted-radar retrieval uses the shifted positions to define new eligible queries and can stay high even for a bad model; fixed original clean-LiDAR anchor coverage is logged separately. S4's large physical voxels can dominate pooled metrics, so inspect S1 and other scales separately, especially for 0.2 m thresholds. A one-frame smoke checkpoint has no meaningful quality claim. Confidence targets are derived from an in-sample probe and can be optimistic; judge calibration on held-out validation and consider out-of-fold targets if a substantial train/validation gap appears.

## Backend and current verification

The local Windows PyTorch 2.8 CUDA installation lacks spconv/MinkowskiEngine/TorchSparse. The portable coordinate-sparse convolution backend was tested on a real VoD frame at base widths; one forward/backward pass with S3/S4 growth used approximately 3.3 GB GPU memory on the local RTX 4090. Full-run throughput must still be measured, especially with multiple data-loader workers. Sparse support growth can produce more S4 active sites than S1; read the logged site counts and isolated fractions before interpreting that level.
