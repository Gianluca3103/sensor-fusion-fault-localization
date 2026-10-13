# Radar → LiDAR Stage 1

This folder learns a **radar-only, sparse 3D geometric representation** with a synchronized clean-LiDAR teacher. It contains no reconstruction, diffusion, BEV, detector or faulty-LiDAR input. The repository/data audit and assumptions are in [`docs/radar_lidar_stage1_audit.md`](../../docs/radar_lidar_stage1_audit.md).

## Interface

The active [`radar_lidar_stage1_surface.json`](../../configs/radar_lidar_stage1_surface.json) configuration now selects `BottleneckSupportStage1`. Its exported radar-only encoder returns **exactly two sparse tensors**: a stride-8 LiDAR-aware radar bottleneck embedding `[N4,64]` and a stride-2 3D confidence field `[M,1]`, each with its voxel coordinates. The field is scored, not a LiDAR point cloud; its scores are not calibrated probabilities.

During training, the clean-LiDAR encoder and radar encoder are compared **only at the stride-8 bottleneck**. Local attention over clean-LiDAR bottleneck features provides a teacher target, and a radar-only student learns to approximate that representation. Clean LiDAR also supplies support-field positives and measured free-space negatives; unobserved space is not labeled empty. The deployed model receives radar alone. No attention weights, clean features, surface positions, or radii cross its inference boundary.

The support head decodes each bottleneck cell onto a stride-2 lattice, using an internal radar skip feature for spatial detail and learning a score per lattice cell. Stage II interpolates the scored field with the edge-aware method, pools only the bottleneck embeddings at the active fine cells, then combines them with faulty-LiDAR features. The previous Stage-II preprocessing cache and all Stage-I/II checkpoints must be rebuilt for this interface.

```bash
python -u -m models.radar_lidar_stage1.train \
  --vod-root /path/to/view_of_delft_PUBLIC \
  --radar-variant radar_20frames_verified_doppler_radial \
  --output-root /path/to/stage1_bottleneck_support \
  --config configs/radar_lidar_stage1_surface.json \
  --epochs 50 --batch-size 8 --validate-every 5 \
  --selection-metric support_f1 --no-object-instances \
  --num-workers 2 --device cuda
```

Validation reports support precision on measured clean/free cells, recall on clean cells inside the radar-derived lattice, unknown selected sites, and clean coverage before scoring. These metrics cannot prove object reconstruction quality; inspect held-out clouds and Stage-II results.

The trainer builds a per-frame cache under `output-root/static_geometry` as it
visits frames. It stores fixed bottleneck neighbor indices and distances plus
the support lattice's clean occupied and measured free-space labels. The first
epoch builds entries; later epochs and resumed runs reuse them even when batch
order changes. Radar and LiDAR encoder features, attention weights, support
scores and gradients are computed afresh every step. The cache is invalidated
when the configuration or source radar/LiDAR/calibration file metadata changes.
Use `--static-cache-root /path/to/cache` to put it on another volume or
`--no-static-cache` to compare with uncached execution. The progress bar and
epoch metrics report reused and built frame counts. Loss statistics stay on
the GPU until periodic progress updates and the epoch summary.

Visualize a held-out frame with `python -m scripts.visualize_stage1_confidence_cloud --vod-root /path/to/view_of_delft_PUBLIC --checkpoint /path/to/stage1_bottleneck_support/best_selected.ckpt --split val --frame-id 08236 --output-root /path/to/stage1_views --device cuda`. The viewer writes both the raw scored support lattice and its interpolated Stage-II region as uncapped PLY files.

### Historical surface-proposal interface (older checkpoints only)

Earlier Stage-I checkpoints may contain two candidate surface positions, scores, and learned 3D radii per grouped radar neighborhood. This path remains loadable for compatibility, but the active config above no longer trains or exports those proposals.

Legacy Stage-II checkpoints retain their own proposal rasterization behavior. They cannot be paired with the new bottleneck-support checkpoint.

## Representation and losses

### PFE encoder from `encoder.py`

The bottleneck-support training config sets `encoder_backend=pfe_spconv`. Both
the radar path and the clean-LiDAR teacher use independent copies of the
downloaded PFE sparse backbone: a 3×3 submanifold stem, residual blocks at
strides 1 and 2, two residual blocks at stride 4, and the stride-8 latent
block with 1×1 compression and final residual block. Its original
mean-XYZ/log-density 4→16→32 voxel MLP is retained. A separate learned
attribute branch preserves radar RCS, Doppler and scan age, and LiDAR
reflectivity. The encoder computes four sparse scales; the active interface
exports only the stride-8 bottleneck features and a stride-2 confidence field.
Cross-modal comparison occurs at stride 8. It does not call the original
encoder's `.dense()` output. The voxel grid remains 0.2 × 0.2 × 0.25 m.
The original `encoder.py`'s separate `config.py` range and voxel size are
not used.

This backend requires `spconv.pytorch` in the training environment and
fails immediately if it is missing. There is no silent zero-feature fallback.
Old checkpoints without `encoder_backend` continue to select `native`, but
their weights cannot initialize the new PFE backbone. Start a fresh Stage-I
run and rebuild Stage-II preprocessing/training from the new Stage-I
checkpoint. The PFE backbone has no site-growth convolution, so the new
config sets `growth_scales=[]`. The learned support field defines the new
Stage-II domain.

The new config sets `bottleneck_cross_attention_only=true` and
`corr_scale_weights=[0,0,0,1]`. Only the stride-8 radar and clean-LiDAR
features enter the training-only cross-modal correspondence module. S1–S3
have no radar–LiDAR attention modules. The new trainer reports bottleneck
correspondence, distillation and support-field losses. Earlier checkpoints
retain their four-scale correspondence architecture and metrics.

- Canonical coordinates are current-frame LiDAR XYZ. Radar data are transformed with the per-frame VoD radar/camera/LiDAR calibration in `vod_io.py`. Raw radar attributes are RCS, raw radial velocity, compensated radial velocity and ordinal scan age. LiDAR has reflectivity. Each modality has its own point MLP and learned voxel mean/max/count aggregation.
- Sparse sites have sorted unique `[batch,z,y,x]` integer coordinates and `[N,C]` features. The native PyTorch backend applies learned 3×3×3 kernels only to active input/output sites. S4 grows axial support by default for context; S3 growth is an optional ablation. Active counts and isolated-site fractions are logged. There is no dense 3D allocation.
- At the enabled correspondence scale, each radar query sees at most `max_neighbors` clean-LiDAR sites within that scale's **physical search radius**. Sites inside the independent `positive_radii_m` threshold are positives; local sites outside it are negatives. Multi-positive InfoNCE uses normalized **learned feature similarity only** for ranking, with configurable temperature and nearest or random local negatives. The new PFE config compares only S4, with a 6.0 m search radius and 3.2 m positive radius. These are geometric proxies, not verified shared radar/LiDAR reflectors. Queries without a positive are counted and excluded from retrieval denominators.
- The lightweight probe is one linear head per scale. It predicts local clean-surface presence and an XYZ offset from each radar sparse site. `L_geom` is binary cross entropy for local presence plus Smooth L1 for the positive site's XYZ in **meters** (`beta=0.2 m`). The target is the nearest *observed* clean voxel within the scale's search radius. This is site-local geometry, not full-scene reconstruction; no Chamfer is used. The occupancy and offset components have configurable weights within `L_geom`.
- At S1, the detached confidence target is `1[probe emits a point AND a clean target exists] × exp(-XYZ error / confidence_sigma_m)`. Thus a confident, inaccurate point or an abstention receives zero quality; clean occupancy alone cannot give a high target. `L_conf` is BCE against that soft target. `L_total = λcorr Lcorr + λgeom Lgeom + λconf Lconf`; all three weights can be zero independently, and all raw/weighted losses are logged. Confidence-only training from a randomly initialized probe gives an unreliable target; keep geometry supervision active or use a separately trained probe.

The default four-scale encoder and deployed feature widths are `[32,64,128,256]`; local correspondence projects these to a separate 64-D comparison space. [`configs/radar_lidar_stage1_small.json`](../../configs/radar_lidar_stage1_small.json) uses `[16,32,64,128]`. Edit a copied JSON config to test one scale, four scales, point resolution, channels or independently disable a loss. For a geometry-only ablation set `correspondence_weight=0`, `confidence_weight=0`, `geometric_weight>0`. The model never reads object labels for training.

## Legacy Stage-I run and metrics

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

Open an output `*_stage1.html` in a browser. The three panels share one 3D scale. Every clean LiDAR point **inside the common display crop** is rendered without a point-count cap. Raw radar is hidden on the third panel by default; its optional overlay is independent of the score slider. For a learned-region checkpoint, the third panel and `*_support_region.ply` show **all active fine voxels** in the proposed patches; `*_surface_proposals.ply` separately stores their learned centers and scores. For a point-only proposal checkpoint, the third panel shows proposal centers. For a legacy checkpoint, `*_confidence.ply` contains radar-derived S1 voxel centers and confidence; those centers are **not** reconstructed LiDAR points. All PLY files retain the full clouds, including points outside the HTML crop. The score is not a calibrated probability unless separately validated.

Previously exported PLYs can be revisualized without rerunning the model or using CUDA:

```bash
python -m scripts.rerender_stage1_clouds \
  --input-root /path/to/old_stage1_views \
  --output-root /path/to/uncapped_views \
  --frame-id 03605 08469 00180 04696 04834
```

**Interpretation limits:** Because positives are *defined* as a clean voxel near a radar query, nearest-neighbor retrieval has Recall@1 of 100% on eligible queries. The required learned-feature Recall@1 is useful for tracking optimization, but cannot alone prove semantic correspondence. Compare it with the logged nearest-neighbor baseline, held-out object-instance retrieval, radar-only probe geometry, confidence reliability, and the shifted-radar audit. Shifted-radar retrieval uses the shifted positions to define new eligible queries and can stay high even for a bad model; fixed original clean-LiDAR anchor coverage is logged separately. S4's large physical voxels can dominate pooled metrics, so inspect S1 and other scales separately, especially for 0.2 m thresholds. A one-frame smoke checkpoint has no meaningful quality claim. Confidence targets are derived from an in-sample probe and can be optimistic; judge calibration on held-out validation and consider out-of-fold targets if a substantial train/validation gap appears.

## Backend and current verification

The local Windows PyTorch 2.8 CUDA installation lacks spconv/MinkowskiEngine/TorchSparse, so the new PFE backend cannot be GPU-tested there until a compatible spconv wheel is installed. The older portable coordinate-sparse backend remains available for old checkpoints and local checks. Full-run throughput of the PFE backend remains to be measured on the training machine.
