# Deterministic coarse LiDAR reconstruction

## Code layout

```text
reconstruction_head/
├── coarse_reconstruction/   coarse model, loss, config, and trainer
├── diffusion_process/       diffusion model, process, pipeline, metrics, and trainer
├── coarse_dataset.py        aligned BEV and cached-mask loading shared by both stages
├── encoders.py              shared encoder building blocks
├── fault_selector.py        shared repair/halo selection
└── fault_selector_cache.py  shared deterministic mask cache
```

`pointpillars.py` contains the optional learned LiDAR/radar pillar encoders used
by the first representation ablation.

The coarse stage directly predicts replacement LiDAR BEV content. It has no
separate reconstruction-latent encoder and does not predict a correction to
hidden LiDAR values. It contains no diffusion process.

## Masks

- `reconstruction_mask` is the complete editable region selected for
  reconstruction. Every LiDAR cell in it is erased and recreated, including
  healthy cells deliberately sacrificed by the rectangular selection.
- `healthy_context_mask` contains only occupied, reliable LiDAR cells in the
  surrounding context.
- `halo_mask` is the complete geometric halo. Together with the reconstruction
  mask, it determines where local radar evidence is retained.

The model erases selected LiDAR before either the local or global branch can
use it:

```text
erased_lidar_bev = (1 - reconstruction_mask) * faulty_lidar_bev
```

The global LiDAR encoder receives this erased BEV concatenated with the
reconstruction mask. The extra binary channel marks erased cells as unknown,
so they cannot be confused with observed-empty LiDAR cells.

The Coarse U-Net receives nine direct 320×320 channels:

```text
healthy_context_mask * faulty_lidar_bev       3 channels
(reconstruction_mask OR halo_mask) * radar    4 channels
reconstruction_mask                           1 channel
healthy_context_mask                          1 channel
```

The default four-level residual U-Net compresses 320×320 to a 40×40 local
bottleneck. Separate lightweight global LiDAR and radar encoders preserve the
complete BEV as spatial maps. Learned global fusion produces
`global_context_map`, and every local bottleneck query attends to every global
map position using absolute BEV positional embeddings.

The decoder returns to 320×320 and predicts:

```text
replacement_raw: [B, 3, 320, 320]
  channel 0: occupancy logits
  channel 1: normalized log point density
  channel 2: normalized robust P90 height
```

For the assembled BEV, the occupancy logits are converted to probabilities;
the two continuous predictions remain unchanged:

```text
replacement_bev = concat(
    sigmoid(occupancy_logits),
    predicted_density,
    predicted_height,
)
```

The structural output merge is:

```text
coarse_lidar_bev =
    (1 - reconstruction_mask) * faulty_lidar_bev
    + reconstruction_mask * replacement_bev
```

Thus, output cells outside the reconstruction mask are exactly unchanged and
all cells inside it come from the replacement prediction.

## Training

### Experiment 1: learned PointPillars sensor representation

`configs/coarse_reconstruction_pointpillars.json` enables two independent
PointPillars-style encoders while leaving the reconstruction U-Net, global
fusion, masks, three-channel target, loss, and metrics unchanged. Generated
sample metadata remains the only source of BEV geometry. With the production
range `x=[0,64)` and `y=[-32,32)`, the aligned 320x320 pillars are 0.20 m by
0.20 m.

LiDAR uses raw `[x,y,z,reflectivity]` plus XYZ cluster offsets and XY
pillar-center offsets (9 decorated values). Radar uses aligned
`[x,y,z,power,doppler]` plus the same decorations (10 values). Each independent
Pillar Feature Network produces `[B,64,320,320]`. The local model therefore
receives 130 channels in the default masked-context experiment, while the
replacement head still predicts the original three target channels.

PointPillars requires generated samples from generator version 11 or newer,
which store `faulty_lidar_points`, and K-Radar RadarV2 caches from format version
9 or newer, which store shared aligned `radar_points`. Re-run the normal sample
generator and RadarV2 cache builder after updating the code; obsolete artifacts
are rejected or regenerated instead of reconstructing points from BEVs.

Train the representation ablation with the same split and schedule as the
baseline, changing only the configuration path:

```powershell
python -m models.reconstruction_head.coarse_reconstruction.train_coarse_reconstruction `
  --data-root "C:\path\to\generated_samples_v11" `
  --radar-root "C:\path\to\radar_v2_cache_v9" `
  --output-root "C:\path\to\coarse_pointpillars_run" `
  --config "configs\coarse_reconstruction_pointpillars.json" `
  --device cuda
```

Keep `configs/coarse_reconstruction.json` selected for the handcrafted-BEV
baseline.

### Experiment: full-resolution HRNet reconstruction backbone

`configs/coarse_reconstruction_pointpillars_hrnet.json` changes only the dense
coarse reconstruction backbone. The PointPillars encoders, 130-channel masked
local input, three-channel target, losses, masks, optimizer, and final masked
replacement rule remain unchanged.

Unlike the U-Net, HRNet has no single encoder bottleneck and decoder. It keeps
parallel `[320,160,80,40]` spatial streams with `[16,32,64,128]` channels.
Every active branch is processed by GroupNorm/SiLU residual blocks, and every
fusion sends information from all active resolutions to all output
resolutions. Low-to-high paths use 1x1 projection and bilinear interpolation;
high-to-low paths use learned stride-2 3x3 convolutions. The final HRNetV2 head
upsamples the three lower-resolution branches, concatenates 240 channels at
320x320, and fuses them to 32 channels before the unchanged three-channel
replacement head.

The 320x320 branch remains present throughout the complete backbone. HRNet
does not use the U-Net global-attention map because its persistent 40x40 branch
and repeated multiresolution fusion provide broad context inside the backbone.

```powershell
python -m models.reconstruction_head.coarse_reconstruction.train_coarse_reconstruction `
  --data-root "C:\path\to\pointpillars_samples" `
  --radar-root "C:\path\to\radar_v2_cache_v9" `
  --output-root "C:\path\to\coarse_hrnet_run" `
  --config "configs\coarse_reconstruction_pointpillars_hrnet.json" `
  --device cuda
```

The independent trainer is
`models.reconstruction_head.coarse_reconstruction.train_coarse_reconstruction`;
its default configuration is `configs/coarse_reconstruction.json`.

Fault Selector masks are precomputed once rather than recalculated in every
epoch. Generate the cache before coarse or diffusion training:

The selector first marks cells whose fraction of original LiDAR returns lost
is at least `min_lidar_loss_fraction` (0.95 by default). It then chooses one
dominant depth band containing the most severe-loss evidence and crops its
left and right edges to the outermost severe-loss cells, avoiding empty side
columns, while requiring
at least `min_repair_fault_fraction` (0.95 by default) of occupied informative
cells inside the box to be severe. Empty cells are neutral. Extending the box
into a healthier depth range is rejected even when that leaves additional
fault cells unreconstructed. Only the separate context halo is dilated outward.

```powershell
python -m models.reconstruction_head.cache_fault_selector_masks `
  --data-root "C:\path\to\grid_reliability_dataset" `
  --config "configs\coarse_reconstruction.json" `
  --num-workers 8
```

By default, a dataset named `grid_reliability_dataset` uses the sibling cache
directory `grid_reliability_dataset_fault_selector_cache`. Masks are compressed
`uint8` arrays and include the complete Fault Selector configuration. Training
fails clearly when the cache is missing or its configuration is stale;
rerunning the command rebuilds only stale entries.

```powershell
python -m models.reconstruction_head.coarse_reconstruction.train_coarse_reconstruction `
  --data-root "C:\path\to\grid_reliability_dataset" `
  --radar-root "C:\path\to\radar_v2_cache" `
  --output-root "C:\path\to\coarse_reconstruction_run" `
  --device cuda
```

Use `--radar-mode full`, `global-only`, or `none` for controlled radar-input
ablations. `global-only` supplies real radar only to the global radar encoder
and zeros the local U-Net radar channels. `none` zeros radar for both branches;
`--disable-radar` remains an alias for that mode. The architecture and parameter
count remain unchanged, and the selected mode is recorded in checkpoints.

Pass `--disable-global-map` for the local-U-Net-only ablation. It bypasses both
global encoders, global fusion, positional cross-attention, and bottleneck
fusion; the local bottleneck is sent directly into the decoder. Local LiDAR,
local radar, masks, skip connections, decoder, and replacement head remain
active. The instantiated architecture is unchanged, but global parameters are
not executed or trained during this ablation.

The baseline objective is channel-aware. Occupancy uses mask-normalized binary
cross entropy with logits plus per-sample soft Dice throughout
`reconstruction_mask`. Density and height use Smooth L1 only where the cell is
both inside `reconstruction_mask` and occupied in the clean target:

```text
M_repair = reconstruction_mask
M_continuous = reconstruction_mask * clean_occupancy

L = lambda_occupancy * (L_BCE + L_Dice)
    + lambda_density * L_SmoothL1_density
    + lambda_height * L_SmoothL1_height
```

The default configuration keeps observability weighting disabled, reproducing
the original baseline exactly. The controlled observability ablation can be
enabled without changing the model or its inputs:

```json
"observability_weighting": {
  "enabled": true,
  "min_empty_weight": 0.1
}
```

For clean-occupied cells the BCE weight is always one. For clean-empty cells:

```text
w_empty = min_empty_weight
          + (1 - min_empty_weight) * observability_confidence

L_BCE_obs = sum(BCEWithLogits * reconstruction_mask * w)
            / sum(reconstruction_mask * w)
```

Only BCE uses this weight. Dice, density Smooth L1, height Smooth L1, the
reconstruction metrics, and every inference path remain unchanged. Enabled
training requires aligned `observability_confidence`; it never silently falls
back to baseline BCE.

Empty masks and masks without clean-occupied cells return differentiable zero
for the corresponding terms. Validation compares coarse and faulty inputs
inside the repair region using exact-cell occupancy precision, recall, F1 and
IoU at 0.2 m resolution; bidirectional tolerance-matched precision, recall and
F1 at 0.5 m; its monotonic IoU-equivalent `F1 / (2 - F1)`; and hallucination
rate. It also reports density MAE/RMSE and robust-height MAE/RMSE in normalized
units and meters. It also verifies that the result is unchanged outside the
repair mask.

The coarse trainer always writes `history.csv`. Pass `--tensorboard` to also
stream all numeric train/validation metrics, active-mask statistics, epoch
runtimes, faulty-input baselines, and optimizer learning rates to
`<output-root>/tensorboard`. Override that location with
`--tensorboard-log-dir`.

During the first training and validation pass, the trainer records the
per-sample fraction of the complete BEV covered by `reconstruction_mask OR
halo_mask`. It writes train, validation, and combined median, 90th percentile,
and maximum coverage to `active_fraction_profile.json`, then prints a
dense-versus-cropped/sparse architecture recommendation. This static mask
profile is collected only once and does not change model behavior.

The trainer writes checkpoints, history, first-batch tensor shapes, and sample
outputs containing `coarse_lidar_bev`, `replacement_raw`, and
`reconstruction_mask`. When attention diagnostics are enabled, attention
weights are requested only for the first validation batch that is saved; later
validation batches do not materialize unused attention matrices.

Aggregate training history does not preserve fault identities. Evaluate a
frozen checkpoint separately to obtain per-sample, per-fault, and
per-fault/severity reconstruction metrics:

```bash
python -m models.reconstruction_head.coarse_reconstruction.evaluate_coarse_by_fault \
  --checkpoint /path/to/coarse_run/best_model.pt \
  --data-root /path/to/generated_samples \
  --radar-root /path/to/radar_cache \
  --output-root /path/to/coarse_run/per_fault_evaluation \
  --config configs/coarse_reconstruction.json \
  --split val --device cuda --batch-size 32 --num-workers 4 \
  --visualize-samples-per-fault 5
```

The evaluator reads radar/global-map ablation settings from the checkpoint and
writes `per_sample_metrics.csv`, `by_fault_metrics.csv`, and `summary.json`.
It reports macro per-sample metrics as well as micro exact-cell IoU/F1 formed
from summed TP/FP/FN counts. Clean/reconstructed/faulty-LiDAR-plus-radar
comparison PNGs are organized under `visualizations/<fault>_s<severity>/`.

## Masked residual diffusion

The second trainer freezes a completed coarse checkpoint and learns only its
remaining error in direct BEV space:

```text
residual_gt = reconstruction_mask * (clean_lidar_bev - coarse_lidar_bev)
```

At every forward and reverse step, the residual and Gaussian noise are zeroed
outside `reconstruction_mask`. The diffusion U-Net receives eleven spatial
channels: three noisy residual channels, three coarse-BEV channels, four local
radar channels masked to the union of the repair region and geometric halo,
and the one-channel reconstruction mask. It does not receive global radar
representations, global context, attention outputs, or latent features.

The default configuration is `configs/residual_diffusion.json`. It uses 1000
cosine DDPM steps, epsilon prediction, a timestep-conditioned convolutional
U-Net, and mask-normalized epsilon MSE. The LiDAR BEVs are already stored as
uint8-derived values divided by 255, so the default per-channel normalization
is identity; configurable training-set means and standard deviations are saved
in checkpoints when supplied.

```powershell
python -m models.reconstruction_head.diffusion_process.train_residual_diffusion `
  --data-root "C:\path\to\grid_reliability_dataset" `
  --radar-root "C:\path\to\radar_v2_cache" `
  --coarse-checkpoint "C:\path\to\coarse_run\best_model.pt" `
  --output-root "C:\path\to\residual_diffusion_run" `
  --device cuda
```

The trainer optimizes only diffusion U-Net parameters and supports AMP,
gradient clipping, exact resume, best/latest checkpoints, first-batch shape
logging, and final validation-set ancestral DDPM sampling. Final metrics compare
erased, coarse, and diffusion-refined BEVs inside the repair mask. Secondary
full-scene metrics and actual-fault/sacrificed-healthy diagnostic regions are
also written. Occupancy is consistently stored in LiDAR channel 0.

The first implementation intentionally supports DDPM only. The sampler API is
separate from the network and schedule so DDIM can be added later without
changing the training model.

## Object-level reconstruction-region evaluation

K-Radar GT boxes can be evaluated against cached reconstruction masks without
running the reconstruction network:

```powershell
python -m models.reconstruction_head.evaluate_object_overlap `
  --data-root "C:\path\to\generated_samples" `
  --kradar-root "C:\path\to\K-Radar_Data" `
  --output-root "C:\path\to\object_overlap_results" `
  --split val `
  --config "configs\coarse_reconstruction.json" `
  --visualize-samples 10
```

Use `--revised-label-root` when the revised K-Radar v2.0 labels are stored
outside the sequence directories. Revised labels are preferred when present;
the frame label recorded in sample metadata is the fallback. The loader follows
the official convention: yaw degrees are converted to radians and stored half
dimensions are doubled into full length, width, and height. Labels are not
calibration-shifted because these BEVs are rasterized in the native LiDAR frame.

The command writes `objects.csv`, `objects.json`, `summary.json`, and GT/mask
visualizations. Footprint coverage uses exact oriented-polygon area against the
union of reconstruction-mask cells. Detector-derived loss, recovery, and
preservation metrics remain explicitly unavailable: coarse output is a
three-channel BEV raster, not a point cloud, and this repository does not have
a frozen detector that consumes that representation. No BEV-to-point-cloud
conversion is fabricated.
