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

The independent trainer is
`models.reconstruction_head.coarse_reconstruction.train_coarse_reconstruction`;
its default configuration is `configs/coarse_reconstruction.json`.

Fault Selector masks are precomputed once rather than recalculated in every
epoch. Generate the cache before coarse or diffusion training:

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
from summed TP/FP/FN counts. Clean/reconstructed/error comparison PNGs are
organized under `visualizations/<fault>_s<severity>/`.

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
