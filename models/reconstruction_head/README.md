# Deterministic coarse LiDAR reconstruction

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

The Coarse U-Net receives nine direct 320×320 channels:

```text
healthy_context_mask * faulty_lidar_bev       3 channels
(reconstruction_mask OR halo_mask) * radar    4 channels
reconstruction_mask                           1 channel
healthy_context_mask                          1 channel
```

The default five-level residual U-Net compresses 320×320 to a 20×20 local
bottleneck. Separate lightweight global LiDAR and radar encoders preserve the
complete BEV as 20×20 spatial maps. Learned global fusion produces
`global_context_map`, and every local bottleneck query attends to every global
map position using absolute BEV positional embeddings.

The decoder returns to 320×320 and predicts:

```text
replacement_raw: [B, 3, 320, 320]
```

The structural output merge is:

```text
coarse_lidar_bev =
    (1 - reconstruction_mask) * faulty_lidar_bev
    + reconstruction_mask * replacement_raw
```

Thus, output cells outside the reconstruction mask are exactly unchanged and
all cells inside it come from the replacement prediction.

## Training

The independent trainer is `training.train_coarse_reconstruction`; its default
configuration is `configs/coarse_reconstruction.json`.

```powershell
python -m training.train_coarse_reconstruction `
  --data-root "C:\path\to\grid_reliability_dataset" `
  --radar-root "C:\path\to\radar_v2_cache" `
  --output-root "C:\path\to\coarse_reconstruction_run" `
  --device cuda
```

The mask-normalized Smooth L1 objective supervises `replacement_raw` against
clean LiDAR over every cell in `reconstruction_mask`. Empty masks return a
differentiable zero loss. Validation logs erased and coarse masked MAE,
reconstruction improvement, relative improvement, and outside-mask change.

The trainer writes checkpoints, history, first-batch tensor shapes, and sample
outputs containing `coarse_lidar_bev`, `replacement_raw`, and
`reconstruction_mask`.

## Masked residual diffusion

The second trainer freezes a completed coarse checkpoint and learns only its
remaining error in direct BEV space:

```text
residual_gt = reconstruction_mask * (clean_lidar_bev - coarse_lidar_bev)
```

At every forward and reverse step, the residual and Gaussian noise are zeroed
outside `reconstruction_mask`. The diffusion U-Net receives only seven spatial
channels: three noisy residual channels, three coarse-BEV channels, and the
one-channel reconstruction mask. It does not receive radar, halo masks, global
context, attention outputs, or latent features.

The default configuration is `configs/residual_diffusion.json`. It uses 1000
cosine DDPM steps, epsilon prediction, a timestep-conditioned convolutional
U-Net, and mask-normalized epsilon MSE. The LiDAR BEVs are already stored as
uint8-derived values divided by 255, so the default per-channel normalization
is identity; configurable training-set means and standard deviations are saved
in checkpoints when supplied.

```powershell
python -m training.train_residual_diffusion `
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
also written. Occupancy is consistently defined as LiDAR channel 2 greater
than zero.

The first implementation intentionally supports DDPM only. The sampler API is
separate from the network and schedule so DDIM can be added later without
changing the training model.
