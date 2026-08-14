# Sensor Fusion Fault Localization

LiDAR reconstruction research pipeline using the View of Delft (VoD) dataset.
The maintained workflow creates aligned LiDAR/radar PointPillars inputs,
injects controlled LiDAR faults, selects reconstruction and halo regions, and
trains deterministic coarse-reconstruction or residual-diffusion models.

Raw VoD data, generated samples, radar caches, checkpoints, and run outputs are
kept outside Git.

## Maintained workflow

1. `Fault_Localization_Model/create_vod_reconstruction_dataset.py` creates
   aligned clean/faulty LiDAR samples and VoD radar caches.
2. `models/reconstruction_head/cache_fault_selector_masks.py` precomputes the
   reconstruction, halo, and healthy-context masks.
3. `models/reconstruction_head/coarse_reconstruction/` contains the U-Net,
   HRNet, SST, and repair-query coarse backbones and their trainer.
4. `models/reconstruction_head/diffusion_process/` contains the masked residual
   diffusion refinement stage.

## Dataset generation

Run each official VoD split independently so the dataset's scene separation is
preserved:

```powershell
python -m Fault_Localization_Model.create_vod_reconstruction_dataset `
  --vod-root "C:\path\to\View-Of-Delft dataset" `
  --output-root "C:\path\to\reconstruction_vod_radar3_unique" `
  --radar-cache-root "C:\path\to\radar3_pointpillars_cache" `
  --split train `
  --radar-variant radar_3frames `
  --fault-plan fog_sim:4 fog_sim:5 `
  --num-workers 8
```

Repeat with `--split val` and `--split test`. The generated samples contain a
320x320 `valid_support_mask`, so selector-cache generation has no dependency on
dataset-specific geometry code.

To inspect the full set of physical and derived channels supported by VoD's
five-frame radar release alongside LiDAR statistics:

```powershell
python -m scripts.export_vod_channel_analysis `
  --vod-root "C:\path\to\View-Of-Delft dataset" `
  --output-root "C:\path\to\vod_5stack_channel_analysis" `
  --split train `
  --limit 50
```

The command writes physical-unit channel arrays to compressed NPZ files and
separate radar/LiDAR channel-grid PNGs. Its manifest explicitly distinguishes
measured fields from radial-only derived velocity projections and unavailable
LiDAR timestamps.

```powershell
python -m models.reconstruction_head.cache_fault_selector_masks `
  --data-root "C:\path\to\reconstruction_vod_radar3_unique" `
  --config configs\coarse_reconstruction_vod_radar3_pointpillars_hrnet.json `
  --num-workers 8
```

## Coarse reconstruction

```powershell
python -m models.reconstruction_head.coarse_reconstruction.train_coarse_reconstruction `
  --data-root "C:\path\to\reconstruction_vod_radar3_unique" `
  --radar-root "C:\path\to\radar3_pointpillars_cache" `
  --output-root "C:\path\to\coarse_vod_hrnet" `
  --config configs\coarse_reconstruction_vod_radar3_pointpillars_hrnet.json `
  --device cuda `
  --epochs 150 `
  --batch-size 12 `
  --num-workers 8
```

VoD-specific HRNet, SST, halo, augmentation, and U-Net cross-attention
configurations are available under `configs/` with the
`coarse_reconstruction_vod_` prefix.

## Tests

```powershell
python -m unittest discover -s tests -v
```
### View-of-Delft 10/20-frame radar accumulation

Generate ego-motion compensated radar histories from the single-frame VoD
release. Histories never cross recording boundaries; RCS and Doppler fields
are retained and `time_index` runs from the oldest scan to zero for the
current scan.

```powershell
python -m scripts.generate_vod_accumulated_radar `
  --vod-root "C:\Users\gianl\Desktop\Thesis\View-Of-Delft dataset" `
  --stack-sizes 10 20 `
  --num-workers 8
```

The output is written under `view_of_delft_PUBLIC/radar_10frames` and
`view_of_delft_PUBLIC/radar_20frames`. Both variants can be selected with
`--radar-variant` when creating reconstruction inputs.
