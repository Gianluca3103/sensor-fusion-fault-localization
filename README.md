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
