# View-of-Delft LiDAR fault reconstruction

This repository now targets one deterministic Stage-I pipeline:

```text
VoD LiDAR/Radar points -> PointPillars -> masked HRNet -> repaired LiDAR BEV
```

The residual-diffusion Stage II remains in the repository but is intentionally
unchanged.

## Main components

- `Fault_Localization_Model/create_vod_reconstruction_dataset.py`: generate
  aligned clean/faulty LiDAR samples and accumulated VoD Radar caches.
- `models/reconstruction_head/fault_selector.py`: select primary and secondary
  repair boxes and their healthy halo context.
- `models/reconstruction_head/pointpillars.py`: encode aligned LiDAR and Radar
  point clouds into 320x320 pseudo-images.
- `models/reconstruction_head/coarse_reconstruction/hrnet_backbone.py`: the only
  supported deterministic reconstruction backbone.
- `models/reconstruction_head/coarse_reconstruction/train_coarse_reconstruction.py`:
  train the coarse model.
- `models/reconstruction_head/coarse_reconstruction/evaluate_coarse_by_fault.py`:
  evaluate exact and tolerant reconstruction metrics by fault.
- `models/reconstruction_head/diffusion_process/`: unchanged Stage-II code.

## Canonical configuration

`configs/coarse_reconstruction_vod.json` records the selected experiment:

- VoD PointPillars for LiDAR and accumulated Radar points
- 64 encoded channels per sensor
- four-stage HRNet with width 16 and dropout 0.1
- reconstruction and healthy-halo masks
- observability-weighted occupancy loss
- physically consistent flip, translation, yaw, and scale augmentation
- AdamW, learning rate 2e-4, weight decay 0.005

## Dataset generation

```powershell
python -m Fault_Localization_Model.create_vod_reconstruction_dataset `
  --vod-root "C:\path\to\View-Of-Delft dataset" `
  --output-root "C:\path\to\reconstruction_vod" `
  --radar-cache-root "C:\path\to\radar20_pointpillars_cache" `
  --radar-variant radar_20frames `
  --split train `
  --num-workers 8
```

Run the command once per `--split` (`train`, `val`, and `test`), changing
`--num-samples` when a split limit is required. Use `--help` for the current
generator options. Fault-selector masks are cached
separately with:

```powershell
python -m models.reconstruction_head.cache_fault_selector_masks `
  --data-root "C:\path\to\reconstruction_vod" `
  --config configs\coarse_reconstruction_vod.json `
  --num-workers 8
```

## Training

```bash
python -u -m models.reconstruction_head.coarse_reconstruction.train_coarse_reconstruction \
  --data-root /workspace/reconstruction_vod \
  --radar-root /workspace/radar20_pointpillars_cache \
  --output-root /workspace/coarse_vod_hrnet \
  --config configs/coarse_reconstruction_vod.json \
  --device cuda \
  --limit-train-samples 5139 \
  --limit-val-samples 1296
```

The sample limits are explicit because they preserve the historical seeded
selection used by the best experiment. Add `--disable-radar` only for a
controlled radar ablation.

## Evaluation

```bash
python -u -m models.reconstruction_head.coarse_reconstruction.evaluate_coarse_by_fault \
  --checkpoint /workspace/coarse_vod_hrnet/best_model.pt \
  --data-root /workspace/reconstruction_vod \
  --radar-root /workspace/radar20_pointpillars_cache \
  --output-root /workspace/coarse_vod_hrnet_test \
  --config configs/coarse_reconstruction_vod.json \
  --split test \
  --device cuda
```

## Tests

The tests use Python's built-in unittest runner:

```bash
python -m unittest discover -s tests
```
