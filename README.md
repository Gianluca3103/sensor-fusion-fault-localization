# Sensor Fusion Fault Localization

LiDAR BEV fault and weather localization pipeline for generating ideal reliability maps and training neural models to predict where a corrupted point cloud is unreliable.

This repository contains the source code for the final thesis model pipeline:

- Hercules LiDAR point-cloud loading and BEV projection.
- Synthetic fault/weather injection for rain, snow, fog, FOV loss, and laser degradation.
- Ideal grid reliability-map generation from clean vs faulty point clouds.
- A compact v5-like U-Net baseline.
- A PFS-inspired reliability-map model adapted from post-fusion feature stabilization.
- Validation metrics for heatmap quality, including IoU, F1, Brier score, MAE, threshold sweeps, and Chamfer distance.

Raw datasets, generated `.npz` samples, model checkpoints, and training outputs are intentionally not committed.

## Repository Layout

```text
Sensor-Fusion_Final_Model_Repo/
  Fault_Localization_Model/
    bev_utils.py                         # LiDAR-to-BEV projection helpers
    create_grid_reliability_heatmaps.py  # dataset/sample generation
    data_injection_utils.py              # wrappers around weather/fault injectors
    fault_injector.py                    # fault-plan routing
    heatmap_metrics.py                   # validation metrics
    model_v5_like.py                     # compact U-Net style baseline
    train_reliability_map.py             # baseline trainer
    test_reliability_map.py              # baseline tester
  PFS/
    pfs_model.py                         # PFS-inspired reliability model
    train_pfs_reliability_map.py         # PFS training
    test_pfs_reliability_map.py          # PFS prediction visualization
    run_generate_7500_and_train.ps1      # example overnight run
  Weather_Injector/
    3D_Corruptions_AD/                   # rain, snow, and FOV corruption support
    LiDAR_fog_sim/                       # LiDAR fog simulator support
  configs/
    reliability_dataset_1000.json        # sample generation config
  tests/
    test_*.py                            # reliability-map and fault-plan tests
  data/
    .gitkeep                             # local data placeholder
```

## Data Setup

Put the Hercules dataset locally under `data/` or point the scripts directly to your dataset path.

Expected local layout:

```text
Sensor-Fusion_Final_Model_Repo/
  data/
    HerculesFiles/
      Data/
        <scene folders>
```

Example external path:

```text
C:\Users\gianl\Desktop\Thesis\HerculesFiles\Data
```

The `data/` folder is ignored by Git so raw datasets stay local.

## Installation

Create and activate an environment, then install dependencies:

```powershell
cd "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For NVIDIA GPUs, install the PyTorch build that matches your machine from the official PyTorch selector. For recent RTX 50-series GPUs, use a CUDA build that supports the GPU compute capability.

## Generate Reliability Samples

This creates BEV inputs and ideal fault/reliability heatmaps from Hercules LiDAR frames.

```powershell
cd "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo"

python Fault_Localization_Model\create_grid_reliability_heatmaps.py `
  --data-root "C:\Users\gianl\Desktop\Thesis\HerculesFiles\Data" `
  --all-scenes `
  --output-root "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo\Fault_Localization_Model\grid_reliability_1000_x64_y32" `
  --num-samples 1000 `
  --fault-plan fog_sim:3 rain_sim:5 snow_sim:5 old_laser_degradation:0 fov_filter:1 `
  --grid-size 100 `
  --x-min 0 `
  --x-max 64 `
  --y-min -32 `
  --y-max 32 `
  --resolution 0.20 `
  --min-range 1.0 `
  --max-range 120.0 `
  --seed 42 `
  --log-level INFO
```

If generation stops, rerun the same command. Existing `.npz` samples are skipped.

## Reliability Map Definition

For each grid cell, the ideal target is computed from clean and faulty point-cloud counts:

```text
reliability = correct_points / (correct_points + faulty_points)
fault_heatmap = 1 - reliability
```

`faulty_points` includes missing points, newly added points, and count differences caused by moved or altered points. This avoids the old cancellation problem where added and missing points in the same grid cell could hide each other.

## Train the PFS Model

Example for a 24 GB GPU:

```powershell
cd "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo"

python PFS\train_pfs_reliability_map.py `
  --dataset-root "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo\Fault_Localization_Model\grid_reliability_1000_x64_y32" `
  --output-root "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo\PFS\runs\pfs_1000_x64_y32" `
  --epochs 200 `
  --batch-size 48 `
  --base-channels 16 `
  --dropout 0.15 `
  --learning-rate 3e-4 `
  --min-learning-rate 1e-6 `
  --warmup-epochs 20 `
  --weight-decay 1e-3 `
  --loss-mode stable `
  --grad-clip 1.0 `
  --early-stop-patience 0 `
  --best-checkpoint-metric val_brier_score `
  --resize-height 320 `
  --resize-width 320 `
  --grid-size 100 `
  --metric-threshold 0.5 `
  --stability-weight 0.05 `
  --pfs-reliability-weight 0.02 `
  --max-val-images 24 `
  --num-workers 8 `
  --device cuda
```

The trainer saves checkpoints, metrics, training history, plots, and validation prediction images under the selected `--output-root`.

## Resume Training

```powershell
python PFS\train_pfs_reliability_map.py `
  --dataset-root "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo\Fault_Localization_Model\grid_reliability_1000_x64_y32" `
  --output-root "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo\PFS\runs\pfs_1000_x64_y32" `
  --resume "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo\PFS\runs\pfs_1000_x64_y32\checkpoints\last_checkpoint.pt" `
  --epochs 200 `
  --batch-size 48 `
  --base-channels 16 `
  --device cuda
```

Keep architecture arguments such as `--base-channels` the same when resuming from a checkpoint.

## Visualize Predictions

```powershell
python PFS\test_pfs_reliability_map.py `
  --dataset-root "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo\Fault_Localization_Model\grid_reliability_1000_x64_y32" `
  --checkpoint "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo\PFS\runs\pfs_1000_x64_y32\checkpoints\best_model.pt" `
  --output-root "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo\PFS\runs\pfs_1000_x64_y32\test_predictions" `
  --max-images 10 `
  --device cuda
```

## Run Tests

```powershell
cd "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo"
python -m unittest discover -s tests -v
```

## Notes

- The weather injectors are included as local support code because the training sample generator calls them directly.
- The repository does not include raw Hercules data or trained checkpoints.
- Generated folders such as `grid_reliability_*`, `runs/`, `checkpoints/`, and preview images are ignored by Git.
