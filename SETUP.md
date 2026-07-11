# Setup Notes

## Recommended GPU Install

Install PyTorch from the official selector for your CUDA driver and GPU. Example for a CUDA 12.9 wheel:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

## Hercules Data

Keep raw Hercules files outside Git. Either place them under:

```text
data/HerculesFiles/Data
```

or pass the absolute path with `--data-root`.

## Suggested Large-Run Settings

For an RTX 4090 with 24 GB VRAM:

```text
batch size: 48 to 64
num workers: 8 to 16
base channels: 16 first, then 24 if stable
epochs: 200
warmup epochs: 20
best checkpoint metric: val_brier_score
```

If GPU utilization is already near 95-100%, increasing `--num-workers` will usually not help.
