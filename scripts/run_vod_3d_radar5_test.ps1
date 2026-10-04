param(
    [int]$TrainSamples = 8,
    [int]$ValSamples = 2,
    [int]$Epochs = 1,
    [int]$BatchSize = 1,
    [string]$Python = 'C:\Users\gianl\miniconda3\envs\dl-lab-anomaly-detection\python.exe',
    [string]$DatasetRoot = 'C:\Users\gianl\Desktop\Thesis\View-Of-Delft dataset'
)

$ErrorActionPreference = 'Stop'
if ($TrainSamples -lt 1 -or $ValSamples -lt 1 -or $Epochs -lt 1 -or $BatchSize -lt 1) {
    throw 'TrainSamples, ValSamples, Epochs, and BatchSize must all be positive.'
}

$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$samples = Join-Path $DatasetRoot 'reconstruction_vod_radar5_unique'
$radar = Join-Path $DatasetRoot 'radar5_pointpillars_cache'
$cache = Join-Path $repo ("outputs_local\vod_3d_radar5_test\supervision_t{0}_v{1}" -f $TrainSamples, $ValSamples)
$run = Join-Path $repo ("outputs_local\vod_3d_radar5_test\train_{0}" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$config = Join-Path $repo 'configs\voxelization_3d.json'

foreach ($path in @($Python, $config, (Join-Path $samples 'train'), (Join-Path $samples 'val'), (Join-Path $radar 'train'), (Join-Path $radar 'val'))) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path is missing: $path" }
}

$env:PYTHONPATH = "$repo;$env:PYTHONPATH"
Push-Location $repo
try {
    foreach ($split in @('train', 'val')) {
        $limit = if ($split -eq 'train') { $TrainSamples } else { $ValSamples }
        Write-Host "Caching $limit $split samples (five-frame radar)..."
        & $Python -u -m scripts.cache_sparse_voxel_supervision `
            --data-root $samples --radar-root $radar --cache-root $cache `
            --config $config --split $split --limit-samples $limit --workers 1
        if ($LASTEXITCODE -ne 0) { throw "The $split supervision-cache build failed (exit $LASTEXITCODE)." }
    }

    Write-Host "Starting 3D training; run directory: $run"
    & $Python -u -m scripts.train_sparse_voxel_diffusion `
        --samples-root $samples --radar-root $radar --supervision-cache-root $cache `
        --cached-samples-only --train-fraction 1 --val-fraction 1 `
        --voxel-config $config --output-root $run `
        --epochs $Epochs --batch-size $BatchSize --hidden-dim 32 --num-blocks 2 `
        --lambda-diffusion 1 --lambda-bce 1 --lambda-chamfer 0.5 `
        --chamfer-chunk-size 64 --chamfer-max-points 64 --device cuda
    if ($LASTEXITCODE -ne 0) { throw "3D training failed (exit $LASTEXITCODE)." }
    Write-Host "Finished. Metrics: $(Join-Path $run 'history.jsonl')"
    Write-Host "Checkpoint: $(Join-Path $run 'last_checkpoint.pt')"
}
finally {
    Pop-Location
}
