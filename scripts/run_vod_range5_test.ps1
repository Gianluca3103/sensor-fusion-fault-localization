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
    throw 'TrainSamples, ValSamples, Epochs, and BatchSize must be positive.'
}

$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$vod = Join-Path $DatasetRoot 'view_of_delft_PUBLIC'
$radar = Join-Path $DatasetRoot 'radar5_pointpillars_cache'
$base = Join-Path $repo 'outputs_local\vod_range5_test'
$data = Join-Path $base ("full_scan_generated_t{0}_v{1}" -f $TrainSamples, $ValSamples)
$geometry = Join-Path $data 'angular_geometry.json'
$run = Join-Path $base ("train_{0}" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

foreach ($path in @($Python, $vod,
                   (Join-Path $radar 'train'), (Join-Path $radar 'val'))) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required path is missing: $path" }
}

$env:PYTHONPATH = "$repo;$env:PYTHONPATH"
Push-Location $repo
try {
    foreach ($split in @('train', 'val')) {
        $limit = if ($split -eq 'train') { $TrainSamples } else { $ValSamples }
        Write-Host "Preparing $limit full-scan $split samples with exactly five radar frames..."
        & $Python -u -m scripts.create_vod_range_view_dataset `
            --vod-root $vod --radar-cache-root $radar --output-root $data `
            --split $split --seed 42 --limit $limit
        if ($LASTEXITCODE -ne 0) { throw "$split full-scan preparation failed (exit $LASTEXITCODE)." }
    }

    & $Python -u -m scripts.build_vod_range_geometry --data-root $data --output $geometry
    if ($LASTEXITCODE -ne 0) { throw "Angular geometry preparation failed (exit $LASTEXITCODE)." }

    Write-Host "Policy: full forward range view; no fault selector; preserve every observed LiDAR point."
    Write-Host "Starting VoD range-view training; run directory: $run"
    & $Python -u -m scripts.train_range_view_reconstruction `
        --data-root $data --radar-root $radar --geometry $geometry --output-root $run `
        --train-limit $TrainSamples --val-limit $ValSamples `
        --epochs $Epochs --batch-size $BatchSize --hidden-channels 8 `
        --add-threshold 0.5 --delete-threshold 0.999 `
        --false-delete-penalty 30 --missed-delete-penalty 1 `
        --num-workers 0 --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Range-view training failed (exit $LASTEXITCODE)." }
    $resolved = Get-Content -LiteralPath (Join-Path $run 'resolved_config.json') -Raw | ConvertFrom-Json
    if ($resolved.representation -ne 'range_view' -or
        $resolved.arguments.use_fault_map_conditioning -or
        $null -ne $resolved.arguments.fault_map_root -or
        $resolved.merge.allow_original_deletion -or
        -not $resolved.merge.forward_only) {
        throw 'Training policy audit failed: expected range view, no fault selector, and original-preserving forward merge.'
    }
    $validation = Get-Content -LiteralPath (Join-Path $run "val_epoch_$Epochs.json") -Raw | ConvertFrom-Json
    foreach ($sample in $validation.samples) {
        if ($sample.deleted_original_count -ne 0) {
            throw "Validation policy audit failed: an original LiDAR point was deleted in $($sample.sample)."
        }
    }
    Write-Host "Policy audit passed: no fault selector and zero original LiDAR deletions in validation."
    Write-Host "Finished. Summary: $(Join-Path $run 'summary.csv')"
    Write-Host "Checkpoint: $(Join-Path $run 'last_checkpoint.pt')"
}
finally {
    Pop-Location
}
