$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo"
$Python = "C:\Users\gianl\miniconda3\python.exe"
$DataRoot = "C:\Users\gianl\Desktop\Thesis\HerculesFiles\Data"
$DatasetRoot = Join-Path $RepoRoot "Fault_Localization_Model\grid_reliability_7500_fog_s3_x64_y32"
$RunRoot = Join-Path $RepoRoot "PFS\runs\pfs_7500_fog_s3_x64_y32"
$LogRoot = Join-Path $RepoRoot "PFS\runs"
$TranscriptPath = Join-Path $LogRoot "pfs_7500_fog_s3_x64_y32_run.transcript.log"

Set-Location $RepoRoot
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
Start-Transcript -Path $TranscriptPath -Append | Out-Null

try {

Write-Host "=== PFS 7500-sample run started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "Dataset root: $DatasetRoot"
Write-Host "Run root:     $RunRoot"
Write-Host "Fault plan:   fog_sim:S3, rain_sim:S5, snow_sim:S5, old_laser_degradation:S0, fov_filter:S1"
Write-Host "Data mode:    all Hercules scenes under $DataRoot"

& $Python Fault_Localization_Model\create_grid_reliability_heatmaps.py `
  --data-root $DataRoot `
  --all-scenes `
  --output-root $DatasetRoot `
  --num-samples 7500 `
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

Write-Host "=== Dataset generation finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "Starting PFS training..."

& $Python PFS\train_pfs_reliability_map.py `
  --dataset-root $DatasetRoot `
  --output-root $RunRoot `
  --epochs 100 `
  --batch-size 24 `
  --base-channels 16 `
  --dropout 0.15 `
  --learning-rate 3e-4 `
  --weight-decay 1e-3 `
  --loss-mode stable `
  --grad-clip 1.0 `
  --early-stop-patience 12 `
  --resize-height 320 `
  --resize-width 320 `
  --grid-size 100 `
  --stability-weight 0.15 `
  --pfs-reliability-weight 0.10 `
  --max-val-images 24 `
  --num-workers 4 `
  --device cuda

Write-Host "=== PFS training finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
}
finally {
  Stop-Transcript | Out-Null
}
