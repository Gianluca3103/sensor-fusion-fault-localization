param(
    [ValidateSet("v1", "v2")]
    [string]$Radar = "v1",
    [string]$Repo = "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo",
    [string]$Python = "C:\Users\gianl\miniconda3\python.exe",
    [string]$OutputBase = "C:\Users\gianl\Desktop\Thesis\sensor_fusion_outputs",
    [string]$LidarName = "grid_reliability_10k_grid320_no_rain_snow",
    [int]$Epochs = 100,
    [int]$MetricsEvery = 10,
    [int]$BatchSize = 4,
    [int]$Workers = 1,
    [double]$LearningRate = 5e-5,
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"

$Lidar = Join-Path $OutputBase $LidarName
if ($Radar -eq "v1") {
    $RadarRoot = Join-Path $OutputBase "radar_v1_cache_10k_grid320_no_rain_snow_30ms"
    $RunRoot = Join-Path $OutputBase "pfs_radar_v1_10k_no_rain_snow_30ms_b${BatchSize}_lr${LearningRate}_metrics${MetricsEvery}"
    $RadarArgs = @(
        "--radar-frame-count", "20",
        "--require-full-radar-stack"
    )
} else {
    $RadarRoot = Join-Path $OutputBase "radar_v2_cache_10k_grid320_no_rain_snow_30ms_interp_v5"
    $RunRoot = Join-Path $OutputBase "pfs_radar_v2_10k_no_rain_snow_30ms_interp_b${BatchSize}_lr${LearningRate}_metrics${MetricsEvery}"
    $RadarArgs = @()
}

foreach ($PathValue in @($Repo, $Python, $Lidar, $RadarRoot)) {
    if (-not (Test-Path $PathValue)) {
        throw "Missing path: $PathValue"
    }
}

Set-Location $Repo

& $Python -u (Join-Path $Repo "PFS_Radar\train_pfs_radar.py") `
    --train-root (Join-Path $Lidar "train") `
    --val-root (Join-Path $Lidar "val") `
    --radar-root $RadarRoot `
    --output-root $RunRoot `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --num-workers $Workers `
    --base-channels 16 `
    --dropout 0.15 `
    --learning-rate $LearningRate `
    --min-learning-rate 1e-6 `
    --warmup-epochs 10 `
    --weight-decay 2e-3 `
    --stability-weight 0.05 `
    --pfs-reliability-weight 0.10 `
    --localization-loss-weight 0.25 `
    --false-positive-weight 0.65 `
    --grad-clip 1.0 `
    --resize-height 320 `
    --resize-width 320 `
    --grid-size 320 `
    --metric-threshold 0.15 `
    --metric-grid-size 320 `
    --metrics-every $MetricsEvery `
    --localization-tolerance-m 0.20 `
    --max-radar-delta-ms 30 `
    --radar-max-abs-velocity 30 `
    --early-stop-patience 20 `
    --device $Device `
    @RadarArgs

exit $LASTEXITCODE
