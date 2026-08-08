# RadarV2 for K-Radar

`PFS_Radar_v2` now builds the existing four-channel RadarV2 representation
from K-Radar rather than HeRCULES Continental radar.

## Input

Use the tensor-derived `from_rdr_cube_xyz/pc10p` files. Each
`rpc_<index>.npy` row contains:

`x, y, z, power, Doppler, range, azimuth, elevation, range index, azimuth index, elevation index`

Do not substitute `sparse_radar_tensor_wide_range`: its rows contain only
`x, y, z, power`, so it cannot produce RadarV2's Doppler-based dynamic
channels.

Frame timestamps and radar/LiDAR pairings come from `info_label`. Each cache
entry uses only that paired radar frame. KITTI-format poses in
`resources/odometry/gt` are used only to estimate ego velocity for Doppler
compensation. `calib_radar_lidar.txt` maps the frame into LiDAR coordinates.

LiDAR frames, labels, and calibration files are organized under
`<K-Radar-root>/lidar/<sequence>/`; tensor-derived radar points remain under
`<K-Radar-root>/radar/pc10p/<sequence>/`.

K-Radar Doppler is ambiguous outside an interval of approximately 3.865 m/s.
The cache builder wraps the ego-motion residual into that interval before
static/dynamic classification. Dynamic clustering is additionally restricted
to the configured high-power quantile because pc10p contains roughly 100,000
tensor cells per frame rather than a small conventional radar point cloud.

## Four output channels

The downstream contract remains `[4, H, W]`:

1. static occupancy
2. robust log-normalized power
3. dynamic speed
4. robust upper height

Power is the maximum point power in each BEV cell, log-scaled
against the 99th percentile of occupied cells. Upper height is the per-cell
90th percentile of physically plausible aligned radar heights in `[-3, 5]`
metres, normalized into `[0, 1]`. Returns outside that vertical range are
excluded from the height statistic, and empty cells remain zero.

At the default ranges and resolution the cache shape is `[4, 320, 320]`.
Cache files are written as `<output-root>/<sequence>/<radar-index>.npz`.

## Build the cache

PowerShell:

```powershell
$PYTHON = "C:\path\to\python.exe"
$KRADAR = "C:\Users\gianl\Desktop\Thesis\K-Radar_Data"
$REPO = "C:\Users\gianl\Desktop\Thesis\Sensor-Fusion_Final_Model_Repo"

Set-Location $REPO
& $PYTHON -m PFS_Radar_v2.prepare_radar_cache `
  --kradar-root $KRADAR `
  --radar-point-root "$KRADAR\radar\pc10p" `
  --odometry-root "$KRADAR\support\official_k_radar\resources\odometry" `
  --output-root "$KRADAR\radar_v2_cache" `
  --num-workers 4
```

For a one-frame smoke test, add `--max-pending-frames 1 --num-workers 1`.
The builder is resumable: compatible cache entries are skipped on later runs.

The cache generator has no temporal accumulation or frame-stacking options.
The default BEV geometry is `x=[0,64)`, `y=[-32,32)`, resolution `0.2 m`.
