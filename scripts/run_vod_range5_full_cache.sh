#!/usr/bin/env bash
# Build a split-isolated five-radar VoD range-view cache for every official split.
set -euo pipefail

if (( $# < 4 || $# > 5 )); then
  echo "Usage: bash scripts/run_vod_range5_full_cache.sh REPO VOD_PUBLIC OUTPUT_ROOT PYTHON [RADAR_WORKERS]" >&2
  exit 2
fi

REPO=$1
VOD=$2
OUT=$3
PYTHON=$4
RADAR_WORKERS=${5:-2}

test -f "$REPO/scripts/create_vod_range_view_dataset.py" || { echo "Missing updated repository code" >&2; exit 1; }
test -x "$PYTHON" || { echo "Python is not executable: $PYTHON" >&2; exit 1; }
for split in train val test; do
  test -f "$VOD/lidar/ImageSets/$split.txt" || { echo "Missing official split: $split" >&2; exit 1; }
  first_id=$(sed -n '1p' "$VOD/lidar/ImageSets/$split.txt" | tr -d '\r')
  if [[ -f "$VOD/lidar/training/velodyne/$first_id.bin" ]]; then
    partition=training
  elif [[ -f "$VOD/lidar/testing/velodyne/$first_id.bin" ]]; then
    partition=testing
  else
    echo "Cannot locate $split frame $first_id in training or testing" >&2
    exit 1
  fi
  test -d "$VOD/lidar/$partition/velodyne" || { echo "Missing $partition LiDAR" >&2; exit 1; }
  test -d "$VOD/radar/$partition/velodyne" || { echo "Missing $partition radar" >&2; exit 1; }
  test -d "$VOD/lidar/$partition/pose" || { echo "Missing $partition LiDAR poses" >&2; exit 1; }
done

mkdir -p "$OUT"
cd "$REPO"
for split in train val test; do
  echo "[$split] Generating split-isolated five-frame radar scans..."
  "$PYTHON" -u -m scripts.generate_vod_accumulated_radar \
    --vod-root "$VOD" --stack-sizes 5 --split "$split" \
    --output-suffix rangeview --isolate-split-history \
    --num-workers "$RADAR_WORKERS"

  echo "[$split] Generating full-scan faults and lean aligned radar cache..."
  "$PYTHON" -u -m scripts.create_vod_range_view_dataset \
    --vod-root "$VOD" --radar-cache-root "$OUT/radar" \
    --output-root "$OUT/samples" --split "$split" \
    --radar-variant radar_5frames_rangeview --min-radar-frames 1
done

"$PYTHON" -u -m scripts.build_vod_range_geometry \
  --data-root "$OUT/samples" --output "$OUT/angular_geometry.json"

echo "Complete. Split summaries: $OUT/samples/range_view_{train,val,test}_summary.json"
echo "Samples: $OUT/samples/{train,val,test} | aligned radar: $OUT/radar/{train,val,test}"
