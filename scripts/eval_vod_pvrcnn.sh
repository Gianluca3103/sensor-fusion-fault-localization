#!/usr/bin/env bash
# Evaluate one clean-trained PV-RCNN checkpoint on matched VoD val inputs.
set -euo pipefail

if (( $# != 5 )); then
  echo "Usage: bash scripts/eval_vod_pvrcnn.sh OPENPCDET_ROOT EXPORT_ROOT MODE CHECKPOINT PYTHON" >&2
  exit 2
fi
PCDET=$1
EXPORT=$2
MODE=$3
CHECKPOINT=$4
PYTHON=$5

[[ "$MODE" == lidar || "$MODE" == lidar_radar ]] || { echo "MODE must be lidar or lidar_radar" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; exit 1; }
for condition in clean faulty reconstructed; do
  root="$EXPORT/$MODE/$condition"
  cfg="cfgs/kitti_models/vod_pvrcnn_${MODE}_${condition}.yaml"
  [[ -f "$root/export_manifest.json" && -f "$root/kitti_infos_val.pkl" && -f "$PCDET/tools/$cfg" ]] || {
    echo "Incomplete $condition export, info, or config" >&2; exit 1;
  }
done

cd "$PCDET/tools"
for condition in clean faulty reconstructed; do
  cfg="cfgs/kitti_models/vod_pvrcnn_${MODE}_${condition}.yaml"
  echo "Evaluating $MODE/$condition on the same validation frame IDs..."
  "$PYTHON" test.py --cfg_file "$cfg" --ckpt "$CHECKPOINT" \
    --batch_size 1 --workers 2 --extra_tag vod_clean_trained \
    --eval_tag "$condition" --save_to_file
done
