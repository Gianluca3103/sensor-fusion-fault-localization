#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-/home/arrubuntu20/anaconda3/envs/sensor-fusion/bin/python}"
REPO_ROOT="${REPO_ROOT:-/mnt/3D10B36523559581/Gianluca/sensor-fusion-fault-localization-100k}"
HERCULES_ROOT="${HERCULES_ROOT:-/mnt/3D10B36523559581/HeRCULES}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/3D10B36523559581/Gianluca/sensor_fusion_outputs}"

DATASET_ROOT="${DATASET_ROOT:-$OUTPUT_BASE/grid_reliability_100k_grid320_id_v1}"
RADAR_ROOT="${RADAR_ROOT:-$OUTPUT_BASE/radar_cache_100k_stack20}"
RUN_ROOT="${RUN_ROOT:-$OUTPUT_BASE/pfs_radar_100k_stack20_localigned}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_BASE/logs}"

GEN_WORKERS="${GEN_WORKERS:-24}"
CACHE_WORKERS="${CACHE_WORKERS:-16}"
TRAIN_WORKERS="${TRAIN_WORKERS:-12}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-48}"

mkdir -p "$DATASET_ROOT" "$RADAR_ROOT" "$RUN_ROOT" "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/100k_stack20_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "FAILED at line $LINENO. See $LOG_FILE"' ERR

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

test -x "$PYTHON"
test -d "$REPO_ROOT"
test -d "$HERCULES_ROOT"
cd "$REPO_ROOT"

available_kb="$(df -Pk "$OUTPUT_BASE" | awk 'NR==2 {print $4}')"
required_kb="$((100 * 1024 * 1024))"
if (( available_kb < required_kb )); then
  echo "At least 100 GiB free is required; only $((available_kb / 1024 / 1024)) GiB is available."
  exit 1
fi

generate_split() {
  local split="$1"
  local count="$2"
  local seed="$3"
  "$PYTHON" Fault_Localization_Model/create_grid_reliability_heatmaps.py \
    --data-root "$HERCULES_ROOT" \
    --injector-root "$REPO_ROOT/Weather_Injector/3D_Corruptions_AD" \
    --fog-root "$REPO_ROOT/Weather_Injector/LiDAR_fog_sim" \
    --output-root "$DATASET_ROOT/$split" \
    --all-scenes \
    --temporal-split "$split" \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --num-samples "$count" \
    --fault-plan fog_sim:5 rain_sim:5 snow_sim:5 old_laser_degradation:0 fov_filter:1 \
    --grid-size 320 \
    --x-min 0 --x-max 64 \
    --y-min -32 --y-max 32 \
    --resolution 0.20 \
    --movement-tolerance-m 0.05 \
    --num-workers "$GEN_WORKERS" \
    --no-previews \
    --seed "$seed"

  local actual
  actual="$(find "$DATASET_ROOT/$split" -maxdepth 1 -type f -name '*.npz' | wc -l)"
  if [[ "$actual" -ne "$count" ]]; then
    echo "$split contains $actual samples; expected $count."
    exit 1
  fi
}

echo "Generating chronological 70/15/15 reliability-map splits..."
generate_split train 70000 42
generate_split val 15000 43
generate_split test 15000 44

echo "Building strict, pose-aligned 20-frame radar cache..."
"$PYTHON" PFS_Radar/prepare_radar_cache.py \
  --dataset-root "$DATASET_ROOT" \
  --hercules-root "$HERCULES_ROOT" \
  --output-root "$RADAR_ROOT" \
  --radar-frame-count 20 \
  --require-full-stack \
  --max-delta-ms 30 \
  --num-workers "$CACHE_WORKERS"

resume_args=()
if [[ -f "$RUN_ROOT/checkpoints/last_checkpoint.pt" ]]; then
  resume_args=(--resume "$RUN_ROOT/checkpoints/last_checkpoint.pt")
  echo "Resuming training from $RUN_ROOT/checkpoints/last_checkpoint.pt"
fi

echo "Starting RTX 4090 training..."
"$PYTHON" PFS_Radar/train_pfs_radar.py \
  --train-root "$DATASET_ROOT/train" \
  --val-root "$DATASET_ROOT/val" \
  --radar-root "$RADAR_ROOT" \
  --output-root "$RUN_ROOT" \
  "${resume_args[@]}" \
  --epochs 150 \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --num-workers "$TRAIN_WORKERS" \
  --base-channels 16 \
  --dropout 0.15 \
  --learning-rate 1e-4 \
  --min-learning-rate 1e-6 \
  --warmup-epochs 5 \
  --weight-decay 2e-3 \
  --stability-weight 0.05 \
  --pfs-reliability-weight 0.10 \
  --localization-loss-weight 0.25 \
  --false-positive-weight 0.70 \
  --localization-radius-cells 1 \
  --grad-clip 1.0 \
  --early-stop-patience 20 \
  --resize-height 320 \
  --resize-width 320 \
  --grid-size 320 \
  --metric-grid-size 320 \
  --metric-threshold 0.15 \
  --metrics-every 5 \
  --localization-tolerance-m 0.20 \
  --target-fault-threshold 0.0 \
  --device cuda

echo "Completed. Best checkpoints:"
ls -lh "$RUN_ROOT/checkpoints"
echo "Log: $LOG_FILE"
