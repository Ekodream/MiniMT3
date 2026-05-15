#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/data/MiniMT3/MiniMT3
CONDA_BIN=/data/app/dp2.2.11/bin/conda
ENV_NAME=MiniMT3
CONFIG=configs/train_8gpu_shared.yaml
LOG_DIR="$PROJECT_DIR/outputs/logs"
RUN_DIR="$PROJECT_DIR/outputs/ckpt_8gpu_shared"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_8gpu_shared_$STAMP.log"

mkdir -p "$LOG_DIR" "$RUN_DIR"
cd "$PROJECT_DIR"

export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MASTER_PORT="${MASTER_PORT:-29615}"

nohup "$CONDA_BIN" run -n "$ENV_NAME" torchrun \
  --nproc_per_node=8 \
  --master_port="$MASTER_PORT" \
  scripts/train.py \
  --config "$CONFIG" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "started_pid=$PID"
echo "log_file=$LOG_FILE"
