#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/data/MiniMT3/MiniMT3
PYTHON_BIN=/data/app/dp2.2.11/envs/MiniMT3/bin/python
CONFIG=configs/train_8gpu_relative.yaml
LOG_DIR="$PROJECT_DIR/outputs/logs"
RUN_DIR="$PROJECT_DIR/outputs/ckpt_8gpu_relative"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="MiniMT3_relative_${STAMP}"
LOG_FILE="$LOG_DIR/train_8gpu_relative_${STAMP}.log"

mkdir -p "$LOG_DIR" "$RUN_DIR"
cd "$PROJECT_DIR"

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MASTER_PORT="${MASTER_PORT:-29627}"

tmux new-session -d -s "$SESSION" \
  "$PYTHON_BIN -m torch.distributed.run --nproc_per_node=8 --master_port=$MASTER_PORT scripts/train.py --config $CONFIG 2>&1 | tee -a $LOG_FILE"

echo "session=$SESSION"
echo "log_file=$LOG_FILE"
