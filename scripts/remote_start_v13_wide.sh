#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=/data/MiniMT3/MiniMT3
PYTHON_BIN=/data/app/dp2.2.11/envs/MiniMT3/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
SESSION="MiniMT3_v13_wide_${STAMP}"

cd "${PROJECT_DIR}"
mkdir -p data/cache log outputs

if [[ ! -s data/cache/amt_train_8s_uniform2048_v13.json ]]; then
  "${PYTHON_BIN}" scripts/build_amt_manifest.py \
    --index data/cache/maestro_index.json \
    --split train \
    --out data/cache/amt_train_8s_uniform2048_v13.json \
    --clip_seconds 8 \
    --sampling uniform \
    --max_clips 2048 \
    --max_clips_per_piece 8 \
    --seed 173
fi

if [[ ! -s data/cache/amt_val_8s_s8_v13.json ]]; then
  "${PYTHON_BIN}" scripts/build_amt_manifest.py \
    --index data/cache/maestro_index.json \
    --split validation \
    --out data/cache/amt_val_8s_s8_v13.json \
    --clip_seconds 8 \
    --stride_seconds 8 \
    --max_clips 128 \
    --seed 173
fi

"${PYTHON_BIN}" scripts/train_amt.py \
  --config configs/train_amt_v13_wide_smoke.yaml \
  2>&1 | tee "log/v13_wide_smoke_${STAMP}.log"

tmux new-session -d -s "${SESSION}" \
  "cd ${PROJECT_DIR} && ${PYTHON_BIN} -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_amt.py --config configs/train_amt_v13_wide.yaml > log/v13_wide_train_${STAMP}.log 2>&1"

echo "started ${SESSION}"
echo "log: ${PROJECT_DIR}/log/v13_wide_train_${STAMP}.log"
