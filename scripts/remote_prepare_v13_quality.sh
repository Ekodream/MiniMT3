#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/data/MiniMT3/MiniMT3}
PYTHON_BIN=${PYTHON_BIN:-/data/app/dp2.2.11/envs/MiniMT3/bin/python}
STAMP=$(date +%Y%m%d_%H%M%S)
SESSION=${SESSION:-MiniMT3_v13_large_${STAMP}}

cd "${PROJECT_DIR}"
mkdir -p data/cache log outputs

if [[ ! -s data/cache/amt_train_8s_uniform64perpiece_v13_large.json ]]; then
  "${PYTHON_BIN}" scripts/build_amt_manifest.py \
    --index data/cache/maestro_index.json \
    --split train \
    --out data/cache/amt_train_8s_uniform64perpiece_v13_large.json \
    --clip_seconds 8 \
    --sampling uniform \
    --max_clips_per_piece 64 \
    --seed 173
fi

if [[ ! -s data/cache/amt_val_8s_s8_calib512_v13.json ]]; then
  "${PYTHON_BIN}" scripts/build_amt_manifest.py \
    --index data/cache/maestro_index.json \
    --split validation \
    --out data/cache/amt_val_8s_s8_calib512_v13.json \
    --clip_seconds 8 \
    --stride_seconds 8 \
    --max_clips 512 \
    --seed 173
fi

if [[ ! -s data/cache/amt_val_score_quality_v13.json ]]; then
  "${PYTHON_BIN}" scripts/build_score_quality_manifest.py \
    --index data/cache/maestro_index.json \
    --split validation \
    --out data/cache/amt_val_score_quality_v13.json \
    --clip_seconds 30 \
    --stride_seconds 30 \
    --max_clips 30
fi

"${PYTHON_BIN}" scripts/amt_model_report.py \
  --config configs/train_amt_v12_crnn_bytedance.yaml \
  --config configs/train_amt_v13_wide.yaml \
  --config configs/train_amt_v13_large_finetune.yaml \
  --config configs/train_amt_v14_mid.yaml \
  --json_out outputs/amt_model_report_v13_quality.json

cat <<EOF
Prepared dense-AMT quality manifests and model report.

Train v13 large fine-tune explicitly with:
  tmux new-session -d -s ${SESSION} "cd ${PROJECT_DIR} && ${PYTHON_BIN} -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_amt.py --config configs/train_amt_v13_large_finetune.yaml > log/v13_large_finetune_${STAMP}.log 2>&1"

Set LAUNCH_TRAIN=1 to launch that command from this script.
EOF

if [[ "${LAUNCH_TRAIN:-0}" == "1" ]]; then
  tmux new-session -d -s "${SESSION}" \
    "cd ${PROJECT_DIR} && ${PYTHON_BIN} -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_amt.py --config configs/train_amt_v13_large_finetune.yaml > log/v13_large_finetune_${STAMP}.log 2>&1"
  echo "started ${SESSION}"
  echo "log: ${PROJECT_DIR}/log/v13_large_finetune_${STAMP}.log"
fi
