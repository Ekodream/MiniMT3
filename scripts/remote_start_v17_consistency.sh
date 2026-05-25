#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/data/MiniMT3/MiniMT3}
PYTHON_BIN=${PYTHON_BIN:-/data/app/dp2.2.11/envs/MiniMT3/bin/python}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
SESSION=${SESSION:-MiniMT3_v17_xlarge_consistency_${STAMP}}
CONFIG=${CONFIG:-configs/train_amt_v17_xlarge_consistency.yaml}
MANIFEST=${MANIFEST:-data/cache/amt_train_8s_hardmix_v16.json}
BASE_MANIFEST=${BASE_MANIFEST:-data/cache/amt_train_8s_uniform64perpiece_v13_large.json}
INIT_CKPT=${INIT_CKPT:-outputs/ckpt_amt_v16_xlarge_hard_duration/best.pt}
LOG_NAME=${LOG_NAME:-v17_xlarge_consistency_${STAMP}.log}

cd "${PROJECT_DIR}"

if [[ ! -s "${MANIFEST}" ]]; then
  "${PYTHON_BIN}" scripts/build_amt_hard_manifest.py \
    --index data/cache/maestro_index.json \
    --split train \
    --base_manifest "${BASE_MANIFEST}" \
    --out "${MANIFEST}" \
    --clip_seconds 8 \
    --stride_seconds 4 \
    --max_hard_clips "${MAX_HARD_CLIPS:-48000}" \
    --max_per_piece "${MAX_PER_PIECE:-80}" \
    --max_per_category "${MAX_PER_CATEGORY:-12000}"
fi

if [[ ! -s "${INIT_CKPT}" ]]; then
  echo "missing init checkpoint: ${INIT_CKPT}" >&2
  exit 2
fi

"${PYTHON_BIN}" scripts/amt_model_report.py --config "${CONFIG}" || true

echo "Prepared v17 xlarge consistency training."
echo "  project:  ${PROJECT_DIR}"
echo "  config:   ${CONFIG}"
echo "  manifest: ${MANIFEST}"
echo "  init:     ${INIT_CKPT}"
echo "  session:  ${SESSION}"
echo "Set LAUNCH_TRAIN=1 to start tmux training."

if [[ "${LAUNCH_TRAIN:-0}" == "1" ]]; then
  mkdir -p log
  tmux new-session -d -s "${SESSION}" \
    "cd ${PROJECT_DIR} && ${PYTHON_BIN} -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_amt.py --config ${CONFIG} > log/${LOG_NAME} 2>&1"
  echo "started ${SESSION}"
  echo "log: ${PROJECT_DIR}/log/${LOG_NAME}"
fi
