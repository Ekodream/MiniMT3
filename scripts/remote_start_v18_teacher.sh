#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/data/MiniMT3/MiniMT3}
PYTHON_BIN=${PYTHON_BIN:-/data/app/dp2.2.11/envs/MiniMT3/bin/python}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
SESSION=${SESSION:-MiniMT3_v18_teacher_calibrated_${STAMP}}
CONFIG=${CONFIG:-configs/train_amt_v18_teacher_calibrated.yaml}
BASE_MANIFEST=${BASE_MANIFEST:-data/cache/amt_train_8s_uniform64perpiece_v13_large.json}
SOURCE_HARDMIX=${SOURCE_HARDMIX:-data/cache/amt_train_8s_hardmix_v16.json}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-data/cache/amt_train_8s_mix70hard30_v18.json}
VAL_MANIFEST=${VAL_MANIFEST:-data/cache/amt_val_8s_s8_calib512_v13.json}
INIT_CKPT=${INIT_CKPT:-outputs/ckpt_amt_v15_xlarge_duration/best.pt}
TEACHER_DIR=${TEACHER_DIR:-outputs/teacher_bytedance/train_v18_hard4096}
LOG_NAME=${LOG_NAME:-v18_teacher_calibrated_${STAMP}.log}

cd "${PROJECT_DIR}"
mkdir -p log

if [[ ! -s "${TRAIN_MANIFEST}" ]]; then
  echo "building ${TRAIN_MANIFEST}"
  if [[ -s "${SOURCE_HARDMIX}" ]]; then
    "${PYTHON_BIN}" scripts/build_amt_mix_manifest.py \
      --source_manifest "${SOURCE_HARDMIX}" \
      --out "${TRAIN_MANIFEST}" \
      --hard_fraction "${HARD_FRACTION:-0.30}" \
      --max_hard_clips "${MAX_HARD_CLIPS:-26386}" \
      --seed "${MANIFEST_SEED:-186}"
  else
    "${PYTHON_BIN}" scripts/build_amt_hard_manifest.py \
      --base_manifest "${BASE_MANIFEST}" \
      --out "${TRAIN_MANIFEST}" \
      --max_hard_clips "${MAX_HARD_CLIPS:-26386}" \
      --max_per_piece "${MAX_PER_PIECE:-64}" \
      --max_per_category "${MAX_PER_CATEGORY:-6000}" \
      --seed "${MANIFEST_SEED:-186}"
  fi
fi

if [[ ! -s "${VAL_MANIFEST}" ]]; then
  echo "missing validation manifest: ${VAL_MANIFEST}" >&2
  exit 2
fi
if [[ ! -s "${INIT_CKPT}" ]]; then
  echo "missing init checkpoint: ${INIT_CKPT}" >&2
  exit 2
fi
if [[ ! -d "${TEACHER_DIR}" ]]; then
  echo "teacher dir not found yet: ${TEACHER_DIR} (training will run without teacher pseudo labels until files exist)"
fi

"${PYTHON_BIN}" scripts/amt_model_report.py --config "${CONFIG}"

echo "Prepared v18 teacher-calibrated training."
echo "  project:  ${PROJECT_DIR}"
echo "  config:   ${CONFIG}"
echo "  manifest: ${TRAIN_MANIFEST}"
echo "  init:     ${INIT_CKPT}"
echo "  teacher:  ${TEACHER_DIR}"
echo "  session:  ${SESSION}"
echo "Set LAUNCH_TRAIN=1 to start tmux training."

if [[ "${LAUNCH_TRAIN:-0}" == "1" ]]; then
  tmux new-session -d -s "${SESSION}" \
    "cd ${PROJECT_DIR} && ${PYTHON_BIN} -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_amt.py --config ${CONFIG} > log/${LOG_NAME} 2>&1"
  echo "started ${SESSION}"
  echo "log: ${PROJECT_DIR}/log/${LOG_NAME}"
fi
