#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/data/MiniMT3/MiniMT3}
PYTHON_BIN=${PYTHON_BIN:-/data/app/dp2.2.11/envs/MiniMT3/bin/python}
PROFILE=${PROFILE:-f1_duration}
STAMP=$(date +%Y%m%d_%H%M%S)
SESSION=${SESSION:-MiniMT3_v15_${PROFILE}_${STAMP}}

case "${PROFILE}" in
  f1_duration)
    CONFIG=configs/train_amt_v15_f1_duration.yaml
    LOG_NAME=v15_f1_duration_${STAMP}.log
    ;;
  xlarge_duration)
    CONFIG=configs/train_amt_v15_xlarge_duration.yaml
    LOG_NAME=v15_xlarge_duration_${STAMP}.log
    ;;
  *)
    echo "Unknown PROFILE=${PROFILE}; use f1_duration or xlarge_duration." >&2
    exit 2
    ;;
esac

cd "${PROJECT_DIR}"
mkdir -p data/cache log outputs

if [[ ! -s data/cache/amt_train_8s_uniform64perpiece_v13_large.json ]] || [[ ! -s data/cache/amt_val_8s_s8_calib512_v13.json ]]; then
  bash scripts/remote_prepare_v13_quality.sh
fi

"${PYTHON_BIN}" scripts/amt_model_report.py \
  --config configs/train_amt_v13_large_finetune.yaml \
  --config configs/train_amt_v15_f1_duration.yaml \
  --config configs/train_amt_v15_xlarge_duration.yaml \
  --json_out outputs/amt_model_report_v15_f1.json

echo "GPU state before launch:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
echo
echo "Ready to launch ${PROFILE}:"
echo "  tmux new-session -d -s ${SESSION} \"cd ${PROJECT_DIR} && ${PYTHON_BIN} -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_amt.py --config ${CONFIG} > log/${LOG_NAME} 2>&1\""
echo
echo "Set LAUNCH_TRAIN=1 to launch. Set FORCE_LAUNCH=1 only after confirming GPUs are free enough."

if [[ "${LAUNCH_TRAIN:-0}" == "1" ]]; then
  if [[ "${FORCE_LAUNCH:-0}" != "1" ]]; then
    active_gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^$/d' | wc -l)
    if [[ "${active_gpu_processes}" -gt 0 ]]; then
      echo "Refusing to launch: nvidia-smi reports ${active_gpu_processes} active GPU processes. Set FORCE_LAUNCH=1 to override." >&2
      exit 3
    fi
  fi
  tmux new-session -d -s "${SESSION}" \
    "cd ${PROJECT_DIR} && ${PYTHON_BIN} -m torch.distributed.run --standalone --nproc_per_node=8 scripts/train_amt.py --config ${CONFIG} > log/${LOG_NAME} 2>&1"
  echo "started ${SESSION}"
  echo "log: ${PROJECT_DIR}/log/${LOG_NAME}"
fi
