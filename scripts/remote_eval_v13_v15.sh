#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/data/MiniMT3/MiniMT3}
PYTHON_BIN=${PYTHON_BIN:-/data/app/dp2.2.11/envs/MiniMT3/bin/python}
GPU_ID=${GPU_ID:-0}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
SESSION=${SESSION:-MiniMT3_eval_v13_v15_${STAMP}}

VAL_MANIFEST=${VAL_MANIFEST:-data/cache/amt_val_8s_s8_calib512_v13.json}
ITEMS=${ITEMS:-512}
V13_CKPT=${V13_CKPT:-outputs/ckpt_amt_v13_large_finetune/best.pt}
V15_CKPT=${V15_CKPT:-outputs/ckpt_amt_v15_xlarge_duration/best.pt}
OUT_DIR=${OUT_DIR:-outputs/eval_compare_v13_v15}
V13_CACHE=${V13_CACHE:-data/cache/amt_v13_large_finetune_calib512_val_8s_10ms_229mel_center6}
V15_CACHE=${V15_CACHE:-data/cache/amt_v15_xlarge_duration_calib512_val_8s_10ms_229mel_center6}
THRESH_ONSET=${THRESH_ONSET:-0.34,0.38,0.42,0.46,0.50}
THRESH_FRAME=${THRESH_FRAME:-0.16,0.20,0.24}
THRESH_OFFSET=${THRESH_OFFSET:-0.20,0.24,0.28}

run_eval() {
  local name=$1
  local ckpt=$2
  local preset=$3
  local cache_dir=$4
  shift 4
  local log_file="log/eval_${name}_${STAMP}.log"
  local json_out="${OUT_DIR}/${name}_val_calib512.json"
  echo "eval ${name}: ckpt=${ckpt} preset=${preset} json=${json_out}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" scripts/eval_amt.py \
    --ckpt "${ckpt}" \
    --manifest "${VAL_MANIFEST}" \
    --items "${ITEMS}" \
    --cache_dir "${cache_dir}" \
    --decode_preset "${preset}" \
    --onset_thresholds "${THRESH_ONSET}" \
    --frame_thresholds "${THRESH_FRAME}" \
    --offset_thresholds "${THRESH_OFFSET}" \
    --score_quality_eval \
    --analysis_json_out "${json_out}" \
    "$@" \
    > "${log_file}" 2>&1
  tail -n 20 "${log_file}"
}

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}" log

if [[ "${RUN_INSIDE:-0}" == "1" ]]; then
  run_eval v13_best "${V13_CKPT}" practice_score "${V13_CACHE}"
  run_eval v15_xlarge_best "${V15_CKPT}" v15_f1 "${V15_CACHE}"
  if [[ "${RUN_HYBRID:-0}" == "1" ]]; then
    run_eval v13_v15_hybrid "${V13_CKPT}" practice_score "${V13_CACHE}" \
      --assistant_ckpt "${V15_CKPT}" \
      --assistant_decode_preset v15_rescue \
      --hybrid_rescue
  fi
  exit 0
fi

echo "Prepared v13/v15 validation comparison."
echo "  project: ${PROJECT_DIR}"
echo "  output:  ${OUT_DIR}"
echo "  session: ${SESSION}"
echo "Set LAUNCH_EVAL=1 to run in tmux on GPU ${GPU_ID}."

if [[ "${LAUNCH_EVAL:-0}" == "1" ]]; then
  tmux new-session -d -s "${SESSION}" \
    "cd ${PROJECT_DIR} && RUN_INSIDE=1 PROJECT_DIR=${PROJECT_DIR} PYTHON_BIN=${PYTHON_BIN} GPU_ID=${GPU_ID} STAMP=${STAMP} RUN_HYBRID=${RUN_HYBRID:-0} bash scripts/remote_eval_v13_v15.sh"
  echo "started ${SESSION}"
fi
