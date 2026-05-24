#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/data/MiniMT3/MiniMT3}
PYTHON_BIN=${PYTHON_BIN:-/data/app/dp2.2.11/envs/MiniMT3/bin/python}
GPU_ID=${GPU_ID:-0}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
SESSION=${SESSION:-MiniMT3_test_new_compare_${STAMP}}

V13_CKPT=${V13_CKPT:-outputs/ckpt_amt_v13_large_finetune/best.pt}
V15_CKPT=${V15_CKPT:-outputs/ckpt_amt_v15_xlarge_duration/best.pt}
OUT_ROOT=${OUT_ROOT:-outputs/demo_compare}
SCORE_KEY=${SCORE_KEY:-C# minor}
SCORE_TIME=${SCORE_TIME:-4/4}
SCORE_TEMPO=${SCORE_TEMPO:-100}

cd "${PROJECT_DIR}"

if [[ -z "${AUDIO:-}" ]]; then
  if [[ -f ../demo/Test_new.wav ]]; then
    AUDIO=../demo/Test_new.wav
  elif [[ -f /data/MiniMT3/demo/Test_new.wav ]]; then
    AUDIO=/data/MiniMT3/demo/Test_new.wav
  else
    echo "Cannot find Test_new.wav. Set AUDIO=/path/to/Test_new.wav." >&2
    exit 2
  fi
fi

run_demo() {
  local name=$1
  local ckpt=$2
  local decode_preset=$3
  local score_preset=$4
  shift 4
  local out_dir="${OUT_ROOT}/${name}"
  echo "infer ${name}: ckpt=${ckpt} decode=${decode_preset} score=${score_preset} out=${out_dir}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" scripts/infer_amt.py \
    --audio "${AUDIO}" \
    --ckpt "${ckpt}" \
    --out "${out_dir}" \
    --decode_preset "${decode_preset}" \
    --score_preset "${score_preset}" \
    --score_time_signature "${SCORE_TIME}" \
    --score_tempo_bpm "${SCORE_TEMPO}" \
    --score_key_signature "${SCORE_KEY}" \
    --score_voice_mode dual_staff_2voice \
    --score_split_ties \
    --score_hide_filler_rests \
    "$@"
  "${PYTHON_BIN}" scripts/eval_score_polish.py \
    --debug_json "${out_dir}/Test_new_debug.json" \
    >> "${OUT_ROOT}/score_summary_${STAMP}.txt"
}

mkdir -p "${OUT_ROOT}" log

if [[ "${RUN_INSIDE:-0}" == "1" ]]; then
  : > "${OUT_ROOT}/score_summary_${STAMP}.txt"
  if [[ "${ONLY_HYBRID:-0}" != "1" ]]; then
    run_demo v13_best_score "${V13_CKPT}" practice_score score_demo_4_4
    run_demo v15_best_score "${V15_CKPT}" v15_f1 score_demo_4_4
    run_demo v15_analysis_midi "${V15_CKPT}" analysis_midi performance_midi
  fi
  run_demo v13_v15_hybrid_score "${V13_CKPT}" practice_score score_demo_4_4 \
    --assistant_ckpt "${V15_CKPT}" \
    --assistant_decode_preset v15_rescue \
    --hybrid_rescue
  echo "summary: ${OUT_ROOT}/score_summary_${STAMP}.txt"
  cat "${OUT_ROOT}/score_summary_${STAMP}.txt"
  exit 0
fi

echo "Prepared Test_new comparison."
echo "  audio:   ${AUDIO}"
echo "  output:  ${OUT_ROOT}"
echo "  session: ${SESSION}"
echo "Set LAUNCH_DEMO=1 to run in tmux on GPU ${GPU_ID}."

if [[ "${LAUNCH_DEMO:-0}" == "1" ]]; then
  tmux new-session -d -s "${SESSION}" \
    "cd ${PROJECT_DIR} && RUN_INSIDE=1 PROJECT_DIR=${PROJECT_DIR} PYTHON_BIN=${PYTHON_BIN} GPU_ID=${GPU_ID} STAMP=${STAMP} AUDIO=${AUDIO} ONLY_HYBRID=${ONLY_HYBRID:-0} bash scripts/remote_run_test_new_compare.sh > log/test_new_compare_${STAMP}.log 2>&1"
  echo "started ${SESSION}"
  echo "log: ${PROJECT_DIR}/log/test_new_compare_${STAMP}.log"
fi
