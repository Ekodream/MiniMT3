#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/data/MiniMT3/MiniMT3}
PYTHON_BIN=${PYTHON_BIN:-/data/app/dp2.2.11/envs/MiniMT3/bin/python}
GPU_ID=${GPU_ID:-0}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
SESSION=${SESSION:-MiniMT3_final_converge_${STAMP}}

V13_CKPT=${V13_CKPT:-outputs/ckpt_amt_v13_large_finetune/best.pt}
V15_CKPT=${V15_CKPT:-outputs/ckpt_amt_v15_xlarge_duration/best.pt}
V19_DIR=${V19_DIR:-outputs/ckpt_amt_v19_xlarge_precision_teacher_gate}
V19_SELECTED_CKPT=${V19_SELECTED_CKPT:-${V19_DIR}/step_1500_selected.pt}
V19_BEST_CKPT=${V19_BEST_CKPT:-${V19_DIR}/best.pt}
V19_STEP1800_CKPT=${V19_STEP1800_CKPT:-${V19_DIR}/step_1800_predebug.pt}
V19_LOG=${V19_LOG:-log/v19_precision_teacher_gate_20260525_034803.log}
V19_SESSION=${V19_SESSION:-MiniMT3_v19_precision_teacher_gate_20260525_034803}

MANIFEST=${MANIFEST:-data/cache/amt_val_8s_s8_calib512_v13.json}
EVAL_DIR=${EVAL_DIR:-outputs/eval_final_mainline}
DEMO_ROOT=${DEMO_ROOT:-outputs/demo_compare}
SCORE_KEY=${SCORE_KEY:-C# minor}
SCORE_TIME=${SCORE_TIME:-4/4}
SCORE_TEMPO=${SCORE_TEMPO:-100}

cd "${PROJECT_DIR}"
mkdir -p "${EVAL_DIR}" "${DEMO_ROOT}" log

wait_for_v19() {
  if [[ "${WAIT_V19:-1}" != "1" ]]; then
    return 0
  fi
  local waited=0
  local limit=${V19_WAIT_SECONDS:-1800}
  while tmux has-session -t "${V19_SESSION}" 2>/dev/null; do
    if [[ -s "${V19_DIR}/last.pt" ]]; then
      break
    fi
    if (( waited >= limit )); then
      echo "v19 wait timeout after ${waited}s; continuing with saved checkpoints."
      return 0
    fi
    echo "waiting for v19 to finish final debug... ${waited}s"
    sleep 60
    waited=$((waited + 60))
  done
  if [[ -s "${V19_DIR}/last.pt" ]] && tmux has-session -t "${V19_SESSION}" 2>/dev/null; then
    echo "v19 has last.pt; closing tmux session ${V19_SESSION}"
    tmux kill-session -t "${V19_SESSION}" || true
  fi
}

preserve_v19_selected() {
  if [[ ! -s "${V19_SELECTED_CKPT}" && -s "${V19_BEST_CKPT}" ]]; then
    cp -n "${V19_BEST_CKPT}" "${V19_SELECTED_CKPT}"
    echo "preserved ${V19_SELECTED_CKPT}"
  fi
}

run_eval() {
  local name=$1
  local ckpt=$2
  local preset=$3
  local cache_dir=$4
  if [[ ! -s "${ckpt}" ]]; then
    echo "skip eval ${name}: missing ${ckpt}" >&2
    return 0
  fi
  echo "final eval ${name}: ckpt=${ckpt} preset=${preset}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    MODEL_NAME="${name}" \
    CKPT="${ckpt}" \
    PRESET="${preset}" \
    MANIFEST="${MANIFEST}" \
    CACHE_DIR="${cache_dir}" \
    ITEMS="${ITEMS:-512}" \
    OUT_DIR="${EVAL_DIR}" \
    SCORE_QUALITY_EVAL="${SCORE_QUALITY_EVAL:-1}" \
    SCORE_QUALITY_ITEMS="${SCORE_QUALITY_ITEMS:-24}" \
    ONSET_THRESHOLDS="${ONSET_THRESHOLDS:-0.50,0.52,0.54}" \
    FRAME_THRESHOLDS="${FRAME_THRESHOLDS:-0.24}" \
    OFFSET_THRESHOLDS="${OFFSET_THRESHOLDS:-0.20,0.24}" \
    FRAME_DIFF_MODES="${FRAME_DIFF_MODES:-false}" \
    FRAME_DIFF_SCALES="${FRAME_DIFF_SCALES:-0.45}" \
    DURATION_EXTENSION_WEIGHTS="${DURATION_EXTENSION_WEIGHTS:-0.25}" \
    bash scripts/remote_eval_mainline_amt.sh
}

copy_existing_baselines() {
  local src_v13=${EXISTING_V13_EVAL:-outputs/eval_compare_v13_v15/v13_best_val_calib512.json}
  local src_v15=${EXISTING_V15_EVAL:-outputs/eval_compare_v13_v15/v15_xlarge_best_val_calib512.json}
  if [[ -s "${src_v13}" ]]; then
    cp -f "${src_v13}" "${EVAL_DIR}/v13_best_practice_score_val_calib512.json"
    echo "copied existing baseline ${src_v13}"
  else
    echo "missing existing v13 eval: ${src_v13}" >&2
  fi
  if [[ -s "${src_v15}" ]]; then
    cp -f "${src_v15}" "${EVAL_DIR}/v15_best_v15_f1_val_calib512.json"
    echo "copied existing baseline ${src_v15}"
  else
    echo "missing existing v15 eval: ${src_v15}" >&2
  fi
}

infer_demo() {
  local audio=$1
  local name=$2
  local ckpt=$3
  local decode_preset=$4
  local score_preset=$5
  shift 5
  if [[ ! -f "${audio}" || ! -s "${ckpt}" ]]; then
    echo "skip demo ${name}: audio=${audio} ckpt=${ckpt}" >&2
    return 0
  fi
  local stem
  stem=$(basename "${audio}")
  stem="${stem%.*}"
  local out_dir="${DEMO_ROOT}/final_${stem}_${name}"
  echo "final demo ${stem}/${name}: decode=${decode_preset} score=${score_preset}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" scripts/infer_amt.py \
    --audio "${audio}" \
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
  "${PYTHON_BIN}" scripts/eval_score_polish.py --debug_json "${out_dir}/${stem}_debug.json"
}

run_demos_for_audio() {
  local audio=$1
  infer_demo "${audio}" v13_score "${V13_CKPT}" practice_score score_demo_4_4
  infer_demo "${audio}" v15_analysis "${V15_CKPT}" analysis_midi performance_midi
  infer_demo "${audio}" v15_score "${V15_CKPT}" v15_f1 score_demo_4_4
  infer_demo "${audio}" v19_recall "${V19_SELECTED_CKPT}" v19_recall performance_midi
  infer_demo "${audio}" v19_score "${V19_SELECTED_CKPT}" v19_precision score_demo_4_4
  infer_demo "${audio}" v13_v15_hybrid_score "${V13_CKPT}" practice_score score_demo_4_4 \
    --assistant_ckpt "${V15_CKPT}" --assistant_decode_preset v15_rescue --hybrid_rescue \
    --hybrid_preset display_chord_long
  infer_demo "${audio}" v13_v19_hybrid_score "${V13_CKPT}" practice_score score_demo_4_4 \
    --assistant_ckpt "${V19_SELECTED_CKPT}" --assistant_decode_preset v19_recall --hybrid_rescue \
    --hybrid_preset display_chord_long
}

run_inside() {
  wait_for_v19
  preserve_v19_selected
  if [[ "${RUN_EVAL:-1}" == "1" ]]; then
    if [[ "${RUN_BASELINE_EVAL:-0}" == "1" ]]; then
      run_eval v13_best "${V13_CKPT}" practice_score data/cache/amt_v13_large_finetune_calib512_val_8s_10ms_229mel_center6
      run_eval v15_best "${V15_CKPT}" v15_f1 data/cache/amt_v15_xlarge_duration_calib512_val_8s_10ms_229mel_center6
    else
      copy_existing_baselines
    fi
    run_eval v19_selected "${V19_SELECTED_CKPT}" v19_precision data/cache/amt_v19_precision_teacher_gate_calib512_val_8s_10ms_229mel_center6
    run_eval v19_step1800 "${V19_STEP1800_CKPT}" v19_precision data/cache/amt_v19_precision_teacher_gate_calib512_val_8s_10ms_229mel_center6
  fi
  if [[ "${RUN_DEMO:-1}" == "1" ]]; then
    if [[ -n "${AUDIO_LIST:-}" ]]; then
      IFS=":" read -r -a audios <<< "${AUDIO_LIST}"
      for audio in "${audios[@]}"; do
        run_demos_for_audio "${audio}"
      done
    else
      for audio in ../demo/Test_new.wav /data/MiniMT3/demo/Test_new.wav ../demo/Tst_new.wav /data/MiniMT3/demo/Tst_new.wav; do
        [[ -f "${audio}" ]] && run_demos_for_audio "${audio}"
      done
    fi
  fi
  "${PYTHON_BIN}" scripts/summarize_final_results.py \
    --eval_dir "${EVAL_DIR}" \
    --demo_root "${DEMO_ROOT}" \
    --out_json "${EVAL_DIR}/final_metrics.json" \
    --out_csv "${EVAL_DIR}/final_metrics.csv" \
    --demo_json "${DEMO_ROOT}/final_demo_report.json"
}

echo "Prepared final MiniMT3 convergence run."
echo "  eval_dir:  ${EVAL_DIR}"
echo "  demo_root: ${DEMO_ROOT}"
echo "  session:   ${SESSION}"
echo "Set LAUNCH_FINAL=1 to run in tmux on GPU ${GPU_ID}."

if [[ "${RUN_INSIDE:-0}" == "1" ]]; then
  run_inside
  exit 0
fi

if [[ "${LAUNCH_FINAL:-0}" == "1" ]]; then
  tmux new-session -d -s "${SESSION}" \
    "cd ${PROJECT_DIR} && RUN_INSIDE=1 PROJECT_DIR=${PROJECT_DIR} PYTHON_BIN=${PYTHON_BIN} GPU_ID=${GPU_ID} STAMP=${STAMP} RUN_EVAL=${RUN_EVAL:-1} RUN_DEMO=${RUN_DEMO:-1} RUN_BASELINE_EVAL=${RUN_BASELINE_EVAL:-0} bash scripts/remote_finalize_mainline.sh > log/final_converge_${STAMP}.log 2>&1"
  echo "started ${SESSION}"
  echo "log: ${PROJECT_DIR}/log/final_converge_${STAMP}.log"
fi
