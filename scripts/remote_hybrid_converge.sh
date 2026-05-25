#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/data/MiniMT3/MiniMT3}
PYTHON_BIN=${PYTHON_BIN:-/data/app/dp2.2.11/envs/MiniMT3/bin/python}
GPU_ID=${GPU_ID:-0}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
SESSION=${SESSION:-MiniMT3_hybrid_converge_${STAMP}}

V13_CKPT=${V13_CKPT:-outputs/ckpt_amt_v13_large_finetune/best.pt}
V15_CKPT=${V15_CKPT:-outputs/ckpt_amt_v15_xlarge_duration/best.pt}
V19_CKPT=${V19_CKPT:-outputs/ckpt_amt_v19_xlarge_precision_teacher_gate/step_1500_selected.pt}

MANIFEST=${MANIFEST:-data/cache/amt_val_8s_s8_calib512_v13.json}
V13_CACHE=${V13_CACHE:-data/cache/amt_v13_large_finetune_calib512_val_8s_10ms_229mel_center6}
V15_CACHE=${V15_CACHE:-data/cache/amt_v15_xlarge_duration_calib512_val_8s_10ms_229mel_center6}
V19_CACHE=${V19_CACHE:-data/cache/amt_v19_precision_teacher_gate_calib512_val_8s_10ms_229mel_center6}
EVAL_DIR=${EVAL_DIR:-outputs/eval_hybrid_converge}
DEMO_ROOT=${DEMO_ROOT:-outputs/demo_compare}
CALIB_JSON=${CALIB_JSON:-${EVAL_DIR}/v19_pitch_calibration.json}
SCORE_KEY=${SCORE_KEY:-C# minor}
SCORE_TIME=${SCORE_TIME:-4/4}
SCORE_TEMPO=${SCORE_TEMPO:-100}

cd "${PROJECT_DIR}"
mkdir -p "${EVAL_DIR}" "${DEMO_ROOT}" log

run_eval() {
  local name=$1
  local ckpt=$2
  local preset=$3
  local cache_dir=$4
  shift 4
  if [[ ! -s "${ckpt}" ]]; then
    echo "skip eval ${name}: missing ${ckpt}" >&2
    return 0
  fi
  echo "hybrid eval ${name}: preset=${preset}"
  env "$@" \
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
    bash scripts/remote_eval_mainline_amt.sh
}

build_calibration() {
  local src="${EVAL_DIR}/v19_selected_v19_precision_val_calib${ITEMS:-512}.json"
  if [[ ! -s "${src}" ]]; then
    src=${CALIB_SOURCE:-outputs/eval_final_mainline/v19_selected_v19_precision_val_calib512.json}
  fi
  if [[ ! -s "${src}" ]]; then
    echo "skip pitch calibration: missing ${src}" >&2
    return 0
  fi
  "${PYTHON_BIN}" scripts/build_pitch_calibration.py \
    --eval_json "${src}" \
    --out "${CALIB_JSON}" \
    --clamp_min "${CALIB_CLAMP_MIN:--0.06}" \
    --clamp_max "${CALIB_CLAMP_MAX:-0.04}" \
    --scale "${CALIB_SCALE:-1.0}" \
    --min_observations "${CALIB_MIN_OBSERVATIONS:-3}"
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
  local out_dir="${DEMO_ROOT}/hybrid_converge_${stem}_${name}"
  echo "hybrid demo ${stem}/${name}: decode=${decode_preset} score=${score_preset}"
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
  local calib_args=()
  if [[ -s "${CALIB_JSON}" ]]; then
    calib_args=(--assistant_pitch_calibration_json "${CALIB_JSON}")
  fi
  infer_demo "${audio}" v13_score "${V13_CKPT}" practice_score score_demo_4_4
  infer_demo "${audio}" v15_score "${V15_CKPT}" v15_f1 score_demo_4_4
  infer_demo "${audio}" v19_score "${V19_CKPT}" v19_precision score_demo_4_4
  infer_demo "${audio}" hybrid_f1 "${V13_CKPT}" v13_recall performance_midi \
    --assistant_ckpt "${V19_CKPT}" --assistant_decode_preset v19_recall --hybrid_rescue \
    --hybrid_preset hybrid_f1 "${calib_args[@]}"
  infer_demo "${audio}" hybrid_score "${V13_CKPT}" practice_score score_demo_4_4 \
    --assistant_ckpt "${V19_CKPT}" --assistant_decode_preset v19_recall --hybrid_rescue \
    --hybrid_preset hybrid_score "${calib_args[@]}"
}

run_inside() {
  if [[ "${RUN_EVAL:-1}" == "1" ]]; then
    run_eval v13_best "${V13_CKPT}" practice_score "${V13_CACHE}" \
      ONSET_THRESHOLDS="${V13_ONSETS:-0.44,0.46,0.48}" FRAME_THRESHOLDS="${V13_FRAMES:-0.22}" \
      OFFSET_THRESHOLDS="${V13_OFFSETS:-0.24,0.26}" FRAME_DIFF_MODES="${V13_FRAME_DIFF:-true}" \
      FRAME_DIFF_SCALES="${V13_FRAME_DIFF_SCALES:-0.70}" DURATION_EXTENSION_WEIGHTS="${V13_DURATION_WEIGHTS:-0.25}"
    run_eval v15_best "${V15_CKPT}" v15_f1 "${V15_CACHE}" \
      ONSET_THRESHOLDS="${V15_ONSETS:-0.40,0.42,0.44}" FRAME_THRESHOLDS="${V15_FRAMES:-0.20}" \
      OFFSET_THRESHOLDS="${V15_OFFSETS:-0.22,0.24}" FRAME_DIFF_MODES="${V15_FRAME_DIFF:-true}" \
      FRAME_DIFF_SCALES="${V15_FRAME_DIFF_SCALES:-0.78}" DURATION_EXTENSION_WEIGHTS="${V15_DURATION_WEIGHTS:-0.25}"
    run_eval v19_selected "${V19_CKPT}" v19_precision "${V19_CACHE}" \
      ONSET_THRESHOLDS="${V19_ONSETS:-0.50,0.52,0.54}" FRAME_THRESHOLDS="${V19_FRAMES:-0.24}" \
      OFFSET_THRESHOLDS="${V19_OFFSETS:-0.20,0.24}" FRAME_DIFF_MODES="${V19_FRAME_DIFF:-false}" \
      FRAME_DIFF_SCALES="${V19_FRAME_DIFF_SCALES:-0.45}" DURATION_EXTENSION_WEIGHTS="${V19_DURATION_WEIGHTS:-0.25}"
    build_calibration
    local calib_env=()
    if [[ -s "${CALIB_JSON}" ]]; then
      calib_env=(ASSISTANT_PITCH_CALIBRATION_JSON="${CALIB_JSON}")
    fi
    run_eval hybrid_score "${V13_CKPT}" practice_score "${V13_CACHE}" \
      ASSISTANT_CKPT="${V19_CKPT}" ASSISTANT_PRESET=v19_recall HYBRID_RESCUE=1 HYBRID_PRESET=hybrid_score \
      "${calib_env[@]}" ONSET_THRESHOLDS="${HYBRID_SCORE_ONSETS:-0.44,0.46,0.48}" FRAME_THRESHOLDS="${HYBRID_SCORE_FRAMES:-0.22}" \
      OFFSET_THRESHOLDS="${HYBRID_SCORE_OFFSETS:-0.24,0.26}" FRAME_DIFF_MODES="${HYBRID_SCORE_FRAME_DIFF:-true}" \
      FRAME_DIFF_SCALES="${HYBRID_SCORE_FRAME_DIFF_SCALES:-0.70}" DURATION_EXTENSION_WEIGHTS="${HYBRID_SCORE_DURATION_WEIGHTS:-0.25}"
    run_eval hybrid_f1 "${V13_CKPT}" v13_recall "${V13_CACHE}" \
      ASSISTANT_CKPT="${V19_CKPT}" ASSISTANT_PRESET=v19_recall HYBRID_RESCUE=1 HYBRID_PRESET=hybrid_f1 \
      "${calib_env[@]}" ONSET_THRESHOLDS="${HYBRID_F1_ONSETS:-0.36,0.40,0.44}" FRAME_THRESHOLDS="${HYBRID_F1_FRAMES:-0.18,0.20}" \
      OFFSET_THRESHOLDS="${HYBRID_F1_OFFSETS:-0.22,0.24}" FRAME_DIFF_MODES="${HYBRID_F1_FRAME_DIFF:-true}" \
      FRAME_DIFF_SCALES="${HYBRID_F1_FRAME_DIFF_SCALES:-0.70,0.85}" DURATION_EXTENSION_WEIGHTS="${HYBRID_F1_DURATION_WEIGHTS:-0.25,0.35}"
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
  "${PYTHON_BIN}" scripts/summarize_hybrid_converge.py \
    --eval_dir "${EVAL_DIR}" \
    --demo_root "${DEMO_ROOT}" \
    --out_json "${EVAL_DIR}/hybrid_metrics.json" \
    --out_csv "${EVAL_DIR}/hybrid_metrics.csv" \
    --demo_json "${DEMO_ROOT}/hybrid_demo_report.json"
}

echo "Prepared MiniMT3 hybrid convergence run."
echo "  eval_dir:  ${EVAL_DIR}"
echo "  demo_root: ${DEMO_ROOT}"
echo "  session:   ${SESSION}"
echo "Set LAUNCH_HYBRID=1 to run in tmux on GPU ${GPU_ID}."

if [[ "${RUN_INSIDE:-0}" == "1" ]]; then
  run_inside
  exit 0
fi

if [[ "${LAUNCH_HYBRID:-0}" == "1" ]]; then
  tmux new-session -d -s "${SESSION}" \
    "cd ${PROJECT_DIR} && RUN_INSIDE=1 PROJECT_DIR=${PROJECT_DIR} PYTHON_BIN=${PYTHON_BIN} GPU_ID=${GPU_ID} STAMP=${STAMP} RUN_EVAL=${RUN_EVAL:-1} RUN_DEMO=${RUN_DEMO:-1} bash scripts/remote_hybrid_converge.sh > log/hybrid_converge_${STAMP}.log 2>&1"
  echo "started ${SESSION}"
  echo "log: ${PROJECT_DIR}/log/hybrid_converge_${STAMP}.log"
fi
