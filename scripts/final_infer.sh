#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
AUDIO=${AUDIO:?set AUDIO to an input wav path}
MODE=${MODE:-hybrid_score}
OUT_ROOT=${OUT_ROOT:-outputs/demo_compare}
SCORE_KEY=${SCORE_KEY:-C# minor}
SCORE_TIME=${SCORE_TIME:-4/4}
SCORE_TEMPO=${SCORE_TEMPO:-100}

V13_CKPT=${V13_CKPT:-outputs/ckpt_amt_v13_large_finetune/best.pt}
V15_CKPT=${V15_CKPT:-outputs/ckpt_amt_v15_xlarge_duration/best.pt}
V19_CKPT=${V19_CKPT:-outputs/ckpt_amt_v19_xlarge_precision_teacher_gate/step_1500_selected.pt}
ASSISTANT_PITCH_CALIBRATION_JSON=${ASSISTANT_PITCH_CALIBRATION_JSON:-}

cd "${PROJECT_DIR}"
stem=$(basename "${AUDIO}")
stem="${stem%.*}"

run_infer() {
  local name=$1
  local ckpt=$2
  local decode_preset=$3
  local score_preset=$4
  shift 4
  local out="${OUT_ROOT}/final_manual_${stem}_${name}"
  "${PYTHON_BIN}" scripts/infer_amt.py \
    --audio "${AUDIO}" \
    --ckpt "${ckpt}" \
    --out "${out}" \
    --decode_preset "${decode_preset}" \
    --score_preset "${score_preset}" \
    --score_time_signature "${SCORE_TIME}" \
    --score_tempo_bpm "${SCORE_TEMPO}" \
    --score_key_signature "${SCORE_KEY}" \
    --score_voice_mode dual_staff_2voice \
    --score_split_ties \
    --score_hide_filler_rests \
    "$@"
  "${PYTHON_BIN}" scripts/eval_score_polish.py --debug_json "${out}/${stem}_debug.json"
}

case "${MODE}" in
  analysis_midi)
    run_infer v15_analysis "${V15_CKPT}" analysis_midi performance_midi
    ;;
  score_demo)
    run_infer v15_score "${V15_CKPT}" v15_f1 score_demo_4_4
    ;;
  hybrid_score)
    extra_args=()
    if [[ -n "${ASSISTANT_PITCH_CALIBRATION_JSON}" ]]; then
      extra_args=(--assistant_pitch_calibration_json "${ASSISTANT_PITCH_CALIBRATION_JSON}")
    fi
    run_infer v13_v19_hybrid_score "${V13_CKPT}" practice_score score_demo_4_4 \
      --assistant_ckpt "${V19_CKPT}" \
      --assistant_decode_preset v19_recall \
      --hybrid_rescue \
      --hybrid_preset hybrid_score \
      "${extra_args[@]}"
    ;;
  hybrid_f1)
    extra_args=()
    if [[ -n "${ASSISTANT_PITCH_CALIBRATION_JSON}" ]]; then
      extra_args=(--assistant_pitch_calibration_json "${ASSISTANT_PITCH_CALIBRATION_JSON}")
    fi
    run_infer v13_v19_hybrid_f1 "${V13_CKPT}" v13_recall performance_midi \
      --assistant_ckpt "${V19_CKPT}" \
      --assistant_decode_preset v19_recall \
      --hybrid_rescue \
      --hybrid_preset hybrid_f1 \
      "${extra_args[@]}"
    ;;
  v13_clean)
    run_infer v13_score "${V13_CKPT}" practice_score score_demo_4_4
    ;;
  v19_score)
    run_infer v19_score "${V19_CKPT}" v19_precision score_demo_4_4
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use analysis_midi, score_demo, hybrid_score, hybrid_f1, v13_clean, or v19_score." >&2
    exit 2
    ;;
esac
