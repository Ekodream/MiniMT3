#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/data/MiniMT3/MiniMT3}
PYTHON_BIN=${PYTHON_BIN:-/data/app/dp2.2.11/envs/MiniMT3/bin/python}
CKPT=${CKPT:?set CKPT to a dense-AMT checkpoint}
MODEL_NAME=${MODEL_NAME:-$(basename "$(dirname "${CKPT}")")}
PRESET=${PRESET:-v17_f1}
MANIFEST=${MANIFEST:-data/cache/amt_val_8s_s8_calib512_v13.json}
CACHE_DIR=${CACHE_DIR:-}
ITEMS=${ITEMS:-512}
OUT_DIR=${OUT_DIR:-outputs/eval_mainline}
DEVICE=${DEVICE:-cuda}

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}"

cache_args=()
if [[ -n "${CACHE_DIR}" ]]; then
  cache_args=(--cache_dir "${CACHE_DIR}")
fi
teacher_args=()
if [[ -n "${TEACHER_MIDI_DIR:-}" ]]; then
  teacher_args=(--teacher_midi_dir "${TEACHER_MIDI_DIR}")
fi
assistant_args=()
if [[ -n "${ASSISTANT_CKPT:-}" ]]; then
  assistant_args=(--assistant_ckpt "${ASSISTANT_CKPT}" --assistant_decode_preset "${ASSISTANT_PRESET:-v19_recall}")
fi
hybrid_args=()
if [[ "${HYBRID_RESCUE:-0}" == "1" ]]; then
  hybrid_args=(--hybrid_rescue --hybrid_preset "${HYBRID_PRESET:-hybrid_score}")
fi
pitch_args=()
if [[ -n "${PITCH_CALIBRATION_JSON:-}" ]]; then
  pitch_args+=(--pitch_calibration_json "${PITCH_CALIBRATION_JSON}")
fi
if [[ -n "${ASSISTANT_PITCH_CALIBRATION_JSON:-}" ]]; then
  pitch_args+=(--assistant_pitch_calibration_json "${ASSISTANT_PITCH_CALIBRATION_JSON}")
fi
score_quality_args=()
if [[ "${SCORE_QUALITY_EVAL:-0}" == "1" ]]; then
  score_quality_args=(--score_quality_eval)
  if [[ -n "${SCORE_QUALITY_ITEMS:-}" ]]; then
    score_quality_args+=(--score_quality_items "${SCORE_QUALITY_ITEMS}")
  fi
fi

"${PYTHON_BIN}" scripts/eval_amt.py \
  --ckpt "${CKPT}" \
  --manifest "${MANIFEST}" \
  --items "${ITEMS}" \
  --device "${DEVICE}" \
  --decode_preset "${PRESET}" \
  "${cache_args[@]}" \
  "${teacher_args[@]}" \
  "${assistant_args[@]}" \
  "${hybrid_args[@]}" \
  "${pitch_args[@]}" \
  --onset_thresholds "${ONSET_THRESHOLDS:-0.46,0.48,0.50,0.52}" \
  --frame_thresholds "${FRAME_THRESHOLDS:-0.24}" \
  --offset_thresholds "${OFFSET_THRESHOLDS:-0.18,0.20,0.24}" \
  --frame_diff_modes "${FRAME_DIFF_MODES:-false,true}" \
  --frame_diff_scales "${FRAME_DIFF_SCALES:-0.50,0.65}" \
  --duration_extension_weights "${DURATION_EXTENSION_WEIGHTS:-0.25,0.50}" \
  --duration_buckets "${DURATION_BUCKETS:-0,0.125,0.5,2.0,inf}" \
  --chord_tolerance_seconds "${CHORD_TOLERANCE_SECONDS:-0.05}" \
  "${score_quality_args[@]}" \
  --analysis_json_out "${OUT_DIR}/${MODEL_NAME}_${PRESET}_val_calib${ITEMS}.json"
