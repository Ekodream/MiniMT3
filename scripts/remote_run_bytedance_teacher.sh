#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/data/MiniMT3/MiniMT3}
PYTHON_BIN=${PYTHON_BIN:-/data/app/dp2.2.11/envs/MiniMT3/bin/python}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
SESSION=${SESSION:-MiniMT3_bytedance_teacher_${STAMP}}
MANIFEST=${MANIFEST:-data/cache/amt_val_8s_s8_calib512_v13.json}
OUT_DIR=${OUT_DIR:-outputs/teacher_bytedance/val_calib512}
SUMMARY_JSON=${SUMMARY_JSON:-${OUT_DIR}/teacher_summary_${STAMP}.json}
DEVICE=${DEVICE:-cuda}
GPU_ID=${GPU_ID:-0}
ITEMS=${ITEMS:-512}
SPLIT=${SPLIT:-validation}
START_INDEX=${START_INDEX:-0}

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}" log

if [[ "${INSTALL_TEACHER_DEPS:-0}" == "1" ]]; then
  "${PYTHON_BIN}" -m pip install piano-transcription-inference
fi

"${PYTHON_BIN}" - <<'PY'
try:
    import piano_transcription_inference  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "piano_transcription_inference is not installed. "
        "Rerun with INSTALL_TEACHER_DEPS=1, or install piano-transcription-inference manually."
    ) from exc
PY

echo "Prepared ByteDance teacher generation."
echo "  manifest: ${MANIFEST}"
echo "  out:      ${OUT_DIR}"
echo "  items:    ${ITEMS}"
echo "  session:  ${SESSION}"
echo "Set LAUNCH_TEACHER=1 to run in tmux."

if [[ "${LAUNCH_TEACHER:-0}" == "1" ]]; then
  tmux new-session -d -s "${SESSION}" \
    "cd ${PROJECT_DIR} && CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} scripts/run_bytedance_teacher.py --manifest ${MANIFEST} --out_dir ${OUT_DIR} --split ${SPLIT} --items ${ITEMS} --start_index ${START_INDEX} --device ${DEVICE} --summary_json ${SUMMARY_JSON} ${EXTRA_TEACHER_ARGS:-} > log/bytedance_teacher_${STAMP}.log 2>&1"
  echo "started ${SESSION}"
  echo "log: ${PROJECT_DIR}/log/bytedance_teacher_${STAMP}.log"
fi
