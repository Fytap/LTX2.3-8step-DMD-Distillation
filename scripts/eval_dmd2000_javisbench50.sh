#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LTX_ENV="${LTX_ENV:-${PROJECT_ROOT}/.venv}"
PROMPTS="${PROMPTS:-${PROJECT_ROOT}/prompts/JavisBench_text_first50_prompts.jsonl}"
LORA="${LORA:-${PROJECT_ROOT}/checkpoints/dmd_phase2/lora_weights_step_02000.safetensors}"
OUT="${OUT:-${PROJECT_ROOT}/outputs/javisbench50_official8_vs_dmd2000}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG="${LOG_DIR}/eval_javisbench50_dmd2000_$(date +%Y%m%d_%H%M%S).out"

mkdir -p "${OUT}" "${LOG_DIR}"

{
  echo "TEST_START $(date --iso-8601=seconds)"
  echo "OUT=${OUT}"
  echo "LORA=${LORA}"
  test -f "${PROMPTS}"
  test -f "${LORA}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}" "${LTX_ENV}/bin/python" "${PROJECT_ROOT}/scripts/infer_distilled_pipeline_prompt_file.py" \
    --label dmd2000 \
    --outdir "${OUT}" \
    --prompt-file "${PROMPTS}" \
    --prompt-count "${PROMPT_COUNT:-50}" \
    --lora-path "${LORA}" \
    --lora-scale "${LORA_SCALE:-1.0}" \
    --width "${WIDTH:-1280}" \
    --height "${HEIGHT:-704}" \
    --frames "${FRAMES:-121}" \
    --fps "${FPS:-24}" \
    > "${OUT}/generate_dmd2000.log" 2>&1
  echo "GENERATE_DONE $(date --iso-8601=seconds)"

  if [ -x "${PROJECT_ROOT}/scripts/run_vbench_prompt_file_two_cases.sh" ]; then
    bash "${PROJECT_ROOT}/scripts/run_vbench_prompt_file_two_cases.sh" "${OUT}" official8 dmd2000 "${PROMPTS}"
    echo "VBENCH_DONE $(date --iso-8601=seconds)"
  else
    echo "VBENCH_SKIP missing scripts/run_vbench_prompt_file_two_cases.sh"
  fi
  echo "TEST_DONE $(date --iso-8601=seconds)"
} >> "${LOG}" 2>&1

ln -sfn "${LOG}" "${LOG_DIR}/eval_javisbench50_dmd2000_latest.out"
