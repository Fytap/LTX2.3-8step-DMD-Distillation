#!/usr/bin/env bash
set -euo pipefail

ROOT=/keyan/lsh/lsh_teacher_cache_8step_distill
ENV=/keyan/lsh/ltx_dmd/envs/ltx
PROMPTS="${ROOT}/prompts/JavisBench_text_first50_prompts.jsonl"
LORA="${ROOT}/runs/official8_student8_av_dmd_rank16_phase2_from1500_10000/checkpoints/lora_weights_step_02000.safetensors"
OUT=$(cat "${ROOT}/last_javisbench50_teacherdev40_1000_dir.txt")
LOG="${ROOT}/logs/eval_javisbench50_dmd2000_$(date +%Y%m%d_%H%M%S).out"

{
  echo "TEST_START $(date --iso-8601=seconds)"
  echo "OUT=${OUT}"
  echo "LORA=${LORA}"
  test -f "${PROMPTS}"
  test -f "${LORA}"
  test "$(find "${OUT}/official8" -maxdepth 1 -name '*.mp4' | wc -l)" -eq 50

  CUDA_VISIBLE_DEVICES=5 "${ENV}/bin/python" "${ROOT}/scripts/hot_distilled_batch_prompt_file.py" \
    --label dmd2000 --outdir "${OUT}" --prompt-file "${PROMPTS}" --prompt-count 50 \
    --lora-path "${LORA}" --lora-scale 1.0 \
    --width 1280 --height 704 --frames 121 --fps 24 > "${OUT}/generate_dmd2000.log" 2>&1
  echo "GENERATE_DONE $(date --iso-8601=seconds)"

  bash "${ROOT}/scripts/run_vbench_prompt_file_two_cases_98.sh" "${OUT}" official8 dmd2000 "${PROMPTS}"
  echo "VBENCH_DONE $(date --iso-8601=seconds)"
  cat "${OUT}/vbench_selected_summary.md"
  echo "TEST_DONE $(date --iso-8601=seconds)"
} >> "${LOG}" 2>&1

ln -sfn "${LOG}" "${ROOT}/logs/eval_javisbench50_dmd2000_latest.out"
