#!/usr/bin/env bash
set -euo pipefail

ROOT=/keyan/lsh/lsh_teacher_cache_8step_distill
LTX_ROOT="${ROOT}/LTX-2"
TRAINER_DIR="${LTX_ROOT}/packages/ltx-trainer"
ENV=/keyan/lsh/ltx_dmd/envs/ltx
LOG_DIR="${ROOT}/logs"

PHASE1_CONFIG="${ROOT}/configs/official8_student8_av_quality_phase1_1500_keeponly.yaml"
DMD_CONFIG="${ROOT}/configs/official8_student8_av_dmd_rank16_phase2_from1500_10000.yaml"
PHASE1_RUN="${ROOT}/runs/official8_student8_av_quality_phase1_1500_keeponly"
DMD_RUN="${ROOT}/runs/official8_student8_av_dmd_rank16_phase2_from1500_10000"
PHASE1_LORA="${PHASE1_RUN}/checkpoints/lora_weights_step_01500.safetensors"

mkdir -p "${LOG_DIR}" "${PHASE1_RUN}" "${DMD_RUN}" "${ROOT}/outputs"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export PYTHONPATH="${LTX_ROOT}/packages/ltx-trainer/src:${LTX_ROOT}/packages/ltx-core/src:${LTX_ROOT}/packages/ltx-pipelines/src:${LTX_ROOT}/packages/ltx-video/src:${PYTHONPATH:-}"

run_train() {
  local config="$1"
  local log_file="$2"
  cd "${TRAINER_DIR}"
  echo "config=${config}" | tee -a "${log_file}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}" | tee -a "${log_file}"
  echo "log_file=${log_file}" | tee -a "${log_file}"
  "${ENV}/bin/accelerate" launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    scripts/train.py "${config}" \
    --disable-progress-bars \
    >> "${log_file}" 2>&1
}

PHASE1_LOG="${LOG_DIR}/train_official8_quality_phase1_1500_$(date +%Y%m%d_%H%M%S).out"
if [ ! -f "${PHASE1_LORA}" ]; then
  echo "PHASE1_START $(date --iso-8601=seconds)" | tee -a "${PHASE1_LOG}"
  run_train "${PHASE1_CONFIG}" "${PHASE1_LOG}"
  echo "PHASE1_DONE $(date --iso-8601=seconds)" | tee -a "${PHASE1_LOG}"
else
  echo "PHASE1_SKIP existing ${PHASE1_LORA}" | tee -a "${PHASE1_LOG}"
fi

if [ ! -f "${PHASE1_LORA}" ]; then
  echo "ERROR missing phase1 checkpoint ${PHASE1_LORA}" | tee -a "${PHASE1_LOG}"
  exit 1
fi

if [ -d "${PHASE1_RUN}/checkpoints" ]; then
  find "${PHASE1_RUN}/checkpoints" -maxdepth 1 -type f ! -name '*01500*' -delete
fi

COMPARE_BASE="${ROOT}/outputs/compare_official8_vs_phase1_1500_1280x704_$(date +%Y%m%d_%H%M%S)"
GEN_LOG="${LOG_DIR}/generate_official8_vs_phase1_1500_$(date +%Y%m%d_%H%M%S).out"
mkdir -p "${COMPARE_BASE}"
echo "${COMPARE_BASE}" > "${PHASE1_RUN}/last_compare_dir.txt"
echo "${COMPARE_BASE}" > "${ROOT}/last_compare_dir.txt"
echo "COMPARE_BASE=${COMPARE_BASE}" | tee -a "${GEN_LOG}"

CUDA_VISIBLE_DEVICES=4 "${ENV}/bin/python" "${ROOT}/scripts/hot_distilled_batch_lora_any.py" \
  --label official8 \
  --outdir "${COMPARE_BASE}" \
  --width 1280 --height 704 --frames 121 --fps 24 --prompt-count 3 \
  >> "${GEN_LOG}" 2>&1 &
pid_official=$!

CUDA_VISIBLE_DEVICES=5 "${ENV}/bin/python" "${ROOT}/scripts/hot_distilled_batch_lora_any.py" \
  --label phase1_1500 \
  --outdir "${COMPARE_BASE}" \
  --lora-path "${PHASE1_LORA}" \
  --lora-scale 1.0 \
  --width 1280 --height 704 --frames 121 --fps 24 --prompt-count 3 \
  >> "${GEN_LOG}" 2>&1 &
pid_phase1=$!

wait "${pid_official}"
wait "${pid_phase1}"
echo "GENERATION_DONE $(date --iso-8601=seconds)" | tee -a "${GEN_LOG}"

VBENCH_LOG="${LOG_DIR}/vbench_official8_vs_phase1_1500_$(date +%Y%m%d_%H%M%S).out"
bash "${ROOT}/scripts/run_vbench_two_cases_selected_98.sh" "${COMPARE_BASE}" official8 phase1_1500 \
  > "${VBENCH_LOG}" 2>&1
echo "VBENCH_DONE $(date --iso-8601=seconds)" | tee -a "${VBENCH_LOG}"

if [ "${START_DMD_AFTER_TEST:-1}" = "1" ]; then
  DMD_LOG="${LOG_DIR}/train_official8_dmd_phase2_from1500_$(date +%Y%m%d_%H%M%S).out"
  echo "DMD_PHASE2_START $(date --iso-8601=seconds)" | tee -a "${DMD_LOG}"
  run_train "${DMD_CONFIG}" "${DMD_LOG}"
  echo "DMD_PHASE2_DONE $(date --iso-8601=seconds)" | tee -a "${DMD_LOG}"
else
  echo "START_DMD_AFTER_TEST=${START_DMD_AFTER_TEST}; skip DMD phase2"
fi
