#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LTX_ROOT="${LTX_ROOT:-${PROJECT_ROOT}/external/LTX-2}"
TRAINER_DIR="${TRAINER_DIR:-${LTX_ROOT}/packages/ltx-trainer}"
LTX_ENV="${LTX_ENV:-${PROJECT_ROOT}/.venv}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"

PHASE1_CONFIG="${PHASE1_CONFIG:-${PROJECT_ROOT}/configs/official8_phase1_1500.yaml}"
DMD_CONFIG="${DMD_CONFIG:-${PROJECT_ROOT}/configs/dmd_phase2_from1500_to10000.yaml}"
PHASE1_RUN="${PHASE1_RUN:-${PROJECT_ROOT}/outputs/runs/official8_student8_av_quality_phase1_1500_keeponly}"
DMD_RUN="${DMD_RUN:-${PROJECT_ROOT}/outputs/runs/official8_student8_av_dmd_rank16_phase2_from1500_10000}"
PHASE1_LORA="${PHASE1_LORA:-${PHASE1_RUN}/checkpoints/lora_weights_step_01500.safetensors}"

mkdir -p "${LOG_DIR}" "${PHASE1_RUN}" "${DMD_RUN}" "${PROJECT_ROOT}/outputs"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export PYTHONPATH="${LTX_ROOT}/packages/ltx-trainer/src:${LTX_ROOT}/packages/ltx-core/src:${LTX_ROOT}/packages/ltx-pipelines/src:${LTX_ROOT}/packages/ltx-video/src:${PYTHONPATH:-}"

run_train() {
  local config="$1"
  local log_file="$2"
  cd "${TRAINER_DIR}"
  echo "config=${config}" | tee -a "${log_file}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}" | tee -a "${log_file}"
  echo "log_file=${log_file}" | tee -a "${log_file}"
  "${LTX_ENV}/bin/accelerate" launch \
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

if [ "${RUN_DMD_PHASE2:-1}" = "1" ]; then
  DMD_LOG="${LOG_DIR}/train_official8_dmd_phase2_from1500_$(date +%Y%m%d_%H%M%S).out"
  echo "DMD_START $(date --iso-8601=seconds)" | tee -a "${DMD_LOG}"
  run_train "${DMD_CONFIG}" "${DMD_LOG}"
  echo "DMD_DONE $(date --iso-8601=seconds)" | tee -a "${DMD_LOG}"
else
  echo "DMD_SKIP RUN_DMD_PHASE2=${RUN_DMD_PHASE2}"
fi
