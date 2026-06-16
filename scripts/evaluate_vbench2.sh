#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VBENCH_ROOT="${VBENCH_ROOT:-${PROJECT_ROOT}/external/VBench-2.0}"
VBENCH_ENV="${VBENCH_ENV:-${VBENCH_ROOT}/.venv}"

# Activate VBench environment.
source "${VBENCH_ENV}/bin/activate"
cd "${VBENCH_ROOT}"

# Point to local checkpoints/models and disable Hugging Face online verification/retries.
export VBENCH2_CACHE_DIR="${VBENCH2_CACHE_DIR:-${VBENCH_ROOT}/models/vbench}"
export HF_HOME="${HF_HOME:-${VBENCH_ROOT}/models/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${VBENCH_ROOT}/models/torch}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1

#改这里即可，其他都不要动
VIDEOS_ROOT="${VIDEOS_ROOT:-${PROJECT_ROOT}/outputs/vbench/distill_lora_480P}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results/official8_vbench2_baseline}"


# Define VBench dimensions
dimensions=(
    "Human_Anatomy" "Human_Identity" "Human_Clothes" "Diversity" "Composition"
    "Dynamic_Spatial_Relationship" "Dynamic_Attribute" "Motion_Order_Understanding"
    "Human_Interaction" "Complex_Landscape" "Complex_Plot" "Camera_Motion"
    "Motion_Rationality" "Mechanics" "Thermotics" "Material"
    "Multi-View_Consistency"
)

mkdir -p "$OUTPUT_ROOT"

# Declare expected video counts for standard VBench 2.0
declare -A expected_counts=(
    ["Camera_Motion"]=324
    ["Complex_Landscape"]=90
    ["Complex_Plot"]=180
    ["Composition"]=159
    ["Diversity"]=200
    ["Dynamic_Attribute"]=273
    ["Dynamic_Spatial_Relationship"]=207
    ["Human_Anatomy"]=360
    ["Human_Clothes"]=225
    ["Human_Identity"]=138
    ["Human_Interaction"]=300
    ["Instance_Preservation"]=171
    ["Material"]=150
    ["Mechanics"]=171
    ["Motion_Order_Understanding"]=297
    ["Motion_Rationality"]=174
    ["Multi-View_Consistency"]=291
    ["Thermotics"]=150
)

echo "==== 正在校验视频生成完整度 (检查 VIDEOS_ROOT) ===="
all_valid=true
for dim in "${!expected_counts[@]}"; do
    expected=${expected_counts[$dim]}
    dim_dir="${VIDEOS_ROOT}/${dim}"
    if [ ! -d "$dim_dir" ]; then
        echo "[错误] 维度目录不存在: ${dim_dir}"
        all_valid=false
        continue
    fi
    # Count .mp4 files
    actual=$(find "$dim_dir" -maxdepth 1 -name "*.mp4" 2>/dev/null | wc -l)
    if [ "$actual" -lt "$expected" ]; then
        echo "[错误] 维度 '${dim}' 视频不完整！预期数量: ${expected}，实际数量: ${actual}。"
        all_valid=false
    fi
done

if [ "$all_valid" = false ]; then
    echo -e "\n========================================================"
    echo "[⚠️ 严重警告] 视频文件未完全生成！评测已被终止。"
    echo "请在视频生成逻辑中重新生成缺少的视频后，再重新运行此评估脚本。"
    echo "========================================================"
    exit 1
fi
echo "==== 视频完整度校验通过！所有 18 个维度的视频已完全生成完毕 ===="


# Healthy GPUs to use (cards 0 to 6)
gpus=(0 1 2 3 4 5 6 7)
num_gpus=${#gpus[@]}

echo "==== 开始 17 个基础维度的背景并发评测 (断点评估已开启) ===="
active_jobs=0
for i in "${!dimensions[@]}"; do
    dimension=${dimensions[i]}
    gpu_id=${gpus[i % num_gpus]}

    videos_path="${VIDEOS_ROOT}/${dimension}"
    output_path="${OUTPUT_ROOT}/${dimension}"

    # 断点检测：如果已经有生成好的 eval_results.json 结果文件，则跳过评估
    if ls "${output_path}"/*eval_results.json 1>/dev/null 2>&1; then
        echo "[断点跳过] 维度 '${dimension}' 已经评估完成，跳过计算。"
        continue
    fi

    mkdir -p "$output_path"
    echo "[评估启动] 维度 '${dimension}' 分配至 GPU ${gpu_id}，视频源: ${videos_path}"

    CUDA_VISIBLE_DEVICES=$gpu_id python evaluate.py \
        --videos_path "$videos_path" \
        --dimension "$dimension" \
        --output_path "$output_path" \
        --load_ckpt_from_local True > "${output_path}/eval_run.log" 2>&1 &

    active_jobs=$((active_jobs + 1))
    sleep 2
done

# Function to check if GPUs 4, 5, 6, 7 are idle (defined as < 4GB memory used)
check_gpus_idle() {
    local target_gpus=("4" "5" "6" "7")
    for gpu in "${target_gpus[@]}"; do
        # Query memory used in MiB
        local mem_used
        mem_used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[[:space:]]')
        if [ -n "$mem_used" ]; then
            if [ "$mem_used" -gt 4000 ]; then # Threshold: 4000 MiB (approx 4GB)
                return 1 # Not idle
            fi
        fi
    done
    return 0 # All idle
}

# ==================== STEP 2: 等待 GPU 4,5,6,7 空闲并启动 IP 评估 ====================
ip_dimension="Instance_Preservation"
ip_videos_path="${VIDEOS_ROOT}/${ip_dimension}"
ip_output_path="${OUTPUT_ROOT}/${ip_dimension}"

echo -e "\n==== 开始 Instance_Preservation 评估准备 ===="
if ls "${ip_output_path}"/*eval_results.json 1>/dev/null 2>&1; then
    echo "[断点跳过] 维度 '${ip_dimension}' 已经评估完成，跳过计算。"
else
    # Polling GPUs 4, 5, 6, 7 until idle
    echo "正在监控 GPU 4, 5, 6, 7 显存占用情况，等待其空闲以避免抢占或 OOM (安全显存占用上限门槛: 4GB)..."
    while true; do
        if check_gpus_idle; then
            echo "GPU 4, 5, 6, 7 已全部空闲！立即启动 '${ip_dimension}' 并行评估！"
            break
        fi
        echo "GPU 4, 5, 6, 7 尚未空闲，30 秒后再次检测..."
        sleep 30
    done

    mkdir -p "$ip_output_path"
    echo "[评估启动] 维度 '${ip_dimension}' 运行于 GPU 4,5,6,7 并发 8 个 Workers..."

    python evaluate.py \
        --videos_path "$ip_videos_path" \
        --dimension "$ip_dimension" \
        --output_path "$ip_output_path" \
        --load_ckpt_from_local True \
        --num_workers 8 \
        --gpu_ids "4,5,6,7" > "${ip_output_path}/eval_run_parallel.log" 2>&1

    echo "Instance_Preservation 并行评估完毕！"
fi

# Wait for any remaining background tasks to be absolutely sure
wait

# ==================== STEP 3: 直接调用本地中文打分脚本 ====================
echo -e "\n==== 评测全部结束，直接调用本地中文翻译与聚合逻辑进行打分 ===="
python3 "${VBENCH_ROOT}/cal_local_scores.py" --dir "$OUTPUT_ROOT" 2>&1 | tee "${OUTPUT_ROOT}/cal_local_scores.log"

echo "==== 评测与打分完整流程已结束，日志已写入 ${OUTPUT_ROOT}/cal_local_scores.log ===="
