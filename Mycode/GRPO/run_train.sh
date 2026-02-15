#!/bin/bash
# ============================================================================
# GRPO 训练启动脚本
# 使用 accelerate + DeepSpeed ZeRO-2 分布式训练
#
# 用法:
#   # 在线下载模型和数据 (默认)
#   bash run_train.sh
#
#   # 使用本地模型和数据
#   MODEL_PATH=/data/models/Qwen2.5-VL-3B-Instruct \
#   DATA_PATH=/data/datasets/gsm8k \
#       bash run_train.sh
#
#   # 指定 GPU 数量
#   NUM_GPUS=4 bash run_train.sh
# ============================================================================

set -e

# ---------- 可配置参数 ----------
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}  # 自动检测 GPU 数量
MASTER_PORT=${MASTER_PORT:-29500}
MODEL_PATH=${MODEL_PATH:-""}       # 本地模型路径 (空=从 Hub 下载)
DATA_PATH=${DATA_PATH:-""}         # 本地数据路径 (空=从 Hub 下载)
OUTPUT_DIR=${OUTPUT_DIR:-"./output/qwen25vl_gsm8k_grpo"}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------- 构造训练脚本参数 ----------
TRAIN_ARGS=""
if [ -n "${MODEL_PATH}" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --model_path ${MODEL_PATH}"
fi
if [ -n "${DATA_PATH}" ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --data_path ${DATA_PATH}"
fi
TRAIN_ARGS="${TRAIN_ARGS} --output_dir ${OUTPUT_DIR}"

echo "============================================"
echo " GRPO Training Launcher"
echo " GPUs:       ${NUM_GPUS}"
echo " Model:      ${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct (Hub)}"
echo " Data:       ${DATA_PATH:-openai/gsm8k (Hub)}"
echo " Output:     ${OUTPUT_DIR}"
echo " DeepSpeed:  ZeRO-2"
echo "============================================"

# ---------- 单卡训练 ----------
if [ "${NUM_GPUS}" -le 1 ]; then
    echo ">> Single-GPU mode"
    python "${SCRIPT_DIR}/train_grpo.py" ${TRAIN_ARGS}
    exit 0
fi

# ---------- 多卡训练 (accelerate + deepspeed) ----------
echo ">> Multi-GPU mode (${NUM_GPUS} GPUs)"
accelerate launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file "${SCRIPT_DIR}/ds_config_zero2.json" \
    --main_process_port "${MASTER_PORT}" \
    "${SCRIPT_DIR}/train_grpo.py" ${TRAIN_ARGS}
