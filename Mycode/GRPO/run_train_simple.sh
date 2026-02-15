#!/bin/bash
# ============================================================================
# GRPO 训练启动脚本 (无 DeepSpeed 版本)
#
# 单卡: 直接 python 运行
# 多卡: 用 torchrun 启动 DDP (PyTorch 原生分布式)
#
# 用法:
#   # 单卡 (自动检测)
#   bash run_train_simple.sh
#
#   # 指定 GPU 数量
#   NUM_GPUS=2 bash run_train_simple.sh
#
#   # 本地模型 + 数据 + 4bit 量化
#   MODEL_PATH=/data/models/Qwen2.5-VL-3B-Instruct \
#   DATA_PATH=/data/datasets/gsm8k \
#   USE_4BIT=1 \
#       bash run_train_simple.sh
# ============================================================================

set -e

# ---------- 可配置参数 ----------
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
MASTER_PORT=${MASTER_PORT:-29500}
MODEL_PATH=${MODEL_PATH:-""}
DATA_PATH=${DATA_PATH:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/qwen25vl_gsm8k_grpo_simple"}
USE_4BIT=${USE_4BIT:-""}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------- 构造参数 ----------
TRAIN_ARGS=""
[ -n "${MODEL_PATH}" ] && TRAIN_ARGS="${TRAIN_ARGS} --model_path ${MODEL_PATH}"
[ -n "${DATA_PATH}" ]  && TRAIN_ARGS="${TRAIN_ARGS} --data_path ${DATA_PATH}"
[ -n "${USE_4BIT}" ]   && TRAIN_ARGS="${TRAIN_ARGS} --use_4bit"
TRAIN_ARGS="${TRAIN_ARGS} --output_dir ${OUTPUT_DIR}"

echo "============================================"
echo " GRPO Training (No DeepSpeed)"
echo " GPUs:       ${NUM_GPUS}"
echo " Model:      ${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct (Hub)}"
echo " Data:       ${DATA_PATH:-openai/gsm8k (Hub)}"
echo " 4-bit:      ${USE_4BIT:-no}"
echo " Output:     ${OUTPUT_DIR}"
echo "============================================"

# ---------- 单卡 ----------
if [ "${NUM_GPUS}" -le 1 ]; then
    echo ">> Single-GPU mode"
    python "${SCRIPT_DIR}/train_grpo_simple.py" ${TRAIN_ARGS}
    exit 0
fi

# ---------- 多卡 DDP (torchrun) ----------
echo ">> Multi-GPU DDP mode (${NUM_GPUS} GPUs)"
torchrun \
    --nproc_per_node "${NUM_GPUS}" \
    --master_port "${MASTER_PORT}" \
    "${SCRIPT_DIR}/train_grpo_simple.py" ${TRAIN_ARGS}
