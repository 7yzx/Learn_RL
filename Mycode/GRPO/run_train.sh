#!/bin/bash
# ============================================================================
# GRPO 训练启动脚本
# 使用 accelerate + DeepSpeed ZeRO-2 分布式训练
# ============================================================================

set -e

# ---------- 可配置参数 ----------
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}  # 自动检测 GPU 数量
MASTER_PORT=${MASTER_PORT:-29500}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo " GRPO Training Launcher"
echo " GPUs: ${NUM_GPUS}"
echo " DeepSpeed: ZeRO-2"
echo "============================================"

# ---------- 单卡训练 ----------
if [ "${NUM_GPUS}" -le 1 ]; then
    echo ">> Single-GPU mode"
    python "${SCRIPT_DIR}/train_grpo.py"
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
    "${SCRIPT_DIR}/train_grpo.py"
