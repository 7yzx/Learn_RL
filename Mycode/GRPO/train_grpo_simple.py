"""
GRPO Training for Qwen2.5-VL on GSM8K (无 DeepSpeed 版本)
========================================================

纯 PyTorch 原生训练, 不依赖 DeepSpeed。
适合单卡调试 或 多卡 DDP 训练。

显存优化策略:
- gradient_checkpointing: 用计算换显存, 减少 ~40% 显存
- bf16 混合精度: 参数用 BF16 存储, 减少一半显存
- gradient_accumulation: 小 micro-batch + 梯度累积 = 大有效 batch

与 DeepSpeed 版本的区别:
- 没有 ZeRO 优化器分片 → 单卡需要更多显存
- 没有 CPU offload → 优化器状态全在 GPU
- 更简单, 调试更方便, 适合入门学习

参考: DeepSeekMath (2402.03300)
"""

import re
import argparse
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig


# ============================================================================
# 1. 数据预处理: 加载 GSM8K 并转换为 GRPO 所需格式
# ============================================================================

def build_prompt(question: str) -> list[dict]:
    """
    构造 chat 格式的 prompt。

    system prompt 引导模型:
    - 在 <think>...</think> 中进行推理
    - 最终答案放在 \\boxed{} 中
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful math assistant. "
                "Reason step by step inside <think>...</think> tags, "
                "then give the final numerical answer inside \\boxed{}."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]


def extract_ground_truth(answer_text: str) -> str:
    """
    从 GSM8K 的 answer 字段中提取数值答案。

    GSM8K 格式: "...#### 42"  →  提取 "42"
    """
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_text.strip()


def prepare_dataset(split: str = "train", data_path: str | None = None) -> Dataset:
    """
    加载 GSM8K 数据集并转换为 GRPOTrainer 所需的格式。

    Args:
        split: 数据集划分 ("train" / "test")
        data_path: 本地数据集路径, 支持多种格式:
            1. save_to_disk Arrow 目录 (含 train/ test/ 子目录)
            2. HuggingFace Hub clone 的仓库 (含 main/ 子目录)
            3. 直接包含 parquet/jsonl 文件的目录
            4. 单个数据文件
            5. None → 从 HuggingFace Hub 在线下载
    """
    if data_path is not None:
        local = Path(data_path)

        if not local.exists():
            raise FileNotFoundError(f"本地数据路径不存在: {data_path}")

        # ---- 情况 1: save_to_disk 的 Arrow 目录 ----
        if (local / split).is_dir() and list((local / split).glob("*.arrow")):
            ds = Dataset.load_from_disk(str(local / split))

        # ---- 情况 2: Hub clone 的仓库 (含 config 子目录如 main/) ----
        elif any(local.glob("*/{}*".format(split))):
            config_name = None
            for sub in local.iterdir():
                if sub.is_dir() and list(sub.glob(f"{split}-*")):
                    config_name = sub.name
                    break
            ds = load_dataset(str(local), name=config_name, split=split)

        # ---- 情况 3: 目录下直接有数据文件 ----
        elif local.is_dir():
            matched = list(local.glob(f"{split}*.*"))
            if matched:
                ds = load_dataset(
                    matched[0].suffix.lstrip("."),
                    data_files=[str(f) for f in matched],
                    split="train",
                )
            else:
                ds = load_dataset(str(local), split=split)

        # ---- 情况 4: 单个文件 ----
        elif local.is_file():
            ds = load_dataset(local.suffix.lstrip("."), data_files=str(local), split="train")

        else:
            raise FileNotFoundError(f"无法从该路径加载数据: {data_path}")

        print(f"[DATA] 从本地加载: {data_path} (split={split}, 样本数={len(ds)})")
    else:
        ds = load_dataset("openai/gsm8k", "main", split=split)
        print(f"[DATA] 从 Hub 加载: openai/gsm8k (split={split}, 样本数={len(ds)})")

    def transform(example):
        return {
            "prompt": build_prompt(example["question"]),
            "ground_truth": extract_ground_truth(example["answer"]),
        }

    ds = ds.map(transform, remove_columns=ds.column_names)
    return ds


# ============================================================================
# 2. 奖励函数
# ============================================================================

def accuracy_reward(completions: list, ground_truth: list[str], **kwargs) -> list[float]:
    """
    准确性奖励: 从 \\boxed{} 提取答案, 与 ground_truth 精确匹配。
    正确 → 1.0, 错误 → 0.0
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        match = re.search(r"\\boxed\{([^}]*)\}", text)
        predicted = match.group(1).strip().replace(",", "") if match else ""
        rewards.append(1.0 if predicted == gt else 0.0)
    return rewards


def format_reward(completions: list, **kwargs) -> list[float]:
    """
    格式奖励: 鼓励结构化输出。
    有 <think>...</think> → +0.5, 有 \\boxed{} → +0.5
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        score = 0.0
        if "<think>" in text and "</think>" in text:
            score += 0.5
        if re.search(r"\\boxed\{.+\}", text):
            score += 0.5
        rewards.append(score)
    return rewards


# ============================================================================
# 3. 训练配置与启动
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training (无 DeepSpeed)")
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="本地模型路径, 不指定则从 Hub 下载 Qwen/Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="本地 GSM8K 数据集路径, 不指定则从 Hub 下载"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/qwen25vl_gsm8k_grpo_simple",
        help="训练输出目录"
    )
    parser.add_argument(
        "--use_4bit", action="store_true",
        help="启用 4-bit 量化加载模型 (QLoRA 风格, 大幅降低显存)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 模型路径 ----
    model_name = args.model_path or "Qwen/Qwen2.5-VL-3B-Instruct"
    print(f"[MODEL] 模型路径: {model_name}")

    # ---- 准备数据集 ----
    train_dataset = prepare_dataset("train", data_path=args.data_path)
    eval_dataset  = prepare_dataset("test",  data_path=args.data_path)

    # ---- 训练参数 (不使用 DeepSpeed) ----
    training_args = GRPOConfig(
        output_dir=args.output_dir,

        # === 训练超参 ===
        num_train_epochs=2,
        per_device_train_batch_size=1,          # 无 ZeRO → 用更小的 batch
        gradient_accumulation_steps=8,           # 补偿小 batch, 有效 batch = 1*8 = 8
        learning_rate=1e-6,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        bf16=True,

        # === GRPO 特有参数 ===
        num_generations=8,                       # 无 ZeRO 显存紧张, 减少到 G=8
        max_completion_length=1024,
        beta=0.001,
        loss_type="dapo",
        scale_rewards=True,

        # === 显存优化 (替代 DeepSpeed 的方案) ===
        gradient_checkpointing=True,             # 用计算换显存, 减少 ~40%
        # 注意: 这里没有 deepspeed 参数!

        # === 日志与保存 ===
        logging_steps=5,
        save_strategy="steps",
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        log_completions=True,
        report_to="none",

        # === 其他 ===
        seed=42,
        remove_unused_columns=False,
    )


    # ---- 加载 processor ----
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ---- 创建 Trainer ----
    trainer = GRPOTrainer(
        model=model_name,
        args=training_args,
        processing_class=processor,
        reward_funcs = [
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # ---- 开始训练 ----
    print("=" * 60)
    print("Starting GRPO Training (无 DeepSpeed)")
    print(f"  Model:             {model_name}")
    print(f"  Grad checkpointing: True")
    print(f"  Dataset:           GSM8K (train={len(train_dataset)}, eval={len(eval_dataset)})")
    print(f"  Generations/prompt: {training_args.num_generations}")
    print(f"  Effective batch:   {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print("=" * 60)

    trainer.train()

    # ---- 保存 ----
    final_dir = str(Path(args.output_dir) / "final")
    trainer.save_model(final_dir)
    print(f"Training complete. Model saved to {final_dir}")


if __name__ == "__main__":
    main()
