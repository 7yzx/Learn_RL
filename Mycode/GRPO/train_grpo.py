"""
GRPO Training for Qwen2.5-VL on GSM8K
======================================

使用 TRL 库的 GRPOTrainer 对 Qwen2.5-VL 模型在 GSM8K 数据集上进行
Group Relative Policy Optimization (GRPO) 强化学习训练。

GRPO 算法核心思想:
1. 对每个 prompt 采样一组 (G 个) completions
2. 用 reward function 对每个 completion 打分
3. 在组内计算相对优势 (advantage): A_i = (r_i - mean(r)) / std(r)
4. 用 clipped surrogate objective 更新策略，保持策略不偏离太远

参考: DeepSeekMath (2402.03300)
"""

import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoProcessor
from trl import GRPOTrainer, GRPOConfig


# ============================================================================
# 1. 数据预处理: 加载 GSM8K 并转换为 GRPO 所需格式
# ============================================================================

def build_prompt(question: str) -> list[dict]:
    """
    构造 chat 格式的 prompt。
    
    system prompt 引导模型使用 <think>...</think> 进行推理，
    最终把答案放在 \\boxed{} 中。
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


def prepare_dataset(split: str = "train") -> Dataset:
    """
    加载 GSM8K 数据集并转换为 GRPOTrainer 所需的格式。
    
    GRPOTrainer 要求:
    - `prompt` 列: 对话格式的 prompt (list[dict])
    - 其他列: 可传递给 reward function 的额外信息
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)

    def transform(example):
        return {
            "prompt": build_prompt(example["question"]),
            "ground_truth": extract_ground_truth(example["answer"]),
        }

    ds = ds.map(transform, remove_columns=ds.column_names)
    return ds


# ============================================================================
# 2. 奖励函数: 评估模型输出的正确性和格式
# ============================================================================

def accuracy_reward(completions: list, ground_truth: list[str], **kwargs) -> list[float]:
    """
    准确性奖励: 检查模型输出是否包含正确答案。
    
    从 completion 中提取 \\boxed{...} 内容，与 ground_truth 比较。
    - 完全匹配 → 1.0
    - 不匹配   → 0.0
    """
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        # 提取 completion 文本 (兼容 str 和 chat 格式)
        text = completion[0]["content"] if isinstance(completion, list) else completion
        # 尝试匹配 \boxed{...}
        match = re.search(r"\\boxed\{([^}]*)\}", text)
        predicted = match.group(1).strip().replace(",", "") if match else ""
        rewards.append(1.0 if predicted == gt else 0.0)
    return rewards


def format_reward(completions: list, **kwargs) -> list[float]:
    """
    格式奖励: 鼓励模型使用 <think>...</think> 和 \\boxed{} 的结构化格式。
    
    检查规则:
    - 包含 <think> 和 </think> 标签 → +0.5
    - 包含 \\boxed{} 答案框        → +0.5
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

def main():
    # ---- 模型名称 ----
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

    # ---- 准备数据集 ----
    train_dataset = prepare_dataset("train")    # ~7.5k 样本
    eval_dataset  = prepare_dataset("test")     # ~1.3k 样本

    # ---- GRPO 训练参数 ----
    training_args = GRPOConfig(
        output_dir="./output/qwen25vl_gsm8k_grpo",

        # === 训练超参 ===
        num_train_epochs=2,
        per_device_train_batch_size=2,         # 每卡 batch size
        gradient_accumulation_steps=4,          # 有效 batch = 2 * 4 * n_gpu
        learning_rate=1e-6,                     # GRPO 推荐较小学习率
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        bf16=True,                              # 使用 bfloat16 混合精度
        
        # === GRPO 特有参数 ===
        num_generations=8,                      # 每个 prompt 生成 G=8 个候选
        max_completion_length=1024,             # 最大生成长度
        beta=0.001,                             # KL 散度系数 (约束策略偏移)
        loss_type="dapo",                       # DAPO loss: 消除长度偏差
        scale_rewards=True,                     # 组内标准差归一化
        
        # === DeepSpeed ===
        deepspeed="ds_config_zero2.json",       # ZeRO-2 + CPU offload

        # === 日志与保存 ===
        logging_steps=5,
        save_strategy="steps",
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        log_completions=True,                   # 打印生成样例到日志
        report_to="tensorboard",

        # === 其他 ===
        seed=42,
        remove_unused_columns=False,            # 保留 ground_truth 列给 reward
    )

    # ---- 加载 processor (tokenizer + image_processor) ----
    processor = AutoProcessor.from_pretrained(model_name)
    # GRPOTrainer 要求 padding_side = "left"
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ---- 创建 Trainer ----
    trainer = GRPOTrainer(
        model=model_name,
        args=training_args,
        processing_class=processor,
        reward_funcs=[accuracy_reward, format_reward],  # 多奖励函数, 分数累加
        reward_weights=[1.0, 0.5],                      # 准确性权重 > 格式权重
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # ---- 开始训练 ----
    print("=" * 60)
    print("Starting GRPO Training")
    print(f"  Model:         {model_name}")
    print(f"  Dataset:       GSM8K (train={len(train_dataset)}, eval={len(eval_dataset)})")
    print(f"  Generations/prompt: {training_args.num_generations}")
    print(f"  Loss type:     {training_args.loss_type}")
    print(f"  DeepSpeed:     ZeRO-2")
    print("=" * 60)

    trainer.train()

    # ---- 保存最终模型 ----
    trainer.save_model("./output/qwen25vl_gsm8k_grpo/final")
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()
