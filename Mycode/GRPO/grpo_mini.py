import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def args():
    parser = argparse.ArgumentParser(description="Train GRPO")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-Math-1.5B", help="预训练模型名称或路径")
    parser.add_argument("--data_path", type=str, default=None, help="GSM8K 数据集路径 (可选)")
    parser.add_argument("--output_dir", type=str, default="./grpo_gsm8k_simple", help="训练输出目录")
    parser.add_argument("--num_generations", type=int, default=4, help="每个 prompt 生成的答案数量")
    return parser.parse_args()


def main():
    model_path = "Qwen/Qwen2-Math-1.5B"
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Input prompts (batch of 2 queries)
    prompts = [
        "Solve y = 2x + 1 for x = 2, y = ",  # Correct answer: 5
        "Solve y = 2x + 1 for x = 4, y = "   # Correct answer: 9
    ]

    inp


