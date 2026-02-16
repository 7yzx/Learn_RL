from datasets import load_dataset

reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""
import re


def load_dataset(data_path: str, split_: str = "train") -> Dataset:
    """
    加载 GSM8K 数据集。

    Args:
        data_path: 数据集路径或名称
        split: 数据集划分，默认为 "train"

    Returns:
        加载后的 Dataset 对象
    """
    dataset = load_dataset(data_path, "main", split=split_)
    dataset = dataset.map(lambda x: {
        "prompt" : [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
    })




def extract_hash_answer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()


match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)
    return scores

