import json
from datasets import Dataset, DatasetDict


def create_legal_qa_dataset(file_path: str):

    prompt_template = """<s>[INST] 你是一名专业的司法专家，请根据以下法律问题，结合法律原则和司法实践，给出准确、严谨的分析与回答。

### 法律问题：
{question}
[/INST]

### 司法专家回答：
{answer}</s>
"""

    formatted_data = []

    # ⭐ 用 utf-8-sig 解决 BOM 问题
    with open(file_path, "r", encoding="utf-8-sig") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()

            # ✅ 跳过空行
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"❌ 第 {idx} 行 JSON 解析失败，已跳过")
                print(f"内容: {line[:200]}")
                print(f"错误: {e}")
                continue

            if "input" in item and "output" in item:
                full_prompt = prompt_template.format(
                    question=item["input"],
                    answer=item["output"]
                )
                formatted_data.append({"text": full_prompt})

    if len(formatted_data) == 0:
        raise ValueError("❌ 没有成功解析任何数据，请检查 JSONL 文件格式")

    dataset = Dataset.from_list(formatted_data)

    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)

    return DatasetDict({
        "train": train_test_split["train"],
        "validation": train_test_split["test"]
    })


if __name__ == "__main__":
    raw_json_path = "./dataset/train.jsonl"

    print("开始处理数据集...")
    processed_dataset = create_legal_qa_dataset(raw_json_path)

    processed_dataset.save_to_disk("./dataset/processed_data")

    print("数据集处理完毕")
    print(f"训练集大小: {len(processed_dataset['train'])}")
    print(f"验证集大小: {len(processed_dataset['validation'])}")

    print("\n示例样本：")
    print(processed_dataset["train"][0]["text"])
