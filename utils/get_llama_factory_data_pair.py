import base64
import json
import os
import random

# You need to consider
DATA_CNT_LIMIT = 7000  # 生成的数据对数
input_file = "<model_name>_data_pair.jsonl"
output_file = "output.json"

# Don't need to change
CONSIDER_DATA_LIMIT = -1  # 只考虑输入文件的前 x 个字段，若全部考虑，设为 -1
PAD_TO_CNT_LIMIT = False  # 是否在数据数量不足 DATA_CNT_LIMIT 时补全
NEED_SHUFFLE = False  # 是否打乱数据


def read_jsonl(file_path: str) -> list[dict]:
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def convert_jsonl_to_json(input_file, output_file):
    data_to_process = read_jsonl(input_file)

    if NEED_SHUFFLE:
        random.shuffle(data_to_process)

    if CONSIDER_DATA_LIMIT != -1:
        data_to_process = data_to_process[:CONSIDER_DATA_LIMIT]

    result = []
    for entry in data_to_process:
        if len(result) >= DATA_CNT_LIMIT:
            break

        instruction = f"<image>{entry['question']}"
        images = [entry["image_path"]]

        result.append(
            {
                "instruction": instruction,
                "context": entry.get("context", ""),
                "chosen": entry.get("win", ""),
                "rejected": entry.get("lose", ""),
                "images": images,
            }
        )

    if PAD_TO_CNT_LIMIT and len(result) < DATA_CNT_LIMIT:
        additional_entries = random.choices(result, k=DATA_CNT_LIMIT - len(result))
        result.extend(additional_entries)

    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(result, outfile, ensure_ascii=False, indent=4)


def main(input_file: str | list[str], output_file: str) -> None:
    assert not os.path.exists(output_file), f"Output file already exists: {output_file}"
    convert_jsonl_to_json(input_file, output_file)


if __name__ == "__main__":
    main(input_file, output_file)
