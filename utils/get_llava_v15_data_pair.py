import json
import os
import random
from dataclasses import asdict, dataclass, field

# You need to consider
# DATA_CNT_LIMIT = 8600  # 生成的数据对数
DATA_CNT_LIMIT = 8600
# input_jsonl_path: str | list[str] = "<model_name>_data_pair.jsonl"
input_jsonl_path: str | list[str] = "/home/yilin/SENTINEL/results/LLaVA_v1_5_7b_data_pair.jsonl"
output_json_path = "output.json"

# Don't need to change
CONSIDER_DATA_LIMIT = -1  # 只考虑输入文件的前 x 个字段，若全部考虑，设为 -1
PAD_TO_CNT_LIMIT = False  # 是否在数据数量不足 DATA_CNT_LIMIT 时补全
NEED_SHUFFLE = False  # 是否打乱数据
SEED = 42  # 随机种子
random.seed(SEED)


@dataclass
class Pair:
    question: str
    context: str
    y_win: str
    y_lose: str
    type: str | None = None


@dataclass
class Data:
    image_id: str
    image_path: str
    nonhallu_objects: set[str] = field(default_factory=set)
    hallu_objects: set[str] = field(default_factory=set)
    pairs: list[Pair] = field(default_factory=list)


def read_jsonl(file_path: str) -> list[dict]:
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def save_to_json(data_objects: dict[str, Data], output_path: str) -> None:
    output_dict = {}
    for image_id, data in data_objects.items():
        output_dict[image_id] = [asdict(pair) for pair in data.pairs]

    with open(output_path, "w") as outfile:
        json.dump(output_dict, outfile, indent=4)


def build_data_objects(jsonl_data: list[dict]) -> tuple[dict[str, Data], int]:
    data_dict: dict[str, Data] = {}  # image_id -> Data
    data_count = 0

    if NEED_SHUFFLE:
        random.shuffle(jsonl_data)

    for entry in jsonl_data[:CONSIDER_DATA_LIMIT]:
        if data_count >= DATA_CNT_LIMIT:
            break

        image_id = entry["image_id"]
        if image_id not in data_dict:
            data_dict[image_id] = Data(image_id=image_id, image_path=entry["image_path"])

        data_dict[image_id].nonhallu_objects.update(entry["nonhallu_objects"])
        data_dict[image_id].hallu_objects.update(entry["hallu_objects_of_y_lose"])

        pair = Pair(
            question=entry["question"],
            context=entry["context"],
            y_win=entry["y_win"],
            y_lose=entry["y_lose"],
            type=entry["type"],
        )
        data_dict[image_id].pairs.append(pair)
        data_count += 1

    return data_dict, data_count


def pad_data_objects(data_objects: dict[str, Data], target_count: int) -> dict[str, Data]:
    all_pairs = [pair for data in data_objects.values() for pair in data.pairs]
    current_count = len(all_pairs)

    if current_count >= target_count:
        return data_objects

    to_add = target_count - current_count
    additional_pairs = random.choices(all_pairs, k=to_add)

    padded_data_objects = {**data_objects}
    for pair in additional_pairs:
        image_ids = list(padded_data_objects.keys())
        chosen_image_id = random.choice(image_ids)
        padded_data_objects[chosen_image_id].pairs.append(pair)

    return padded_data_objects


def main(input_path: str | list[str], output_path: str) -> None:
    assert not os.path.exists(output_json_path), f"Output file already exists: {output_json_path}"
    if isinstance(input_path, str):
        input_path = [input_path]

    data_to_process = []
    for path in input_path:
        assert os.path.exists(path), f"Input file not found: {path}"
        data_to_process.extend(read_jsonl(path))

    data_objects, actual_count = build_data_objects(data_to_process)

    if PAD_TO_CNT_LIMIT and actual_count < DATA_CNT_LIMIT:
        data_objects = pad_data_objects(data_objects, DATA_CNT_LIMIT)

    save_to_json(data_objects, output_path)


if __name__ == "__main__":
    main(input_jsonl_path, output_json_path)
