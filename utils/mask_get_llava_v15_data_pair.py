"""
Mask DPO 训练数据转换脚本
==========================

本脚本将 SENTINEL mask 模式生成的幻觉检测数据（JSONL 格式）转换为 DPO 训练所需的格式（JSON 格式）。

数据流程:
    输入: mask_preference_pairs.jsonl (mask 模式生成的幻觉检测数据)
         ↓
    处理: 按 image_id 聚合，构建 Data 对象
         ↓
    输出: mask_output.json (DPO 训练格式)

输入格式 (mask_preference_pairs.jsonl，每行一个):
    {
        "image_id": "5",
        "image_path": "/path/to/original_image.jpg",
        "question": "What is this photo about?",
        "masked_object": "person",                    # 被遮挡的物体
        "masked_image_path": "/path/to/masked.jpg",  # 遮挡后的图像路径
        "detection_scores": [...],
        "detection_source": "both",
        "hallu_ratio": 1.0,                          # 幻觉比例
        "total_samples": 10,
        "hallu_responses": ["...", "..."],           # 幻觉回复列表
        "ground_truth": "..."                        # 正确的描述（无幻觉）
    }

输出格式 (JSON):
    {
        "/path/to/original_image.jpg": [  # 使用原图像路径作为 key
            {
                "question": "What is this photo about?",
                "context": null,
                "y_win": "The correct description...",  # ground_truth
                "y_lose": null,
                "type": "y+",
                "masked_image": "/path/to/masked_image.jpg"  # 遮挡后的图像路径
            },
            ...
        ],
        ...
    }

使用方法:
    1. 修改 input_jsonl_path 为输入文件路径
    2. 修改 DATA_CNT_LIMIT 为需要的数据对数量
    3. 运行: python mask_get_llava_v15_data_pair.py
    4. 输出文件: mask_output.json
"""

import json
import os
import random
from dataclasses import asdict, dataclass, field


# ==================== 配置参数 ====================

DATA_CNT_LIMIT = 8600
"""
生成的数据对数量上限

说明:
    - 脚本会从输入文件中读取数据，直到达到此数量
    - 如果输入数据不足，可开启 PAD_TO_CNT_LIMIT 进行补全
    - 建议根据训练需求和 GPU 显存调整此值
"""

input_jsonl_path: str | list[str] = "/home/yilin/SENTINEL/results/LLaVA_v1_5_7b_mask/mask_preference_pairs.jsonl"
"""
输入文件路径

格式:
    - 单个文件: str 类型，如 "/path/to/file.jsonl"
    - 多个文件: list[str] 类型，如 ["/path/to/file1.jsonl", "/path/to/file2.jsonl"]
    
说明:
    - 支持合并多个 JSONL 文件的数据
    - 文件格式必须是 SENTINEL mask 模式生成的 mask_preference_pairs.jsonl
"""

output_json_path = "mask_output.json"
"""输出文件路径，DPO 训练将使用此文件"""


# ==================== 高级配置（通常无需修改） ====================

CONSIDER_DATA_LIMIT = -1
"""
只考虑输入文件的前 x 条数据

说明:
    - 设为 -1 表示考虑全部数据
    - 设为正整数 n 表示只考虑前 n 条（用于调试或快速测试）
"""

PAD_TO_CNT_LIMIT = False
"""
是否在数据数量不足 DATA_CNT_LIMIT 时补全

说明:
    - True: 如果实际数据不足，会随机复制现有数据进行填充
    - False: 有多少数据就输出多少，不进行填充
"""

NEED_SHUFFLE = False
"""
是否打乱数据顺序

说明:
    - True: 在处理前随机打乱输入数据顺序
    - False: 保持输入顺序（通常按图像处理顺序）
"""

SEED = 42
"""随机种子，确保实验可复现"""

random.seed(SEED)


# ==================== 数据结构定义 ====================

@dataclass
class Pair:
    """
    单个偏好对数据结构
    
    在 DPO (Direct Preference Optimization) 训练中，每个 Pair 代表一组偏好数据：
    - y_win: 人类偏好的回答（无幻觉，来自 ground_truth）
    - y_lose: 不被偏好的回答（设置为 None）
    
    训练目标: 让模型学会生成更像 y_win 的回答
    
    Attributes:
        question: 用户问题，如 "What is this photo about?"
        context: 回答的上下文/前缀，设置为 None
        y_win: 正样本句子，来自 ground_truth（正确的描述）
        y_lose: 负样本句子，设置为 None
        type: 偏好对类型标识，默认为 "y+"
        masked_image: 遮挡后的图像路径
    
    Example:
        Pair(
            question="What is in this image?",
            context=None,
            y_win="The image shows a cat sleeping on the sofa.",
            y_lose=None,
            type="y+",
            masked_image="/path/to/masked_image.jpg"
        )
    """
    question: str
    context: str | None
    y_win: str
    y_lose: str | None
    type: str | None = "y+"
    masked_image: str | None = None


@dataclass
class Data:
    """
    单张图像的完整数据结构
    
    聚合了一张图像相关的所有偏好对，用于按图像组织训练数据。
    
    Attributes:
        image_id: 图像唯一标识符，如 "5"
        image_path: 图像文件路径
        masked_objects: 该图像中被检测为幻觉的物体集合
        pairs: 该图像的所有偏好对列表
    """
    image_id: str
    image_path: str
    masked_objects: set[str] = field(default_factory=set)
    pairs: list[Pair] = field(default_factory=list)


# ==================== 工具函数 ====================

def read_jsonl(file_path: str) -> list[dict]:
    """
    读取 JSONL 文件
    
    JSONL (JSON Lines) 格式：每行是一个独立的 JSON 对象，
    适合处理大规模数据，支持流式读取。
    
    Args:
        file_path: JSONL 文件路径
    
    Returns:
        解析后的字典列表，每个字典对应文件中的一行
    """
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_to_json(data_objects: dict[str, Data], output_path: str) -> None:
    """
    将数据对象保存为 JSON 文件
    
    输出格式为按 image_path（原图像路径）组织的字典，每个 image_path 对应一个偏好对列表。
    这种格式便于 DPO 训练时按图像批量加载数据。
    
    Args:
        data_objects: image_path -> Data 的映射字典
        output_path: 输出 JSON 文件路径
    
    输出格式:
        {
            "/path/to/original_image.jpg": [
                {"question": "...", "context": null, "y_win": "...", "y_lose": null, "type": "y+", "masked_image": "/path/to/masked.jpg"},
                ...
            ],
            "/path/to/another_image.jpg": [...],
            ...
        }
    """
    output_dict = {}
    for image_path, data in data_objects.items():
        # 将 Pair 对象转换为字典，只保留偏好对信息
        # 使用 image_path（原图像路径）作为 key
        output_dict[image_path] = [asdict(pair) for pair in data.pairs]

    with open(output_path, "w") as outfile:
        json.dump(output_dict, outfile, indent=4)


# ==================== 核心处理函数 ====================

def build_data_objects(jsonl_data: list[dict]) -> tuple[dict[str, Data], int]:
    """
    从 mask_preference_pairs.jsonl 数据构建 Data 对象字典
    
    处理流程:
        1. （可选）打乱输入数据顺序
        2. 遍历每条数据，按 image_id 聚合
        3. 为每张图像创建 Data 对象，收集偏好对
        4. 达到 DATA_CNT_LIMIT 时停止
    
    Args:
        jsonl_data: 从 JSONL 文件读取的原始数据列表
    
    Returns:
        tuple: (data_dict, data_count)
            - data_dict: image_id -> Data 的映射
            - data_count: 实际处理的偏好对数量
    
    数据转换逻辑:
        - question: 直接使用原数据的 question
        - context: 设置为 None
        - y_win: 使用 ground_truth（正确的无幻觉描述）
        - y_lose: 设置为 None
        - type: 默认为 "y+"
    """
    data_dict: dict[str, Data] = {}  # image_path -> Data 映射（使用原图像路径作为 key）
    data_count = 0  # 已处理的偏好对计数

    # 可选：打乱数据顺序（用于随机采样）
    if NEED_SHUFFLE:
        random.shuffle(jsonl_data)

    # 遍历数据（支持限制处理数量）
    for entry in jsonl_data[:CONSIDER_DATA_LIMIT]:
        # 达到目标数量时停止
        if data_count >= DATA_CNT_LIMIT:
            break

        image_id = entry["image_id"]
        image_path = entry["image_path"]  # 原图像路径，作为 key
        
        # 如果是新图像，创建 Data 对象（使用 image_path 作为 key）
        if image_path not in data_dict:
            data_dict[image_path] = Data(
                image_id=image_id, 
                image_path=image_path
            )

        # 记录被遮挡的物体（用于统计）
        data_dict[image_path].masked_objects.add(entry["masked_object"])

        # 创建偏好对：
        # - y_win 使用 ground_truth（正确的描述）
        # - y_lose 和 context 设置为 None
        # - type 默认为 "y+"
        # - masked_image 使用 masked_image_path（遮挡后的图像路径）
        pair = Pair(
            question=entry["question"],
            context=None,
            y_win=entry["ground_truth"],
            y_lose=None,
            type="y+",
            masked_image=entry["masked_image_path"],
        )
        data_dict[image_path].pairs.append(pair)
        data_count += 1

    return data_dict, data_count


def pad_data_objects(data_objects: dict[str, Data], target_count: int) -> dict[str, Data]:
    """
    填充数据对象到目标数量
    
    当实际数据量不足 target_count 时，通过随机复制现有偏好对来补全。
    
    Args:
        data_objects: 现有的 image_id -> Data 映射
        target_count: 目标偏好对数量
    
    Returns:
        填充后的 data_objects
    """
    # 收集所有现有的偏好对
    all_pairs = [pair for data in data_objects.values() for pair in data.pairs]
    current_count = len(all_pairs)

    # 如果已经满足目标数量，直接返回
    if current_count >= target_count:
        return data_objects

    # 计算需要补充的数量
    to_add = target_count - current_count
    
    # 随机选择要复制的偏好对（允许重复选择）
    additional_pairs = random.choices(all_pairs, k=to_add)

    # 将补充的偏好对随机分配到现有图像
    padded_data_objects = {**data_objects}
    for pair in additional_pairs:
        image_ids = list(padded_data_objects.keys())
        chosen_image_id = random.choice(image_ids)
        padded_data_objects[chosen_image_id].pairs.append(pair)

    return padded_data_objects


# ==================== 主函数 ====================

def main(input_path: str | list[str], output_path: str) -> None:
    """
    主处理函数
    
    完整的数据转换流程:
        1. 检查输出文件是否已存在（防止覆盖）
        2. 读取输入 JSONL 文件（支持多文件合并）
        3. 构建 Data 对象字典
        4. （可选）填充数据到目标数量
        5. 保存为 JSON 格式
    
    Args:
        input_path: 输入 JSONL 文件路径（单个或列表）
        output_path: 输出 JSON 文件路径
    """
    # 安全检查：防止覆盖已有文件
    assert not os.path.exists(output_path), f"Output file already exists: {output_path}"
    
    # 统一处理为列表格式
    if isinstance(input_path, str):
        input_path = [input_path]

    # 读取并合并所有输入文件
    data_to_process = []
    for path in input_path:
        assert os.path.exists(path), f"Input file not found: {path}"
        data_to_process.extend(read_jsonl(path))

    print(f"Loaded {len(data_to_process)} entries from input files")

    # 构建数据对象
    data_objects, actual_count = build_data_objects(data_to_process)
    print(f"Processed {actual_count} preference pairs from {len(data_objects)} images")

    # 可选：填充到目标数量
    if PAD_TO_CNT_LIMIT and actual_count < DATA_CNT_LIMIT:
        data_objects = pad_data_objects(data_objects, DATA_CNT_LIMIT)
        print(f"Padded to {DATA_CNT_LIMIT} preference pairs")

    # 保存结果
    save_to_json(data_objects, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main(input_jsonl_path, output_json_path)
