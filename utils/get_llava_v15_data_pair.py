"""
LLaVA v1.5 DPO 训练数据对生成脚本
==================================

本脚本将 SENTINEL 生成的偏好对数据（JSONL 格式）转换为 DPO 训练所需的格式（JSON 格式）。

数据流程:
    输入: LLaVA_v1_5_7b_data_pair.jsonl (SENTINEL 生成的句子级偏好对)
         ↓
    处理: 按 image_id 聚合，构建 Data 对象
         ↓
    输出: output.json (DPO 训练格式)

输入格式 (JSONL，每行一个):
    {
        "image_id": "COCO_train2014_000000123456",
        "image_path": "/path/to/image.jpg",
        "question": "Describe this image.",
        "context": "The image shows...",           # 上下文（y_win 的前缀）
        "y_win": "a cat sitting on a couch.",     # 正样本（无幻觉句子）
        "y_lose": "a dog running in the park.",   # 负样本（有幻觉句子）
        "type": "y+",                              # 偏好对类型
        "nonhallu_objects": ["cat", "couch"],     # 正样本中的物体
        "hallu_objects_of_y_lose": ["dog", "park"] # 负样本中的幻觉物体
    }

输出格式 (JSON):
    {
        "COCO_train2014_000000123456": [
            {
                "question": "Describe this image.",
                "context": "The image shows...",
                "y_win": "a cat sitting on a couch.",
                "y_lose": "a dog running in the park.",
                "type": "y+"
            },
            ... # 该图像的其他偏好对
        ],
        ... # 其他图像
    }

使用方法:
    1. 修改 input_jsonl_path 为输入文件路径
    2. 修改 DATA_CNT_LIMIT 为需要的数据对数量
    3. 运行: python get_llava_v15_data_pair.py
    4. 输出文件: output.json
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
    - 脚本会从输入文件中读取偏好对，直到达到此数量
    - 如果输入数据不足，可开启 PAD_TO_CNT_LIMIT 进行补全
    - 建议根据训练需求和 GPU 显存调整此值
"""

input_jsonl_path: str | list[str] = "/home/yilin/SENTINEL/results/LLaVA_v1_5_7b_data_pair.jsonl"
"""
输入文件路径

格式:
    - 单个文件: str 类型，如 "/path/to/file.jsonl"
    - 多个文件: list[str] 类型，如 ["/path/to/file1.jsonl", "/path/to/file2.jsonl"]
    
说明:
    - 支持合并多个 JSONL 文件的数据
    - 文件格式必须是 SENTINEL 生成的 _data_pair.jsonl
"""

output_json_path = "output.json"
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
    - y_win: 人类偏好的回答（无幻觉）
    - y_lose: 不被偏好的回答（有幻觉）
    
    训练目标: 让模型学会生成更像 y_win、远离 y_lose 的回答
    
    Attributes:
        question: 用户问题，如 "Describe this image in detail."
        context: 回答的上下文/前缀，y_win 和 y_lose 共享相同的 context
                这确保了两个回答只在关键部分（context 之后）有差异
        y_win: 正样本句子，不包含幻觉物体的描述
        y_lose: 负样本句子，包含幻觉物体的描述
        type: 偏好对类型标识
              - "y+": y_win 是原始生成的无幻觉句子
              - "y-": y_lose 是原始生成的有幻觉句子
              - None: 未指定类型
    
    Example:
        Pair(
            question="What is in this image?",
            context="The image shows a living room with",
            y_win="a cat sleeping on the sofa.",      # 图中确实有猫和沙发
            y_lose="a dog playing with a ball.",      # 图中没有狗和球（幻觉）
            type="y+"
        )
    """
    question: str
    context: str
    y_win: str
    y_lose: str
    type: str | None = None


@dataclass
class Data:
    """
    单张图像的完整数据结构
    
    聚合了一张图像相关的所有偏好对，用于按图像组织训练数据。
    
    Attributes:
        image_id: 图像唯一标识符，如 "COCO_train2014_000000123456"
        image_path: 图像文件路径
        nonhallu_objects: 该图像中检测到的非幻觉物体集合（真实存在的物体）
        hallu_objects: 该图像相关偏好对中出现的所有幻觉物体集合
        pairs: 该图像的所有偏好对列表
    
    Note:
        - nonhallu_objects 和 hallu_objects 目前主要用于统计和调试
        - 实际训练只使用 pairs 中的数据
    """
    image_id: str
    image_path: str
    nonhallu_objects: set[str] = field(default_factory=set)
    hallu_objects: set[str] = field(default_factory=set)
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
    
    Example:
        >>> data = read_jsonl("data.jsonl")
        >>> print(data[0])  # 第一行数据
        {"image_id": "xxx", "question": "...", ...}
    """
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


def save_to_json(data_objects: dict[str, Data], output_path: str) -> None:
    """
    将数据对象保存为 JSON 文件
    
    输出格式为按 image_id 组织的字典，每个 image_id 对应一个偏好对列表。
    这种格式便于 DPO 训练时按图像批量加载数据。
    
    Args:
        data_objects: image_id -> Data 的映射字典
        output_path: 输出 JSON 文件路径
    
    输出格式:
        {
            "image_id_1": [
                {"question": "...", "context": "...", "y_win": "...", "y_lose": "...", "type": "..."},
                ...
            ],
            "image_id_2": [...],
            ...
        }
    """
    output_dict = {}
    for image_id, data in data_objects.items():
        # 将 Pair 对象转换为字典，只保留偏好对信息
        output_dict[image_id] = [asdict(pair) for pair in data.pairs]

    with open(output_path, "w") as outfile:
        json.dump(output_dict, outfile, indent=4)


# ==================== 核心处理函数 ====================

def build_data_objects(jsonl_data: list[dict]) -> tuple[dict[str, Data], int]:
    """
    从 JSONL 数据构建 Data 对象字典
    
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
    
    数据聚合逻辑:
        - 同一 image_id 的多个偏好对会被聚合到同一个 Data 对象
        - nonhallu_objects 和 hallu_objects 会累积合并
        - 每个偏好对作为独立的 Pair 添加到 pairs 列表
    """
    data_dict: dict[str, Data] = {}  # image_id -> Data 映射
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
        
        # 如果是新图像，创建 Data 对象
        if image_id not in data_dict:
            data_dict[image_id] = Data(
                image_id=image_id, 
                image_path=entry["image_path"]
            )

        # 更新该图像的物体集合（用于统计）
        data_dict[image_id].nonhallu_objects.update(entry["nonhallu_objects"])
        data_dict[image_id].hallu_objects.update(entry["hallu_objects_of_y_lose"])

        # 创建并添加偏好对
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
    """
    填充数据对象到目标数量
    
    当实际数据量不足 target_count 时，通过随机复制现有偏好对来补全。
    这在数据稀缺时可以增加训练数据量，但可能导致过拟合。
    
    Args:
        data_objects: 现有的 image_id -> Data 映射
        target_count: 目标偏好对数量
    
    Returns:
        填充后的 data_objects（原地修改 + 返回）
    
    填充策略:
        1. 收集所有现有的偏好对
        2. 随机选择需要补充的数量
        3. 随机将补充的偏好对分配到现有的图像中
    
    Note:
        - 填充的偏好对会被随机分配到已有的图像中
        - 这意味着某些图像可能会有重复的偏好对
        - 建议在数据充足时将 PAD_TO_CNT_LIMIT 设为 False
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
    
    Raises:
        AssertionError: 输出文件已存在或输入文件不存在时抛出
    """
    # 安全检查：防止覆盖已有文件
    assert not os.path.exists(output_json_path), f"Output file already exists: {output_json_path}"
    
    # 统一处理为列表格式
    if isinstance(input_path, str):
        input_path = [input_path]

    # 读取并合并所有输入文件
    data_to_process = []
    for path in input_path:
        assert os.path.exists(path), f"Input file not found: {path}"
        data_to_process.extend(read_jsonl(path))

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
