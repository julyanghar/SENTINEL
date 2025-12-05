"""
转换 mm-ref-data-filtered.json 为 mask 模式数据集
================================================

保留字段: id, image, prompt, chosen
输出格式: 
    - image_id: 原 id
    - image_path: 拼接后的完整图片路径
    - question: 从候选列表随机选择
    - ground_truth: 原 chosen

使用方法:
    python convert_masked.py
"""

import json
import random


# Question 候选列表
CANDIDATE_QUESTIONS = [
    "What is this photo about? Please answer in great detail.",
    "Describe this image in detail.",
    "Provide a thorough description of the given picture.",
    "Please provide a detailed description of the picture."
]


def convert_dataset(input_path: str, output_path: str, image_path_prefix: str) -> None:
    """
    将 mm-ref-data-filtered.json 转换为 mask 模式所需格式
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        image_path_prefix: 图片路径前缀
    """
    # 读取原始数据
    print(f"正在读取文件: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        print(f"成功读取 {len(source_data)} 条数据")
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {input_path} 不是有效的 JSON 格式")
        return

    # 设置随机种子，保证每次运行结果一致
    random.seed(42)
    
    target_data = []

    # 遍历并转换格式
    for item in source_data:
        new_item = {
            "image_id": item.get("id"),
            "image_path": f"{image_path_prefix}{item.get('image')}",
            "question": random.choice(CANDIDATE_QUESTIONS),  # 随机选择 question
            "ground_truth": item.get("chosen"),
        }
        target_data.append(new_item)

    # 保存为 JSONL 格式
    print(f"正在保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in target_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"转换完成！共 {len(target_data)} 条数据")


# --- 配置区域 ---
if __name__ == "__main__":
    # 输入文件名
    INPUT_FILE = "mm-ref-data-filtered.json"
    
    # 输出文件名
    OUTPUT_FILE = "mask_data.jsonl"
    
    # 图片路径前缀
    IMG_PREFIX = "/home/yilin/new-DPO/dataset/train2014/"

    # 执行转换
    convert_dataset(INPUT_FILE, OUTPUT_FILE, IMG_PREFIX)
