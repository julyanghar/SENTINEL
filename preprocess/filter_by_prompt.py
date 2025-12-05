"""
按 prompt 筛选数据脚本
======================

从 mm-ref-data-full.json 中筛选出 prompt 为指定描述类问题的数据。

使用方法:
    python filter_by_prompt.py

输出:
    mm-ref-data-filtered.json - 筛选后的数据
"""

import json

# ==================== 配置区域 ====================

# 输入文件
INPUT_FILE = "mm-ref-data-full.json"

# 输出文件
OUTPUT_FILE = "mm-ref-data-filtered.json"

# 要筛选的 prompt 列表
TARGET_PROMPTS = [
    "What is this photo about? Please answer in great detail.",
    "Describe this image in detail.",
    "Provide a thorough description of the given picture.",
    "Please provide a detailed description of the picture."
]


def filter_by_prompt(input_path: str, output_path: str, target_prompts: list[str]) -> None:
    """
    按 prompt 筛选数据
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        target_prompts: 要保留的 prompt 列表
    """
    # 转换为 set 加速查找
    target_prompts_set = set(target_prompts)
    
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
    
    # 筛选数据
    filtered_data = []
    prompt_counts = {p: 0 for p in target_prompts}
    
    for item in source_data:
        prompt = item.get("prompt", "")
        if prompt in target_prompts_set:
            filtered_data.append(item)
            prompt_counts[prompt] += 1
    
    # 统计信息
    print(f"\n筛选完成！")
    print(f"原始数据量: {len(source_data)}")
    print(f"筛选后数据量: {len(filtered_data)}")
    print(f"\n各 prompt 数量统计:")
    for prompt, count in prompt_counts.items():
        print(f"  - \"{prompt[:50]}...\": {count} 条")
    
    # 保存筛选后的数据
    print(f"\n正在保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"保存完成！")


if __name__ == "__main__":
    filter_by_prompt(INPUT_FILE, OUTPUT_FILE, TARGET_PROMPTS)

