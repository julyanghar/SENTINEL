import json
import os
import random  # 1. 引入 random 库

def convert_dataset(input_path, output_path, image_path_prefix):
    """
    将原始数据集格式转换为目标格式，并随机分配 question
    """
    
    # 2. 定义 Question 候选列表
    candidate_questions = [
        "What is this photo about? Please answer in great detail.",
        "Describe this image in detail.",
        "Provide a thorough description of the given picture.",
        "Please provide a detailed description of the picture."
    ]

    # (可选) 设置随机种子，保证每次运行生成的随机结果一致
    random.seed(42)

    # 读取原始数据
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        print(f"成功读取 {len(source_data)} 条数据。")
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {input_path} 不是有效的 JSON 格式。")
        return

    target_data = []

    # 遍历并转换格式
    for item in source_data:
        # 构建新的字典对象
        new_item = {
            "image_id": item.get("id"),
            
            # 路径拼接
            "image_path": f"{image_path_prefix}{item.get('image')}",
            
            # 3. 从列表中随机选择一个 question
            "question": random.choice(candidate_questions)
        }
        target_data.append(new_item)

    # 保存为 JSONL 格式
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in target_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"转换完成！新数据集已保存至: {output_path}")

# --- 配置区域 ---
if __name__ == "__main__":
    # 输入文件名
    # INPUT_FILE = "mm-ref-data-full.json" 
    INPUT_FILE = "mm-ref-data-filtered.json"
    
    # 输出文件名
    OUTPUT_FILE = "image_data.jsonl"
    
    # 图片路径前缀
    IMG_PREFIX = "/home/yilin/new-DPO/dataset/train2014/"

    # 执行转换
    convert_dataset(INPUT_FILE, OUTPUT_FILE, IMG_PREFIX)