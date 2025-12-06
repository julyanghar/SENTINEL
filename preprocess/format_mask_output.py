"""
格式化 mask_output.json
=======================

将 image 路径（JSON 的 key）和 masked_image 路径从完整路径转换为纯文件名。

使用方法:
    python preprocess/format_mask_output.py

输入:
    {
        "/home/yilin/new-DPO/dataset/train2014/COCO_train2014_000000239580.jpg": [
            {
                "masked_image": "./results/LLaVA_v1_5_7b_masked_images/COCO_xxx_masked_person.jpg"
            }
        ]
    }

输出:
    {
        "COCO_train2014_000000239580.jpg": [
            {
                "masked_image": "COCO_xxx_masked_person.jpg"
            }
        ]
    }
"""

import json
import os
from pathlib import Path


def format_mask_output(
    input_path: str,
    output_path: str = "./mask_output_formatted.json",
) -> None:
    """
    格式化 mask_output.json，将 image 路径和 masked_image 路径转换为纯文件名
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径，默认为当前目录下的 mask_output_formatted.json
    """
    
    # 读取原始数据
    print(f"Reading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 转换 key（image 路径 -> 纯文件名）和 masked_image 路径
    new_data = {}
    for image_path, pairs in data.items():
        # 提取纯文件名（原图像）
        filename = os.path.basename(image_path)
        
        # 同时格式化 masked_image 路径
        for pair in pairs:
            if "masked_image" in pair and pair["masked_image"]:
                pair["masked_image"] = os.path.basename(pair["masked_image"])
        
        new_data[filename] = pairs
    
    # 保存
    print(f"Writing: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    
    print(f"Done! Converted {len(new_data)} entries.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="格式化 mask_output.json")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="/home/yilin/SENTINEL/results/mask_output.json",
        help="输入文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./mask_output_formatted.json",
        help="输出文件路径（默认为当前目录下的 mask_output_formatted.json）"
    )
    
    args = parser.parse_args()
    
    format_mask_output(
        input_path=args.input,
        output_path=args.output,
    )

