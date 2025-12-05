"""
SENTINEL 参数配置模块
=====================

本模块负责解析命令行参数和配置项目路径。

使用示例:
    # 命令行使用
    python main.py --model Qwen2_VL_7B --batch_size 5 --dataset_path dataset/my_data.jsonl
    
    # 代码中使用
    from utils.setup_utils import parse_arg
    args = parse_arg()
    print(args.model)  # 输出: Qwen2_VL_7B
"""

import os
from argparse import ArgumentParser, Namespace
from logging import Logger

if __name__ == "__main__":
    print("Please run main.py")
    exit(0)


def parse_arg() -> Namespace:
    """
    解析命令行参数
    
    核心参数:
        --model: 选择用于生成数据的 MLLM 模型
        --batch_size: 批处理大小，影响显存使用和处理速度
        --dataset_path: 输入数据集路径
    
    辅助参数:
        --gpu_num: GPU 数量
        --log_level: 日志级别
        --num_of_data: 处理数据量（-1 表示全部）
        --seed: 随机种子
        --log_dir: 日志目录
    
    自动计算的参数:
        model_size: 从 model 名称中提取的模型大小（如 7B, 13B）
        model_version: 从 model 名称中提取的版本号（如 1.5, 1.6, 2.0）
    
    Returns:
        Namespace: 包含所有参数的命名空间对象
    
    Example:
        >>> args = parse_arg()
        >>> print(args.model)
        'Qwen2_VL_7B'
        >>> print(args.model_size)
        '7B'
        >>> print(args.model_version)
        '2.0'
    """
    parser = ArgumentParser(
        description="SENTINEL: 通过句子级早期干预缓解对象幻觉"
    )
    
    # ==================== 核心参数 ====================
    
    parser.add_argument(
        "--gpu_num", 
        type=int, 
        default=2,
        help="使用的 GPU 数量（1 或 2）"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=5,
        help="批处理大小，建议值：24GB 显存用 3-5，48GB 显存用 8-10"
    )
    
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="dataset/mask_data.jsonl",
        help="输入数据集路径（jsonl 格式）"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="LLaVA_v1_5_7b",
        choices=[
            "LLaVA_v1_5_7b",           # LLaVA v1.5 7B
            "LLaVA_v1_5_13b",          # LLaVA v1.5 13B
            "LLaVA_v1_6_vicuna_7b",    # LLaVA v1.6 (Next) 7B
            "LLaVA_v1_6_vicuna_13b",   # LLaVA v1.6 (Next) 13B
            "Qwen2_VL_2B",             # Qwen2-VL 2B
            "Qwen2_VL_7B",             # Qwen2-VL 7B
            "Qwen2_5_VL_7B",           # Qwen2.5-VL 7B
        ],
        help="用于生成候选句子的 MLLM 模型名称"
    )

    # ==================== 辅助参数 ====================
    
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO", 
        choices=["INFO", "WARNING"],
        help="日志级别"
    )
    
    parser.add_argument(
        "--num_of_data", 
        type=int, 
        default=-1,
        help="处理的数据点数量，-1 表示处理全部数据"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="随机种子，用于结果复现"
    )
    
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="./log/",
        help="日志文件保存目录"
    )
    
    # ==================== 运行模式参数 ====================
    
    parser.add_argument(
        "--mode",
        type=str,
        default="mask",
        choices=["default", "mask"],
        help="运行模式：default（原始偏好对生成）或 mask（基于遮挡的偏好对生成）"
    )
    
    parser.add_argument(
        "--masked_images_dir",
        type=str,
        default="./results/masked_images",
        help="遮挡图像保存目录（仅 mask 模式使用）"
    )

    args: Namespace = parser.parse_args()
    
    # ==================== 自动推断参数 ====================
    
    # 从模型名称中提取模型大小（如 7b, 13b, 2B, 7B）
    args.model_size = args.model.split("-")[-1] if "-" in args.model else args.model.split("_")[-1]
    
    # 从模型名称中提取版本号
    # v1_5 → 1.5, v1_6 → 1.6, 其他 → 2.0
    args.model_version = "1.5" if "v1_5" in args.model else "1.6" if "v1_6" in args.model else "2.0"
    
    return args


def get_save_path(logger: Logger, args: Namespace) -> str:
    """
    获取结果保存路径
    
    路径格式: ./results/<model_name>
    例如: ./results/Qwen2_VL_7B
    
    Args:
        logger: 日志器，用于记录路径信息
        args: 命令行参数
    
    Returns:
        str: 保存路径（不包含扩展名）
    
    Note:
        实际保存时会添加 .jsonl 后缀：
        - ./results/<model_name>.jsonl: 完整分析结果
        - ./results/<model_name>_data_pair.jsonl: 偏好对数据
    """
    save_path: str = os.path.join("./results", args.model)
    if save_path.endswith("/"):
        save_path = save_path[:-1]
    logger.info(f"The save_path is: {save_path}")
    return save_path
