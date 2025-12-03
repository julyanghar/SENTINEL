"""
SENTINEL 全局变量管理模块
=========================

本模块定义了项目中使用的所有全局变量，包括：
- 命令行参数
- 日志配置
- 设备信息
- 模型路径

使用方法:
    from model.auxiliary.global_vars import GVars
    
    # 初始化全局变量
    GVars.init()
    
    # 访问变量
    args = GVars.args
    device = GVars.device
    logger = GVars.logger
"""

import logging
import os
import sys
from argparse import Namespace
from logging import Logger

import torch

from utils.setup_utils import get_save_path, parse_arg


class GVars:
    """
    全局变量管理类
    
    所有变量都是类变量，可以在任何地方通过 GVars.xxx 访问。
    在使用前必须调用 GVars.init() 进行初始化。
    
    Attributes:
        args: 命令行参数（Namespace对象）
        save_path: 结果保存路径
        model_dir: 模型缓存目录
        hf_home: HuggingFace 缓存目录
        gpu_count: 可用 GPU 数量
        device: 默认设备
        main_device: 主设备（用于模型推理）
        alter_device: 备用设备（用于检测器等辅助模型）
        openai_key: OpenAI API Key（用于评估）
        logger: 日志器
    """
    
    args: Namespace | None = None
    """命令行参数，包含 model, batch_size, dataset_path 等"""
    
    save_path: str | None = None
    """结果保存路径，格式为 ./results/<model_name>.jsonl"""
    
    model_dir: str | None = None
    """模型缓存目录，从环境变量 MODEL_PATH 读取"""
    
    hf_home: str | None = None
    """HuggingFace 缓存目录，从环境变量 HF_HOME 读取"""
    
    gpu_count: int | None = None
    """可用 GPU 数量"""
    
    device: str | None = None
    """默认设备，等同于 main_device"""
    
    main_device: str | None = None
    """主设备，通常为 'cuda:0'，用于 MLLM 推理"""
    
    alter_device: str | None = None
    """备用设备，通常为 'cuda:1'，用于检测器和 NLP 工具"""
    
    openai_key: str | None = None
    """OpenAI API Key，用于 Object HalBench 评估"""
    
    logger: Logger = logging.getLogger()
    """日志器，用于记录运行信息"""

    @classmethod
    def init(cls, save: bool = True) -> None:
        """
        初始化所有全局变量
        
        初始化顺序很重要，因为某些初始化依赖于之前的初始化结果：
        1. 命令行参数（其他初始化需要用到）
        2. 日志器（后续初始化可以记录日志）
        3. 模型目录
        4. 保存路径
        5. 设备配置
        6. OpenAI Key
        
        Args:
            save: 是否初始化保存路径，默认为 True
        """
        cls.init_args()           # 1. 解析命令行参数
        cls.init_logger()         # 2. 配置日志系统
        cls.init_model_dir()      # 3. 设置模型目录
        if save:
            cls.init_save_file_path()  # 4. 设置保存路径
        cls.init_device()         # 5. 配置 GPU/CPU 设备
        cls.init_openai_key()     # 6. 读取 OpenAI Key
        cls.logger.info("Global variables (Gvars) have been initialized")

    @classmethod
    def init_args(cls, alter: dict | None = None) -> None:
        """
        解析命令行参数
        
        支持的参数:
            --model: 模型名称（如 Qwen2_VL_7B）
            --batch_size: 批处理大小
            --dataset_path: 数据集路径
            --gpu_num: 使用的 GPU 数量
            --log_level: 日志级别
            --num_of_data: 处理的数据数量（-1 表示全部）
        
        Args:
            alter: 可选的参数覆盖字典，用于程序化修改参数
        """
        cls.args = parse_arg()
        if alter is not None:
            for key in alter:
                if alter[key]:
                    setattr(cls.args, key, alter[key])

    @classmethod
    def init_model_dir(cls) -> None:
        """
        从环境变量读取模型目录配置
        
        环境变量:
            HF_HOME: HuggingFace 缓存目录
            MODEL_PATH: 模型文件存储目录
        """
        cls.hf_home = os.getenv("HF_HOME")
        cls.model_dir = os.getenv("MODEL_PATH")

    @classmethod
    def init_openai_key(cls) -> None:
        """
        从环境变量读取 OpenAI API Key
        
        用于 Object HalBench 评估，该评估需要调用 GPT 提取物体
        """
        cls.openai_key = os.getenv("OPENAI_KEY")

    @classmethod
    def init_save_file_path(cls) -> None:
        """
        初始化结果保存路径
        
        路径格式: ./results/<model_name>.jsonl
        """
        cls.save_path = get_save_path(cls.logger, cls.args) + ".jsonl"

    @classmethod
    def init_device(cls) -> None:
        """
        初始化设备配置
        
        设备分配策略:
            - 无 GPU: 使用 CPU
            - 1 个 GPU: main_device 和 alter_device 都使用 cuda:0
            - 2+ 个 GPU: main_device 使用 cuda:0，alter_device 使用 cuda:1
        
        这样可以将 MLLM（显存需求大）和检测器（显存需求小）分开，
        充分利用多 GPU 资源。
        """
        if not torch.cuda.is_available():
            cls.main_device, cls.alter_device, cls.gpu_count = "cpu", "cpu", 0
        elif torch.cuda.device_count() == 1:
            cls.main_device, cls.alter_device, cls.gpu_count = "cuda:0", "cuda:0", 1
        else:
            cls.main_device, cls.alter_device, cls.gpu_count = "cuda:0", "cuda:1", 2
        cls.device = cls.main_device

    @classmethod
    def init_logger(cls) -> None:
        """
        初始化日志系统
        
        日志配置:
            - 同时输出到控制台和文件
            - 日志文件位于 ./log/<model>-<num_of_data>.log
            - 格式: [级别|文件:行号] 时间 >> 消息
            - Transformers 库的日志也会被捕获
        """
        from logging import INFO, WARNING

        from transformers.utils import logging as transformers_logging

        _nameToLevel = {"WARNING": WARNING, "INFO": INFO}

        cls.logger.setLevel(_nameToLevel["INFO"])
        args = cls.args
        log_filename = f"{args.model}-{args.num_of_data}.log"
        log_path = os.path.join(args.log_dir, log_filename)

        # 确保日志目录存在
        os.makedirs(args.log_dir, exist_ok=True)

        # 配置日志格式和处理器
        logging.basicConfig(
            format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stdout),           # 控制台输出
                logging.FileHandler(log_path, mode="a"),     # 文件输出（追加模式）
            ],
        )
        
        # 同步配置 Transformers 库的日志
        transformers_logging.set_verbosity(_nameToLevel[args.log_level])
        transformers_logging.enable_default_handler()
        transformers_logging.add_handler(logging.FileHandler(log_path, mode="a"))
        transformers_logging.enable_explicit_format()


if __name__ == "__main__":
    print("Please run main.py")
    exit(0)
