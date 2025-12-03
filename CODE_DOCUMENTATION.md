# SENTINEL 项目代码详细解读与注释

## 📋 目录
1. [项目概述](#1-项目概述)
2. [整体架构](#2-整体架构)
3. [核心模块详解](#3-核心模块详解)
   - [3.1 主入口模块](#31-主入口模块-mainpy)
   - [3.2 全局变量管理](#32-全局变量管理-modelauxiliaryglobal_varspy)
   - [3.3 数据集模块](#33-数据集模块-modelauxiliarydatasetpy)
   - [3.4 数据状态管理](#34-数据状态管理-modelauxiliarydatastatepy)
   - [3.5 运行入口](#35-运行入口-runrunpy)
   - [3.6 数据集生成核心](#36-数据集生成核心-rungenerate_datasetpy)
   - [3.7 工具函数](#37-工具函数-runutilspy)
4. [模型模块详解](#4-模型模块详解)
   - [4.1 视觉语言模型生成器](#41-视觉语言模型生成器)
   - [4.2 目标检测器](#42-目标检测器)
   - [4.3 NLP工具](#43-nlp工具)
5. [数据流程图](#5-数据流程图)
6. [关键算法解析](#6-关键算法解析)

---

## 1. 项目概述

**SENTINEL** (Sentence-Level Early Intervention) 是一个用于缓解多模态大型语言模型(MLLMs)中对象幻觉问题的项目。该项目通过以下核心思想工作：

### 核心思想
1. **早期干预阻止幻觉传播**：幻觉通常在早期句子中产生并传播到后续输出
2. **无需人工标注的偏好学习**：通过检测器交叉验证构建幻觉/真实样本
3. **上下文感知的偏好数据构建**：构建context-aware的DPO训练数据

### 主要功能
- 自动生成用于DPO训练的偏好数据对
- 支持多种视觉语言模型 (LLaVA, Qwen2-VL, Qwen2.5-VL)
- 使用YOLO和Grounding DINO进行对象检测验证

---

## 2. 整体架构

```
SENTINEL/
├── main.py                 # 主入口文件
├── run/                    # 运行逻辑模块
│   ├── run.py              # 运行入口
│   ├── generate_dataset.py # 数据集生成核心逻辑
│   ├── utils.py            # 运行工具函数
│   └── object_utils.py     # 对象处理工具
├── model/                  # 模型模块
│   ├── auxiliary/          # 辅助类
│   │   ├── global_vars.py  # 全局变量管理
│   │   ├── dataset.py      # 数据集类
│   │   └── datastate.py    # 数据状态类
│   ├── generator/          # 生成器模型
│   │   ├── llava.py        # LLaVA模型封装
│   │   ├── qwen2_vl.py     # Qwen2-VL模型封装
│   │   └── qwen2_5_vl.py   # Qwen2.5-VL模型封装
│   ├── detector/           # 检测器模型
│   │   ├── grounding_dino.py # Grounding DINO检测器
│   │   └── yolo_model.py   # YOLO检测器
│   ├── others/             # 其他NLP工具
│   │   ├── spacy_model.py  # Spacy NLP模型
│   │   ├── wordnet.py      # WordNet词汇工具
│   │   └── sg_parser.py    # 场景图解析器
│   └── utils/              # 模型工具
│       ├── gen_utils.py    # 生成工具函数
│       └── utils.py        # 通用工具函数
├── utils/                  # 通用工具
│   ├── setup_utils.py      # 配置和参数解析
│   ├── get_llama_factory_data_pair.py  # 数据格式转换
│   └── .env                # 环境变量配置
└── train/                  # 训练相关代码
```

---

## 3. 核心模块详解

### 3.1 主入口模块 (main.py)

```python
"""
main.py - 程序主入口

这是整个SENTINEL项目的启动文件，负责：
1. 加载环境变量配置
2. 初始化全局变量
3. 启动数据生成流程
"""

from dotenv import load_dotenv

# 加载 .env 文件中的环境变量配置
# 包括模型路径、HuggingFace目录等关键配置
print("Load dot env result:", load_dotenv("./utils/.env"))


def main():
    """
    主函数入口
    
    执行流程：
    1. 导入并初始化全局变量类 GVars
    2. 调用 run() 函数启动数据生成流程
    """
    # 导入全局变量管理类
    from model.auxiliary.global_vars import GVars

    # 初始化全局变量（包括参数解析、日志配置、设备检测等）
    GVars.init()

    # 导入并执行主运行逻辑
    from run.run import run
    run()


if __name__ == "__main__":
    main()
```

---

### 3.2 全局变量管理 (model/auxiliary/global_vars.py)

```python
"""
global_vars.py - 全局变量管理类

这个模块定义了一个单例模式的全局变量管理类 GVars，用于：
1. 管理命令行参数
2. 配置日志系统
3. 管理模型目录和设备信息
4. 提供全局共享的配置信息
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
    全局变量管理类 (单例模式)
    
    类属性（所有实例共享）：
    - args: 命令行参数对象
    - save_path: 结果保存路径
    - model_dir: 模型缓存目录
    - hf_home: HuggingFace 主目录
    - gpu_count: GPU数量
    - device: 默认设备
    - main_device: 主GPU设备
    - alter_device: 备用GPU设备（用于辅助模型）
    - openai_key: OpenAI API密钥（可选）
    - logger: 日志记录器
    """
    args: Namespace | None = None          # 命令行参数
    save_path: str | None = None           # 结果保存路径
    model_dir: str | None = None           # 模型缓存目录
    hf_home: str | None = None             # HuggingFace Home目录
    gpu_count: int | None = None           # 可用GPU数量
    device: str | None = None              # 默认计算设备
    main_device: str | None = None         # 主GPU（运行生成器）
    alter_device: str | None = None        # 备用GPU（运行检测器等）
    openai_key: str | None = None          # OpenAI API密钥
    logger: Logger = logging.getLogger()   # 日志记录器

    @classmethod
    def init(cls, save: bool = True) -> None:
        """
        初始化所有全局变量
        
        参数:
            save: 是否初始化保存路径，默认为True
        
        初始化顺序很重要：
        1. 首先解析命令行参数
        2. 配置日志系统
        3. 设置模型目录
        4. 确定保存路径
        5. 检测并配置设备
        6. 加载可选的API密钥
        """
        cls.init_args()
        cls.init_logger()
        cls.init_model_dir()
        if save:
            cls.init_save_file_path()
        cls.init_device()
        cls.init_openai_key()
        cls.logger.info("Global variables (Gvars) have been initialized")

    @classmethod
    def init_args(cls, alter: dict | None = None) -> None:
        """
        解析命令行参数并存储
        
        参数:
            alter: 可选的参数覆盖字典，用于测试或特殊场景
        """
        cls.args = parse_arg()
        if alter is not None:
            for key in alter:
                if alter[key]:
                    setattr(cls.args, key, alter[key])

    @classmethod
    def init_model_dir(cls) -> None:
        """
        从环境变量中获取模型目录配置
        
        环境变量:
            HF_HOME: HuggingFace缓存主目录
            MODEL_PATH: 自定义模型存储路径
        """
        cls.hf_home = os.getenv("HF_HOME")
        cls.model_dir = os.getenv("MODEL_PATH")

    @classmethod
    def init_device(cls) -> None:
        """
        自动检测并配置计算设备
        
        设备分配策略：
        - 无GPU: 使用CPU
        - 单GPU: 主设备和备用设备都使用同一GPU
        - 多GPU: 主模型使用cuda:0，辅助模型使用cuda:1
        
        这种分配可以优化内存使用，避免生成器和检测器抢占显存
        """
        if not torch.cuda.is_available():
            cls.main_device, cls.alter_device, cls.gpu_count = "cpu", "cpu", 0
        elif torch.cuda.device_count() == 1:
            cls.main_device, cls.alter_device, cls.gpu_count = "cuda:0", "cuda:0", 1
        else:
            # 多GPU时分离主模型和辅助模型
            cls.main_device, cls.alter_device, cls.gpu_count = "cuda:0", "cuda:1", 2
        cls.device = cls.main_device

    @classmethod
    def init_logger(cls) -> None:
        """
        配置日志系统
        
        特点：
        - 同时输出到控制台和文件
        - 日志文件名包含模型名称和数据量信息
        - 配置Transformers库的日志级别
        """
        from logging import INFO, WARNING
        from transformers.utils import logging as transformers_logging

        _nameToLevel = {"WARNING": WARNING, "INFO": INFO}

        cls.logger.setLevel(_nameToLevel["INFO"])
        args = cls.args
        
        # 日志文件命名: 模型名-数据量.log
        log_filename = f"{args.model}-{args.num_of_data}.log"
        log_path = os.path.join(args.log_dir, log_filename)

        os.makedirs(args.log_dir, exist_ok=True)

        # 配置日志格式和处理器
        logging.basicConfig(
            format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stdout),           # 控制台输出
                logging.FileHandler(log_path, mode="a"),     # 文件输出
            ],
        )
        
        # 配置Transformers库日志
        transformers_logging.set_verbosity(_nameToLevel[args.log_level])
        transformers_logging.enable_default_handler()
        transformers_logging.add_handler(logging.FileHandler(log_path, mode="a"))
        transformers_logging.enable_explicit_format()
```

---

### 3.3 数据集模块 (model/auxiliary/dataset.py)

```python
"""
dataset.py - 数据集管理模块

定义了数据点(DataPoint)和数据集(DataSet)类，用于：
1. 加载和解析原始数据
2. 管理数据的生命周期
3. 支持断点续传（过滤已处理数据）
"""

import os
import random
from argparse import Namespace
from dataclasses import dataclass, field
from logging import Logger

from ..utils.utils import read_json


@dataclass
class DataPoint:
    """
    单个数据点类
    
    表示一个待处理的图像-问题对
    
    属性:
        image_id: 图像唯一标识符
        image_path: 图像文件路径
        question: 关于图像的问题/提示
        attributes: 额外属性字典（可扩展）
    """
    image_id: str           # 图像ID
    image_path: str         # 图像路径
    question: str           # 问题/提示语
    attributes: dict[str] = field(default_factory=dict)  # 扩展属性

    def __getitem__(self, key: str) -> str:
        """支持字典式访问"""
        if key == "image_id":
            return self.image_id
        elif key == "image_path":
            return self.image_path
        elif key == "question":
            return self.question
        elif key in self.attributes:
            return self.attributes[key]
        else:
            raise KeyError(f"Key {key} not found in DataPoint")

    def __repr__(self) -> str:
        return f"DataPoint(image_id={self.image_id}, image_path={self.image_path}, question={self.question})"


@dataclass
class DataSet:
    """
    数据集管理类
    
    负责加载、管理和过滤数据集
    
    属性:
        args: 命令行参数
        logger: 日志记录器
        data: 数据点列表
    
    主要功能:
        1. 从文件加载数据
        2. 支持数据采样
        3. 支持断点续传（过滤已处理数据）
    """
    args: Namespace
    logger: Logger | None = None
    data: list[DataPoint] = field(init=False)

    def __post_init__(self):
        """初始化后自动加载数据集"""
        if self.logger is not None and self.args is not None:
            self.logger.info(f"Loading dataset from {self.args.dataset_path}")
        self.data = self._load_dataset(self.args)

    def _load_dataset(self, args: Namespace) -> list[DataPoint]:
        """
        加载数据集
        
        参数:
            args: 包含dataset_path和num_of_data的参数对象
        
        返回:
            DataPoint列表
        
        处理逻辑:
            1. 从JSON/JSONL文件加载原始数据
            2. 如果指定了数据量限制，随机采样
            3. 转换为DataPoint对象
        """
        dataset_path: str = args.dataset_path
        assert os.path.exists(dataset_path), f"Dataset file not found at {dataset_path}"

        dataset: list[dict] = read_json(dataset_path)

        # 数据采样：如果num_of_data有效且小于总量，随机采样
        num_of_data: int = args.num_of_data
        if 0 <= num_of_data < len(dataset):
            random.seed(args.seed)  # 保证可复现
            dataset = random.sample(dataset, num_of_data)

        return [self._create_datapoint(item) for item in dataset]

    @staticmethod
    def _create_datapoint(item: dict) -> DataPoint:
        """
        从字典创建DataPoint对象
        
        支持多种数据格式，自动适配字段名
        """
        image_id: str = item["image_id"] if "image_id" in item else item["image"]
        image_path: str = item["image_path"] if "image_path" in item else item["image"]
        question: str = item["question"] if "question" in item else "Describe this image."
        
        # 其他字段作为扩展属性
        attributes: dict[str] = {
            k: v for k, v in item.items() if k not in {"image_id", "image", "image_path", "question"}
        }
        return DataPoint(image_id, image_path, question, attributes=attributes)

    def filter(self, save_path: str) -> None:
        """
        过滤已处理的数据（断点续传功能）
        
        参数:
            save_path: 已保存结果的文件路径
        
        功能:
            读取已保存的结果，从数据集中移除已处理的数据点，
            使得程序可以从中断处继续处理
        """
        if not os.path.exists(save_path) or not os.path.isfile(save_path) or not self.data:
            return

        exist_data: list[dict] = read_json(save_path)
        done_image_id: list = [d["image_id"] for d in exist_data]
        # 保留未处理的数据
        self.data = [d for d in self.data if d.image_id not in done_image_id]
```

---

### 3.4 数据状态管理 (model/auxiliary/datastate.py)

```python
"""
datastate.py - 数据处理状态管理模块

定义了跟踪数据处理过程的状态类，用于：
1. 维护当前处理的上下文
2. 记录生成的句子和检测到的对象
3. 管理幻觉/非幻觉对象的分类
"""

from dataclasses import dataclass, field
from PIL import Image
from model.detector.yolo_model import YoloResult
from .dataset import DataPoint


@dataclass
class DataState:
    """
    基础数据状态类
    
    跟踪单个数据点的处理状态
    
    属性:
        data: 原始数据点
        image_path: 图像路径
        image: 加载的图像对象
        question: 问题/提示
        is_finished: 是否完成处理
        assistant: 当前生成的完整描述
        nonhallu_objects: 已确认真实存在的对象列表
        uncertain_objects: 不确定的对象列表
        hallu_objects: 已确认为幻觉的对象列表
    """
    data: DataPoint
    image_path: str = field(init=False)
    image: Image.Image = field(init=False)
    question: str = field(init=False)
    is_finished: bool = False           # 处理完成标志
    assistant: str = ""                  # 当前生成的回复
    
    # 三类对象缓存
    nonhallu_objects: list[str] = field(default_factory=list)   # 非幻觉对象
    uncertain_objects: list[str] = field(default_factory=list)  # 不确定对象
    hallu_objects: list[str] = field(default_factory=list)      # 幻觉对象

    def __post_init__(self):
        """初始化时加载图像"""
        from run.utils import open_images
        self.image_path = self.data.image_path
        self.image = open_images(self.image_path)  # 打开并转换为RGB
        self.question = self.data.question


@dataclass
class DataStateForBuildDataset(DataState):
    """
    用于构建训练数据集的数据状态类
    
    继承自DataState，添加了更多用于偏好对构建的状态信息
    
    核心设计:
        这个类维护了整个迭代式生成过程的完整状态，包括：
        1. 每一步生成的所有候选句子
        2. 每个句子中提取的对象
        3. 对象的幻觉/非幻觉分类
        4. 最终选择的句子形成的上下文
    
    重要属性:
        yolo_detected: YOLO是否已检测过此图像
        yolo_result: YOLO检测结果
        detector_reject: 记录被各检测器拒绝的对象
        generated_sentences: 每步生成的候选句子列表
        generated_objects: 每步每个候选句子中的对象
        generated_hallu_objects: 每步每个候选句子中的幻觉对象
        generated_nonhallu_objects: 每步每个候选句子中的非幻觉对象
    """
    # YOLO检测相关
    yolo_detected: bool = False
    yolo_result: YoloResult = None

    # 检测器拒绝记录：记录哪些对象被哪个检测器判定为不存在
    detector_reject: dict[str, list[str]] = field(init=False)
    
    # 生成过程记录
    # 结构: generated_sentences[step_idx] = [sent_1, sent_2, ..., sent_n]
    generated_sentences: list[list[str]] = field(default_factory=list)
    
    # 记录最终选定的assistant句子
    generated_assistents: list[str] = field(default_factory=list)

    # 对象分析记录
    # 结构: generated_objects[step_idx][candidate_idx] = [obj_1, obj_2, ...]
    generated_objects: list[list[list[str]]] = field(default_factory=list)
    generated_hallu_objects: list[list[list[str]]] = field(default_factory=list)
    generated_nonhallu_objects: list[list[list[str]]] = field(default_factory=list)

    # 当前assistant中累积的对象
    assistant_objects: list[str] = field(default_factory=list)
    assistant_hallu_objects: list[str] = field(default_factory=list)
    assistant_nonhallu_objects: list[str] = field(default_factory=list)

    # 用于POPE数据集构建
    ground_truth: bool = field(init=False)

    # 调试信息
    gt_objects: list[str] = field(default_factory=list)

    # 难样本相关
    hard_positive: list[str] = field(default_factory=list)   # 未被模型提及但确实存在的对象
    small_objects: list[str] = field(default_factory=list)   # 小对象
    edge_objects: list[str] = field(default_factory=list)    # 边缘对象

    # 自然上下文缓存（贪婪搜索生成）
    nature_context: str = None
    nature_objects: list[list[str]] = field(default_factory=list)
    nature_hallu_objects: list[list[str]] = field(default_factory=list)
    nature_nonhallu_objects: list[list[str]] = field(default_factory=list)

    def __post_init__(self):
        """初始化特有状态"""
        super().__post_init__()
        self.generated_assistents.append("")  # 第一步需要空字符串
        self.detector_reject = {"dino": [], "yolo": []}  # 初始化检测器拒绝记录

        if "ground_truth" in self.data.attributes:
            self.ground_truth = self.data.attributes["ground_truth"]

    def app_assistant(self, new_sents: list[str], idx: int) -> None:
        """
        追加选中的句子到assistant
        
        参数:
            new_sents: 当前步骤的所有候选句子
            idx: 选中的句子索引
        
        功能:
            1. 将选中句子追加到assistant
            2. 记录assistant历史
            3. 更新assistant中的对象列表
        """
        # 拼接新句子到现有assistant
        self.assistant = self.assistant + " " + new_sents[idx] if self.assistant else new_sents[idx]
        self.generated_assistents.append(self.assistant)

        # 更新assistant中的对象统计
        if len(self.generated_objects) == self.gen_sents_cnt:
            self.assistant_objects.extend(self.generated_objects[-1][idx])
        if len(self.generated_hallu_objects) == self.gen_sents_cnt:
            self.assistant_hallu_objects.extend(self.generated_hallu_objects[-1][idx])
        if len(self.generated_nonhallu_objects) == self.gen_sents_cnt:
            self.assistant_nonhallu_objects.extend(self.generated_nonhallu_objects[-1][idx])

    @property
    def gen_sents_cnt(self) -> int:
        """返回已生成的句子步数"""
        return len(self.generated_sentences)

    @property
    def now_step_idx(self) -> int:
        """返回当前步骤索引（0-based）"""
        return self.gen_sents_cnt - 1

    @property
    def context_gen_objects(self) -> set[str]:
        """返回上下文中已生成的所有对象（去重）"""
        return set(self.assistant_objects)

    @property
    def context_gen_hallu_objects(self) -> set[str]:
        """返回上下文中已生成的幻觉对象（去重）"""
        return set(self.assistant_hallu_objects)

    @property
    def flat_gen_objs(self) -> list[str]:
        """返回所有生成过程中提及的对象（未去重）"""
        return [obj for objs in self.generated_objects for obj_list in objs for obj in obj_list]

    def gen_objs(self, index: int) -> list[list[str]]:
        """获取指定步骤的所有候选句子的对象"""
        return self.generated_objects[index]

    def hallu_objs(self, index: int) -> list[list[str]]:
        """获取指定步骤的所有候选句子的幻觉对象"""
        return self.generated_hallu_objects[index]

    def nonhallu_objs(self, index: int) -> list[list[str]]:
        """获取指定步骤的所有候选句子的非幻觉对象"""
        return self.generated_nonhallu_objects[index]
```

---

### 3.5 运行入口 (run/run.py)

```python
"""
run.py - 运行入口模块

负责初始化数据集并启动生成流程
"""

def run() -> None:
    """
    主运行函数
    
    执行流程:
        1. 导入必要的模块和全局变量
        2. 加载数据集
        3. 过滤已处理的数据（支持断点续传）
        4. 调用数据集生成函数
    """
    from model.auxiliary.dataset import DataSet
    from model.auxiliary.global_vars import GVars
    from run.generate_dataset import run_gen_dataset

    args, save_path, logger = GVars.args, GVars.save_path, GVars.logger
    batch_size = args.batch_size

    logger.info(f"Current batch size: {batch_size}")
    logger.info(f"Start loading dataset with dataset path: {args.dataset_path}")
    
    # 创建数据集并过滤已处理的数据
    dataset: DataSet = DataSet(args=args, logger=logger)
    dataset.filter(save_path)  # 断点续传：移除已处理的数据
    
    logger.info(f"Finish loading dataset, dataset size: {len(dataset.data)}")

    if not dataset.data:
        logger.info("All data has been processed, exit run function")
        return
        
    # 启动数据集生成
    run_gen_dataset(dataset.data, batch_size)
```

---

### 3.6 数据集生成核心 (run/generate_dataset.py)

```python
"""
generate_dataset.py - 数据集生成核心模块

这是SENTINEL项目的核心逻辑，负责：
1. 迭代式地生成图像描述
2. 识别幻觉对象和真实对象
3. 构建偏好数据对用于DPO训练

核心算法:
    对于每张图像，循环执行以下步骤直到生成结束:
    1. 使用VLM生成多个候选句子（采样n=10）
    2. 从句子中提取对象
    3. 使用YOLO和DINO验证对象是否存在
    4. 构建偏好对：非幻觉句子 vs 幻觉句子
    5. 选择最佳句子作为下一轮上下文
"""

import random
from time import time

from model.auxiliary.dataset import DataPoint
from model.auxiliary.datastate import DataStateForBuildDataset
from model.auxiliary.global_vars import GVars
from model.detector.grounding_dino import DINO
from model.detector.yolo_model import YoloModel
from model.others.sg_parser import SGParser
from model.others.spacy_model import SpacyModel
from model.others.wordnet import WordnetModel
from model.utils.gen_utils import GenOutput, get_generator
from run.utils import (
    b_get_hallu_objects,
    extract_obj_from_textgraphs,
    extract_obj_w_gt,
    get_finish_flag,
    log_progress,
    object_in_set,
    objects_in_set,
    refModel,
    resolve_corefs,
    save_result,
    yolo_detect,
)

DEBUG = True           # 调试模式
HALLUCI_CONTEXT = False  # 是否使用含幻觉的句子添加到context
CHECK_TYPE = "any"     # 对象检查类型


def save_data_state(
    res_save_path: str,
    s: DataStateForBuildDataset,
    spacy: SpacyModel | None = None,
    wn: WordnetModel | None = None,
    inv_synonym_map: dict[str, list[str]] | None = None,
) -> None:
    """
    保存数据状态到文件
    
    参数:
        res_save_path: 保存路径
        s: 数据状态对象
        spacy: Spacy模型（用于同义词检查）
        wn: WordNet模型
        inv_synonym_map: 逆同义词映射
    
    保存内容:
        - 基本信息：image_id, image_path, question, caption
        - 统计信息：句子数量、幻觉对象、非幻觉对象
        - 分析信息：困难正例、小对象、边缘对象
    """
    # 困难正例：图像中存在但模型未提及的对象
    s.hard_positive = [
        obj for obj in s.yolo_result.labels 
        if not object_in_set(obj, set(s.flat_gen_objs), spacy, wn, inv_synonym_map)
    ]
    
    # 小对象：面积小于2%的非幻觉对象
    s.small_objects = [
        obj
        for obj in s.yolo_result.labels
        if object_in_set(obj, set(s.flat_nonhallu_objs), spacy, wn, inv_synonym_map)
        and s.yolo_result.get_largest(obj)
        and (s.yolo_result.get_largest(obj)["xywhn"][2] * s.yolo_result.get_largest(obj)["xywhn"][3] < 0.02)
    ]
    
    # 边缘对象：距离图像边缘小于10%的非幻觉对象
    s.edge_objects = [
        obj
        for obj in s.yolo_result.labels
        if object_in_set(obj, set(s.flat_nonhallu_objs), spacy, wn, inv_synonym_map)
        if s.yolo_result.get_farthest_to_edge(obj)
        and (
            min(
                s.yolo_result.get_farthest_to_edge(obj)["xywhn"][0],
                1 - s.yolo_result.get_farthest_to_edge(obj)["xywhn"][0],
                s.yolo_result.get_farthest_to_edge(obj)["xywhn"][1],
                1 - s.yolo_result.get_farthest_to_edge(obj)["xywhn"][1],
            )
            < 0.1
        )
    ]

    save_result(
        res_save_path,
        {
            "image_id": s.data.image_id,
            "image_path": s.data.image_path,
            "question": s.question,
            "caption": s.assistant,
            "sentences_cnt": s.gen_sents_cnt,
            "hallu_objects": s.hallu_objects,
            "uncertain_objects": s.uncertain_objects,
            "nonhallu_objects": s.nonhallu_objects,
            "hard_positive": s.hard_positive,
            "small_objects": s.small_objects,
            "edge_objects": s.edge_objects,
        },
    )


def maybe_build_pair(
    save_path: str,
    s: DataStateForBuildDataset,
    spacy: SpacyModel,
    wn: WordnetModel,
    inv_synonym_map: dict[str, list[str]] | None = None,
) -> int:
    """
    构建偏好数据对并返回最佳句子索引
    
    参数:
        save_path: 保存路径
        s: 数据状态
        spacy: Spacy模型
        wn: WordNet模型
        inv_synonym_map: 逆同义词映射
    
    返回:
        最佳句子的索引（用于更新上下文）
    
    核心逻辑:
        1. 分类候选句子：非幻觉句子 vs 幻觉句子
        2. 进一步分类非幻觉句子：探索新对象的 vs 重复旧对象的
        3. 配对构建偏好对：y+ (非幻觉) vs y- (幻觉)
        4. 选择策略：优先选择探索新对象的非幻觉句子
    """
    
    def create_pairs(win_candidates: list[tuple[int, list[str]]], lose_candidates, pair_type: str) -> list[dict]:
        """构建偏好数据对"""
        return [
            {
                "image_id": s.data.image_id,
                "image_path": s.data.image_path,
                "question": s.data.question,
                "context": s.assistant,           # 当前上下文
                "y_win": new_sentences[win_idx],  # 胜出句子（非幻觉）
                "y_lose": new_sentences[lose_idx], # 失败句子（幻觉）
                # 附加分析信息
                "nonhallu_objects": s.nonhallu_objects,
                "context_gen_objects": s.context_gen_objects,
                "context_gen_hallu_objects": s.context_gen_hallu_objects,
                "objects_of_y_win": objects,
                "hallu_objects_of_y_lose": hallu_objects,
                "is_last_sent": s.is_finished,
                "type": pair_type,
            }
            for (win_idx, objects), (lose_idx, hallu_objects) in zip(win_candidates, lose_candidates)
        ]

    new_sentences: list[str] = s.generated_sentences[-1]
    if len(new_sentences) <= 1:
        return 0

    step_idx = s.now_step_idx
    # 获取当前步骤所有候选句子的对象信息
    objects_list, nonhallu_objects_list, hallu_objects_list = (
        s.gen_objs(step_idx),
        s.nonhallu_objs(step_idx),
        s.hallu_objs(step_idx),
    )

    # 筛选非幻觉候选：至少有一个对象且无幻觉对象且不在不确定列表中
    nonhallu_candidates: list = [
        (i, objects)
        for i, (objects, hallu_objects) in enumerate(zip(objects_list, hallu_objects_list))
        if len(objects) >= 1
        and not hallu_objects
        and not objects_in_set(objects, s.uncertain_objects, spacy, wn, inv_synonym_map, check_type="any")
    ]
    
    # 筛选幻觉候选：包含至少一个幻觉对象
    hallu_candidates: list = [
        (i, hallu_objects) for i, hallu_objects in enumerate(hallu_objects_list) if len(hallu_objects) >= 1
    ]

    # 将非幻觉候选进一步分为：成功探索新对象的 vs 重复旧对象的
    success_explore_candidates, normal_nonhallu_candidates = [], []
    for idx, objects in nonhallu_candidates:
        if not objects_in_set(objects, s.context_gen_objects, spacy, wn, inv_synonym_map, check_type=CHECK_TYPE):
            # 探索了上下文中没有的新对象
            success_explore_candidates.append((idx, objects))
        else:
            # 只提及上下文中已有的对象
            normal_nonhallu_candidates.append((idx, objects))

    # 构建偏好对：数量取两者最小值
    num_pairs = min(len(normal_nonhallu_candidates), len(hallu_candidates))
    all_results_list = create_pairs(normal_nonhallu_candidates[:num_pairs], hallu_candidates[:num_pairs], "y+")

    # 保存偏好对
    save_result(save_path.replace(".jsonl", "_data_pair.jsonl"), all_results_list)

    # 选择最佳句子用于下一轮生成
    if HALLUCI_CONTEXT:
        # 测试用：故意选择幻觉句子
        if hallu_candidates:
            return random.choice([idx for idx, _ in hallu_candidates])
        else:
            return random.choice(range(len(new_sentences)))
    else:
        # 正常策略：优先探索新对象，其次选择非幻觉句子
        if success_explore_candidates:
            return random.choice([idx for idx, _ in success_explore_candidates])
        elif normal_nonhallu_candidates:
            return random.choice([i for i, _ in normal_nonhallu_candidates])
        else:
            return random.choice(range(len(new_sentences)))


def run_gen_dataset(datalist: list[DataPoint], batch_size: int) -> None:
    """
    主数据集生成函数
    
    参数:
        datalist: 待处理的数据点列表
        batch_size: 批处理大小
    
    主循环逻辑:
        while 还有未处理的数据:
            1. 装载batch_size个数据到状态列表
            2. 对新数据进行YOLO检测
            3. 使用VLM生成候选句子（采样n=10）
            4. 检查生成是否结束
            5. 执行指代消解
            6. 提取对象（使用GT词表 + 场景图解析）
            7. 判断幻觉对象（YOLO + DINO交叉验证）
            8. 构建偏好对并选择最佳句子
            9. 保存完成的数据，移除已完成状态
    """
    logger, save_path, model_dir, alter_device = GVars.logger, GVars.save_path, GVars.model_dir, GVars.alter_device
    
    # 初始化生成器（VLM模型）
    generator = get_generator(use_vllm=True, debug=DEBUG)

    # 初始化目标检测器
    DINO_detector = DINO("base", model_dir=model_dir, device=alter_device, logger=logger)
    yolo = YoloModel("yolo11x", model_dir=model_dir, logger=logger)

    # 初始化NLP工具
    SG_parser = SGParser(DEBUG, "base", model_dir, device=alter_device, logger=logger)  # 场景图解析
    spacy = SpacyModel(model_size="md", model_dir=model_dir, device=alter_device, logger=logger)  # 词性分析
    wn = WordnetModel(logger=logger)  # WordNet同义词
    ref = refModel(args=GVars.args)  # 参考模型（包含有效名词列表）

    data_states: list[DataStateForBuildDataset] = []  # 当前正在处理的数据状态
    num_of_data, finished_data_num = len(datalist), 0

    logger.info(f"Start processing {num_of_data} data points.")

    # 主循环
    while len(datalist) > 0 or len(data_states) > 0:
        start_time = time()

        # 步骤1: 装载数据到状态列表，保持batch_size个活跃状态
        while len(data_states) < batch_size and len(datalist) > 0:
            tmp_data = datalist.pop(0)
            data_states.append(DataStateForBuildDataset(data=tmp_data))

        # 步骤2: 对新图像进行YOLO检测（只检测一次）
        yolo_detect(yolo, data_states)

        # 步骤3: 使用VLM生成候选句子
        # 采样参数：n=10（每个样本生成10个候选），temp=0.7，单句子生成
        out: GenOutput = generator.gen(
            images=[s.image for s in data_states],
            users=[s.question for s in data_states],
            assistants=[s.assistant for s in data_states],  # 使用累积的上下文
            do_sample=True,
            n=10,           # 每个样本生成10个候选句子
            temp=0.7,       # 采样温度
            force_list=True,
            single_sentence=True,  # 只生成一个句子
        )
        b_new_sents: list[list[str]] = out.outputs

        # 步骤4: 检查是否生成结束
        for idx, (new_sents, s) in enumerate(zip(b_new_sents, data_states)):
            b_new_sents[idx], s.is_finished = get_finish_flag(new_sents, remove_duplicates=True)

        # 步骤5: 执行指代消解
        # 将代词替换为其指代的名词，便于后续对象提取
        context = [s.assistant for s in data_states]
        b_resolved_new_sents: list[list[str]] = resolve_corefs(spacy, b_new_sents, context, 1)

        # 步骤6: 提取对象
        b_object_lists: list[list[list[str]]] = []
        for s, new_sents in zip(data_states, b_resolved_new_sents):
            # 方法1: 基于GT词表的对象提取
            object_lists: list[list[str]] = extract_obj_w_gt(
                new_sents,
                ref.valid_nouns,
                ref.double_words,
                ref.inv_syn_map,
                wn,
                force_list=True,
                return_repr=False,
            )

            # 方法2: 基于场景图解析的对象提取
            textgraphs: list[list[list[str]]] = SG_parser.pharse(new_sents, force_list=True)
            new_object_lists: list[list[str]] = extract_obj_from_textgraphs(textgraphs, spacy, wn, force_list=True)
            
            # 合并两种方法的结果
            object_lists = [objects + new_objects for objects, new_objects in zip(object_lists, new_object_lists)]
            b_object_lists.append(object_lists)

        # 步骤7: 判断幻觉对象（核心：YOLO + DINO交叉验证）
        b_haluci_objects_list, b_nonhallu_objects_list = b_get_hallu_objects(
            b_object_lists,
            [s.nonhallu_objects for s in data_states],    # 已知非幻觉对象
            [s.hallu_objects for s in data_states],       # 已知幻觉对象
            spacy=spacy,
            wn=wn,
            images=[s.image for s in data_states],
            dino=DINO_detector,
            b_yolo_results=[s.yolo_result.labels for s in data_states] if yolo else None,
            yolo_labels=yolo.labels if yolo else None,
            b_uncertain_objects=[s.uncertain_objects for s in data_states],
            b_detector_rejects=[s.detector_reject for s in data_states],
            inv_syn_map=ref.inv_syn_map,
        )

        # 步骤8: 更新状态并构建偏好对
        for s, new_sents, object_lists, haluci_objects_list, nonhallu_objects_list in zip(
            data_states, b_resolved_new_sents, b_object_lists, b_haluci_objects_list, b_nonhallu_objects_list
        ):
            if not new_sents:
                continue
            # 记录生成结果
            s.generated_sentences.append(new_sents)
            s.generated_objects.append(object_lists)
            s.generated_hallu_objects.append(haluci_objects_list)
            s.generated_nonhallu_objects.append(nonhallu_objects_list)

            # 构建偏好对并选择最佳句子
            best_idx: int = maybe_build_pair(save_path, s, spacy, wn, ref.inv_syn_map)
            s.app_assistant(new_sents, best_idx)  # 将最佳句子追加到上下文

        # 步骤9: 保存并清理已完成的状态
        [save_data_state(save_path, s, spacy, wn, ref.inv_syn_map) for s in data_states if s.is_finished]

        finished_data_num += len([s for s in data_states if s.is_finished])
        log_progress(logger, finished_data_num, num_of_data, batch_size, time() - start_time)
        data_states = [s for s in data_states if not s.is_finished]  # 只保留未完成的状态
```

---

### 3.7 工具函数 (run/utils.py) - 部分关键函数

```python
"""
utils.py - 运行工具函数模块

包含数据处理过程中使用的各种辅助函数
"""

@dataclass
class refModel:
    """
    参考模型类
    
    存储用于对象识别的参考数据，包括：
    - valid_nouns: 有效名词列表（MSCOCO对象）
    - inv_syn_map: 同义词到代表词的映射
    - double_words: 双词短语的映射
    """
    args: Namespace
    valid_nouns: list[str] = field(init=False)
    inv_syn_map: dict[str, str] = field(init=False)
    double_words: dict[str, str] = field(init=False)

    def __post_init__(self):
        self.valid_nouns, self.inv_syn_map, self.double_words = self._get_nouns()

    def _get_nouns(self) -> tuple[list[str], dict[str, str], dict[str, str]]:
        """获取MSCOCO对象词表和同义词映射"""
        mscoco_objects, inverse_syn_map = get_object_n_represent()
        valid_nouns: list[str] = mscoco_objects
        double_word_dict = get_double_word_dict()
        return valid_nouns, inverse_syn_map, double_word_dict


def get_hallu_objects(
    objects_list: list[list[str]],
    nonhallu_objects: list[str] | None,
    hallu_objects: list[str],
    spacy: SpacyModel,
    wn: WordnetModel,
    image: Image.Image | None = None,
    dino: DINO | None = None,
    yolo_results: list[str] | None = None,
    yolo_labels: list[str] | None = None,
    uncertain_objects: list[str] | None = None,
    detector_reject: dict[str, list[str]] | None = None,
    inv_syn_map: dict[str, str] | None = None,
) -> tuple[list[list[str]], list[list[str]]]:
    """
    核心函数：判断对象是否为幻觉
    
    判断策略（YOLO + DINO交叉验证）：
    1. 如果对象已在缓存中（已确认为幻觉/非幻觉/不确定），直接使用缓存结果
    2. 对于新对象：
       - YOLO认可 AND DINO认可 → 非幻觉对象
       - YOLO不认可 AND DINO不认可 → 幻觉对象
       - 其他情况 → 不确定对象
    
    特殊处理：
    - 如果对象不在YOLO的检测范围内（不在标签列表中），视为YOLO认可
    - 这样可以处理YOLO无法检测的对象类别
    
    参数:
        objects_list: 所有候选句子的对象列表
        nonhallu_objects: 已确认的非幻觉对象（会被更新）
        hallu_objects: 已确认的幻觉对象（会被更新）
        其他参数: 检测器和NLP工具
    
    返回:
        (幻觉对象列表, 非幻觉对象列表) - 对应每个候选句子
    """
    # ... 详细实现见源代码 ...


def resolve_corefs(
    spacy: SpacyModel,
    descriptions: list[str] | list[list[str]],
    previous: list[str],
    retro_num: int,
    force_list: bool = True,
) -> list[str] | list[list[str]]:
    """
    指代消解函数
    
    将代词替换为其指代的具体名词，例如：
    "A man is sitting. He is holding a book." 
    → "A man is sitting. A man is holding a book."
    
    这对于准确提取对象至关重要
    
    参数:
        spacy: Spacy模型（带fastcoref组件）
        descriptions: 当前句子
        previous: 上下文句子
        retro_num: 回溯句子数量
    """
    # ... 详细实现见源代码 ...


def object_in_set(
    obj: str,
    target_set: list[str] | set[str],
    spacy: SpacyModel,
    wn: WordnetModel,
    inv_synonym_map: dict[str, str] | None = None,
    allow_synonym: bool = False,
) -> bool:
    """
    检查对象是否在目标集合中
    
    匹配策略：
    1. 直接匹配代表词
    2. 词干匹配（lemma）
    3. 可选：同义词匹配
    """
    # ... 详细实现见源代码 ...
```

---

## 4. 模型模块详解

### 4.1 视觉语言模型生成器

#### LLaVA模型 (model/generator/llava.py)

```python
"""
llava.py - LLaVA模型封装

支持LLaVA v1.5和v1.6版本，提供统一的生成接口
可以选择使用vLLM（高效）或HuggingFace（标准）后端
"""

class LlavaModel:
    """
    LLaVA模型封装类
    
    初始化参数:
        use_vllm: 是否使用vLLM后端（推荐，更高效）
        version: 模型版本 "1.5" 或 "1.6"
        model_size: 模型大小 "7b" 或 "13b"
        gpu_util: GPU显存利用率
    
    主要方法:
        gen(): 生成响应
    """
    
    def gen(
        self,
        images: Image.Image | list[Image.Image],
        users: str | list[str],
        assistants: str | list[str] = "",
        do_sample: bool = False,
        n: int = 1,
        temp: float = 0.3,
        max_tokens: int = 512,
        single_sentence: bool = False,
    ) -> list[str] | list[list[str]]:
        """
        生成响应
        
        参数:
            images: 输入图像
            users: 用户问题/提示
            assistants: 已有的assistant响应（用于续写）
            do_sample: 是否采样（True=多样性，False=贪婪）
            n: 每个样本生成n个候选
            temp: 采样温度
            single_sentence: 是否只生成单个句子
        
        返回:
            生成的文本或文本列表
        """
        # ... 实现细节 ...
```

### 4.2 目标检测器

#### Grounding DINO (model/detector/grounding_dino.py)

```python
"""
grounding_dino.py - Grounding DINO目标检测器

Grounding DINO是一个开放词汇的目标检测器，可以根据文本描述检测任意对象
与YOLO配合使用进行交叉验证
"""

class DINO:
    """
    Grounding DINO检测器封装
    
    特点:
        - 开放词汇：可以检测任意文本描述的对象
        - 支持批量检测
        - 提供置信度阈值控制
    """
    
    def detect(
        self,
        images: Image.Image | list[Image.Image],
        captions: str | list[str],
        box_threshold=0.35,
        text_threshold=0.25,
    ) -> list[dict[str]] | dict[str]:
        """
        检测对象
        
        参数:
            images: 输入图像
            captions: 要检测的对象描述（格式: "cat.dog.person."）
            box_threshold: 边界框置信度阈值
            text_threshold: 文本匹配置信度阈值
        
        返回:
            检测结果字典，包含 scores, boxes, labels
        """
        # ... 实现细节 ...
```

#### YOLO检测器 (model/detector/yolo_model.py)

```python
"""
yolo_model.py - YOLO目标检测器

使用YOLO11x进行封闭词汇的目标检测
提供80类COCO对象的高精度检测
"""

@dataclass
class YoloResult:
    """
    YOLO检测结果封装
    
    提供便捷的结果查询接口：
        - labels: 检测到的所有标签（去重）
        - get_largest(): 获取某类别中最大的对象
        - get_smallest(): 获取某类别中最小的对象
        - get_closest_to_edge(): 获取最接近边缘的对象
    """
    original_result: Results
    result: dict[str, list[dict[str]]] = field(default_factory=dict)


class YoloModel:
    """
    YOLO模型封装
    
    支持模型:
        - yolo11x: 最新最准确
        - yolov8x-worldv2: 支持开放词汇
    """
    
    @property
    def labels(self) -> list[str]:
        """返回模型支持的所有类别标签"""
        return list(self.yolo.names.values())
    
    def predict(self, images: Image.Image | list[Image.Image]) -> list[YoloResult]:
        """执行检测并返回结果"""
        # ... 实现细节 ...
```

### 4.3 NLP工具

#### Spacy模型 (model/others/spacy_model.py)

```python
"""
spacy_model.py - Spacy NLP工具封装

提供以下功能:
1. 指代消解（使用fastcoref）
2. 词性标注
3. 词干提取
4. 名词提取
"""

class SpacyModel:
    """
    Spacy NLP模型封装
    
    主要功能:
        resolve_coref(): 指代消解
        is_noun(): 判断是否为名词
        lemma(): 词干提取
        extract_nouns_from_text(): 从文本提取名词
    """
    
    def resolve_coref(self, text: list[str] | str) -> list[str] | str:
        """
        指代消解
        
        将代词替换为其指代的具体名词
        例如: "He is eating." → "The man is eating."
        """
        # 延迟加载fastcoref组件
        if not self._loaded_fastcoref:
            self._load_fastcoref()
        # ... 实现细节 ...
    
    def lemma(self, word: str) -> str:
        """
        词干提取
        
        获取单词的词干形式
        例如: "running" → "run", "dogs" → "dog"
        """
        # ... 实现细节 ...
```

#### 场景图解析器 (model/others/sg_parser.py)

```python
"""
sg_parser.py - 场景图解析器

使用T5模型将自然语言描述转换为结构化的场景图（三元组）
用于更准确地提取句子中的对象和关系
"""

class SGParser:
    """
    场景图解析器
    
    将句子解析为(主语, 谓语, 宾语)三元组
    例如: "A man is holding a book" → [("man", "holding", "book")]
    """
    
    def pharse(self, discriptions: list[str] | str) -> list[list[list[str]]]:
        """
        解析句子为场景图
        
        输入: 句子列表
        输出: 三元组列表的列表
        """
        # ... 实现细节 ...
```

---

## 5. 数据流程图

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    SENTINEL 数据流程                      │
                    └─────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  1. 初始化阶段                                                                    │
│  ├── 加载环境变量 (.env)                                                          │
│  ├── 解析命令行参数                                                               │
│  ├── 初始化全局变量 (GVars)                                                       │
│  ├── 加载生成器 (LLaVA/Qwen2-VL)                                                  │
│  ├── 加载检测器 (YOLO + Grounding DINO)                                           │
│  └── 加载NLP工具 (Spacy, WordNet, SG Parser)                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  2. 数据加载阶段                                                                  │
│  ├── 读取数据集 (image_data.jsonl)                                                │
│  ├── 过滤已处理数据 (断点续传)                                                    │
│  └── 创建DataPoint列表                                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  3. 迭代生成循环 (每批batch_size个样本)                                           │
│                                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │  3.1 装载数据                                                              │   │
│  │      创建DataStateForBuildDataset对象                                      │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                            │
│                                      ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │  3.2 YOLO预检测                                                            │   │
│  │      对每张图像执行YOLO检测，缓存结果                                       │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                            │
│                                      ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │  3.3 候选句子生成                                                          │   │
│  │      VLM生成n=10个候选句子 (采样temp=0.7)                                   │   │
│  │      输入: 图像 + 问题 + 当前上下文                                         │   │
│  │      输出: 10个候选下一句                                                   │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                            │
│                                      ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │  3.4 指代消解                                                              │   │
│  │      使用Spacy+fastcoref解析代词                                            │   │
│  │      "He" → "The man"                                                       │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                            │
│                                      ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │  3.5 对象提取                                                              │   │
│  │      方法1: 基于MSCOCO词表的匹配                                            │   │
│  │      方法2: 场景图解析 (T5模型)                                              │   │
│  │      合并结果得到每个候选句子的对象列表                                      │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                            │
│                                      ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │  3.6 幻觉判断 (核心!)                                                       │   │
│  │      ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │      │  对于每个新提到的对象:                                           │  │   │
│  │      │  ├── YOLO检测: 对象是否在YOLO结果中?                             │  │   │
│  │      │  │   (如果不在YOLO类别中，视为通过)                              │  │   │
│  │      │  ├── DINO检测: 使用Grounding DINO验证                            │  │   │
│  │      │  │                                                               │  │   │
│  │      │  └── 判定逻辑:                                                   │  │   │
│  │      │      ├── YOLO✓ AND DINO✓ → 非幻觉对象                           │  │   │
│  │      │      ├── YOLO✗ AND DINO✗ → 幻觉对象                             │  │   │
│  │      │      └── 其他情况 → 不确定对象                                   │  │   │
│  │      └─────────────────────────────────────────────────────────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                            │
│                                      ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │  3.7 偏好对构建                                                            │   │
│  │      ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │      │  分类候选句子:                                                   │  │   │
│  │      │  ├── 非幻觉候选: 有对象且无幻觉对象                              │  │   │
│  │      │  │   ├── 探索型: 包含上下文中未出现的新对象 ★优先                │  │   │
│  │      │  │   └── 重复型: 只提及已有对象                                  │  │   │
│  │      │  └── 幻觉候选: 包含幻觉对象                                      │  │   │
│  │      │                                                                   │  │   │
│  │      │  构建偏好对:                                                      │  │   │
│  │      │  y_win (胜出) = 非幻觉句子                                        │  │   │
│  │      │  y_lose (失败) = 幻觉句子                                         │  │   │
│  │      └─────────────────────────────────────────────────────────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                            │
│                                      ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │  3.8 更新上下文                                                            │   │
│  │      选择最佳句子追加到assistant                                            │   │
│  │      选择策略: 优先探索型 > 重复型 > 随机                                   │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                            │
│                                      ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │  3.9 检查终止条件                                                          │   │
│  │      如果>50%的候选为空 → 标记完成                                          │   │
│  │      否则 → 继续下一轮迭代                                                  │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                            │
│                         ┌────────────┴────────────┐                              │
│                         ▼                          ▼                              │
│                    [未完成]                   [已完成]                            │
│                    继续迭代                   保存结果                            │
│                         │                          │                              │
│                         └──────────┬───────────────┘                              │
│                                    │                                              │
└────────────────────────────────────┼──────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  4. 输出文件                                                                     │
│  ├── <model_name>.jsonl                                                          │
│  │   包含: image_id, caption, hallu_objects, nonhallu_objects等                  │
│  │                                                                               │
│  └── <model_name>_data_pair.jsonl                                                │
│      包含: image_path, question, context, y_win, y_lose                          │
│      用于DPO训练的偏好对数据                                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. 关键算法解析

### 6.1 幻觉检测算法

```
输入: 对象列表 objects, 已知非幻觉对象 nonhallu, 已知幻觉对象 hallu
输出: 更新后的 nonhallu, hallu, 以及每个句子的分类结果

算法流程:
1. 获取缓存对象 cached = nonhallu ∪ hallu ∪ uncertain
2. 获取未缓存对象 uncached = objects - cached

3. 对于每个 obj ∈ uncached:
   a. yolo_ok = (obj ∈ YOLO检测结果) OR (obj ∉ YOLO标签集)
   b. dino_ok = DINO.detect(image, obj) 返回非空结果
   
   c. 判定:
      if yolo_ok AND dino_ok:
          nonhallu.add(obj)  # 确认存在
      elif NOT yolo_ok AND NOT dino_ok:
          hallu.add(obj)     # 确认幻觉
      else:
          uncertain.add(obj) # 检测器不一致

4. 返回分类结果
```

### 6.2 偏好对构建算法

```
输入: 候选句子集合 S, 当前上下文 C
输出: 偏好对列表, 最佳句子索引

算法流程:
1. 对每个句子 s ∈ S:
   a. 提取对象: objs = extract_objects(s)
   b. 分类对象: hallu_objs, nonhallu_objs = classify(objs)
   
2. 分类句子:
   a. 非幻觉句子: nonhallu_sents = {s | hallu_objs(s) = ∅ AND objs(s) ≠ ∅}
   b. 幻觉句子: hallu_sents = {s | hallu_objs(s) ≠ ∅}

3. 进一步分类非幻觉句子:
   a. 探索型: explore = {s | objs(s) ∩ context_objs = ∅}
   b. 重复型: repeat = nonhallu_sents - explore

4. 构建偏好对:
   pairs = zip(repeat, hallu_sents)  # y_win=非幻觉, y_lose=幻觉

5. 选择最佳句子:
   if explore ≠ ∅:
       return random.choice(explore)  # 优先探索新对象
   elif repeat ≠ ∅:
       return random.choice(repeat)   # 其次选非幻觉
   else:
       return random.choice(S)        # 随机选择
```

### 6.3 上下文迭代引导算法

```
算法核心思想:
通过选择非幻觉句子作为上下文，引导模型在后续生成中避免幻觉

迭代过程:
Context_0 = ""
for step in range(max_steps):
    candidates = VLM.generate(Image, Question, Context_{step-1}, n=10)
    
    # 分类候选
    nonhallu, hallu = classify_candidates(candidates)
    
    # 构建偏好对用于训练
    pairs.append({
        "context": Context_{step-1},
        "y_win": nonhallu[random],
        "y_lose": hallu[random]
    })
    
    # 选择最佳句子更新上下文
    best = select_best(nonhallu, explore_first=True)
    Context_{step} = Context_{step-1} + best
    
    if generation_finished:
        break

关键洞察:
- 通过选择非幻觉句子，上下文逐渐累积真实信息
- 这种累积使得后续生成更不容易产生幻觉（利用上下文一致性）
- 偏好对捕获了"给定相同上下文，选择非幻觉输出"的偏好
```

---

## 7. 使用说明

### 7.1 生成训练数据

```bash
# 基本用法
python main.py

# 指定模型
python main.py --model LLaVA_v1_5_7b

# 指定批处理大小
python main.py --batch_size 10

# 指定数据量
python main.py --num_of_data 1000
```

### 7.2 转换为训练格式

```bash
# 转换为LLaMA-Factory格式
python utils/get_llama_factory_data_pair.py

# 转换为LLaVA-v1.5格式
python utils/get_llava_v15_data_pair.py
```

### 7.3 关键配置

```python
# generate_dataset.py 中的配置
DEBUG = True           # 调试模式（不编译模型）
HALLUCI_CONTEXT = False  # 是否使用幻觉句子更新上下文
CHECK_TYPE = "any"     # 对象检查类型

# 生成参数
n = 10                 # 每步生成候选数
temp = 0.7             # 采样温度
single_sentence = True # 单句生成
```

---

## 8. 总结

SENTINEL项目通过以下创新点解决VLM幻觉问题：

1. **句子级早期干预**: 在每个句子生成后立即检测和干预
2. **检测器交叉验证**: 使用YOLO和DINO双重验证减少误判
3. **迭代式上下文构建**: 通过选择非幻觉句子累积可靠上下文
4. **无需人工标注**: 完全自动化的偏好数据构建流程

项目架构清晰，模块化程度高，易于扩展到新的VLM模型。
