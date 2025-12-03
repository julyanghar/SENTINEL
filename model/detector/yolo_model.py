"""
YOLO 目标检测器模块
===================

YOLO（You Only Look Once）是一个快速的封闭词汇目标检测器，
可以实时检测图像中的物体。本模块使用 YOLO11x 或 YOLOv8 模型。

在 SENTINEL 中的作用:
    - 快速预检测图像中的物体，建立物体基准
    - 与 Grounding DINO 交叉验证，提高幻觉检测准确性
    
特点:
    - 速度快：比 Grounding DINO 快很多
    - 封闭词汇：只能检测预定义的 80 类 COCO 物体
    - 准确：在标准物体上准确率很高

使用示例:
    >>> yolo = YoloModel("yolo11x", model_dir="/models")
    >>> result = yolo.predict(image)
    >>> print(result.labels)  # ['person', 'dog', 'car']
"""

import os
from dataclasses import dataclass, field
from logging import Logger

import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from ..utils.utils import ensure_lists, maybe_return_ls


@dataclass
class YoloResult:
    """
    YOLO 检测结果封装类
    
    将 YOLO 的原始检测结果转换为更易用的格式，并提供便捷的访问方法。
    
    Attributes:
        original_result: YOLO 原始检测结果
        result: 按标签分组的检测结果
                格式: {标签: [{conf: 置信度, xywhn: 归一化坐标}, ...]}
    
    Example:
        >>> result = yolo.predict(image)
        >>> print(result.labels)  # ['person', 'dog']
        >>> print(result.get_largest('person'))  # 返回最大的 person 框
    """
    
    original_result: Results
    """YOLO 原始检测结果对象"""
    
    result: dict[str, list[dict[str]]] = field(default_factory=dict)
    """按标签分组的检测结果"""

    def __post_init__(self):
        """
        初始化后处理：将原始结果转换为按标签分组的格式
        
        转换逻辑:
        1. 获取类别ID到名称的映射
        2. 遍历所有检测框
        3. 按标签分组存储置信度和坐标信息
        """
        # 类别ID到名称的映射 (如 {0: 'person', 1: 'bicycle', ...})
        mapping_dict: dict[int, str] = self.original_result.names
        
        # 提取检测结果
        classes: list[int] = self.original_result.boxes.cls.cpu().to(dtype=int).tolist()
        confidences: list[float] = self.original_result.boxes.conf.cpu().tolist()
        # xywhn: 归一化的中心点坐标和宽高 (x_center, y_center, width, height)
        xywhns: list[list[float]] = self.original_result.boxes.xywhn.cpu().tolist()

        # 按标签分组
        for cls, conf, xywhn in zip(classes, confidences, xywhns):
            label = mapping_dict[cls]
            if label not in self.result:
                self.result[label] = []
            self.result[label].append({"conf": conf, "xywhn": xywhn})

    def _calculate_area(self, xywhn: list[float]) -> float:
        """
        计算归一化边界框的面积
        
        Args:
            xywhn: 归一化坐标 [x_center, y_center, width, height]
        
        Returns:
            归一化面积（0-1 之间）
        """
        return xywhn[2] * xywhn[3]

    def _calculate_min_dist_to_edge(self, xywhn: list[float]) -> float:
        """
        计算边界框中心到图像边缘的最小距离
        
        用于判断物体是否靠近图像边缘（可能被裁切）
        
        Args:
            xywhn: 归一化坐标 [x_center, y_center, width, height]
        
        Returns:
            到最近边缘的归一化距离（0-0.5 之间）
        """
        return min(xywhn[0], 1 - xywhn[0], xywhn[1], 1 - xywhn[1])

    @property
    def labels(self) -> list[str]:
        """
        获取检测到的所有物体标签（无重复）
        
        Returns:
            标签列表，如 ['person', 'dog', 'car']
        """
        return list(self.result.keys())

    def get_largest(self, label: str) -> dict[str, float | list[float]] | None:
        """
        获取指定标签中面积最大的物体
        
        Args:
            label: 物体标签（如 'person'）
        
        Returns:
            最大物体的信息 {'conf': 置信度, 'xywhn': 坐标}，
            如果标签不存在则返回 None
        """
        if label not in self.result:
            return None
        return max(self.result[label], key=lambda x: self._calculate_area(x["xywhn"]))

    def get_smallest(self, label: str) -> dict[str, float | list[float]] | None:
        """
        获取指定标签中面积最小的物体
        
        用于识别小物体（可能更容易被模型忽略）
        
        Args:
            label: 物体标签
        
        Returns:
            最小物体的信息，如果标签不存在则返回 None
        """
        if label not in self.result:
            return None
        return min(self.result[label], key=lambda x: self._calculate_area(x["xywhn"]))

    def get_closest_to_edge(self, label: str) -> dict[str, float | list[float]] | None:
        """
        获取指定标签中最接近图像边缘的物体
        
        用于识别边缘物体（可能更容易产生幻觉）
        
        Args:
            label: 物体标签
        
        Returns:
            最接近边缘的物体信息，如果标签不存在则返回 None
        """
        if label not in self.result:
            return None
        return min(self.result[label], key=lambda x: self._calculate_min_dist_to_edge(x["xywhn"]))

    def get_farthest_to_edge(self, label: str) -> dict[str, float | list[float]] | None:
        """
        获取指定标签中最远离图像边缘的物体
        
        Args:
            label: 物体标签
        
        Returns:
            最远离边缘的物体信息，如果标签不存在则返回 None
        """
        if label not in self.result:
            return None
        return max(self.result[label], key=lambda x: self._calculate_min_dist_to_edge(x["xywhn"]))

    def __repr__(self):
        """返回结果的字符串表示"""
        return str(self.result)


class YoloModel:
    """
    YOLO 目标检测器封装类
    
    支持 YOLO11x 和 YOLOv8-World 模型，用于快速检测图像中的物体。
    
    Attributes:
        model_path: 模型文件路径
        yolo: YOLO 模型实例
        labels: 可检测的物体标签列表（80 类 COCO 物体）
    
    Example:
        >>> yolo = YoloModel("yolo11x", model_dir="/models", device="cuda:0")
        >>> 
        >>> # 单图像检测
        >>> result = yolo.predict(image)
        >>> print(result.labels)  # ['person', 'dog']
        >>> 
        >>> # 批量检测
        >>> results = yolo.predict([img1, img2], force_list=True)
    """
    
    def __init__(
        self,
        model_name: str,
        model_dir: str | None = None,
        device: torch.device | str = "cpu",
        logger: Logger | None = None,
    ):
        """
        初始化 YOLO 检测器
        
        Args:
            model_name: 模型名称
                - 'yolo11x': YOLO11 Extra Large，精度最高
                - 'yolov8x-worldv2': YOLOv8-World，支持开放词汇
            model_dir: 模型文件目录
                      模型文件应位于 {model_dir}/../yolo/{model_name}.pt
            device: 运行设备（如 'cuda:0' 或 'cpu'）
            logger: 可选的日志器
        
        Note:
            首次使用时会自动下载模型文件
        """
        assert model_name in [
            "yolo11x",
            "yolov8x-worldv2",
        ], "Invalid model name."

        if logger:
            logger.info(f"Loading YOLO model {model_name} from {model_dir}")
        
        # 构建模型路径
        self.model_path: str = (
            os.path.join(os.path.dirname(os.path.expanduser(model_dir)), "yolo", f"{model_name}.pt")
            if model_dir
            else model_name
        )
        
        # 加载模型并移动到指定设备
        self.yolo = YOLO(self.model_path, verbose=False).to(device)

    @property
    def labels(self) -> list[str]:
        """
        获取模型可检测的所有物体标签
        
        Returns:
            80 类 COCO 物体标签列表，如
            ['person', 'bicycle', 'car', ..., 'toothbrush']
        """
        return list(self.yolo.names.values())

    def predict(
        self,
        images: Image.Image | list[Image.Image],
        force_list: bool = False,
        save_predict_result: bool = False,
    ) -> YoloResult | list[YoloResult]:
        """
        对图像进行目标检测
        
        Args:
            images: 输入图像，可以是单个 PIL.Image 或图像列表
            force_list: 是否强制返回列表
            save_predict_result: 是否保存检测结果图像（用于调试）
        
        Returns:
            YoloResult 对象或对象列表，包含检测到的物体信息
        
        Example:
            >>> result = yolo.predict(image)
            >>> print(result.labels)  # ['person', 'dog', 'car']
            >>> largest_person = result.get_largest('person')
            >>> print(largest_person['conf'])  # 0.95
        """
        images = ensure_lists(images)

        # 调用 YOLO 进行检测
        results = self.yolo.predict(source=images, verbose=False, save=save_predict_result)

        # 将原始结果封装为 YoloResult 对象
        return maybe_return_ls(force_list, [YoloResult(original_result=result) for result in results])
