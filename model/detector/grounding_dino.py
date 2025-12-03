"""
Grounding DINO 目标检测器模块
=============================

Grounding DINO 是一个开放词汇目标检测器，可以根据文本描述检测图像中的任意物体。
与封闭词汇检测器（如 YOLO）不同，它不限于预定义的物体类别。

在 SENTINEL 中的作用:
    - 验证模型生成的物体是否真实存在于图像中
    - 与 YOLO 交叉验证，提高幻觉检测的准确性
    
工作原理:
    输入: 图像 + 物体名称列表（如 "cat. dog. person."）
    输出: 检测到的物体边界框、置信度和标签

使用示例:
    >>> dino = DINO("base", device="cuda:0")
    >>> result = dino.detect(image, "cat. dog.")
    >>> print(result["labels"])  # ['cat'] - 只检测到了猫
"""

import warnings
from logging import Logger

import torch
from PIL import Image
from transformers import (
    BertTokenizerFast,
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
)
from transformers.models.grounding_dino.modeling_grounding_dino import (
    GroundingDinoObjectDetectionOutput,
)

from ..utils.utils import ensure_lists, is_dino_sentences_equal, maybe_return_ls

# ==================== 检测阈值配置 ====================

BOX_THRESHOLD = 0.35
"""边界框置信度阈值，低于此值的检测结果会被过滤"""

TEXT_THRESHOLD = 0.25
"""文本匹配置信度阈值，低于此值的标签会被过滤"""


class DINO:
    """
    Grounding DINO 目标检测器封装类
    
    Grounding DINO 是一个基于 Transformer 的开放词汇目标检测模型，
    可以根据自然语言描述检测图像中的物体。
    
    模型来源: IDEA-Research/grounding-dino-{size}
    
    Attributes:
        size: 模型大小 ('tiny' 或 'base')
        model_dir: 模型缓存目录
        device: 运行设备
        model: GroundingDino 模型实例
        processor: 数据预处理器
        tokenizer: BERT 分词器
    
    Example:
        >>> dino = DINO("base", model_dir="/models", device="cuda:0")
        >>> 
        >>> # 检测单个图像
        >>> result = dino.detect(image, "cat. dog. person.")
        >>> print(result["labels"])  # ['cat', 'person']
        >>> print(result["scores"])  # tensor([0.89, 0.76])
        >>> 
        >>> # 批量检测
        >>> results = dino.detect([img1, img2], ["cat.", "dog."], force_list=True)
    """
    
    def __init__(
        self,
        size: str = "base",
        model_dir: str = "",
        torch_dtype: torch.dtype | str = "auto",  # Don't change
        device: torch.device | str = "cpu",
        logger: Logger | None = None,
    ):
        """
        初始化 Grounding DINO 检测器
        
        Args:
            size: 模型大小
                - 'tiny': 172M 参数，速度快但精度略低
                - 'base': 233M 参数，精度高（推荐）
            model_dir: 模型缓存目录，模型会自动下载到此目录
            torch_dtype: 数据类型，'auto' 会自动选择最优类型
            device: 运行设备，如 'cuda:0' 或 'cpu'
            logger: 可选的日志器
        
        Note:
            首次使用时会自动从 HuggingFace 下载模型
        """
        assert size in [
            "tiny",
            "base",
        ], "Invalid model size. Choose from 'tiny' or 'base'."

        if logger:
            logger.info(f"Loading grounding DINO model with size {size} on {device}")
        
        self.size = size
        self.model_dir = model_dir
        self.torch_dtype = torch_dtype
        self.device = device
        
        # 加载模型和处理器
        self.model, self.processor = self._create()
        self.tokenizer: BertTokenizerFast = self.processor.tokenizer
        
        # 空结果模板，用于处理空输入
        self.empty_dino_result: dict = {
            "scores": torch.tensor([]),
            "boxes": torch.tensor([]),
            "labels": [],
        }

    def _create(self) -> tuple[GroundingDinoForObjectDetection, GroundingDinoProcessor]:
        """
        Create a grounding DINO model and processor based on the specified size.

        Returns:
            A tuple containing the model and processor.
        """
        model_name = f"IDEA-Research/grounding-dino-{self.size}"

        model: GroundingDinoForObjectDetection = (
            GroundingDinoForObjectDetection.from_pretrained(
                model_name,
                cache_dir=self.model_dir,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                disable_custom_kernels=True,
            )
            .to(self.device)
            .eval()
        )

        processor: GroundingDinoProcessor = GroundingDinoProcessor.from_pretrained(
            model_name,
            cache_dir=self.model_dir,
            # Do not set padding_side!
        )

        return model, processor

    def detect(
        self,
        images: Image.Image | list[Image.Image],
        captions: str | list[str],
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        force_list: bool = False,
    ) -> list[dict[str]] | dict[str]:
        """
        使用 Grounding DINO 检测图像中的物体
        
        Args:
            images: 输入图像，可以是单个 PIL.Image 或图像列表
            captions: 要检测的物体名称，格式为 "object1. object2. object3."
                     必须小写，以点号分隔，如 "cat. dog. person."
            box_threshold: 边界框置信度阈值（默认 0.35）
            text_threshold: 文本匹配置信度阈值（默认 0.25）
            force_list: 是否强制返回列表
        
        Returns:
            检测结果字典（或字典列表），包含：
            - 'scores': torch.Tensor - 每个检测框的置信度
            - 'boxes': torch.Tensor - 边界框坐标 (x1, y1, x2, y2)
            - 'labels': list[str] - 检测到的物体名称
        
        Example:
            >>> # 单图像检测
            >>> result = dino.detect(image, "cat. dog.")
            >>> print(result)
            {
                'scores': tensor([0.89, 0.76]),
                'boxes': tensor([[10, 20, 100, 150], [200, 50, 300, 200]]),
                'labels': ['cat', 'dog']
            }
            
            >>> # 批量检测
            >>> results = dino.detect([img1, img2], ["cat.", "dog."], force_list=True)
        
        Note:
            - 如果 caption 为空字符串，返回空结果
            - caption 中的物体不一定都能被检测到
            - 相同的图像-文本对会被缓存，避免重复计算
        """
        images, captions = ensure_lists(images, captions)

        # 初始化结果列表，空 caption 直接使用空结果
        results: list = [None if caption else self.empty_dino_result for caption in captions]

        # 过滤出有效的 caption 和对应的图像
        valid_captions: list[str] = [c for c in captions if c]
        filtered_images: list[Image.Image] = [img for i, img in enumerate(images) if captions[i]]

        # 对有效输入进行检测
        if valid_captions:
            detection_res = self._detect_wo_repetition(
                filtered_images,
                valid_captions,
                box_threshold,
                text_threshold,
            )

            # 将检测结果填充到对应位置
            result_index = 0
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = detection_res[result_index] if isinstance(detection_res, list) else detection_res
                    result_index += 1

        return maybe_return_ls(force_list, results)

    def _detect_wo_repetition(
        self,
        images: list[Image.Image],
        captions: list[str],
        box_threshold: float = BOX_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
    ) -> list[dict[str]]:
        # 收集唯一的图像和文本对
        unique_pairs: dict[tuple[str, str], tuple[Image.Image, str]] = {}
        for image, caption in zip(images, captions):
            for key in unique_pairs.keys():
                if key[0] == image.tobytes():
                    if is_dino_sentences_equal(key[1], caption):
                        break
            else:
                unique_pairs[(image.tobytes(), caption)] = (image, caption)

        unique_images, unique_captions = zip(*unique_pairs.values())

        # 调用 _detect 方法，处理所有唯一的图像文本对
        output: list[dict] = self._detect(
            list(unique_images),
            list(unique_captions),
            box_threshold,
            text_threshold,
        )

        # 构建结果列表，按原顺序填充
        results: list[dict[str]] = []
        for image, caption in zip(images, captions):
            for key, _ in unique_pairs.items():
                if key[0] == image.tobytes() and is_dino_sentences_equal(key[1], caption):
                    results.append(output[list(unique_pairs.keys()).index(key)])

        return results

    def _detect(
        self,
        images: list[Image.Image],
        captions: list[str],
        box_threshold: float = BOX_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
    ) -> list[dict[str]]:
        with torch.inference_mode():
            encoded_inputs = self.processor(
                images=images,
                text=captions,
                max_length=200,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            outputs: GroundingDinoObjectDetectionOutput = self.model(**encoded_inputs)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                target_sizes = [image.size[::-1] for image in images]
                # 字典中包含 "scores", "boxes", "labels" 三个字段
                results: list[dict] = self.processor.post_process_grounded_object_detection(
                    outputs,
                    encoded_inputs["input_ids"],
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes,
                )
        return results
