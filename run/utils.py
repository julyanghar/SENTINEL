"""
SENTINEL 核心工具函数模块
========================

本模块包含 SENTINEL 项目中使用的所有核心工具函数，主要分为以下几类：

1. 数据处理工具
   - set_to_list: 递归转换 set 为 list
   - save_result: 保存结果到文件
   - log_progress: 记录处理进度

2. 图像处理工具
   - open_image / open_images: 打开图像文件
   - annotate_with_dino_result: 在图像上绘制检测框
   - crop_with_dino_boxes: 根据检测框裁剪图像

3. 物体提取工具
   - extract_obj_w_gt: 从文本中提取 MSCOCO 物体
   - extract_obj_from_textgraphs: 从场景图中提取物体
   - process_triple: 处理场景图三元组
   - add_to_nouns_if_valid: 验证并添加有效名词

4. 物体匹配工具
   - object_in_set: 检查物体是否在集合中（支持同义词）
   - objects_in_set: 批量检查物体是否在集合中

5. DINO 检测工具
   - pack_objs_for_dino: 打包物体列表为 DINO 输入格式
   - unpack_objs_from_dino: 解包 DINO 输出格式
   - get_dino_detected_objects: 从 DINO 结果中提取检测到的物体

6. 幻觉检测工具
   - get_hallu_objects: 单样本幻觉检测
   - b_get_hallu_objects: 批量幻觉检测（核心函数）

7. 文本处理工具
   - tokenize_sent: 句子分词
   - get_finish_flag: 判断生成是否结束
   - concat_sents: 拼接句子形成上下文
   - resolve_corefs: 指代消解
   - pharse_w_context: 带上下文的句子解析

8. YOLO 检测工具
   - yolo_detect: 批量 YOLO 检测

9. 辅助工具
   - pop_first_sents: 提取句子的第一句
"""

import json
import os
from argparse import Namespace
from dataclasses import dataclass, field
from logging import Logger

import cv2
import numpy as np
import requests
import supervision as sv
import torch
from PIL import Image
from spacy.tokens import Doc, Token

from model.auxiliary.datastate import DataStateForBuildDataset
from model.detector.grounding_dino import DINO
from model.detector.yolo_model import YoloModel, YoloResult
from model.others.sg_parser import SGParser
from model.others.spacy_model import SpacyModel
from model.others.wordnet import WordnetModel
from model.utils.utils import ensure_lists
from run.object_utils import get_double_word_dict, get_object_n_represent

if __name__ == "__main__":
    print("Please run main.py")
    exit(0)


# ==================== 数据结构 ====================


@dataclass
class refModel:
    """
    参考模型数据类 - 存储物体识别所需的映射表
    
    用于存储 MSCOCO 物体列表、同义词映射等参考数据，
    在物体提取和匹配过程中提供查找表支持。
    
    Attributes:
        args: 命令行参数
        valid_nouns: MSCOCO 80 类物体及其同义词列表
            例如: ["person", "girl", "boy", "bicycle", "bike", ...]
        inv_syn_map: 逆同义词映射，将同义词映射到代表词
            例如: {"puppy": "dog", "sedan": "car", ...}
        double_words: 双词物体映射
            例如: {"hot dog": "hot dog", "cell phone": "cell phone", ...}
        gt_anno: Ground Truth 标注（可选，用于评估）
    
    Example:
        >>> ref = refModel(args=args)
        >>> "dog" in ref.valid_nouns  # True
        >>> ref.inv_syn_map["puppy"]  # "dog"
    """
    args: Namespace
    valid_nouns: list[str] = field(init=False)
    inv_syn_map: dict[str, str] = field(init=False)
    double_words: dict[str, str] = field(init=False)
    gt_anno: dict[str, dict] = field(init=False, default=None)

    def __post_init__(self):
        """初始化后自动加载物体映射表"""
        self.valid_nouns, self.inv_syn_map, self.double_words = self._get_nouns()

    def _get_nouns(self) -> tuple[list[str], dict[str, str], dict[str, str]]:
        """
        加载 MSCOCO 物体列表和映射表
        
        Returns:
            tuple: (有效物体列表, 逆同义词映射, 双词物体映射)
        """
        # Step 1: 获取 MSCOCO 物体列表和同义词映射
        mscoco_objects, inverse_syn_map = get_object_n_represent()
        valid_nouns: list[str] = mscoco_objects
        
        # Step 2: 获取双词物体映射（如 "hot dog", "cell phone"）
        double_word_dict = get_double_word_dict()
        
        return valid_nouns, inverse_syn_map, double_word_dict


# ==================== 数据处理工具 ====================


def set_to_list(obj: set | dict | list) -> list | dict:
    """
    递归地将数据结构中的 set 转换为 list
    
    JSON 不支持 set 类型，在保存结果前需要将所有 set 转换为 list。
    本函数递归处理嵌套的 dict 和 list 结构。
    
    Args:
        obj: 待转换的对象，可以是 set、dict、list 或其他类型
    
    Returns:
        转换后的对象：
        - set → list
        - dict → dict（值递归转换）
        - list → list（元素递归转换）
        - 其他类型 → 原样返回
    
    Example:
        >>> set_to_list({1, 2, 3})
        [1, 2, 3]
        >>> set_to_list({"a": {1, 2}, "b": [3, {4, 5}]})
        {"a": [1, 2], "b": [3, [4, 5]]}
    """
    if isinstance(obj, set):
        # Step 1: set 直接转换为 list
        return list(obj)
    elif isinstance(obj, dict):
        # Step 2: dict 递归处理每个值
        return {k: set_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Step 3: list 递归处理每个元素
        return [set_to_list(element) for element in obj]
    else:
        # Step 4: 其他类型原样返回
        return obj


def save_result(file_path: str, results: dict | list[dict] | None) -> None:
    """
    保存结果到指定路径的文件中（追加模式）
    
    支持 .jsonl 和 .json 格式，使用追加模式写入，支持断点续传。
    自动创建目录，自动将 set 转换为 list。
    
    Args:
        file_path: 保存文件的路径
            - .jsonl: 每行一个 JSON 对象（推荐，支持追加）
            - .json: 整个文件是一个 JSON 对象
        results: 需要保存的结果
            - dict: 单个结果
            - list[dict]: 多个结果
            - 迭代器: 自动转换为列表
    
    Returns:
        None
    
    Example:
        >>> save_result("results.jsonl", {"id": 1, "score": 0.95})
        >>> save_result("results.jsonl", [{"id": 2}, {"id": 3}])
    
    Note:
        - 使用 "a+" 追加模式，不会覆盖已有内容
        - 自动创建目录结构
        - 自动处理 set → list 转换
    """
    # Step 1: 空值检查
    if not file_path or not results:
        return

    # Step 2: 如果 results 是迭代器，转换为列表
    if hasattr(results, "__iter__") and not isinstance(results, (dict, list)):
        results = list(results)

    # Step 3: 空列表检查
    if isinstance(results, list) and not results:
        return

    # Step 4: 将 set 转换为 list（JSON 不支持 set）
    results = set_to_list(results)

    # Step 5: 获取文件扩展名并创建目录
    ext: str = os.path.splitext(file_path)[-1]
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Step 6: 根据扩展名选择保存格式
    with open(file_path, "a+", encoding="utf-8") as f:
        if ext == ".jsonl":
            # JSONL 格式：每行一个 JSON 对象
            if isinstance(results, list):
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(results, ensure_ascii=False) + "\n")
        elif ext in {".json", ".jsonfile"}:
            # JSON 格式：整个文件是一个 JSON 对象
            json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unspported extension {ext} for file: {file_path}")


def log_progress(
    logger: Logger | None, finished_data_num: int, num_of_data: int, batch_size: int, taken_time: float
) -> None:
    """
    记录处理进度日志
    
    在批处理过程中定期调用，输出当前进度信息。
    
    Args:
        logger: 日志记录器，如果为 None 则不输出
        finished_data_num: 已完成的数据数量
        num_of_data: 总数据数量
        batch_size: 当前批处理大小
        taken_time: 本批次处理所用时间（秒）
    
    Returns:
        None
    
    输出格式:
        "Progress: 100/1000, Batch size: 5, Time: 3.25s"
    """
    if logger:
        logger.info(f"Progress: {finished_data_num}/{num_of_data}, Batch size: {batch_size}, Time: {taken_time:.2f}s")


# ==================== 图像处理工具 ====================


def open_image(image: Image.Image | str) -> Image.Image:
    """
    打开单个图像并确保模式为 RGB
    
    支持多种输入格式：PIL.Image 对象、本地路径、URL。
    自动将非 RGB 模式（如 RGBA、L）转换为 RGB。
    
    Args:
        image: 图像来源
            - PIL.Image: 直接使用
            - str (本地路径): 从文件读取，如 "/path/to/image.jpg"
            - str (URL): 从网络下载，如 "http://example.com/image.jpg"
    
    Returns:
        PIL.Image.Image: RGB 模式的图像对象
    
    Raises:
        TypeError: 如果输入类型不支持
    
    Example:
        >>> img = open_image("/path/to/image.jpg")
        >>> img = open_image("http://example.com/image.png")
        >>> img.mode
        'RGB'
    """
    # Step 1: 根据输入类型获取图像
    if isinstance(image, Image.Image):
        # 已经是 PIL.Image，直接使用
        img = image
    elif isinstance(image, str):
        if image.startswith("http"):
            # URL：使用 requests 下载
            img = Image.open(requests.get(image, stream=True).raw)
        else:
            # 本地路径：直接打开
            img = Image.open(image)
    else:
        raise TypeError(f"image must be a URL string or a PIL.Image object, but got {type(image)}")

    # Step 2: 确保图像模式为 RGB（某些模型只接受 RGB）
    if img.mode != "RGB":
        img: Image.Image = img.convert("RGB")
    return img


def open_images(images: Image.Image | str | list[Image.Image | str]) -> Image.Image | list[Image.Image]:
    """
    批量打开图像（支持单个或列表输入）
    
    对 open_image 的封装，支持批量处理。
    
    Args:
        images: 图像来源
            - 单个: PIL.Image | str
            - 批量: list[PIL.Image | str]
    
    Returns:
        - 单个输入: PIL.Image.Image
        - 列表输入: list[PIL.Image.Image]
    
    Example:
        >>> img = open_images("/path/to/image.jpg")  # 单个
        >>> imgs = open_images(["/path/1.jpg", "/path/2.jpg"])  # 批量
    """
    if isinstance(images, list):
        # 批量处理
        return [open_image(i) for i in images]
    else:
        # 单个处理
        return open_image(images)


def annotate_with_dino_result(
    image: Image.Image | str,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: list[str],
) -> np.ndarray:
    """
    在图像上绘制 DINO 检测结果（边界框和标签）
    
    使用 supervision 库在图像上绘制检测框和标签，用于可视化调试。
    
    Args:
        image: 原始图像（PIL.Image 或路径）
        boxes: 边界框坐标张量，形状 [N, 4]，格式 (x1, y1, x2, y2)
        scores: 置信度张量，形状 [N]
        labels: 标签列表，长度 N
    
    Returns:
        np.ndarray: 标注后的图像（BGR 格式，OpenCV 格式）
    
    Example:
        >>> result = dino.detect(image, "cat. dog.")
        >>> annotated = annotate_with_dino_result(
        ...     image, result["boxes"], result["scores"], result["labels"]
        ... )
        >>> cv2.imwrite("annotated.jpg", annotated)
    
    Note:
        返回的是 BGR 格式，可直接用 cv2.imwrite 保存
    """
    # Step 1: 确保图像是 PIL.Image 对象
    image: Image.Image = open_images(image)

    # Step 2: 创建 supervision 检测对象
    detections: sv.Detections = sv.Detections(xyxy=boxes.cpu().numpy())

    # Step 3: 构建标签字符串（格式："物体名 置信度"）
    labels: list[str] = [f"{phrase} {score:.2f}" for phrase, score in zip(labels, scores)]

    # Step 4: 创建标注器
    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    
    # Step 5: 将图像转换为 OpenCV 格式 (RGB → BGR)
    annotated_img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    
    # Step 6: 绘制边界框和标签
    annotated_img = bbox_annotator.annotate(scene=annotated_img, detections=detections)
    annotated_img = label_annotator.annotate(scene=annotated_img, detections=detections, labels=labels)
    
    return annotated_img


def crop_with_dino_boxes(image: Image.Image, boxes: torch.Tensor) -> list[Image.Image]:
    """
    根据检测框裁剪图像
    
    将图像按照 DINO 检测到的边界框裁剪成多个子图像。
    
    Args:
        image: 原始 PIL 图像
        boxes: 边界框坐标张量，形状 [N, 4]，格式 (x1, y1, x2, y2)
    
    Returns:
        list[PIL.Image.Image]: 裁剪后的图像列表，每个框对应一个
    
    Example:
        >>> result = dino.detect(image, "cat.")
        >>> crops = crop_with_dino_boxes(image, result["boxes"])
        >>> crops[0].save("cat_crop.jpg")
    """
    # 对每个边界框裁剪图像
    return [image.crop(tuple(box.tolist())) for box in boxes]


# ==================== 物体提取工具 ====================


def extract_obj_w_gt(
    discriptions: list[str] | str,
    valid_nouns: list[str],
    double_word_dict: dict,
    inv_synonym_map: dict,
    wn: WordnetModel,
    force_list: bool = False,
    return_repr: bool = True,
) -> list[list[str]]:
    """
    从文本中提取 MSCOCO 物体（基于预定义物体列表的正则匹配方法）
    
    这是物体提取的核心函数之一，通过词形还原和同义词映射，
    从自然语言文本中提取出 MSCOCO 80 类物体。
    
    Args:
        discriptions: 输入文本
            - str: 单个句子
            - list[str]: 多个句子
        valid_nouns: 有效物体列表（MSCOCO 80 类及其同义词）
        double_word_dict: 双词物体映射
            例如: {"hot dog": "hot dog", "cell phone": "cell phone"}
        inv_synonym_map: 逆同义词映射，将同义词映射到代表词
            例如: {"puppy": "dog", "sedan": "car"}
        wn: WordNet 模型实例，用于分词和词形还原
        force_list: 是否强制返回列表格式
        return_repr: 是否返回代表词（True）还是原始词（False）
    
    Returns:
        list[list[str]]: 每个句子提取出的物体列表
            - force_list=True 或多句子输入: [[obj1, obj2], [obj3], ...]
            - force_list=False 且单句子输入: [obj1, obj2]
    
    Example:
        >>> extract_obj_w_gt(
        ...     "A puppy is playing with a hot dog",
        ...     valid_nouns, double_word_dict, inv_syn_map, wn
        ... )
        ["dog", "hot dog"]  # puppy → dog (同义词映射)
    
    处理流程:
        1. 分词并词形还原（dogs → dog）
        2. 合并双词物体（hot dog）
        3. 特殊处理（toilet seat → toilet）
        4. 过滤出 MSCOCO 物体
        5. 转换为代表词
    """
    # 转换为 set 加速查找
    valid_nouns_set: set[str] = set(valid_nouns)

    def get_repr(noun: str) -> str:
        """将名词转换为其代表词（同义词组的第一个词）"""
        return inv_synonym_map.get(noun, noun) if inv_synonym_map else noun

    def extract(disc: str) -> list[str]:
        """从单个句子中提取物体"""
        # Step 1: 分词并词形还原
        # "dogs" → "dog", "running" → "run"
        all_words: list[str] = [wn.lemma(w) for w in wn.word_tokenize(disc.lower())]

        # Step 2: 合并双词物体
        # ["hot", "dog"] → ["hot dog"]
        i = 0
        double_words, idxs = [], []
        while i < len(all_words):
            idxs.append(i)
            double_word = " ".join(all_words[i : i + 2])  # 两个词组成的双词短语
            if double_word in double_word_dict:
                # 找到双词物体，合并
                double_words.append(double_word_dict[double_word])
                i += 2
            else:
                # 不是双词物体，保留原词
                double_words.append(all_words[i])
                i += 1
        all_words = double_words

        # Step 3: 特殊处理 - "toilet seat" 只保留 "toilet"
        if ("toilet" in all_words) & ("seat" in all_words):
            all_words = [word for word in all_words if word != "seat"]

        # Step 4: 过滤出 MSCOCO 物体
        idxs = [idxs[idx] for idx, word in enumerate(all_words) if word in valid_nouns_set]
        all_objects: list[str] = [word for word in all_words if word in valid_nouns_set]
        
        # Step 5: 转换为代表词
        # "puppy" → "dog", "sedan" → "car"
        node_words: list[str] = [get_repr(obj) for obj in all_objects]

        return node_words if return_repr else all_objects

    # 统一输入格式
    if not isinstance(discriptions, list):
        discriptions = [discriptions]

    # 对每个句子提取物体
    objects: list[list[str]] = [extract(disc) for disc in discriptions]

    # 根据参数决定返回格式
    return objects if force_list or len(objects) > 1 else objects[0]


def extract_obj_from_textgraphs(
    textgraphs: list[list[str]] | list[list[list[str]]],
    spacy: SpacyModel,
    wn: WordnetModel,
    valid_nouns: list[str] | None = None,
    inverse_synonym_map: dict | None = None,
    force_list: bool = False,
) -> list[list[str]]:
    """
    从场景图（Text Graph）中提取物体（基于场景图解析的方法）
    
    场景图是将句子解析为 (主语, 谓语, 宾语) 三元组的结构，
    本函数从这些三元组中提取出物体名词。
    
    Args:
        textgraphs: 场景图结构
            - 单个场景图: [[subj, pred, obj], [subj, pred, obj], ...]
            - 多个场景图: [[[subj, pred, obj], ...], [[subj, pred, obj], ...], ...]
        spacy: Spacy 模型实例，用于词性判断
        wn: WordNet 模型实例，用于具体名词判断
        valid_nouns: 有效名词列表（可选，如果提供则只提取列表中的物体）
        inverse_synonym_map: 逆同义词映射（可选）
        force_list: 是否强制返回列表格式
    
    Returns:
        list[list[str]]: 每个场景图提取出的物体列表
    
    Example:
        >>> # 场景图: "A dog is running" → [["dog", "is", "running"]]
        >>> extract_obj_from_textgraphs([[["dog", "is", "running"]]], spacy, wn)
        [["dog"]]
    """
    # Step 1: 空值检查
    if not textgraphs:
        return [[]]
    
    # Step 2: 统一格式（确保是多个场景图的列表）
    if not isinstance(textgraphs[0][0], list):
        textgraphs = [textgraphs]

    # Step 3: 遍历每个场景图，提取物体
    objects_list: list[list[str]] = []
    for textgraph in textgraphs:
        objects: list[str] = []
        for triple in textgraph:
            # 处理每个三元组，提取名词
            process_triple(triple, objects, [], [], spacy, wn, valid_nouns, inverse_synonym_map)
        objects_list.append(objects)

    # Step 4: 根据参数决定返回格式
    if force_list or len(objects_list) > 1:
        return objects_list
    else:
        return objects_list[0]


def process_triple(
    triple: list[str],
    nouns: list[str],
    attributes: list[str],
    relations: list[str],
    spacy: SpacyModel,
    wn: WordnetModel,
    valid_nouns: list[str] | None = None,
    inverse_synonym_map: dict[str, str] | None = None,
) -> None:
    """
    处理场景图三元组，提取名词、属性和关系
    
    根据谓语类型决定如何处理主语和宾语：
    - has/have: 主语和宾语都是物体
    - is/are: 主语是物体，可能是属性描述
    - 其他谓语: 主语和宾语都是物体，记录关系
    
    Args:
        triple: 三元组 [主语, 谓语, 宾语]，如 ["dog", "is", "running"]
        nouns: 名词列表（会被原地修改）
        attributes: 属性列表（会被原地修改）
        relations: 关系列表（会被原地修改）
        spacy: Spacy 模型实例
        wn: WordNet 模型实例
        valid_nouns: 有效名词列表（可选）
        inverse_synonym_map: 逆同义词映射（可选）
    
    Returns:
        None（通过修改传入的列表返回结果）
    
    Example:
        >>> nouns = []
        >>> process_triple(["dog", "has", "tail"], nouns, [], [], spacy, wn)
        >>> nouns
        ["dog", "tail"]
    """
    # Step 1: 检查三元组格式
    if len(triple) != 3:
        return
    subject, predicate, obj = triple

    def add_noun(word: str) -> None:
        """辅助函数：尝试将词添加到名词列表"""
        add_to_nouns_if_valid(word, nouns, spacy, wn, valid_nouns, inverse_synonym_map)

    # Step 2: 根据谓语类型处理
    if predicate in ["has", "have"]:
        # "has/have" 表示所属关系，主语和宾语都是物体
        # 例如: "dog has tail" → 提取 dog, tail
        add_noun(subject)
        add_noun(obj)
    elif predicate in ["is", "are"]:
        # "is/are" 可能是属性描述
        # 例如: "dog is brown" → 提取 dog，记录属性
        add_noun(subject)
        if spacy.is_noun(subject) or spacy.is_noun(obj):
            attributes.append([subject, predicate, obj])
    else:
        # 其他谓语表示关系
        # 例如: "dog chases cat" → 提取 dog, cat，记录关系
        add_noun(subject)
        add_noun(obj)
        relations.append([subject, predicate, obj])


def add_to_nouns_if_valid(
    word: str,
    nouns: list[str],
    spacy: SpacyModel,
    wn: WordnetModel,
    valid_nouns: list[str] | None = None,
    inverse_synonym_map: dict[str, str] | None = None,
) -> bool:
    """
    验证单词是否为有效名词，如果是则添加到名词列表
    
    验证逻辑：
    1. 如果提供了 valid_nouns 列表，检查单词是否在列表中
    2. 否则，使用 spaCy 和 WordNet 判断是否为具体名词
    
    Args:
        word: 要检查的单词
        nouns: 名词列表（会被原地修改）
        spacy: Spacy 模型实例，用于词性判断
        wn: WordNet 模型实例，用于具体名词判断
        valid_nouns: 有效名词列表（可选）
        inverse_synonym_map: 逆同义词映射（可选）
    
    Returns:
        bool: 如果单词被添加到列表则返回 True，否则返回 False
    
    Example:
        >>> nouns = []
        >>> add_to_nouns_if_valid("dog", nouns, spacy, wn, valid_nouns)
        True
        >>> nouns
        ["dog"]
    """
    # Step 1: 转小写
    word = word.lower()
    
    # Step 2: 检查是否已存在（避免重复）
    if word in nouns:
        return False

    # Step 3: 根据是否提供了有效名词列表选择验证方式
    if valid_nouns:
        # 方式 A: 检查是否在预定义列表中
        if object_in_set(word, valid_nouns, spacy, wn, inverse_synonym_map):
            nouns.append(word)
            return True
        else:
            return False
    else:
        # 方式 B: 使用 NLP 工具判断是否为具体名词
        doc: Doc = spacy(word)
        if len(doc) == 1:
            # 单词情况
            token: Token = doc[0]
            lemma: str = token.lemma_
            # 检查：是名词 + 不重复 + 是具体名词（非抽象名词）
            if spacy.is_noun(token) and lemma not in nouns and wn.is_concrete_noun(lemma):
                nouns.append(lemma)
                return True
        else:
            # 多词情况（如 "hot dog"）
            for token in doc:
                lemma = token.lemma_
                if spacy.is_noun(token) and lemma not in nouns and wn.is_concrete_noun(lemma):
                    nouns.append(word)
                    return True
        return False


# ==================== 物体匹配工具 ====================


def object_in_set(
    obj: str,
    target_set: list[str] | set[str],
    spacy: SpacyModel,
    wn: WordnetModel,
    inv_synonym_map: dict[str, str] | None = None,
    allow_synonym: bool = False,
) -> bool:
    """
    检查物体是否在目标集合中（支持同义词匹配）
    
    这是物体匹配的核心函数，处理同义词和词形变化：
    - "puppy" 可以匹配包含 "dog" 的集合
    - "dogs" 可以匹配包含 "dog" 的集合
    
    Args:
        obj: 要检查的物体名称
        target_set: 目标集合（可以是 list 或 set）
        spacy: Spacy 模型实例，用于词形还原
        wn: WordNet 模型实例，用于同义词查找
        inv_synonym_map: 逆同义词映射（将同义词映射到代表词）
            例如: {"puppy": "dog", "sedan": "car"}
        allow_synonym: 是否允许使用 WordNet 同义词扩展匹配
    
    Returns:
        bool: 如果物体在集合中返回 True，否则返回 False
    
    匹配流程:
        1. 将 obj 转换为代表词
        2. 将 target_set 中的词都转换为代表词
        3. 比较代表词是否匹配
        4. 尝试词形还原后再比较
        5. (可选) 使用 WordNet 同义词扩展匹配
    
    Example:
        >>> object_in_set("puppy", ["dog", "cat"], spacy, wn, inv_syn_map)
        True  # puppy → dog (通过 inv_synonym_map)
    """
    def get_repr(noun: str) -> str:
        """将名词转换为其代表词（同义词组的第一个词）"""
        return inv_synonym_map.get(noun, noun) if inv_synonym_map else noun

    # Step 1: 转换为代表词
    repr_obj: str = get_repr(obj)
    repr_target_set: list[str] = [get_repr(obj) for obj in target_set]
    
    # Step 2: 直接比较或词形还原后比较
    if repr_obj in repr_target_set or (spacy is not None and get_repr(spacy.lemma(obj)) in repr_target_set):
        return True

    # Step 3: (可选) 使用 WordNet 同义词扩展匹配
    if allow_synonym and wn is not None:
        for synonym in wn.get_synset_list(repr_obj):
            if synonym in target_set:
                return True

    return False


def objects_in_set(
    object_list: list[str] | str,
    target_set: list[str] | set[str],
    spacy: SpacyModel,
    wn: WordnetModel,
    inv_synonym_map: dict[str, str] | None = None,
    check_type: str = "all",
) -> bool:
    """
    批量检查多个物体是否在目标集合中
    
    Args:
        object_list: 要检查的物体列表（或单个物体）
        target_set: 目标集合
        spacy: Spacy 模型实例
        wn: WordNet 模型实例
        inv_synonym_map: 逆同义词映射
        check_type: 检查类型
            - "all": 所有物体都必须在集合中才返回 True
            - "any": 任一物体在集合中就返回 True
    
    Returns:
        bool: 根据 check_type 返回检查结果
    
    Example:
        >>> objects_in_set(["dog", "cat"], ["dog"], spacy, wn, None, "all")
        False  # cat 不在集合中
        >>> objects_in_set(["dog", "cat"], ["dog"], spacy, wn, None, "any")
        True   # dog 在集合中
    """
    # Step 1: 统一输入格式
    if not isinstance(object_list, list):
        object_list = [object_list]

    # Step 2: 根据 check_type 执行检查
    if check_type.lower() == "all":
        # 所有物体都必须匹配
        return all(object_in_set(obj, target_set, spacy, wn, inv_synonym_map, False) for obj in object_list)
    elif check_type.lower() == "any":
        # 任一物体匹配即可
        return any(object_in_set(obj, target_set, spacy, wn, inv_synonym_map, False) for obj in object_list)
    else:
        raise ValueError(f"Invalid check type: {check_type}")


# ==================== DINO 检测工具 ====================


def pack_objs_for_dino(objects: list[list[str]] | list[str]) -> list[str] | str:
    """
    将物体列表打包为 DINO 检测器的输入格式
    
    DINO 检测器需要的输入格式是用点号分隔的字符串，如 "cat. dog. person."
    本函数将物体列表转换为这种格式。
    
    Args:
        objects: 物体列表
            - list[str]: 单个列表，如 ["cat", "dog"]
            - list[list[str]]: 嵌套列表，如 [["cat"], ["dog", "person"]]
            - str: 直接返回
    
    Returns:
        - str: 单个列表 → "cat.dog."
        - list[str]: 嵌套列表 → ["cat.", "dog.person."]
        - "": 空输入
    
    Example:
        >>> pack_objs_for_dino(["cat", "dog"])
        "cat.dog."
        >>> pack_objs_for_dino([["cat"], ["dog", "person"]])
        ["cat.", "dog.person."]
    """
    # Step 1: 空值或无效输入检查
    if (not isinstance(objects, list) and not isinstance(objects, str)) or not objects:
        return ""
    
    # Step 2: 如果已经是字符串，直接返回
    elif isinstance(objects, str):
        return objects
    
    # Step 3: 嵌套列表，递归处理
    elif isinstance(objects[0], list):
        return [pack_objs_for_dino(obj) for obj in objects]
    
    # Step 4: 普通列表，用点号连接（去重）
    else:
        return ".".join(set(objects)) + "." if objects else ""


def unpack_objs_from_dino(objects: list[str] | str) -> list[list[str]] | list[str]:
    """
    将 DINO 格式的字符串解包为物体列表
    
    与 pack_objs_for_dino 相反的操作。
    
    Args:
        objects: DINO 格式的输入
            - str: "cat.dog." → ["cat", "dog"]
            - list[str]: 递归处理每个元素
    
    Returns:
        物体列表
    
    Example:
        >>> unpack_objs_from_dino("cat.dog.")
        ["cat", "dog"]
    """
    if isinstance(objects, list):
        # 递归处理列表
        return [unpack_objs_from_dino(obj) for obj in objects]
    else:
        # 字符串：去掉末尾点号，按点号分割
        return objects.rstrip(".").split(".") if objects else []


def get_dino_detected_objects(obj_to_detects: str, dino_result_labels: list[str]) -> set[str]:
    """
    从 DINO 检测结果中提取被检测到的物体
    
    比较待检测物体列表和 DINO 返回的标签，找出被检测到的物体。
    
    Args:
        obj_to_detects: 待检测物体字符串，格式 "cat.dog.person."
        dino_result_labels: DINO 返回的检测标签列表，如 ["cat", "dog"]
    
    Returns:
        set[str]: 被检测到的物体集合
    
    Example:
        >>> get_dino_detected_objects("cat.dog.person.", ["cat", "dog"])
        {"cat", "dog"}  # person 没有被检测到
    
    Note:
        使用子字符串匹配（obj in label），因为 DINO 标签可能包含额外信息
    """
    detected_obj: set[str] = set()

    # 遍历待检测物体
    for obj in filter(None, obj_to_detects.split(".")):
        # 检查是否在任一标签中出现
        if any(obj in label for label in dino_result_labels):
            detected_obj.add(obj)

    return detected_obj


# ==================== 幻觉检测工具 ====================


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
    单样本幻觉检测 - 使用 YOLO + DINO 交叉验证判断物体是否为幻觉
    
    这是幻觉检测的核心函数，通过目标检测器验证文本中提到的物体是否真实存在。
    
    判断逻辑:
        - YOLO ✅ + DINO ✅ → 非幻觉（真实物体）
        - YOLO ❌ + DINO ❌ → 幻觉（虚假物体）
        - YOLO ✅ + DINO ❌ 或 YOLO ❌ + DINO ✅ → 不确定
    
    Args:
        objects_list: 待检测物体列表，每个元素是一个句子中的物体列表
            例如: [["dog", "cat"], ["person", "car"]]
        nonhallu_objects: 已确认的非幻觉物体列表（会被原地修改）
        hallu_objects: 已确认的幻觉物体列表（会被原地修改）
        spacy: Spacy 模型实例，用于词形还原
        wn: WordNet 模型实例
        image: 图像对象（用于 DINO 检测）
        dino: DINO 检测器实例
        yolo_results: YOLO 检测到的物体标签列表
        yolo_labels: YOLO 模型支持的所有标签（80 类）
        uncertain_objects: 不确定物体列表（会被原地修改）
        detector_reject: 记录哪个检测器拒绝了哪些物体
        inv_syn_map: 逆同义词映射
    
    Returns:
        tuple:
            - hallu_objects_list: 每个句子中的幻觉物体列表
            - nonhallu_objects_list: 每个句子中的非幻觉物体列表
    
    Example:
        >>> hallu_list, nonhallu_list = get_hallu_objects(
        ...     [["dog", "unicorn"]], nonhallu, hallu, spacy, wn, image, dino, ...
        ... )
        >>> hallu_list
        [["unicorn"]]  # unicorn 没有被检测到
        >>> nonhallu_list
        [["dog"]]      # dog 被检测到了
    """

    def get_repr(noun: str) -> str:
        """将名词转换为其代表词"""
        return inv_syn_map.get(noun, noun) if inv_syn_map else noun

    def get_cached_objects() -> list[str]:
        """获取所有已缓存（已判定）的物体"""
        cached_objs: list[str] = []
        if nonhallu_objects:
            cached_objs.extend(nonhallu_objects)
        if hallu_objects:
            cached_objs.extend(hallu_objects)
        if uncertain_objects:
            cached_objs.extend(uncertain_objects)
        return cached_objs

    def get_set() -> set[str]:
        """获取"非幻觉"集合（包括不确定的，用于后续过滤）"""
        _set = set()
        if nonhallu_objects:
            _set.update(nonhallu_objects)
        if uncertain_objects:
            _set.update(uncertain_objects)
        return _set

    def recognize_by_yolo(obj: str) -> bool:
        """
        检查物体是否被 YOLO 认可
        
        规则：如果物体不在 YOLO 的检测范围内（不是 80 类之一），视为认可
        """
        if yolo_results is None or yolo_labels is None:
            return True
        repr_obj: str = spacy.lemma(get_repr(obj))
        if repr_obj in yolo_labels:
            # 在 YOLO 检测范围内，检查是否被检测到
            return repr_obj in yolo_results
        else:
            # 不在 YOLO 检测范围内，视为认可
            return True

    def recognize_by_dino(obj: str, detected_obj: list[str] | set[str]) -> bool:
        """检查物体是否被 DINO 认可"""
        return object_in_set(obj, detected_obj, spacy, wn, inv_syn_map, False)

    def get_uncached_objects(objects_list: list[list[str]], cached_objs: list[str]) -> list[str]:
        """获取未缓存的物体（需要检测的新物体），去重"""
        return list(
            set(
                [
                    spacy.lemma(obj)
                    for objects in objects_list
                    for obj in objects
                    if not object_in_set(obj, cached_objs, spacy, wn, inv_syn_map, False)
                ]
            )
        )

    # Step 1: 获取已缓存和未缓存的物体
    cached_objs: list[str] = get_cached_objects()
    uncached_objs: list[str] = get_uncached_objects(objects_list, cached_objs)

    # Step 2: 处理未缓存的物体（需要检测）
    if uncached_objs:
        if image is not None and dino is not None:
            # 使用 YOLO + DINO 双重检测
            # Step 2a: DINO 检测
            obj_for_dino_to_detect: str = pack_objs_for_dino(uncached_objs)
            dino_results: dict[str] = dino.detect(image, obj_for_dino_to_detect, force_list=False)
            detected_obj: set[str] = get_dino_detected_objects(obj_for_dino_to_detect, dino_results["labels"])

            # Step 2b: 对每个物体进行判定
            for obj in uncached_objs:
                yolo_recognized, dino_recognized = recognize_by_yolo(obj), recognize_by_dino(obj, detected_obj)
                
                if dino_recognized and yolo_recognized and obj not in nonhallu_objects:
                    # 两个都认可 → 非幻觉
                    nonhallu_objects.append(obj)
                elif not dino_recognized and not yolo_recognized and obj not in hallu_objects:
                    # 两个都不认可 → 幻觉
                    hallu_objects.append(obj)
                elif uncertain_objects is not None and obj not in uncertain_objects:
                    # 一个认可一个不认可 → 不确定
                    if not yolo_recognized and detector_reject is not None and "yolo" in detector_reject:
                        detector_reject["yolo"].append(obj)
                    if not dino_recognized and detector_reject is not None and "dino" in detector_reject:
                        detector_reject["dino"].append(obj)
                    uncertain_objects.append(obj)
        else:
            # 只使用 YOLO 检测
            for obj in uncached_objs:
                yolo_recognized: bool = recognize_by_yolo(obj)
                if yolo_recognized and obj not in nonhallu_objects:
                    nonhallu_objects.append(obj)
                elif not yolo_recognized and obj not in hallu_objects:
                    hallu_objects.append(obj)
    
    del cached_objs, uncached_objs

    # Step 3: 根据判定结果，为每个句子分类物体
    hallu_objects_list: list[list[str]] = []
    nonhallu_objects_list: list[list[str]] = []
    _set = get_set()  # 非幻觉 + 不确定的集合
    
    for objects in objects_list:
        # 幻觉物体：在幻觉列表中，或者不在非幻觉集合中
        hallu_objects_list.append(
            [
                obj
                for obj in objects
                if spacy.lemma(obj) in hallu_objects or not object_in_set(obj, _set, spacy, wn, inv_syn_map)
            ]
        )
        # 非幻觉物体：在非幻觉列表中
        nonhallu_objects_list.append(
            [obj for obj in objects if object_in_set(obj, nonhallu_objects, spacy, wn, inv_syn_map)]
        )
    
    return hallu_objects_list, nonhallu_objects_list


def b_get_hallu_objects(
    b_object_lists: list[list[list[str]]],
    b_nonhallu_objects: list[list[str]],
    b_hallu_objects: list[list[str]],
    spacy: SpacyModel,
    wn: WordnetModel,
    images: list[Image.Image],
    dino: DINO,
    b_yolo_results: list[list[str]],
    yolo_labels: list[str],
    b_uncertain_objects: list[list[str]],
    b_detector_rejects: list[dict[str, list[str]]],
    inv_syn_map: dict[str, str],
) -> tuple[list[list[list[str]]], list[list[list[str]]]]:
    """
    批量幻觉检测 - 对多个样本同时进行 YOLO + DINO 交叉验证
    
    这是 SENTINEL 的核心函数，批量处理多个图像的幻觉检测。
    通过批处理 DINO 检测提高效率。
    
    Args:
        b_object_lists: 批量物体列表
            形状: [batch_size][num_sentences][num_objects]
            例如: [[["dog", "cat"], ["person"]], [["car"], ["bike", "tree"]]]
        b_nonhallu_objects: 每个样本已确认的非幻觉物体列表（会被原地修改）
        b_hallu_objects: 每个样本已确认的幻觉物体列表（会被原地修改）
        spacy: Spacy 模型实例
        wn: WordNet 模型实例
        images: 图像列表，长度 = batch_size
        dino: DINO 检测器实例
        b_yolo_results: 每个样本的 YOLO 检测结果
        yolo_labels: YOLO 模型支持的所有标签（80 类）
        b_uncertain_objects: 每个样本的不确定物体列表（会被原地修改）
        b_detector_rejects: 每个样本的检测器拒绝记录
        inv_syn_map: 逆同义词映射
    
    Returns:
        tuple:
            - b_hallu_objects_list: 每个样本每个句子的幻觉物体
                形状: [batch_size][num_sentences][num_hallu_objects]
            - b_nonhallu_objects_list: 每个样本每个句子的非幻觉物体
                形状: [batch_size][num_sentences][num_nonhallu_objects]
    
    处理流程:
        1. 收集所有未缓存的物体
        2. 批量调用 DINO 检测（提高效率）
        3. 使用 YOLO + DINO 交叉验证判定每个物体
        4. 更新各物体列表
        5. 为每个句子分类物体
    
    Example:
        >>> b_hallu, b_nonhallu = b_get_hallu_objects(
        ...     b_object_lists, b_nonhallu, b_hallu, spacy, wn,
        ...     images, dino, b_yolo_results, yolo_labels, ...
        ... )
    """
    # 获取批处理大小
    b_size: int = len(b_object_lists)

    # ===== 辅助函数定义 =====
    
    def get_repr(noun: str) -> str:
        """将名词转换为其代表词（同义词组的第一个词）"""
        return inv_syn_map.get(noun, noun) if inv_syn_map else noun

    def get_cached_objects(idx: int) -> list[str]:
        """
        获取第 idx 个样本的所有已缓存物体
        包括：非幻觉 + 幻觉 + 不确定
        """
        cached_objs: list[str] = []
        if b_nonhallu_objects:
            cached_objs.extend(b_nonhallu_objects[idx])
        if b_hallu_objects:
            cached_objs.extend(b_hallu_objects[idx])
        if b_uncertain_objects:
            cached_objs.extend(b_uncertain_objects[idx])
        return cached_objs

    def get_set(idx: int) -> set[str]:
        """
        获取第 idx 个样本的"非幻觉"集合
        包括：非幻觉 + 不确定（保守策略，不确定的不算幻觉）
        """
        _set = set()
        if b_nonhallu_objects:
            _set.update(b_nonhallu_objects[idx])
        if b_uncertain_objects:
            _set.update(b_uncertain_objects[idx])
        return _set

    def recognize_by_yolo(obj: str, idx: int) -> bool:
        """
        检查物体是否被 YOLO 认可
        
        规则：如果物体不在 YOLO 的 80 类检测范围内，视为认可
        这是保守策略，避免将 YOLO 无法检测的物体错误标记为幻觉
        """
        if b_yolo_results is None or yolo_labels is None:
            return True
        repr_obj: str = spacy.lemma(get_repr(obj))
        if repr_obj in yolo_labels:
            # 在 YOLO 范围内，检查是否被检测到
            return repr_obj in b_yolo_results[idx]
        else:
            # 不在 YOLO 范围内，视为认可
            return True

    def recognize_by_dino(obj: str, detected_obj: list[str] | set[str]) -> bool:
        """检查物体是否被 DINO 认可"""
        return object_in_set(obj, detected_obj, spacy, wn, inv_syn_map, False)

    def get_uncached_objects(objects_list: list[list[str]], cached_objs: list[str]) -> list[str]:
        """获取未缓存的物体（需要检测的新物体），去重并词形还原"""
        return list(
            set(
                [
                    spacy.lemma(obj)
                    for objects in objects_list
                    for obj in objects
                    if not object_in_set(obj, cached_objs, spacy, wn, inv_syn_map, False)
                ]
            )
        )

    # ===== Step 1: 收集已缓存和未缓存的物体 =====
    b_cached_objs: list[list[str]] = [get_cached_objects(i) for i in range(b_size)]
    b_uncached_objs: list[list[str]] = [
        get_uncached_objects(b_object_lists[i], b_cached_objs[i]) for i in range(b_size)
    ]
    
    # ===== Step 2: 批量 DINO 检测并判定 =====
    if b_uncached_objs and any(b_uncached_objs):
        # Step 2a: 打包所有未缓存物体并批量检测
        b_obj_for_dino: list[str] = pack_objs_for_dino(b_uncached_objs)
        dino_results: list[dict[str]] = dino.detect(images, b_obj_for_dino, force_list=True)
        
        # Step 2b: 提取 DINO 检测到的物体
        b_detected_obj: list[set[str]] = [
            get_dino_detected_objects(obj, dino_results[i]["labels"]) for i, obj in enumerate(b_obj_for_dino)
        ]

        # Step 2c: 对每个样本的每个未缓存物体进行判定
        for idx, uncached_objs in enumerate(b_uncached_objs):
            for obj in uncached_objs:
                if b_yolo_results is not None:
                    # === YOLO + DINO 双重验证模式 ===
                    yolo_recognized = recognize_by_yolo(obj, idx)
                    dino_recognized = recognize_by_dino(obj, b_detected_obj[idx])

                    if dino_recognized and yolo_recognized:
                        # 两个都认可 → 非幻觉
                        if obj not in b_nonhallu_objects[idx]:
                            b_nonhallu_objects[idx].append(obj)
                    elif not dino_recognized and not yolo_recognized:
                        # 两个都不认可 → 幻觉
                        if obj not in b_hallu_objects[idx]:
                            b_hallu_objects[idx].append(obj)
                    elif b_uncertain_objects[idx] is not None and obj not in b_uncertain_objects[idx]:
                        # 一个认可一个不认可 → 不确定
                        b_uncertain_objects[idx].append(obj)

                        # 记录哪个检测器拒绝了
                        if not yolo_recognized and b_detector_rejects is not None and "yolo" in b_detector_rejects[idx]:
                            b_detector_rejects[idx]["yolo"].append(obj)
                        if not dino_recognized and b_detector_rejects is not None and "dino" in b_detector_rejects[idx]:
                            b_detector_rejects[idx]["dino"].append(obj)
                else:
                    # === 仅 DINO 模式 ===
                    dino_recognized = recognize_by_dino(obj, b_detected_obj[idx])
                    if dino_recognized and obj not in b_nonhallu_objects[idx]:
                        b_nonhallu_objects[idx].append(obj)
                    elif not dino_recognized and obj not in b_hallu_objects[idx]:
                        b_hallu_objects[idx].append(obj)

    # 释放临时变量
    del b_cached_objs, b_uncached_objs

    # ===== Step 3: 为每个句子分类物体 =====
    b_hallu_objects_list: list[list[list[str]]] = []
    b_nonhallu_objects_list: list[list[list[str]]] = []
    
    for idx, objects_list in enumerate(b_object_lists):
        _set = get_set(idx)  # 非幻觉 + 不确定的集合

        _hallu_objects_list, _nonhallu_objects_list = [], []
        for objects in objects_list:
            # 幻觉物体：在幻觉列表中，或者不在非幻觉集合中
            _hallu_objects_list.append(
                [
                    obj
                    for obj in objects
                    if spacy.lemma(obj) in b_hallu_objects[idx] or not object_in_set(obj, _set, spacy, wn, inv_syn_map)
                ]
            )
            # 非幻觉物体：在非幻觉列表中
            _nonhallu_objects_list.append(
                [obj for obj in objects if object_in_set(obj, b_nonhallu_objects[idx], spacy, wn, inv_syn_map)]
            )
        b_hallu_objects_list.append(_hallu_objects_list)
        b_nonhallu_objects_list.append(_nonhallu_objects_list)

    return b_hallu_objects_list, b_nonhallu_objects_list


# ==================== 文本处理工具 ====================


def tokenize_sent(text: str) -> list[str]:
    """
    将文本分割成句子列表
    
    使用 NLTK 的 sent_tokenize 进行句子分割。
    
    Args:
        text: 输入文本
    
    Returns:
        list[str]: 句子列表
    
    Example:
        >>> tokenize_sent("Hello world. How are you?")
        ["Hello world.", "How are you?"]
    """
    from nltk.tokenize import sent_tokenize

    return sent_tokenize(text)


def get_finish_flag(
    sentences: list[str],
    stop_threshold: float = 0.5,
    remove_duplicates: bool = False,
) -> tuple[list[str], bool]:
    """
    判断生成是否应该停止
    
    当空句子比例超过阈值时，停止生成。用于控制迭代生成的终止条件。
    
    Args:
        sentences: 新生成的句子列表（可能包含空字符串）
        stop_threshold: 停止阈值，空句子比例超过此值时停止
            默认 0.5 表示超过 50% 是空句子就停止
        remove_duplicates: 是否去重
    
    Returns:
        tuple:
            - valid_sentences: 非空句子列表
            - should_stop: 是否应该停止生成
    
    Example:
        >>> get_finish_flag(["Hello.", "", "World.", ""], 0.5)
        (["Hello.", "World."], False)  # 50% 空句子，不超过阈值
        >>> get_finish_flag(["Hello.", "", "", ""], 0.5)
        (["Hello."], True)  # 75% 空句子，超过阈值
    """
    # Step 1: 过滤出非空句子
    valid_sentences: list[str] = [s for s in sentences if s]
    
    # Step 2: 可选去重
    if remove_duplicates:
        valid_sentences = list(set(valid_sentences))
    
    # Step 3: 计算空句子比例，判断是否停止
    should_stop = (len(sentences) - len(valid_sentences)) / len(sentences) > stop_threshold
    
    return valid_sentences, should_stop


def concat_sents(description: str, previous: str, retrospect_num: int = 1) -> str:
    """
    将当前句子与前文拼接，形成带上下文的句子
    
    用于指代消解：将 "It is cute" 变成 "A dog is running. It is cute"，
    这样 "It" 可以被解析为 "dog"。
    
    Args:
        description: 当前句子
        previous: 前文（可能包含多个句子）
        retrospect_num: 回溯句子数量（从前文末尾取几个句子）
    
    Returns:
        str: 拼接后的句子
    
    Example:
        >>> concat_sents("It is cute.", "A dog is running. It barks.", 1)
        "It barks. It is cute."
        >>> concat_sents("It is cute.", "A dog is running. It barks.", 2)
        "A dog is running. It barks. It is cute."
    """
    # Step 1: 空前文或不需要回溯，直接返回
    if not previous or retrospect_num <= 0:
        return description

    # Step 2: 分割前文为句子
    sentences: list[str] = tokenize_sent(previous)
    
    # Step 3: 取最后 retrospect_num 个句子作为上下文
    context: str = " ".join(sentences[-retrospect_num:]) if len(sentences) >= retrospect_num else " ".join(sentences)
    
    # Step 4: 拼接上下文和当前句子
    return " ".join([context, description])


def resolve_corefs(
    spacy: SpacyModel,
    descriptions: list[str] | list[list[str]],
    previous: list[str],
    retro_num: int,
    force_list: bool = True,
) -> list[str] | list[list[str]]:
    """
    批量指代消解 - 将代词替换为其指代的实体
    
    例如："A dog is running. It is cute." → "A dog is running. The dog is cute."
    
    Args:
        spacy: Spacy 模型实例（需要加载 fastcoref）
        descriptions: 待处理的句子
            - list[str]: 单层列表
            - list[list[str]]: 嵌套列表（批处理多个样本）
        previous: 每个句子对应的前文
        retro_num: 回溯句子数量
        force_list: 强制返回列表格式
    
    Returns:
        指代消解后的句子，保持输入格式
    
    Example:
        >>> resolve_corefs(
        ...     spacy,
        ...     ["It is cute."],
        ...     ["A dog is running."],
        ...     retro_num=1
        ... )
        ["The dog is cute."]
    
    处理流程:
        1. 展平嵌套列表（如果有）
        2. 拼接上下文
        3. 使用 fastcoref 进行指代消解
        4. 提取最后一句（即原始待处理句子）
        5. 恢复原始格式
    """
    # Step 1: 处理嵌套列表（展平）
    if isinstance(descriptions[0], list):
        # 嵌套列表：展平并扩展 previous
        flat_descriptions: list[str] = [sent for sublist in descriptions for sent in sublist]
        expanded_previous: list[str] = [previous[idx] for idx, des in enumerate(descriptions) for _ in des]
    else:
        flat_descriptions = descriptions
        expanded_previous = previous

    # Step 2: 拼接上下文并指代消解
    if retro_num > 0 and any(pre for pre in previous):
        # 需要指代消解
        concated_sents: list[str] = [
            concat_sents(d, p, retro_num) for d, p in zip(flat_descriptions, expanded_previous)
        ]
        resolved_texts: list[str] = spacy.resolve_coref(concated_sents, force_list=force_list)
    else:
        # 不需要指代消解
        resolved_texts: list[str] = flat_descriptions

    # Step 3: 提取最后一句（原始待处理句子的消解结果）
    resolved_last_sentences: list[str] = [t.split(". ")[-1] for t in resolved_texts]

    # Step 4: 恢复原始格式
    if isinstance(descriptions[0], list):
        # 还原为嵌套列表
        resolved_texts_nested = []
        idx = 0
        for sublist in descriptions:
            resolved_texts_nested.append(resolved_last_sentences[idx : idx + len(sublist)])
            idx += len(sublist)
        return resolved_texts_nested
    else:
        return resolved_last_sentences


def pharse_w_context(
    spacy: SpacyModel,
    sg_parser: SGParser,
    descriptions: list[str] | str,
    previous: list[str] | str | None = None,
    retro_num: int = 2,
    force_list: bool = True,
) -> list[list[list[str]]]:
    """
    带上下文的句子解析 - 先指代消解，再解析为场景图
    
    组合了指代消解和场景图解析两个步骤。
    
    Args:
        spacy: Spacy 模型实例
        sg_parser: 场景图解析器实例
        descriptions: 待解析的句子
        previous: 前文（用于指代消解）
        retro_num: 回溯句子数量
        force_list: 强制返回列表格式
    
    Returns:
        list[list[list[str]]]: 场景图结构
            外层：每个句子
            中层：每个三元组
            内层：[主语, 谓语, 宾语]
    
    Example:
        >>> pharse_w_context(
        ...     spacy, sg_parser,
        ...     "The dog chases a cat.",
        ...     None
        ... )
        [[["dog", "chases", "cat"]]]
    
    处理流程:
        1. 统一输入格式
        2. 指代消解（将代词替换为实体）
        3. 场景图解析（提取三元组）
    """
    # Step 1: 统一输入格式
    descriptions, previous = ensure_lists(descriptions, previous)
    
    # Step 2: 指代消解
    resolved_texts: list[str] = resolve_corefs(spacy, descriptions, previous, retro_num, force_list)
    
    # Step 3: 场景图解析
    textgraphs: list[list[list[str]]] = sg_parser.pharse(resolved_texts, force_list=force_list)
    
    return textgraphs


# ==================== YOLO 检测工具 ====================


def yolo_detect(yolo: YoloModel | None, data_states: list[DataStateForBuildDataset]) -> None:
    """
    批量 YOLO 检测 - 对数据状态列表中的图像进行检测
    
    检测结果会保存在 DataState 对象中，供后续幻觉检测使用。
    
    Args:
        yolo: YOLO 模型实例，如果为 None 则不执行
        data_states: 数据状态列表，每个包含一个图像
    
    Returns:
        None（结果保存在 data_states 中）
    
    处理流程:
        1. 过滤出未检测的状态
        2. 批量检测
        3. 将结果保存到对应的 DataState 对象
    
    Note:
        只检测 yolo_detected=False 的状态，避免重复检测
    """
    # Step 1: 空值检查
    if not yolo:
        return

    # Step 2: 过滤出未检测的状态
    states_to_detect = [s for s in data_states if not s.yolo_detected]
    if not states_to_detect:
        return

    # Step 3: 提取图像并批量检测
    images_to_detect: list[Image.Image] = [s.image for s in states_to_detect]
    detection_results: list[YoloResult] = yolo.predict(images_to_detect, force_list=True)

    # Step 4: 保存检测结果
    for s, result in zip(states_to_detect, detection_results):
        s.yolo_result = result
        s.yolo_detected = True


# ==================== 辅助工具 ====================


def pop_first_sents(sents: list[str]) -> tuple[list[str], list[str]]:
    """
    从每个句子中提取第一句，返回第一句和剩余部分
    
    用于逐句处理长文本。
    
    Args:
        sents: 句子列表（每个可能包含多个句子）
    
    Returns:
        tuple:
            - first_sents: 每个元素的第一句（带句号）
            - remaining_sents: 每个元素的剩余部分
    
    Example:
        >>> pop_first_sents(["Hello. World.", "Foo. Bar. Baz."])
        (["Hello.", "Foo."], ["World.", "Bar. Baz."])
    """
    first_sents = []
    remaining_sents = []

    for sent in sents:
        # 按第一个句号分割
        split_sents = sent.split(".", 1)
        
        # 第一句（加回句号）
        first_sents.append(split_sents[0] + ".")
        
        # 剩余部分（去掉开头空格）
        if len(split_sents) > 1:
            remaining_sents.append(split_sents[1].lstrip(" "))
        else:
            remaining_sents.append("")

    return first_sents, remaining_sents
