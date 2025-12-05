"""
Mask-based DPO 数据生成模块
============================

本模块实现基于图像遮挡的偏好数据生成逻辑：
1. 读取参考数据集 {image_path, question, ground_truth}
2. 从 ground_truth 中提取物体
3. 使用 YOLO + DINO 检测物体，记录详细信息（名称、置信度、检测框）
4. 对每个检测到的物体进行遮挡，生成 masked_image
5. 将 masked_image 送入 VLM，得到 masked_response
6. 如果 masked_response 中仍然提到被遮挡的物体，记录为幻觉偏好对

核心思想：
    如果遮挡某个物体后，模型仍然声称看到该物体，说明是幻觉。
"""

import json
import os
from dataclasses import dataclass, field
from logging import Logger
from time import time

import numpy as np
import torch
from PIL import Image

# 检测器导入
from model.detector.grounding_dino import DINO
from model.detector.yolo_model import YoloModel

# NLP 工具导入
from model.others.spacy_model import SpacyModel
from model.others.wordnet import WordnetModel

# 生成器工具导入
from model.utils.gen_utils import GenOutput, get_generator

# 工具函数导入
from run.utils import (
    extract_obj_w_gt,
    open_images,
    pack_objs_for_dino,
    save_result,
)
from run.object_utils import get_object_n_represent, get_double_word_dict


# ==================== 配置常量 ====================

DEBUG = True
"""调试模式"""

MASK_COLOR = (128, 128, 128)  # 灰色遮挡
"""遮挡区域的颜色 (R, G, B)"""

MASK_EXPAND_RATIO = 0.05
"""遮挡框扩展比例，避免边缘泄露"""


# ==================== 数据结构 ====================

@dataclass
class DetectionInfo:
    """单个物体的检测信息"""
    object_name: str
    scores: list[float] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)  # [[x1, y1, x2, y2], ...]
    source: str = ""  # "yolo", "dino", or "both"


@dataclass
class DetectionResult:
    """完整的检测结果，包含每个物体的信息和原始检测结果"""
    detections: list[DetectionInfo] = field(default_factory=list)  # 每个物体的检测信息
    yolo_raw: dict = field(default_factory=dict)  # YOLO 原始结果 {label: [{xywhn, conf}, ...]}
    dino_raw: dict = field(default_factory=dict)  # DINO 原始结果 {labels, boxes, scores}


@dataclass
class MaskDataPoint:
    """遮挡数据点"""
    image_id: str
    image_path: str
    question: str
    masked_object: str
    masked_image_path: str
    original_response: str = ""
    masked_response: str = ""
    is_hallucination: bool = False


# ==================== 核心函数 ====================

def read_reference_dataset(dataset_path: str) -> list[dict]:
    """
    读取参考数据集
    
    Args:
        dataset_path: 数据集路径（jsonl 格式）
    
    Returns:
        数据列表，每个元素包含 {image_path, question, ground_truth}
    """
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data


def load_processed_image_ids(processed_path: str) -> set[str]:
    """
    加载已处理的 image_id 集合，用于断点续传
    
    Args:
        processed_path: 已处理记录文件路径（jsonl 格式）
    
    Returns:
        已处理的 image_id 集合
    """
    processed = set()
    if not os.path.exists(processed_path):
        return processed
    
    try:
        with open(processed_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    image_id = item.get("image_id", "")
                    if image_id:
                        processed.add(image_id)
    except Exception:
        pass
    
    return processed


def extract_objects_from_text(
    text: str,
    valid_nouns: list[str],
    double_word_dict: dict[str, str],
    inv_syn_map: dict[str, str],
    wn: WordnetModel,
) -> list[str]:
    """
    从文本中提取物体
    
    Args:
        text: 输入文本 (ground_truth)
        valid_nouns: 有效物体列表 (MSCOCO 80 类)
        double_word_dict: 双词物体映射
        inv_syn_map: 逆同义词映射
        wn: WordNet 模型
    
    Returns:
        提取的物体列表（去重）
    """
    objects = extract_obj_w_gt(
        text,
        valid_nouns,
        double_word_dict,
        inv_syn_map,
        wn,
        force_list=True,
        return_repr=True,
    )
    # 去重
    return list(set(objects[0])) if objects else []


def detect_objects_with_details(
    image: Image.Image,
    objects: list[str],
    yolo: YoloModel,
    dino: DINO,
    spacy: SpacyModel,
) -> DetectionResult:
    """
    使用 YOLO + DINO 检测物体，记录详细信息
    
    Args:
        image: 输入图像
        objects: 待检测的物体列表
        yolo: YOLO 检测器
        dino: DINO 检测器
        spacy: Spacy 模型（用于词形还原）
    
    Returns:
        DetectionResult: 包含每个物体的检测信息和原始检测结果
    """
    detection_infos: list[DetectionInfo] = []
    img_width, img_height = image.size
    
    # YOLO 检测
    yolo_result = yolo.predict(image, force_list=False)
    yolo_labels = set(yolo_result.labels)
    
    # DINO 检测
    dino_caption = pack_objs_for_dino(objects)
    dino_result = dino.detect(image, dino_caption, force_list=False) if dino_caption else {"labels": [], "boxes": torch.tensor([]), "scores": torch.tensor([])}
    
    for obj in objects:
        info = DetectionInfo(object_name=obj)
        obj_lemma = spacy.lemma(obj)
        
        # 从 YOLO 获取检测信息
        if obj_lemma in yolo_labels or obj in yolo_labels:
            label_to_use = obj_lemma if obj_lemma in yolo_labels else obj
            for det in yolo_result.result.get(label_to_use, []):
                # 转换 xywhn 到 xyxy
                x_center, y_center, w, h = det["xywhn"]
                x1 = (x_center - w / 2) * img_width
                y1 = (y_center - h / 2) * img_height
                x2 = (x_center + w / 2) * img_width
                y2 = (y_center + h / 2) * img_height
                info.boxes.append([x1, y1, x2, y2])
                info.scores.append(det["conf"])
            info.source = "yolo"
        
        # 从 DINO 获取检测信息
        if dino_result["labels"]:
            for i, label in enumerate(dino_result["labels"]):
                if obj.lower() in label.lower() or obj_lemma.lower() in label.lower():
                    box = dino_result["boxes"][i].tolist()
                    score = dino_result["scores"][i].item()
                    # 检查是否与 YOLO 框重叠度低（避免重复）
                    if not _is_box_duplicate(box, info.boxes):
                        info.boxes.append(box)
                        info.scores.append(score)
                    if info.source == "yolo":
                        info.source = "both"
                    elif not info.source:
                        info.source = "dino"
        
        # 只保留有检测结果的物体
        if info.boxes:
            detection_infos.append(info)
    
    # 构建原始结果（转换 tensor 为可序列化格式）
    dino_raw = {
        "labels": dino_result["labels"],
        "boxes": dino_result["boxes"].tolist() if hasattr(dino_result["boxes"], "tolist") else [],
        "scores": dino_result["scores"].tolist() if hasattr(dino_result["scores"], "tolist") else [],
    }
    
    return DetectionResult(
        detections=detection_infos,
        yolo_raw=yolo_result.result,  # {label: [{xywhn, conf}, ...]}
        dino_raw=dino_raw,
    )


def _is_box_duplicate(new_box: list[float], existing_boxes: list[list[float]], iou_threshold: float = 0.2) -> bool:
    """检查新框是否与现有框重叠"""
    for box in existing_boxes:
        iou = _calculate_iou(new_box, box)
        if iou > iou_threshold:
            return True
    return False


def _calculate_iou(box1: list[float], box2: list[float]) -> float:
    """计算两个框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


MASK_SCORE_THRESHOLD = 0.8
"""遮挡框的置信度阈值，低于此阈值的框不进行遮挡"""


def create_masked_image(
    image: Image.Image,
    boxes: list[list[float]],
    scores: list[float] | None = None,
    score_threshold: float = 0,
    expand_ratio: float = MASK_EXPAND_RATIO,
    mask_color: tuple = MASK_COLOR,
) -> Image.Image:
    """
    创建遮挡后的图像
    
    Args:
        image: 原始图像
        boxes: 检测框列表 [[x1, y1, x2, y2], ...]
        scores: 每个框的置信度列表，与 boxes 一一对应
        score_threshold: 置信度阈值，低于此阈值的框不进行遮挡
        expand_ratio: 框扩展比例
        mask_color: 遮挡颜色
    
    Returns:
        遮挡后的图像
    """
    masked_image = image.copy()
    img_array = np.array(masked_image)
    img_height, img_width = img_array.shape[:2]
    
    for i, box in enumerate(boxes):
        # 检查置信度是否超过阈值
        if scores is not None and i < len(scores):
            if scores[i] < score_threshold:
                continue  # 跳过低置信度的框
        
        x1, y1, x2, y2 = box
        
        # 扩展框
        w, h = x2 - x1, y2 - y1
        x1 = max(0, x1 - w * expand_ratio)
        y1 = max(0, y1 - h * expand_ratio)
        x2 = min(img_width, x2 + w * expand_ratio)
        y2 = min(img_height, y2 + h * expand_ratio)
        
        # 遮挡
        img_array[int(y1):int(y2), int(x1):int(x2)] = mask_color
    
    return Image.fromarray(img_array)


def save_masked_image(
    masked_image: Image.Image,
    original_path: str,
    masked_object: str,
    output_dir: str,
) -> str:
    """
    保存遮挡后的图像
    
    Args:
        masked_image: 遮挡后的图像
        original_path: 原始图像路径
        masked_object: 被遮挡的物体名称
        output_dir: 输出目录
    
    Returns:
        保存的文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建文件名: 原文件名_masked_物体名.后缀
    base_name = os.path.basename(original_path)
    name, ext = os.path.splitext(base_name)
    new_name = f"{name}_masked_{masked_object}{ext}"
    save_path = os.path.join(output_dir, new_name)
    
    masked_image.save(save_path)
    return save_path


def check_hallucination_in_response(
    response: str,
    masked_object: str,
    spacy: SpacyModel,
    wn: WordnetModel,
    valid_nouns: list[str],
    double_word_dict: dict[str, str],
    inv_syn_map: dict[str, str],
) -> bool:
    """
    检查响应中是否仍然提到被遮挡的物体（幻觉）
    
    Args:
        response: VLM 的响应文本
        masked_object: 被遮挡的物体
        spacy: Spacy 模型
        wn: WordNet 模型
        valid_nouns: 有效物体列表
        double_word_dict: 双词映射
        inv_syn_map: 逆同义词映射
    
    Returns:
        True 如果响应中仍然提到被遮挡物体（幻觉），否则 False
    """
    # 从响应中提取物体
    response_objects = extract_obj_w_gt(
        response,
        valid_nouns,
        double_word_dict,
        inv_syn_map,
        wn,
        force_list=True,
        return_repr=True,
    )
    
    if not response_objects or not response_objects[0]:
        return False
    
    response_objects = set(response_objects[0])
    masked_repr = inv_syn_map.get(masked_object, masked_object)
    
    # 检查被遮挡物体是否在响应中
    return masked_repr in response_objects or masked_object in response_objects


def run_mask_dataset_generation(
    dataset_path: str,
    output_dir: str,
    masked_images_dir: str,
    batch_size: int = 4,
    logger: Logger | None = None,
) -> None:
    """
    运行 Mask-based DPO 数据生成（支持断点续传）
    
    Args:
        dataset_path: 参考数据集路径
        output_dir: 输出目录（保存偏好对 jsonl）
        masked_images_dir: 遮挡图像保存目录
        batch_size: 批处理大小
        logger: 日志器
    
    断点续传机制:
        - 输出文件使用追加模式，每处理完一个物体立即保存
        - 重启时自动加载已处理的 (image_id, masked_object) 组合
        - 跳过已处理的物体，只处理未完成的
    """
    from model.auxiliary.global_vars import GVars
    
    # 获取全局配置
    model_dir = GVars.model_dir if hasattr(GVars, 'model_dir') else ""
    alter_device = GVars.alter_device if hasattr(GVars, 'alter_device') else "cuda:1"
    
    # ==================== 输出路径 ====================
    os.makedirs(output_dir, exist_ok=True)
    output_jsonl_path = os.path.join(output_dir, "mask_preference_pairs.jsonl")  # 主输出：幻觉记录
    processed_jsonl_path = os.path.join(output_dir, "mask_processed.jsonl")  # 断点续传：已处理记录
    
    # ==================== 断点续传：加载已处理记录 ====================
    if logger:
        logger.info("Loading processed records for resume...")
    
    done_image_ids = load_processed_image_ids(processed_jsonl_path)
    
    if logger:
        logger.info(f"Found {len(done_image_ids)} already processed images")
    
    # ==================== 初始化模型 ====================
    if logger:
        logger.info("Initializing models...")
    
    # 生成器
    generator = get_generator(use_vllm=True, debug=DEBUG)
    
    # 检测器
    yolo = YoloModel("yolo11x", model_dir=model_dir, logger=logger)
    dino = DINO("base", model_dir=model_dir, device=alter_device, logger=logger)
    
    # NLP 工具
    spacy = SpacyModel(model_size="md", model_dir=model_dir, device=alter_device, logger=logger)
    wn = WordnetModel(logger=logger)
    
    # 物体映射
    valid_nouns, inv_syn_map = get_object_n_represent()
    double_word_dict = get_double_word_dict()
    
    # ==================== 读取数据集 ====================
    if logger:
        logger.info(f"Loading dataset from {dataset_path}")
    
    dataset = read_reference_dataset(dataset_path)
    total_images = len(dataset)
    
    if logger:
        logger.info(f"Loaded {total_images} data points")
    
    # ==================== 主循环 ====================
    total_pairs = 0  # 幻觉对数量
    skipped_by_resume = len(done_image_ids)  # 断点续传跳过的图片数量
    
    # 用于保存 json 格式（方便实时查看）
    output_json_path = output_jsonl_path.replace(".jsonl", ".json")
    
    # 加载已有的 json 结果（断点续传）
    all_results: list[dict] = []
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            if logger:
                logger.info(f"Loaded {len(all_results)} existing results from json")
        except Exception:
            all_results = []
    
    for idx, item in enumerate(dataset):
        start_time = time()
        
        image_path = item["image_path"]
        question = item.get("question", "Describe this image in detail.")
        ground_truth = item.get("ground_truth", "")
        image_id = item.get("image_id", str(idx))
        
        # ===== 断点续传：检查该图片是否已处理 =====
        if image_id in done_image_ids:
            continue  # 跳过已处理的图片
        
        # 打开图像
        try:
            image = open_images(image_path)
        except Exception as e:
            if logger:
                logger.warning(f"Failed to open image {image_path}: {e}")
            continue
        
        # Step 1: 从 ground_truth 提取物体
        objects = extract_objects_from_text(
            ground_truth, valid_nouns, double_word_dict, inv_syn_map, wn
        )
        
        if not objects:
            if logger and DEBUG:
                logger.debug(f"No objects found in ground_truth for image {image_id}")
            continue
        
        # Step 2: 检测物体
        detection_result = detect_objects_with_details(image, objects, yolo, dino, spacy)
        detections = detection_result.detections
        
        if not detections:
            if logger and DEBUG:
                logger.debug(f"No objects detected in image {image_id}")
            continue
        
        # Step 2.5: 过滤低置信度的物体
        # 只要某类物体有一个 box 的置信度高于阈值，就保留该物体的所有 boxes
        filtered_detections = []
        for det in detections:
            if det.scores and det.boxes:
                # 检查是否有任意一个 box 高于阈值
                has_high_confidence = any(score >= MASK_SCORE_THRESHOLD for score in det.scores)
                if has_high_confidence:
                    # 保留该物体的所有 boxes
                    filtered_detections.append(det)
        
        if not filtered_detections:
            if logger and DEBUG:
                logger.debug(f"No high-confidence detections for image {image_id}")
            continue
        
        detections = filtered_detections
        
        # Step 3: 对每个检测到的物体创建遮挡图像并测试
        for det in detections:
            # 创建遮挡图像（内存中，不保存）
            masked_image = create_masked_image(image, det.boxes, det.scores)
            
            # Step 4: VLM 推理
            out: GenOutput = generator.gen(
                images=[masked_image],      # 输入图像
                users=[question],           # 用户问题
                assistants=[""],            # 当前上下文
                do_sample=True,             # 启用采样
                n=10,                       # 每个图像生成 10 个候选
                temp=0.7,                   # 温度参数，控制多样性
                force_list=True,            # 强制返回列表
            )
            masked_responses: list[str] = out.outputs[0] if out.outputs else []
            
            # Step 5: 对每个采样句子检查是否提到被遮挡物体
            hallu_count = 0
            hallu_responses = []  # 记录出现幻觉的句子
            
            for response in masked_responses:
                if check_hallucination_in_response(
                    response, det.object_name, spacy, wn,
                    valid_nouns, double_word_dict, inv_syn_map
                ):
                    hallu_count += 1
                    hallu_responses.append(response)
            
            # Step 6: 如果超过一半（>=5）的句子出现幻觉，记录
            hallu_ratio = hallu_count / len(masked_responses) if masked_responses else 0
            is_hallucination = hallu_count >= len(masked_responses) // 2  # 超过一半
            
            if is_hallucination:
                # 只有检测到幻觉时才保存遮挡图像
                masked_image_path = save_masked_image(
                    masked_image, image_path, det.object_name, masked_images_dir
                )
                
                # 幻觉情况：保存到主输出文件
                result = {
                    "image_id": image_id,
                    "image_path": image_path,
                    "question": question,
                    "masked_object": det.object_name,
                    "masked_image_path": masked_image_path,
                    "detection_scores": det.scores,
                    "detection_source": det.source,
                    "hallu_ratio": hallu_ratio,                # 幻觉比例
                    "total_samples": len(masked_responses),    # 总采样数量
                    "hallu_responses": hallu_responses,        # 出现幻觉的句子列表
                    "ground_truth": ground_truth,
                }
                save_result(output_jsonl_path, result)
                all_results.append(result)  # 收集结果用于 json 保存
                total_pairs += 1
                
                if logger:
                    logger.info(f"[Hallucination] image={image_id}, object={det.object_name}, ratio={hallu_count}/{len(masked_responses)}")
        
        # Step 7: 处理完该图片的所有物体后，记录已处理（断点续传）
        save_result(processed_jsonl_path, {"image_id": image_id, "image_path": image_path})
        done_image_ids.add(image_id)  # 更新内存中的集合
        
        # 实时保存 json 格式（每处理完一张图片就保存，防止中断丢失）
        if all_results:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=6)
        
        # 日志：每处理 10 张图像输出一次进度
        if logger and (idx + 1) % 10 == 0:
            elapsed = time() - start_time
            logger.info(
                f"Progress: {idx + 1}/{total_images} images, "
                f"Hallucinations: {total_pairs}, "
                f"Skipped(resume): {skipped_by_resume}, Time: {elapsed:.2f}s"
            )
    
    # ==================== 完成 ====================
    # 最终保存 json 格式
    if all_results:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    if logger:
        logger.info(
            f"Finished! "
            f"Hallucination pairs: {total_pairs}, "
            f"Skipped by resume: {skipped_by_resume}"
        )


# ==================== 主函数 ====================

if __name__ == "__main__":
    # 示例用法
    run_mask_dataset_generation(
        dataset_path="./dataset/reference_data.jsonl",
        output_dir="./results/mask_dpo",
        masked_images_dir="./results/mask_dpo/masked_images",
        batch_size=4,
    )

