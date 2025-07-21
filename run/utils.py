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


@dataclass
class refModel:
    args: Namespace
    valid_nouns: list[str] = field(init=False)
    inv_syn_map: dict[str, str] = field(init=False)
    double_words: dict[str, str] = field(init=False)
    gt_anno: dict[str, dict] = field(init=False, default=None)

    def __post_init__(self):
        self.valid_nouns, self.inv_syn_map, self.double_words = self._get_nouns()

    def _get_nouns(self) -> tuple[list[str], dict[str, str], dict[str, str]]:
        mscoco_objects, inverse_syn_map = get_object_n_represent()
        valid_nouns: list[str] = mscoco_objects
        double_word_dict = get_double_word_dict()
        return valid_nouns, inverse_syn_map, double_word_dict


def set_to_list(obj: set | dict | list) -> list | dict:
    """
    如果 obj 是 set 类型，则转换为 list；
    如果 obj 是字典或列表，则递归地转换其中的 set。
    """
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: set_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [set_to_list(element) for element in obj]
    else:
        return obj


def save_result(file_path: str, results: dict | list[dict] | None) -> None:
    """
    保存结果到指定路径的文件中。

    Args:
        file_path: 保存文件的路径。
        results: 需要保存的结果，可以是字典、字典的列表或迭代器。
    """
    if not file_path or not results:
        return

    # 如果 results 是一个迭代器，将其转换为列表
    if hasattr(results, "__iter__") and not isinstance(results, (dict, list)):
        results = list(results)

    if isinstance(results, list) and not results:
        return

    # 转换可能存在的 set 为 list
    results = set_to_list(results)

    ext: str = os.path.splitext(file_path)[-1]
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "a+", encoding="utf-8") as f:
        if ext == ".jsonl":
            if isinstance(results, list):
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(results, ensure_ascii=False) + "\n")
        elif ext in {".json", ".jsonfile"}:
            json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unspported extension {ext} for file: {file_path}")


def log_progress(
    logger: Logger | None, finished_data_num: int, num_of_data: int, batch_size: int, taken_time: float
) -> None:
    """
    记录进度日志。

    Args:
        logger: 日志记录器。
        finished_data_num: 已完成的数据数量。
        num_of_data: 总数据数量。
        batch_size: 批处理大小。
        taken_time: 处理所用时间。
    """
    if logger:
        logger.info(f"Progress: {finished_data_num}/{num_of_data}, Batch size: {batch_size}, Time: {taken_time:.2f}s")


def open_image(image: Image.Image | str) -> Image.Image:
    """
    打开图像并确保图像的模式为 RGB。

    Args:
        image: 图像，可以是 PIL.Image 对象或图像路径（本地路径或 URL）。

    Returns:
        打开的 PIL.Image 对象。
    """
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, str):
        if image.startswith("http"):
            img = Image.open(requests.get(image, stream=True).raw)
        else:
            img = Image.open(image)
    else:
        raise TypeError(f"image must be a URL string or a PIL.Image object, but got {type(image)}")

    if img.mode != "RGB":
        img: Image.Image = img.convert("RGB")
    return img


def open_images(images: Image.Image | str | list[Image.Image | str]) -> Image.Image | list[Image.Image]:
    """
    从文件路径或 URL 中打开图像，并确保图像的模式为 RGB。如果输入已经是 PIL.Image 对象，则直接返回。
    """
    if isinstance(images, list):
        return [open_image(i) for i in images]
    else:
        return open_image(images)


def annotate_with_dino_result(
    image: Image.Image | str,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: list[str],
) -> np.ndarray:
    """
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image (Image.Image): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates.
    scores (torch.Tensor): A tensor containing confidence scores for each bounding box.
    labels (list[str]): A list of labels for each bounding box.

    Returns:
    np.ndarray: The annotated image in numpy array format.
    """

    image: Image.Image = open_images(image)

    detections: sv.Detections = sv.Detections(xyxy=boxes.cpu().numpy())

    # 标签的格式为 "phrase score"，例如 "cat 0.95"
    labels: list[str] = [f"{phrase} {score:.2f}" for phrase, score in zip(labels, scores)]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    # Convert image to OpenCV format
    annotated_img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    # Annotate image with bounding boxes and labels
    annotated_img = bbox_annotator.annotate(scene=annotated_img, detections=detections)
    annotated_img = label_annotator.annotate(scene=annotated_img, detections=detections, labels=labels)
    return annotated_img


def crop_with_dino_boxes(image: Image.Image, boxes: torch.Tensor) -> list[Image.Image]:
    """
    This function crops an image based on bounding box coordinates.

    Parameters:
    image (Image.Image): The source image to be cropped.
    boxes (torch.Tensor): A tensor containing bounding box coordinates.

    Returns:
    list[Image.Image]: A list of cropped images, each corresponding to a given bounding box.
    """
    return [image.crop(tuple(box.tolist())) for box in boxes]


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
    从输入的句子当中提取物体。
    """
    valid_nouns_set: set[str] = set(valid_nouns)

    def get_repr(noun: str) -> str:
        """将名词转换为其代表词"""
        return inv_synonym_map.get(noun, noun) if inv_synonym_map else noun

    def extract(disc: str) -> list[str]:
        all_words: list[str] = [wn.lemma(w) for w in wn.word_tokenize(disc.lower())]

        # 将文本中的双词对象合并成一个对象
        i = 0
        double_words, idxs = [], []
        while i < len(all_words):
            idxs.append(i)
            double_word = " ".join(all_words[i : i + 2])  # 两个词组成的双词短语，用于查找映射关系
            if double_word in double_word_dict:
                double_words.append(double_word_dict[double_word])
                i += 2
            else:
                double_words.append(all_words[i])
                i += 1
        all_words = double_words

        if ("toilet" in all_words) & ("seat" in all_words):
            all_words = [word for word in all_words if word != "seat"]

        # 仅保留 caption 中的 MSCOCO 对象作为需要检测的对象
        idxs = [idxs[idx] for idx, word in enumerate(all_words) if word in valid_nouns_set]
        all_objects: list[str] = [word for word in all_words if word in valid_nouns_set]
        node_words: list[str] = [get_repr(obj) for obj in all_objects]  # 代表词，即每个词的同义词的第一个词

        # return all_objects, node_words, idxs, all_words
        return node_words if return_repr else all_objects

    if not isinstance(discriptions, list):
        discriptions = [discriptions]

    objects: list[list[str]] = [extract(disc) for disc in discriptions]

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
    Extract objects from text graphs. If valid_nouns is provided, only objects in the valid_nouns list will be extracted.

    Return:
    - list of objects extracted from text graphs, every element is a list of objects extracted from a text graph.
    """
    if not textgraphs:
        return [[]]
    if not isinstance(textgraphs[0][0], list):
        textgraphs = [textgraphs]

    objects_list: list[list[str]] = []
    for textgraph in textgraphs:
        objects: list[str] = []
        for triple in textgraph:
            process_triple(triple, objects, [], [], spacy, wn, valid_nouns, inverse_synonym_map)
        objects_list.append(objects)

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
    Process a triple and add its subject, predicate, and object to the noun, attributes and relation lists.

    Args:
    - triple: A list containing subject, predicate, and object.

    Returns:
    - None, but the subject, predicate, and object will be added to the noun, attributes and relation lists.
    """
    if len(triple) != 3:
        return
    subject, predicate, obj = triple

    def add_noun(word: str) -> None:
        add_to_nouns_if_valid(word, nouns, spacy, wn, valid_nouns, inverse_synonym_map)

    if predicate in ["has", "have"]:  # Means belongs to
        add_noun(subject)
        add_noun(obj)
    elif predicate in ["is", "are"]:
        add_noun(subject)
        if spacy.is_noun(subject) or spacy.is_noun(obj):
            attributes.append([subject, predicate, obj])
    else:
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
    如果单词是有效名词，则将其添加到名词列表中。
    若提供了有效名词列表，则只有在单词在有效名词列表中时才会被添加。若没有提供有效名词列表，则会检查单词是否是具体名词。

    参数:
    - word: 要检查的单词。
    - nouns: 名词列表。
    - spacy: Spacy 模型实例。
    - wn: Wordnet 模型实例。
    - valid_nouns: 有效名词列表（可选）。
    - inverse_synonym_map: 逆同义词映射字典（可选）。

    返回:
    - bool: 如果单词被添加到名词列表中，则返回 True；否则返回 False。
    """
    word = word.lower()
    if word in nouns:
        return False

    if valid_nouns:
        if object_in_set(word, valid_nouns, spacy, wn, inverse_synonym_map):
            nouns.append(word)
            return True
        else:
            return False
    else:
        doc: Doc = spacy(word)
        if len(doc) == 1:
            token: Token = doc[0]
            lemma: str = token.lemma_
            if spacy.is_noun(token) and lemma not in nouns and wn.is_concrete_noun(lemma):
                nouns.append(lemma)
                return True
        else:
            for token in doc:
                lemma = token.lemma_
                if spacy.is_noun(token) and lemma not in nouns and wn.is_concrete_noun(lemma):
                    nouns.append(word)
                    return True
        return False


def object_in_set(
    obj: str,
    target_set: list[str] | set[str],
    spacy: SpacyModel,
    wn: WordnetModel,
    inv_synonym_map: dict[str, str] | None = None,
    allow_synonym: bool = False,
) -> bool:
    def get_repr(noun: str) -> str:
        """将名词转换为其代表词"""
        return inv_synonym_map.get(noun, noun) if inv_synonym_map else noun

    """检查物体（obj）是否在目标集合（target_set）中"""
    repr_obj: str = get_repr(obj)
    repr_target_set: list[str] = [get_repr(obj) for obj in target_set]
    if repr_obj in repr_target_set or (spacy is not None and get_repr(spacy.lemma(obj)) in repr_target_set):
        return True

    # 如果允许查找同义词列表，则遍历同义词列表，检查每个同义词是否在 target_set 中
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
    if not isinstance(object_list, list):
        object_list = [object_list]

    if check_type.lower() == "all":
        return all(object_in_set(obj, target_set, spacy, wn, inv_synonym_map, False) for obj in object_list)
    elif check_type.lower() == "any":
        return any(object_in_set(obj, target_set, spacy, wn, inv_synonym_map, False) for obj in object_list)
    else:
        raise ValueError(f"Invalid check type: {check_type}")


def pack_objs_for_dino(objects: list[list[str]] | list[str]) -> list[str] | str:
    """
    pack objects for DINO detector, support nested list.
    """
    if (not isinstance(objects, list) and not isinstance(objects, str)) or not objects:
        return ""
    elif isinstance(objects, str):
        return objects
    elif isinstance(objects[0], list):  # objects is list[list[str]]
        return [pack_objs_for_dino(obj) for obj in objects]
    else:  # objects is list[str]
        return ".".join(set(objects)) + "." if objects else ""


def unpack_objs_from_dino(objects: list[str] | str) -> list[list[str]] | list[str]:
    if isinstance(objects, list):
        return [unpack_objs_from_dino(obj) for obj in objects]
    else:
        return objects.rstrip(".").split(".") if objects else []


def get_dino_detected_objects(obj_to_detects: str, dino_result_labels: list[str]) -> set[str]:
    detected_obj: set[str] = set()

    for obj in filter(None, obj_to_detects.split(".")):
        if any(obj in label for label in dino_result_labels):
            detected_obj.add(obj)

    return detected_obj


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
    获取幻觉物体列表，如果传入了图像和 DINO 检测器，则会使用 DINO 检测器检测未知物体。

    参数:
    - objects_list: 所有物体的列表，每个元素是一个物体列表，代表一个句子中所包含的物体。
    - nonhallu_objects: 真实物体的列表。
    - hallu_objects: 幻觉物体的列表。
    - spacy: Spacy 模型实例。
    - wn: Wordnet 模型实例。
    - image: 图像对象。
    - dino: DINODetector 实例。
    - yolo_results: YOLO 检测结果。
    - yolo_labels: YOLO 标签。
    - uncertain_objects: 不能确定的物体列表。
    - inv_synonym_map: 逆同义词映射字典。

    返回:
    - 幻觉物体的列表，每个元素是一个列表，代表单个句子中所包含的幻觉物体，如果没有幻觉物体，则返回空列表。
    - 非幻觉物体的列表，每个元素是一个列表，代表单个句子中所包含的非幻觉物体，如果没有非幻觉物体，则返回空列表。
    """

    def get_repr(noun: str) -> str:
        """将名词转换为其代表词"""
        return inv_syn_map.get(noun, noun) if inv_syn_map else noun

    def get_cached_objects() -> list[str]:
        cached_objs: list[str] = []
        if nonhallu_objects:
            cached_objs.extend(nonhallu_objects)
        if hallu_objects:
            cached_objs.extend(hallu_objects)
        if uncertain_objects:
            cached_objs.extend(uncertain_objects)
        return cached_objs

    def get_set() -> set[str]:
        _set = set()
        if nonhallu_objects:
            _set.update(nonhallu_objects)
        if uncertain_objects:
            _set.update(uncertain_objects)
        return _set

    def recognize_by_yolo(obj: str) -> bool:
        """返回该物体是否被 YOLO 模型认可。如果不在 YOLO 模型的检测范围内（标签中），视为认可"""
        if yolo_results is None or yolo_labels is None:
            return True
        repr_obj: str = spacy.lemma(get_repr(obj))
        if repr_obj in yolo_labels:
            return repr_obj in yolo_results
        else:
            return True

    def recognize_by_dino(obj: str, detected_obj: list[str] | set[str]) -> bool:
        """返回该物体是否被 DINO 模型认可"""
        return object_in_set(obj, detected_obj, spacy, wn, inv_syn_map, False)

    def get_uncached_objects(objects_list: list[list[str]], cached_objs: list[str]) -> list[str]:
        """根据已缓存的物体列表获取未缓存的物体列表，已去重"""
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

    cached_objs: list[str] = get_cached_objects()
    uncached_objs: list[str] = get_uncached_objects(objects_list, cached_objs)

    # Handle uncached objects
    if uncached_objs:
        if image is not None and dino is not None:
            obj_for_dino_to_detect: str = pack_objs_for_dino(uncached_objs)
            dino_results: dict[str] = dino.detect(image, obj_for_dino_to_detect, force_list=False)
            detected_obj: set[str] = get_dino_detected_objects(obj_for_dino_to_detect, dino_results["labels"])

            for obj in uncached_objs:
                yolo_recognized, dino_recognized = recognize_by_yolo(obj), recognize_by_dino(obj, detected_obj)
                if dino_recognized and yolo_recognized and obj not in nonhallu_objects:
                    nonhallu_objects.append(obj)
                elif not dino_recognized and not yolo_recognized and obj not in hallu_objects:
                    hallu_objects.append(obj)
                elif uncertain_objects is not None and obj not in uncertain_objects:
                    if not yolo_recognized and detector_reject is not None and "yolo" in detector_reject:
                        detector_reject["yolo"].append(obj)
                    if not dino_recognized and detector_reject is not None and "dino" in detector_reject:
                        detector_reject["dino"].append(obj)
                    uncertain_objects.append(obj)
        else:
            for obj in uncached_objs:
                yolo_recognized: bool = recognize_by_yolo(obj)
                if yolo_recognized and obj not in nonhallu_objects:
                    nonhallu_objects.append(obj)
                elif not yolo_recognized and obj not in hallu_objects:
                    hallu_objects.append(obj)
    del cached_objs, uncached_objs

    hallu_objects_list: list[list[str]] = []
    nonhallu_objects_list: list[list[str]] = []
    _set = get_set()  # 用于过滤掉已知物体
    for objects in objects_list:
        hallu_objects_list.append(
            [
                obj
                for obj in objects
                if spacy.lemma(obj) in hallu_objects or not object_in_set(obj, _set, spacy, wn, inv_syn_map)
            ]
        )
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
    获取幻觉物体列表，如果传入了图像和 DINO 检测器，则会使用 DINO 检测器检测未知物体。

    参数:
    - b_object_lists: 所有物体的列表集合，每个元素是一个物体列表，代表一个句子中所包含的物体。
    - b_nonhallu_objects: 真实物体的列表集合。
    - b_hallu_objects: 幻觉物体的列表集合。
    - spacy: Spacy 模型实例。
    - wn: Wordnet 模型实例。
    - images: 图像对象集合。
    - dino: DINODetector。
    - b_yolo_results: YOLO 检测结果集合。
    - yolo_labels: YOLO 标签集合。
    - uncertain_objectses: 不能确定的物体列表集合。
    - inv_syn_maps: 逆同义词映射字典集合。

    返回:
    - 包含每组输入的结果字典列表，每个字典包括幻觉物体列表和非幻觉物体列表。
    """
    b_size: int = len(b_object_lists)

    def get_repr(noun: str) -> str:
        """将名词转换为其代表词"""
        return inv_syn_map.get(noun, noun) if inv_syn_map else noun

    def get_cached_objects(idx: int) -> list[str]:
        cached_objs: list[str] = []
        if b_nonhallu_objects:
            cached_objs.extend(b_nonhallu_objects[idx])
        if b_hallu_objects:
            cached_objs.extend(b_hallu_objects[idx])
        if b_uncertain_objects:
            cached_objs.extend(b_uncertain_objects[idx])
        return cached_objs

    def get_set(idx: int) -> set[str]:
        _set = set()
        if b_nonhallu_objects:
            _set.update(b_nonhallu_objects[idx])
        if b_uncertain_objects:
            _set.update(b_uncertain_objects[idx])
        return _set

    def recognize_by_yolo(obj: str, idx: int) -> bool:
        """返回该物体是否被 YOLO 模型认可。如果不在 YOLO 模型的检测范围内（标签中），视为认可"""
        if b_yolo_results is None or yolo_labels is None:
            return True
        repr_obj: str = spacy.lemma(get_repr(obj))
        if repr_obj in yolo_labels:
            return repr_obj in b_yolo_results[idx]
        else:
            return True

    def recognize_by_dino(obj: str, detected_obj: list[str] | set[str]) -> bool:
        return object_in_set(obj, detected_obj, spacy, wn, inv_syn_map, False)

    def get_uncached_objects(objects_list: list[list[str]], cached_objs: list[str]) -> list[str]:
        """根据已缓存的物体列表获取未缓存的物体列表，已去重"""
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

    b_cached_objs: list[list[str]] = [get_cached_objects(i) for i in range(b_size)]
    b_uncached_objs: list[list[str]] = [
        get_uncached_objects(b_object_lists[i], b_cached_objs[i]) for i in range(b_size)
    ]
    if b_uncached_objs and any(b_uncached_objs):
        b_obj_for_dino: list[str] = pack_objs_for_dino(b_uncached_objs)
        dino_results: list[dict[str]] = dino.detect(images, b_obj_for_dino, force_list=True)
        b_detected_obj: list[set[str]] = [
            get_dino_detected_objects(obj, dino_results[i]["labels"]) for i, obj in enumerate(b_obj_for_dino)
        ]

        for idx, uncached_objs in enumerate(b_uncached_objs):
            for obj in uncached_objs:
                if b_yolo_results is not None:  # YOLO + DINO
                    yolo_recognized = recognize_by_yolo(obj, idx)
                    dino_recognized = recognize_by_dino(obj, b_detected_obj[idx])

                    if dino_recognized and yolo_recognized:
                        if obj not in b_nonhallu_objects[idx]:
                            b_nonhallu_objects[idx].append(obj)
                    elif not dino_recognized and not yolo_recognized:
                        if obj not in b_hallu_objects[idx]:
                            b_hallu_objects[idx].append(obj)
                    elif b_uncertain_objects[idx] is not None and obj not in b_uncertain_objects[idx]:
                        b_uncertain_objects[idx].append(obj)

                        if not yolo_recognized and b_detector_rejects is not None and "yolo" in b_detector_rejects[idx]:
                            b_detector_rejects[idx]["yolo"].append(obj)
                        if not dino_recognized and b_detector_rejects is not None and "dino" in b_detector_rejects[idx]:
                            b_detector_rejects[idx]["dino"].append(obj)
                else:  # DINO only
                    dino_recognized = recognize_by_dino(obj, b_detected_obj[idx])
                    if dino_recognized and obj not in b_nonhallu_objects[idx]:
                        b_nonhallu_objects[idx].append(obj)
                    elif not dino_recognized and obj not in b_hallu_objects[idx]:
                        b_hallu_objects[idx].append(obj)

    del b_cached_objs, b_uncached_objs

    b_hallu_objects_list: list[list[list[str]]] = []
    b_nonhallu_objects_list: list[list[list[str]]] = []
    for idx, objects_list in enumerate(b_object_lists):
        _set = get_set(idx)  # 用于过滤掉已知物体

        _hallu_objects_list, _nonhallu_objects_list = [], []
        for objects in objects_list:
            _hallu_objects_list.append(
                [
                    obj
                    for obj in objects
                    if spacy.lemma(obj) in b_hallu_objects[idx] or not object_in_set(obj, _set, spacy, wn, inv_syn_map)
                ]
            )
            _nonhallu_objects_list.append(
                [obj for obj in objects if object_in_set(obj, b_nonhallu_objects[idx], spacy, wn, inv_syn_map)]
            )
        b_hallu_objects_list.append(_hallu_objects_list)
        b_nonhallu_objects_list.append(_nonhallu_objects_list)

    return b_hallu_objects_list, b_nonhallu_objects_list


def tokenize_sent(text: str) -> list[str]:
    """
    将输入文本分割成句子列表。
    """
    from nltk.tokenize import sent_tokenize

    return sent_tokenize(text)


def get_finish_flag(
    sentences: list[str],
    stop_threshold: float = 0.5,
    remove_duplicates: bool = False,
) -> tuple[list[str], bool]:
    """
    输入新生成的所有句子的列表，当句子列表中为空的元素的比率大于 stop_threshold 时停止生成，返回新的非空句子列表和是否停止生成的标志。

    Args:
    - sentences: 新生成的所有句子的列表。
    - stop_threshold: 停止生成的阈值。
    - remove_duplicates: 是否去重，默认为 False。

    Returns:
    - 新的非空句子列表。
    - 是否停止生成的标志。
    """
    valid_sentences: list[str] = [s for s in sentences if s]
    if remove_duplicates:
        valid_sentences = list(set(valid_sentences))
    should_stop = (len(sentences) - len(valid_sentences)) / len(sentences) > stop_threshold
    return valid_sentences, should_stop


def concat_sents(description: str, previous: str, retrospect_num: int = 1) -> str:
    """
    将当前描述字符串与前面的描述字符串连接起来，形成上下文字符串。

    Args:
    - description: 当前描述的字符串。
    - previous: 上文，可能包含多个句子，也可能为空。
    - retrospect_num: 回溯的句子数量。

    Returns:
    - 生成的上下文字符串。
    """
    if not previous or retrospect_num <= 0:
        return description

    sentences: list[str] = tokenize_sent(previous)
    context: str = " ".join(sentences[-retrospect_num:]) if len(sentences) >= retrospect_num else " ".join(sentences)
    return " ".join([context, description])


def resolve_corefs(
    spacy: SpacyModel,
    descriptions: list[str] | list[list[str]],
    previous: list[str],
    retro_num: int,
    force_list: bool = True,
) -> list[str] | list[list[str]]:
    """
    根据输入的句子和前置句子，使用 Spacy 模型进行指代消解，返回指代消解后的“当前”句子。

    参数:
    - spacy (SpacyModel): 一个 Spacy 模型实例，用于进行指代消解。
    - descriptions (list[str] | list[list[str]]): 主要的句子列表或句子列表的列表。
    - previous (list[str]): 提供上下文的前置句子列表。
    - retro_num (int): 每个主要句子要考虑的前置句子数量，用于指代消解。
    - force_list (bool, optional): 强制输出为列表，即使只有一个结果。默认为 True。

    返回:
    - list[str] | list[list[str]]: 进行指代消解后的句子列表，保留与原始 descriptions 一样的格式和位置顺序。
    """

    if isinstance(descriptions[0], list):
        # 展平 descriptions 并扩展 previous
        flat_descriptions: list[str] = [sent for sublist in descriptions for sent in sublist]  # len = (batch_size * n)
        expanded_previous: list[str] = [previous[idx] for idx, des in enumerate(descriptions) for _ in des]
    else:
        flat_descriptions = descriptions
        expanded_previous = previous

    if retro_num > 0 and any(pre for pre in previous):  # Need to resolve corefs
        concated_sents: list[str] = [
            concat_sents(d, p, retro_num) for d, p in zip(flat_descriptions, expanded_previous)
        ]  # len = (batch_size * n)
        resolved_texts: list[str] = spacy.resolve_coref(concated_sents, force_list=force_list)
    else:
        resolved_texts: list[str] = flat_descriptions

    # 获取最后一句话
    resolved_last_sentences: list[str] = [t.split(". ")[-1] for t in resolved_texts]

    # 如果之前展平过，则将 resolved_texts 恢复为原始 descriptions 的格式
    if isinstance(descriptions[0], list):
        resolved_texts_nested = []
        idx = 0
        for sublist in descriptions:
            resolved_texts_nested.append(resolved_last_sentences[idx : idx + len(sublist)])
            idx += len(sublist)
        return resolved_texts_nested  # len = batch_size
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
    处理一组句子（描述）并进行指代消解，然后将句子解析为文本图。

    参数:
    - spacy (SpacyModel): 一个 Spacy 模型实例，用于进行指代消解。
    - sg_parser (SGParser): 一个句子图解析器实例，用于将句子解析为结构。
    - descriptions (list[str] | str): 主要的句子列表。
    - previous (list[str] | str, optional): 提供上下文的前置句子列表。
    - retro_num (int): 每个主要句子要考虑的前置句子数量，用于指代消解。
    - force_list (bool, optional): 强制输出为列表，即使只有一个结果。默认为 True。

    返回:
    - list[list[list[str]]]: 一个嵌套列表，每个子列表表示句子的解析结构。
    """
    descriptions, previous = ensure_lists(descriptions, previous)
    resolved_texts: list[str] = resolve_corefs(spacy, descriptions, previous, retro_num, force_list)
    textgraphs: list[list[list[str]]] = sg_parser.pharse(resolved_texts, force_list=force_list)
    return textgraphs


def yolo_detect(yolo: YoloModel | None, data_states: list[DataStateForBuildDataset]) -> None:
    """
    使用 YOLO 检测器对数据状态中未被检测过的状态进行检测，并将检测结果保存在状态对象中，无返回值。
    """
    if not yolo:
        return

    states_to_detect = [s for s in data_states if not s.yolo_detected]
    if not states_to_detect:
        return

    images_to_detect: list[Image.Image] = [s.image for s in states_to_detect]
    detection_results: list[YoloResult] = yolo.predict(images_to_detect, force_list=True)

    for s, result in zip(states_to_detect, detection_results):
        s.yolo_result = result
        s.yolo_detected = True


def pop_first_sents(sents: list[str]) -> tuple[list[str], list[str]]:
    """输入一个句子列表，返回每个句子的第一个句子，以及剩余的句子列表。"""
    first_sents = []
    remaining_sents = []

    for sent in sents:
        split_sents = sent.split(".", 1)
        first_sents.append(split_sents[0] + ".")
        if len(split_sents) > 1:
            remaining_sents.append(split_sents[1].lstrip(" "))
        else:
            remaining_sents.append("")

    return first_sents, remaining_sents
