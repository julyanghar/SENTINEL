"""
================================================================================
幻觉检测算法详解（带详细注释）
================================================================================

本文件包含从 run/utils.py 提取的幻觉检测核心算法的详细注释版本。
这是SENTINEL项目中最关键的算法之一。

核心思想:
    使用YOLO和Grounding DINO两个检测器进行交叉验证，
    根据两个检测器的一致性判断对象是否为幻觉。

检测策略:
    1. YOLO: 封闭词汇检测器，80类COCO对象
    2. Grounding DINO: 开放词汇检测器，可检测任意文本描述的对象
    
    判断逻辑:
    ┌─────────────┬──────────────┬─────────────────┐
    │   YOLO      │    DINO      │     结果        │
    ├─────────────┼──────────────┼─────────────────┤
    │    认可     │     认可     │   非幻觉对象    │
    │   不认可    │    不认可    │   幻觉对象      │
    │    认可     │    不认可    │   不确定对象    │
    │   不认可    │     认可     │   不确定对象    │
    └─────────────┴──────────────┴─────────────────┘

特殊处理:
    - 如果对象不在YOLO的80类标签中，视为YOLO认可
    - 这样可以处理YOLO无法检测的对象类别（如"sky", "grass"等）
================================================================================
"""

from PIL import Image
from model.detector.grounding_dino import DINO
from model.others.spacy_model import SpacyModel
from model.others.wordnet import WordnetModel


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
    
    这个函数考虑了以下匹配方式:
    1. 直接匹配：对象的代表词与目标集合中的代表词相同
    2. 词干匹配：对象的词干形式在目标集合中
    3. 同义词匹配（可选）：对象的同义词在目标集合中
    
    参数:
        obj: str
            待检查的对象名称
        target_set: list[str] | set[str]
            目标集合
        spacy: SpacyModel
            Spacy模型，用于词干提取
        wn: WordnetModel
            WordNet模型，用于同义词查找
        inv_synonym_map: dict[str, str] | None
            逆同义词映射表：同义词 -> 代表词
        allow_synonym: bool
            是否允许同义词匹配
    
    返回:
        bool: 对象是否在集合中
    
    示例:
        # "dog" 和 "dogs" 应该匹配
        # "cat" 和 "kitten" 通过同义词映射应该匹配
    """
    
    def get_repr(noun: str) -> str:
        """
        获取名词的代表词
        
        通过同义词映射，将所有同义词映射到一个统一的代表词。
        例如: "kitten" -> "cat", "puppy" -> "dog"
        """
        return inv_synonym_map.get(noun, noun) if inv_synonym_map else noun

    # 获取对象的代表词
    repr_obj: str = get_repr(obj)
    
    # 获取目标集合中所有元素的代表词
    repr_target_set: list[str] = [get_repr(obj) for obj in target_set]
    
    # 方式1: 直接匹配代表词
    if repr_obj in repr_target_set:
        return True
    
    # 方式2: 词干匹配（使用Spacy的lemma）
    if spacy is not None and get_repr(spacy.lemma(obj)) in repr_target_set:
        return True

    # 方式3: 同义词匹配（可选）
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
    检查多个对象是否在目标集合中
    
    参数:
        object_list: 对象列表
        target_set: 目标集合
        check_type: "all" - 所有对象都在集合中
                    "any" - 任一对象在集合中
    
    返回:
        bool: 根据check_type的检查结果
    """
    if not isinstance(object_list, list):
        object_list = [object_list]

    if check_type.lower() == "all":
        # 所有对象都必须在集合中
        return all(
            object_in_set(obj, target_set, spacy, wn, inv_synonym_map, False) 
            for obj in object_list
        )
    elif check_type.lower() == "any":
        # 任一对象在集合中即可
        return any(
            object_in_set(obj, target_set, spacy, wn, inv_synonym_map, False) 
            for obj in object_list
        )
    else:
        raise ValueError(f"Invalid check type: {check_type}")


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
    核心函数：获取幻觉对象列表
    
    这是SENTINEL的核心算法，使用YOLO和Grounding DINO交叉验证
    来判断每个对象是否为幻觉。
    
    算法流程:
    ┌────────────────────────────────────────────────────────────────┐
    │  1. 识别已缓存对象                                              │
    │     cached = nonhallu ∪ hallu ∪ uncertain                      │
    │                                                                │
    │  2. 获取未缓存对象                                              │
    │     uncached = {obj ∈ objects | obj ∉ cached}                  │
    │                                                                │
    │  3. 对每个未缓存对象进行检测                                    │
    │     ├── YOLO检测: yolo_ok = (obj ∈ yolo_results) OR            │
    │     │                       (obj ∉ yolo_labels)                │
    │     │                                                          │
    │     └── DINO检测: dino_ok = DINO.detect(image, obj) ≠ ∅        │
    │                                                                │
    │  4. 根据检测结果分类                                            │
    │     if yolo_ok AND dino_ok:                                    │
    │         nonhallu.add(obj)  # 确认存在                          │
    │     elif NOT yolo_ok AND NOT dino_ok:                          │
    │         hallu.add(obj)     # 确认幻觉                          │
    │     else:                                                      │
    │         uncertain.add(obj) # 检测器不一致                       │
    │                                                                │
    │  5. 返回每个句子的幻觉/非幻觉对象列表                           │
    └────────────────────────────────────────────────────────────────┘
    
    参数:
        objects_list: list[list[str]]
            所有候选句子的对象列表
            objects_list[i] = 第i个句子中的所有对象
        nonhallu_objects: list[str]
            已确认的非幻觉对象列表（会被更新）
        hallu_objects: list[str]
            已确认的幻觉对象列表（会被更新）
        spacy: SpacyModel
            用于词干提取
        wn: WordnetModel
            用于同义词查找
        image: Image.Image
            当前图像（用于DINO检测）
        dino: DINO
            Grounding DINO检测器
        yolo_results: list[str]
            YOLO在当前图像上的检测结果（标签列表）
        yolo_labels: list[str]
            YOLO支持的所有标签（80类）
        uncertain_objects: list[str]
            不确定对象列表（会被更新）
        detector_reject: dict[str, list[str]]
            记录被各检测器拒绝的对象
        inv_syn_map: dict[str, str]
            同义词映射
    
    返回:
        tuple[list[list[str]], list[list[str]]]
            (幻觉对象列表, 非幻觉对象列表)
            每个列表的长度与输入objects_list相同
    
    重要说明:
        - 此函数会修改传入的nonhallu_objects, hallu_objects, 
          uncertain_objects列表（原地更新缓存）
        - 使用缓存机制避免重复检测同一对象
    """

    def get_repr(noun: str) -> str:
        """获取名词的代表词"""
        return inv_syn_map.get(noun, noun) if inv_syn_map else noun

    def get_cached_objects() -> list[str]:
        """
        获取所有已缓存的对象
        包括：非幻觉对象 + 幻觉对象 + 不确定对象
        """
        cached_objs: list[str] = []
        if nonhallu_objects:
            cached_objs.extend(nonhallu_objects)
        if hallu_objects:
            cached_objs.extend(hallu_objects)
        if uncertain_objects:
            cached_objs.extend(uncertain_objects)
        return cached_objs

    def get_set() -> set[str]:
        """
        获取"可接受"对象集合
        用于最终分类时判断对象是否为幻觉
        包括：非幻觉对象 + 不确定对象
        """
        _set = set()
        if nonhallu_objects:
            _set.update(nonhallu_objects)
        if uncertain_objects:
            _set.update(uncertain_objects)
        return _set

    def recognize_by_yolo(obj: str) -> bool:
        """
        YOLO识别判断
        
        逻辑:
            1. 如果没有YOLO结果或标签，返回True（视为认可）
            2. 获取对象的代表词的词干形式
            3. 如果对象在YOLO标签中:
               - 检查是否在YOLO检测结果中
            4. 如果对象不在YOLO标签中:
               - 返回True（不在检测范围内，视为认可）
        
        返回:
            bool: YOLO是否认可该对象
        """
        if yolo_results is None or yolo_labels is None:
            return True
            
        # 获取对象的标准形式
        repr_obj: str = spacy.lemma(get_repr(obj))
        
        if repr_obj in yolo_labels:
            # 对象在YOLO的检测范围内，检查是否被检测到
            return repr_obj in yolo_results
        else:
            # 对象不在YOLO的检测范围内（如"sky", "grass"等）
            # 视为YOLO认可，交给DINO判断
            return True

    def recognize_by_dino(obj: str, detected_obj: list[str] | set[str]) -> bool:
        """
        DINO识别判断
        
        检查对象是否在DINO检测结果中
        
        返回:
            bool: DINO是否认可该对象
        """
        return object_in_set(obj, detected_obj, spacy, wn, inv_syn_map, False)

    def get_uncached_objects(objects_list: list[list[str]], cached_objs: list[str]) -> list[str]:
        """
        获取未缓存的对象列表
        
        从所有对象中过滤掉已经处理过的对象
        
        返回:
            list[str]: 未缓存对象列表（去重）
        """
        return list(
            set(
                [
                    spacy.lemma(obj)  # 使用词干形式去重
                    for objects in objects_list
                    for obj in objects
                    if not object_in_set(obj, cached_objs, spacy, wn, inv_syn_map, False)
                ]
            )
        )

    # ========== 步骤1: 识别缓存状态 ==========
    cached_objs: list[str] = get_cached_objects()
    uncached_objs: list[str] = get_uncached_objects(objects_list, cached_objs)

    # ========== 步骤2: 处理未缓存对象 ==========
    if uncached_objs:
        if image is not None and dino is not None:
            # ---------- 使用YOLO + DINO双重验证 ----------
            
            # 准备DINO检测的输入
            # 格式: "cat.dog.person." （以点分隔的对象列表）
            obj_for_dino_to_detect: str = pack_objs_for_dino(uncached_objs)
            
            # 执行DINO检测
            dino_results: dict[str] = dino.detect(image, obj_for_dino_to_detect, force_list=False)
            
            # 从DINO结果中提取检测到的对象
            detected_obj: set[str] = get_dino_detected_objects(
                obj_for_dino_to_detect, 
                dino_results["labels"]
            )

            # 对每个未缓存对象进行分类
            for obj in uncached_objs:
                yolo_recognized = recognize_by_yolo(obj)
                dino_recognized = recognize_by_dino(obj, detected_obj)
                
                if dino_recognized and yolo_recognized:
                    # 两个检测器都认可 -> 非幻觉对象
                    if obj not in nonhallu_objects:
                        nonhallu_objects.append(obj)
                        
                elif not dino_recognized and not yolo_recognized:
                    # 两个检测器都不认可 -> 幻觉对象
                    if obj not in hallu_objects:
                        hallu_objects.append(obj)
                        
                elif uncertain_objects is not None and obj not in uncertain_objects:
                    # 检测器结果不一致 -> 不确定对象
                    
                    # 记录是哪个检测器拒绝了
                    if not yolo_recognized and detector_reject is not None and "yolo" in detector_reject:
                        detector_reject["yolo"].append(obj)
                    if not dino_recognized and detector_reject is not None and "dino" in detector_reject:
                        detector_reject["dino"].append(obj)
                        
                    uncertain_objects.append(obj)
        else:
            # ---------- 只使用YOLO验证 ----------
            for obj in uncached_objs:
                yolo_recognized: bool = recognize_by_yolo(obj)
                if yolo_recognized and obj not in nonhallu_objects:
                    nonhallu_objects.append(obj)
                elif not yolo_recognized and obj not in hallu_objects:
                    hallu_objects.append(obj)
    
    # 清理临时变量
    del cached_objs, uncached_objs

    # ========== 步骤3: 生成每个句子的分类结果 ==========
    hallu_objects_list: list[list[str]] = []
    nonhallu_objects_list: list[list[str]] = []
    
    _set = get_set()  # 可接受对象集合
    
    for objects in objects_list:
        # 幻觉对象：在hallu_objects中 或 不在可接受集合中
        hallu_objects_list.append(
            [
                obj
                for obj in objects
                if spacy.lemma(obj) in hallu_objects 
                or not object_in_set(obj, _set, spacy, wn, inv_syn_map)
            ]
        )
        
        # 非幻觉对象：在nonhallu_objects中
        nonhallu_objects_list.append(
            [
                obj 
                for obj in objects 
                if object_in_set(obj, nonhallu_objects, spacy, wn, inv_syn_map)
            ]
        )
    
    return hallu_objects_list, nonhallu_objects_list


def pack_objs_for_dino(objects: list[list[str]] | list[str]) -> list[str] | str:
    """
    为DINO检测器打包对象列表
    
    DINO需要特定格式的输入: "cat.dog.person."
    这个函数将对象列表转换为这种格式
    
    参数:
        objects: 对象列表，可以是嵌套的
    
    返回:
        格式化的字符串或字符串列表
    
    示例:
        ["cat", "dog"] -> "cat.dog."
        [["cat"], ["dog", "bird"]] -> ["cat.", "dog.bird."]
    """
    if (not isinstance(objects, list) and not isinstance(objects, str)) or not objects:
        return ""
    elif isinstance(objects, str):
        return objects
    elif isinstance(objects[0], list):
        # 嵌套列表，递归处理
        return [pack_objs_for_dino(obj) for obj in objects]
    else:
        # 普通列表，转换为点分隔格式
        return ".".join(set(objects)) + "." if objects else ""


def get_dino_detected_objects(obj_to_detects: str, dino_result_labels: list[str]) -> set[str]:
    """
    从DINO检测结果中提取检测到的对象
    
    参数:
        obj_to_detects: DINO输入格式的对象字符串 ("cat.dog.")
        dino_result_labels: DINO返回的标签列表
    
    返回:
        检测到的对象集合
    """
    detected_obj: set[str] = set()

    for obj in filter(None, obj_to_detects.split(".")):
        if any(obj in label for label in dino_result_labels):
            detected_obj.add(obj)

    return detected_obj


# ============================================================================
# 批量版本（用于多个样本同时处理）
# ============================================================================

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
    批量版本的幻觉对象检测
    
    与get_hallu_objects类似，但处理多个样本（batch）
    主要优化：合并DINO检测请求，减少检测器调用次数
    
    参数:
        b_object_lists: 批量对象列表，b_object_lists[样本][句子][对象]
        b_nonhallu_objects: 批量非幻觉对象列表
        其他参数类似get_hallu_objects
    
    返回:
        (批量幻觉对象列表, 批量非幻觉对象列表)
    
    性能优化:
        - 批量调用DINO，减少GPU调用开销
        - 共享词汇处理，减少重复计算
    """
    b_size: int = len(b_object_lists)
    
    # ... (实现逻辑与get_hallu_objects类似，但处理批量数据)
    # 完整实现见源代码 run/utils.py
    
    pass  # 省略，实际代码见run/utils.py
