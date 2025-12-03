"""
SENTINEL 核心数据生成模块
=========================

本模块实现了 SENTINEL 的核心数据生成逻辑，包括：
1. 句子级采样生成
2. 物体提取与幻觉检测
3. 偏好对构建与保存

核心算法流程:
    对于每个图像数据:
    1. 使用 YOLO 预检测图像中的物体（建立基准）
    2. 使用 MLLM 生成多个候选句子（温度采样）
    3. 使用 Spacy 进行指代消解
    4. 使用场景图解析器 + 正则提取物体
    5. 使用 YOLO + DINO 交叉验证物体真实性
    6. 构建偏好对 (y_win: 非幻觉句子, y_lose: 幻觉句子)
    7. 选择最佳非幻觉句子作为下一步的上下文
    8. 重复步骤 2-7 直到生成完成
"""

import random
from time import time

# 数据结构导入
from model.auxiliary.dataset import DataPoint
from model.auxiliary.datastate import DataStateForBuildDataset
from model.auxiliary.global_vars import GVars

# 目标检测器导入
from model.detector.grounding_dino import DINO
from model.detector.yolo_model import YoloModel

# NLP 工具导入
from model.others.sg_parser import SGParser
from model.others.spacy_model import SpacyModel
from model.others.wordnet import WordnetModel

# 生成器工具导入
from model.utils.gen_utils import GenOutput, get_generator

# 工具函数导入
from run.utils import (  # noqa
    b_get_hallu_objects,        # 批量获取幻觉物体
    extract_obj_from_textgraphs, # 从场景图提取物体
    extract_obj_w_gt,            # 基于预定义物体列表提取
    get_finish_flag,             # 判断是否停止生成
    log_progress,                # 记录进度
    object_in_set,               # 检查物体是否在集合中
    objects_in_set,              # 检查多个物体是否在集合中
    refModel,                    # 参考模型（存储物体同义词映射等）
    resolve_corefs,              # 指代消解
    save_result,                 # 保存结果
    yolo_detect,                 # YOLO 检测
)

# ==================== 配置常量 ====================

DEBUG = True  
"""调试模式：开启时会使用 eager 模式，便于调试但速度较慢"""

HALLUCI_CONTEXT = False  
"""是否允许幻觉句子进入上下文
   False: 只选择非幻觉句子作为上下文（推荐）
   True: 允许幻觉句子进入上下文（用于对比实验）"""

CHECK_TYPE = "any"
"""物体匹配检查类型
   "any": 任一物体匹配即可
   "all": 所有物体都需匹配"""


def save_data_state(
    res_save_path: str,
    s: DataStateForBuildDataset,
    spacy: SpacyModel | None = None,
    wn: WordnetModel | None = None,
    inv_synonym_map: dict[str, list[str]] | None = None,
) -> None:
    """
    保存数据状态到文件（用于分析和调试）
    
    Args:
        res_save_path: 结果保存路径
        s: 数据状态对象，包含该图像的所有生成信息
        spacy: Spacy 模型，用于词形还原
        wn: WordNet 模型，用于同义词处理
        inv_synonym_map: 逆同义词映射表
    
    保存内容说明:
        - hard_positive: YOLO 检测到但模型未提及的物体（难正样本）
        - small_objects: 图像中面积较小的物体（可能更容易漏检）
        - edge_objects: 靠近图像边缘的物体（可能更容易产生幻觉）
    """
    # 计算难正样本：YOLO 检测到但模型从未提及的物体
    # 这些物体可以用于构建更有挑战性的训练数据
    s.hard_positive = [
        obj for obj in s.yolo_result.labels 
        if not object_in_set(obj, set(s.flat_gen_objs), spacy, wn, inv_synonym_map)
    ]
    
    # 计算小物体：面积小于图像 2% 的物体
    # 小物体更容易被模型忽略或产生幻觉
    s.small_objects = [
        obj
        for obj in s.yolo_result.labels
        if object_in_set(obj, set(s.flat_nonhallu_objs), spacy, wn, inv_synonym_map)
        and s.yolo_result.get_largest(obj)
        and (s.yolo_result.get_largest(obj)["xywhn"][2] * s.yolo_result.get_largest(obj)["xywhn"][3] < 0.02)
    ]
    
    # 计算边缘物体：距离图像边缘小于 10% 的物体
    # 边缘物体可能因为不完整显示而更容易产生幻觉
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
            < 0.1  # 如果最小距离小于 0.1，则认为物体接近边缘
        )
    ]

    # 保存完整的数据状态到 jsonl 文件
    save_result(
        res_save_path,
        {
            "image_id": s.data.image_id,           # 图像 ID
            "image_path": s.data.image_path,       # 图像路径
            "question": s.question,                 # 输入问题
            "caption": s.assistant,                 # 最终生成的完整描述
            "sentences_cnt": s.gen_sents_cnt,       # 生成的句子数量
            "hallu_objects": s.hallu_objects,       # 所有幻觉物体
            "uncertain_objects": s.uncertain_objects, # 不确定的物体
            "nonhallu_objects": s.nonhallu_objects, # 所有非幻觉物体
            "hard_positive": s.hard_positive,       # 难正样本
            "small_objects": s.small_objects,       # 小物体
            "edge_objects": s.edge_objects,         # 边缘物体
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
    构建偏好对并返回最佳句子的索引
    
    这是 SENTINEL 的核心函数之一，负责：
    1. 从候选句子中筛选出非幻觉句子和幻觉句子
    2. 构建偏好对 (y_win, y_lose)
    3. 保存偏好对到文件
    4. 返回最佳句子的索引，用于构建下一步的上下文
    
    Args:
        save_path: 保存路径
        s: 数据状态对象
        spacy: Spacy 模型
        wn: WordNet 模型
        inv_synonym_map: 逆同义词映射
    
    Returns:
        int: 最佳句子的索引，该句子将被添加到上下文中
    
    偏好对构建策略:
        - y_win: 非幻觉句子，且包含至少一个真实物体
        - y_lose: 包含至少一个幻觉物体的句子
        
    上下文选择策略（按优先级）:
        1. 成功探索的句子（包含新物体的非幻觉句子）
        2. 普通非幻觉句子
        3. 随机选择
    """

    def create_pairs(win_candidates: list[tuple[int, list[str]]], lose_candidates, pair_type: str) -> list[dict]:
        """
        创建偏好对列表
        
        Args:
            win_candidates: 正样本候选 [(句子索引, 物体列表), ...]
            lose_candidates: 负样本候选 [(句子索引, 幻觉物体列表), ...]
            pair_type: 偏好对类型标记
        
        Returns:
            偏好对字典列表，每个字典包含完整的偏好对信息
        """
        return [
            {
                "image_id": s.data.image_id,
                "image_path": s.data.image_path,
                "question": s.data.question,
                "context": s.assistant,                    # 当前上下文
                "y_win": new_sentences[win_idx],           # 正样本：非幻觉句子
                "y_lose": new_sentences[lose_idx],         # 负样本：幻觉句子
                # ===== 以下为分析用的附加信息 =====
                "nonhallu_objects": s.nonhallu_objects,    # 累计非幻觉物体
                "context_gen_objects": s.context_gen_objects,  # 上下文中的物体
                "context_gen_hallu_objects": s.context_gen_hallu_objects,  # 上下文中的幻觉物体
                "objects_of_y_win": objects,               # y_win 中的物体
                "hallu_objects_of_y_lose": hallu_objects,  # y_lose 中的幻觉物体
                "is_last_sent": s.is_finished,             # 是否是最后一句
                "type": pair_type,                         # 偏好对类型
            }
            for (win_idx, objects), (lose_idx, hallu_objects) in zip(win_candidates, lose_candidates)
        ]

    # 获取当前步骤生成的所有候选句子
    new_sentences: list[str] = s.generated_sentences[-1]
    if len(new_sentences) <= 1:
        return 0  # 只有一个句子，无法构建偏好对

    # 获取当前步骤的物体分析结果
    step_idx = s.now_step_idx
    objects_list, nonhallu_objects_list, hallu_objects_list = (  # noqa
        s.gen_objs(step_idx),        # 所有句子中的物体
        s.nonhallu_objs(step_idx),   # 非幻觉物体
        s.hallu_objs(step_idx),      # 幻觉物体
    )

    # ==================== 筛选候选句子 ====================
    
    # 非幻觉候选：包含至少一个物体，且没有幻觉物体，且物体不在不确定列表中
    nonhallu_candidates: list = [
        (i, objects)
        for i, (objects, hallu_objects) in enumerate(zip(objects_list, hallu_objects_list))
        if len(objects) >= 1                    # 至少包含一个物体
        and not hallu_objects                   # 没有幻觉物体
        and not objects_in_set(objects, s.uncertain_objects, spacy, wn, inv_synonym_map, check_type="any")
    ]
    
    # 幻觉候选：包含至少一个幻觉物体
    hallu_candidates: list = [
        (i, hallu_objects) 
        for i, hallu_objects in enumerate(hallu_objects_list) 
        if len(hallu_objects) >= 1
    ]

    # 进一步区分非幻觉候选
    # success_explore: 包含新物体（不在上下文中的物体）的非幻觉句子
    # normal_nonhallu: 只包含已有物体的非幻觉句子
    success_explore_candidates, normal_nonhallu_candidates = [], []
    for idx, objects in nonhallu_candidates:
        if not objects_in_set(objects, s.context_gen_objects, spacy, wn, inv_synonym_map, check_type=CHECK_TYPE):
            success_explore_candidates.append((idx, objects))  # 包含新物体
        else:
            normal_nonhallu_candidates.append((idx, objects))  # 只有已知物体

    # ==================== 构建偏好对 ====================
    
    # 偏好对数量取决于正负样本的最小数量
    num_pairs = min(len(normal_nonhallu_candidates), len(hallu_candidates))

    # 创建偏好对并保存
    all_results_list = create_pairs(normal_nonhallu_candidates[:num_pairs], hallu_candidates[:num_pairs], "y+")
    save_result(save_path.replace(".jsonl", "_data_pair.jsonl"), all_results_list)

    # ==================== 选择下一步的上下文 ====================
    
    if HALLUCI_CONTEXT:
        # 实验模式：允许幻觉句子进入上下文
        if hallu_candidates:
            return random.choice([idx for idx, _ in hallu_candidates])
        else:
            return random.choice(range(len(new_sentences)))
    else:
        # 正常模式：优先选择非幻觉句子
        if success_explore_candidates:
            # 优先选择包含新物体的非幻觉句子（迭代自举）
            return random.choice([idx for idx, _ in success_explore_candidates])
        elif normal_nonhallu_candidates:
            # 其次选择普通非幻觉句子
            return random.choice([i for i, _ in normal_nonhallu_candidates])
        else:
            # 最后随机选择
            return random.choice(range(len(new_sentences)))


def run_gen_dataset(datalist: list[DataPoint], batch_size: int) -> None:
    """
    核心数据生成函数
    
    这是 SENTINEL 最核心的函数，实现了完整的偏好数据生成流程：
    
    算法流程:
        1. 初始化模型（生成器、检测器、NLP工具）
        2. 批量处理图像数据
        3. 对每个 batch:
           a. YOLO 预检测
           b. MLLM 生成候选句子
           c. 指代消解
           d. 物体提取
           e. 幻觉检测
           f. 构建偏好对
           g. 选择最佳句子更新上下文
        4. 迭代直到生成完成
    
    Args:
        datalist: 待处理的数据点列表
        batch_size: 批处理大小，影响 GPU 显存使用
    
    输出文件:
        - ./results/<model_name>.jsonl: 完整分析结果
        - ./results/<model_name>_data_pair.jsonl: 偏好对数据
    """
    # ==================== 获取全局配置 ====================
    logger, save_path, model_dir, alter_device = GVars.logger, GVars.save_path, GVars.model_dir, GVars.alter_device
    
    # ==================== 初始化模型 ====================
    
    # 1. 初始化 MLLM 生成器（使用 vLLM 加速）
    generator = get_generator(use_vllm=True, debug=DEBUG)

    # 2. 初始化目标检测器
    # DINO: 开放词汇检测器，可以检测任意物体
    DINO_detector = DINO("base", model_dir=model_dir, device=alter_device, logger=logger)
    # YOLO: 快速封闭词汇检测器，80类 COCO 物体
    yolo = YoloModel("yolo11x", model_dir=model_dir, logger=logger)

    # 3. 初始化 NLP 工具
    # 场景图解析器：将句子转换为 (主语, 谓语, 宾语) 三元组
    SG_parser = SGParser(DEBUG, "base", model_dir, device=alter_device, logger=logger)
    # Spacy: 用于指代消解和词性分析
    spacy = SpacyModel(model_size="md", model_dir=model_dir, device=alter_device, logger=logger)
    # WordNet: 用于同义词处理和词形还原
    wn = WordnetModel(logger=logger)
    # 参考模型：存储 MSCOCO 物体列表和同义词映射
    ref = refModel(args=GVars.args)

    # ==================== 初始化处理状态 ====================
    
    # 正在处理的数据状态列表，长度不超过 batch_size
    data_states: list[DataStateForBuildDataset] = []
    num_of_data, finished_data_num = len(datalist), 0

    logger.info(f"Start processing {num_of_data} data points.")

    # ==================== 主循环 ====================
    # 持续处理直到所有数据完成
    while len(datalist) > 0 or len(data_states) > 0:
        start_time = time()

        # ---------- Step 1: 填充 batch ----------
        # 从待处理列表中取出数据，填充到当前 batch
        while len(data_states) < batch_size and len(datalist) > 0:
            tmp_data = datalist.pop(0)
            data_states.append(DataStateForBuildDataset(data=tmp_data))

        # ---------- Step 2: YOLO 预检测 ----------
        # 对新加入的图像进行 YOLO 检测，建立物体基准
        yolo_detect(yolo, data_states)

        # ---------- Step 3: MLLM 生成候选句子 ----------
        # 使用温度采样生成多个候选句子
        out: GenOutput = generator.gen(
            images=[s.image for s in data_states],      # 输入图像
            users=[s.question for s in data_states],    # 用户问题
            assistants=[s.assistant for s in data_states],  # 当前上下文
            do_sample=True,     # 启用采样
            n=10,               # 每个图像生成 10 个候选
            temp=0.7,           # 温度参数，控制多样性
            force_list=True,    # 强制返回列表
            single_sentence=True,  # 只生成一个句子
        )
        b_new_sents: list[list[str]] = out.outputs

        # ---------- Step 4: 判断停止条件 ----------
        # 如果大多数候选是空字符串，说明生成应该结束
        for idx, (new_sents, s) in enumerate(zip(b_new_sents, data_states)):
            b_new_sents[idx], s.is_finished = get_finish_flag(new_sents, remove_duplicates=True)
            del idx, new_sents

        # ---------- Step 5: 指代消解 ----------
        # 将句子中的代词（he, it, they 等）替换为具体物体
        context = [s.assistant for s in data_states]
        b_resolved_new_sents: list[list[str]] = resolve_corefs(spacy, b_new_sents, context, 1)
        del context

        # ---------- Step 6: 物体提取 ----------
        b_object_lists: list[list[list[str]]] = []
        for s, new_sents in zip(data_states, b_resolved_new_sents):
            # 方法1: 基于预定义物体列表的正则匹配
            object_lists: list[list[str]] = extract_obj_w_gt(
                new_sents,
                ref.valid_nouns,      # MSCOCO 物体列表
                ref.double_words,     # 双词物体映射（如 "hot dog"）
                ref.inv_syn_map,      # 同义词映射
                wn,
                force_list=True,
                return_repr=False,
            )

            # 方法2: 基于场景图解析的物体提取
            textgraphs: list[list[list[str]]] = SG_parser.pharse(new_sents, force_list=True)
            new_object_lists: list[list[str]] = extract_obj_from_textgraphs(textgraphs, spacy, wn, force_list=True)
            
            # 合并两种方法的结果
            object_lists = [objects + new_objects for objects, new_objects in zip(object_lists, new_object_lists)]

            del textgraphs
            b_object_lists.append(object_lists)

        # ---------- Step 7: 幻觉检测（核心！）----------
        # 使用 YOLO + DINO 交叉验证物体是否真实存在
        b_haluci_objects_list, b_nonhallu_objects_list = b_get_hallu_objects(
            b_object_lists,                                    # 提取的物体
            [s.nonhallu_objects for s in data_states],         # 已知非幻觉物体
            [s.hallu_objects for s in data_states],            # 已知幻觉物体
            spacy=spacy,
            wn=wn,
            images=[s.image for s in data_states],             # 图像（用于 DINO 检测）
            dino=DINO_detector,
            b_yolo_results=[s.yolo_result.labels for s in data_states] if yolo else None,  # YOLO 结果
            yolo_labels=yolo.labels if yolo else None,         # YOLO 可检测标签
            b_uncertain_objects=[s.uncertain_objects for s in data_states],  # 不确定物体
            b_detector_rejects=[s.detector_reject for s in data_states],
            inv_syn_map=ref.inv_syn_map,
        )

        # ---------- Step 8: 更新状态并构建偏好对 ----------
        for s, new_sents, object_lists, haluci_objects_list, nonhallu_objects_list in zip(
            data_states, b_resolved_new_sents, b_object_lists, b_haluci_objects_list, b_nonhallu_objects_list
        ):
            if not new_sents:
                continue
            
            # 记录本步生成的句子和物体
            s.generated_sentences.append(new_sents)
            s.generated_objects.append(object_lists)
            s.generated_hallu_objects.append(haluci_objects_list)
            s.generated_nonhallu_objects.append(nonhallu_objects_list)

            # 构建偏好对并获取最佳句子索引
            best_idx: int = maybe_build_pair(save_path, s, spacy, wn, ref.inv_syn_map)
            
            # 将最佳句子添加到上下文（用于下一步生成）
            s.app_assistant(new_sents, best_idx)

        # ---------- Step 9: 保存已完成的数据 ----------
        [save_data_state(save_path, s, spacy, wn, ref.inv_syn_map) for s in data_states if s.is_finished]

        # ---------- Step 10: 更新进度 ----------
        finished_data_num += len([s for s in data_states if s.is_finished])
        log_progress(logger, finished_data_num, num_of_data, batch_size, time() - start_time)
        
        # 移除已完成的数据状态
        data_states = [s for s in data_states if not s.is_finished]
