"""
================================================================================
generate_dataset.py - SENTINEL 数据集生成核心模块（带详细注释版本）
================================================================================

本模块是SENTINEL项目的核心，负责：
1. 迭代式地使用视觉语言模型(VLM)生成图像描述
2. 识别生成文本中的幻觉对象和真实对象
3. 构建用于DPO(Direct Preference Optimization)训练的偏好数据对

核心思想:
    SENTINEL方法的核心创新在于"句子级早期干预"：
    - 幻觉通常在描述的早期句子中产生，并会"传播"到后续句子
    - 通过在每个句子生成后立即检测并干预，可以防止幻觉的累积和传播
    - 使用YOLO和Grounding DINO的交叉验证来判断对象是否为幻觉

算法流程:
    对于每张输入图像:
    while 未结束:
        1. VLM采样生成n个候选下一句 (n=10, temp=0.7)
        2. 对候选句子进行指代消解
        3. 从句子中提取对象名词
        4. 使用YOLO+DINO验证对象是否真实存在
        5. 分类句子：幻觉句子 vs 非幻觉句子
        6. 构建偏好对：(非幻觉句子, 幻觉句子)
        7. 选择最佳非幻觉句子作为下一轮的上下文
        8. 检查是否生成结束（>50%候选为空）

输出格式:
    - <model_name>.jsonl: 每行包含一张图像的完整生成信息
    - <model_name>_data_pair.jsonl: 偏好训练对，每行一个pair

作者: SENTINEL团队
================================================================================
"""

import random
from time import time

# ============================================================================
# 模块导入
# ============================================================================

# 数据结构相关
from model.auxiliary.dataset import DataPoint
from model.auxiliary.datastate import DataStateForBuildDataset
from model.auxiliary.global_vars import GVars

# 检测器
from model.detector.grounding_dino import DINO
from model.detector.yolo_model import YoloModel

# NLP工具
from model.others.sg_parser import SGParser       # 场景图解析器
from model.others.spacy_model import SpacyModel   # Spacy NLP
from model.others.wordnet import WordnetModel     # WordNet词汇工具

# 生成器工具
from model.utils.gen_utils import GenOutput, get_generator

# 运行时工具函数
from run.utils import (
    b_get_hallu_objects,       # 批量幻觉对象检测
    extract_obj_from_textgraphs,  # 从场景图提取对象
    extract_obj_w_gt,          # 基于GT词表提取对象
    get_finish_flag,           # 判断生成是否结束
    log_progress,              # 记录进度
    object_in_set,             # 对象集合匹配
    objects_in_set,            # 多对象集合匹配
    refModel,                  # 参考模型（词表等）
    resolve_corefs,            # 指代消解
    save_result,               # 保存结果
    yolo_detect,               # YOLO检测
)

# ============================================================================
# 全局配置
# ============================================================================

DEBUG = True  
"""
调试模式开关

当 DEBUG=True 时:
    - 不使用 torch.compile() 编译模型，加快初始化
    - 使用 enforce_eager=True 模式运行vLLM
    - 适合开发和调试阶段

当 DEBUG=False 时:
    - 启用模型编译优化
    - 适合生产环境，生成速度更快
"""

HALLUCI_CONTEXT = False  
"""
是否使用含幻觉的句子添加到上下文

当 HALLUCI_CONTEXT=True 时:
    - 故意选择幻觉句子作为下一轮的上下文
    - 用于研究幻觉如何在上下文中传播
    - 仅用于实验/消融研究

当 HALLUCI_CONTEXT=False 时 (默认):
    - 优先选择非幻觉句子作为上下文
    - 这是正常的数据生成模式
"""

CHECK_TYPE = "any"
"""
对象集合检查类型

"any": 只要有一个对象在集合中就返回True
"all": 所有对象都在集合中才返回True

用于判断句子中的对象是否与上下文重复
"""


# ============================================================================
# 辅助函数
# ============================================================================

def save_data_state(
    res_save_path: str,
    s: DataStateForBuildDataset,
    spacy: SpacyModel | None = None,
    wn: WordnetModel | None = None,
    inv_synonym_map: dict[str, list[str]] | None = None,
) -> None:
    """
    保存单个数据点的处理状态到文件
    
    这个函数在一个数据点处理完成后被调用，负责：
    1. 计算额外的分析信息（困难正例、小对象、边缘对象）
    2. 将所有信息保存到JSONL文件
    
    参数:
        res_save_path: str
            结果保存路径（.jsonl格式）
        s: DataStateForBuildDataset
            包含完整处理状态的数据对象
        spacy: SpacyModel | None
            Spacy模型，用于词干提取和同义词匹配
        wn: WordnetModel | None
            WordNet模型，用于同义词查找
        inv_synonym_map: dict[str, list[str]] | None
            逆同义词映射表
    
    保存的字段:
        - image_id: 图像唯一标识
        - image_path: 图像文件路径
        - question: 输入的问题/提示
        - caption: 最终生成的完整描述
        - sentences_cnt: 生成的句子数量
        - hallu_objects: 幻觉对象列表
        - uncertain_objects: 不确定对象列表
        - nonhallu_objects: 非幻觉对象列表
        - hard_positive: 困难正例（存在但未被提及的对象）
        - small_objects: 小对象（面积<2%的对象）
        - edge_objects: 边缘对象（距边缘<10%的对象）
    """
    
    # ========== 计算困难正例 ==========
    # 困难正例定义：图像中确实存在（被YOLO检测到），但模型从未提及的对象
    # 这类对象可能是因为：
    # 1. 太小或不显眼
    # 2. 不是图像的主要内容
    # 3. 模型的注意力没有关注到
    s.hard_positive = [
        obj for obj in s.yolo_result.labels  # 遍历所有YOLO检测到的对象
        if not object_in_set(obj, set(s.flat_gen_objs), spacy, wn, inv_synonym_map)
        # 如果该对象没有出现在任何生成的文本中，则为困难正例
    ]
    
    # ========== 计算小对象 ==========
    # 小对象定义：面积小于整张图像2%的非幻觉对象
    # 小对象更容易被模型忽略或误识别
    s.small_objects = [
        obj
        for obj in s.yolo_result.labels
        # 首先确保这个对象被正确识别为非幻觉对象
        if object_in_set(obj, set(s.flat_nonhallu_objs), spacy, wn, inv_synonym_map)
        and s.yolo_result.get_largest(obj)  # 获取该类别最大的实例
        and (s.yolo_result.get_largest(obj)["xywhn"][2]  # 归一化宽度
             * s.yolo_result.get_largest(obj)["xywhn"][3]  # 归一化高度
             < 0.02)  # 面积阈值：2%
    ]
    
    # ========== 计算边缘对象 ==========
    # 边缘对象定义：距离图像边缘最近距离小于10%的非幻觉对象
    # 边缘对象可能被截断或难以完整识别
    s.edge_objects = [
        obj
        for obj in s.yolo_result.labels
        if object_in_set(obj, set(s.flat_nonhallu_objs), spacy, wn, inv_synonym_map)
        if s.yolo_result.get_farthest_to_edge(obj)  # 获取离边缘最远的实例
        and (
            min(
                # 计算到四个边的距离，取最小值
                s.yolo_result.get_farthest_to_edge(obj)["xywhn"][0],      # 到左边的距离
                1 - s.yolo_result.get_farthest_to_edge(obj)["xywhn"][0],  # 到右边的距离
                s.yolo_result.get_farthest_to_edge(obj)["xywhn"][1],      # 到上边的距离
                1 - s.yolo_result.get_farthest_to_edge(obj)["xywhn"][1],  # 到下边的距离
            )
            < 0.1  # 距离阈值：10%
        )
    ]

    # ========== 保存结果 ==========
    save_result(
        res_save_path,
        {
            "image_id": s.data.image_id,
            "image_path": s.data.image_path,
            "question": s.question,
            "caption": s.assistant,           # 最终生成的完整描述
            "sentences_cnt": s.gen_sents_cnt, # 总句子数
            "hallu_objects": s.hallu_objects, # 所有幻觉对象
            "uncertain_objects": s.uncertain_objects,
            "nonhallu_objects": s.nonhallu_objects,
            "hard_positive": s.hard_positive, # 困难正例
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
    构建偏好数据对并返回最佳句子的索引
    
    这是SENTINEL的核心函数之一，负责：
    1. 分析当前步骤的所有候选句子
    2. 识别哪些是幻觉句子，哪些是非幻觉句子
    3. 构建(非幻觉, 幻觉)偏好对用于DPO训练
    4. 选择最佳句子作为下一轮生成的上下文
    
    参数:
        save_path: str
            偏好对保存路径
        s: DataStateForBuildDataset
            当前数据的处理状态
        spacy: SpacyModel
            Spacy模型
        wn: WordnetModel
            WordNet模型
        inv_synonym_map: dict[str, list[str]] | None
            逆同义词映射
    
    返回:
        int: 最佳句子在候选列表中的索引
        
    偏好对格式:
        {
            "image_id": 图像ID,
            "image_path": 图像路径,
            "question": 问题,
            "context": 当前上下文（之前选择的句子累积）,
            "y_win": 非幻觉句子（应该被模型偏好）,
            "y_lose": 幻觉句子（应该被模型拒绝）,
            "type": 偏好对类型标记
        }
    """

    def create_pairs(
        win_candidates: list[tuple[int, list[str]]], 
        lose_candidates: list[tuple[int, list[str]]], 
        pair_type: str
    ) -> list[dict]:
        """
        内部辅助函数：创建偏好数据对
        
        参数:
            win_candidates: 胜出候选列表，每个元素是(索引, 对象列表)
            lose_candidates: 失败候选列表，每个元素是(索引, 幻觉对象列表)
            pair_type: 偏好对类型标记（如"y+"表示正常偏好对）
        
        返回:
            偏好对字典列表
        """
        return [
            {
                # 基本信息
                "image_id": s.data.image_id,
                "image_path": s.data.image_path,
                "question": s.data.question,
                
                # 核心字段：上下文和偏好对
                "context": s.assistant,           # 当前累积的上下文
                "y_win": new_sentences[win_idx],  # 胜出句子（非幻觉）
                "y_lose": new_sentences[lose_idx], # 失败句子（幻觉）
                
                # 附加分析信息（用于后续分析和调试）
                "nonhallu_objects": s.nonhallu_objects,
                "context_gen_objects": s.context_gen_objects,
                "context_gen_hallu_objects": s.context_gen_hallu_objects,
                "objects_of_y_win": objects,        # y_win中的对象
                "hallu_objects_of_y_lose": hallu_objects,  # y_lose中的幻觉对象
                "is_last_sent": s.is_finished,      # 是否是最后一个句子
                "type": pair_type,                  # 偏好对类型
            }
            for (win_idx, objects), (lose_idx, hallu_objects) 
            in zip(win_candidates, lose_candidates)
        ]

    # ========== 获取当前步骤的候选句子 ==========
    new_sentences: list[str] = s.generated_sentences[-1]  # 最后一步生成的所有候选
    
    # 如果只有一个候选句子，无法构建偏好对
    if len(new_sentences) <= 1:
        return 0

    step_idx = s.now_step_idx  # 当前步骤索引
    
    # 获取当前步骤所有候选句子的对象分析结果
    objects_list = s.gen_objs(step_idx)           # 每个候选的所有对象
    nonhallu_objects_list = s.nonhallu_objs(step_idx)  # 每个候选的非幻觉对象
    hallu_objects_list = s.hallu_objs(step_idx)        # 每个候选的幻觉对象

    # ========== 分类候选句子 ==========
    
    # 筛选非幻觉候选句子
    # 条件：
    # 1. 至少包含一个对象 (len(objects) >= 1)
    # 2. 不包含任何幻觉对象 (not hallu_objects)
    # 3. 对象不在"不确定"列表中
    nonhallu_candidates: list = [
        (i, objects)
        for i, (objects, hallu_objects) in enumerate(zip(objects_list, hallu_objects_list))
        if len(objects) >= 1  # 至少有一个对象
        and not hallu_objects  # 无幻觉对象
        and not objects_in_set(  # 对象不在不确定列表中
            objects, s.uncertain_objects, spacy, wn, inv_synonym_map, check_type="any"
        )
    ]
    
    # 筛选幻觉候选句子
    # 条件：包含至少一个幻觉对象
    hallu_candidates: list = [
        (i, hallu_objects) 
        for i, hallu_objects in enumerate(hallu_objects_list) 
        if len(hallu_objects) >= 1  # 至少有一个幻觉对象
    ]

    # ========== 进一步分类非幻觉句子 ==========
    # 将非幻觉句子分为两类：
    # 1. 探索型：包含上下文中从未出现的新对象
    # 2. 重复型：只提及上下文中已有的对象
    
    success_explore_candidates = []  # 探索型候选
    normal_nonhallu_candidates = []  # 重复型候选
    
    for idx, objects in nonhallu_candidates:
        # 检查对象是否都已在上下文中出现过
        if not objects_in_set(
            objects, s.context_gen_objects, spacy, wn, inv_synonym_map, check_type=CHECK_TYPE
        ):
            # 包含新对象 -> 探索型
            success_explore_candidates.append((idx, objects))
        else:
            # 只有旧对象 -> 重复型
            normal_nonhallu_candidates.append((idx, objects))

    # ========== 构建偏好对 ==========
    # 配对数量取两者的最小值
    num_pairs = min(len(normal_nonhallu_candidates), len(hallu_candidates))
    
    # 创建偏好对：非幻觉句子 vs 幻觉句子
    all_results_list = create_pairs(
        normal_nonhallu_candidates[:num_pairs],  # 胜出候选
        hallu_candidates[:num_pairs],            # 失败候选
        "y+"  # 类型标记
    )

    # 保存偏好对到文件
    save_result(save_path.replace(".jsonl", "_data_pair.jsonl"), all_results_list)

    # ========== 选择最佳句子用于下一轮生成 ==========
    if HALLUCI_CONTEXT:
        # 实验模式：故意选择幻觉句子（用于研究幻觉传播）
        if hallu_candidates:
            return random.choice([idx for idx, _ in hallu_candidates])
        else:
            return random.choice(range(len(new_sentences)))
    else:
        # 正常模式：优先选择探索型，其次重复型，最后随机
        if success_explore_candidates:
            # 优先选择探索新对象的句子
            # 这样可以增加上下文的多样性和覆盖面
            return random.choice([idx for idx, _ in success_explore_candidates])
        elif normal_nonhallu_candidates:
            # 其次选择非幻觉句子
            return random.choice([i for i, _ in normal_nonhallu_candidates])
        else:
            # 如果都没有，随机选择
            return random.choice(range(len(new_sentences)))


# ============================================================================
# 主函数
# ============================================================================

def run_gen_dataset(datalist: list[DataPoint], batch_size: int) -> None:
    """
    主数据集生成函数
    
    这是整个数据生成流程的入口函数，负责：
    1. 初始化所有需要的模型（生成器、检测器、NLP工具）
    2. 管理批处理循环
    3. 协调各个处理步骤
    
    参数:
        datalist: list[DataPoint]
            待处理的数据点列表，每个数据点包含图像路径和问题
        batch_size: int
            批处理大小，同时处理的数据点数量
    
    处理流程概述:
        while 还有未处理的数据:
            1. 装载数据到状态列表
            2. YOLO预检测
            3. VLM生成候选句子
            4. 指代消解
            5. 对象提取
            6. 幻觉判断
            7. 构建偏好对
            8. 更新上下文
            9. 检查终止条件
    """
    
    # ========== 获取全局配置 ==========
    logger = GVars.logger          # 日志记录器
    save_path = GVars.save_path    # 结果保存路径
    model_dir = GVars.model_dir    # 模型缓存目录
    alter_device = GVars.alter_device  # 辅助模型使用的设备
    
    # ========== 初始化生成器（VLM模型） ==========
    # 根据命令行参数选择对应的生成器模型
    # 支持：LLaVA v1.5/1.6, Qwen2-VL, Qwen2.5-VL
    generator = get_generator(use_vllm=True, debug=DEBUG)
    
    # ========== 初始化目标检测器 ==========
    
    # Grounding DINO：开放词汇检测器
    # 可以根据任意文本描述检测对象
    # size="base" 表示使用基础模型（233M参数）
    DINO_detector = DINO(
        "base", 
        model_dir=model_dir, 
        device=alter_device,  # 使用备用GPU避免与生成器抢占显存
        logger=logger
    )
    
    # YOLO：封闭词汇检测器
    # 高精度检测80类COCO对象
    # yolo11x 是最大最准确的版本
    yolo = YoloModel(
        "yolo11x", 
        model_dir=model_dir, 
        logger=logger
    )

    # ========== 初始化NLP工具 ==========
    
    # 场景图解析器：将句子解析为(主语,谓语,宾语)三元组
    # 基于T5模型，用于结构化地提取对象和关系
    SG_parser = SGParser(
        DEBUG, 
        "base",  # T5-base模型
        model_dir, 
        device=alter_device, 
        logger=logger
    )
    
    # Spacy模型：通用NLP工具
    # 用于词性标注、词干提取、指代消解等
    # model_size="md" 是中等大小模型，平衡速度和准确性
    spacy = SpacyModel(
        model_size="md", 
        model_dir=model_dir, 
        device=alter_device, 
        logger=logger
    )
    
    # WordNet模型：词汇知识库
    # 用于同义词查找、词干提取等
    wn = WordnetModel(logger=logger)
    
    # 参考模型：包含有效名词列表和同义词映射
    ref = refModel(args=GVars.args)

    # ========== 初始化处理状态 ==========
    
    # 当前正在处理的数据状态列表
    # 长度动态变化，最多保持 batch_size 个活跃状态
    data_states: list[DataStateForBuildDataset] = []
    
    # 统计信息
    num_of_data = len(datalist)  # 总数据量
    finished_data_num = 0        # 已完成数量

    logger.info(f"Start processing {num_of_data} data points.")

    # ========== 主处理循环 ==========
    # 条件：还有待处理数据 或 还有未完成的活跃状态
    while len(datalist) > 0 or len(data_states) > 0:
        start_time = time()  # 记录本轮开始时间

        # ---------- 步骤1: 装载数据 ----------
        # 保持 data_states 中有 batch_size 个活跃状态
        while len(data_states) < batch_size and len(datalist) > 0:
            tmp_data = datalist.pop(0)  # 从队列头部取出
            # 创建新的处理状态对象
            data_states.append(DataStateForBuildDataset(data=tmp_data))

        # ---------- 步骤2: YOLO预检测 ----------
        # 对新加入的图像执行YOLO检测
        # 每张图像只检测一次，结果会被缓存
        yolo_detect(yolo, data_states)

        # ---------- 步骤3: VLM生成候选句子 ----------
        # 使用视觉语言模型生成下一个句子的多个候选
        out: GenOutput = generator.gen(
            images=[s.image for s in data_states],        # 批量图像
            users=[s.question for s in data_states],      # 用户问题
            assistants=[s.assistant for s in data_states], # 当前上下文
            do_sample=True,      # 启用采样（而非贪婪解码）
            n=10,                # 每个样本生成10个候选
            temp=0.7,            # 采样温度，控制多样性
            force_list=True,     # 强制返回列表格式
            single_sentence=True, # 每次只生成一个句子
        )
        
        # 提取生成结果
        # 格式：b_new_sents[样本索引][候选索引] = 句子字符串
        b_new_sents: list[list[str]] = out.outputs

        # ---------- 步骤4: 检查生成是否结束 ----------
        # 判断标准：如果超过50%的候选是空的，认为生成结束
        for idx, (new_sents, s) in enumerate(zip(b_new_sents, data_states)):
            # get_finish_flag 返回 (过滤后的句子列表, 是否结束)
            b_new_sents[idx], s.is_finished = get_finish_flag(
                new_sents, 
                remove_duplicates=True  # 去除重复句子
            )

        # ---------- 步骤5: 指代消解 ----------
        # 将代词替换为其指代的具体名词
        # 例如："He is eating." -> "The man is eating."
        # 这对于准确提取对象至关重要
        context = [s.assistant for s in data_states]  # 上下文列表
        b_resolved_new_sents: list[list[str]] = resolve_corefs(
            spacy, 
            b_new_sents,  # 待消解的句子
            context,      # 上下文
            1,            # 回溯1个句子
        )

        # ---------- 步骤6: 对象提取 ----------
        # 使用两种互补的方法从句子中提取对象
        b_object_lists: list[list[list[str]]] = []
        
        for s, new_sents in zip(data_states, b_resolved_new_sents):
            # 方法1: 基于MSCOCO词表的对象提取
            # 直接匹配句子中出现的COCO对象名词
            object_lists: list[list[str]] = extract_obj_w_gt(
                new_sents,
                ref.valid_nouns,     # 有效名词列表（COCO对象）
                ref.double_words,    # 双词短语映射（如"traffic light"）
                ref.inv_syn_map,     # 同义词映射
                wn,
                force_list=True,
                return_repr=False,   # 返回原始词而非代表词
            )

            # 方法2: 基于场景图解析的对象提取
            # 使用T5模型将句子解析为结构化三元组
            # 然后从三元组中提取对象
            textgraphs: list[list[list[str]]] = SG_parser.pharse(
                new_sents, 
                force_list=True
            )
            new_object_lists: list[list[str]] = extract_obj_from_textgraphs(
                textgraphs, 
                spacy, 
                wn, 
                force_list=True
            )
            
            # 合并两种方法的结果
            object_lists = [
                objects + new_objects 
                for objects, new_objects in zip(object_lists, new_object_lists)
            ]

            b_object_lists.append(object_lists)

        # ---------- 步骤7: 幻觉判断（核心步骤） ----------
        # 使用YOLO和DINO交叉验证判断对象是否真实存在
        # 判断逻辑：
        #   - YOLO认可 AND DINO认可 -> 非幻觉对象
        #   - YOLO不认可 AND DINO不认可 -> 幻觉对象
        #   - 其他情况 -> 不确定对象
        b_haluci_objects_list, b_nonhallu_objects_list = b_get_hallu_objects(
            b_object_lists,                                    # 所有对象
            [s.nonhallu_objects for s in data_states],        # 已知非幻觉对象
            [s.hallu_objects for s in data_states],           # 已知幻觉对象
            spacy=spacy,
            wn=wn,
            images=[s.image for s in data_states],            # 图像列表
            dino=DINO_detector,                               # DINO检测器
            b_yolo_results=[s.yolo_result.labels for s in data_states] if yolo else None,
            yolo_labels=yolo.labels if yolo else None,        # YOLO支持的标签
            b_uncertain_objects=[s.uncertain_objects for s in data_states],
            b_detector_rejects=[s.detector_reject for s in data_states],
            inv_syn_map=ref.inv_syn_map,
        )

        # ---------- 步骤8: 更新状态并构建偏好对 ----------
        for s, new_sents, object_lists, haluci_objects_list, nonhallu_objects_list in zip(
            data_states, 
            b_resolved_new_sents, 
            b_object_lists, 
            b_haluci_objects_list, 
            b_nonhallu_objects_list
        ):
            # 跳过空句子列表
            if not new_sents:
                continue
                
            # 记录本步骤的生成结果
            s.generated_sentences.append(new_sents)
            s.generated_objects.append(object_lists)
            s.generated_hallu_objects.append(haluci_objects_list)
            s.generated_nonhallu_objects.append(nonhallu_objects_list)

            # 构建偏好对并选择最佳句子
            best_idx: int = maybe_build_pair(save_path, s, spacy, wn, ref.inv_syn_map)
            
            # 将最佳句子追加到上下文
            # 这会更新 s.assistant，用于下一轮生成
            s.app_assistant(new_sents, best_idx)

        # ---------- 步骤9: 保存并清理已完成的状态 ----------
        # 对于已完成的数据点，保存最终结果
        [save_data_state(save_path, s, spacy, wn, ref.inv_syn_map) 
         for s in data_states if s.is_finished]

        # 更新统计信息
        finished_data_num += len([s for s in data_states if s.is_finished])
        
        # 记录进度
        log_progress(logger, finished_data_num, num_of_data, batch_size, time() - start_time)
        
        # 只保留未完成的状态，继续下一轮迭代
        data_states = [s for s in data_states if not s.is_finished]


# ============================================================================
# 模块入口
# ============================================================================

if __name__ == "__main__":
    print("Please run main.py")
    exit(0)
