"""
SENTINEL 数据状态管理模块
=========================

本模块定义了用于跟踪数据处理状态的数据类。

主要类:
    - DataState: 基础数据状态类
    - DataStateForBuildDataset: 扩展的数据状态类，用于偏好数据构建

在迭代生成过程中，每个图像对应一个 DataStateForBuildDataset 实例，
用于跟踪该图像的所有生成历史、物体检测结果和累积上下文。
"""

from dataclasses import dataclass, field

from PIL import Image

from model.detector.yolo_model import YoloResult

from .dataset import DataPoint


@dataclass
class DataState:
    """
    基础数据状态类
    
    用于跟踪单个图像处理过程中的基本状态信息。
    
    Attributes:
        data: 原始数据点
        image_path: 图像文件路径
        image: PIL 图像对象
        question: 输入问题
        is_finished: 是否完成生成
        assistant: 当前累积的生成文本（上下文）
        nonhallu_objects: 已确认的非幻觉物体列表
        uncertain_objects: 不确定的物体列表
        hallu_objects: 已确认的幻觉物体列表
    
    物体分类说明:
        在检测过程中，物体会被分为三类：
        - nonhallu: YOLO 和 DINO 都检测到
        - hallu: YOLO 和 DINO 都未检测到
        - uncertain: 两个检测器结果不一致
    """
    
    data: DataPoint
    """原始数据点"""
    
    image_path: str = field(init=False)
    """图像文件路径"""
    
    image: Image.Image = field(init=False)
    """加载的 PIL 图像对象"""
    
    question: str = field(init=False)
    """输入问题（如 "Describe this image."）"""
    
    is_finished: bool = False
    """是否完成生成（当大多数采样为空时结束）"""
    
    assistant: str = ""
    """当前累积的生成文本，作为后续生成的上下文"""
    
    # 物体分类缓存
    nonhallu_objects: list[str] = field(default_factory=list)
    """非幻觉物体列表（检测器确认存在）"""
    
    uncertain_objects: list[str] = field(default_factory=list)
    """不确定物体列表（检测器结果不一致）"""
    
    hallu_objects: list[str] = field(default_factory=list)
    """幻觉物体列表（检测器确认不存在）"""

    def __post_init__(self):
        """
        初始化后处理：加载图像并提取关键信息
        """
        from run.utils import open_images

        self.image_path = self.data.image_path
        self.image = open_images(self.image_path)
        self.question = self.data.question


@dataclass
class DataStateForBuildDataset(DataState):
    """
    扩展的数据状态类，用于偏好数据集构建
    
    继承自 DataState，添加了更详细的生成历史跟踪功能。
    在迭代生成过程中，记录每一步的所有候选句子和物体分析结果。
    
    数据结构说明:
        generated_sentences[i][j]: 第 i 步生成的第 j 个候选句子
        generated_objects[i][j][k]: 第 i 步第 j 个句子中的第 k 个物体
    
    Example:
        >>> state = DataStateForBuildDataset(data=data_point)
        >>> 
        >>> # 第一步生成
        >>> state.generated_sentences.append(["sent1", "sent2", ..., "sent10"])
        >>> state.generated_objects.append([["obj1", "obj2"], ["obj3"], ...])
        >>> 
        >>> # 选择最佳句子更新上下文
        >>> state.app_assistant(sentences, best_idx=3)
        >>> print(state.assistant)  # "The selected sentence."
    """
    
    # ==================== YOLO 检测结果 ====================
    
    yolo_detected: bool = False
    """是否已对该图像进行 YOLO 检测"""
    
    yolo_result: YoloResult = None
    """YOLO 检测结果对象"""

    # ==================== 检测器拒绝记录 ====================
    
    detector_reject: dict[str, list[str]] = field(init=False)
    """记录哪个检测器拒绝了哪些物体
       格式: {"dino": ["obj1", "obj2"], "yolo": ["obj3"]}"""
    
    # ==================== 生成历史记录 ====================
    
    generated_sentences: list[list[str]] = field(default_factory=list)
    """每步生成的候选句子列表
       generated_sentences[step][candidate_idx] = "sentence text" """
    
    generated_assistents: list[str] = field(default_factory=list)
    """每步结束后的累积上下文
       generated_assistents[step] = "context after step" """

    generated_objects: list[list[list[str]]] = field(default_factory=list)
    """每步每个候选句子中提取的物体
       generated_objects[step][candidate_idx] = ["obj1", "obj2"] """
    
    generated_hallu_objects: list[list[list[str]]] = field(default_factory=list)
    """每步每个候选句子中的幻觉物体"""
    
    generated_nonhallu_objects: list[list[list[str]]] = field(default_factory=list)
    """每步每个候选句子中的非幻觉物体"""

    # ==================== 上下文物体追踪 ====================
    
    assistant_objects: list[str] = field(default_factory=list)
    """当前上下文中包含的所有物体"""
    
    assistant_hallu_objects: list[str] = field(default_factory=list)
    """当前上下文中的幻觉物体"""
    
    assistant_nonhallu_objects: list[str] = field(default_factory=list)
    """当前上下文中的非幻觉物体"""

    # ==================== 其他信息 ====================
    
    ground_truth: bool = field(init=False)
    """POPE 数据集的标签（如果有）"""

    gt_objects: list[str] = field(default_factory=list)
    """Ground truth 物体列表（用于调试）"""

    # ==================== 难样本分析 ====================
    
    hard_positive: list[str] = field(default_factory=list)
    """难正样本：YOLO 检测到但模型未提及的物体"""
    
    small_objects: list[str] = field(default_factory=list)
    """小物体：面积小于 2% 的物体"""
    
    edge_objects: list[str] = field(default_factory=list)
    """边缘物体：距离边缘小于 10% 的物体"""

    # ==================== 自然上下文（实验用）====================
    
    nature_context: str = None
    """贪婪生成的自然上下文（用于对比实验）"""
    
    nature_objects: list[list[str]] = field(default_factory=list)
    nature_hallu_objects: list[list[str]] = field(default_factory=list)
    nature_nonhallu_objects: list[list[str]] = field(default_factory=list)

    def __post_init__(self):
        """
        初始化后处理
        
        - 调用父类初始化
        - 初始化生成历史（添加空字符串作为第一步的初始上下文）
        - 初始化检测器拒绝记录
        - 提取 ground truth 标签（如果有）
        """
        super().__post_init__()
        self.generated_assistents.append("")  # 第一步的初始上下文为空
        self.detector_reject = {"dino": [], "yolo": []}

        if "ground_truth" in self.data.attributes:
            self.ground_truth = self.data.attributes["ground_truth"]

    def app_assistant(self, new_sents: list[str], idx: int) -> None:
        """
        将选中的句子添加到上下文
        
        这是迭代生成的关键函数，将最佳候选句子追加到上下文中，
        并同步更新相关的物体列表。
        
        Args:
            new_sents: 当前步骤生成的所有候选句子
            idx: 选中的句子索引
        
        Example:
            >>> state.app_assistant(["sent1", "sent2", "sent3"], idx=1)
            >>> print(state.assistant)  # "Previous context. sent2"
        """
        # 拼接上下文
        self.assistant = self.assistant + " " + new_sents[idx] if self.assistant else new_sents[idx]
        self.generated_assistents.append(self.assistant)

        # 同步更新上下文中的物体列表
        if len(self.generated_objects) == self.gen_sents_cnt:
            self.assistant_objects.extend(self.generated_objects[-1][idx])
        if len(self.generated_hallu_objects) == self.gen_sents_cnt:
            self.assistant_hallu_objects.extend(self.generated_hallu_objects[-1][idx])
        if len(self.generated_nonhallu_objects) == self.gen_sents_cnt:
            self.assistant_nonhallu_objects.extend(self.generated_nonhallu_objects[-1][idx])

    # ==================== 状态属性 ====================

    @property
    def gen_sents_cnt(self) -> int:
        """返回已生成的步数（句子组数）"""
        return len(self.generated_sentences)

    @property
    def now_step_idx(self) -> int:
        """返回当前步骤的索引（0-based）"""
        return self.gen_sents_cnt - 1

    @property
    def is_in_the_first_step(self) -> bool:
        """判断是否处于第一步生成"""
        return self.gen_sents_cnt <= 1

    @property
    def context_gen_objects(self) -> set[str]:
        """返回当前上下文中包含的所有物体（去重）"""
        return set(self.assistant_objects)

    @property
    def context_gen_hallu_objects(self) -> set[str]:
        """返回当前上下文中的幻觉物体（去重）"""
        return set(self.assistant_hallu_objects)

    # ==================== 访问器方法 ====================

    def is_last_sent(self, index: int) -> bool:
        """判断给定索引是否是最后一步"""
        return index == self.gen_sents_cnt - 1

    def gen_sents(self, index: int) -> list[str]:
        """获取第 index 步生成的所有候选句子"""
        return self.generated_sentences[index]

    def assist(self, index: int) -> str:
        """获取第 index 步结束后的上下文"""
        return self.generated_assistents[index]

    # ==================== 物体列表访问器 ====================

    @property
    def flat_gen_objs(self) -> list[str]:
        """返回所有步骤生成的所有物体（未去重，展平列表）"""
        return [obj for objs in self.generated_objects for obj_list in objs for obj in obj_list]

    def gen_objs(self, index: int) -> list[list[str]]:
        """获取第 index 步每个候选句子中的物体列表"""
        return self.generated_objects[index]

    @property
    def flat_hallu_objs(self) -> list[str]:
        """返回所有步骤的所有幻觉物体（未去重）"""
        return [obj for objs in self.generated_hallu_objects for obj_list in objs for obj in obj_list]

    def hallu_objs(self, index: int) -> list[list[str]]:
        """获取第 index 步每个候选句子中的幻觉物体列表"""
        return self.generated_hallu_objects[index]

    @property
    def flat_nonhallu_objs(self) -> list[str]:
        """返回所有步骤的所有非幻觉物体（未去重）"""
        return [obj for objs in self.generated_nonhallu_objects for obj_list in objs for obj in obj_list]

    def nonhallu_objs(self, index: int) -> list[list[str]]:
        """获取第 index 步每个候选句子中的非幻觉物体列表"""
        return self.generated_nonhallu_objects[index]

    # ==================== 自然上下文相关（实验用）====================

    @property
    def next_step_assist_from_nature_context(self) -> str:
        """从自然上下文中获取下一步的句子"""
        sent = self.nature_context.split(".")[self.now_step_idx] + "."
        return sent.lstrip(" ")

    def get_step_assist_from_nature_context(self, index: int) -> str:
        """从自然上下文中获取第 index 步的句子"""
        sent = self.nature_context.split(".")[index] + "."
        return sent.lstrip(" ") if sent != "." else ""
