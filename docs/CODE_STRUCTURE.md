# SENTINEL 代码结构说明

本文档详细说明项目的代码结构、模块职责和核心算法。

---

## 目录

- [1. 核心流程](#1-核心流程)
- [2. 模块说明](#2-模块说明)
- [3. 关键类说明](#3-关键类说明)
- [4. 数据流说明](#4-数据流说明)
- [5. 算法细节](#5-算法细节)

---

## 1. 核心流程

### 1.1 数据生成流程

```
main.py
    │
    ├── 1. 加载环境变量 (.env)
    │
    ├── 2. 初始化全局变量 (GVars.init())
    │   ├── 解析命令行参数
    │   ├── 配置日志系统
    │   ├── 设置模型路径
    │   └── 配置 GPU 设备
    │
    └── 3. 运行数据生成 (run.run())
        │
        ├── 3.1 加载数据集 (DataSet)
        ├── 3.2 过滤已处理数据 (断点续传)
        │
        └── 3.3 核心生成循环 (run_gen_dataset)
            │
            ├── 初始化模型
            │   ├── MLLM 生成器 (LLaVA/Qwen2-VL)
            │   ├── YOLO 检测器
            │   ├── DINO 检测器
            │   └── NLP 工具 (Spacy, SGParser, WordNet)
            │
            └── 对每个 batch:
                │
                ├── Step 1: YOLO 预检测
                │   └── 获取图像中的物体基准
                │
                ├── Step 2: MLLM 生成候选句子
                │   ├── 输入: 图像 + 问题 + 上下文
                │   ├── 温度采样 (n=10, temp=0.7)
                │   └── 输出: 10 个候选句子
                │
                ├── Step 3: 指代消解
                │   └── "he" → "the man"
                │
                ├── Step 4: 物体提取
                │   ├── 方法1: 正则匹配 MSCOCO 物体
                │   └── 方法2: 场景图解析
                │
                ├── Step 5: 幻觉检测 ⭐
                │   ├── YOLO 验证
                │   ├── DINO 验证
                │   └── 交叉验证判定
                │
                ├── Step 6: 构建偏好对
                │   ├── y_win: 非幻觉句子
                │   └── y_lose: 幻觉句子
                │
                └── Step 7: 更新上下文
                    └── 选择最佳非幻觉句子
```

### 1.2 训练流程

```
train/models/dpo_llava.py
    │
    ├── 1. 解析训练参数
    │
    ├── 2. 设置 LLaVA 模型
    │   ├── 加载预训练权重
    │   ├── 添加 LoRA 适配器
    │   └── 配置视觉编码器
    │
    ├── 3. 准备数据集
    │   ├── 加载偏好对数据
    │   └── 数据预处理
    │
    └── 4. DPO 训练
        ├── 配置 DPO 超参数 (β=0.1)
        ├── 初始化 Trainer
        └── 开始训练
```

---

## 2. 模块说明

### 2.1 模型模块 (`model/`)

```
model/
├── auxiliary/              # 辅助工具
│   ├── dataset.py         # 数据集加载和管理
│   ├── datastate.py       # 数据状态跟踪
│   └── global_vars.py     # 全局变量管理
│
├── detector/               # 目标检测器
│   ├── grounding_dino.py  # Grounding DINO 开放词汇检测
│   └── yolo_model.py      # YOLO 封闭词汇检测
│
├── generator/              # MLLM 生成器
│   ├── llava.py           # LLaVA v1.5/v1.6 模型
│   ├── qwen2_vl.py        # Qwen2-VL 模型
│   └── qwen2_5_vl.py      # Qwen2.5-VL 模型
│
├── others/                 # NLP 工具
│   ├── sg_parser.py       # 场景图解析器
│   ├── spacy_model.py     # Spacy NLP
│   └── wordnet.py         # WordNet 同义词
│
└── utils/                  # 工具函数
    ├── gen_utils.py       # 生成相关工具
    └── utils.py           # 通用工具
```

### 2.2 运行模块 (`run/`)

```
run/
├── run.py                 # 主运行逻辑
├── generate_dataset.py    # 核心数据生成
├── object_utils.py        # 物体处理工具
└── utils.py               # 运行时工具函数
```

### 2.3 训练模块 (`train/`)

```
train/
├── models/
│   ├── dpo_llava.py      # LLaVA DPO 训练脚本
│   ├── dpo_llava.sh      # 训练启动脚本
│   └── llava_utils/      # LLaVA 训练工具
│       ├── arguments.py  # 参数定义
│       ├── callback.py   # 训练回调
│       ├── data.py       # 数据处理
│       └── llava_trainer.py  # 自定义 Trainer
│
└── accelerate_configs/    # 分布式训练配置
    ├── zero2.json        # DeepSpeed ZeRO-2
    ├── zero3.json        # DeepSpeed ZeRO-3
    └── zero3_offload.json
```

---

## 3. 关键类说明

### 3.1 DataPoint

```python
@dataclass
class DataPoint:
    """表示单个数据点"""
    image_id: str       # 图像唯一标识
    image_path: str     # 图像文件路径
    question: str       # 输入问题
    attributes: dict    # 额外属性
```

### 3.2 DataStateForBuildDataset

```python
@dataclass
class DataStateForBuildDataset:
    """跟踪单个图像的生成状态"""
    
    # 基本信息
    data: DataPoint
    image: Image.Image
    question: str
    
    # YOLO 检测结果
    yolo_result: YoloResult
    
    # 生成历史
    generated_sentences: list[list[str]]    # 每步生成的候选句子
    generated_objects: list[list[list[str]]]  # 提取的物体
    generated_hallu_objects: list[...]       # 幻觉物体
    generated_nonhallu_objects: list[...]    # 非幻觉物体
    
    # 当前上下文
    assistant: str                 # 累积的描述文本
    nonhallu_objects: list[str]    # 已确认的非幻觉物体
    hallu_objects: list[str]       # 已确认的幻觉物体
    uncertain_objects: list[str]   # 不确定的物体
    
    # 状态标志
    is_finished: bool              # 是否完成生成
```

### 3.3 GenOutput

```python
@dataclass
class GenOutput:
    """生成器输出"""
    outputs: list[str] | list[list[str]]  # 生成的文本
    generated_ids: torch.Tensor           # token IDs
    true_gen_length: list[int]            # 实际生成长度
    log_probs: list[...]                  # 对数概率（可选）
```

### 3.4 YoloResult

```python
@dataclass
class YoloResult:
    """YOLO 检测结果"""
    result: dict[str, list[dict]]
    # 格式: {"person": [{"conf": 0.95, "xywhn": [0.5, 0.5, 0.3, 0.4]}, ...]}
    
    @property
    def labels(self) -> list[str]:
        """检测到的物体标签"""
    
    def get_largest(self, label: str) -> dict:
        """获取最大的物体"""
    
    def get_smallest(self, label: str) -> dict:
        """获取最小的物体"""
```

---

## 4. 数据流说明

### 4.1 输入数据

```
dataset/image_data.jsonl
    │
    │  {"image_id": "123", "image_path": "/path/to/img.jpg", "question": "Describe..."}
    │
    ▼
DataSet.data: list[DataPoint]
```

### 4.2 生成过程数据

```
DataPoint
    │
    ▼
DataStateForBuildDataset (每个图像一个状态对象)
    │
    ├── 迭代生成
    │   ├── generated_sentences[0] = ["句子1a", "句子1b", ..., "句子1j"]  # 第1步，10个候选
    │   ├── generated_sentences[1] = ["句子2a", "句子2b", ..., "句子2k"]  # 第2步
    │   └── ...
    │
    └── 累积状态
        ├── assistant = "句子1. 句子2. 句子3."  # 选中的句子拼接
        └── nonhallu_objects = ["person", "table", "cup"]  # 累积的非幻觉物体
```

### 4.3 输出数据

```
./results/<model_name>.jsonl  (分析数据)
    │
    │  {"image_id": "123", "caption": "...", "hallu_objects": [...], ...}
    │
./results/<model_name>_data_pair.jsonl  (偏好对数据)
    │
    │  {"context": "...", "y_win": "...", "y_lose": "...", ...}
    │
    ▼
训练数据 (经转换后)
```

---

## 5. 算法细节

### 5.1 幻觉检测算法

```python
def detect_hallucination(object: str, image: Image) -> str:
    """
    判断物体是否为幻觉
    
    Returns:
        "nonhallu" | "hallu" | "uncertain"
    """
    # Step 1: YOLO 验证
    if object in YOLO_LABELS:
        yolo_detected = object in yolo.predict(image).labels
    else:
        yolo_detected = True  # YOLO 不能检测的物体，视为通过
    
    # Step 2: DINO 验证
    dino_result = dino.detect(image, f"{object}.")
    dino_detected = object in dino_result["labels"]
    
    # Step 3: 交叉验证
    if yolo_detected and dino_detected:
        return "nonhallu"
    elif not yolo_detected and not dino_detected:
        return "hallu"
    else:
        return "uncertain"
```

### 5.2 偏好对构建算法

```python
def build_preference_pairs(candidates: list[str], object_analysis: list) -> list[dict]:
    """
    从候选句子中构建偏好对
    
    策略:
    1. 筛选非幻觉候选（包含物体但无幻觉）
    2. 筛选幻觉候选（包含幻觉物体）
    3. 一一配对形成偏好对
    """
    nonhallu_candidates = [
        (i, sent) for i, sent in enumerate(candidates)
        if has_objects(sent) and not has_hallucination(sent)
    ]
    
    hallu_candidates = [
        (i, sent) for i, sent in enumerate(candidates)
        if has_hallucination(sent)
    ]
    
    pairs = []
    for (win_idx, _), (lose_idx, _) in zip(nonhallu_candidates, hallu_candidates):
        pairs.append({
            "y_win": candidates[win_idx],
            "y_lose": candidates[lose_idx],
        })
    
    return pairs
```

### 5.3 上下文选择算法

```python
def select_best_context(candidates: list[str], object_analysis: list) -> int:
    """
    选择最佳句子作为下一步上下文
    
    优先级:
    1. 成功探索句子（包含新物体的非幻觉句子）
    2. 普通非幻觉句子
    3. 随机选择
    """
    # 分类句子
    success_explore = []  # 包含新物体
    normal_nonhallu = []  # 只有已知物体
    
    for i, sent in enumerate(candidates):
        if is_nonhallu(sent):
            if has_new_objects(sent):
                success_explore.append(i)
            else:
                normal_nonhallu.append(i)
    
    # 按优先级选择
    if success_explore:
        return random.choice(success_explore)
    elif normal_nonhallu:
        return random.choice(normal_nonhallu)
    else:
        return random.randint(0, len(candidates) - 1)
```

### 5.4 C-DPO 训练算法

```python
def compute_cdpo_loss(model, context, y_win, y_lose):
    """
    计算 Context-aware DPO 损失
    
    关键: 只对新生成部分计算损失，忽略上下文部分
    """
    # 拼接上下文和响应
    full_win = context + " " + y_win
    full_lose = context + " " + y_lose
    
    # 获取 token IDs
    win_ids = tokenize(full_win)
    lose_ids = tokenize(full_lose)
    context_ids = tokenize(context)
    
    # 创建 labels，上下文部分标记为 IGNORE_INDEX
    win_labels = [IGNORE_INDEX] * len(context_ids) + win_ids[len(context_ids):]
    lose_labels = [IGNORE_INDEX] * len(context_ids) + lose_ids[len(context_ids):]
    
    # 计算标准 DPO 损失
    return dpo_loss(model, win_ids, win_labels, lose_ids, lose_labels)
```

---

## 附录：文件依赖图

```
main.py
├── utils/setup_utils.py
├── model/auxiliary/global_vars.py
│   └── utils/setup_utils.py
└── run/run.py
    ├── model/auxiliary/dataset.py
    │   └── model/utils/utils.py
    └── run/generate_dataset.py
        ├── model/auxiliary/datastate.py
        ├── model/detector/grounding_dino.py
        ├── model/detector/yolo_model.py
        ├── model/generator/llava.py
        ├── model/generator/qwen2_vl.py
        ├── model/others/sg_parser.py
        ├── model/others/spacy_model.py
        ├── model/others/wordnet.py
        ├── model/utils/gen_utils.py
        └── run/utils.py
            └── run/object_utils.py
```

---

*文档版本：1.0*
