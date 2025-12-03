# SENTINEL 详细帮助文档

<div align="center">

## 通过句子级早期干预缓解对象幻觉

</div>

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 核心概念](#2-核心概念)
- [3. 项目架构](#3-项目架构)
- [4. 安装指南](#4-安装指南)
- [5. 快速开始](#5-快速开始)
- [6. 核心模块详解](#6-核心模块详解)
- [7. 数据格式说明](#7-数据格式说明)
- [8. 训练指南](#8-训练指南)
- [9. 评估指南](#9-评估指南)
- [10. 常见问题](#10-常见问题)
- [11. API 参考](#11-api-参考)

---

## 1. 项目简介

### 1.1 什么是 SENTINEL？

SENTINEL（**S**entence-level **E**arly i**NT**ervention for hall**I**ucination mitigatio**N** in **E**fficient **L**arge models）是一种自动化的、句子级别的早期干预策略，用于防止和缓解多模态大语言模型（MLLMs）中的对象幻觉问题。

### 1.2 核心发现

研究发现，MLLMs 的幻觉具有以下特点：

1. **早期产生**：幻觉主要在生成的前几个句子中产生
2. **链式传播**：一旦产生幻觉，会在后续输出中不断传播和放大
3. **上下文依赖**：后续句子的生成高度依赖前文上下文

### 1.3 解决方案

SENTINEL 通过以下方式解决幻觉问题：

1. **句子级采样**：对每个句子进行多次采样，获取多个候选
2. **检测器交叉验证**：使用 YOLO 和 Grounding DINO 双检测器验证物体真实性
3. **偏好对构建**：自动构建（非幻觉句子, 幻觉句子）偏好对
4. **C-DPO 训练**：使用上下文感知的 DPO 训练模型

---

## 2. 核心概念

### 2.1 对象幻觉 (Object Hallucination)

指模型生成的文本中描述了图像中**不存在**的物体。例如：

```
图像：一个人站在桌子旁边
模型输出：一个人站在桌子旁边，手里拿着一本书。  ← "书"是幻觉
```

### 2.2 偏好学习 (Preference Learning)

通过对比"好"的输出和"坏"的输出来训练模型：

```
y_win (正样本)：一个人站在桌子旁边，桌上有一杯咖啡。  ← 物体真实存在
y_lose (负样本)：一个人站在桌子旁边，手里拿着一本书。 ← 物体是幻觉
```

### 2.3 C-DPO (Context-aware DPO)

在标准 DPO 基础上，SENTINEL 引入了上下文感知机制：

```
Context: "一个人站在桌子旁边。"
y_win: "桌上有一杯咖啡。"
y_lose: "手里拿着一本书。"
```

训练时，**只对新生成的句子部分计算损失**，忽略上下文部分。

### 2.4 检测器交叉验证

使用两个独立的目标检测器进行验证：

| YOLO 结果 | DINO 结果 | 最终判定 |
|-----------|-----------|----------|
| 检测到 | 检测到 | **非幻觉** |
| 未检测到 | 未检测到 | **幻觉** |
| 检测到 | 未检测到 | 不确定 |
| 未检测到 | 检测到 | 不确定 |

---

## 3. 项目架构

### 3.1 目录结构

```
SENTINEL/
├── main.py                      # 主入口文件
├── requirements.txt             # 依赖包列表
├── README.md                    # 项目说明
│
├── model/                       # 核心模型模块
│   ├── __init__.py
│   ├── auxiliary/               # 辅助工具
│   │   ├── dataset.py          # 数据集加载
│   │   ├── datastate.py        # 数据状态管理
│   │   └── global_vars.py      # 全局变量
│   ├── detector/                # 目标检测器
│   │   ├── grounding_dino.py   # Grounding DINO
│   │   └── yolo_model.py       # YOLO 模型
│   ├── generator/               # 文本生成器
│   │   ├── llava.py            # LLaVA 模型
│   │   ├── qwen2_vl.py         # Qwen2-VL 模型
│   │   └── qwen2_5_vl.py       # Qwen2.5-VL 模型
│   ├── others/                  # NLP 工具
│   │   ├── sg_parser.py        # 场景图解析器
│   │   ├── spacy_model.py      # Spacy NLP
│   │   └── wordnet.py          # WordNet 同义词
│   └── utils/                   # 工具函数
│       ├── gen_utils.py        # 生成相关工具
│       └── utils.py            # 通用工具
│
├── run/                         # 运行逻辑
│   ├── run.py                  # 主运行逻辑
│   ├── generate_dataset.py     # 数据生成核心
│   ├── object_utils.py         # 物体处理工具
│   └── utils.py                # 运行时工具
│
├── train/                       # 训练模块
│   ├── models/                 # 训练脚本
│   │   ├── dpo_llava.py       # LLaVA DPO 训练
│   │   ├── dpo_llava.sh       # 训练启动脚本
│   │   └── llava_utils/       # LLaVA 训练工具
│   ├── accelerate_configs/     # 分布式配置
│   │   ├── zero2.json
│   │   ├── zero3.json
│   │   └── deepspeed_zero3.yaml
│   └── utils/                  # 训练工具
│
├── llava/                       # LLaVA 评估
│   ├── eval/                   # 评估脚本
│   │   ├── model_vqa.py       # VQA 推理
│   │   └── utils/             # 评估工具
│   └── eval_script/           # Shell 脚本
│
├── llamafactory/                # LLaMA-Factory 集成
│   ├── data/                   # 数据配置
│   ├── examples/               # 训练示例
│   └── src/                    # 修改的源码
│
├── utils/                       # 数据转换工具
│   ├── .env                    # 环境变量
│   ├── setup_utils.py          # 参数设置
│   ├── get_llava_v15_data_pair.py
│   └── get_llama_factory_data_pair.py
│
├── dataset/                     # 数据集目录
│   └── .gitkeep
│
└── docs/                        # 文档
    ├── Evaluation.md
    ├── Evaluation_zh.md
    └── figures/
```

### 3.2 模块依赖关系

```
main.py
    ├── model/auxiliary/global_vars.py    # 初始化全局变量
    │       ├── utils/setup_utils.py      # 解析命令行参数
    │       └── torch, transformers       # 设备和日志初始化
    │
    └── run/run.py                        # 主运行逻辑
            ├── model/auxiliary/dataset.py    # 加载数据集
            └── run/generate_dataset.py       # 核心数据生成
                    ├── model/generator/*     # MLLM 生成器
                    ├── model/detector/*      # 目标检测器
                    └── model/others/*        # NLP 工具
```

---

## 4. 安装指南

### 4.1 环境要求

- Python 3.10+
- CUDA 12.x（推荐）
- GPU 显存 ≥ 24GB（数据生成），≥ 40GB（训练）

### 4.2 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/pspdada/SENTINEL.git --depth=1
cd SENTINEL

# 2. 创建 Conda 环境
conda create -y -n SENTINEL python=3.10
conda activate SENTINEL

# 3. 安装基础依赖
pip install -r requirements.txt

# 4. 安装 Flash Attention（加速推理）
pip install -U flash-attn --no-build-isolation

# 5. 安装 NLTK 数据包
python -c "
import nltk
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('cmudict')
nltk.download('averaged_perceptron_tagger_eng')
"

# 6. 安装 Spacy 模型
pip install -U pip setuptools wheel
pip install 'spacy[cuda12x]==3.8.0'
python -m spacy download en_core_web_md  # 数据生成用
python -m spacy download en_core_web_trf  # 评估用

# 7. 安装 CLIP（YOLO 依赖）
pip install git+https://github.com/openai/CLIP.git
```

### 4.3 环境变量配置

编辑 `utils/.env` 文件：

```bash
# Hugging Face 模型缓存目录
HF_HOME=/path/to/huggingface/cache

# 模型下载目录
MODEL_PATH=/path/to/models

# OpenAI API Key（Object HalBench 评估需要）
OPENAI_KEY=sk-xxx

# CUDA 设备
CUDA_VISIBLE_DEVICES=0,1
```

---

## 5. 快速开始

### 5.1 数据生成

```bash
# 1. 准备数据
# 下载 Visual Genome 图像：https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
# 下载 image_data.jsonl 到 dataset/ 目录

# 2. 修改配置（可选）
vim utils/setup_utils.py
# 修改 --model 参数选择目标模型

# 3. 运行数据生成
python main.py

# 生成结果保存在 ./results/<model_name>.jsonl
```

### 5.2 数据转换

```bash
# 转换为 LLaMA-Factory 格式
python utils/get_llama_factory_data_pair.py

# 或转换为 LLaVA-v1.5 格式
python utils/get_llava_v15_data_pair.py
```

### 5.3 模型训练

```bash
# LLaVA-v1.5 训练
export INPUT_MODEL=/path/to/llava-v1.5-7b
export TRAINING_DATA_PATH=/path/to/training_data.json
export OUTPUT_NAME=my_sentinel_model
export VISUAL_GENOME_PATH=/path/to/visual_genome
bash train/models/dpo_llava.sh
```

### 5.4 模型评估

```bash
# Object HalBench 评估
bash llava/eval_script/eval_object_halbench.sh
```

---

## 6. 核心模块详解

### 6.1 数据生成流程 (`run/generate_dataset.py`)

```python
def run_gen_dataset(datalist: list[DataPoint], batch_size: int) -> None:
    """
    核心数据生成函数
    
    工作流程：
    1. 初始化模型（生成器、检测器、NLP工具）
    2. 批量处理图像数据
    3. 对每个图像：
       a. YOLO 预检测图像中的物体
       b. MLLM 生成多个候选句子
       c. 指代消解处理
       d. 提取句子中的物体
       e. 使用双检测器验证物体真实性
       f. 构建偏好对并保存
    4. 迭代直到所有数据处理完成
    """
```

### 6.2 幻觉检测 (`run/utils.py`)

```python
def b_get_hallu_objects(
    b_object_lists,      # 批量物体列表
    b_nonhallu_objects,  # 已知非幻觉物体
    b_hallu_objects,     # 已知幻觉物体
    spacy, wn,           # NLP 工具
    images,              # 图像列表
    dino,                # DINO 检测器
    b_yolo_results,      # YOLO 检测结果
    ...
) -> tuple[list, list]:
    """
    批量获取幻觉物体
    
    检测逻辑：
    1. 获取未缓存的物体（新出现的物体）
    2. 使用 DINO 检测未缓存物体
    3. 交叉验证 YOLO 和 DINO 结果
    4. 更新物体分类（幻觉/非幻觉/不确定）
    """
```

### 6.3 生成器接口 (`model/generator/`)

所有生成器遵循统一接口：

```python
class BaseGenerator:
    def gen(
        self,
        images: list[Image.Image],    # 输入图像
        users: list[str],              # 用户问题
        assistants: list[str],         # 已有上下文
        do_sample: bool = False,       # 是否采样
        n: int = 1,                    # 采样数量
        temp: float = 0.3,             # 温度
        max_tokens: int = 512,         # 最大生成长度
        single_sentence: bool = False, # 是否只生成一句
    ) -> GenOutput:
        """
        生成文本响应
        
        Returns:
            GenOutput: 包含 outputs, generated_ids, log_probs 等
        """
```

### 6.4 检测器接口 (`model/detector/`)

```python
# Grounding DINO
class DINO:
    def detect(
        self,
        images: list[Image.Image],
        captions: list[str],           # 要检测的物体名称
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> list[dict]:
        """
        开放词汇目标检测
        
        Returns:
            [{'scores': tensor, 'boxes': tensor, 'labels': list}, ...]
        """

# YOLO
class YoloModel:
    def predict(
        self,
        images: list[Image.Image],
    ) -> list[YoloResult]:
        """
        YOLO 目标检测
        
        Returns:
            [YoloResult(labels=['person', 'dog', ...]), ...]
        """
```

---

## 7. 数据格式说明

### 7.1 输入数据格式

#### image_data.jsonl

```jsonl
{"image_id": "2417997", "image_path": "/path/to/vg/VG_100K/2417997.jpg", "question": "Describe this image in detail."}
{"image_id": "2382741", "image_path": "/path/to/vg/VG_100K/2382741.jpg", "question": "What objects are in this image?"}
```

### 7.2 中间数据格式

#### <model_name>.jsonl（分析用）

```jsonl
{
    "image_id": "2417997",
    "image_path": "/path/to/image.jpg",
    "question": "Describe this image in detail.",
    "caption": "A man is standing near a table. He is wearing a blue shirt.",
    "sentences_cnt": 2,
    "hallu_objects": ["laptop", "phone"],
    "uncertain_objects": ["cup"],
    "nonhallu_objects": ["man", "table", "shirt"],
    "hard_positive": ["chair"],
    "small_objects": [],
    "edge_objects": []
}
```

### 7.3 偏好对数据格式

#### <model_name>_data_pair.jsonl

```jsonl
{
    "image_id": "2417997",
    "image_path": "/path/to/image.jpg",
    "question": "Describe this image in detail.",
    "context": "A man is standing near a table.",
    "y_win": "There is a coffee cup on the table.",
    "y_lose": "He is holding a laptop.",
    "nonhallu_objects": ["man", "table", "coffee cup"],
    "context_gen_objects": ["man", "table"],
    "context_gen_hallu_objects": [],
    "objects_of_y_win": ["coffee cup"],
    "hallu_objects_of_y_lose": ["laptop"],
    "is_last_sent": false,
    "type": "y+"
}
```

### 7.4 训练数据格式

#### LLaVA-v1.5 格式

```json
[
    {
        "id": "unique_id",
        "image": "path/to/image.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nDescribe this image."},
            {"from": "gpt", "value": "A man is standing..."}
        ],
        "chosen": "There is a coffee cup on the table.",
        "rejected": "He is holding a laptop.",
        "context": "A man is standing near a table."
    }
]
```

#### LLaMA-Factory 格式

```json
[
    {
        "instruction": "<image>Describe this image.",
        "context": "A man is standing near a table.",
        "chosen": "There is a coffee cup on the table.",
        "rejected": "He is holding a laptop.",
        "images": ["path/to/image.jpg"]
    }
]
```

---

## 8. 训练指南

### 8.1 LLaVA-v1.5 训练

#### 8.1.1 配置参数

编辑 `train/models/dpo_llava.sh`：

```bash
# GPU 配置
GPU_LIST="0,1"              # 使用的 GPU
PER_DEVICE_BATCH_SIZE=16    # 每 GPU batch size（40GB用8，80GB用16）

# 训练超参数
--lora_r 128                # LoRA rank
--lora_alpha 256            # LoRA alpha
--learning_rate 2e-6        # 学习率
--beta 0.1                  # DPO beta
--num_train_epochs 1        # 训练轮数
```

#### 8.1.2 运行训练

```bash
export INPUT_MODEL=/path/to/llava-v1.5-7b
export TRAINING_DATA_PATH=/path/to/data.json
export OUTPUT_NAME=sentinel_llava_v1_5_7b
export VISUAL_GENOME_PATH=/path/to/visual_genome

bash train/models/dpo_llava.sh
```

### 8.2 LLaMA-Factory 训练

#### 8.2.1 环境配置

```bash
# 创建新环境
conda create -n LLaMA-Factory-SENTINEL python=3.10
conda activate LLaMA-Factory-SENTINEL

# 安装 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .

# 复制 SENTINEL 修改的文件
cp -r /path/to/SENTINEL/llamafactory/data/* data/
cp -r /path/to/SENTINEL/llamafactory/examples/* examples/
cp /path/to/SENTINEL/llamafactory/src/llamafactory/data/*.py src/llamafactory/data/
```

#### 8.2.2 运行训练

```bash
# 编辑训练脚本
vim examples/SENTINEL/Qwen_2_5_VL/qwen_2_5_vl_7B_dpo.sh

# 修改以下参数
MODEL_NAME="/path/to/Qwen2.5-VL-7B-Instruct"
GPU_LIST="0,1,2,3"

# 运行
bash examples/SENTINEL/Qwen_2_5_VL/qwen_2_5_vl_7B_dpo.sh
```

### 8.3 训练技巧

1. **Batch Size 计算**：
   ```
   global_batch_size = num_gpus × per_device_batch_size × gradient_accumulation_steps
   推荐 global_batch_size = 64
   ```

2. **显存优化**：
   - 使用 DeepSpeed ZeRO-2 或 ZeRO-3
   - 启用 gradient checkpointing
   - 使用 BF16 混合精度

3. **学习率**：
   - LLaVA: 2e-6
   - Qwen-VL: 3e-6

---

## 9. 评估指南

### 9.1 Object HalBench

```bash
# 1. 准备数据
mkdir -p llava/data/MSCOCO/coco2014
cd llava/data/MSCOCO/coco2014
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip

# 2. 下载问题文件到 llava/data/eval/object_halbench/

# 3. 运行评估
bash llava/eval_script/eval_object_halbench.sh
```

**输出指标**：
- **CHAIRs**: 包含幻觉的句子比例（越低越好）
- **CHAIRi**: 幻觉物体占总物体的比例（越低越好）

### 9.2 POPE

```bash
bash llava/eval_script/eval_pope.sh
```

**输出指标**：
- Accuracy
- Precision
- Recall
- F1 Score

### 9.3 AMBER

```bash
# 判别式评估
bash llava/eval_script/eval_amber_dis.sh

# 生成式评估
bash llava/eval_script/eval_amber_gen.sh
```

### 9.4 通用能力评估

```bash
# VQAv2
bash llava/eval_script/eval_vqav2.sh

# ScienceQA
bash llava/eval_script/eval_ScienceQA.sh

# TextVQA
bash llava/eval_script/eval_textvqa.sh

# MM-Vet
bash llava/eval_script/eval_mmvet.sh
```

---

## 10. 常见问题

### Q1: CUDA OOM 错误

**解决方案**：
```bash
# 减小 batch size
--batch_size 3

# 使用 DeepSpeed ZeRO-3
--deepspeed train/accelerate_configs/zero3.json

# 降低 GPU 利用率
# 在 model/utils/gen_utils.py 中修改 gpu_util
gpu_util: float = 0.6
```

### Q2: 模型下载失败

**解决方案**：
```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型到本地
# 然后设置 MODEL_PATH 环境变量
```

### Q3: Spacy 模型加载失败

**解决方案**：
```bash
# 重新下载
python -m spacy download en_core_web_md

# 或手动安装
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0.tar.gz
```

### Q4: NLTK 数据下载失败

**解决方案**：
```python
import nltk
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('wordnet')
```

### Q5: vLLM 初始化失败

**解决方案**：
```bash
# 检查 CUDA 版本
nvcc --version

# 重新安装 vLLM
pip uninstall vllm
pip install vllm==0.7.3
```

---

## 11. API 参考

### 11.1 GVars 全局变量

```python
from model.auxiliary.global_vars import GVars

# 初始化
GVars.init()

# 访问变量
args = GVars.args           # 命令行参数
logger = GVars.logger       # 日志器
device = GVars.device       # 主设备
model_dir = GVars.model_dir # 模型目录
```

### 11.2 DataSet 类

```python
from model.auxiliary.dataset import DataSet, DataPoint

# 加载数据集
dataset = DataSet(args=args, logger=logger)

# 过滤已处理数据
dataset.filter(save_path)

# 访问数据
for data_point in dataset.data:
    print(data_point.image_id)
    print(data_point.image_path)
    print(data_point.question)
```

### 11.3 生成器

```python
from model.utils.gen_utils import get_generator

# 获取生成器
generator = get_generator(use_vllm=True, debug=False)

# 生成文本
output = generator.gen(
    images=[image],
    users=["Describe this image."],
    assistants=[""],
    do_sample=True,
    n=10,
    temp=0.7,
    single_sentence=True
)

# 访问结果
for text in output.outputs:
    print(text)
```

### 11.4 检测器

```python
from model.detector.grounding_dino import DINO
from model.detector.yolo_model import YoloModel

# DINO
dino = DINO("base", model_dir=model_dir, device="cuda:0")
results = dino.detect(images, captions=["cat. dog."])
print(results[0]["labels"])  # ['cat']

# YOLO
yolo = YoloModel("yolo11x", model_dir=model_dir)
results = yolo.predict(images)
print(results[0].labels)  # ['person', 'dog', 'car']
```

### 11.5 NLP 工具

```python
from model.others.spacy_model import SpacyModel
from model.others.sg_parser import SGParser
from model.others.wordnet import WordnetModel

# Spacy
spacy = SpacyModel(model_size="md", device="cuda:0")
resolved = spacy.resolve_coref("The man picked up the ball. He threw it.")
print(resolved)  # "The man picked up the ball. The man threw the ball."

# 场景图解析
sg_parser = SGParser(size="base", device="cuda:0")
graphs = sg_parser.pharse("A man is holding a cup.")
print(graphs)  # [["man", "holding", "cup"]]

# WordNet
wn = WordnetModel()
synonyms = wn.get_synset_list("dog")
print(synonyms)  # ['dog', 'canine', 'hound', ...]
```

---

## 附录

### A. 支持的模型列表

| 模型 | 参数量 | 支持状态 |
|------|--------|----------|
| LLaVA-v1.5-7B | 7B | ✅ |
| LLaVA-v1.5-13B | 13B | ✅ |
| LLaVA-v1.6-Vicuna-7B | 7B | ✅ |
| LLaVA-v1.6-Vicuna-13B | 13B | ✅ |
| Qwen2-VL-2B-Instruct | 2B | ✅ |
| Qwen2-VL-7B-Instruct | 7B | ✅ |
| Qwen2.5-VL-7B-Instruct | 7B | ✅ |

### B. 硬件要求

| 任务 | 最低显存 | 推荐显存 |
|------|----------|----------|
| 数据生成 | 24GB | 48GB |
| 训练（7B） | 40GB×2 | 80GB×2 |
| 训练（13B） | 80GB×2 | 80GB×4 |
| 推理 | 16GB | 24GB |

### C. 引用

```bibtex
@article{peng2025mitigating,
  title={Mitigating Object Hallucinations via Sentence-Level Early Intervention},
  author={Peng, Shangpin and Yang, Senqiao and Jiang, Li and Tian, Zhuotao},
  journal={arXiv preprint arXiv:2507.12455},
  year={2025}
}
```

---

*文档版本：1.0 | 最后更新：2025年7月*
