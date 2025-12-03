# SENTINEL 快速开始指南

本指南帮助你在 10 分钟内开始使用 SENTINEL。

---

## 🚀 快速安装

```bash
# 1. 克隆仓库
git clone https://github.com/pspdada/SENTINEL.git --depth=1
cd SENTINEL

# 2. 创建环境
conda create -y -n SENTINEL python=3.10
conda activate SENTINEL

# 3. 安装依赖
pip install -r requirements.txt
pip install -U flash-attn --no-build-isolation

# 4. 下载 NLP 模型
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab')"
python -m spacy download en_core_web_md
```

---

## 📊 使用已发布的数据集

如果你只想使用我们发布的数据集进行训练，可以直接跳到训练步骤。

### 下载数据集

从 HuggingFace 下载：https://huggingface.co/datasets/psp-dada/SENTINEL

```bash
# 下载特定模型的数据
wget https://huggingface.co/datasets/psp-dada/SENTINEL/resolve/main/Qwen2_VL_7B_Instruct_SENTINEL_7k.json
```

---

## 🔧 生成自己的数据

### Step 1: 准备数据

```bash
# 下载 Visual Genome 图像
# https://homes.cs.washington.edu/~ranjay/visualgenome/api.html

# 下载输入数据
wget -O dataset/image_data.jsonl \
  https://huggingface.co/datasets/psp-dada/SENTINEL/resolve/main/image_data.jsonl

# 修改 image_data.jsonl 中的 image_path 为你的本地路径
```

### Step 2: 配置环境变量

创建 `utils/.env` 文件：

```bash
# 模型缓存目录
HF_HOME=/path/to/huggingface
MODEL_PATH=/path/to/models

# GPU 设备
CUDA_VISIBLE_DEVICES=0,1
```

### Step 3: 运行数据生成

```bash
# 使用默认模型 (Qwen2_VL_2B)
python main.py

# 或指定模型
python main.py --model Qwen2_VL_7B --batch_size 5

# 处理部分数据（测试）
python main.py --model Qwen2_VL_2B --num_of_data 100
```

### Step 4: 转换数据格式

```bash
# 转换为 LLaMA-Factory 格式
python utils/get_llama_factory_data_pair.py
```

---

## 🎯 训练模型

### 方式 1: LLaVA-v1.5 训练

```bash
# 设置环境变量
export INPUT_MODEL=/path/to/llava-v1.5-7b
export TRAINING_DATA_PATH=/path/to/training_data.json
export OUTPUT_NAME=my_sentinel_model
export VISUAL_GENOME_PATH=/path/to/visual_genome

# 运行训练
bash train/models/dpo_llava.sh
```

### 方式 2: LLaMA-Factory 训练 (Qwen2-VL)

```bash
# 1. 安装 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e .

# 2. 复制 SENTINEL 文件
cp -r /path/to/SENTINEL/llamafactory/data/* data/
cp -r /path/to/SENTINEL/llamafactory/examples/* examples/
cp /path/to/SENTINEL/llamafactory/src/llamafactory/data/*.py src/llamafactory/data/

# 3. 运行训练
bash examples/SENTINEL/Qwen_2_5_VL/qwen_2_5_vl_7B_dpo.sh
```

---

## 📈 评估模型

### 准备评估数据

```bash
# 下载 LLaVA 评估数据包
# https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view
unzip eval.zip -d llava/data/

# 下载 COCO 标注
mkdir -p llava/data/MSCOCO/coco2014
cd llava/data/MSCOCO/coco2014
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip
```

### 运行评估

```bash
# Object HalBench
bash llava/eval_script/eval_object_halbench.sh

# POPE
bash llava/eval_script/eval_pope.sh

# AMBER
bash llava/eval_script/eval_amber_dis.sh
```

---

## 📁 输出文件说明

### 数据生成输出

```
./results/
├── Qwen2_VL_7B.jsonl           # 完整分析结果
└── Qwen2_VL_7B_data_pair.jsonl # 偏好对数据
```

### 训练输出

```
./train/results/<OUTPUT_NAME>/
├── adapter_model.bin    # LoRA 权重
├── adapter_config.json  # LoRA 配置
└── training_args.bin    # 训练参数
```

---

## ❓ 常见问题

### Q: CUDA OOM 错误
```bash
# 减小 batch size
python main.py --batch_size 3
```

### Q: 模型下载慢
```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: vLLM 初始化失败
```bash
# 尝试禁用 vLLM
# 修改 run/generate_dataset.py 中的 use_vllm=False
```

---

## 📚 更多资源

- [详细帮助文档](HELP_zh.md)
- [代码结构说明](CODE_STRUCTURE.md)
- [评估指南](Evaluation.md)
- [论文](https://arxiv.org/abs/2507.12455)

---

*如有问题，请提交 Issue 或 PR！*
