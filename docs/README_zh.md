<div align="center">

# Mitigating Object Hallucinations via <br> Sentence-Level Early Intervention <!-- omit in toc -->

<a href='https://arxiv.org/abs/2507.12455'>
<img src='https://img.shields.io/badge/Paper-Arxiv-purple'></a>
<a href='https://huggingface.co/datasets/psp-dada/SENTINEL'>
<img src='https://img.shields.io/badge/Datasets-HF-Green'></a>
<a href='https://huggingface.co/collections/psp-dada/sentinel-686ea70912079af142015286'>
<img src='https://img.shields.io/badge/Models-HF-orange'></a>
<a href='https://huggingface.co/papers/2507.12455'>
<img src='https://img.shields.io/badge/Discussion-HF-blue'></a>
<a href='https://github.com/pspdada/SENTINEL/blob/main/LICENSE'>
<img src='https://img.shields.io/badge/LICENSE-Apache_2.0-yellow'></a>

<b>中文</b> | <a href="/README.md">English</a>

**[Shangpin Peng](https://scholar.google.com/citations?user=mKnBrRAAAAAJ&hl=zh-CN)\*<sup>1</sup>**,&emsp;
**[Senqiao Yang](https://scholar.google.com/citations?user=NcJc-RwAAAAJ)\*<sup>2</sup>**,&emsp;
**[Li Jiang](https://scholar.google.com/citations?user=5cIodxsAAAAJ)<sup>3</sup>**,&emsp;
**[Zhuotao Tian](https://scholar.google.com/citations?user=mEjhz-IAAAAJ&hl=zh-CN)<sup>1</sup>**<sup>✉️</sup>

<sup>1</sup>Harbin Institute of Technology, Shenzhen<br>
<sup>2</sup>The Chinese University of Hong Kong<br>
<sup>3</sup>The Chinese University of Hong Kong, Shenzhen

\* Equal contribution.<br>
<sup>✉️</sup> Corresponding author: tianzhuotao@hit.edu.cn.

</div>

## 🎊 新闻 <!-- omit in toc -->

- [2025.07.21] 所有代码、数据和模型已发布！
- [2025.06.26] 🎉 我们的 SENTINEL 被 **ICCV 2025** 接收！

## 🚀 概览 <!-- omit in toc -->

**SENTINEL** 引入了一种自动化、句子级别的早期干预策略，以防止和缓解多模态大语言模型（MLLM）中的对象幻觉。主要优势：

- **无需标注**：不需要人工标注。
- **模型无关**：兼容任何 MLLM 架构。
- **高效**：轻量级 LoRA 微调。

## 📌 目录 <!-- omit in toc -->

- [🔑 主要特性](#-主要特性)
- [📚 数据集](#-数据集)
- [📦 模型权重](#-模型权重)
- [💻 环境配置](#-环境配置)
- [🔨 数据生成](#-数据生成)
- [⚙️ 训练](#️-训练)
- [📈 评估](#-评估)
- [📝 引用](#-引用)

## 🔑 主要特性

- 🧠 **早期干预阻断幻觉传播**。我们发现 MLLMs 的幻觉主要在前几句中产生，并在后续输出中不断传播。SENTINEL 提前打断该链条，以最大程度地缓解幻觉。
<table align="center">
    <p align="center">
      <img src="/docs/figures/figure2.png" width="80%" />
    </p>
</table>

- 🔍 **无需人工标注的域内上下文偏好学习**。SENTINEL 通过检测器交叉验证构造*幻觉/真实*样本，并在不依赖专有 LLM 或手动标注的情况下构建域内偏好数据。
<table align="center">
    <p align="center">
      <img src="/docs/figures/figure3.png" width="80%" />
    </p>
</table>

- 💡 **上下文至关重要：丰富的连贯性提升鲁棒性**。通过优先选择上下文连贯的正样本，SENTINEL 显著提升了泛化能力。
<table align="center">
    <p align="center">
      <img src="/docs/figures/figure4.png" width="80%" />
    </p>
</table>

- ♻️ **迭代式上下文自举，生成多样无幻觉上下文**。我们的框架动态地扩展非幻觉上下文，覆盖更多场景，提升模型在推理阶段的鲁棒性。
<table align="center">
    <p align="center">
      <img src="/docs/figures/figure5.png" width="80%" />
    </p>
</table>

- 📊 **各基准测试中达到优秀水平**。
SENTINEL 在减少幻觉方面最多可达 **92%** 的降低，并在 Object HalBench、AMBER 和 HallusionBench 上超越先前 SOTA 方法，同时保持或提升通用任务性能。
<table align="center">
    <p align="center">
      <img src="/docs/figures/table1.png" width="80%" />
    </p>
</table>

## 📚 数据集

我们发布了 [**SENTINEL 数据集**](https://huggingface.co/datasets/psp-dada/SENTINEL)，这是一个无需人工标注即可构建的领域内多模态偏好数据集，用于缓解对象幻觉。

<details>
<summary>数据集详情</summary>

SENTINEL 数据集记录了 `LLaVA-v1.5`、`LLaVA-v1.6`、`Qwen2-VL` 和 `Qwen2.5-VL` 系列模型的偏好对，支持在无外部监督下进行稳健且可扩展的幻觉缓解。

包含以下部分：

- `image_data.jsonl` 文件

  该文件包含从 Visual Genome 数据集中筛选的开源图像，仅包含 `image_id`、`image_path` 和 `question` 三个字段，用于构建图像描述任务的偏好训练数据。

  **注意**：使用此文件数据时，请将 `image_path` 字段替换为本地 Visual Genome 数据集的路径。

- `<model_name>.json` 文件

  这些文件是训练数据构建步骤生成的偏好训练集，每个文件对应特定模型，包含进行 **C-DPO 训练** 所需的字段，如 `"question"`、`"context"`、`"y_win"` 和 `"y_lose"`。

<table align="center">
    <p align="center">
      <img src="/docs/figures/dataset.png" width="80%" />
    </p>
</table>
</details>

## 📦 模型权重

我们提供了论文中提到的所有模型权重，均通过 **LoRA** 训练。可直接插入对应基础模型进行推理或进一步微调。

| 基础模型               | 训练数据量 | LoRA |                                                                                                                                  下载                                                                                                                                  |
| ---------------------- | ---------- | :--: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| LLaVA-v1.5-7B          | 8.6K       |  ✅  |            [🤗 基础模型](https://huggingface.co/liuhaotian/llava-v1.5-7b) / [📄 数据](https://huggingface.co/datasets/psp-dada/SENTINEL/blob/main/LLaVA_v1_5_7b_SENTINEL_8_6k.json) / [🤗 SENTINEL](https://huggingface.co/psp-dada/LLaVA-v1.5-7B-SENTINEL)            |
| LLaVA-v1.5-13B         | 7.0K       |  ✅  |           [🤗 基础模型](https://huggingface.co/liuhaotian/llava-v1.5-13b) / [📄 数据](https://huggingface.co/datasets/psp-dada/SENTINEL/blob/main/LLaVA_v1_5_13b_SENTINEL_7k.json) / [🤗 SENTINEL](https://huggingface.co/psp-dada/LLaVA-v1.5-13B-SENTINEL)            |
| LLaVA-v1.6-Vicuna-7B   | 7.0K       |  ✅  |  [🤗 基础模型](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) / [📄 数据](https://huggingface.co/datasets/psp-dada/SENTINEL/blob/main/LLaVA_v1_6_Vicuna_7b_SENTINEL_7k.json) / [🤗 SENTINEL](https://huggingface.co/psp-dada/LLaVA-v1.6-Vicuna-7B-SENTINEL)   |
| LLaVA-v1.6-Vicuna-13B  | 7.0K       |  ✅  | [🤗 基础模型](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b) / [📄 数据](https://huggingface.co/datasets/psp-dada/SENTINEL/blob/main/LLaVA_v1_6_Vicuna_13b_SENTINEL_7k.json) / [🤗 SENTINEL](https://huggingface.co/psp-dada/LLaVA-v1.6-Vicuna-13B-SENTINEL) |
| Qwen2-VL-2B-Instruct   | 12K        |  ✅  |         [🤗 基础模型](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) / [📄 数据](https://huggingface.co/datasets/psp-dada/SENTINEL/blob/main/Qwen2_VL_2B_Instruct_SENTINEL_12k.json) / [🤗 SENTINEL](https://huggingface.co/psp-dada/Qwen2-VL-2B-SENTINEL)          |
| Qwen2-VL-7B-Instruct   | 7.0K       |  ✅  |          [🤗 基础模型](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) / [📄 数据](https://huggingface.co/datasets/psp-dada/SENTINEL/blob/main/Qwen2_VL_7B_Instruct_SENTINEL_7k.json) / [🤗 SENTINEL](https://huggingface.co/psp-dada/Qwen2-VL-7B-SENTINEL)          |
| Qwen2.5-VL-7B-Instruct | 7.0K       |  ✅  |       [🤗 基础模型](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) / [📄 数据](https://huggingface.co/datasets/psp-dada/SENTINEL/blob/main/Qwen2_5_VL_7B_Instruct_SENTINEL_7k.json) / [🤗 SENTINEL](https://huggingface.co/psp-dada/Qwen2.5-VL-7B-SENTINEL)       |

## 💻 环境配置

1. 克隆本仓库并进入 _SENTINEL_ 目录

   ```bash
   git clone https://github.com/pspdada/SENTINEL.git --depth=1
   cd SENTINEL
   ```

2. 安装依赖

   ```bash
   conda create -y -n SENTINEL python=3.10
   conda activate SENTINEL
   pip install -r requirements.txt
   pip install -U flash-attn --no-build-isolation
   ```

3. 安装额外依赖
   <details>
   <summary>详情</summary>

   下载 NLTK 数据包

   ```python
   import nltk
   nltk.download("wordnet")
   nltk.download("punkt_tab")
   nltk.download("cmudict")
   nltk.download("averaged_perceptron_tagger_eng")
   ```

   下载 Spacy 包

   ```bash
   pip install -U pip setuptools wheel
   pip install 'spacy[cuda12x]==3.8.0'
   python -m spacy download en_core_web_md    # 用于生成训练数据
   python -m spacy download en_core_web_trf   # 用于 Object Halbench 评估
   ```

   安装 YOLO 模型依赖：

   ```bash
   pip install git+https://github.com/openai/CLIP.git
   ```

   </details>

## 🔨 数据生成

> 如果只想使用我们发布的数据集，可跳过本节。

1. （可选）检查 `.env` 文件

   可查看 <a href="./utils/.env">.env</a> 配置环境变量。文件在运行时自动加载，大多数项默认注释，可按需修改。

2. 选择用于生成数据的模型

   需选择某个 MLLM 来生成针对该模型的训练数据。我们已支持 LLaVA-v1.5、LLaVA-v1.6 和 Qwen-VL 系列。

   可通过修改 [`utils/setup_utils.py`](./utils/setup_utils.py) 中的 `--model` 参数切换模型，更多细节请参阅 [`model/generator`](./model/generator) 目录。

3. 下载 [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html) 图像，以及用于数据生成的 [dataset](https://huggingface.co/datasets/psp-dada/SENTINEL/blob/main/image_data.jsonl)，并放置于 [`dataset`](dataset) 目录中。

4. 生成训练数据

   运行以下命令，将结果保存在 `./results` 目录：

   ```bash
   python main.py
   ```

5. 数据生成完成

   <details>
   <summary>生成数据详情</summary>

   生成的文件包含两个 `.jsonl` 文件：
   - `<model_name>.jsonl`：用于分析的辅助文件，每行对应一张图像的生成结果，包含：
     - `sentences_cnt`：描述该图像的句子总数
     - `hallu_objects`：采样过程中生成的幻觉物体总数
     - `uncertain_objects`：不确定物体数
     - `nonhallu_objects`：非幻觉物体数

   - `<model_name>_data_pair.jsonl`：偏好样本对文件，每行是一组样本对，包含 `"image_path"`、`"context"`、`"y_win"`、`"y_lose"` 等字段，以及用于分析的附加字段。
   </details>

6. 将训练样本转换为所需格式
   - 若使用 **LLaVA-v1.5**，请用 [`utils/get_llava_v15_data_pair.py`](./utils/get_llava_v15_data_pair.py) 转换，以保持与原仓库一致。
   - 若使用 **LLaVA-v1.6**、**Qwen2-VL** 或 **Qwen2.5-VL**，请用 [`utils/get_llama_factory_data_pair.py`](./utils/get_llama_factory_data_pair.py) 转换为 **LLaMA-Factory** 格式。

## ⚙️ 训练

1. **准备数据**
   - 训练数据
     - 若想复现我们的实验，可使用我们构建的 [SENTINEL 数据集](https://huggingface.co/datasets/psp-dada/SENTINEL)。
     - 若想自建数据集，可使用上节生成的数据。

   - 图像数据
     我们使用 Visual Genome 数据集中的图像进行训练，可从 [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html) 下载并记录路径。

2. **训练**
   - LLaVA-v1.5
     我们基于 HA-DPO 库（其本身基于官方 LLaVA-v1.5 实现）进行修改，以便与先前工作进行公平对比。
     提供了 [训练脚本](./train/models/dpo_llava.sh)，运行以下命令开始 LoRA 训练：

     ```bash
     export INPUT_MODEL=/your/path/to/llava-v1.5-7b/or/13b
     export TRAINING_DATA_PATH=/your/path/to/training/data/file
     export OUTPUT_NAME=/the/name/of/directory/to/save
     export VISUAL_GENOME_PATH=/your/path/to/visual/genome
     bash "train/models/dpo_llava.sh"
     ```

     最终权重将保存在 `./train/results/${OUTPUT_NAME}` 目录中。

   - 其他模型
     对于 **LLaVA-v1.6**、**Qwen2-VL** 或 **Qwen2.5-VL**，我们采用通用的微调框架 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 实现方法。
     训练流程请参阅 [Train SENTINEL via LLaMA-Factory](llamafactory/README.md)。

## 📈 评估

我们严格遵循官方 [LLaVA](https://github.com/haotian-liu/LLaVA) 评估设置，以确保公平比较。详情请参阅官方指南：[Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)。
更多信息请见我们的 [评估说明](/docs/Evaluation.md)。

## 🙏 致谢 <!-- omit in toc -->

- [LLaVA](https://github.com/haotian-liu/LLaVA)：提供了 LLaVA-v1.5 模型的出色的 MLLM 开源项目。
- [HA-DPO](https://github.com/opendatalab/HA-DPO)：一个对多模态模型物体幻觉领域有较大贡献的仓库，我们 LLaVA-v1.5 部分代码基于此。
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)：统一高效的大模型微调框架，我们的 LLaVA-v1.6、Qwen2-VL 和 Qwen2.5-VL 实现基于此。

## 📝 引用

如果我们的模型/代码/数据/论文对您有帮助，请引用我们的论文并为我们点 ⭐️！

```bibtex
@article{peng2025mitigating,
  title={Mitigating Object Hallucinations via Sentence-Level Early Intervention},
  author={Peng, Shangpin and Yang, Senqiao and Jiang, Li and Tian, Zhuotao},
  journal={arXiv preprint arXiv:2507.12455},
  year={2025}
}
```

## 📧 联系我们 <!-- omit in toc -->

如果您有任何问题、意见或建议，欢迎提交 issue 或 PR，共同推动该方向的研究进展。

## License <!-- omit in toc -->

[Apache License 2.0](/LICENSE)
