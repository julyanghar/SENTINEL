# 评测说明

<b>中文</b> | <a href="/docs/Evaluation.md">English</a>

**本文档提供了模型评测的详细说明。**

对于所有模型，我们在 7 个多样化的基准测试集上进行评测。为保证可复现性，所有模型均采用贪婪解码进行评测。

## 脚本

在准备各基准测试集数据前，**你必须先从 LLaVA 下载 [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing)**。该压缩包包含自定义标注、脚本以及 LLaVA v1.5 的预测文件。解压到 [`llava/data`](/llava/data/)。这也为所有数据集提供了统一的数据结构。

### Object Halbench

1. 准备 COCO2014 标注文件

   ```shell
    mkdir llava/data/MSCOCO/coco2014
    cd llava/data/MSCOCO/coco2014

    wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    unzip annotations_trainval2014.zip
   ```

2. 按照 [说明](https://github.com/RLHF-V/RLHF-V?tab=readme-ov-file#object-halbench) 下载问题文件，并放置到 [`llava/data/eval/object_halbench`](/llava/data/eval/object_halbench) 下，文件名应为 `Object_HalBench.jsonl`。

3. 使用 [`llava/eval_script/eval_object_halbench.sh`](/llava/eval_script/eval_object_halbench.sh) 进行推理。

### AMBER

1. 下载 [`images`](https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view) 并解压到 [`llava/data/eval/AMBER/images`](/llava/data/eval/AMBER/images)。

2. 下载 [AMBER 官方仓库](https://github.com/junyangwang0410/AMBER/tree/master/data) 的 `data` 文件夹到 [`llava/data/eval/AMBER/data`](/llava/data/eval/AMBER/data)。

3. 使用 [`llava/eval_script/eval_amber_dis.sh`](/llava/eval_script/eval_amber_dis.sh) 进行判别部分推理，或使用 [`llava/eval_script/eval_amber_gen.sh`](/llava/eval_script/eval_amber_gen.sh) 进行生成部分推理。

### HallusionBench

1. 下载 [`images`](https://drive.google.com/file/d/1eeO1i0G9BSZTE1yd5XeFwmrbe1hwyf_0/view) 并解压到 `llava/data/eval/HallusionBench/images`（`llava/data/eval/HallusionBench/images` 下应直接包含 `VD` 和 `VS` 文件夹）。

2. 下载 [`HallusionBench.json`](https://github.com/tianyi-lab/HallusionBench/blob/main/HallusionBench.json) 并放置到 [`llava/eval/utils/HallusionBench`](llava/eval/utils/HallusionBench)。

3. 使用 [`convert_questions_file.py`](/llava/eval/utils/HallusionBench/convert_questions_file.py) 将 `HallusionBench.json` 转换为 `questions.jsonl`。

4. 使用 [`llava/eval_script/eval_hallusion_bench.sh`](/llava/eval_script/eval_hallusion_bench.sh) 进行推理。

### VQAv2

1. 下载 COCO [`test2015`](http://images.cocodataset.org/zips/test2015.zip) 并放置到 [`llava/data/MSCOCO`](/llava/data/MSCOCO)。

2. 使用 [`llava/eval_script/eval_vqav2.sh`](/llava/eval_script/eval_vqav2.sh) 进行推理。

3. 将 `llava/data/eval/vqav2/answers_upload/llava_vqav2_mscoco_test-dev2015` 下的结果文件提交到 [评测服务器](https://eval.ai/web/challenges/challenge-page/830/my-submission)。

### ScienceQA

1. 在 `llava/data/eval/scienceqa` 下，从 ScienceQA [仓库](https://github.com/lupantech/ScienceQA) 的 `data/scienceqa` 文件夹下载 `images`、`pid_splits.json`、`problems.json`。

2. 使用 [`llava/eval_script/eval_ScienceQA.sh`](/llava/eval_script/eval_ScienceQA.sh) 进行推理。

### TextVQA

1. 下载 [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) 和 [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)，解压到 `llava/data/eval/textvqa`。

2. 使用 [`llava/eval_script/eval_textvqa.sh`](/llava/eval_script/eval_textvqa.sh) 进行推理。

### MM-Vet

1. 下载并解压图片 [`mm-vet.zip`](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) 到 `llava/data/eval/mm-vet`。

2. 使用 [`llava/eval_script/eval_mmvet.sh`](/llava/eval_script/eval_mmvet.sh) 进行推理。

3. 使用 [官方评测器](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator) 对 `llava/data/eval/mm-vet/results` 下的预测结果进行评测。
