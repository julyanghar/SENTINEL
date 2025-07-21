# Evaluation

<a href="/docs/Evaluation_zh.md">中文</a> | <b>English</b>

**This document provides instructions for evaluating the models.**

For all the models, we evaluate on a diverse set of 7 benchmarks. To ensure the reproducibility, we evaluate the models with greedy decoding.

## Scripts

Before preparing benchmark-specific data, **you MUST first download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing)** from LLaVA. It contains custom annotations, scripts, and the prediction files with LLaVA v1.5. Extract to [`llava/data`](/llava/data/). This also provides a general structure for all datasets.

### Object Halbench

1. Prepare COCO2014 annotations
   ```shell
    mkdir llava/data/MSCOCO/coco2014
    cd llava/data/MSCOCO/coco2014

    wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    unzip annotations_trainval2014.zip
    ```
2. Download the question file follow [the instruction](https://github.com/RLHF-V/RLHF-V?tab=readme-ov-file#object-halbench) and put it under [`llava/data/eval/object_halbench`](/llava/data/eval/object_halbench). The file should be named `Object_HalBench.jsonl`.

3. Inference using [`llava/eval_script/eval_object_halbench.sh`](/llava/eval_script/eval_object_halbench.sh).

### AMBER

1. Download [`images`](https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view) and extract it under [`llava/data/eval/AMBER/images`](/llava/data/eval/AMBER/images).

2. Download the `data` folder of the [AMBER official repo](https://github.com/junyangwang0410/AMBER/tree/master/data) to [`llava/data/eval/AMBER/data`](/llava/data/eval/AMBER/data).

3. Inference the *Discriminative part* using [`llava/eval_script/eval_amber_dis.sh`](/llava/eval_script/eval_amber_dis.sh) or *Generative part* using [`llava/eval_script/eval_amber_gen.sh`](/llava/eval_script/eval_amber_gen.sh).

### HallusionBench

1. Download [`images`](https://drive.google.com/file/d/1eeO1i0G9BSZTE1yd5XeFwmrbe1hwyf_0/view) and extract it under `llava/data/eval/HallusionBench/images` (`llava/data/eval/HallusionBench/images` should directly contain dirs `VD` and `VS`).

2. Download [`HallusionBench.json`](https://github.com/tianyi-lab/HallusionBench/blob/main/HallusionBench.json) and put it under [`llava/eval/utils/HallusionBench`](llava/eval/utils/HallusionBench).

3. Use  [`convert_questions_file.py`](/llava/eval/utils/HallusionBench/convert_questions_file.py) to generate the questions file from `HallusionBench.json` to `questions.jsonl`.

4. Inference using [`llava/eval_script/eval_hallusion_bench.sh`](/llava/eval_script/eval_hallusion_bench.sh).

### VQAv2

1. Download [`test2015`](http://images.cocodataset.org/zips/test2015.zip) of COCO and put it under [`llava/data/MSCOCO`](/llava/data/MSCOCO).

2. Inference using [`llava/eval_script/eval_vqav2.sh`](/llava/eval_script/eval_vqav2.sh).

3. Submit the result files in `llava/data/eval/vqav2/answers_upload/llava_vqav2_mscoco_test-dev2015` to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission).

### ScienceQA

1. Under `llava/data/eval/scienceqa`, download `images`, `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).

2. Inference using [`llava/eval_script/eval_ScienceQA.sh`](/llava/eval_script/eval_ScienceQA.sh).

### TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `llava/data/eval/textvqa`.

2. Inference using [`llava/eval_script/eval_textvqa.sh`](/llava/eval_script/eval_textvqa.sh).

### MM-Vet

1. Download and extract the images [`mm-vet.zip`](https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip) to `llava/data/eval/mm-vet`.

2. Inference using [`llava/eval_script/eval_mmvet.sh`](/llava/eval_script/eval_mmvet.sh).

3. Evaluate the predictions in `llava/data/eval/mm-vet/results` using the [official evaluator](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator).