## Train SENTINEL via LLaMA-Factory

[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) is a widely used framework for fine-tuning LLMs. We adapt our C-DPO training method based on this framework. For convenience, we provide all necessary additions required to integrate our C-DPO training into the original LLaMA-Factory, including training data configuration, training scripts, and the C-DPO object itself.

The setup steps are as follows:

1. Follow the [official LLaMA-Factory tutorial](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#getting-started) to set up the environment. To distinguish it, we recommend naming the conda environment `LLaMA-Factory-SENTINEL`.

2. Replace or update the contents of [`llamafactory/data`](/llamafactory/data) in the original framework with our files. The [`llamafactory/data/ours`](/llamafactory/data/ours/) directory contains training data downloaded from the [SENTINEL Dataset](https://huggingface.co/datasets/psp-dada/SENTINEL), or your own dataset converted into the LLaMA-Factory format in the previous section.

3. Replace or add the contents from [`llamafactory/example`](/llamafactory/example) in this project into the original framework. This directory contains training scripts used in our experiments.

4. Update the corresponding files under [`llamafactory/src/llamafactory/data`](/llamafactory/src/llamafactory/data/) in the original repo.
   
   **Note**: Only three files: [`processor/pairwise.py`](/llamafactory/src/llamafactory/data/processor/pairwise.py), [`converter.py`](/llamafactory/src/llamafactory/data/converter.py), and [`parser.py`](/llamafactory/src/llamafactory/data/parser.py) differ from the original repository. All modifications are clearly marked using `#! <-- ADD HERE START -->` and `#! <-- ADD HERE END -->` comments.

5. install necessary dependencies:
   ```bash
   pip install deepspeed==0.15.4
   ```

After completing the steps above, simply modify the `MODEL_NAME` and `GPU_LIST` fields in the provided `.sh` script of [`examples/SENTINEL`](/llamafactory/examples/SENTINEL), and you're ready to run training.
