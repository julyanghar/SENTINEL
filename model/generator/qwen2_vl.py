from logging import Logger

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2TokenizerFast,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)
from vllm import LLM

from model.utils.gen_utils import GenOutput


class Qwen2VLModel:
    def __init__(
        self,
        use_vllm: bool = True,
        debug: bool = False,
        model_size: str = "7B",
        model_dir: str = "",
        seed: int = 42,
        gpu_util: float = 0.9,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        logger: Logger | None = None,
    ):
        """
        初始化 Qwen2VL 模型，根据版本和大小加载不同的模型。

        Args:
            size: 模型大小，'2B'，'7B' 或 '72B'。
            model_dir: 模型缓存目录。
            device: 使用的设备（CPU 或 GPU）。
        """
        assert model_size in [
            "2B",
            "7B",
            "72B",
        ], "Invalid model size. Choose from '2B', '7B' or '72B', pay attention to the capitalization."
        if logger:
            logger.info(f"Loading Qwen2 VL model with size {model_size} on {device} util {gpu_util*100}%.")
        self.use_vllm = use_vllm
        self.debug = debug
        self.model_size = model_size
        self.model_dir = model_dir
        self.seed = seed
        self.gpu_memory_util = gpu_util
        self.torch_dtype = torch_dtype
        self.device = device
        self.model, self.processor = self._create()
        self.tokenizer: Qwen2TokenizerFast = self.processor.tokenizer
        self.eos_id: list[int] = [self.tokenizer.eos_token_id]
        self.stop_ids: list[int] = self._get_stop_words_ids()

    def _get_stop_words_ids(self) -> list[int]:
        eos_token_id = self.tokenizer.eos_token_id
        stop_words_ids = [self.tokenizer.encode(w)[0] for w in [".", "!", "?", ".\n", ".\n\n"]]
        stop_words_ids.append(eos_token_id)
        return stop_words_ids

    def _create(self) -> tuple[LLM | Qwen2VLForConditionalGeneration, Qwen2VLProcessor]:
        if self.use_vllm:
            return self._create_vllm()
        else:
            return self._create_hf()

    def _create_vllm(self) -> list[LLM, Qwen2VLProcessor]:
        from ..utils.gen_utils import init_vllm

        model_name = f"/home/zhuotaotian/psp/llm/utils/models/repo/Qwen2-VL-2B-Instruct"

        llm: LLM = init_vllm(
            model_name,
            self.model_dir,
            self.gpu_memory_util,
            self.torch_dtype,
            self.seed,
            model_context_len=20480,
            enforce_eager=self.debug,
        )

        processor: Qwen2VLProcessor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=self.model_dir,
            padding_side="left",
        )

        return llm, processor

    def _create_hf(self) -> tuple[Qwen2VLForConditionalGeneration, Qwen2VLProcessor]:
        model_name = f"Qwen/Qwen2-VL-{self.model_size}-Instruct"

        model: Qwen2VLForConditionalGeneration = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=self.model_dir,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            device_map=self.device,
            attn_implementation="flash_attention_2" if "cuda" in str(self.device) else None,
        ).eval()
        if not self.debug:
            model = torch.compile(model)

        processor: Qwen2VLProcessor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=self.model_dir,
            padding_side="left",
        )

        return model, processor

    def gen(
        self,
        images: Image.Image | list[Image.Image],
        users: str | list[str] = "Describe this image.",
        assistants: str | list[str] = "",
        do_sample: bool = False,
        n: int = 5,
        temp: float = 0.3,
        max_tokens: int = 512,
        force_list: bool = False,
        single_sentence: bool = False,
    ) -> list[str] | list[list[str]]:
        """
        生成图片的描述。
        """
        from ..utils.gen_utils import gen_hf, gen_vllm, u_a_to_prompts
        from ..utils.utils import ensure_lists

        images, users, assistants = ensure_lists(images, users, assistants)

        # modifier 去掉 <|im_end|>\n 共 11 个字符
        prompts: list[str] = u_a_to_prompts(users, assistants, self.processor, lambda x: x[:-11])
        stop_token_ids = self.stop_ids if single_sentence else self.eos_id
        if self.use_vllm:
            out: GenOutput = gen_vllm(
                self.model,
                images,
                prompts,
                do_sample,
                temp,
                max_tokens,
                stop_token_ids,
                n=n,
                strip_slash_n=False,
            )
        else:
            out: GenOutput = gen_hf(
                self.model,
                self.processor,
                self.device,
                self.torch_dtype,
                images,
                prompts,
                do_sample,
                temp,
                max_tokens,
                eos_token_id=stop_token_ids,
                strip_slash_n=False,
            )

        return out.maybe_change_to_list(force_list)
