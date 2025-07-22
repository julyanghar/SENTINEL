from logging import Logger

import torch
from PIL import Image
from transformers import (
    LlamaTokenizerFast,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    LlavaProcessor,
)
from vllm import LLM

from ..utils.gen_utils import GenOutput, gen_hf, gen_vllm, get_stop_words_ids, init_vllm, u_a_to_prompts
from ..utils.utils import ensure_lists


class LlavaModel:
    def __init__(
        self,
        use_vllm: bool = True,
        debug: bool = False,
        version: str = "1.5",
        model_size: str = "7b",
        model_dir: str = "",
        seed: int = 42,
        gpu_util: float = 0.9,
        torch_dtype: torch.dtype = torch.float16,
        device: str = "cpu",
        logger: Logger | None = None,
    ):
        """
        Initialize LLaVA model, load different models according to version and size.

        Args:
            use_vllm (bool): Whether to use vLLM.
            debug (bool): Whether to enable debug mode.
            version (str): Model version, can be '1.5' or '1.6'.
            model_size (str): Model size, '7b' or '13b'.
            model_dir (str): Model cache directory.
            seed (int): Random seed.
            gpu_util (float): GPU memory utilization.
            torch_dtype (torch.dtype): PyTorch data type.
            device (str): Device to use (CPU or GPU).
            logger (Logger | None): Logger.
        """
        assert version in ["1.5", "1.6"], "Invalid LLaVA version. Choose from '1.5' or '1.6'."
        assert model_size in ["7b", "13b"], "Invalid size. Pay attention to the capitalization."

        if logger:
            logger.info(f"Loading LLaVA {version} model with size {model_size} on {device} util {gpu_util*100}%.")
        self.use_vllm = use_vllm
        self.debug = debug
        self.version = version
        self.model_size = model_size
        self.model_dir = model_dir
        self.seed = seed
        self.gpu_memory_util = gpu_util
        self.torch_dtype = torch_dtype
        self.device = device
        self.model, self.processor = self._create()
        self.tokenizer: LlamaTokenizerFast = self.processor.tokenizer
        self.eos_id: list[int] = [self.tokenizer.eos_token_id]
        self.stop_ids: list[int] = get_stop_words_ids(self.tokenizer)

    def _create(
        self,
    ) -> tuple[
        LLM | LlavaForConditionalGeneration | LlavaNextForConditionalGeneration, LlavaProcessor | LlavaNextProcessor
    ]:
        if self.use_vllm:
            return self._create_vllm()
        else:
            return self._create_hf()

    def _create_vllm(self) -> list[LLM, LlavaProcessor | LlavaNextProcessor]:
        if self.version == "1.5":
            model_name: str = f"llava-hf/llava-1.5-{self.model_size}-hf"
            processor_class: type = LlavaProcessor
            model_context_len = 1024
        else:
            model_name: str = f"llava-hf/llava-v1.6-vicuna-{self.model_size}-hf"
            processor_class: type = LlavaNextProcessor
            model_context_len = 3500

        llm: LLM = init_vllm(
            model_name,
            self.model_dir,
            self.gpu_memory_util,
            self.torch_dtype,
            self.seed,
            model_context_len=model_context_len,
            enforce_eager=self.debug,
        )
        processor = processor_class.from_pretrained(
            model_name,
            cache_dir=self.model_dir,
            padding_side="left",
            use_fast=True,
        )

        return llm, processor

    def _create_hf(
        self,
    ) -> tuple[LlavaForConditionalGeneration | LlavaNextForConditionalGeneration, LlavaProcessor | LlavaNextProcessor]:
        if self.version == "1.5":
            model_name = f"llava-hf/llava-1.5-{self.model_size}-hf"
            model_class: type = LlavaForConditionalGeneration
            processor_class: type = LlavaProcessor
        else:
            model_name = f"llava-hf/llava-v1.6-vicuna-{self.model_size}-hf"
            model_class: type = LlavaNextForConditionalGeneration
            processor_class: type = LlavaNextProcessor

        attn_implementation = "flash_attention_2"

        model = model_class.from_pretrained(
            model_name,
            cache_dir=self.model_dir,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            device_map=self.device,
            attn_implementation=attn_implementation,
        ).eval()

        if not self.debug:
            model = torch.compile(model)

        processor = processor_class.from_pretrained(
            model_name,
            cache_dir=self.model_dir,
            padding_side="left",
            patch_size=model.config.vision_config.patch_size,
            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
            use_fast=True,
        )

        return model, processor

    def gen(
        self,
        images: Image.Image | list[Image.Image],
        users: str | list[str] = "Please describe this image in detail.",
        assistants: str | list[str] = "",
        do_sample: bool = False,
        n: int = 1,
        temp: float = 0.3,
        max_tokens: int = 512,
        force_list: bool = False,
        single_sentence: bool = False,
        return_log_probs=False,
    ) -> list[str] | list[list[str]]:
        """
        Generate responses based on input images and prompts.
        """

        images, users, assistants = ensure_lists(images, users, assistants)

        # modifier: remove the last character
        prompts: list[str] = u_a_to_prompts(users, assistants, self.processor, lambda x: x[:-1])
        stop_token_ids: list[int] = self.stop_ids if single_sentence else self.eos_id
        if self.use_vllm:
            out: GenOutput = gen_vllm(self.model, images, prompts, do_sample, temp, max_tokens, stop_token_ids, n=n)
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
                n=n,
                eos_token_id=stop_token_ids,
                truncate_gen_ids=True,
            )

        return out.maybe_change_to_list(force_list)
