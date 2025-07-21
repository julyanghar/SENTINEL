from logging import Logger

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

from ..utils.utils import maybe_return_ls


class SGParser:
    def __init__(
        self,
        debug: bool = False,
        size: str = "base",
        model_dir: str = "",
        device: str = "cpu",
        logger: Logger | None = None,
    ):
        """
        Initialize the SGParser with the specified model size.

        Args:
            size: The size of the parser ('small', 'base', or 'large') with parameters 77M/248M/783M.
            model_dir: Directory for model caching.
            device: The device to run the model on (e.g., CPU or GPU).
        """
        assert size in ["small", "base", "large"], "Invalid model size. Choose from 'small', 'base', or 'large'."

        if logger:
            logger.info(f"Loading SG parser model with size {size} on {device}")
        self.debug = debug
        self.size = size
        self.model_dir = model_dir
        self.device = device
        self.model, self.tokenizer = self._create()

    def _create(self) -> tuple[T5ForConditionalGeneration, T5TokenizerFast]:
        """
        Create a SG parser model and tokenizer based on the specified size.

        Returns:
            A tuple containing the model and tokenizer.
        """
        model_name = f"lizhuang144/flan-t5-{self.size}-VG-factual-sg"

        model: T5ForConditionalGeneration = (
            AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=self.model_dir,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            .to(self.device)
            .eval()
        )
        if not self.debug:
            model = torch.compile(model)

        processor: T5TokenizerFast = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=self.model_dir,
            padding_side="left",
        )

        return model, processor

    def _gen(self, discriptions: list[str], max_length: int = 200) -> list[str]:
        with torch.inference_mode():
            encoded_inputs = self.tokenizer(
                [f"Generate Scene Graph: {d.strip('.').strip().lower()}" for d in discriptions],
                max_length=max_length,
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).to(self.device)

            generated_ids = self.model.generate(
                **encoded_inputs,
                use_cache=True,
                num_beams=1,
                max_length=max_length,
            )

            decoded_outputs: list[str] = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,  # remove leading/trailing whitespaces
            )
        return decoded_outputs

    def _get_text_graphs_from_str(self, model_output: str) -> list[list[str]]:
        """
        Extract text scene graphs from the output string.

        Args:
            output: The single output string from the SG parser model.
        """
        return [triplet.split(", ") for triplet in model_output[2:-2].split(" ), ( ")]

    def pharse(
        self,
        discriptions: list[str] | str,
        force_list: bool = False,
    ) -> list[list[list[str]]] | list[list[str]]:
        """
        Generate text scene graphs that convert the input descriptions
        into a series of triplets representing relationships in the scene.

        Args:
            discriptions: Input description texts, can be a single string or a list of strings.
            force_list: Whether to force the output to be a list of text scene graphs.

        Returns:
            A list of text scene graphs, where each graph is a list of triplets, where each triplet is a list of strings.
        """
        if isinstance(discriptions, str):
            discriptions = [discriptions]

        # 初始化
        text_graphs: list[list[list[str]]] = [[] if d == "" else None for d in discriptions]

        valid_desc: list[str] = [d for d in discriptions if d]
        if valid_desc:
            decoded_outputs: list[str] = self._gen(valid_desc)

            result_index = 0
            for i in range(len(text_graphs)):
                if text_graphs[i] is None:
                    text_graphs[i] = self._get_text_graphs_from_str(decoded_outputs[result_index])
                    result_index += 1

        return maybe_return_ls(force_list, text_graphs)
