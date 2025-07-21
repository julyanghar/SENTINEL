import warnings
from logging import Logger

import torch
from PIL import Image
from transformers import (
    BertTokenizerFast,
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
)
from transformers.models.grounding_dino.modeling_grounding_dino import (
    GroundingDinoObjectDetectionOutput,
)

from ..utils.utils import ensure_lists, is_dino_sentences_equal, maybe_return_ls

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25


class DINO:
    def __init__(
        self,
        size: str = "base",
        model_dir: str = "",
        torch_dtype: torch.dtype | str = "auto",  # Don't change
        device: torch.device | str = "cpu",
        logger: Logger | None = None,
    ):
        """
        Initialize the DINO detector with the specified model size.

        Args:
            size: The size of the model ('tiny' or 'base') with parameters 172M or 233M.
            model_dir: Directory for model caching.
            device: The device to run the model on (e.g., CPU or GPU).
        """
        assert size in [
            "tiny",
            "base",
        ], "Invalid model size. Choose from 'tiny' or 'base'."

        if logger:
            logger.info(f"Loading grounding DINO model with size {size} on {device}")
        self.size = size
        self.model_dir = model_dir
        self.torch_dtype = torch_dtype
        self.device = device
        self.model, self.processor = self._create()
        self.tokenizer: BertTokenizerFast = self.processor.tokenizer
        self.empty_dino_result: dict = {
            "scores": torch.tensor([]),
            "boxes": torch.tensor([]),
            "labels": [],
        }

    def _create(self) -> tuple[GroundingDinoForObjectDetection, GroundingDinoProcessor]:
        """
        Create a grounding DINO model and processor based on the specified size.

        Returns:
            A tuple containing the model and processor.
        """
        model_name = f"IDEA-Research/grounding-dino-{self.size}"

        model: GroundingDinoForObjectDetection = (
            GroundingDinoForObjectDetection.from_pretrained(
                model_name,
                cache_dir=self.model_dir,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                disable_custom_kernels=True,
            )
            .to(self.device)
            .eval()
        )

        processor: GroundingDinoProcessor = GroundingDinoProcessor.from_pretrained(
            model_name,
            cache_dir=self.model_dir,
            # Do not set padding_side!
        )

        return model, processor

    def detect(
        self,
        images: Image.Image | list[Image.Image],
        captions: str | list[str],
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        force_list: bool = False,
    ) -> list[dict[str]] | dict[str]:
        """
        Use the DINO model to detect objects in an image based on a given caption.

        Args:
            images: Image path or PIL.Image object.
            captions: Caption for the detection (must be lowercased and end with a dot, the spaces won't affect the result).
            box_threshold: Threshold for box detection.
            text_threshold: Threshold for text detection.
            force_list: Whether to force the return value to be a list.
            return_confidence: Whether to form the return value to the confidence for each word in the caption.
        Returns:
            A dictionary with detection results including scores, boxes, and labels.
            {'scores': torch.Tensor, 'boxes': torch.Tensor, 'labels': list[str]}.
            Default values for empty captions.
        """
        images, captions = ensure_lists(images, captions)

        results: list = [None if caption else self.empty_dino_result for caption in captions]  # None is a placeholder

        # Filter out empty captions and corresponding images
        valid_captions: list[str] = [c for c in captions if c]
        filtered_images: list[Image.Image] = [img for i, img in enumerate(images) if captions[i]]

        # Process images if there are valid captions
        if valid_captions:
            detection_res = self._detect_wo_repetition(
                filtered_images,
                valid_captions,
                box_threshold,
                text_threshold,
            )

            # Fill valid results into the results list
            result_index = 0
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = detection_res[result_index] if isinstance(detection_res, list) else detection_res
                    result_index += 1

        return maybe_return_ls(force_list, results)

    def _detect_wo_repetition(
        self,
        images: list[Image.Image],
        captions: list[str],
        box_threshold: float = BOX_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
    ) -> list[dict[str]]:
        # 收集唯一的图像和文本对
        unique_pairs: dict[tuple[str, str], tuple[Image.Image, str]] = {}
        for image, caption in zip(images, captions):
            for key in unique_pairs.keys():
                if key[0] == image.tobytes():
                    if is_dino_sentences_equal(key[1], caption):
                        break
            else:
                unique_pairs[(image.tobytes(), caption)] = (image, caption)

        unique_images, unique_captions = zip(*unique_pairs.values())

        # 调用 _detect 方法，处理所有唯一的图像文本对
        output: list[dict] = self._detect(
            list(unique_images),
            list(unique_captions),
            box_threshold,
            text_threshold,
        )

        # 构建结果列表，按原顺序填充
        results: list[dict[str]] = []
        for image, caption in zip(images, captions):
            for key, _ in unique_pairs.items():
                if key[0] == image.tobytes() and is_dino_sentences_equal(key[1], caption):
                    results.append(output[list(unique_pairs.keys()).index(key)])

        return results

    def _detect(
        self,
        images: list[Image.Image],
        captions: list[str],
        box_threshold: float = BOX_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
    ) -> list[dict[str]]:
        with torch.inference_mode():
            encoded_inputs = self.processor(
                images=images,
                text=captions,
                max_length=200,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            outputs: GroundingDinoObjectDetectionOutput = self.model(**encoded_inputs)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                target_sizes = [image.size[::-1] for image in images]
                # 字典中包含 "scores", "boxes", "labels" 三个字段
                results: list[dict] = self.processor.post_process_grounded_object_detection(
                    outputs,
                    encoded_inputs["input_ids"],
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes,
                )
        return results
