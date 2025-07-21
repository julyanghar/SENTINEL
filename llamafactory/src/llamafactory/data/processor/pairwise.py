# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen

if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class PairwiseDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        #! <-- ADD HERE START -->
        context: Optional[str] = None,
        #! <-- ADD HERE END -->
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        #! <-- ADD HERE START -->
        def mark_context(input_ids, labels, context_ids):
            context_len = len(context_ids)
            for i in range(len(input_ids) - context_len + 1):
                if input_ids[i : i + context_len] == context_ids:
                    labels[: i + context_len] = [IGNORE_INDEX] * (i + context_len)
                    break

        if context is not None and isinstance(context, str):
            response[0]["content"] = context + " " + response[0]["content"]
            response[1]["content"] = context + " " + response[1]["content"]
        #! <-- ADD HERE END -->

        chosen_messages = self.template.mm_plugin.process_messages(
            prompt + [response[0]], images, videos, audios, self.processor
        )
        rejected_messages = self.template.mm_plugin.process_messages(
            prompt + [response[1]], images, videos, audios, self.processor
        )
        prompt_ids, chosen_ids = self.template.encode_oneturn(self.tokenizer, chosen_messages, system, tools)
        _, rejected_ids = self.template.encode_oneturn(self.tokenizer, rejected_messages, system, tools)

        if self.template.efficient_eos:
            chosen_ids += [self.tokenizer.eos_token_id]
            rejected_ids += [self.tokenizer.eos_token_id]

        #! <-- ADD HERE START -->
        if context:
            period_id: int = self.tokenizer.convert_tokens_to_ids(".")
            # chosen_id 和 reject id 都将最后一个 period 之后的内容截断（即去掉 EOS token）
            if period_id in chosen_ids:
                last_period_index = len(chosen_ids) - 1 - chosen_ids[::-1].index(period_id)
                chosen_ids = chosen_ids[: last_period_index + 1]

            if period_id in rejected_ids:
                last_period_index = len(rejected_ids) - 1 - rejected_ids[::-1].index(period_id)
                rejected_ids = rejected_ids[: last_period_index + 1]
        #! <-- ADD HERE END -->

        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        # consider the response is more important
        source_len, target_len = infer_seqlen(
            len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), self.data_args.cutoff_len
        )
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids

        #! <-- ADD HERE START -->
        if context:
            context_ids = self.tokenizer.encode(context, add_special_tokens=False)
            # 在 chosen_input_ids 和 rejected_input_ids 中找到 context_ids，在 labels 对应的位置标记为 IGNORE_INDEX，之前的位置也都标记为 IGNORE_INDEX
            mark_context(chosen_input_ids, chosen_labels, context_ids)
            mark_context(rejected_input_ids, rejected_labels, context_ids)
        #! <-- ADD HERE END -->

        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                #! <-- ADD HERE START -->
                context=examples["_context"][i] if "_context" in examples else None,
                #! <-- ADD HERE END -->
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
        valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
        print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
        print(
            "chosen_inputs:\n{}".format(self.tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False))
        )
        print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
        print(f"chosen_labels:\n{self.tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
        print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
        print(
            "rejected_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)
            )
        )
        print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
        print(f"rejected_labels:\n{self.tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")
