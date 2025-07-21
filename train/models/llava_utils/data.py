import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.utils.rnn as rnn_utils
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, PreTrainedTokenizer

from llava.constants import IGNORE_INDEX
from llava.train.train import preprocess, preprocess_multimodal
from train.models.llava_utils.arguments import LlavaDataArguments
from train.models.llava_utils.utils import read_json


class LazySupervisedDataset(Dataset):
    """
    A lazy dataset for supervised fine-tuning.
    """

    def __init__(
        self,
        vg_path: str,
        train_data_path: str,
        tokenizer: PreTrainedTokenizer,
        data_args: LlavaDataArguments,
        seed: int = 42,
    ):
        super().__init__()

        vg_image_data: list[dict] = json.load(open(os.path.join(vg_path, "image_data.json")))
        self.id2path: dict[int, str] = {
            d["image_id"]: os.path.join(vg_path, d["url"].split("/")[-2], d["url"].split("/")[-1])
            for d in vg_image_data
        }

        # preprocess
        desc_data_dict: dict[str, dict] = read_json(train_data_path)
        random.seed(seed)
        if train_data_path.endswith(".jsonl"):
            data: list[dict] = self.desc_w_context_jsonl_process(desc_data_dict)
        else:
            data: list[dict] = self.desc_w_context_process(desc_data_dict)
        random.shuffle(data)

        self.tokenizer = tokenizer
        self.data = data[:100]
        self.data_args = data_args

        del desc_data_dict, vg_image_data, self.id2path

    def desc_w_context_process(self, desc_data: dict[str, list[dict[str, str]]]) -> list[dict]:
        # A list of `{'id': 2324811, 'image': '...', 'chosen_conversations': [{...}, {...}], 'reject_conversations': [{...}, {...}]}`
        desc_data_dict: list[dict[str]] = []
        for image_id in desc_data.keys():
            for pair in desc_data[image_id]:
                y_win = pair["y_win"] if "y_win" in pair else pair["win"]
                y_lose = pair["y_lose"] if "y_lose" in pair else pair["lose"]
                question = "<image>\n" + pair["question"]

                has_context = "context" in pair and pair["context"] is not None
                chosen = pair["context"] + " " + y_win if has_context else y_win
                rejected = pair["context"] + " " + y_lose if has_context else y_lose
                desc_data_dict.append(
                    {
                        "id": int(image_id),
                        "image": self.id2path[int(image_id)],
                        "chosen_conversations": [
                            {"from": "human", "value": question},
                            {"from": "gpt", "value": chosen},
                        ],
                        "reject_conversations": [
                            {"from": "human", "value": question},
                            {"from": "gpt", "value": rejected},
                        ],
                        "has_context": has_context,
                        "context": pair["context"],
                    }
                )

        return desc_data_dict

    def desc_w_context_jsonl_process(self, desc_data: list[dict[str]]) -> list[dict]:
        # A list of `{'id': 2324811, 'image': '...', 'chosen_conversations': [{...}, {...}], 'reject_conversations': [{...}, {...}]}`
        desc_data_dict: list[dict[str]] = []
        for pair in desc_data:
            y_win = pair["y_win"] if "y_win" in pair else pair["win"]
            y_lose = pair["y_lose"] if "y_lose" in pair else pair["lose"]
            question = "<image>\n" + pair["question"]

            has_context = "context" in pair and pair["context"] is not None
            chosen = pair["context"] + " " + y_win if has_context else y_win
            rejected = pair["context"] + " " + y_lose if has_context else y_lose
            image_id = int(pair["image_id"])

            desc_data_dict.append(
                {
                    "id": image_id,
                    "image": self.id2path[image_id],
                    "chosen_conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": chosen},
                    ],
                    "reject_conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": rejected},
                    ],
                    "has_context": has_context,
                    "context": pair["context"] if has_context else None,
                }
            )

        return desc_data_dict

    def __len__(self):
        return len(self.data)

    @property
    def lengths(self):
        length_list = []
        for sample in self.data:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.data:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "images" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def handle_data(self, data: dict, image: Image.Image, processor: CLIPImageProcessor):
        if self.data_args.image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image: torch.Tensor = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image: torch.Tensor = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        chosen_sources: list[list[dict]] = preprocess_multimodal(
            copy.deepcopy([data["chosen_conversations"]]), self.data_args
        )

        has_image = "image" in data
        chosen_data_dict: dict[str, torch.Tensor] = preprocess(chosen_sources, self.tokenizer, has_image=has_image)
        data_dict = dict(
            chosen_input_ids=chosen_data_dict["input_ids"][0],
            chosen_labels=chosen_data_dict["labels"][0],
            images=image,
        )

        if data["reject_conversations"]:
            reject_sources: list[list[dict]] = preprocess_multimodal(
                copy.deepcopy([data["reject_conversations"]]), self.data_args
            )
            reject_data_dict: dict[str, torch.Tensor] = preprocess(reject_sources, self.tokenizer, has_image=has_image)
            data_dict.update(
                {
                    "reject_input_ids": reject_data_dict["input_ids"][0],
                    "reject_labels": reject_data_dict["labels"][0],
                }
            )

        if "has_context" in data and data["has_context"] and data["context"]:
            context = data["context"]
            context_ids = self.tokenizer.encode(context, add_special_tokens=False, return_tensors="pt")[0]

            context_len = len(context_ids)
            chosen_input_ids: torch.Tensor = data_dict["chosen_input_ids"]
            for i in range(len(chosen_input_ids) - context_len + 1):
                if torch.equal(chosen_input_ids[i : i + context_len], context_ids):
                    data_dict["chosen_labels"][i : i + context_len] = IGNORE_INDEX
                    break

            # 去掉最后一个 <EOS> token 的预测
            data_dict["chosen_input_ids"] = data_dict["chosen_input_ids"][:-1]
            data_dict["chosen_labels"] = data_dict["chosen_labels"][:-1]

            if "reject_input_ids" in data_dict:
                reject_input_ids = data_dict["reject_input_ids"]
                for i in range(len(reject_input_ids) - context_len + 1):
                    if torch.equal(reject_input_ids[i : i + context_len], context_ids):
                        data_dict["reject_labels"][i : i + context_len] = IGNORE_INDEX
                        break

                data_dict["reject_input_ids"] = data_dict["reject_input_ids"][:-1]
                data_dict["reject_labels"] = data_dict["reject_labels"][:-1]

        return data_dict

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data: dict = self.data[idx]

        image_file: str = self.data[idx]["image"]
        image_folder: str = self.data_args.image_folder
        processor: CLIPImageProcessor = self.data_args.image_processor
        image: Image.Image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")

        data_dict = self.handle_data(data, image, processor)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __post_init__(self):
        self.pad_token_id: int = self.tokenizer.pad_token_id
        self.max_length: int = self.tokenizer.model_max_length

    def pad_and_truncate(self, sequences, padding_value):
        padded = rnn_utils.pad_sequence(sequences, batch_first=True, padding_value=padding_value)
        return padded[:, : self.max_length]

    def __call__(self, instances: list[dict]) -> Dict[str, torch.Tensor]:
        chosen_input_ids, chosen_labels, reject_input_ids, reject_labels = tuple(
            [i[key] for i in instances]
            for key in ("chosen_input_ids", "chosen_labels", "reject_input_ids", "reject_labels")
        )
        # Pad and truncate inputs and labels
        chosen_input_ids = self.pad_and_truncate(chosen_input_ids, padding_value=self.pad_token_id)
        chosen_labels = self.pad_and_truncate(chosen_labels, padding_value=IGNORE_INDEX)
        reject_input_ids = self.pad_and_truncate(reject_input_ids, padding_value=self.pad_token_id)
        reject_labels = self.pad_and_truncate(reject_labels, padding_value=IGNORE_INDEX)

        batch = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            reject_input_ids=reject_input_ids,
            reject_labels=reject_labels,
            chosen_attention_mask=chosen_input_ids.ne(self.pad_token_id),
            reject_attention_mask=reject_input_ids.ne(self.pad_token_id),
        )
        if "sft_images" in instances[0]:
            sft_input_ids, sft_labels = tuple([i[key] for i in instances] for key in ("sft_input_ids", "sft_labels"))
            # Pad inputs and labels
            sft_input_ids = self.pad_and_truncate(sft_input_ids, padding_value=self.pad_token_id)
            sft_labels = self.pad_and_truncate(sft_labels, padding_value=IGNORE_INDEX)
            batch.update(
                dict(
                    sft_input_ids=sft_input_ids,
                    sft_labels=sft_labels,
                    sft_attention_mask=sft_input_ids.ne(self.pad_token_id),
                )
            )

        images = [instance["images"] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch["images"] = torch.stack(images)
        else:
            batch["images"] = images

        if "sft_images" in instances[0]:
            sft_images = [instance["sft_images"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in sft_images):
                batch["sft_images"] = torch.stack(sft_images)
            else:
                batch["sft_images"] = sft_images

        return batch
