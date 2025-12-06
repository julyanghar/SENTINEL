"""
Mask DPO 数据集模块
==================

基于图像遮挡的 DPO 训练数据集。与原版 LazySupervisedDataset 的主要区别：
1. 数据格式不同：y_lose=None, context=None
2. 同时使用原图 (images) 和遮挡图 (masked_images) 作为输入
3. 训练目标：对比原图和遮挡图的输出，让模型学会不提及被遮挡的物体

数据格式 (mask_output.json):
    {
        "原图路径": [  # key 是原图路径（如 COCO_train2014_000000123456.jpg）
            {
                "question": "What is this photo about?",
                "context": null,
                "y_win": "The correct description...",
                "y_lose": null,
                "type": "y+",
                "masked_image": "COCO_train2014_000000123456_masked_person.jpg"  # 遮挡图文件名
            },
            ...
        ],
        ...
    }

训练原理：
- images: 原始图像，用于计算 chosen log probability
- masked_images: 遮挡后的图像，用于计算 masked_chosen log probability
- 通过对比两者的 log probability 来训练模型
"""

import copy
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


class MaskLazySupervisedDataset(Dataset):
    """
    基于遮挡图像的 DPO 训练数据集
    
    特点：
    - 同时加载原图 (images) 和遮挡图 (masked_images)
    - y_lose 设为 None（不使用传统的 rejected 样本）
    - 训练时对比原图和遮挡图的模型输出
    
    Attributes:
        tokenizer: 分词器
        data: 处理后的训练数据列表
        data_args: 数据相关参数
    """

    def __init__(
        self,
        train_data_path: str,
        tokenizer: PreTrainedTokenizer,
        data_args: LlavaDataArguments,
        seed: int = 42,
    ):
        """
        初始化 Mask 数据集
        
        Args:
            train_data_path: mask_output.json 文件路径
            tokenizer: 分词器
            data_args: 数据参数，需要包含：
                - image_folder: 原图目录
                - masked_image_folder: 遮挡图目录
            seed: 随机种子
        """
        super().__init__()

        # 读取数据
        desc_data_dict: dict = read_json(train_data_path)
        random.seed(seed)
        
        # 处理数据
        data: list[dict] = self.mask_data_process(desc_data_dict)
        random.shuffle(data)

        self.tokenizer = tokenizer
        self.data = data
        self.data_args = data_args

    def mask_data_process(self, desc_data: dict[str, list[dict]]) -> list[dict]:
        """
        处理 mask_output.json 格式的数据
        
        输入格式:
            {
                "原图路径": [
                    {"question": "...", "y_win": "...", "y_lose": null, "masked_image": "遮挡图文件名"},
                    ...
                ],
                ...
            }
        
        输出格式:
            [
                {
                    "id": 原图路径,
                    "image": 原图文件名,           # 原始图像
                    "masked_image": 遮挡图文件名,  # 遮挡图像
                    "chosen_conversations": [...],
                    "reject_conversations": None,
                },
                ...
            ]
        
        Args:
            desc_data: 从 mask_output.json 读取的数据字典
        
        Returns:
            处理后的数据列表
        """
        data_list: list[dict] = []
        
        for image_path, pairs in desc_data.items():
            for pair in pairs:
                # 获取数据字段
                y_win = pair.get("y_win", "")
                question = "<image>\n" + pair.get("question", "")
                masked_image = pair.get("masked_image", "")
                
                # 跳过无效数据
                if not y_win or not masked_image:
                    continue
                
                # 从 image_path（原图路径）提取文件名
                # image_path 可能是完整路径或仅文件名
                original_image = os.path.basename(image_path)
                
                # 构建训练数据
                data_list.append(
                    {
                        "id": image_path,
                        "image": original_image,      # 原图文件名
                        "masked_image": masked_image,  # 遮挡图文件名
                        "chosen_conversations": [
                            {"from": "human", "value": question},
                            {"from": "gpt", "value": y_win},
                        ],
                        "reject_conversations": None,
                        "has_context": False,
                        "context": None,
                    }
                )
        
        return data_list

    def __len__(self):
        return len(self.data)

    @property
    def lengths(self):
        """计算每个样本的长度（用于动态 batching）"""
        length_list = []
        for sample in self.data:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["chosen_conversations"]) + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        """计算模态长度（用于分组）"""
        length_list = []
        for sample in self.data:
            cur_len = sum(len(conv["value"].split()) for conv in sample["chosen_conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def _expand2square(self, pil_img: Image.Image, background_color: tuple) -> Image.Image:
        """
        将图像扩展为正方形
        
        Args:
            pil_img: PIL 图像
            background_color: 填充背景色（RGB 元组）
        
        Returns:
            正方形 PIL 图像
        """
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

    def _process_image(self, image: Image.Image, processor: CLIPImageProcessor) -> torch.Tensor:
        """
        处理单张图像
        
        Args:
            image: PIL 图像
            processor: CLIP 图像处理器
        
        Returns:
            处理后的图像张量
        """
        if self.data_args.image_aspect_ratio == "pad":
            image = self._expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        return processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

    def handle_data(
        self,
        data: dict,
        image: Image.Image,
        masked_image: Image.Image,
        processor: CLIPImageProcessor,
    ) -> dict:
        """
        处理单个数据样本
        
        Args:
            data: 数据字典
            image: 原始 PIL 图像
            masked_image: 遮挡后的 PIL 图像
            processor: CLIP 图像处理器
        
        Returns:
            处理后的数据字典，包含：
            - chosen_input_ids: token ids
            - chosen_labels: 标签
            - images: 原图张量
            - masked_images: 遮挡图张量
            - reject_input_ids: 与 chosen 相同（用于 DPO 框架兼容）
            - reject_labels: 与 chosen 相同
        """
        # 处理原图和遮挡图
        image_tensor = self._process_image(image, processor)
        masked_image_tensor = self._process_image(masked_image, processor)
        
        # 处理 chosen 对话
        chosen_sources: list[list[dict]] = preprocess_multimodal(
            copy.deepcopy([data["chosen_conversations"]]), self.data_args
        )
        has_image = "image" in data
        chosen_data_dict: dict[str, torch.Tensor] = preprocess(
            chosen_sources, self.tokenizer, has_image=has_image
        )
        
        data_dict = dict(
            chosen_input_ids=chosen_data_dict["input_ids"][0],
            chosen_labels=chosen_data_dict["labels"][0],
            images=image_tensor,              # 原图
            masked_images=masked_image_tensor, # 遮挡图
        )
        
        # reject 使用与 chosen 相同的 input_ids 和 labels
        # 在 Mask DPO 中，我们不使用传统的 rejected 样本
        # 而是通过对比原图和遮挡图的输出来计算 loss
        data_dict.update(
            {
                "reject_input_ids": data_dict["chosen_input_ids"].clone(),
                "reject_labels": data_dict["chosen_labels"].clone(),
            }
        )
        
        return data_dict

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个数据样本
        
        同时加载原图和遮挡图
        
        Args:
            idx: 样本索引
        
        Returns:
            处理后的数据字典
        """
        data: dict = self.data[idx]
        processor: CLIPImageProcessor = self.data_args.image_processor
        
        # 获取图像路径
        image_file: str = data["image"]           # 原图文件名
        masked_image_file: str = data["masked_image"]  # 遮挡图文件名
        
        # 原图目录和遮挡图目录
        image_folder: str = self.data_args.image_folder
        masked_folder: str = self.data_args.masked_image_folder or self.data_args.image_folder
        
        # 加载原图
        if os.path.isabs(image_file):
            image_path = image_file
        else:
            image_path = os.path.join(image_folder, image_file)
        image: Image.Image = Image.open(image_path).convert("RGB")
        
        # 加载遮挡图
        if os.path.isabs(masked_image_file):
            masked_image_path = masked_image_file
        else:
            masked_image_path = os.path.join(masked_folder, masked_image_file)
        masked_image: Image.Image = Image.open(masked_image_path).convert("RGB")
        
        data_dict = self.handle_data(data, image, masked_image, processor)
        return data_dict


@dataclass
class MaskDataCollatorForSupervisedDataset:
    """
    Mask DPO 训练数据整理器
    
    与原版 DataCollatorForSupervisedDataset 的区别：
    - 同时处理 images（原图）和 masked_images（遮挡图）
    
    负责将多个样本整理成一个 batch，包括：
    1. 填充（Padding）：将不同长度的序列填充到相同长度
    2. 截断（Truncation）：将超长序列截断到最大长度
    3. 生成 attention mask：标记哪些位置是真实 token
    4. 堆叠图像：将原图和遮挡图分别堆叠成 batch
    
    Attributes:
        tokenizer: 分词器，用于获取 pad_token_id 和 max_length
    """

    tokenizer: PreTrainedTokenizer

    def __post_init__(self):
        """dataclass 初始化后的设置"""
        self.pad_token_id: int = self.tokenizer.pad_token_id
        self.max_length: int = self.tokenizer.model_max_length

    def pad_and_truncate(self, sequences: list[torch.Tensor], padding_value: int) -> torch.Tensor:
        """
        对序列进行填充和截断
        
        Args:
            sequences: 序列列表，每个元素是一个 1D tensor
            padding_value: 填充值
        
        Returns:
            填充并截断后的 2D tensor，shape: (batch_size, min(max_seq_len, max_length))
        """
        padded = rnn_utils.pad_sequence(sequences, batch_first=True, padding_value=padding_value)
        return padded[:, : self.max_length]

    def __call__(self, instances: list[dict]) -> Dict[str, torch.Tensor]:
        """
        将多个样本整理成一个 batch
        
        Args:
            instances: 样本列表，每个样本是 __getitem__ 返回的字典
        
        Returns:
            batch 字典，包含：
            - chosen_input_ids: (batch_size, seq_len)
            - chosen_labels: (batch_size, seq_len)
            - reject_input_ids: (batch_size, seq_len)
            - reject_labels: (batch_size, seq_len)
            - chosen_attention_mask: (batch_size, seq_len)
            - reject_attention_mask: (batch_size, seq_len)
            - images: (batch_size, 3, H, W) 原图
            - masked_images: (batch_size, 3, H, W) 遮挡图
        """
        # ==================== 处理文本序列 ====================
        chosen_input_ids, chosen_labels, reject_input_ids, reject_labels = tuple(
            [i[key] for i in instances]
            for key in ("chosen_input_ids", "chosen_labels", "reject_input_ids", "reject_labels")
        )
        
        # 填充和截断
        chosen_input_ids = self.pad_and_truncate(chosen_input_ids, padding_value=self.pad_token_id)
        chosen_labels = self.pad_and_truncate(chosen_labels, padding_value=IGNORE_INDEX)
        reject_input_ids = self.pad_and_truncate(reject_input_ids, padding_value=self.pad_token_id)
        reject_labels = self.pad_and_truncate(reject_labels, padding_value=IGNORE_INDEX)

        batch = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            reject_input_ids=reject_input_ids,
            reject_labels=reject_labels,
            # 注意力掩码：非 pad 位置为 True，pad 位置为 False
            chosen_attention_mask=chosen_input_ids.ne(self.pad_token_id),
            reject_attention_mask=reject_input_ids.ne(self.pad_token_id),
        )

        # ==================== 处理原图 ====================
        images = [instance["images"] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch["images"] = torch.stack(images)
        else:
            batch["images"] = images

        # ==================== 处理遮挡图 ====================
        masked_images = [instance["masked_images"] for instance in instances]
        if all(x is not None and x.shape == masked_images[0].shape for x in masked_images):
            batch["masked_images"] = torch.stack(masked_images)
        else:
            batch["masked_images"] = masked_images

        return batch
