"""
DPO 训练数据集模块
==================

本模块实现了用于 LLaVA 模型 DPO（Direct Preference Optimization）训练的数据集类。

主要组件：
1. LazySupervisedDataset: 懒加载的监督训练数据集
2. DataCollatorForSupervisedDataset: 数据整理器，负责 batch 的填充和截断

数据格式说明：
--------------
输入数据格式（output.json）:
{
    "image_id": [
        {
            "question": "问题文本",
            "y_win": "正确/偏好的回答",
            "y_lose": "错误/不偏好的回答",
            "context": "可选的上下文前缀",
            "type": "y+" 或其他类型标记
        },
        ...
    ],
    ...
}

处理后的内部格式:
{
    "id": 图像ID (int),
    "image": 图像路径 (str),
    "chosen_conversations": [  # 偏好的对话
        {"from": "human", "value": "<image>\n问题"},
        {"from": "gpt", "value": "正确回答"}
    ],
    "reject_conversations": [  # 不偏好的对话
        {"from": "human", "value": "<image>\n问题"},
        {"from": "gpt", "value": "错误回答"}
    ],
    "has_context": 是否有上下文 (bool),
    "context": 上下文文本 (str | None)
}

训练原理：
---------
DPO 通过对比 chosen（偏好）和 rejected（不偏好）响应来训练模型：
- chosen_conversations: 模型应该学习生成的正确响应
- reject_conversations: 模型应该避免生成的错误响应
- context: 如果存在，会被添加到响应前面，但在计算 loss 时被忽略（IGNORE_INDEX）
"""

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
    懒加载的 DPO 监督训练数据集
    
    该数据集用于加载和处理 DPO 训练所需的图文对数据。
    "懒加载"指的是图像只在 __getitem__ 时才被加载，而非初始化时全部加载到内存。
    
    Attributes:
        tokenizer (PreTrainedTokenizer): 分词器，用于将文本转换为 token ids
        data (list[dict]): 处理后的训练数据列表
        data_args (LlavaDataArguments): 数据相关的参数配置
    
    数据流程：
        1. 加载 Visual Genome 图像元数据，建立 image_id -> 图像路径 的映射
        2. 加载训练数据（JSON 或 JSONL 格式）
        3. 将原始数据转换为统一的内部格式
        4. 随机打乱数据顺序
    """

    def __init__(
        self,
        vg_path: str,
        train_data_path: str,
        tokenizer: PreTrainedTokenizer,
        data_args: LlavaDataArguments,
        seed: int = 42,
    ):
        """
        初始化数据集
        
        Args:
            vg_path: Visual Genome 数据集根目录路径
                     该目录下应包含 image_data.json 文件
            train_data_path: 训练数据文件路径（.json 或 .jsonl 格式）
            tokenizer: HuggingFace 分词器
            data_args: 数据参数配置对象
            seed: 随机种子，用于数据打乱
        """
        super().__init__()

        # Step 1: 加载 Visual Genome 图像元数据
        # image_data.json 包含图像 ID 和 URL 的映射
        vg_image_data: list[dict] = json.load(open(os.path.join(vg_path, "image_data.json")))
        
        # 构建 image_id -> 本地图像路径 的映射字典
        # URL 格式: https://cs.stanford.edu/people/rak248/VG_100K/2324811.jpg
        # 本地路径格式: vg_path/VG_100K/2324811.jpg
        self.id2path: dict[int, str] = {
            d["image_id"]: os.path.join(vg_path, d["url"].split("/")[-2], d["url"].split("/")[-1])
            for d in vg_image_data
        }

        # Step 2: 加载并处理训练数据
        desc_data_dict: dict[str, dict] = read_json(train_data_path)
        random.seed(seed)
        
        # 根据文件格式选择不同的处理函数
        if train_data_path.endswith(".jsonl"):
            # JSONL 格式：每行一个 JSON 对象
            data: list[dict] = self.desc_w_context_jsonl_process(desc_data_dict)
        else:
            # JSON 格式：以 image_id 为 key 的字典
            data: list[dict] = self.desc_w_context_process(desc_data_dict)
        
        # Step 3: 随机打乱数据
        random.shuffle(data)

        self.tokenizer = tokenizer
        self.data = data[:100]  # 注意：这里限制了数据量为 100 条（可能是调试用）
        self.data_args = data_args

        # 清理不再需要的中间数据，释放内存
        del desc_data_dict, vg_image_data, self.id2path

    def desc_w_context_process(self, desc_data: dict[str, list[dict[str, str]]]) -> list[dict]:
        """
        处理 JSON 格式的训练数据（带上下文支持）
        
        输入格式:
            {
                "image_id_1": [
                    {"question": "...", "y_win": "...", "y_lose": "...", "context": "..."},
                    ...
                ],
                "image_id_2": [...],
                ...
            }
        
        Args:
            desc_data: 以 image_id 为 key 的数据字典
        
        Returns:
            处理后的数据列表，每个元素包含:
            - id: 图像 ID
            - image: 图像路径
            - chosen_conversations: 偏好对话列表
            - reject_conversations: 不偏好对话列表
            - has_context: 是否有上下文
            - context: 上下文文本
        """
        desc_data_dict: list[dict[str]] = []
        
        for image_id in desc_data.keys():
            for pair in desc_data[image_id]:
                # 兼容不同的字段命名（y_win/win, y_lose/lose）
                y_win = pair["y_win"] if "y_win" in pair else pair["win"]
                y_lose = pair["y_lose"] if "y_lose" in pair else pair["lose"]
                
                # 添加 <image> 标记，告诉模型这里需要插入图像 embedding
                question = "<image>\n" + pair["question"]

                # 处理上下文：如果存在 context，将其添加到回答前面
                # context 通常是 y_win 和 y_lose 共享的前缀部分
                has_context = "context" in pair and pair["context"] is not None
                chosen = pair["context"] + " " + y_win if has_context else y_win
                rejected = pair["context"] + " " + y_lose if has_context else y_lose
                
                desc_data_dict.append(
                    {
                        "id": int(image_id),
                        "image": self.id2path[int(image_id)],  # 通过 id2path 映射获取图像路径
                        "chosen_conversations": [
                            {"from": "human", "value": question},  # 用户提问
                            {"from": "gpt", "value": chosen},      # 偏好的模型回答
                        ],
                        "reject_conversations": [
                            {"from": "human", "value": question},  # 用户提问（相同）
                            {"from": "gpt", "value": rejected},    # 不偏好的模型回答
                        ],
                        "has_context": has_context,
                        "context": pair["context"],
                    }
                )

        return desc_data_dict

    def desc_w_context_jsonl_process(self, desc_data: list[dict[str]]) -> list[dict]:
        """
        处理 JSONL 格式的训练数据（带上下文支持）
        
        输入格式（每行一个 JSON）:
            {"image_id": "...", "question": "...", "y_win": "...", "y_lose": "...", "context": "..."}
            {"image_id": "...", "question": "...", "y_win": "...", "y_lose": "...", "context": "..."}
            ...
        
        Args:
            desc_data: 数据列表（read_json 已将 JSONL 解析为列表）
        
        Returns:
            处理后的数据列表，格式与 desc_w_context_process 返回值相同
        """
        desc_data_dict: list[dict[str]] = []
        
        for pair in desc_data:
            # 兼容不同的字段命名
            y_win = pair["y_win"] if "y_win" in pair else pair["win"]
            y_lose = pair["y_lose"] if "y_lose" in pair else pair["lose"]
            question = "<image>\n" + pair["question"]

            # 处理上下文
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
        """返回数据集大小"""
        return len(self.data)

    @property
    def lengths(self):
        """
        计算每个样本的 token 长度
        
        用于动态 batching，可以将长度相近的样本放在同一个 batch 中，
        减少 padding 带来的计算浪费。
        
        Returns:
            长度列表，每个元素是对应样本的估计 token 数
        """
        length_list = []
        for sample in self.data:
            # 图像会被编码为固定数量的 token（这里估计为 128）
            img_tokens = 128 if "image" in sample else 0
            # 文本长度用单词数估计
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        """
        计算模态长度（用于按模态分组）
        
        正值表示有图像，负值表示纯文本。
        这允许训练时将有图像和无图像的样本分开处理。
        
        Returns:
            长度列表，正值=图文样本，负值=纯文本样本
        """
        length_list = []
        for sample in self.data:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            # 有图像的样本长度为正，无图像的为负
            cur_len = cur_len if "images" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def handle_data(self, data: dict, image: Image.Image, processor: CLIPImageProcessor):
        """
        处理单个数据样本
        
        将原始数据转换为模型可以接受的格式，包括：
        1. 图像预处理（填充为正方形 + CLIP 处理）
        2. 文本分词（转换为 token ids）
        3. 上下文处理（将 context 部分的 loss 设为 IGNORE）
        
        Args:
            data: 单个样本的数据字典
            image: PIL 图像对象
            processor: CLIP 图像处理器
        
        Returns:
            dict: 包含以下字段：
                - chosen_input_ids: 偏好响应的 token ids
                - chosen_labels: 偏好响应的标签（用于计算 loss）
                - reject_input_ids: 不偏好响应的 token ids
                - reject_labels: 不偏好响应的标签
                - images: 处理后的图像张量
        """
        # ==================== 图像预处理 ====================
        if self.data_args.image_aspect_ratio == "pad":
            # 将图像填充为正方形，保持原始宽高比

            def expand2square(pil_img, background_color):
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
                    # 宽图：上下填充
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    # 高图：左右填充
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            # 使用 CLIP 图像均值作为填充颜色
            image: torch.Tensor = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image: torch.Tensor = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            # 直接使用 CLIP 处理（可能会裁剪或变形）
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        
        # ==================== 文本预处理 ====================
        # 处理 chosen（偏好）对话
        chosen_sources: list[list[dict]] = preprocess_multimodal(
            copy.deepcopy([data["chosen_conversations"]]), self.data_args
        )

        has_image = "image" in data
        # preprocess 将对话转换为 input_ids 和 labels
        # labels 中，用户输入部分被设为 IGNORE_INDEX（不计算 loss）
        chosen_data_dict: dict[str, torch.Tensor] = preprocess(chosen_sources, self.tokenizer, has_image=has_image)
        
        data_dict = dict(
            chosen_input_ids=chosen_data_dict["input_ids"][0],
            chosen_labels=chosen_data_dict["labels"][0],
            images=image,
        )

        # 处理 reject（不偏好）对话
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

        # ==================== 上下文处理 ====================
        # 如果存在 context，需要在 labels 中将其标记为 IGNORE_INDEX
        # 这样 context 部分不会被计入 loss，只有 y_win/y_lose 部分才计算 loss
        if "has_context" in data and data["has_context"] and data["context"]:
            context = data["context"]
            # 将 context 文本编码为 token ids（不加特殊 token）
            context_ids = self.tokenizer.encode(context, add_special_tokens=False, return_tensors="pt")[0]

            context_len = len(context_ids)
            
            # 在 chosen 序列中找到 context 的位置，将其 labels 设为 IGNORE_INDEX
            chosen_input_ids: torch.Tensor = data_dict["chosen_input_ids"]
            for i in range(len(chosen_input_ids) - context_len + 1):
                if torch.equal(chosen_input_ids[i : i + context_len], context_ids):
                    data_dict["chosen_labels"][i : i + context_len] = IGNORE_INDEX
                    break

            # 去掉最后一个 <EOS> token 的预测
            # 这是因为 context 处理后序列结构可能变化
            data_dict["chosen_input_ids"] = data_dict["chosen_input_ids"][:-1]
            data_dict["chosen_labels"] = data_dict["chosen_labels"][:-1]

            # 对 reject 序列做同样的处理
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
        """
        获取单个数据样本
        
        Args:
            idx: 样本索引
        
        Returns:
            处理后的数据字典，包含 input_ids, labels, images 等
        """
        data: dict = self.data[idx]

        # 加载图像（懒加载：只在需要时才读取图像文件）
        image_file: str = self.data[idx]["image"]
        image_folder: str = self.data_args.image_folder
        processor: CLIPImageProcessor = self.data_args.image_processor
        image: Image.Image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")

        # 处理数据并返回
        data_dict = self.handle_data(data, image, processor)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    监督训练数据整理器
    
    负责将多个样本整理成一个 batch，包括：
    1. 填充（Padding）：将不同长度的序列填充到相同长度
    2. 截断（Truncation）：将超长序列截断到最大长度
    3. 生成 attention mask：标记哪些位置是真实 token，哪些是填充
    
    Attributes:
        tokenizer: 分词器，用于获取 pad_token_id 和 max_length
        pad_token_id: 填充 token 的 ID
        max_length: 最大序列长度
    """

    tokenizer: PreTrainedTokenizer

    def __post_init__(self):
        """dataclass 初始化后的设置"""
        self.pad_token_id: int = self.tokenizer.pad_token_id
        self.max_length: int = self.tokenizer.model_max_length

    def pad_and_truncate(self, sequences, padding_value):
        """
        对序列进行填充和截断
        
        Args:
            sequences: 序列列表，每个元素是一个 1D tensor
            padding_value: 填充值（input_ids 用 pad_token_id，labels 用 IGNORE_INDEX）
        
        Returns:
            填充并截断后的 2D tensor，shape: (batch_size, min(max_seq_len, max_length))
        """
        # pad_sequence 会将所有序列填充到最长序列的长度
        padded = rnn_utils.pad_sequence(sequences, batch_first=True, padding_value=padding_value)
        # 截断到 max_length
        return padded[:, : self.max_length]

    def __call__(self, instances: list[dict]) -> Dict[str, torch.Tensor]:
        """
        将多个样本整理成一个 batch
        
        Args:
            instances: 样本列表，每个样本是 __getitem__ 返回的字典
        
        Returns:
            batch 字典，包含：
            - chosen_input_ids: (batch_size, seq_len) 偏好响应的 token ids
            - chosen_labels: (batch_size, seq_len) 偏好响应的标签
            - reject_input_ids: (batch_size, seq_len) 不偏好响应的 token ids
            - reject_labels: (batch_size, seq_len) 不偏好响应的标签
            - chosen_attention_mask: (batch_size, seq_len) 偏好响应的注意力掩码
            - reject_attention_mask: (batch_size, seq_len) 不偏好响应的注意力掩码
            - images: (batch_size, 3, H, W) 图像张量
            - [可选] sft_* 字段：用于 SFT 训练的数据
        """
        # 提取各个字段
        chosen_input_ids, chosen_labels, reject_input_ids, reject_labels = tuple(
            [i[key] for i in instances]
            for key in ("chosen_input_ids", "chosen_labels", "reject_input_ids", "reject_labels")
        )
        
        # 填充和截断
        # input_ids 使用 pad_token_id 填充
        chosen_input_ids = self.pad_and_truncate(chosen_input_ids, padding_value=self.pad_token_id)
        reject_input_ids = self.pad_and_truncate(reject_input_ids, padding_value=self.pad_token_id)
        # labels 使用 IGNORE_INDEX 填充（填充位置不计算 loss）
        chosen_labels = self.pad_and_truncate(chosen_labels, padding_value=IGNORE_INDEX)
        reject_labels = self.pad_and_truncate(reject_labels, padding_value=IGNORE_INDEX)

        batch = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            reject_input_ids=reject_input_ids,
            reject_labels=reject_labels,
            # 注意力掩码：非 pad 位置为 True，pad 位置为 False
            # .ne() 是 "not equal" 的简写
            chosen_attention_mask=chosen_input_ids.ne(self.pad_token_id),
            reject_attention_mask=reject_input_ids.ne(self.pad_token_id),
        )
        
        # ==================== 处理 SFT 数据（可选） ====================
        # 如果样本中包含 SFT 训练数据
        if "sft_images" in instances[0]:
            sft_input_ids, sft_labels = tuple([i[key] for i in instances] for key in ("sft_input_ids", "sft_labels"))
            sft_input_ids = self.pad_and_truncate(sft_input_ids, padding_value=self.pad_token_id)
            sft_labels = self.pad_and_truncate(sft_labels, padding_value=IGNORE_INDEX)
            batch.update(
                dict(
                    sft_input_ids=sft_input_ids,
                    sft_labels=sft_labels,
                    sft_attention_mask=sft_input_ids.ne(self.pad_token_id),
                )
            )

        # ==================== 处理图像 ====================
        images = [instance["images"] for instance in instances]
        # 如果所有图像 shape 相同，可以 stack 成一个 tensor
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch["images"] = torch.stack(images)
        else:
            # 否则保持为列表（模型需要特殊处理）
            batch["images"] = images

        # 处理 SFT 图像（如果存在）
        if "sft_images" in instances[0]:
            sft_images = [instance["sft_images"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in sft_images):
                batch["sft_images"] = torch.stack(sft_images)
            else:
                batch["sft_images"] = sft_images

        return batch
