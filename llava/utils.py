import math
import random
from io import BytesIO

import numpy as np
import requests
import shortuuid
import torch
import torch.backends.cudnn as cudnn
from PIL import Image


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def setup_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def load_image(image_file: str | Image.Image) -> Image.Image:
    if isinstance(image_file, str):
        if image_file.startswith("http"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
    else:
        image = image_file.convert("RGB")
    return image


def load_images(image_files: list) -> list[Image.Image]:
    return [load_image(image_file) for image_file in image_files]


def split_list(lst: list, n: int) -> list:
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst: list, n: int, k: int) -> list:
    chunks = split_list(lst, n)
    return chunks[k]


def get_id_from_sample(sample: dict) -> str:
    if "question_id" in sample:
        return sample["question_id"]
    elif "id" in sample:
        return sample["id"]
    elif "image_id" in sample:
        return sample["image_id"]
    else:
        return shortuuid.uuid()


def get_question_from_sample(sample: dict) -> str:
    if "text" in sample:
        return sample["text"]
    elif "query" in sample:
        return sample["query"]
    elif "question" in sample:
        return sample["question"]


def get_image_path_from_sample(sample: dict) -> str | None:
    image_path = None
    if "image" in sample:
        image_path = sample["image"]
    elif "image_path" in sample:
        image_path = sample["image_path"]
    elif "image_file" in sample:
        image_path = sample["image_file"]
    elif "image_id" in sample:
        image_path = sample["image_id"]
    if isinstance(image_path, str) and "." not in image_path:
        image_path += ".jpg"
    return image_path
