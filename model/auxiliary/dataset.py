import os
import random
from argparse import Namespace
from dataclasses import dataclass, field
from logging import Logger

from ..utils.utils import read_json


@dataclass
class DataPoint:
    image_id: str
    image_path: str
    question: str
    attributes: dict[str] = field(default_factory=dict)

    def __getitem__(self, key: str) -> str:
        if key == "image_id":
            return self.image_id
        elif key == "image_path":
            return self.image_path
        elif key == "question":
            return self.question
        elif key in self.attributes:
            return self.attributes[key]
        else:
            raise KeyError(f"Key {key} not found in DataPoint")

    def __repr__(self) -> str:
        return f"DataPoint(image_id={self.image_id}, image_path={self.image_path}, question={self.question})"


@dataclass
class DataSet:
    args: Namespace
    logger: Logger | None = None
    data: list[DataPoint] = field(init=False)

    def __post_init__(self):
        if self.logger is not None and self.args is not None:
            self.logger.info(f"Loading dataset from {self.args.dataset_path}")
        self.data = self._load_dataset(self.args)

    def _load_dataset(self, args: Namespace) -> list[DataPoint]:
        dataset_path: str = args.dataset_path
        assert os.path.exists(dataset_path), f"Dataset file not found at {dataset_path}"

        dataset: list[dict] = read_json(dataset_path)

        num_of_data: int = args.num_of_data
        if 0 <= num_of_data < len(dataset):
            random.seed(args.seed)
            dataset = random.sample(dataset, num_of_data)

        return [self._create_datapoint(item) for item in dataset]

    @staticmethod
    def _create_datapoint(item: dict) -> DataPoint:
        """
        Create a DataPoint object from a dictionary.
        """
        image_id: str = item["image_id"] if "image_id" in item else item["image"]
        image_path: str = item["image_path"] if "image_path" in item else item["image"]
        question: str = item["question"] if "question" in item else "Describe this image."
        attributes: dict[str] = {
            k: v for k, v in item.items() if k not in {"image_id", "image", "image_path", "question"}
        }
        return DataPoint(image_id, image_path, question, attributes=attributes)

    def filter(self, save_path: str) -> None:
        """
        Modify the dataset by filtering out items that already exist in the save_path directory.
        If select_idx is provided, only items with image_id in select_idx will be kept.
        """
        if not os.path.exists(save_path) or not os.path.isfile(save_path) or not self.data:
            return

        exist_data: list[dict] = read_json(save_path)
        done_image_id: list = [d["image_id"] for d in exist_data]
        self.data = [d for d in self.data if d.image_id not in done_image_id]

    @property
    def get_batches(self, batch_size: int) -> list[list[DataPoint]]:
        assert batch_size > 0, "Invalid batch size"
        if not self.data:
            return []

        n = len(self.data) // batch_size  # number of batches
        batched_data: list[list[DataPoint]] = [self.data[i * batch_size : (i + 1) * batch_size] for i in range(n)]

        last_start = n * batch_size
        if last_start < len(self.data):
            batched_data.append(self.data[last_start:])
        return batched_data

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        return f"DataSet(data_count={len(self.data)})"


if __name__ == "__main__":
    pass
