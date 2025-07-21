import os
from dataclasses import dataclass, field
from logging import Logger

import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from ..utils.utils import ensure_lists, maybe_return_ls


@dataclass
class YoloResult:
    original_result: Results
    result: dict[str, list[dict[str]]] = field(default_factory=dict)

    def __post_init__(self):
        mapping_dict: dict[int, str] = self.original_result.names
        classes: list[int] = self.original_result.boxes.cls.cpu().to(dtype=int).tolist()
        confidences: list[float] = self.original_result.boxes.conf.cpu().tolist()
        xywhns: list[list[float]] = self.original_result.boxes.xywhn.cpu().tolist()

        for cls, conf, xywhn in zip(classes, confidences, xywhns):
            label = mapping_dict[cls]
            if label not in self.result:
                self.result[label] = []
            self.result[label].append({"conf": conf, "xywhn": xywhn})

    def _calculate_area(self, xywhn: list[float]) -> float:
        """计算给定 xywhn 的面积"""
        return xywhn[2] * xywhn[3]

    def _calculate_min_dist_to_edge(self, xywhn: list[float]) -> float:
        """计算给定 xywhn 到边缘的最小距离"""
        return min(xywhn[0], 1 - xywhn[0], xywhn[1], 1 - xywhn[1])

    @property
    def labels(self) -> list[str]:
        """获取无重复的标签列表"""
        return list(self.result.keys())

    def get_largest(self, label: str) -> dict[str, float | list[float]] | None:
        """获取指定标签的最大对象"""
        if label not in self.result:
            return None
        return max(self.result[label], key=lambda x: self._calculate_area(x["xywhn"]))

    def get_smallest(self, label: str) -> dict[str, float | list[float]] | None:
        """获取指定标签的最小对象"""
        if label not in self.result:
            return None
        return min(self.result[label], key=lambda x: self._calculate_area(x["xywhn"]))

    def get_closest_to_edge(self, label: str) -> dict[str, float | list[float]] | None:
        """获取指定标签最接近边缘的对象"""
        if label not in self.result:
            return None
        return min(self.result[label], key=lambda x: self._calculate_min_dist_to_edge(x["xywhn"]))

    def get_farthest_to_edge(self, label: str) -> dict[str, float | list[float]] | None:
        """获取指定标签最远离边缘的对象"""
        if label not in self.result:
            return None
        return max(self.result[label], key=lambda x: self._calculate_min_dist_to_edge(x["xywhn"]))

    def __repr__(self):
        return str(self.result)


class YoloModel:
    def __init__(
        self,
        model_name: str,
        model_dir: str | None = None,
        device: torch.device | str = "cpu",
        logger: Logger | None = None,
    ):
        """
        Initialize the YoloModel with the specified model size.

        Args:
            model_dir: Directory for model caching.
            device: The device to run the model on (e.g., CPU or GPU).
        """
        assert model_name in [
            "yolo11x",
            "yolov8x-worldv2",
        ], "Invalid model name."

        if logger:
            logger.info(f"Loading YOLO model {model_name} from {model_dir}")
        self.model_path: str = (
            os.path.join(os.path.dirname(os.path.expanduser(model_dir)), "yolo", f"{model_name}.pt")
            if model_dir
            else model_name
        )
        self.yolo = YOLO(self.model_path, verbose=False).to(device)

    @property
    def labels(self) -> list[str]:
        return list(self.yolo.names.values())

    def predict(
        self,
        images: Image.Image | list[Image.Image],
        force_list: bool = False,
        save_predict_result: bool = False,  # For debugging
    ):
        images = ensure_lists(images)

        results = self.yolo.predict(source=images, verbose=False, save=save_predict_result)

        return maybe_return_ls(force_list, [YoloResult(original_result=result) for result in results])
