import logging
import os
import sys
from argparse import Namespace
from logging import Logger

import torch

from utils.setup_utils import get_save_path, parse_arg


# 定义全局变量
class GVars:
    args: Namespace | None = None
    """命令行参数"""
    save_path: str | None = None
    """保存路径"""
    model_dir: str | None = None
    """模型目录"""
    hf_home: str | None = None
    """Hugging Face Home"""
    gpu_count: int | None = None
    device: str | None = None
    main_device: str | None = None
    alter_device: str | None = None
    openai_key: str | None = None
    logger: Logger = logging.getLogger()

    @classmethod
    def init(cls, save: bool = True) -> None:
        cls.init_args()
        cls.init_logger()
        cls.init_model_dir()
        if save:
            cls.init_save_file_path()
        cls.init_device()
        cls.init_openai_key()
        cls.logger.info("Global variables (Gvars) have been initialized")

    @classmethod
    def init_args(cls, alter: dict | None = None) -> None:
        cls.args = parse_arg()
        if alter is not None:
            for key in alter:
                if alter[key]:
                    setattr(cls.args, key, alter[key])

    @classmethod
    def init_model_dir(cls) -> None:
        cls.hf_home = os.getenv("HF_HOME")
        cls.model_dir = os.getenv("MODEL_PATH")

    @classmethod
    def init_openai_key(cls) -> None:
        cls.openai_key = os.getenv("OPENAI_KEY")

    @classmethod
    def init_save_file_path(cls) -> None:
        cls.save_path = get_save_path(cls.logger, cls.args) + ".jsonl"

    @classmethod
    def init_device(cls) -> None:
        if not torch.cuda.is_available():
            cls.main_device, cls.alter_device, cls.gpu_count = "cpu", "cpu", 0
        elif torch.cuda.device_count() == 1:
            cls.main_device, cls.alter_device, cls.gpu_count = "cuda:0", "cuda:0", 1
        else:
            cls.main_device, cls.alter_device, cls.gpu_count = "cuda:0", "cuda:1", 2
        cls.device = cls.main_device

    @classmethod
    def init_logger(cls) -> None:
        from logging import INFO, WARNING

        from transformers.utils import logging as transformers_logging

        _nameToLevel = {"WARNING": WARNING, "INFO": INFO}

        cls.logger.setLevel(_nameToLevel["INFO"])
        args = cls.args
        log_filename = f"{args.model}-{args.num_of_data}.log"
        log_path = os.path.join(args.log_dir, log_filename)

        os.makedirs(args.log_dir, exist_ok=True)

        logging.basicConfig(
            format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_path, mode="a"),
            ],
        )
        transformers_logging.set_verbosity(_nameToLevel[args.log_level])
        transformers_logging.enable_default_handler()
        transformers_logging.add_handler(logging.FileHandler(log_path, mode="a"))
        transformers_logging.enable_explicit_format()


if __name__ == "__main__":
    print("Please run main.py")
    exit(0)
