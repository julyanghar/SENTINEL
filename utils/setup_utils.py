import os
from argparse import ArgumentParser, Namespace
from logging import Logger

if __name__ == "__main__":
    print("Please run main.py")
    exit(0)


def parse_arg() -> Namespace:
    parser = ArgumentParser()
    # How many GPUs to use, 1 or 2
    parser.add_argument("--gpu_num", type=int, default=2)
    # Sample batch size
    parser.add_argument("--batch_size", type=int, default=5)
    # Dataset path
    parser.add_argument("--dataset_path", type=str, default="dataset/image_data.jsonl")
    # Model name to sample candidates
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen2_VL_2B",
        choices=[
            "LLaVA_v1_5_7b",
            "LLaVA_v1_5_13b",
            "LLaVA_v1_6_vicuna_7b",
            "LLaVA_v1_6_vicuna_13b",
            "Qwen2_VL_2B",
            "Qwen2_VL_7B",
            "Qwen2_5_VL_7B",
        ],
    )
    
    # <--- Uninportant --->
    # log level
    parser.add_argument("--log_level", type=str, default="INFO", choices=["INFO", "WARNING"])
    # num of datapoints to process (-1 means all)
    parser.add_argument("--num_of_data", type=int, default=-1)
    # random seed for reproducibility
    parser.add_argument("--seed", type=int, default=42)
    # log directory
    parser.add_argument("--log_dir", type=str, default="./log/")

    args: Namespace = parser.parse_args()
    args.model_size = args.model.split("-")[-1] if "-" in args.model else args.model.split("_")[-1]
    args.model_version = "1.5" if "v1_5" in args.model else "1.6" if "v1_6" in args.model else "2.0"
    return args


def get_save_path(logger: Logger, args: Namespace) -> str:
    save_path: str = os.path.join("./results", args.model)
    if save_path.endswith("/"):
        save_path = save_path[:-1]
    logger.info(f"The save_path is: {save_path}")
    return save_path
