"""
SENTINEL Mask 模式运行模块
===========================

本模块负责协调基于图像遮挡的偏好数据生成流程：
1. 加载参考数据集
2. 过滤已处理的数据点
3. 调用 mask 数据生成函数

模块依赖:
    - model.auxiliary.global_vars: 全局变量
    - run.generate_mask_dataset: Mask 核心生成逻辑
"""

if __name__ == "__main__":
    print("Please run main.py with --mode mask")
    exit(0)


def run_mask() -> None:
    """
    Mask 模式主运行函数：协调基于图像遮挡的偏好数据生成流程
    
    执行流程:
        1. 从全局变量获取配置参数
        2. 调用 mask 数据生成函数
    
    数据流:
        reference_data.jsonl 
            → 物体提取 
            → YOLO + DINO 检测
            → 图像遮挡
            → VLM 推理
            → 幻觉检测
            → 输出到 ./results/mask_preference_pairs.jsonl
    
    使用方法:
        python main.py --mode mask --dataset_path dataset/reference_data.jsonl
    
    Notes:
        - 支持断点续传：如果程序中断，重新运行会自动跳过已处理的数据
        - 遮挡图像会保存到 --masked_images_dir 指定的目录
    """
    from model.auxiliary.global_vars import GVars
    from run.generate_mask_dataset import run_mask_dataset_generation

    # 从全局变量获取配置
    args, save_path, logger = GVars.args, GVars.save_path, GVars.logger
    batch_size = args.batch_size

    # 输出目录
    output_dir = save_path.replace(".jsonl", "_mask")
    
    # 遮挡图像保存目录：./results/{model}_masked_images
    masked_images_dir = f"./results/{args.model}_masked_images"
    
    # 记录当前配置
    logger.info(f"Running in MASK mode")
    logger.info(f"Current batch size: {batch_size}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Masked images will be saved to: {masked_images_dir}")
    
    # 调用 mask 数据生成函数
    run_mask_dataset_generation(
        dataset_path=args.dataset_path,
        output_dir=output_dir,
        masked_images_dir=masked_images_dir,
        batch_size=batch_size,
        logger=logger,
    )
    
    logger.info("Mask dataset generation completed!")

