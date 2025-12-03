"""
SENTINEL 主运行模块
===================

本模块负责协调整个数据生成流程：
1. 加载和预处理数据集
2. 过滤已处理的数据点
3. 调用核心数据生成函数

模块依赖:
    - model.auxiliary.dataset: 数据集加载
    - model.auxiliary.global_vars: 全局变量
    - run.generate_dataset: 核心生成逻辑
"""

if __name__ == "__main__":
    print("Please run main.py")
    exit(0)


def run() -> None:
    """
    主运行函数：协调数据生成流程
    
    执行流程:
        1. 从全局变量获取配置参数
        2. 加载数据集（image_data.jsonl）
        3. 过滤已处理的数据（支持断点续传）
        4. 调用核心生成函数处理剩余数据
    
    数据流:
        image_data.jsonl 
            → DataSet 对象 
            → 过滤已处理数据 
            → run_gen_dataset()
            → 输出到 ./results/<model_name>.jsonl
    
    Notes:
        - 支持断点续传：如果程序中断，重新运行会自动跳过已处理的数据
        - batch_size 影响 GPU 显存使用和处理速度
    """
    from model.auxiliary.dataset import DataSet
    from model.auxiliary.global_vars import GVars
    from run.generate_dataset import run_gen_dataset

    # 从全局变量获取配置
    args, save_path, logger = GVars.args, GVars.save_path, GVars.logger
    batch_size = args.batch_size

    # 记录当前配置
    logger.info(f"Current batch size: {batch_size}")
    logger.info(f"Start loading dataset with dataset path: {args.dataset_path}")
    
    # 加载数据集
    # DataSet 会自动解析 jsonl 文件并创建 DataPoint 对象列表
    dataset: DataSet = DataSet(args=args, logger=logger)
    
    # 过滤已处理的数据（实现断点续传）
    # 通过比对 save_path 中已存在的 image_id 来过滤
    dataset.filter(save_path)
    logger.info(f"Finish loading dataset, dataset size: {len(dataset.data)}")

    # 检查是否还有数据需要处理
    if not dataset.data:
        logger.info("All data has been processed, exit run function")
        return
    
    # 调用核心数据生成函数
    # 这是整个项目最核心的处理逻辑
    run_gen_dataset(dataset.data, batch_size)
