"""
SENTINEL 主入口文件
====================

本文件是 SENTINEL 项目的主入口，负责：
1. 加载环境变量配置
2. 初始化全局变量
3. 启动数据生成流程

使用方法:
    python main.py [--model MODEL_NAME] [--batch_size N] [--dataset_path PATH]

示例:
    python main.py --model Qwen2_VL_7B --batch_size 5
"""

from dotenv import load_dotenv

# 加载环境变量配置文件
# 配置文件位于 ./utils/.env，包含模型路径、HuggingFace缓存路径等配置
print("Load dot env result:", load_dotenv("./utils/.env"))


def main():
    """
    主函数：初始化环境并启动数据生成流程
    
    执行流程:
        1. 初始化全局变量 (GVars)
           - 解析命令行参数
           - 配置日志系统
           - 设置设备（GPU/CPU）
           - 初始化模型目录路径
        
        2. 运行数据生成流程 (run)
           - 加载数据集
           - 批量处理图像
           - 生成偏好训练数据对
    """
    # 导入全局变量管理类
    from model.auxiliary.global_vars import GVars

    # 初始化所有全局变量
    # 包括：命令行参数、日志器、设备配置、模型路径等
    GVars.init()

    # 导入并执行主运行逻辑
    from run.run import run

    run()


if __name__ == "__main__":
    main()
