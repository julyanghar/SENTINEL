#!/bin/bash
# Description: Training script for LLaVA with Mask DPO
# 
# 与原版 dpo_llava.sh 的区别：
# 1. 使用 mask_dpo_llava.py 作为训练入口
# 2. 不需要 VISUAL_GENOME_PATH（遮挡图像路径在 mask_output.json 中）
# 3. 图像文件夹 image_folder 指向项目根目录（因为 masked_image 使用相对路径）

# ==================== HuggingFace 缓存配置 ====================
# 设置缓存目录，避免每次重新下载 CLIP vision tower
export HF_HOME="${HF_HOME:-/home/yilin/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

# Mask DPO 需要配置的环境变量（运行前请修改）：
INPUT_MODEL="liuhaotian/llava-v1.5-7b"           # LLaVA 基础模型路径
TRAINING_DATA_PATH="/home/yilin/SENTINEL/results/mask_output.json"  # mask_output.json 路径
OUTPUT_NAME="mask-dpo-llava-v1.5-7b"                # 输出目录名称
MASKED_IMAGE_FOLDER="/home/yilin/SENTINEL/results/LLaVA_v1_5_7b_masked_images"                 # 遮挡图像根目录（masked_image 相对路径的基准）
IMAGE_FOLDER="/home/yilin/new-DPO/dataset/train2014"  # 原图像根目录（image_path 相对路径的基准）



# ==================== 参数说明 ====================
# TRAINING_DATA_PATH: mask_output.json 文件路径
# IMAGE_FOLDER: 遮挡图像根目录，masked_image 路径如 "./results/xxx.jpg" 会拼接成 "$IMAGE_FOLDER/./results/xxx.jpg"
# ORIGINAL_IMAGE_FOLDER: 原图像根目录，用于加载原始图像
# 注意：Mask DPO 不需要 VISUAL_GENOME_PATH

# ==================== GPU 配置 ====================
CUDA_VISIBLE_DEVICES=2,3,4,5
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1}"
# per device train batch size: 8 (for 40GB GPU), 16 (for 80GB GPU)
PER_DEVICE_BATCH_SIZE=8

# ==================== 调试配置 ====================
# 设置为 1 启用 debugpy 调试模式
DEBUG_MODE="${DEBUG_MODE:0}"
DEBUG_PORT="${DEBUG_PORT:-5679}"
MASTER_PORT="${MASTER_PORT:-44565}"

# ==================== 目录配置（通常无需修改） ====================
TRAIN_DIR="./train"
TRAIN_RESULT_DIR="$TRAIN_DIR/results"
ACCELERATE_CONFIG_DIR="$TRAIN_DIR/accelerate_configs"


# ==================== 检查文件是否存在 ====================
for folder in "$ACCELERATE_CONFIG_DIR"; do
    [ ! -d "$folder" ] && echo "Error: Folder '$folder' not found." && exit 1
done
for file in "$TRAINING_DATA_PATH"; do
    [ ! -f "$file" ] && echo "Error: File '$file' not found." && exit 1
done

# ==================== 计算梯度累积步数 ====================
# number_of_GPU * per_device_train_batch_size * gradient_accumulation_steps = global_batch_size (should be 64)
GLOBAL_BATCH_SIZE=64
NUM_GPUS=$(echo "$GPU_LIST" | tr ',' '\n' | wc -l)
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (NUM_GPUS * PER_DEVICE_BATCH_SIZE)))

echo "============================================"
echo "Mask DPO Training Configuration:"
echo "  - Training data: $TRAINING_DATA_PATH"
echo "  - Input model: $INPUT_MODEL"
echo "  - Output: $TRAIN_RESULT_DIR/$OUTPUT_NAME"
echo "  - GPUs: $GPU_LIST ($NUM_GPUS GPUs)"
echo "  - Batch size: $PER_DEVICE_BATCH_SIZE per device"
echo "  - Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "  - Global batch size: $GLOBAL_BATCH_SIZE"
echo "  - Debug mode: $DEBUG_MODE (port: $DEBUG_PORT)"
echo "============================================"

# ==================== 开始训练 ====================
if [ "$DEBUG_MODE" = "1" ]; then
    echo "Starting in DEBUG mode, connecting to debugpy on port $DEBUG_PORT..."
    echo "Make sure VS Code/Cursor debugger is listening on port $DEBUG_PORT"
    python -m debugpy --connect $DEBUG_PORT $(which deepspeed) \
        --include=localhost:2 --master_port $MASTER_PORT \
        "$TRAIN_DIR/models/mask_dpo_llava.py" \
        --train_data_path "$TRAINING_DATA_PATH" \
        --image_folder "$IMAGE_FOLDER" \
        --masked_image_folder "$MASKED_IMAGE_FOLDER" \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 0 \
        --deepspeed "$ACCELERATE_CONFIG_DIR/zero2.json" \
        --output_dir "$TRAIN_RESULT_DIR/$OUTPUT_NAME" \
        --model_name_or_path "$INPUT_MODEL" \
        --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
        --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --num_train_epochs 1 \
        --eval_strategy "no" \
        --save_strategy "no" \
        --save_total_limit 1 \
        --learning_rate 2e-6 \
        --weight_decay 0. \
        --warmup_steps 0 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb \
        --conv_version v1 \
        --run_name "mask-dpo-llava-v1.5" \
        --beta 0.1
    exit 0
fi

# 正常模式
deepspeed --include localhost:"$GPU_LIST" --master_port $MASTER_PORT "$TRAIN_DIR/models/mask_dpo_llava.py" \
    --train_data_path "$TRAINING_DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --masked_image_folder "$MASKED_IMAGE_FOLDER" \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 0 \
    --deepspeed "$ACCELERATE_CONFIG_DIR/zero2.json" \
    --output_dir "$TRAIN_RESULT_DIR/$OUTPUT_NAME" \
    --model_name_or_path "$INPUT_MODEL" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 1 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 1 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_steps 0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --conv_version v1 \
    --run_name "mask-dpo-llava-v1.5" \
    --beta 0.1
