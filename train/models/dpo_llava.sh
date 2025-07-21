#!/bin/bash
# Description: Training script for LLaVA with DPO

# Set environment variables
GPU_LIST="${CUDA_VISIBLE_DEVICES:-1,2}"

# per device train batch size: 8 (for 40GB GPU), 16 (for 80GB GPU)
PER_DEVICE_BATCH_SIZE=16

# Don't change
TRAIN_DIR="./train"
TRAIN_DATA_DIR="$TRAIN_DIR/data"
TRAIN_RESULT_DIR="$TRAIN_DIR/results"
ACCELERATE_CONFIG_DIR="$TRAIN_DIR/accelerate_configs"

# Check if folders or files exist
for folder in "$VISUAL_GENOME_PATH" "$ACCELERATE_CONFIG_DIR" "$INPUT_MODEL" "$TRAIN_DATA_DIR"; do
    [ ! -d "$folder" ] && echo "Error: Folder '$folder' not found." && exit 1
done
for file in "$TRAINING_DATA_PATH"; do
    [ ! -f "$file" ] && echo "Error: File '$file' not found." && exit 1
done

# number_of_GPU * per_device_train_batch_size * gradient_accumulation_steps = global_batch_size (should be 64)
GLOBAL_BATCH_SIZE=64

# Calculate gradient accumulation steps
NUM_GPUS=$(echo "$GPU_LIST" | tr ',' '\n' | wc -l)
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (NUM_GPUS * PER_DEVICE_BATCH_SIZE)))

# Please check:
# global_batch_size * num_of_steps = total_training_size
# For example: 64 * 150 = 9600

# Run training with deepspeed
deepspeed --include localhost:"$GPU_LIST" --master_port 44564 "$TRAIN_DIR/models/dpo_llava.py" \
    --VISUAL_GENOME_PATH "$VISUAL_GENOME_PATH" --train_data_path "$TRAINING_DATA_PATH" \
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
    --report_to "none" \
    --conv_version v1 \
    --run_name "dpo-llava-v1.5" \
    --beta 0.1
