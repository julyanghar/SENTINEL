#!/bin/bash

MODEL_NAME="/your/path/to/Qwen2-VL-2B-Instruct"
OUTPUT_DIR="saves/Qwen2_VL_2B_Instruct_12k_2e_6"
DATASET="Qwen2_VL_2B_Instruct_12k"
GPU_LIST="0,1,2,3"

# Make sure: global batch size = 64

deepspeed --include localhost:$GPU_LIST src/train.py \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --model_name_or_path $MODEL_NAME \
    --stage dpo \
    --flash_attn fa2 \
    --do_train \
    --finetuning_type lora \
    --lora_rank 128 \
    --lora_target all \
    --pref_beta 0.1 \
    --pref_loss sigmoid \
    --dataset $DATASET \
    --template qwen2_vl \
    --output_dir $OUTPUT_DIR \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 1000 \
    --learning_rate 2.0e-6 \
    --num_train_epochs 1 \
    --plot_loss \
    --bf16 \
    --warmup_ratio 0 \
    --ddp_timeout 180000000 \
    --cutoff_len 4096 \
    --max_samples 1000000 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16
