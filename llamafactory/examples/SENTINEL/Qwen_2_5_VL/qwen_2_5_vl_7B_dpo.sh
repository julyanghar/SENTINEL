#!/bin/bash

MODEL_NAME="/your/path/to/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR="saves/Qwen2_5_VL_7B_Instruct_7k_3e_6"
DATASET="Qwen2_5_VL_7B_Instruct_7k"
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
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 1000 \
    --learning_rate 3.0e-6 \
    --num_train_epochs 1 \
    --plot_loss \
    --bf16 \
    --warmup_ratio 0 \
    --ddp_timeout 180000000 \
    --cutoff_len 4096 \
    --max_samples 1000000 \
    --overwrite_output_dir \
    --preprocessing_num_workers 16
