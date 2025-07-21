#!/bin/bash

GPU_LIST="0,1"

MODEL_NAME="/your/path/to/Qwen2-VL-7B-Instruct"
LORA_NAME="psp-dada/Qwen2-VL-7B-Instruct-SENTINEL"
OUTPUT_NAME="Qwen2_VL_7B_Instruct_SENTINEL_7k"

TEXTVQA_DIR="llava/data/eval/textvqa"
IMAGE_FOLDER="$TEXTVQA_DIR/train_images"

IFS=',' read -ra GPULIST <<<"$GPU_LIST"
CHUNKS=${#GPULIST[@]}
for IDX in $(seq 0 $((CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa \
        --model-name "$MODEL_NAME" \
        --lora-name "$LORA_NAME" \
        --question-file "$TEXTVQA_DIR/llava_textvqa_val_v051_ocr.jsonl" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$TEXTVQA_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --conv-mode vicuna_v1 &

    echo "Now running GPU $IDX at PID $!"

    sleep 1
done
wait

FINAL_OUTPUT_FILE="$TEXTVQA_DIR/answers/$OUTPUT_NAME.jsonl"

# Clear out the output file if it exists.
true >"$FINAL_OUTPUT_FILE"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS - 1))); do
    echo "Concatenating $TEXTVQA_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
    cat "$TEXTVQA_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" >>"$FINAL_OUTPUT_FILE"
    rm "$TEXTVQA_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
done

echo "Converting TextVQA results to TextVQA format"

python -m llava.eval.eval_textvqa \
    --annotation-file "$TEXTVQA_DIR"/TextVQA_0.5.1_val.json \
    --result-file "$FINAL_OUTPUT_FILE" >>"$TEXTVQA_DIR/answers/${OUTPUT_NAME}_res.txt"
