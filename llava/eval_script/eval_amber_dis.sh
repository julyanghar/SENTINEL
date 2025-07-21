#!/bin/bash

GPU_LIST="0,1"

MODEL_NAME="/your/path/to/Qwen2-VL-7B-Instruct"
LORA_NAME="psp-dada/Qwen2-VL-7B-Instruct-SENTINEL"
OUTPUT_NAME="Qwen2_VL_7B_Instruct_SENTINEL_7k"

AMBER_DIR="llava/data/eval/AMBER"
AMBER_DATA_DIR="$AMBER_DIR/data"
AMBER_DIS_DIR="$AMBER_DIR/amber_dis"
IMAGE_FOLDER_DIR="$AMBER_DIR/images"
QUESTION_FILE="$AMBER_DATA_DIR/query/query_discriminative.json"

AMBER_EVAL_FILE="llava/eval/utils/AMBER_EVAL_FILE.py"

IFS=',' read -ra GPULIST <<<"$GPU_LIST"
CHUNKS=${#GPULIST[@]}
for IDX in $(seq 0 $((CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa \
        --model-name "$MODEL_NAME" \
        --lora-name "$LORA_NAME" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$IMAGE_FOLDER_DIR" \
        --answers-file "$AMBER_DIS_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" \
        --temperature 0 \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --max-new-tokens 1 \
        --seed 0 \
        --conv-mode vicuna_v1 &

    echo "Now running GPU $IDX at PID $!"

    sleep 1
done
wait

FINAL_OUTPUT_FILE="$AMBER_DIS_DIR/answers/$OUTPUT_NAME.jsonl"

# Clear out the output file if it exists.
true >"$FINAL_OUTPUT_FILE"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS - 1))); do
    echo "Concatenating $AMBER_DIS_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
    cat "$AMBER_DIS_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" >>"$FINAL_OUTPUT_FILE"
    rm "$AMBER_DIS_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
done

# Identical to: save_path.replace(".jsonl", "_eval_amber.jsonl")
EVAL_OUTPUT_FILE="${FINAL_OUTPUT_FILE/.jsonl/_eval_amber.jsonl}"

python "$AMBER_EVAL_FILE" \
    --evaluation_type d \
    --similar_score_threshold 0.8 \
    --inference_data "$FINAL_OUTPUT_FILE" \
    --metrics "$AMBER_DATA_DIR/metrics.txt" \
    --safe_words "$AMBER_DATA_DIR/safe_words.txt" \
    --annotation "$AMBER_DATA_DIR/annotations.json" \
    --word_association "$AMBER_DATA_DIR/relation.json" \
    --save_file "$EVAL_OUTPUT_FILE"
