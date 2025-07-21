#!/bin/bash

GPU_LIST="0,1"

MODEL_NAME="/your/path/to/Qwen2-VL-7B-Instruct"
LORA_NAME="psp-dada/Qwen2-VL-7B-Instruct-SENTINEL"
OUTPUT_NAME="Qwen2_VL_7B_Instruct_SENTINEL_7k"

SQA_DIR="llava/data/eval/scienceqa"
EVAL_SQA="llava/eval/utils/eval_science_qa.py"

IFS=',' read -ra GPULIST <<<"$GPU_LIST"
CHUNKS=${#GPULIST[@]}
for IDX in $(seq 0 $((CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_science \
        --model-name "$MODEL_NAME" \
        --lora-name "$LORA_NAME" \
        --question-file "$SQA_DIR/llava_test_CQM-A.json" \
        --image-folder "$SQA_DIR/images/test" \
        --answers-file "$SQA_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" \
        --single-pred-prompt \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --conv-mode vicuna_v1 &

    echo "Now running GPU $IDX at PID $!"

    sleep 1
done
wait

FINAL_OUTPUT_FILE="$SQA_DIR/answers/${OUTPUT_NAME}.jsonl"

# Clear out the output file if it exists.
true >"$FINAL_OUTPUT_FILE"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS - 1))); do
    echo "Concatenating $SQA_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
    cat "$SQA_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" >>"$FINAL_OUTPUT_FILE"
    rm "$SQA_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
done

mkdir -p "$SQA_DIR/temp"
mkdir -p "$SQA_DIR/results"

python "$EVAL_SQA" \
    --base-dir "$SQA_DIR" \
    --result-file "$FINAL_OUTPUT_FILE" \
    --output-file "$SQA_DIR/temp/${OUTPUT_NAME}_output.jsonl" \
    --output-result "$SQA_DIR/temp/${OUTPUT_NAME}_result.json" >>"$SQA_DIR/results/${OUTPUT_NAME}_result.txt"
