#!/bin/bash

GPU_LIST="0,1,2,3,4,5,6,7"


MODEL_NAME="/your/path/to/Qwen2-VL-7B-Instruct"
LORA_NAME="psp-dada/Qwen2-VL-7B-Instruct-SENTINEL"
OUTPUT_NAME="Qwen2_VL_7B_Instruct_SENTINEL_7k"

POPE_DIR="llava/data/eval/pope"
POPE_QUESTION_FILE="llava_pope_test.jsonl"
POPE_ANNO_DIR="$POPE_DIR/coco"
IMAGE_FOLDER=/home/zhuotaotian/psp/llm/utils/data/MSCOCO/coco2014/val2014

IFS=',' read -ra GPULIST <<<"$GPU_LIST"
CHUNKS=${#GPULIST[@]}
for IDX in $(seq 0 $((CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa \
        --model-name "$MODEL_NAME" \
        --lora-name "$LORA_NAME" \
        --question-file "$POPE_DIR"/$POPE_QUESTION_FILE \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$POPE_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" \
        --temperature 0 \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --seed 0 \
        --conv-mode vicuna_v1 &
done
wait

FINAL_OUTPUT_FILE=$POPE_DIR/answers/$OUTPUT_NAME.jsonl
# Clear out the output file if it exists.
true >"$FINAL_OUTPUT_FILE"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS - 1))); do
    echo "Concatenating $POPE_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
    cat "$POPE_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" >>"$FINAL_OUTPUT_FILE"
    rm "$POPE_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
done

python llava/eval/utils/eval_pope.py \
    --annotation-dir "$POPE_ANNO_DIR" \
    --question-file "$POPE_DIR"/$POPE_QUESTION_FILE \
    --answer-file "$FINAL_OUTPUT_FILE" \
    --output-file "$POPE_DIR/answers/${OUTPUT_NAME}_eval.txt"
