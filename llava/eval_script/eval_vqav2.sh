#!/bin/bash

GPU_LIST="0,1"

MODEL_NAME="/your/path/to/Qwen2-VL-7B-Instruct"
LORA_NAME="psp-dada/Qwen2-VL-7B-Instruct-SENTINEL"
OUTPUT_NAME="Qwen2_VL_7B_Instruct_SENTINEL_7k"

VQAV2_DIR="llava/data/eval/vqav2"
IMAGE_FOLDER="llava/data/MSCOCO/coco2015/test2015"
CONVERT_FILE_PATH="llava/eval/utils/convert_vqav2_for_submission.py"

SPLIT="llava_vqav2_mscoco_test-dev2015"

IFS=',' read -ra GPULIST <<<"$GPU_LIST"
CHUNKS=${#GPULIST[@]}
for IDX in $(seq 0 $((CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa \
        --model-name "$MODEL_NAME" \
        --lora-name "$LORA_NAME" \
        --question-file "$VQAV2_DIR/$SPLIT.jsonl" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$VQAV2_DIR/answers/$SPLIT/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --seed 0 \
        --conv-mode vicuna_v1 &

    echo "Now running GPU $IDX at PID $!"

    sleep 1
done
wait

FINAL_OUTPUT_FILE=$VQAV2_DIR/answers/$SPLIT/$OUTPUT_NAME.jsonl

# Clear out the output file if it exists.
true >"$FINAL_OUTPUT_FILE"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS - 1))); do
    echo "Concatenating $VQAV2_DIR/answers/$SPLIT/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
    cat "$VQAV2_DIR/answers/$SPLIT/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" >>"$FINAL_OUTPUT_FILE"
    rm "$VQAV2_DIR/answers/$SPLIT/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
done

python "$CONVERT_FILE_PATH" --dir "$VQAV2_DIR" --split $SPLIT --output $OUTPUT_NAME
