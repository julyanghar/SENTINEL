#!/bin/bash

GPU_LIST="0,1"

MODEL_NAME="/your/path/to/Qwen2-VL-7B-Instruct"
LORA_NAME="psp-dada/Qwen2-VL-7B-Instruct-SENTINEL"
OUTPUT_NAME="Qwen2_VL_7B_Instruct_SENTINEL_7k"

MMVET_DIR="llava/data/eval/mm-vet"
IMAGE_FOLDER="$MMVET_DIR/images"

IFS=',' read -ra GPULIST <<<"$GPU_LIST"
CHUNKS=${#GPULIST[@]}
for IDX in $(seq 0 $((CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa \
        --model-name "$MODEL_NAME" \
        --lora-name "$LORA_NAME" \
        --question-file "$MMVET_DIR/llava-mm-vet.jsonl" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$MMVET_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --conv-mode vicuna_v1 &

    echo "Now running GPU $IDX at PID $!"

    sleep 1
done
wait

FINAL_OUTPUT_FILE="$MMVET_DIR/answers/$OUTPUT_NAME.jsonl"

# Clear out the output file if it exists.
true >"$FINAL_OUTPUT_FILE"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS - 1))); do
    echo "Concatenating $MMVET_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
    cat "$MMVET_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" >>"$FINAL_OUTPUT_FILE"
    rm "$MMVET_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
done

echo "Converting MM-Vet results to MM-Vet format"

python "$PROJECT_DIR"/llava/eval/convert_mmvet_for_eval.py \
    --src "$FINAL_OUTPUT_FILE" \
    --dst "$MMVET_DIR/results/$OUTPUT_NAME.json"

echo "Please submit the file at $MMVET_DIR/results/$OUTPUT_NAME.json to https://huggingface.co/spaces/whyu/MM-Vet_Evaluator"
echo "done"
