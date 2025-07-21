#!/bin/bash

GPU_LIST="0,1"

MODEL_NAME="/your/path/to/Qwen2-VL-7B-Instruct"
LORA_NAME="psp-dada/Qwen2-VL-7B-Instruct-SENTINEL"
OUTPUT_NAME="Qwen2_VL_7B_Instruct_SENTINEL_7k"

COCO_PATH="llava/data/MSCOCO"
OBJ_HALBENCH_DIR="llava/data/eval/object_halbench"
QUESTION_FILE="$OBJ_HALBENCH_DIR/Object_HalBench.jsonl"
# The question file is a jsonl file with each line containing a dictionary with:
# {"image_id": ..., "image_path": ..., "question": ...}

EVAL_FILE="llava/eval/utils/eval_gpt_obj_halbench.py"

IFS=',' read -ra GPULIST <<<"$GPU_LIST"
CHUNKS=${#GPULIST[@]}
for IDX in $(seq 0 $((CHUNKS - 1))); do
    echo "Now GPU ${GPULIST[$IDX]} is processing chunk $IDX"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa \
        --model-name "$MODEL_NAME" \
        --lora-name "$LORA_NAME" \
        --question-file "$QUESTION_FILE" \
        --answers-file "./${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --seed 0 \
        --conv-mode vicuna_v1 &

    echo "Now running GPU $IDX at PID $!"

    sleep 1
done
wait

FINAL_OUTPUT_FILE="$OBJ_HALBENCH_DIR/$OUTPUT_NAME.jsonl"

# Clear out the output file if it exists.
true >"$FINAL_OUTPUT_FILE"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS - 1))); do
    echo "Concatenating ./${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
    cat "./${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" >>"$FINAL_OUTPUT_FILE"
    rm "./${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
done

python "$EVAL_FILE" \
    --cap_file "$FINAL_OUTPUT_FILE" \
    --cap_folder "" \
    --coco_path "$COCO_PATH/coco2014/annotations" \
    --sample_num -1
