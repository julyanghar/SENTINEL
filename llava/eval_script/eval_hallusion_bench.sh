#!/bin/bash

GPU_LIST="0,1"

MODEL_NAME="/your/path/to/Qwen2-VL-7B-Instruct"
LORA_NAME="psp-dada/Qwen2-VL-7B-Instruct-SENTINEL"
OUTPUT_NAME="Qwen2_VL_7B_Instruct_SENTINEL_7k"

HALLUSION_BENCH_DIR="llava/data/eval/HallusionBench"
IMAGE_FOLDER="$HALLUSION_BENCH_DIR/images"

HALLUSION_BENCH_EVAL_DIR="llava/eval/utils/HallusionBench"
QUESTION_FILE="$HALLUSION_BENCH_EVAL_DIR/questions.jsonl"

IFS=',' read -ra GPULIST <<<"$GPU_LIST"
CHUNKS=${#GPULIST[@]}
for IDX in $(seq 0 $((CHUNKS - 1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa \
        --model-name "$MODEL_NAME" \
        --lora-name "$LORA_NAME" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$HALLUSION_BENCH_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" \
        --temperature 0.0 \
        --num-chunks "$CHUNKS" \
        --max-new-tokens 256 \
        --chunk-idx "$IDX" \
        --conv-mode vicuna_v1 &

    echo "Now running GPU $IDX at PID $!"

    sleep 1
done
wait

FINAL_OUTPUT_FILE="$HALLUSION_BENCH_DIR/answers/$OUTPUT_NAME.jsonl"

# Clear out the output file if it exists.
true >"$FINAL_OUTPUT_FILE"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS - 1))); do
    echo "$HALLUSION_BENCH_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
    cat "$HALLUSION_BENCH_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl" >>"$FINAL_OUTPUT_FILE"
    rm "$HALLUSION_BENCH_DIR/answers/${OUTPUT_NAME}_${CHUNKS}_${IDX}.jsonl"
done

# Create necessary directories for evaluation
mkdir -p "$HALLUSION_BENCH_DIR/temp"
mkdir -p "$HALLUSION_BENCH_DIR/reviews"

CONVERT_FILE="$HALLUSION_BENCH_DIR/answers/${OUTPUT_NAME}_convert.json"

# Convert the output format to the evaluation format
python3 "$HALLUSION_BENCH_EVAL_DIR/convert_format_to_eval.py" \
    --questions_file "$HALLUSION_BENCH_EVAL_DIR/questions.jsonl" \
    --answers_file "$FINAL_OUTPUT_FILE" \
    --output_file "$CONVERT_FILE"

# Run the evaluation script
TEMP_FILE="$HALLUSION_BENCH_DIR/temp/${OUTPUT_NAME}_temp.json"
python "$HALLUSION_BENCH_EVAL_DIR/evaluation.py" \
    --input_file "$CONVERT_FILE" \
    --temp_file "$TEMP_FILE" \
    --output_file "$HALLUSION_BENCH_DIR/reviews/${OUTPUT_NAME}_review.json" \
    --load_json
