import json

# Convert HallusionBench JSON file to JSONL file for evaluation
input_file = "llava/eval/utils/HallusionBench/HallusionBench.json"
output_file = "llava/eval/utils/HallusionBench/questions.jsonl"


def reorder_dict(item: dict):
    image: str | None = item.get("filename")
    if image:
        image: str = image.replace("./", "")

    # Place the "image" and "question" fields at the beginning
    reordered_item = {"image": image, "question": item.get("question")}
    # Add other fields
    for key, value in item.items():
        if key not in ["question"]:
            reordered_item[key] = value
    return reordered_item


def convert_json_to_jsonl(input_file, output_file):
    # Read the JSON file
    with open(input_file, "r", encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    # Write to the JSONL file
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(reorder_dict(item)) + "\n")


if __name__ == "__main__":
    convert_json_to_jsonl(input_file, output_file)
