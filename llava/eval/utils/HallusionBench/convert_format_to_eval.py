import json
import os
from argparse import ArgumentParser

import jsonlines


def load_jsonl(file_path: str) -> list:
    with jsonlines.open(file_path) as reader:
        return list(reader)


def extract_image(path):
    """
    Extract the key part of the path, for example:
    Input: '/home/.../HallusionBench/images/VS/chart/0_1.png'
    Output: 'VS/chart/0_1.png'
    """
    if path is None:
        return None
    # Normalize the path to handle differences between ./ and /
    normalized_path = os.path.normpath(path)

    # Split the path into parts
    parts = normalized_path.split(os.sep)

    # Extract the last three parts (VS/chart/0_1.png)
    if len(parts) >= 3:
        return os.path.join(*parts[-3:])
    else:
        # If the path has less than three parts, return the full path directly
        return normalized_path


def merge_data(questions: list, answers: list) -> list:
    merged_list = []
    answer_prompt_key: str = "prompt" if "prompt" in answers[0] else "question"
    answer_caption_key: str = "text" if "text" in answers[0] else "caption"
    answer_id_key: str = "question_id" if "question_id" in answers[0] else "image_id"
    for question in questions:
        for answer in answers:
            if (
                question["question"] == answer[answer_prompt_key]
                and extract_image(question["image"]) == extract_image(answer["image_path"])
                and question["question_id"] == answer[answer_id_key]
            ):
                merged_dict: dict = question.copy()
                merged_dict["model_prediction"] = answer[answer_caption_key]
                merged_dict.pop("image")
                merged_list.append(merged_dict)
                break
    return merged_list


def write_output(data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    parser = ArgumentParser(
        description="Merge questions and answers into a single JSON file based on matching criteria."
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default="HallusionBench/questions.jsonl",
        help="Path to the input questions JSONL file.",
    )
    parser.add_argument(
        "--answers_file",
        type=str,
        required=True,
        help="Path to the input answers JSONL file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="HallusionBench/answers/llava-v1.5-7b_convert.json",
        help="Path to the output JSON file.",
    )

    args = parser.parse_args()
    questions = load_jsonl(args.questions_file)
    answers = load_jsonl(args.answers_file)

    merged_data = merge_data(questions, answers)

    print(f"The merged data has {len(merged_data)} items.")

    write_output(merged_data, args.output_file)


if __name__ == "__main__":
    main()
