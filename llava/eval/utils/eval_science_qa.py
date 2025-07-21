import json
import os
import re
from argparse import ArgumentParser, Namespace


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--base-dir", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--output-file", type=str, default="a_output.jsonl")
    parser.add_argument("--output-result", type=str, default="a_result.json")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--options", type=list, default=["A", "B", "C", "D", "E"])
    args = parser.parse_args()
    return args


def convert_caps(results):
    fakecaps = []
    for result in results:
        image_id = result["question_id"]
        caption = result["text"]
        fakecaps.append({"image_id": int(image_id), "caption": caption})
    return fakecaps


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[: len(choices)]:
        return options.index(prediction)
    else:
        return -1


if __name__ == "__main__":
    args: Namespace = get_args()

    base_dir: str = args.base_dir
    split_indices: list[str] = json.load(open(os.path.join(base_dir, "pid_splits.json")))[args.split]
    # idx -> data (qproblem)
    problems: dict[str, dict] = json.load(open(os.path.join(base_dir, "problems.json")))

    # data (result)
    predictions: list[dict] = [json.loads(line) for line in open(args.result_file)]
    # question_id -> data (result)
    predictions: dict[str, dict] = {pred["question_id"]: pred for pred in predictions}
    # idx -> data
    split_problems: dict[str, dict] = {idx: problems[idx] for idx in split_indices}

    results = {"correct": [], "incorrect": []}
    sqa_results = {}
    sqa_results["acc"] = None
    sqa_results["correct"] = None
    sqa_results["count"] = None
    sqa_results["results"] = {}
    sqa_results["outputs"] = {}

    for prob_id, prob in split_problems.items():
        if prob_id not in predictions:
            pred = {"text": "FAILED", "prompt": "Unknown"}
            pred_text = "FAILED"
        else:
            pred: dict[str, str] = predictions[prob_id]
            pred_text: str = pred["text"]  # The output of model to be evaluated

        if pred_text in args.options:
            answer = pred_text
        elif len(pred_text) >= 3 and pred_text[0] in args.options and pred_text[1:3] == ". ":
            answer = pred_text[0]
        elif len(pred_text) == 2 and pred_text[0] in args.options and pred_text[1] == ".":
            answer = pred_text[0]
        else:
            pattern = re.compile(r"The answer is ([A-Z]).")
            res = pattern.findall(pred_text)
            if len(res) == 1:
                answer = res[0]  # 'A', 'B', ...
            else:
                answer = "FAILED"

        pred_idx: int = get_pred_idx(answer, prob["choices"], args.options)

        analysis = {
            "question_id": prob_id,
            "parsed_ans": answer,
            "ground_truth": args.options[prob["answer"]],
            "question": pred["prompt"],
            "pred": pred_text,
            "is_multimodal": "<image>" in pred["prompt"],
        }

        sqa_results["results"][prob_id] = get_pred_idx(answer, prob["choices"], args.options)
        sqa_results["outputs"][prob_id] = pred_text

        if pred_idx == prob["answer"]:
            results["correct"].append(analysis)
        else:
            results["incorrect"].append(analysis)

    correct = len(results["correct"])
    total = len(results["correct"]) + len(results["incorrect"])

    ###### IMG ######
    multimodal_correct = len([x for x in results["correct"] if x["is_multimodal"]])
    multimodal_incorrect = len([x for x in results["incorrect"] if x["is_multimodal"]])
    multimodal_total = multimodal_correct + multimodal_incorrect
    ###### IMG ######

    acc = correct / total * 100
    image_acc = multimodal_correct / multimodal_total * 100
    print(f"Total: {total}, Correct: {correct}, Accuracy: {acc:.2f}%, IMG-Accuracy: {image_acc:.2f}%")

    sqa_results["acc"] = acc
    sqa_results["correct"] = correct
    sqa_results["count"] = total

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    with open(args.output_result, "w") as f:
        json.dump(sqa_results, f, indent=2)
