import argparse
import json
import os


def eval_pope(answers, label_file):
    label_list = [json.loads(q)["label"] for q in open(label_file, "r")]

    for answer in answers:
        if "text" in answer:
            text = answer["text"]
        elif "caption" in answer:
            text = answer["caption"]
        else:
            text = answer["answer"]

        # Only keep the first sentence
        if text.find(".") != -1:
            text = text.split(".")[0]

        text = text.replace(",", "")
        words = text.split(" ")
        if "No" in words or "not" in words or "no" in words:
            answer["text"] = "no"
        else:
            answer["text"] = "yes"

    for i in range(len(label_list)):
        if label_list[i] == "no":
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer["text"] == "no":
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print("TP\tFP\tTN\tFN\t")
    print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print("Accuracy: {}".format(acc))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 score: {}".format(f1))
    print("Yes ratio: {}".format(yes_ratio))
    return acc, precision, recall, f1, yes_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answer-file", type=str, help="The file containing the answers to the questions")
    parser.add_argument("--output-file", type=str, help="The file to write the results to")

    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question["question_id"]: question for question in questions}
    answers = [json.loads(q) for q in open(args.answer_file)]

    # 初始化字典来记录每个类别的结果
    categories = ["random", "popular", "adversarial"]
    metrics = {category: {"acc": 0, "precision": 0, "recall": 0, "f1": 0} for category in categories}

    for file in os.listdir(args.annotation_dir):
        assert file.startswith("coco_pope_")
        assert file.endswith(".json")
        category = file[10:-5]

        if "image_id" in answers[0]:
            for answer in answers:
                answer["question_id"] = answer["image_id"]
        cur_answers = [x for x in answers if questions[x["question_id"]]["category"] == category]
        print("Category: {}, # samples: {}".format(category, len(cur_answers)))
        acc, precision, recall, f1, yes_ratio = eval_pope(cur_answers, os.path.join(args.annotation_dir, file))
        print("====================================")

        # 记录每个类别的指标
        metrics[category]["acc"] = acc
        metrics[category]["precision"] = precision
        metrics[category]["recall"] = recall
        metrics[category]["f1"] = f1
        metrics[category]["yes_ratio"] = yes_ratio

    # 打印每个类别的指标
    with open(args.output_file, "w") as f:
        f.write("\t\tRandom/Popular/Adversarial\n")
        f.write(
            "Accuracy: {:.2f}/{:.2f}/{:.2f}\n".format(
                metrics["random"]["acc"] * 100, metrics["popular"]["acc"] * 100, metrics["adversarial"]["acc"] * 100
            )
        )
        f.write(
            "Precision: {:.2f}/{:.2f}/{:.2f}\n".format(
                metrics["random"]["precision"] * 100,
                metrics["popular"]["precision"] * 100,
                metrics["adversarial"]["precision"] * 100,
            )
        )
        f.write(
            "Recall: {:.2f}/{:.2f}/{:.2f}\n".format(
                metrics["random"]["recall"] * 100,
                metrics["popular"]["recall"] * 100,
                metrics["adversarial"]["recall"] * 100,
            )
        )
        f.write(
            "F1 score: {:.2f}/{:.2f}/{:.2f}\n".format(
                metrics["random"]["f1"] * 100, metrics["popular"]["f1"] * 100, metrics["adversarial"]["f1"] * 100
            )
        )
        f.write(
            "Yes ratio: {:.2f}/{:.2f}/{:.2f}\n".format(
                metrics["random"]["yes_ratio"] * 100,
                metrics["popular"]["yes_ratio"] * 100,
                metrics["adversarial"]["yes_ratio"] * 100,
            )
        )
