import json
import os
import sys
import warnings
from argparse import ArgumentParser, Namespace

import nltk
import spacy
from nltk.stem import WordNetLemmatizer
from spacy.language import Language
from spacy.tokens import Doc
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from llava.eval.utils.utils import read_json


def save_result(save_path: str, results: dict | list[dict]):
    if not save_path or not results:
        return
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, "a+", encoding="utf-8") as f:
        if isinstance(results, list):
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            f.write(json.dumps(results, ensure_ascii=False) + "\n")


def main(args: Namespace) -> None:
    spacy.require_gpu()

    nlp: Language = spacy.load("en_core_web_trf")
    lemmatizer = WordNetLemmatizer()
    warnings.filterwarnings("ignore", category=UserWarning)

    def check_synonyms_word(word1: str, word2: str, similar_score_threshold: float) -> bool:
        token1: Doc = nlp(word1)
        token2: Doc = nlp(word2)
        similarity: float = token1.similarity(token2)
        return similarity > similar_score_threshold

    def extract_nouns(text: str) -> list[str]:
        tokens: list[str] = nltk.word_tokenize(text)
        tagged: list[tuple] = nltk.pos_tag(tokens)
        nouns: list[str] = [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith("NN")]
        return nouns

    def init(args: Namespace) -> dict[str, float]:
        metrics: dict[str, float] = {}
        with open(args.metrics, "r") as file:
            lines: list[str] = file.readlines()

        for line in lines:
            parts: list[str] = line.strip().split("=")
            if len(parts) == 2:
                variable_name: str = parts[0].strip()
                variable_value: float = eval(parts[1].strip())
                metrics[variable_name] = variable_value

        return metrics

    ## Main
    metrics: dict[str, float] = init(args)
    association: dict = json.load(open(args.word_association, "r", encoding="utf-8"))
    hallucination_words: list[str] = []
    for word1 in association.keys():
        hallucination_words.append(word1)
        for word2 in association[word1]:
            hallucination_words.append(word2)

    global_safe_words: list[str] = []
    with open(args.safe_words, "r", encoding="utf-8") as safe_file:
        for line in safe_file:
            line = line.split("\n")[0]
            global_safe_words.append(line)

    dimension = {"g": False, "de": False, "da": False, "dr": False}
    if args.evaluation_type == "a":
        for key in dimension.keys():
            dimension[key] = True
    elif args.evaluation_type == "g":
        dimension["g"] = True
    elif args.evaluation_type == "d":
        dimension["de"] = True
        dimension["da"] = True
        dimension["dr"] = True
    else:
        dimension[args.evaluation_type] = True

    inference_data: list[dict] = read_json(args.inference_data)
    ground_truth: list[dict] = read_json(args.annotation)
    print(f"len(inference_data): {len(inference_data)}")

    for i in tqdm(range(len(inference_data))):
        if "id" in inference_data[i]:
            id: int = inference_data[i]["id"]
        elif "question_id" in inference_data[i]:
            id: int = inference_data[i]["question_id"]
        elif "image_id" in inference_data[i]:
            id: int = inference_data[i]["image_id"]

        if "response" in inference_data[i]:
            response: str = inference_data[i]["response"]
        elif "caption" in inference_data[i]:
            response: str = inference_data[i]["caption"]
        elif "text" in inference_data[i]:
            response: str = inference_data[i]["text"]
        if ground_truth[id - 1]["type"] == "generative":
            nouns = extract_nouns(response)
            after_process_nouns = []
            for noun in nouns:
                if noun in hallucination_words:
                    after_process_nouns.append(noun)

            safe_words = []
            safe_list = []
            for idx, word in enumerate(ground_truth[id - 1]["truth"]):
                safe_words += association[word]
                safe_list += [idx] * len(association[word])

            ha_words = []
            ha_list = []
            for idx, word in enumerate(ground_truth[id - 1]["hallu"]):
                ha_words += association[word]
                ha_list += [idx] * len(association[word])

            safe_words += ground_truth[id - 1]["truth"]
            safe_len = len(ground_truth[id - 1]["truth"])
            safe_list += [0] * safe_len
            safe_flag_list = [0] * len(after_process_nouns)

            ha_words += ground_truth[id - 1]["hallu"]
            ha_len = len(ground_truth[id - 1]["hallu"])
            ha_list += [0] * ha_len

            for idx, noun in enumerate(after_process_nouns):
                if noun in global_safe_words:
                    continue

                if noun in safe_words:
                    for j in range(len(safe_words)):
                        if noun == safe_words[j]:
                            if j < (len(safe_list) - safe_len):
                                safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                            else:
                                safe_list[j] = 1
                            break
                    continue

                if noun in ha_words:
                    for j in range(len(ha_words)):
                        if noun == ha_words[j]:
                            if j < (len(ha_list) - ha_len):
                                ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                            else:
                                ha_list[j] = 1
                            break

                for j, check_word in enumerate(ha_words):
                    if check_synonyms_word(noun, check_word, args.similar_score_threshold):
                        if j < (len(ha_list) - ha_len):
                            ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                        else:
                            ha_list[j] = 1
                        break

                flag: bool = False
                for j, check_word in enumerate(safe_words):
                    if check_synonyms_word(noun, check_word, args.similar_score_threshold):
                        flag = True
                        if j < (len(safe_list) - safe_len):
                            safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                        else:
                            safe_list[j] = 1
                        break
                if flag:
                    continue

                safe_flag_list[idx] = 1

            metrics["chair_score"] += sum(safe_flag_list)
            metrics["chair_num"] += len(safe_flag_list)
            metrics["safe_cover_score"] += sum(safe_list[-safe_len:])
            metrics["safe_cover_num"] += len(safe_list[-safe_len:])
            metrics["hallu_cover_score"] += sum(ha_list[-ha_len:])
            metrics["hallu_cover_num"] += len(ha_list[-ha_len:])
            if sum(safe_flag_list) == 0:
                metrics["non_hallu_score"] += 1
            metrics["non_hallu_num"] += 1

        else:
            metrics["qa_correct_num"] += 1
            if ground_truth[id - 1]["type"] == "discriminative-attribute-state":
                metrics["as_qa_correct_num"] += 1
            elif ground_truth[id - 1]["type"] == "discriminative-attribute-number":
                metrics["an_qa_correct_num"] += 1
            elif ground_truth[id - 1]["type"] == "discriminative-attribute-action":
                metrics["aa_qa_correct_num"] += 1
            elif ground_truth[id - 1]["type"] == "discriminative-hallucination":
                metrics["ha_qa_correct_num"] += 1
            else:
                metrics["asso_qa_correct_num"] += 1

            truth = ground_truth[id - 1]["truth"]
            if truth == "yes":
                if response in ["Yes", "yes"]:
                    metrics["qa_correct_score"] += 1
                    if ground_truth[id - 1]["type"] == "discriminative-attribute-state":
                        metrics["as_qa_correct_score"] += 1
                    elif ground_truth[id - 1]["type"] == "discriminative-attribute-number":
                        metrics["an_qa_correct_score"] += 1
                    elif ground_truth[id - 1]["type"] == "discriminative-attribute-action":
                        metrics["aa_qa_correct_score"] += 1
                    elif ground_truth[id - 1]["type"] == "discriminative-hallucination":
                        metrics["ha_qa_correct_score"] += 1
                    else:
                        metrics["asso_qa_correct_score"] += 1
            else:
                metrics["qa_no_num"] += 1
                if ground_truth[id - 1]["type"] == "discriminative-attribute-state":
                    metrics["as_qa_no_num"] += 1
                elif ground_truth[id - 1]["type"] == "discriminative-attribute-number":
                    metrics["an_qa_no_num"] += 1
                elif ground_truth[id - 1]["type"] == "discriminative-attribute-action":
                    metrics["aa_qa_no_num"] += 1
                elif ground_truth[id - 1]["type"] == "discriminative-hallucination":
                    metrics["ha_qa_no_num"] += 1
                else:
                    metrics["asso_qa_no_num"] += 1

                if response in ["No", "no"]:
                    metrics["qa_correct_score"] += 1
                    metrics["qa_no_score"] += 1
                    if ground_truth[id - 1]["type"] == "discriminative-attribute-state":
                        metrics["as_qa_correct_score"] += 1
                        metrics["as_qa_no_score"] += 1
                    elif ground_truth[id - 1]["type"] == "discriminative-attribute-number":
                        metrics["an_qa_correct_score"] += 1
                        metrics["an_qa_no_score"] += 1
                    elif ground_truth[id - 1]["type"] == "discriminative-attribute-action":
                        metrics["aa_qa_correct_score"] += 1
                        metrics["aa_qa_no_score"] += 1
                    elif ground_truth[id - 1]["type"] == "discriminative-hallucination":
                        metrics["ha_qa_correct_score"] += 1
                        metrics["ha_qa_no_score"] += 1
                    else:
                        metrics["asso_qa_correct_score"] += 1
                        metrics["asso_qa_no_score"] += 1

            if response in ["No", "no"]:
                metrics["qa_ans_no_num"] += 1
                if ground_truth[id - 1]["type"] == "discriminative-attribute-state":
                    metrics["as_qa_ans_no_num"] += 1
                elif ground_truth[id - 1]["type"] == "discriminative-attribute-number":
                    metrics["an_qa_ans_no_num"] += 1
                elif ground_truth[id - 1]["type"] == "discriminative-attribute-action":
                    metrics["aa_qa_ans_no_num"] += 1
                elif ground_truth[id - 1]["type"] == "discriminative-hallucination":
                    metrics["ha_qa_ans_no_num"] += 1
                else:
                    metrics["asso_qa_ans_no_num"] += 1
                if truth == "no":
                    metrics["qa_ans_no_score"] += 1
                    if ground_truth[id - 1]["type"] == "discriminative-attribute-state":
                        metrics["as_qa_ans_no_score"] += 1
                    elif ground_truth[id - 1]["type"] == "discriminative-attribute-number":
                        metrics["an_qa_ans_no_score"] += 1
                    elif ground_truth[id - 1]["type"] == "discriminative-attribute-action":
                        metrics["aa_qa_ans_no_score"] += 1
                    elif ground_truth[id - 1]["type"] == "discriminative-hallucination":
                        metrics["ha_qa_ans_no_score"] += 1
                    else:
                        metrics["asso_qa_ans_no_score"] += 1

    # 保存结果，用于输出
    results: dict[dict] = {}
    if dimension["g"]:
        CHAIR = round(metrics["chair_score"] / metrics["chair_num"] * 100, 1)
        Cover = round(metrics["safe_cover_score"] / metrics["safe_cover_num"] * 100, 1)
        Ha = round(metrics["hallu_cover_score"] / metrics["hallu_cover_num"] * 100, 1)
        Ha_p = round(100 - metrics["non_hallu_score"] / metrics["non_hallu_num"] * 100, 1)
        results["Generative Task"] = {"CHAIR": CHAIR, "Cover": Cover, "Hal": Ha_p, "Cog": Ha}

    if dimension["de"] and dimension["da"] and dimension["dr"]:
        Accuracy = round(metrics["qa_correct_score"] / metrics["qa_correct_num"] * 100, 1)
        Precision = round(metrics["qa_ans_no_score"] / metrics["qa_ans_no_num"] * 100, 1)
        Recall = round(metrics["qa_no_score"] / metrics["qa_no_num"] * 100, 1)
        F1 = round(2 * (Precision / 100) * (Recall / 100) / ((Precision / 100) + (Recall / 100) + 0.0001) * 100, 1)
        results["Descriminative Task"] = {"Accuracy": Accuracy, "Precision": Precision, "Recall": Recall, "F1": F1}

    if dimension["de"]:
        hallucination_Accuracy = round(metrics["ha_qa_correct_score"] / metrics["ha_qa_correct_num"] * 100, 1)
        hallucination_Precision = round(metrics["ha_qa_ans_no_score"] / metrics["ha_qa_ans_no_num"] * 100, 1)
        hallucination_Recall = round(metrics["ha_qa_no_score"] / metrics["ha_qa_no_num"] * 100, 1)
        hallucination_F1 = round(
            2
            * (hallucination_Precision / 100)
            * (hallucination_Recall / 100)
            / ((hallucination_Precision / 100) + (hallucination_Recall / 100) + 0.001)
            * 100,
            1,
        )
        results["Exsitence"] = {
            "Accuracy": hallucination_Accuracy,
            "Precision": hallucination_Precision,
            "Recall": hallucination_Recall,
            "F1": hallucination_F1,
        }

    if dimension["da"]:
        attr_Accuracy = round(
            (metrics["as_qa_correct_score"] + metrics["an_qa_correct_score"] + metrics["aa_qa_correct_score"])
            / (metrics["as_qa_correct_num"] + metrics["an_qa_correct_num"] + metrics["aa_qa_correct_num"])
            * 100,
            1,
        )
        attr_Precision = round(
            (metrics["as_qa_ans_no_score"] + metrics["an_qa_ans_no_score"] + metrics["aa_qa_ans_no_score"])
            / (metrics["as_qa_ans_no_num"] + metrics["an_qa_ans_no_num"] + metrics["aa_qa_ans_no_num"])
            * 100,
            1,
        )
        attr_Recall = round(
            (metrics["as_qa_no_score"] + metrics["an_qa_no_score"] + metrics["aa_qa_no_score"])
            / (metrics["as_qa_no_num"] + metrics["an_qa_no_num"] + metrics["aa_qa_no_num"])
            * 100,
            1,
        )
        attr_F1 = round(
            2
            * (attr_Precision / 100)
            * (attr_Recall / 100)
            / ((attr_Precision / 100) + (attr_Recall / 100) + 0.0001)
            * 100,
            1,
        )
        state_Accuracy = round(metrics["as_qa_correct_score"] / metrics["as_qa_correct_num"] * 100, 1)
        state_Precision = round(metrics["as_qa_ans_no_score"] / metrics["as_qa_ans_no_num"] * 100, 1)
        state_Recall = round(metrics["as_qa_no_score"] / metrics["as_qa_no_num"] * 100, 1)
        state_F1 = round(
            2
            * (state_Precision / 100)
            * (state_Recall / 100)
            / ((state_Precision / 100) + (state_Recall / 100) + 0.0001)
            * 100,
            1,
        )
        number_Accuracy = round(metrics["an_qa_correct_score"] / metrics["an_qa_correct_num"] * 100, 1)
        number_Precision = round(metrics["an_qa_ans_no_score"] / metrics["an_qa_ans_no_num"] * 100, 1)
        number_Recall = round(metrics["an_qa_no_score"] / metrics["an_qa_no_num"] * 100, 1)
        number_F1 = round(
            2
            * (number_Precision / 100)
            * (number_Recall / 100)
            / ((number_Precision / 100) + (number_Recall / 100) + 0.0001)
            * 100,
            1,
        )
        action_Accuracy = round(metrics["aa_qa_correct_score"] / metrics["aa_qa_correct_num"] * 100, 1)
        action_Precision = round(metrics["aa_qa_ans_no_score"] / metrics["aa_qa_ans_no_num"] * 100, 1)
        action_Recall = round(metrics["aa_qa_no_score"] / metrics["aa_qa_no_num"] * 100, 1)
        action_F1 = round(
            2
            * (action_Precision / 100)
            * (action_Recall / 100)
            / ((action_Precision / 100) + (action_Recall / 100) + 0.0001)
            * 100,
            1,
        )
        results["Attribute"] = {
            "Accuracy": attr_Accuracy,
            "Precision": attr_Precision,
            "Recall": attr_Recall,
            "F1": attr_F1,
        }
        results["State"] = {
            "Accuracy": state_Accuracy,
            "Precision": state_Precision,
            "Recall": state_Recall,
            "F1": state_F1,
        }
        results["Number"] = {
            "Accuracy": number_Accuracy,
            "Precision": number_Precision,
            "Recall": number_Recall,
            "F1": number_F1,
        }
        results["Action"] = {
            "Accuracy": action_Accuracy,
            "Precision": action_Precision,
            "Recall": action_Recall,
            "F1": action_F1,
        }

    if dimension["dr"]:
        relation_Accuracy = round(metrics["asso_qa_correct_score"] / metrics["asso_qa_correct_num"] * 100, 1)
        relation_Precision = round(metrics["asso_qa_ans_no_score"] / metrics["asso_qa_ans_no_num"] * 100, 1)
        relation_Recall = round(metrics["asso_qa_no_score"] / metrics["asso_qa_no_num"] * 100, 1)
        relation_F1 = round(
            2
            * (relation_Precision / 100)
            * (relation_Recall / 100)
            / ((relation_Precision / 100) + (relation_Recall / 100) + 0.0001)
            * 100,
            1,
        )
        results["Relation"] = {
            "Accuracy": relation_Accuracy,
            "Precision": relation_Precision,
            "Recall": relation_Recall,
            "F1": relation_F1,
        }

    save_result(args.save_file, results)


def phase_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--inference_data", type=str, required=True)
    parser.add_argument("--similar_score_threshold", type=float, default=0.8)
    parser.add_argument("--metrics", type=str, default="data/metrics.txt")
    parser.add_argument("--safe_words", type=str, default="data/safe_words.txt")
    parser.add_argument("--annotation", type=str, default="data/annotations.json")
    parser.add_argument("--word_association", type=str, default="data/relation.json")
    parser.add_argument("--save_file", type=str, required=True, help="The path to save the evaluation result.")
    parser.add_argument(
        "--evaluation_type",
        choices=["a", "g", "d", "de", "da", "dr"],
        help="a: all tasks and dimensions    g: generative task    d: descriminative task    de, da, dr: existence, attribute, relation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args: Namespace = phase_args()
    main(args)
