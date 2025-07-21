# ruff: noqa: F405, F403
import json
from argparse import ArgumentParser, Namespace

from prettytable import PrettyTable

from .utils import (
    evaluate,
    check_same,
    assign_correctness,
    get_eval_all,
    get_eval_pair_all,
    get_eval_pair_easy,
    get_eval_pair_hard,
    get_eval_fig,
    yes_ratio_stats,
)

MODEL_OUTPUT_KEY = "model_prediction"
MODEL_CORRECTNESS_KEY = "gpt4v_output_gpt_check"


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Evaluate your method")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file",
    )
    parser.add_argument(
        "--temp_file",
        type=str,
        default="./temp.json",
        help="Path to the temporary JSON file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./review.json",
        help="Path to the output JSON file",
    )
    parser.add_argument("--load_json", action="store_true", default=True, help="Flag to load existing JSON data")

    return parser.parse_args()


def get_str(table: PrettyTable) -> str:
    return table.get_string().replace("+", "").replace("-", "").replace(" ", "")


def main(input_file, temp_file, output_file: str, load_json):
    vd_temp_file = temp_file.replace(".json", "_vd.json")  # Visual Dependent
    vs_temp_file = temp_file.replace(".json", "_vs.json")  # Visual Supplement

    data_vd, data_vs = [], []
    with open(input_file, "r", encoding="utf-8") as f:
        data: list[dict] = json.load(f)
    assert len(data) == 1129

    data_vd = [item for item in data if item["category"] == "VD"]
    data_vs = [item for item in data if item["category"] == "VS"]
    assert len(data_vd) + len(data_vs) == len(data)
    del data

    data_vd = evaluate(data_vd, MODEL_OUTPUT_KEY, MODEL_CORRECTNESS_KEY, load_json, vd_temp_file)
    data_vd = check_same(data_vd, MODEL_OUTPUT_KEY, MODEL_CORRECTNESS_KEY, vd_temp_file)
    data_vs = evaluate(data_vs, MODEL_OUTPUT_KEY, MODEL_CORRECTNESS_KEY, load_json, vs_temp_file)
    data_vs = check_same(data_vs, MODEL_OUTPUT_KEY, MODEL_CORRECTNESS_KEY, vs_temp_file)

    data_vd = assign_correctness(data_vd, MODEL_CORRECTNESS_KEY)
    data_vs = assign_correctness(data_vs, MODEL_CORRECTNESS_KEY)
    data = data_vd + data_vs

    all_data = get_eval_all(data, MODEL_CORRECTNESS_KEY)
    all_vd = get_eval_all(data_vd, MODEL_CORRECTNESS_KEY)
    all_vs = get_eval_all(data_vs, MODEL_CORRECTNESS_KEY)

    table1 = [
        ["per question", "Total"],
        ["VD", round(100 * all_vd["correct"] / all_vd["total"], 4)],
        ["VS", round(100 * all_vs["correct"] / all_vs["total"], 4)],
        ["Overall", round(100 * all_data["correct"] / all_data["total"], 4)],
    ]
    tab1 = PrettyTable(table1[0])
    tab1.add_rows(table1[1:])

    q_acc_gpt = round(100 * all_data["correct"] / all_data["total"], 4)

    all_data = get_eval_pair_all(data, MODEL_CORRECTNESS_KEY)
    easy = get_eval_pair_easy(data)
    hard = get_eval_pair_hard(data)
    all_vd = get_eval_pair_all(data_vd, MODEL_CORRECTNESS_KEY)
    easy_vd = get_eval_pair_easy(data_vd)
    hard_vd = get_eval_pair_hard(data_vd)
    all_vs = get_eval_pair_all(data_vs, MODEL_CORRECTNESS_KEY)
    easy_vs = get_eval_pair_easy(data_vs)
    hard_vs = get_eval_pair_hard(data_vs)
    # question pair level
    table3 = [
        ["per question pair", "Easy", "Hard", "Total"],
        [
            "VD",
            round(100 * easy_vd["correct"] / easy_vd["total"], 4),
            round(100 * hard_vd["correct"] / hard_vd["total"], 4),
            round(100 * all_vd["correct"] / all_vd["total"], 4),
        ],
        [
            "VS",
            round(100 * easy_vs["correct"] / easy_vs["total"], 4),
            round(100 * hard_vs["correct"] / hard_vs["total"], 4),
            round(100 * all_vs["correct"] / all_vs["total"], 4),
        ],
        [
            "Overall",
            round(100 * easy["correct"] / easy["total"], 4),
            round(100 * hard["correct"] / hard["total"], 4),
            round(100 * all_data["correct"] / all_data["total"], 4),
        ],
    ]
    tab3 = PrettyTable(table3[0])
    tab3.add_rows(table3[1:])

    fig_all = get_eval_fig(data)
    fig_vd = get_eval_fig(data_vd)
    fig_vs = get_eval_fig(data_vs)

    # image level
    table2 = [
        ["per figure", "Correct", "Wrong", "Score"],
        [
            "VD",
            round(100 * fig_vd["correct"] / fig_vd["total"], 4),
            round(100 * fig_vd["inconsistent"] / fig_vd["total"], 4)
            + round(100 * fig_vd["wrong"] / fig_vd["total"], 4),
            round(fig_vd["score"], 4),
        ],
        [
            "VS",
            round(100 * fig_vs["correct"] / fig_vs["total"], 4),
            round(100 * fig_vs["inconsistent"] / fig_vs["total"], 4)
            + round(100 * fig_vs["wrong"] / fig_vs["total"], 4),
            round(fig_vs["score"], 4),
        ],
        [
            "Overall",
            round(100 * fig_all["correct"] / fig_all["total"], 4),
            round(100 * fig_all["inconsistent"] / fig_all["total"], 4)
            + round(100 * fig_all["wrong"] / fig_all["total"], 4),
            round(fig_all["score"], 4),
        ],
    ]
    tab2 = PrettyTable(table2[0])
    tab2.add_rows(table2[1:])

    pair_acc_gpt = round(100 * all_data["correct"] / all_data["total"], 4)
    figure_acc_gpt = round(100 * fig_all["correct"] / fig_all["total"], 4)
    easy_acc_gpt = round(100 * easy["correct"] / easy["total"], 4)
    hard_acc_gpt = round(100 * hard["correct"] / hard["total"], 4)

    table = [
        [
            "",
            "Acc per question pair (qAcc)",
            "Acc per figure (fAcc)",
            "Acc per easy question (easy aAcc)",
            "Acc per hard question (hard aAcc)",
            "Acc per question (aAcc)",
        ],
        ["GPT Eval", pair_acc_gpt, figure_acc_gpt, easy_acc_gpt, hard_acc_gpt, q_acc_gpt],
    ]
    leaderboard = PrettyTable(table[0])
    leaderboard.add_rows(table[1:])
    print(leaderboard)

    stats = yes_ratio_stats(data)

    table = [
        [
            "",
            "Yes/No Bias (Pct Diff)",
            "Yes/No Bias (FP Ratio)",
            "Consistency Test (correct)",
            "Consistency Test (inconsistent)",
            "Consistency Test (wrong)",
            "LH",
            "VI",
            "Mixed",
        ],
        [
            "GPT Eval",
            stats["diff"],
            stats["fp"],
            round(100 * fig_all["correct"] / fig_all["total"], 4),
            round(100 * fig_all["inconsistent"] / fig_all["total"], 4),
            round(100 * fig_all["wrong"] / fig_all["total"], 4),
            round(100 * all_data["LH_cg"] / (all_data["LH_cg"] + all_data["VI_cg"] + all_data["Mix_cg"]), 4),
            round(100 * all_data["VI_cg"] / (all_data["LH_cg"] + all_data["VI_cg"] + all_data["Mix_cg"]), 4),
            round(100 * all_data["Mix_cg"] / (all_data["LH_cg"] + all_data["VI_cg"] + all_data["Mix_cg"]), 4),
        ],
    ]
    test = PrettyTable(table[0])
    test.add_rows(table[1:])

    results = {
        "question_pair_level": get_str(tab3),
        "image_level": get_str(tab2),
        "leaderboard_stats": get_str(leaderboard),
        "yes_no_bias": get_str(test),
    }

    with open(output_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args: Namespace = parse_args()
    main(args.input_file, args.temp_file, args.output_file, args.load_json)
