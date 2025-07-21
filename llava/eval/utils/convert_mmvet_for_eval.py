import json
import os
from argparse import ArgumentParser, Namespace

parser = ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--dst", type=str)
args: Namespace = parser.parse_args()

cur_result = {}

for line in open(args.src):
    data = json.loads(line)
    qid = data["question_id"] if "question_id" in data else data["image_id"]
    cur_result[f"v1_{qid}"] = data["text"] if "text" in data else data["caption"]

os.makedirs(os.path.dirname(args.dst), exist_ok=True)

with open(args.dst, "w") as f:
    json.dump(cur_result, f, indent=2)
