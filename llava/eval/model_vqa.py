import argparse
import json
import os
import sys
from argparse import Namespace

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.eval.utils.utils import init_model, read_json
from llava.mm_utils import process_images, tokenizer_image_token
from llava.utils import (
    get_chunk,
    get_id_from_sample,
    get_image_path_from_sample,
    get_question_from_sample,
    load_image,
    setup_seeds,
)


def eval_model(args):
    setup_seeds(args.seed)

    tokenizer, model, image_processor = init_model(args)

    questions = read_json(args.question_file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    for line in tqdm(questions):
        idx: str = get_id_from_sample(line)
        question: str = get_question_from_sample(line)
        image_path: str | None = get_image_path_from_sample(line)

        ori_question = question

        if image_path:
            if hasattr(args, "image_folder") and args.image_folder:
                image_path = os.path.join(args.image_folder, image_path)
            image: Image.Image = load_image(image_path)

            image_tensor = process_images([image], image_processor, model.config)[0]
            images: torch.Tensor = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
            if model.config.mm_use_im_start_end:
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
            else:
                question = DEFAULT_IMAGE_TOKEN + "\n" + question
        else:
            images = None
            image_sizes = None

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        with open(answers_file, "a") as ans_file:
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "image_path": image_path,
                        "prompt": ori_question,
                        "text": outputs,
                        "answer_id": shortuuid.uuid(),
                        "category": line["category"] if "category" in line else None,
                        "metadata": {},
                    }
                )
                + "\n"
            )


def eval_hf_model(args: Namespace) -> None:
    from llava.eval.utils.hf_utils import init_hf_model, truncate_gen_ids

    setup_seeds(args.seed)

    model, processor = init_hf_model(args)

    questions = read_json(args.question_file)
    questions: list[dict] = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    for line in tqdm(questions):
        idx: str = get_id_from_sample(line)
        question: str = get_question_from_sample(line)
        image_path: str | None = get_image_path_from_sample(line)
        # question += "\n" + "Answer with 'yes' or 'no' directly."

        if image_path:
            if hasattr(args, "image_folder") and args.image_folder:
                image_path = os.path.join(args.image_folder, image_path)
            image = Image.open(image_path)

            conversation: list[dict[str]] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
            ]
        else:
            image = None
            conversation: list[dict[str]] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                    ],
                },
            ]

        prompt: str = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        with torch.inference_mode():
            encoded_inputs: dict[str, torch.Tensor] = processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                return_token_type_ids=False,
                padding=True,
            ).to(model.device, model.dtype)

            output_ids = model.generate(
                **encoded_inputs,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p if args.temperature > 0 else None,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        output_ids = truncate_gen_ids(encoded_inputs, output_ids)
        output: str = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        with open(answers_file, "a") as ans_file:
            ans_file.write(
                json.dumps(
                    {
                        "image_id": idx,
                        "image_path": image_path,
                        "question": question,
                        "caption": output,
                        "answer_id": shortuuid.uuid(),
                        "category": line["category"] if "category" in line else None,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--lora-name", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _name = args.model_name.lower()
    if "llava" in _name and ("1.5" in _name or "1_5" in _name):
        if args.lora_name:
            args.model_path = args.lora_name
            args.model_base = args.model_name
        else:
            args.model_path = args.model_name
            args.model_base = None
        eval_model(args)
    else:
        eval_hf_model(args)
