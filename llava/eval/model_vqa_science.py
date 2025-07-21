import argparse
import json
import os
import sys

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from llava.eval.utils.utils import init_model, read_json
from llava.utils import get_chunk, setup_seeds


def eval_model(args):
    from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    setup_seeds(args.seed)
    tokenizer, model, image_processor = init_model(args)

    questions = read_json(args.question_file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    for line in tqdm(questions):
        idx = line["id"]
        question = line["conversations"][0]
        qs = question["value"].replace("<image>", "").strip()
        cur_prompt = qs

        if "image" in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([image], image_processor, model.config)[0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
            if getattr(model.config, "mm_use_im_start_end", False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
            cur_prompt = "<image>" + "\n" + cur_prompt
        else:
            images = None
            image_sizes = None

        if args.single_pred_prompt:
            qs = qs + "\n" + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + "\n" + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
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
                max_new_tokens=1024,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        with open(answers_file, "a") as ans_file:
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "prompt": cur_prompt,
                        "text": outputs,
                        "answer_id": shortuuid.uuid(),
                        "model_id": args.model_path,
                        "metadata": {},
                    }
                )
                + "\n"
            )


def eval_hf_model(args):
    from llava.eval.utils.hf_utils import init_hf_model, truncate_gen_ids

    setup_seeds(args.seed)

    model, processor = init_hf_model(args)

    questions = read_json(args.question_file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    for line in tqdm(questions):
        idx = line["id"]
        question = line["conversations"][0]
        qs: str = question["value"].replace("<image>", "").strip()
        cur_prompt: str = qs

        if args.single_pred_prompt:
            qs: str = qs + "\n" + "Answer with the option's letter from the given choices directly."
            cur_prompt: str = cur_prompt + "\n" + "Answer with the option's letter from the given choices directly."

        if "image" in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            conversation: list[dict[str]] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": qs},
                    ],
                },
            ]
        else:
            image = None
            conversation: list[dict[str]] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": qs},
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
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        output_ids = truncate_gen_ids(encoded_inputs, output_ids)
        outputs: str = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        outputs = outputs.strip()

        with open(answers_file, "a") as ans_file:
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "prompt": cur_prompt.replace("Context:", "<image>\nContext:"),
                        "text": outputs,
                        "answer_id": shortuuid.uuid(),
                        "model_id": args.model_name,
                        "metadata": {},
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
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--single-pred-prompt", action="store_true")
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
