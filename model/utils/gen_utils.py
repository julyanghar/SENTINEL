from dataclasses import dataclass, field
from typing import Callable

import torch
from PIL import Image
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin
from transformers.generation.utils import GenerateDecoderOnlyOutput
from vllm import LLM, RequestOutput, SamplingParams
from vllm.sequence import Logprob


@dataclass
class GenOutput:
    outputs: list[str] | list[list[str]] = field(default_factory=list)
    generated_ids: torch.Tensor | None = None
    true_gen_length: list[int] | list[list[int]] = field(default_factory=list)
    log_probs: list[list[dict[int, Logprob]]] | list[list[list[dict[int, Logprob]]]] | None = None

    def maybe_change_to_list(self, force_list: bool) -> "GenOutput":
        def maybe_ls(ls: list, force_list: bool) -> list:
            return ls if force_list or not ls or len(ls) > 1 else ls[0]

        self.outputs = maybe_ls(self.outputs, force_list)
        if self.true_gen_length is not None:
            self.true_gen_length = maybe_ls(self.true_gen_length, force_list)
        if self.log_probs is not None:
            self.log_probs = maybe_ls(self.log_probs, force_list)

        return self


def init_vllm(
    model_name: str,
    model_dir: str,
    gpu_memory_util: float,
    dtype: torch.dtype,
    seed: int,
    model_context_len: int,
    enforce_eager: bool,
) -> LLM:
    return LLM(
        model=model_name,
        trust_remote_code=True,
        download_dir=model_dir,
        tensor_parallel_size=1,  # How many GPUs to use
        gpu_memory_utilization=gpu_memory_util,
        pipeline_parallel_size=1,
        dtype=dtype,
        seed=seed,
        max_model_len=model_context_len,  # Model context length
        enforce_eager=enforce_eager,  # True for faster init, False for faster batch generation
        swap_space=14,  # The size (GiB) of CPU memory per GPU to use as swap space
        task="generate",
        # DO NOT SET num_scheduler_steps for VLLMs!!!
    )


def get_generator(use_vllm: bool = True, debug: bool = False):
    """
    Get the generator model from the args of the global variables.
    """
    from model.auxiliary.global_vars import GVars

    args, model_dir, device, logger = (
        GVars.args,
        GVars.model_dir,
        GVars.main_device,
        GVars.logger,
    )
    gpu_util: float = 0.7

    if "llava" in args.model.lower():
        from model.generator.llava import LlavaModel

        return LlavaModel(
            use_vllm=use_vllm,
            debug=debug,
            version=args.model_version,
            model_size=args.model_size,
            model_dir=model_dir,
            gpu_util=gpu_util,
            device=device,
            logger=logger,
        )
    elif "qwen2.5vl" in args.model.lower() or "qwen2_5_vl" in args.model.lower():
        from model.generator.qwen2_5_vl import Qwen2_5_VLModel

        return Qwen2_5_VLModel(
            use_vllm=use_vllm,
            debug=debug,
            model_size=args.model_size,
            model_dir=model_dir,
            gpu_util=gpu_util,
            device=device,
            logger=logger,
        )
    elif "qwen2vl" in args.model.lower() or "qwen2_vl" in args.model.lower():
        from model.generator.qwen2_vl import Qwen2VLModel

        return Qwen2VLModel(
            use_vllm=use_vllm,
            debug=debug,
            model_size=args.model_size,
            model_dir=model_dir,
            gpu_util=gpu_util,
            device=device,
            logger=logger,
        )
    else:
        raise ValueError(f"Unknown generator model: {args.model}")


def get_stop_words_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    eos_token_id = tokenizer.eos_token_id
    stop_words_ids = [tokenizer.convert_tokens_to_ids(w) for w in [".", "!", "?"]]
    stop_words_ids.append(eos_token_id)
    return stop_words_ids


def _get_vllm_inputs(prompts: list[str], images: list[Image.Image]) -> list[dict]:
    return [
        {
            "prompt": p,
            "multi_modal_data": {"image": i},
        }
        for p, i in zip(prompts, images)
    ]


def gen_vllm(
    llm: LLM,
    images: list[Image.Image],
    prompts: list[str],
    do_sample: bool,
    temperature: float | int,
    max_tokens: int,
    stop_token_ids: list[int] | None = None,
    n: int = 1,
    return_log_probs: bool = True,
    seed: int | None = None,
    strip_slash_n: bool = True,
) -> GenOutput:
    inputs: list[dict] = _get_vllm_inputs(prompts, images)
    n: int = n if do_sample else 1

    params = SamplingParams(
        n=n,
        temperature=temperature if do_sample else 0,  # 0 means greedy sampling
        max_tokens=max_tokens,
        seed=seed,
        stop_token_ids=stop_token_ids,
        logprobs=0 if return_log_probs else None,
        include_stop_str_in_output=True,  # 包含停止词，如 '.'、'!'、'?' 等
    )
    completions: list[RequestOutput] = llm.generate(inputs, params, use_tqdm=False)

    outputs: list[str] | list[list[str]] = []
    log_probs: list[dict] | list[list[dict]] = [] if return_log_probs else None
    lengths: int | list[int] = []

    for c in completions:
        if len(c.outputs) == 1:
            if strip_slash_n is True:
                outputs.append(c.outputs[0].text.rstrip("\n").strip(" "))
            else:
                outputs.append(c.outputs[0].text.strip(" "))
            lengths.append(len(c.outputs[0].token_ids))
            if return_log_probs:
                log_probs.append(c.outputs[0].logprobs)
        else:
            if strip_slash_n is True:
                outputs.append([o.text.rstrip("\n").strip(" ") for o in c.outputs])
            else:
                outputs.append([o.text.strip(" ") for o in c.outputs])
            lengths.append([len(o.token_ids) for o in c.outputs])
            if return_log_probs:
                log_probs.append([o.logprobs for o in c.outputs])

    return GenOutput(outputs=outputs, true_gen_length=lengths, log_probs=log_probs)


def u_a_to_prompts(
    users: list[str],
    assistants: list[str],
    processor: ProcessorMixin | None = None,
    prompt_modifier: Callable[[str], str] | None = None,
) -> list[str]:
    """
    Convert user and assistant strings to prompts.

    Args:
        users (list[str]): User strings.
        assistants (list[str]): Assistant strings.
        processor (ProcessorMixin | None, optional): Processor. Defaults to None.
        prompt_modifier (Callable[[str], str] | None, optional): Prompt modifier. Defaults to None.
    """
    prompts: list[str] = []
    for u, a in zip(users, assistants):
        if processor is not None:
            conversation: list[dict[str]] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": u},
                    ],
                },
            ]
            if a:
                conversation.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": a}],
                    }
                )
            add_gen_prompt: bool = len(conversation) == 1
            prompt: str = processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=add_gen_prompt,
            )
            if a and prompt_modifier is not None:
                prompt = prompt_modifier(prompt)
        else:
            prompt = u + " " + a if a else u
        prompts.append(prompt)
    return prompts


def _gen_hf(
    model: PreTrainedModel,
    processor: ProcessorMixin,
    device: torch.device | str,
    torch_dtype: torch.dtype,
    images: list[Image.Image],
    prompts: list[str],
    do_sample: bool,
    temperature: float,
    max_tokens: int,
    eos_token_id: list[int] | None = None,
) -> tuple[GenerateDecoderOnlyOutput, dict[str, torch.Tensor]]:
    with torch.inference_mode():
        encoded_inputs: dict[str, torch.Tensor] = processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            return_token_type_ids=False,
            padding=True,
        ).to(device, torch_dtype)

        out: GenerateDecoderOnlyOutput = model.generate(
            **encoded_inputs,
            max_new_tokens=max_tokens,
            num_beams=1,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            eos_token_id=eos_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            return_legacy_cache=True,
        )
        return out, encoded_inputs


def _get_b_gen_length(generated_ids: torch.Tensor, pad_token_id: int) -> list[int]:
    """
    通过 generated_ids 以及 pad_token_id 获得生成的每个序列的实际长度。

    Args:
        generated_ids (torch.Tensor): 生成的 ID，形状为 [batch_size, generated_length]。
        pad_token_id (int): PAD token 的 ID。

    Returns:
        list[int]: 每个序列的实际长度。
    """
    b_gen_length: list[int] = []
    for ids in generated_ids:
        # 寻找PAD token的位置
        pad_idx = (ids == pad_token_id).nonzero(as_tuple=True)[0]

        if len(pad_idx) > 0:  # 如果存在 PAD token，取第一个出现的位置（因为要排除PAD token本身）
            length = pad_idx[0].item()
        else:  # 如果不存在PAD token，使用整个序列的长度
            length = len(ids)
        b_gen_length.append(length)
    return b_gen_length


def _truncate_gen_ids(encoded_inputs: dict, generated_ids: torch.Tensor) -> torch.Tensor:
    """
    截断 generated_ids 到其有效长度并保留原来的格式。

    Args:
        encoded_inputs: 编码后的输入数据，包含 input_ids。
        generated_ids: 生成的 ID 列表。

    Returns:
        list: 截断后的 generated_ids。
    """
    return generated_ids[:, encoded_inputs.input_ids.size(1) :]


def gen_hf(
    model: PreTrainedModel,
    processor: ProcessorMixin,
    device: torch.device | str,
    torch_dtype: torch.dtype,
    images: list[Image.Image],
    prompts: list[str],
    do_sample: bool,
    temperature: float,
    max_tokens: int,
    n: int = 1,
    eos_token_id: list[int] | None = None,
    truncate_gen_ids: bool = True,
    strip_slash_n: bool = True,
) -> GenOutput:
    from .utils import repeat_n, split_n

    images, prompts = repeat_n(n, images, prompts)
    tokenizer: PreTrainedTokenizerBase = processor.tokenizer

    out, encoded_inputs = _gen_hf(
        model,
        processor,
        device,
        torch_dtype,
        images,
        prompts,
        do_sample,
        temperature,
        max_tokens,
        eos_token_id=eos_token_id,
    )

    generated_ids = out.sequences  # Tensor: (batch_size, sequence_length)
    if truncate_gen_ids:
        generated_ids = _truncate_gen_ids(encoded_inputs, generated_ids)

    true_gen_length: list[int] = _get_b_gen_length(generated_ids, tokenizer.pad_token_id)
    del out, encoded_inputs

    outputs: list[str] = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    if strip_slash_n is True:
        outputs = [d.rstrip("\n").strip(" ") for d in outputs]
    else:
        outputs = [d.strip(" ") for d in outputs]

    outputs = split_n(n, outputs)

    return GenOutput(
        outputs=outputs,
        generated_ids=generated_ids,
        true_gen_length=true_gen_length,
    )


if __name__ == "__main__":
    pass
