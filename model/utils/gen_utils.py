"""
SENTINEL 生成器工具模块
=======================

本模块提供 MLLM 生成相关的工具函数和类，包括：
- GenOutput: 生成输出的数据类
- 生成器初始化和调用函数
- vLLM 和 HuggingFace 两种后端支持

核心功能:
    - init_vllm(): 初始化 vLLM 引擎
    - get_generator(): 根据配置获取对应的生成器
    - gen_vllm(): 使用 vLLM 进行生成
    - gen_hf(): 使用 HuggingFace 进行生成
"""

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
    """
    生成器输出的统一封装类
    
    无论使用 vLLM 还是 HuggingFace 后端，都返回此格式的输出。
    
    Attributes:
        outputs: 生成的文本
            - 单样本单候选: str
            - 单样本多候选: list[str]
            - 多样本: list[list[str]]
        generated_ids: 生成的 token IDs（仅 HF 后端）
        true_gen_length: 实际生成的 token 数量
        log_probs: 每个 token 的对数概率（用于分析）
    
    Example:
        >>> output = generator.gen(images, prompts, n=10)
        >>> print(output.outputs[0])  # 第一个样本的第一个候选
        >>> print(output.true_gen_length[0])  # 第一个样本的生成长度
    """
    
    outputs: list[str] | list[list[str]] = field(default_factory=list)
    """生成的文本内容"""
    
    generated_ids: torch.Tensor | None = None
    """生成的 token IDs，形状为 [batch_size, seq_len]"""
    
    true_gen_length: list[int] | list[list[int]] = field(default_factory=list)
    """每个样本的实际生成长度（不包括 padding）"""
    
    log_probs: list[list[dict[int, Logprob]]] | list[list[list[dict[int, Logprob]]]] | None = None
    """每个 token 的对数概率"""

    def maybe_change_to_list(self, force_list: bool) -> "GenOutput":
        """
        根据 force_list 参数决定是否保持列表格式
        
        当只有一个样本时，默认会解包成单个元素，
        设置 force_list=True 可以强制保持列表格式。
        
        Args:
            force_list: 是否强制返回列表
        
        Returns:
            self (支持链式调用)
        """
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
    """
    初始化 vLLM 引擎
    
    vLLM 是一个高性能的 LLM 推理引擎，通过 PagedAttention 等技术
    实现高效的批处理和内存管理。
    
    Args:
        model_name: HuggingFace 模型名称（如 "Qwen/Qwen2-VL-7B-Instruct"）
        model_dir: 模型下载目录
        gpu_memory_util: GPU 显存使用比例（0-1）
        dtype: 数据类型（torch.float16 或 torch.bfloat16）
        seed: 随机种子
        model_context_len: 模型上下文长度
        enforce_eager: 是否使用 eager 模式
            - True: 启动快但批处理慢（适合调试）
            - False: 启动慢但批处理快（适合生产）
    
    Returns:
        vLLM LLM 实例
    
    Note:
        - tensor_parallel_size=1 表示使用单 GPU
        - swap_space=14 设置 14GB CPU 内存用于 swap
        - 不要设置 num_scheduler_steps，否则会导致问题
    """
    return LLM(
        model=model_name,
        trust_remote_code=True,
        download_dir=model_dir,
        tensor_parallel_size=1,  # 使用的 GPU 数量
        gpu_memory_utilization=gpu_memory_util,
        pipeline_parallel_size=1,
        dtype=dtype,
        seed=seed,
        max_model_len=model_context_len,  # 模型上下文长度
        enforce_eager=enforce_eager,  # True: 快速初始化，False: 快速批处理
        swap_space=14,  # CPU swap 空间大小 (GiB)
        task="generate",
        # 注意: 不要设置 num_scheduler_steps！
    )


def get_generator(use_vllm: bool = True, debug: bool = False):
    """
    根据全局配置获取对应的 MLLM 生成器
    
    工厂函数，根据 GVars.args.model 的值自动选择并初始化相应的生成器。
    
    Args:
        use_vllm: 是否使用 vLLM 后端
            - True: 使用 vLLM，批处理速度快
            - False: 使用 HuggingFace，兼容性好
        debug: 是否开启调试模式
            - True: 使用 eager 模式，便于调试
            - False: 使用编译模式，速度快
    
    Returns:
        生成器实例，支持的类型:
        - LlavaModel: LLaVA v1.5/v1.6
        - Qwen2VLModel: Qwen2-VL
        - Qwen2_5_VLModel: Qwen2.5-VL
    
    Raises:
        ValueError: 不支持的模型类型
    
    Example:
        >>> generator = get_generator(use_vllm=True, debug=False)
        >>> output = generator.gen(images, prompts)
    
    Note:
        - 默认 GPU 利用率为 70%，避免 OOM
        - 模型类型从 GVars.args.model 自动推断
    """
    from model.auxiliary.global_vars import GVars

    args, model_dir, device, logger = (
        GVars.args,
        GVars.model_dir,
        GVars.main_device,
        GVars.logger,
    )
    gpu_util: float = 0.7  # GPU 显存使用率，设置较低以避免 OOM

    # 根据模型名称选择对应的生成器类
    if "llava" in args.model.lower():
        from model.generator.llava import LlavaModel

        return LlavaModel(
            use_vllm=use_vllm,
            debug=debug,
            version=args.model_version,  # "1.5" 或 "1.6"
            model_size=args.model_size,   # "7b" 或 "13b"
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
            model_size=args.model_size,  # "7B"
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
            model_size=args.model_size,  # "2B" 或 "7B"
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
