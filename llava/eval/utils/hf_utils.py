import torch
from peft import PeftModel  # noqa
from transformers import (
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    PreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)


def init_hf_model(args) -> tuple[PreTrainedModel, AutoProcessor]:
    model_name: str = args.model_name
    model_dir: str = args.model_dir
    print(f"Loading model name {model_name} from dir {model_dir}")

    if "llava" in model_name.lower():
        model_type = LlavaNextForConditionalGeneration
        processor_type = LlavaNextProcessor
    elif "qwen2_5" in model_name.lower() or "qwen2.5" in model_name.lower() or "qwen25" in model_name.lower():
        model_type = Qwen2_5_VLForConditionalGeneration
        processor_type = AutoProcessor
    elif "qwen" in model_name.lower():
        model_type = Qwen2VLForConditionalGeneration
        processor_type = AutoProcessor

    model: PreTrainedModel = model_type.from_pretrained(
        model_name,
        cache_dir=model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    ).eval()

    if args.lora_name is not None and len(args.lora_name) > 10:
        lora_name: str = args.lora_name
        print(f"Loading LoRA model from {lora_name}")
        model = PeftModel.from_pretrained(
            model,
            args.lora_name,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
    else:
        print("No LoRA model specified, using base model.")

    processor: AutoProcessor = processor_type.from_pretrained(
        model_name,
        cache_dir=model_dir,
        padding_side="left",
        use_fast=True,
    )
    return model, processor


def truncate_gen_ids(encoded_inputs: dict, generated_ids: torch.Tensor) -> torch.Tensor:
    """
    截断 generated_ids 到其有效长度并保留原来的格式。

    Args:
        encoded_inputs: 编码后的输入数据，包含 input_ids。
        generated_ids: 生成的 ID 列表。

    Returns:
        list: 截断后的 generated_ids。
    """
    return generated_ids[:, encoded_inputs.input_ids.size(1) :]
