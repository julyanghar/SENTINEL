# Mask DPO Trainer for LLaVA
# 
# 针对 mask 数据集的训练器。与原版 LlavaDPOTrainer 的主要区别：
# 1. mask 数据没有真正的 reject 样本（y_lose=None）
# 2. 使用 SFT loss 代替 DPO loss 进行训练
# 3. 训练目标：让模型在看到遮挡图像时生成正确的描述（不提及被遮挡物体）
#
# 训练原理：
# - 标准 DPO 需要 (chosen, rejected) 对来计算偏好 loss
# - mask 数据只有 chosen（正确描述），没有 rejected
# - 因此我们使用 SFT（监督微调）方式训练，直接最大化 P(chosen|masked_image)

from typing import Any, Dict, Literal, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from llava.constants import IGNORE_INDEX
from train.models.base_trainer import BaseDPOTrainer


def concatenate_chosen_rejected(
    chosen_input_ids: torch.Tensor,
    chosen_labels: torch.Tensor,
    chosen_attention_mask: torch.Tensor,
    reject_input_ids: torch.Tensor,
    reject_labels: torch.Tensor,
    reject_attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    将 chosen 和 rejected 序列拼接成一个 batch
    
    用于 DPO 训练，将 chosen 和 rejected 序列拼接在一起，
    方便一次前向传播计算两者的 log probabilities。
    
    Args:
        chosen_input_ids: chosen 序列的 input ids, shape: (batch_size, chosen_len)
        chosen_labels: chosen 序列的 labels, shape: (batch_size, chosen_len)
        chosen_attention_mask: chosen 序列的 attention mask, shape: (batch_size, chosen_len)
        reject_input_ids: rejected 序列的 input ids, shape: (batch_size, reject_len)
        reject_labels: rejected 序列的 labels, shape: (batch_size, reject_len)
        reject_attention_mask: rejected 序列的 attention mask, shape: (batch_size, reject_len)
    
    Returns:
        batch_input_ids: 拼接后的 input ids, shape: (batch_size * 2, max_len)
        batch_labels: 拼接后的 labels, shape: (batch_size * 2, max_len)
        batch_attention_mask: 拼接后的 attention mask, shape: (batch_size * 2, max_len)
        batch_size: 原始 batch size
    
    Note:
        - 前 batch_size 行是 chosen，后 batch_size 行是 rejected
        - 较短的序列会用 0（input_ids）、IGNORE_INDEX（labels）、False（mask）填充
    """
    batch_size: int = chosen_input_ids.shape[0]
    dtype, device = chosen_input_ids.dtype, chosen_input_ids.device
    max_dim = max(chosen_input_ids.shape[1], reject_input_ids.shape[1])

    # 创建拼接后的 tensors
    batch_input_ids = torch.zeros((batch_size * 2, max_dim), dtype=dtype, device=device)
    batch_labels = torch.ones((batch_size * 2, max_dim), dtype=dtype, device=device) * IGNORE_INDEX
    batch_attention_mask = torch.zeros((batch_size * 2, max_dim), device=device).to(torch.bool)

    # 填充 chosen（前半部分）
    batch_input_ids[:batch_size, :chosen_input_ids.shape[1]] = chosen_input_ids
    batch_labels[:batch_size, :chosen_labels.shape[1]] = chosen_labels
    batch_attention_mask[:batch_size, :chosen_attention_mask.shape[1]] = chosen_attention_mask

    # 填充 rejected（后半部分）
    batch_input_ids[batch_size:, :reject_input_ids.shape[1]] = reject_input_ids
    batch_labels[batch_size:, :reject_labels.shape[1]] = reject_labels
    batch_attention_mask[batch_size:, :reject_attention_mask.shape[1]] = reject_attention_mask

    return batch_input_ids, batch_labels, batch_attention_mask, batch_size


class MaskLlavaDPOTrainer(BaseDPOTrainer):
    """
    Mask DPO Trainer with DPO Loss
    
    如果需要使用 DPO loss 进行训练，可以使用这个 trainer。
    在 mask 数据中，reject 被设置为 chosen 的副本，
    这样 DPO loss 会产生一个基线 loss（约 0.693），
    模型会学习到减少这个 loss 的方向。
    
    注意：这种方式的训练效果可能不如纯 SFT。
    """

    def dpo_concatenated_forward(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        DPO 拼接前向传播
        
        将 chosen 和 rejected 拼接在一起进行前向传播，
        然后分别计算它们的 log probabilities。
        
        对于 mask 数据，chosen 和 rejected 是相同的（因为 y_lose=None）。
        """
        images: torch.Tensor = inputs["images"]
        masked_images: torch.Tensor = inputs["masked_images"]

        # 拼接 chosen 和 rejected 序列
        # batch_input_ids, batch_labels, batch_attention_mask, batch_size = concatenate_chosen_rejected(
        #     chosen_input_ids=inputs["chosen_input_ids"],
        #     chosen_labels=inputs["chosen_labels"],
        #     chosen_attention_mask=inputs["chosen_attention_mask"],
        #     reject_input_ids=inputs["reject_input_ids"],
        #     reject_labels=inputs["reject_labels"],
        #     reject_attention_mask=inputs["reject_attention_mask"],
        # )

        batch_input_ids, batch_labels, batch_attention_mask, batch_size = concatenate_chosen_rejected(
            chosen_input_ids=inputs["chosen_input_ids"],
            chosen_labels=inputs["chosen_labels"],
            chosen_attention_mask=inputs["chosen_attention_mask"],
            reject_input_ids=inputs["chosen_input_ids"],
            reject_labels=inputs["chosen_labels"],
            reject_attention_mask=inputs["chosen_attention_mask"],
        )

        # 准备多模态输入
        (
            batch_input_ids,
            batch_position_ids,
            batch_attention_mask,
            batch_past_key_values,
            batch_inputs_embeds,
            batch_labels,
        ) = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=batch_input_ids,
            position_ids=None,
            attention_mask=batch_attention_mask,
            past_key_values=None,
            labels=batch_labels,
            images=torch.cat([images, masked_images], dim=0),
        )

        # 前向传播计算 logits
        all_logits = model.forward(
            inputs_embeds=batch_inputs_embeds,
            labels=None,
            attention_mask=batch_attention_mask,
        ).logits.to(torch.float32)

        # 计算 log probabilities
        all_logps = self._get_batch_logps(all_logits, batch_labels)

        chosen_logps = all_logps[:batch_size]
        # rejected_logps = all_logps[batch_size:]
        masked_chosen_logps = all_logps[batch_size:]

        # 计算平均 logits（用于 metrics）
        loss_mask = batch_labels != IGNORE_INDEX
        logits = [all_logits[i][loss_mask[i]] for i in range(loss_mask.shape[0])]
        chosen_logits = logits[:batch_size]
        # rejected_logits = logits[batch_size:]
        masked_chosen_logits = logits[batch_size:]
        chosen_logits = [logit.detach().cpu().mean() for logit in chosen_logits]
        # rejected_logits = [logit.detach().cpu().mean() for logit in masked_chosen_logits]
        masked_chosen_logits = [logit.detach().cpu().mean() for logit in masked_chosen_logits]
        chosen_logits = sum(chosen_logits) / batch_size
        # rejected_logits = sum(rejected_logits) / batch_size
        masked_chosen_logits = sum(masked_chosen_logits) / batch_size

        return (chosen_logps, masked_chosen_logps, chosen_logits, masked_chosen_logits)

    def compute_ref_log_probs(self, inputs: dict[str, torch.LongTensor]):
        """计算参考模型的 log probabilities"""
        with torch.no_grad():
            if self.ref_model is None:
                with self.null_ref_context():
                    chosen_logps, rejected_logps, _, _ = self.dpo_concatenated_forward(self.model, inputs)
            else:
                chosen_logps, rejected_logps, _, _ = self.dpo_concatenated_forward(self.ref_model, inputs)
        return chosen_logps, rejected_logps

    def get_batch_loss_metrics(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """计算 batch 的 DPO loss 和 metrics"""
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.dpo_concatenated_forward(model, inputs)
        reference_chosen_logps, reference_rejected_logps = self.compute_ref_log_probs(inputs)

        (
            dpo_losses,
            chosen_rewards,
            rejected_rewards,
        ) = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        loss = dpo_losses.mean()

        # Log metrics
        metrics = {}
        metrics["rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics["rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics["rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().cpu().mean()
        metrics["rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics["policy_logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics["policy_logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics["referece_logps/chosen"] = reference_chosen_logps.detach().cpu().mean()
        metrics["referece_logps/rejected"] = reference_rejected_logps.detach().cpu().mean()
        metrics["logits/chosen"] = policy_chosen_logits.detach().cpu().mean()
        metrics["logits/rejected"] = policy_rejected_logits.detach().cpu().mean()

        return loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """计算总 loss"""
        loss, metrics = self.get_batch_loss_metrics(model, inputs, "train")

        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

