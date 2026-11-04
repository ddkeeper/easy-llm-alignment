"""Assignment 5 · 4.2 GRPO 组件（对应 PDF 4.2.1 - 4.2.3 里的各个 Problem）。

只给函数定义、签名，以及 torch 张量的形状。实现和提示一律留空。

当前进度：4.2.1 的 tokenize_prompt_and_output、get_response_log_probs 已实现并通过测试；
4.2.3 的 compute_rollout_rewards 已给出签名。
"""
from __future__ import annotations

import math
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizer
from einops import rearrange, einsum

# =====================================================================
# 4.2.1  Problem (tokenize_prompt_and_output): Prompt and output tokenization
#        [1 point]
# =====================================================================
# 写法一
def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    """把 prompt 与 output 分别 tokenize 后拼接，并构造与 labels 对齐的 response_mask。

    Args:
        prompt_strs: list[str]，长度 batch_size。
        output_strs: list[str]，长度 batch_size。
        tokenizer: 预训练分词器。

    Returns:
        记 prompt_and_output_lens 为每条「prompt 长度 + output 长度」组成的列表，
        max_len = max(prompt_and_output_lens)，则：
        {
            "input_ids":     (batch_size, max_len - 1), int64
            "labels":        (batch_size, max_len - 1), int64
            "response_mask": (batch_size, max_len - 1), bool
        }
    """
    prompt_ids = [tokenizer.encode(s, add_special_tokens=False) for s in prompt_strs]
    response_ids = [tokenizer.encode(s, add_special_tokens=False) for s in output_strs]
    # 要以最长输入 ID 序列为基准，在其他短序列末尾进行填充，对齐长度以便能组装成一个 batch
    max_seq_len = max(len(ids1) + len(ids2) for ids1, ids2 in zip(prompt_ids, response_ids))

    complete_ids = []
    response_mask = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for ids1, ids2 in zip(prompt_ids, response_ids):
        pad_len = max_seq_len - len(ids1) - len(ids2)
        complete_id = ids1 + ids2 + [pad_id] * pad_len
        complete_ids.append(complete_id)
        mask = [0] * len(ids1) + [1] * len(ids2) + [0] * pad_len
        response_mask.append(mask)

    # 基于完整拼接结果构造输入与标签
    complete_ids = torch.tensor(complete_ids)
    input_ids, labels = complete_ids[:, :max_seq_len - 1], complete_ids[:, 1:]
    response_mask = torch.tensor(response_mask, dtype=bool)[:, 1:]

    # 构造返回字典
    res_dict = {}
    res_dict["input_ids"] = input_ids
    res_dict["labels"] = labels
    res_dict["response_mask"] = response_mask
    return res_dict

# 写法二（先拼接 prompt 和 output 串再分词，pdf 要求的是分别做分词，有出入）
def tokenize_prompt_and_output_v2(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    """把 prompt 与 output 分别 tokenize 后拼接，并构造与 labels 对齐的 response_mask。

    Args:
        prompt_strs: list[str]，长度 batch_size。
        output_strs: list[str]，长度 batch_size。
        tokenizer: 预训练分词器。

    Returns:
        {
            "input_ids":     (batch_size, max_len - 1), int64
            "labels":        (batch_size, max_len - 1), int64
            "response_mask": (batch_size, max_len - 1), bool
        }
    """
    batch_size = len(prompt_strs)
    # 1. 单独tokenize prompt，不padding，拿到每条prompt真实长度
    tok_prompt = tokenizer(prompt_strs, add_special_tokens=False, padding=False)
    prompt_lens = torch.tensor([len(ids) for ids in tok_prompt["input_ids"]], dtype=torch.long) # [B]

    # 2. 拼接字符串，整体tokenize，自动padding
    full_strs = [p + o for p, o in zip(prompt_strs, output_strs)]
    tok_full = tokenizer(
        full_strs,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    full_ids = tok_full["input_ids"]          # [B, T]
    attn_mask_full = tok_full["attention_mask"] # [B, T], bool/int, 1=有效token

    B, T = full_ids.shape

    # 3. 构造 prompt_mask: True 代表该位置属于prompt
    pos = torch.arange(T, device=full_ids.device) # [T]
    prompt_mask = pos.unsqueeze(0) < prompt_lens.unsqueeze(-1) # [1,T] < [B,1] → [B,T]

    # 4. response_mask：不是prompt 并且 不是padding
    response_mask = (~prompt_mask) & attn_mask_full.bool()

    # 5. shift：输入是[:-1], label是[1:]，mask同步切掉第一列
    input_ids = full_ids[:, :-1]
    labels = full_ids[:, 1:]
    response_mask = response_mask[:, 1:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

# =====================================================================
# 4.2.1  Problem (get_response_log_probs): Response log-probs (and entropy)
#        [1 point]
# =====================================================================
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """计算每个位置的条件对数概率，可选返回每个位置的熵。

    Args:
        model: HuggingFace 因果语言模型。
        input_ids: (batch_size, sequence_length)
        labels: (batch_size, sequence_length)
        return_token_entropy: 为 True 时额外返回 token_entropy。

    Returns:
        {
            "log_probs":     (batch_size, sequence_length),
            "token_entropy": (batch_size, sequence_length),  # 仅 return_token_entropy=True
        }
    """
    logits = model(input_ids).logits # b * s * v
    log_softmax = F.log_softmax(logits, dim=-1) # b * s * v
    log_probs = torch.gather(log_softmax, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    token_entropy = None
    if return_token_entropy:
        token_entropy = - (log_softmax.exp() * log_softmax).sum(dim=-1)
        return {"log_probs": log_probs, "token_entropy": token_entropy}

    return {"log_probs": log_probs}


# =====================================================================
# 4.2.3  Problem (compute_rollout_rewards): Computing the rewards of rollouts
#        [1 point]
# =====================================================================
def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """对每个 rollout response 计算原始 reward，并汇总可记录的统计量。

    Args:
        reward_fn: 打分函数，签名 (response, ground_truth) -> dict，
            返回的 dict 含 "reward"、"format_reward"、"answer_reward" 三个键。
        rollout_responses: list[str]，长度 rollout_batch_size。
        repeated_ground_truths: list[str]，长度同为 rollout_batch_size，
            每个 prompt 的标准答案按 group_size 重复。

    Returns:
        (raw_rewards, metadata)
        raw_rewards: (rollout_batch_size,)，未归一化的逐条 reward。
        metadata: 至少包含整个 rollout batch 上的
                  - 平均总 reward
                  - 平均 format reward
                  键名自定。
    """
    raw_rewards = []
    tot_reward = 0.0
    answer_reward = 0.0
    format_reward = 0.0
    n = len(rollout_responses)
    for rr, gt in zip(rollout_responses, repeated_ground_truths):
        out = reward_fn(rr, gt)
        raw_rewards.append(out["reward"])
        tot_reward += out["reward"]
        format_reward += out["format_reward"]
        answer_reward += out["answer_reward"]

    raw_rewards = torch.tensor(raw_rewards)
    return raw_rewards, {"mean_tot_reward": tot_reward / n, "mean_format_reward": format_reward / n, "mean_answer_reward": answer_reward / n}

# =====================================================================
# 统计量：pass@k（不是 PDF 要求的 problem，但训练/评估日志都要用）
# =====================================================================
def compute_pass_at_k(
    raw_rewards: torch.Tensor,
    group_size: int,
    ks: list[int] | None = None,
) -> dict[str, float]:
    """从一批 rollout 的原始 reward 估计 pass@k，k = 1..group_size。

    Args:
        raw_rewards: (n_prompts * group_size,)，同一 prompt 的 group_size 条必须连续。
        group_size: 每个 prompt 的 rollout 条数，也是能估的最大 k。
        ks: 要估的 k 列表，默认 1..group_size。

    Returns:
        {"pass@1": ..., "pass@2": ..., ...}，每个是 prompt 上的比例（float），
        另附两个 GRPO 诊断量：
          - "group_all_wrong"：一个 prompt 的 group_size 条全错的比例；
          - "group_all_correct"：全对的比例。
        这两类组的 advantage 恒为 0（减组均值后全零 / 组内无差异），贡献不了任何梯度。
        它们占比高就直接解释了 loss 不动、reward 不涨，是排查"训练卡住"的第一手依据。

    估计量说明（为什么要这么算）：
        每个 prompt 只有 group_size 条样本，想估 k>1 的 pass@k 时不能「取前 k 条」——
        那样只有 group_size/k 条样本在起作用，且与采样顺序强相关。
        无偏做法：从 n 条里无放回抽 k 条共 C(n,k) 种，其中一条都没答对的有 C(n-c,k) 种
        （c 为答对条数），于是
            pass@k = 1 - C(n-c, k) / C(n, k)
        k=1 时退化为 c/n，即普通 accuracy —— 所以 pass@1 就是「至少一条正确」的 k=1 特例。
        注意：这里假设 reward 是二值的（正确=1），对这两个 reward_fn 成立。
    """
    n = group_size
    rewards = rearrange(
        raw_rewards.detach().reshape(-1),
        "(n_prompt group_size) -> n_prompt group_size",
        group_size=group_size,
    )
    correct = (rewards >= 0.5).sum(dim=-1).tolist()  # (n_prompts,) 每个 prompt 答对几条

    results = {}
    for k in (ks if ks is not None else range(1, n + 1)):
        if k > n:
            continue
        # math.comb(a, b) 在 a < b 时返回 0，正好对应"该 prompt 的 c 条全抽中也凑不出 k 条对的"
        estimates = [1.0 - math.comb(n - c, k) / math.comb(n, k) for c in correct]
        results[f"pass@{k}"] = sum(estimates) / max(len(estimates), 1)

    if correct:
        results["group_all_wrong"] = sum(1 for c in correct if c == 0) / len(correct)
        results["group_all_correct"] = sum(1 for c in correct if c == n) / len(correct)
    return results

# =====================================================================
# 4.2.3  Problem (compute_group_normalized_rewards_grpo): Group normalization
#        [1 point]  —— 本阶段只需 baseline="mean" 且 advantage_normalizer="std"
# =====================================================================
def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """在组内做 baseline 相减与归一化，得到 advantages。

    Args:
        raw_rewards: (rollout_batch_size,) 未归一化 reward，
            rollout_batch_size = n_prompts_per_rollout_batch * group_size。
        group_size: 每个 prompt 的 response 条数。
        baseline: "mean" 时减去组内均值；"none" 表示不减（后续才实现）。
        advantage_eps: 防止除零的小常数。
        advantage_normalizer: "std" 时除以组内标准差；"none" 不归一化；
            "mean" 时除以组内均值（后两者后续才实现）。

    Returns:
        (advantages, metadata)
        advantages: (rollout_batch_size,)，组内归一化后的 reward。
        metadata: 想记录的统计量（eg. mean, std, max/min of rewards），键名自定。
    """
    raw_rewards = rearrange(raw_rewards, '(n_prompt group_size)-> n_prompt group_size', group_size=group_size) # n_p * g_s
    if baseline == "mean": 
        advantage = raw_rewards - raw_rewards.mean(dim=-1, keepdim=True) # n_p * g_s  n_p * 1 -> n_p * g_s
        if advantage_normalizer == "std":
            advantage /= (raw_rewards.std(dim=-1, keepdim=True) + advantage_eps)
        elif advantage_normalizer == "mean":
            advantage /= (raw_rewards.mean(dim=-1, keepdim=True) + advantage_eps)
        else:
            pass
    else:
        raise NameError

    advantage = rearrange(advantage, 'n_prompt group_size -> (n_prompt group_size)', group_size=group_size) # (n_p * g_s)

    metadata = {}
    metadata = {"mean": advantage.mean().item(), "std": advantage.std().item()}
    return (advantage, metadata)

# =====================================================================
# 4.2.3  Problem (compute_policy_gradient_loss_on_policy): On-policy policy gradient
#        [1 point]  —— 本阶段只需 importance_reweighting_method="none"
# =====================================================================
def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """逐 token 计算 policy gradient loss。

    注意：这个量不是通常意义上"训练中会下降"的 loss，
    它只是这样一个表达式——对它求导能得到 GRPO 的策略梯度。实现时返回该目标的负值。

    Args:
        raw_rewards_or_advantages: (batch_size,) 或 (batch_size, 1)，
            每条 rollout response 的标量 reward / advantage。
        policy_log_probs: (batch_size, sequence_length)，逐 token 的 log 概率。
        importance_reweighting_method: "none" 不做重要性重加权；
            其余取值本阶段不用。
        old_log_probs: (batch_size, sequence_length)，method != "none" 时必填。
        cliprange: 裁剪参数 ε，method 为 "grpo" / "gspo" 时必填。
        response_mask: (batch_size, sequence_length)，可选，
            用于只在 response token 上做序列级平均（GSPO 需要）。

    Returns:
        (per_token_policy_gradient_loss, metadata)
        per_token_policy_gradient_loss: (batch_size, sequence_length)，
            待后续在 batch 与 sequence 维上聚合。
        metadata: 底层 loss 调用产生的统计量，如 clip 比例相关项（dict[str, Tensor]）。
    """
    if  len(raw_rewards_or_advantages.shape) == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(-1)

    per_token_policy_gradient_loss = -raw_rewards_or_advantages.detach() * policy_log_probs # 在对目标取反后得到的 loss 上跑梯度下降，等价于在原目标上跑梯度上升
    if importance_reweighting_method == "none":
        pass

    metadata = {} # 后续补上
    return per_token_policy_gradient_loss, metadata 

# =====================================================================
# 4.2.3  Problem (aggregate_loss_across_microbatch_sequence):
#        Aggregate loss across tokens and sequences  [0.5 point]
#        —— 本阶段只需 loss_normalization="sequence"
# =====================================================================
def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """按 mask 与归一化策略，把逐 token loss 聚合成标量。

    Args:
        per_token_policy_gradient_loss: (batch_size, sequence_length)。
        mask: (batch_size, sequence_length)，哪些位置计入 loss。
        loss_normalization: "sequence" 先在每条序列内平均、再在序列间平均；
            "constant" 用固定常数归一化。
        normalization_constant: loss_normalization="constant" 时必填。

    Returns:
        loss: 标量张量，且之后必须能对它调用 backward。
    """
    if loss_normalization == "sequence":
        per_seq_policy_gradient_loss = (mask * per_token_policy_gradient_loss).sum(dim=-1) / mask.sum(dim=-1) # bs bs -> bs
        loss = per_seq_policy_gradient_loss.mean()
    elif loss_normalization == "constant":
        loss = (mask * per_token_policy_gradient_loss).sum() / normalization_constant
    else:
        raise NameError
     
    return loss

# =====================================================================
# 4.2.4  Problem (grpo_train_step_standard_on_policy): GRPO train step
#        [5 points]  —— 本阶段只需 baseline="mean"、advantage_normalizer="std"、
#        importance_reweighting_method="none"、loss_normalization="sequence"
# =====================================================================
def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """用一批 rollout 完成一次参数更新（含梯度累积）。

    Args:
        model / tokenizer / optimizer: 待训练模型、分词器、优化器。
        gradient_accumulation_steps: 每个 optimizer step 拆成几个 microbatch。
        max_grad_norm: 不为 None 时，在 optimizer.step() 前按该值裁剪梯度范数。
        reward_fn: (response, ground_truth) -> {"reward", "format_reward", "answer_reward"}。
        repeated_prompts: 长度 rollout_batch_size，每个 prompt 重复 group_size 次。
        rollout_responses: 长度 rollout_batch_size。
        repeated_ground_truths: 长度 rollout_batch_size，gt 同样重复 group_size 次。
        group_size: 每个 prompt 的 response 条数。
        其余超参与上面几个函数同名参数含义一致。

    Returns:
        (loss, metadata)
        loss: 标量，已按梯度累积做过缩放，供记录用。
        metadata: 至少包含裁剪前的梯度范数，以及其它想记录的统计量。
    """
    
    batch_size = len(repeated_prompts) # batch_size = num_prompts * group_size
    microbatch_size =  batch_size // gradient_accumulation_steps
    
    # 预处理：分词得到张量
    out_tok = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids, labels, response_mask = out_tok["input_ids"], out_tok["labels"], out_tok["response_mask"] # batch_size * seq_len
    seq_len = input_ids.shape[1]

    # tokenize 产出的张量在 CPU、reward 算出来的 advantages 也在 CPU，
    # 而模型可能在 GPU：不对齐 device 的话 forward 第一步 embedding 就报
    # "Expected all tensors to be on the same device"。
    device = next(model.parameters()).device
    input_ids, labels, response_mask = input_ids.to(device), labels.to(device), response_mask.to(device)

    # 计算出每一个样本的 advantages (常量值，不参与梯度计算)
    raw_rewards, metadata_reward = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, _ = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)
    advantages = advantages.to(device)

    tot_train_loss = torch.zeros((), device=device)
    tot_token_entropy = torch.empty(0, seq_len, device=device)

    # 梯度累积
    for i in range(0, batch_size, microbatch_size):
        inputs_microbatch = input_ids[i:i+microbatch_size] # micro_bs * seq_len
        labels_microbatch = labels[i:i+microbatch_size]
        response_mask_microbatch = response_mask[i:i+microbatch_size]
        advantages_microbatch = advantages[i:i+microbatch_size] 
        # old_log_probs 是整批 (rollout_batch_size, seq_len)，必须跟着 microbatch 一起切，
        # 否则 off-policy 变体里 (micro_bs, S) 会和 (bs, S) 形状对不上。
        old_log_probs_microbatch = None if old_log_probs is None else old_log_probs[i:i+microbatch_size]

        # 前向传播并根据计算 loss（实际上是梯度估计器反推出来的 loss 表达式)
        out_log_probs = get_response_log_probs(model, inputs_microbatch, labels_microbatch, return_token_entropy=True)
        policy_log_probs_microbatch,  token_entropy_microbatch = out_log_probs["log_probs"], out_log_probs["token_entropy"]
        per_token_loss, _ = compute_policy_gradient_loss(advantages_microbatch, policy_log_probs_microbatch, importance_reweighting_method, old_log_probs_microbatch, cliprange, response_mask_microbatch)
        microbatch_loss = aggregate_loss_across_microbatch(per_token_loss, response_mask_microbatch, loss_normalization, normalization_constant)
        loss = microbatch_loss * (len(inputs_microbatch) / batch_size) # 不同微批次的权重调节量可能不一样，取决于微批次实际大小

        loss.backward() # 反向传播
        
        # 累积 logging 量
        tot_train_loss += loss.detach()
        tot_token_entropy = torch.concat([tot_token_entropy, token_entropy_microbatch.detach()]) 

    max_norm = max_grad_norm if max_grad_norm is not None else math.inf # 根据传入参数决定是否执行梯度裁剪，发生在梯度累积完成后
    # clip_grad_norm_ 的返回值是「裁剪前」的全局范数：max_norm=inf 时不会改动任何参数，
    # 但仍然把真实范数吐出来，所以不裁剪时也能记录。
    grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)

    # 梯度下降
    optimizer.step() # Update weights once across entire batch.
    optimizer.zero_grad() # Zero gradients once across entire batch.

    metadata = {"loss": tot_train_loss, "grad_norm": grad_norm, "token_entropy": tot_token_entropy, "train_rewards": metadata_reward}
    metadata.update(compute_pass_at_k(raw_rewards, group_size))
    return tot_train_loss, metadata
