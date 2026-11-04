"""Problem (grpo_experiments_standard_on_policy): 4.3 GRPO 训练脚本脚手架。

工程脚手架都已写好：argparse、数据载入、prompt 模板（一次性读入内存）、
rollout 后端抽象（vllm / hf 两种）、指标与 rollout 落盘、wandb 可选接入。
核心逻辑留空，需要你自己实现：
  - sample_rollouts()  把"一个 prompt 采 group_size 条"摊平成训练所需的三个平行 list
  - evaluate()         验证集指标（total / format / answer reward、平均响应长度、pass@k）
  - train()            主循环（权重同步时机、每步更新、周期评估、rollout 落盘）

两种后端：
  --backend vllm   Linux + GPU，真正用于实验。会拉起 vLLM 进程并用 NCCL 同步权重，
                   因此训练进程和 vLLM 必须在同一台机器上（vllm_utils 里 pkill / killpg
                   是 POSIX 专用，Windows 上调用 VLLMServer.stop() 会 AttributeError）。
  --backend hf     本地 smoke test 用。直接用 HF 模型 .generate() 采样，不涉及 vLLM，
                   Windows 可跑。注意：它验证的是"流程能走通"，不是算法效果——
                   HF 与 vLLM 的采样实现不同，数值不可迁移。

本地跑通（Windows / RTX 3060，6 GB）建议：小模型 + 小 batch + 2 步
    python grpo_train.py --backend hf --dry-run \\
        --model-id HuggingFaceTB/SmolLM2-135M-Instruct --attn-implementation eager --device cpu
服务器上正式跑：
    python grpo_train.py --backend vllm --gpu 1 --launch-server

服务器首次跑前记得：export HF_ENDPOINT=https://hf-mirror.com
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch

# reference/ 放算法实现；公共模块（奖励函数、vLLM 封装、checkpoint）、提示词、数据、产物都在本章根目录
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from grpo import compute_pass_at_k, grpo_train_step
from prompting_baselines import load_gsm8k
from drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn

PROMPT_FILES = {
    "question_only": "question_only.prompt",
    "r1_zero": "r1_zero.prompt",
    "r1_zero_three_shot": "r1_zero_three_shot_gsm8k.prompt",
}
PROMPT_DIR = _ROOT / "prompts"
# ----------------------------------------------------------------------


# =====================================================================
# 命令行
# =====================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。全部超参都有默认值，直接 `python grpo_train.py` 也能跑。

    默认值即 PDF 4.3 的建议超参，直接写在下面各参数上。
    """
    p = argparse.ArgumentParser(description="GRPO (standard, on-policy) on GSM8K")

    # 模型与数据
    p.add_argument("--model-id", default="allenai/OLMo-2-0425-1B")
    p.add_argument("--train-file", type=Path, default=_ROOT / "data" / "gsm8k" / "train.jsonl")
    p.add_argument("--val-file", type=Path, default=_ROOT / "data" / "gsm8k" / "test.jsonl")
    p.add_argument("--prompt-mode", choices=sorted(PROMPT_FILES), default="r1_zero")
    p.add_argument("--n-train-examples", type=int, default=6400)
    p.add_argument("--n-val-examples", type=int, default=1024)

    # 训练超参：batch 数的是"响应条数"而非 prompt 数，256 = 32 prompt × group_size 8
    # microbatch = n_prompts_per_batch * group_size // gradient_accumulation_steps，
    # 所以 32 累积步 → 每个 microbatch 8 条响应。实测（3090 24 GB，max_tokens 512，
    # 1.48B 参数）microbatch 4 时 peak_alloc 已达 21.5 GB，8 条必然 OOM，
    # 故默认取 64（microbatch 4）。步数与切分不改变梯度总和，只影响显存和速度。
    p.add_argument("--num-rollout-steps", type=int, default=200)
    p.add_argument("--n-prompts-per-batch", type=int, default=32)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--gradient-accumulation-steps", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    # 采样超参
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=512)

    # 日志与评估
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--log-rollouts-every", type=int, default=40)
    p.add_argument("--preview-groups", type=int, default=4,
                   help="rollouts.md 里每组展开几个 prompt（0 = 全展开）；全量始终在 rollouts.jsonl")
    p.add_argument("--eval-batch-prompts", type=int, default=128,
                   help="评估时一次送多少 prompt；1024 条全塞一次会撑爆内存")
    p.add_argument("--output-dir", type=Path, default=_ROOT / "outputs" / "grpo",
                   help="结果根目录；实际落盘在 <output-dir>/<run-name>/")
    p.add_argument("--experiment", default="base",
                   help="实验类型，作为 run 名的前缀，如 base / lr_sweep / bs_sweep / prompt_ablation")
    p.add_argument("--run-name", default=None,
                   help="留空则按「实验类型-数据集-关键超参-seed」自动生成，"
                        "如 base-gsm8k-lr1e-5-bs256-g8-s0；指定则完全覆盖")
    p.add_argument("--wandb-project", default="cs336-a5-grpo",
                   help="wandb 项目名；传空字符串关掉")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online",
                   help="服务器无外网时用 offline，事后 wandb sync")
    p.add_argument("--print-every", type=int, default=10,
                   help="控制台打印间隔步数；0 表示完全不打印（正式跑只留 wandb）")
    p.add_argument("--save-every", type=int, default=0,
                   help="每 N 步存一次 checkpoint；0 表示不存（1B 模型一份约 2 GB）")

    # 后端
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--launch-server", action="store_true", help="本机拉起 vLLM 进程（仅 Linux）")
    p.add_argument("--vllm-host", default="127.0.0.1")
    p.add_argument("--vllm-port", type=int, default=8000)
    p.add_argument("--gpu", type=int, default=1, help="vLLM 只占这张卡；policy 用另一张")
    p.add_argument("--vllm-gpu-util", type=float, default=0.9,
                   help="vLLM 的 --gpu-memory-utilization，占**整卡**显存的比例（不是剩余空闲）。"
                        "共用显卡时必须调低，否则 vLLM 启动时校验空闲显存会直接失败")
    p.add_argument("--rollout-request-batch-size", type=int, default=None,
                   help="一次 HTTP 塞几个 prompt；默认等于 n_prompts_per_batch（一次发完）。"
                        "不确定 vLLM 的 index 语义时设成 1，退化为逐 prompt 请求")
    p.add_argument("--policy-device", default="cuda:0")
    p.add_argument("--attn-implementation", default="sdpa",
                   help="sdpa 是通用默认；装了 flash-attn 且序列很长时才值得换 flash_attention_2")
    p.add_argument("--device", default=None, help="仅 hf 后端有效，如 cpu / cuda:0")

    # 调试
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="极小规模跑 2 步，验证流程")
    p.add_argument("--verify-order", action="store_true",
                   help="只发一次多 prompt 请求，打印 choices 的 index，确认返回顺序语义")

    args = p.parse_args(argv)
    if args.dry_run:
        args.num_rollout_steps = min(args.num_rollout_steps, 2)
        args.n_train_examples = min(args.n_train_examples, 8)
        args.n_val_examples = min(args.n_val_examples, 4)
        args.n_prompts_per_batch = min(args.n_prompts_per_batch, 2)
        args.group_size = min(args.group_size, 2)
        args.max_tokens = min(args.max_tokens, 64)
        args.eval_every = 1
        args.log_rollouts_every = 1
        args.eval_batch_prompts = min(args.eval_batch_prompts, 4)
        args.gradient_accumulation_steps = min(args.gradient_accumulation_steps, 1)
    if args.rollout_request_batch_size is None:
        args.rollout_request_batch_size = args.n_prompts_per_batch
    return args


# =====================================================================
# 数据 / prompt / 采样参数
# =====================================================================
def prepare_data(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    """读 GSM8K 并按需要截断。返回 (train, val)，每条是 {"question", "gold"}。"""
    train = load_gsm8k(args.train_file)[: args.n_train_examples]
    val = load_gsm8k(args.val_file)[: args.n_val_examples]
    return train, val


def make_prompt_fn(mode: str) -> Callable[[str], str]:
    """把模板一次性读进内存，返回 question -> prompt 的闭包。

    不要每次调用都 read_text：6400 条 × 200 步会把 IO 变成瓶颈。
    用 str.format 而不是 str.replace —— r1_zero 模板正文里含有 "question" 字样。
    """
    template = (PROMPT_DIR / PROMPT_FILES[mode]).read_text(encoding="utf-8")
    return lambda question: template.format(question=question)


def make_sampling_params(args: argparse.Namespace) -> dict:
    """rollout 用的采样参数。n 必须等于 group_size（一次请求采一组）。"""
    params = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "n": args.group_size,
    }
    # question_only 不要求输出 </answer>，加 stop 会把正常回答截断
    if args.prompt_mode != "question_only":
        params["stop"] = ["</answer>"]
        params["include_stop_str_in_output"] = True
    return params


def make_reward_fn(mode: str) -> Callable[[str, str], dict[str, float]]:
    return question_only_reward_fn if mode == "question_only" else r1_zero_reward_fn


# =====================================================================
# rollout 后端
# =====================================================================
@dataclass
class Completion:
    """与 vllm_utils.VLLMCompletion 同形，HF 后端也复用它，避免上层区分后端。"""
    text: str
    token_ids: list[int]
    finish_reason: str | None


class VLLMBackend:
    """vLLM 后端（Linux + GPU）。

    - vllm_utils 只在实例化时 import，保证 Windows 上仅用 hf 后端时不会踩到 vllm。
    - 批量请求依赖 vLLM 的 `index=len(choices)`（跨 batch 的扁平下标）：
      generate_completions 内部 `sorted(choices, key=index)` 后即得 prompt-major 顺序。
      旧版 vLLM 用的是组内下标（每个 prompt 从 0 重数），那时排序会交错 ——
      用 `--verify-order` 确认一次；不确定时把 --rollout-request-batch-size 设成 1。
    """

    def __init__(self, model_id: str, host: str, port: int, gpu: int,
                 seed: int, launch_server: bool, request_batch_size: int,
                 gpu_memory_utilization: float = 0.9) -> None:
        from vllm_utils import VLLMServer

        self.request_batch_size = request_batch_size
        self.server = VLLMServer(
            model_id=model_id,
            host=host,
            port=port,
            gpu=gpu,
            seed=seed,
            launch_server=launch_server,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def start(self) -> None:
        self.server.start()

    def stop(self) -> None:
        self.server.stop()

    def init_weight_sync(self, policy_device: str):
        return self.server.init_weight_sync(policy_device)

    def sync_weights(self, policy: torch.nn.Module) -> None:
        self.server.sync_policy_weights(policy)

    def generate_groups(self, prompt_texts: list[str],
                        sampling_params: dict) -> list[list[Completion]]:
        flat = self.server.generate_completions(
            prompt_texts, sampling_params, batch_size=self.request_batch_size
        )
        n = sampling_params["n"]
        if len(flat) != len(prompt_texts) * n:
            raise RuntimeError(f"返回 {len(flat)} 条，期望 {len(prompt_texts) * n} 条")
        completions = [Completion(c.text, c.token_ids, c.finish_reason) for c in flat]
        return [completions[i: i + n] for i in range(0, len(completions), n)]


class HFBackend:
    """HF 后端（本地 smoke test）。直接用 policy 自己采样，不涉及 vLLM 与权重同步。"""

    def __init__(self, model, tokenizer, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def init_weight_sync(self, policy_device: str):
        return None  # 本地没有独立推理进程，权重天然一致

    def sync_weights(self, policy: torch.nn.Module) -> None:
        pass  # 同一个模型对象，无需同步

    def generate_groups(self, prompt_texts: list[str],
                        sampling_params: dict) -> list[list[Completion]]:
        return [self._generate_one(text, sampling_params) for text in prompt_texts]

    @torch.no_grad()
    def _generate_one(self, prompt_text: str, sampling_params: dict) -> list[Completion]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id  # 注意别写成 `pad_id or eos`：pad 的 id 可能是 0
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=sampling_params["temperature"],
            max_new_tokens=sampling_params["max_tokens"],
            num_return_sequences=sampling_params["n"],
            pad_token_id=pad_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        stop_strings = sampling_params.get("stop") or []
        completions = []
        for row in outputs:
            new_ids = row[prompt_len:].tolist()
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            # HF 的 generate 没有 stop 参数，手动截断，否则本地指标会跟 vLLM 不可比
            truncated = False
            for stop in stop_strings:
                position = text.find(stop)
                if position != -1:
                    text = text[: position + len(stop)]
                    truncated = True
                    break
            if truncated:
                # 截断后 text 与 new_ids 不再对应，重新 tokenize 保证长度口径与文本一致
                new_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            # 只有没被 stop 截断、且用完 max_new_tokens 才算 length
            hit_length = not truncated and len(new_ids) >= sampling_params["max_tokens"]
            completions.append(
                Completion(text, new_ids, "length" if hit_length else "stop")
            )
        return completions


def build_backend(args: argparse.Namespace, policy, tokenizer) -> VLLMBackend | HFBackend:
    if args.backend == "hf":
        return HFBackend(policy, tokenizer, args.device or args.policy_device)
    return VLLMBackend(
        model_id=args.model_id,
        host=args.vllm_host,
        port=args.vllm_port,
        gpu=args.gpu,
        seed=args.seed,
        launch_server=args.launch_server,
        request_batch_size=args.rollout_request_batch_size,
        gpu_memory_utilization=args.vllm_gpu_util,
    )


# =====================================================================
# 采样：一组 rollout 摊平成训练用的平行 list
# =====================================================================
@dataclass
class RolloutBatch:
    """三个 list 长度都是 len(prompts) * group_size，且按 (prompt, j) 顺序排列。"""
    prompts: list[str]
    responses: list[str]
    ground_truths: list[str]
    finish_reasons: list[str]
    # 响应 token 数（= len(completion.token_ids)）。评估要按 token 算平均长度：
    # 字符数会被长公式/格式差异带偏，token 数才对应真实的解码开销。
    token_counts: list[int]

def sample_rollouts(
    backend: VLLMBackend | HFBackend,
    examples: list[dict],
    prompt_fn: Callable[[str], str],
    sampling_params: dict,
) -> RolloutBatch:
    """对每个 example 采 group_size 条响应，摊平成训练所需的平行 list。

    为什么要摊平：grpo_train_step 的 repeated_prompts / rollout_responses /
    repeated_ground_truths 都是按"响应"计的，长度 = n_prompts * group_size，
    且同一 prompt 的 group_size 条必须**连续**排在一起，后续
    compute_group_normalized_rewards 才能按 group_size reshape 回组。
    """
    prompt_texts = [prompt_fn(example["question"]) for example in examples]
    groups = backend.generate_groups(prompt_texts, sampling_params)

    batch = RolloutBatch(
        prompts=[], responses=[], ground_truths=[], finish_reasons=[], token_counts=[]
    )
    for example, prompt_text, completions in zip(examples, prompt_texts, groups):
        if len(completions) != sampling_params["n"]:
            raise RuntimeError(
                f"backend 返回 {len(completions)} 条，期望 {sampling_params['n']} 条"
            )
        for completion in completions:
            batch.prompts.append(prompt_text)
            batch.responses.append(completion.text)
            batch.ground_truths.append(example["gold"])
            batch.finish_reasons.append(completion.finish_reason)
            batch.token_counts.append(len(completion.token_ids))
    return batch


# =====================================================================
# 评估与训练循环（核心逻辑，留给你实现）
# =====================================================================
def evaluate(
    backend: VLLMBackend | HFBackend,
    examples: list[dict],
    prompt_fn: Callable[[str], str],
    reward_fn: Callable[[str, str], dict[str, float]],
    sampling_params: dict,
    eval_batch_prompts: int = 128,
    prefix: str = "val",
) -> dict[str, float]:
    """在整个验证集上跑一遍，返回要记录的指标（每个都是 float）。

    指标口径（键名带 prefix，train/eval 在 wandb 里天然分组）：
      - reward / format_reward / answer_reward：所有 rollout 上的算术平均；
      - response_tokens：**token** 数均值。不用字符数——GSM8K 的 CoT 里公式和数字
        长短差异很大，字符数会被排版带偏，token 数才对应真实的解码开销；
      - truncation_rate：finish_reason == "length" 的比例。这条比长度均值更能说明问题：
        如果它接近 1，说明模型根本没机会写完 </answer>，此时 reward 低是截断导致的；
      - pass@k，k = 1..group_size：至少一条正确的 prompt 占比（无偏估计，见 compute_pass_at_k）。
        pass@1 就是普通 accuracy，pass@group_size 是"这一组里只要有一条对就算对"。

    分块采样而不是一次塞完：1024 个 prompt × group_size 条文本 + 打分都在内存里，
    一次做完容易把显存/内存打满；顺带每块都会做一次权重无关的采样，出错时进展可见。
    """
    group_size = sampling_params["n"]
    n_responses = 0
    n_prompts = 0
    sums = {
        "reward": 0.0,
        "format_reward": 0.0,
        "answer_reward": 0.0,
        "response_tokens": 0.0,
        "truncated": 0.0,
    }
    pass_sums: dict[str, float] = {}

    for start in range(0, len(examples), eval_batch_prompts):
        chunk = examples[start: start + eval_batch_prompts]
        batch = sample_rollouts(backend, chunk, prompt_fn, sampling_params)

        # 直接自己打分，而不是调 compute_rollout_rewards：后者只给汇总，
        # 而 pass@k 需要逐条 reward 才能按 group_size 分组。
        scores = [
            reward_fn(response, gold)
            for response, gold in zip(batch.responses, batch.ground_truths)
        ]
        rewards = torch.tensor([s["reward"] for s in scores], dtype=torch.float32)

        n_responses += len(scores)
        n_prompts += len(chunk)
        sums["reward"] += float(rewards.sum())
        sums["format_reward"] += sum(s["format_reward"] for s in scores)
        sums["answer_reward"] += sum(s["answer_reward"] for s in scores)
        sums["response_tokens"] += sum(batch.token_counts)
        sums["truncated"] += sum(1 for reason in batch.finish_reasons if reason == "length")

        # 每块的 pass@k 是该块 prompt 上的均值，按块内 prompt 数加权即得全集均值
        for key, value in compute_pass_at_k(rewards, group_size).items():
            pass_sums[key] = pass_sums.get(key, 0.0) + value * len(chunk)

    metrics = {
        f"{prefix}/reward": sums["reward"] / n_responses,
        f"{prefix}/format_reward": sums["format_reward"] / n_responses,
        f"{prefix}/answer_reward": sums["answer_reward"] / n_responses,
        f"{prefix}/response_tokens": sums["response_tokens"] / n_responses,
        f"{prefix}/truncation_rate": sums["truncated"] / n_responses,
        f"{prefix}/n_prompts": float(n_prompts),
    }
    for key, value in pass_sums.items():
        metrics[f"{prefix}/{key}"] = value / n_prompts
    return metrics


def _epoch_batches(
    examples: list[dict], n_prompts_per_batch: int, seed: int
) -> Iterator[list[dict]]:
    """按 epoch 轮转地产出 mini-batch：每个 epoch 把数据打乱一次，用完再打乱。

    不要用 random.sample(range(n_prompts_per_batch), k=len(train_examples)) ——
    那是从"下标 0..31"里抽 6400 个，直接 ValueError（样本数大于总体）。
    正确顺序是 population=全部样本，k=n_prompts_per_batch。
    用独立的 Random(seed) 而不是全局 random，避免影响别的随机源。
    """
    rng = random.Random(seed)
    order = list(range(len(examples)))
    while True:
        rng.shuffle(order)
        # 尾部不足一个 batch 就丢掉：不扔的话最后一个 batch 会比别人小，
        # gradient_accumulation 的整除假设会被破坏。
        for start in range(0, len(order) - n_prompts_per_batch + 1, n_prompts_per_batch):
            yield [examples[i] for i in order[start: start + n_prompts_per_batch]]


def train(
    args: argparse.Namespace,
    policy,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    backend: VLLMBackend | HFBackend,
    reward_fn: Callable[[str, str], dict[str, float]],
    train_examples: list[dict],
    val_examples: list[dict],
    prompt_fn: Callable[[str], str],
    sampling_params: dict,
    run_dir: Path,
    wandb_run=None,
) -> None:
    """主循环：num_rollout_steps 次「采样 → 更新 → 记录」，外加周期性验证。"""
    batch_size = args.n_prompts_per_batch * args.group_size
    if batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError(
            f"gradient_accumulation_steps={args.gradient_accumulation_steps} "
            f"不能整除 batch_size={batch_size}（= n_prompts_per_batch × group_size）"
        )

    def run_eval() -> dict[str, float]:
        t_eval = time.time()
        metrics = evaluate(
            backend, val_examples, prompt_fn, reward_fn, sampling_params,
            eval_batch_prompts=args.eval_batch_prompts, prefix="val",
        )
        metrics["time/eval"] = time.time() - t_eval
        return metrics

    # step 0 基线：必须放在任何一次参数更新之前，否则"reward 有没有涨"没有参照点。
    backend.sync_weights(policy)
    policy.eval()
    if val_examples:
        log_metrics(run_dir, 0, run_eval(), wandb_run, args.print_every)

    batches = _epoch_batches(train_examples, args.n_prompts_per_batch, args.seed)
    for step in range(1, args.num_rollout_steps + 1):
        t_start = time.time()
        examples = next(batches)

        # 顺序不能反：先把训练进程的权重推给 vLLM，再让它采样。
        # 反了的话这一步的 rollout 是上一步权重产出的，off-policy 味道且难查。
        policy.eval()
        backend.sync_weights(policy)
        t_sync = time.time()
        rollout = sample_rollouts(backend, examples, prompt_fn, sampling_params)
        t_rollout = time.time()

        policy.train()
        loss, metadata = grpo_train_step(
            model=policy,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            reward_fn=reward_fn,
            repeated_prompts=rollout.prompts,
            rollout_responses=rollout.responses,
            repeated_ground_truths=rollout.ground_truths,
            group_size=args.group_size,
        )
        t_train = time.time()

        n_rollouts = len(rollout.responses)
        metrics = {
            "train/loss": metadata["loss"],
            "train/grad_norm": metadata["grad_norm"],
            "train/token_entropy": metadata["token_entropy"],
            "train/reward": metadata["train_rewards"]["mean_tot_reward"],
            "train/format_reward": metadata["train_rewards"]["mean_format_reward"],
            "train/answer_reward": metadata["train_rewards"]["mean_answer_reward"],
            "train/response_tokens": sum(rollout.token_counts) / n_rollouts,
            "train/truncation_rate": sum(
                1 for reason in rollout.finish_reasons if reason == "length"
            ) / n_rollouts,
            # 耗时拆开记：sync 慢说明权重传输是瓶颈，rollout 慢是解码/打分，
            # train 慢才是 GPU 算力。调吞吐时先看这个比例再决定优化哪块。
            "time/sync": t_sync - t_start,
            "time/rollout": t_rollout - t_sync,
            "time/train": t_train - t_rollout,
            # 与别的用户共用显卡时，OOM 只会崩在第 1 次 optimizer.step()（AdamW 的
            # 两个 moment 那时才分配），所以把峰值显存打出来，跌到接近卡容量就该提前收手。
            "mem/peak_alloc_gb": (
                torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            ),
            "mem/peak_reserved_gb": (
                torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
            ),
        }
        # pass@k 与 group_all_correct / group_all_wrong 由 grpo_train_step 的 metadata 带出：
        # 后两个是 GRPO 特有的诊断量——全对或全错的组 advantage 恒为 0，贡献不了梯度，
        # 这类组占比高就直接解释了 loss 不动。
        for key, value in metadata.items():
            if key.startswith("pass@") or key.startswith("group_"):
                metrics[f"train/{key}"] = value

        # 验证指标并进同一行：wandb 的 history 以 step 为行，同一步两次 log 会被合并，
        # 一次 log 更可控，jsonl 里也正好是"一行一步"，方便直接 pandas 读。
        if args.eval_every and (step % args.eval_every == 0 or step == args.num_rollout_steps):
            metrics.update(run_eval())

        log_metrics(run_dir, step, metrics, wandb_run, args.print_every)

        if args.log_rollouts_every and step % args.log_rollouts_every == 0:
            log_rollouts(
                run_dir, step, rollout, reward_fn, wandb_run,
                n_show=2 if args.print_every else 0,
                group_size=args.group_size,
                preview_groups=args.preview_groups,
            )
        if args.save_every and step % args.save_every == 0:
            save_checkpoint(policy, tokenizer, run_dir / "checkpoints" / f"step{step}")

# =====================================================================
# 落盘 / 日志
# =====================================================================
def _fmt_num(value: float) -> str:
    """把超参值压成适合进目录名的短串：1e-05 → 1e-5，2.4e-03 → 2.4e-3。"""
    mantissa, exponent = f"{value:.6e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def make_run_name(args: argparse.Namespace) -> str:
    """按「实验类型 + 数据集 + 关键超参 + seed」自动拼 run 名。

    形如 base-gsm8k-lr1e-5-bs256-g8-s0；prompt 不是默认的 r1_zero 时额外插一段
    （如 prompt_ablation-gsm8k-r1_zero_three_shot-lr1e-5-bs256-g8-s0），免得 ablation 之间撞名。

    为什么不用时间戳：同一配置重跑会得到两个无法区分的目录；反过来，用超参命名时
    看目录名就知道什么配置，排序时同实验的 run 天然聚在一起，也方便脚本按前缀过滤。
    """
    dataset = Path(args.train_file).parent.name or Path(args.train_file).stem
    parts = [args.experiment, dataset]
    if args.prompt_mode != "r1_zero":
        parts.append(args.prompt_mode)
    parts += [
        f"lr{_fmt_num(args.learning_rate)}",
        # bs 按 PDF 的口径数"响应条数"（= n_prompts_per_batch × group_size），不是 prompt 数；
        # g 单独标出来，才知道一组里有几条
        f"bs{args.n_prompts_per_batch * args.group_size}",
        f"g{args.group_size}",
        f"s{args.seed}",
    ]
    return "-".join(parts)


def prepare_run_dir(args: argparse.Namespace) -> Path:
    """建结果目录 <output-dir>/<run-name>/，并写入 args.json。

    目录名就是 run 名（不带时间戳），wandb 上的 run 名与本地目录一一对应，
    看到曲线就能定位到盘上的日志。args.json 里除超参外还记了 argv / hostname / torch 版本：
    复现某个 run 时不用猜它当时在哪台机器、用什么命令跑起来的。
    """
    name = args.run_name or make_run_name(args)
    args.run_name = name  # 回写，保证 wandb 名、args.json、目录名三处一致
    run_dir = args.output_dir / name
    if (run_dir / "metrics.jsonl").exists():
        # 同名目录里已有日志，说明这个配置跑过了。直接追加会把两次的曲线混在一起，
        # 而这恰恰是 4.3 拿曲线下结论时最怕的静默错误，所以宁可报错。
        raise FileExistsError(
            f"{run_dir} 已存在且含 metrics.jsonl。同名 run 通常是重复提交："
            f"删掉该目录重跑，或用 --run-name 指定新名字。"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv,
        "hostname": socket.gethostname(),
        "platform": sys.platform,
        "torch": torch.__version__,
        "args": {key: str(value) for key, value in vars(args).items()},
    }
    (run_dir / "args.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir


def _as_float(value) -> float:
    """把 metadata 里的各种标量统一成 float。

    逐 token 的 token_entropy 是 (batch, seq) 的大张量，直接 float() 会报错，
    这里取均值——记录它的目的是看"输出分布是否越来越自信"，均值就够。
    """
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean())
    return float(value)


def _flatten_into(out: dict[str, float], metrics: dict, prefix: str = "") -> None:
    """把嵌套 dict 拍平成 "a/b/c" 形式的键，wandb 会按 "/" 自动分组画图。"""
    for key, value in metrics.items():
        if isinstance(value, dict):
            _flatten_into(out, value, f"{prefix}{key}/")
        else:
            out[f"{prefix}{key}"] = _as_float(value)


def log_metrics(
    run_dir: Path,
    step: int,
    metrics: dict,
    wandb_run=None,
    print_every: int = 10,
) -> None:
    """一行一个 step 追加到 metrics.jsonl，同时推 wandb（若已 init）。

    注意 wandb 的 step 必须通过 log(..., step=) 传，**不要**把它放进 dict：
    两边都出现会报错；即使不报错，dict 里的 step 也会被当成普通指标。
    """
    flat = {}
    _flatten_into(flat, metrics)

    with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"step": step, **flat}, ensure_ascii=False) + "\n")

    if wandb_run is not None:
        wandb_run.log(flat, step=step)

    if print_every and (step % print_every == 0 or step <= 1):
        summary = "  ".join(f"{key}={value:.4f}" for key, value in flat.items())
        print(f"[step {step:4d}] {summary}", flush=True)


def log_rollouts(
    run_dir: Path,
    step: int,
    batch: RolloutBatch,
    reward_fn: Callable[[str, str], dict[str, float]],
    wandb_run=None,
    n_show: int = 2,
    group_size: int | None = None,
    preview_groups: int = 4,
) -> None:
    """把一批 rollout 落三份：jsonl（全量、机读）、md（按组预览、人读）、wandb Table。

    jsonl 一行一条响应，适合 pandas / jq；但通读一串转义过的 JSON 判断格式对错很痛苦，
    所以再写一份 rollouts.md，按 prompt 分组，把同一组 group_size 条响应并排放在一起。
    分组的意义在于：reward 是组内相对比较出来的，一条 response 单独看没信息量。
    """
    rows = []
    for prompt, response, gold, finish, n_tokens in zip(
        batch.prompts, batch.responses, batch.ground_truths,
        batch.finish_reasons, batch.token_counts,
    ):
        score = reward_fn(response, gold)
        rows.append({
            "step": step,
            "gold": gold,
            "reward": score["reward"],
            "format_reward": score["format_reward"],
            "answer_reward": score["answer_reward"],
            "tokens": n_tokens,
            "finish_reason": finish,
            "response": response,
        })

    with open(run_dir / "rollouts.jsonl", "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if group_size is None:
        group_size = _infer_group_size(batch.prompts)
    with open(run_dir / "rollouts.md", "a", encoding="utf-8") as f:
        f.write(_render_rollouts_md(step, batch, rows, group_size, preview_groups))

    if wandb_run is not None:
        import wandb

        table = wandb.Table(columns=list(rows[0].keys()))
        for row in rows:
            table.add_data(*row.values())
        # 表格没有 step 轴，用 key 里的步号区分，方便事后对比"训练前/训练后"
        wandb_run.log({f"rollouts/step_{step:04d}": table}, step=step)

    # 终端里抽 n_show 条：smoke test 时没有 wandb，靠这个眼看格式对不对
    for row in rows[:n_show]:
        print(
            f"  [rollout step {step}] reward={row['reward']} format={row['format_reward']} "
            f"tokens={row['tokens']} finish={row['finish_reason']}\n"
            f"    gold={row['gold']!r}\n"
            f"    response={row['response'][:400]!r}",
            flush=True,
        )


def _infer_group_size(prompts: list[str]) -> int:
    """数第一段连续相同的 prompt 有多长，作为 group_size（sample_rollouts 保证了这点）。"""
    if not prompts:
        return 1
    return sum(1 for prompt in prompts if prompt == prompts[0])


def _render_rollouts_md(
    step: int,
    batch: "RolloutBatch",
    rows: list[dict],
    group_size: int,
    preview_groups: int,
) -> str:
    """把一步的 rollout 渲染成 markdown 片段，按组排列，追加进 rollouts.md。"""
    n_groups = max(len(rows) // max(group_size, 1), 1)
    shown = min(preview_groups, n_groups) if preview_groups > 0 else n_groups
    mean_tokens = sum(row["tokens"] for row in rows) / max(len(rows), 1)
    n_truncated = sum(1 for row in rows if row["finish_reason"] == "length")
    lines = [
        f"\n## step {step}",
        "",
        f"- 组数 {n_groups}，响应 {len(rows)} 条；平均 {mean_tokens:.0f} tokens，"
        f"截断 {n_truncated}/{len(rows)}",
        f"- 总 reward 均值 {_mean(row['reward'] for row in rows):.2f}，"
        f"format {_mean(row['format_reward'] for row in rows):.2f}，"
        f"answer {_mean(row['answer_reward'] for row in rows):.2f}",
    ]
    if shown < n_groups:
        lines.append(f"- 下面只展开前 {shown}/{n_groups} 组，全量见 rollouts.jsonl")

    for group_index in range(shown):
        start = group_index * group_size
        group_rows = rows[start : start + group_size]
        gold = batch.ground_truths[start]
        lines += [
            "",
            f"### [{group_index + 1}/{n_groups}] gold={gold}",
            "",
            "<details><summary>prompt</summary>",
            "",
            "```text",
            batch.prompts[start].rstrip(),
            "```",
            "",
            "</details>",
        ]
        for inner_index, row in enumerate(group_rows):
            lines += [
                "",
                f"**#{inner_index}** reward={row['reward']} format={row['format_reward']} "
                f"answer={row['answer_reward']} tokens={row['tokens']} "
                f"finish={row['finish_reason']}",
                "",
                "```text",
                row["response"].rstrip(),
                "```",
            ]
    return "\n".join(lines) + "\n"


def _mean(values) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


def save_checkpoint(policy, tokenizer, path: Path) -> None:
    """存一份 HF 格式的权重，方便之后单独对某个 step 做评估/采样。"""
    path.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"[checkpoint] saved to {path}", flush=True)


def init_wandb(args: argparse.Namespace, run_dir: Path):
    """按需初始化 wandb；缺失 wandb 或显式关闭时返回 None，其余流程不受影响。

    WANDB_MODE=offline 时日志会落到 run_dir/wandb/ 下，训练结束后
    `wandb sync <dir>` 再上传——服务器没有外网时的常规做法。
    """
    if args.wandb_mode == "disabled" or not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print("[warn] 未安装 wandb，仅写本地 metrics.jsonl / rollouts.jsonl", flush=True)
        return None

    # 命令行优先于环境变量：否则 shell 里留着的 WANDB_MODE=offline 会让 --wandb-mode online 静默失效
    os.environ["WANDB_MODE"] = args.wandb_mode
    return wandb.init(
        project=args.wandb_project,
        # 与本地目录同名，两边能一一对上（experiment 已经在 run 名里）
        name=run_dir.name,
        config={key: str(value) for key, value in vars(args).items()},
        dir=str(run_dir),
    )


# =====================================================================
# 模型加载 / 主入口
# =====================================================================
def load_policy(args: argparse.Namespace):
    """加载 policy 模型与 tokenizer。

    没有直接用 checkpoint.get_model_and_tokenizer：它把 dtype 固定成 bfloat16、
    attn 固定成 flash_attention_2，而这台 Windows 机器没有 flash-attn。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device or args.policy_device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# =====================================================================
# 调试：确认 vLLM 多 prompt + n>1 时的返回顺序
# =====================================================================
def verify_rollout_order(args: argparse.Namespace) -> None:
    """打印 /v1/completions 返回的 index 序列，用来判断能否一次发多个 prompt。

    要区分的是两种语义：
      - index = i * n + j（全局扁平下标）→ choices 已按 prompt 分组，全局排序是空操作，可批量请求；
      - index = j（组内下标，每个 prompt 都从 0 重新数）→ 全局排序会把不同 prompt 交错。
    这里直接打原始 HTTP，因为 vllm_utils.generate_completions 把 index 丢掉了。
    """
    from vllm_utils import VLLMServer, _http_json

    server = VLLMServer(
        model_id=args.model_id, host=args.vllm_host, port=args.vllm_port,
        gpu=args.gpu, seed=args.seed, launch_server=args.launch_server,
        gpu_memory_utilization=args.vllm_gpu_util,
    )
    server.start()
    try:
        payload = {
            "model": args.model_id,
            "prompt": ["Q: what is 1+1? A:", "Q: what is 2+2? A:"],
            "max_tokens": 4,
            "temperature": 1.0,
            "n": 2,
            "seed": 0,
            "return_token_ids": True,
        }
        response = _http_json("POST", f"{server.base_url}/v1/completions", payload, timeout=300)
        for position, choice in enumerate(response["choices"]):
            print(f"返回位置 {position}: index={choice['index']}  text={choice['text']!r}")
    finally:
        server.stop()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.verify_order:
        verify_rollout_order(args)
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = prepare_run_dir(args)
    wandb_run = init_wandb(args, run_dir)

    train_examples, val_examples = prepare_data(args)
    prompt_fn = make_prompt_fn(args.prompt_mode)
    sampling_params = make_sampling_params(args)
    reward_fn = make_reward_fn(args.prompt_mode)

    policy, tokenizer = load_policy(args)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )

    backend = build_backend(args, policy, tokenizer)
    backend.start()
    try:
        backend.init_weight_sync(args.policy_device)
        train(
            args=args,
            policy=policy,
            tokenizer=tokenizer,
            optimizer=optimizer,
            backend=backend,
            reward_fn=reward_fn,
            train_examples=train_examples,
            val_examples=val_examples,
            prompt_fn=prompt_fn,
            sampling_params=sampling_params,
            run_dir=run_dir,
            wandb_run=wandb_run,
        )
    finally:
        backend.stop()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
