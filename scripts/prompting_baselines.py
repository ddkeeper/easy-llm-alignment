# -*- coding: utf-8 -*-
"""2.4 实验：在 GSM8K 上评估 OLMo-2-0425-1B 的三种提示词。

每种提示词跑一遍测试集，逐条打分，统计三类输出：
    (1, 1) 格式对、答案也对
    (1, 0) 格式对、答案错
    (0, 0) 格式就不对，压根没解析出答案
后两类按分组写进 review.md，方便人工翻看「到底是真答错，还是没被解析出来」。

vLLM 只能跑在 Linux / WSL，Windows 走客户 / 服务器分离，见 README「环境配置」。
运行（在仓库根目录下）：python scripts/prompting_baselines.py
"""

import json
import sys
from pathlib import Path

# 公共模块在根目录的 utils/ 下，加进搜索路径才 import 得到
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from utils.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from utils.vllm_utils import VLLMServer

# ---------------------------------- 配置 ----------------------------------
MODEL_ID = "allenai/OLMo-2-0425-1B"

LAUNCH_SERVER = False  # True = 本机拉起 vLLM（仅 Linux）；False = 连已经跑着的服务
HOST = "127.0.0.1"
PORT = 8001  # 走 SSH 隧道时填本地端口；服务器那边监听的是 8000
GPU = 0  # 只在 LAUNCH_SERVER=True 时生效

SAMPLING = {
    "max_tokens": 512,  # 文档 3.4
    "temperature": 1.0,  # 文档 3.4（top-p 用 vLLM 默认的 1.0）
    "seed": 0,
    "n": 1,  # generate_completions 必定读这个键，只采一条也要写
}
# 停止串只给 r1_zero 系列。question_only 没要求输出 </answer>，
# 给它设这个停止符会把正常回答截断
STOP = {"stop": ["</answer>"], "include_stop_str_in_output": True}

BATCH_SIZE = 64  # 每次 HTTP 请求塞多少条 prompt，只决定分块，不影响返回顺序
NUM_EXAMPLES = None  # 调试时填个小数字；None = 跑完整测试集 1319 条
REVIEW_LIMIT = 20  # review.md 里每类列多少条

# 一个 mode 就三件事：用哪份模板、用哪个 reward fn、要不要停止串
MODES = {
    "question_only": {
        "template": "question_only.prompt",
        "reward_fn": question_only_reward_fn,  # 在 \boxed{} 里找答案
        "stop": False,
    },
    "r1_zero": {
        "template": "r1_zero.prompt",
        "reward_fn": r1_zero_reward_fn,  # 看 <think> / <answer> 标签
        "stop": True,
    },
    "r1_zero_three_shot": {
        "template": "r1_zero_three_shot_gsm8k.prompt",
        "reward_fn": r1_zero_reward_fn,
        "stop": True,
    },
}

DATA_DIR = _ROOT / "data" / "gsm8k"
PROMPT_DIR = _ROOT / "prompts"
OUTPUT_DIR = _ROOT / "outputs" / "prompting_baselines"
# --------------------------------------------------------------------------


def load_gsm8k(path) -> list[dict]:
    """读 jsonl，返回 [{"question": str, "gold": str}]。

    原始 answer 是 "{rationale} #### {answer}"，打分只要最后那个答案，
    所以按最后一个 #### 切开再 strip。
    """
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            examples.append(
                {
                    "question": record["question"],
                    "gold": record["answer"].rsplit("####", 1)[-1].strip(),
                }
            )
    return examples


def summarize(rows) -> dict:
    """三分类计数、准确率、平均生成长度。作业要你实现的就是这个函数。

    文档 3.3 规定不给 partial credit，total reward 就等于 answer reward，
    所以准确率的分子是 answer_reward == 1 的条数，分母是全部条数。
    """
    num_fmt1_ans1 = sum(r["format_reward"] == 1 and r["answer_reward"] == 1 for r in rows)
    num_fmt1_ans0 = sum(r["format_reward"] == 1 and r["answer_reward"] == 0 for r in rows)
    num_fmt0_ans0 = sum(r["format_reward"] == 0 and r["answer_reward"] == 0 for r in rows)
    return {
        "num_fmt1_ans1": num_fmt1_ans1,
        "num_fmt1_ans0": num_fmt1_ans0,
        "num_fmt0_ans0": num_fmt0_ans0,
        "accuracy": num_fmt1_ans1 / len(rows) if rows else 0.0,
        "average_num_tokens": sum(r["num_tokens"] for r in rows) / len(rows) if rows else 0.0,
        "num_truncated": sum(r["finish_reason"] == "length" for r in rows),
    }


def save(mode, rows) -> None:
    """全量写 all.jsonl 给程序读；两类待复查的样本按分组写进 review.md 给人眼读。"""
    out_dir = OUTPUT_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "all.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    groups = [
        ("格式对、答案错", [r for r in rows if (r["format_reward"], r["answer_reward"]) == (1, 0)]),
        ("格式没解析出来", [r for r in rows if r["format_reward"] == 0]),
    ]
    lines = [f"# {mode}", "", f"共 {len(rows)} 条", ""]
    for title, group in groups:
        lines += [f"## {title}（{len(group)} 条，下面列前 {min(REVIEW_LIMIT, len(group))} 条）", ""]
        for row in group[:REVIEW_LIMIT]:
            lines += [f"### gold={row['gold']} · tokens={row['num_tokens']} · {row['finish_reason']}", ""]
            lines += ["**Question**", "", "> " + row["question"].replace("\n", " "), "", "**Response**", ""]
            body = row["response"].splitlines() or ["(空输出)"]
            # 缩进 4 空格 = markdown 代码块，回答里带任何符号都不会破坏排版
            lines += ["    " + ln if ln.strip() else "    " for ln in body]
            lines += ["", "---", ""]
    (out_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    examples = load_gsm8k(DATA_DIR / "test.jsonl")
    if NUM_EXAMPLES is not None:
        examples = examples[:NUM_EXAMPLES]
    print(f"测试集：{len(examples)} 条")

    server = VLLMServer(
        model_id=MODEL_ID,
        host=HOST,
        port=PORT,
        gpu=GPU,
        seed=SAMPLING["seed"],
        launch_server=LAUNCH_SERVER,
    )
    server.start()

    summary = {}
    try:
        for mode, cfg in MODES.items():
            # 1. 拼提示词。模板里只有 {question} 一个占位符，用 format；
            #    不能用 replace——r1_zero 正文里有 "The User asks a question"，会被一起换掉
            template = (PROMPT_DIR / cfg["template"]).read_text(encoding="utf-8")
            prompts = [template.format(question=ex["question"]) for ex in examples]

            # 2. 生成
            params = {**SAMPLING, **STOP} if cfg["stop"] else SAMPLING
            completions = server.generate_completions(prompts, params, BATCH_SIZE)
            assert len(completions) == len(prompts)  # 返回顺序与 prompts 一一对应

            # 3. 打分：格式和答案各拿一个 reward，total 等于 answer（不给 partial credit）
            rows = [
                {
                    "question": ex["question"],
                    "gold": ex["gold"],
                    "response": c.text,
                    "finish_reason": c.finish_reason,
                    "num_tokens": len(c.token_ids),
                    **cfg["reward_fn"](c.text, ex["gold"]),
                }
                for ex, c in zip(examples, completions)
            ]

            save(mode, rows)
            summary[mode] = summarize(rows)
            print(f"{mode}: {summary[mode]}")
    finally:
        server.stop()

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
