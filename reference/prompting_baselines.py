"""Problem (prompting_baselines): 在 GSM8K 上评估 OLMo-2-0425-1B 的三种 prompting baseline。

基于 C1_prompting.py 的思路补全：数据读取、服务启动、prompt 组装、采样参数、逐条打分都已填好，
只留 tally()（作业 (a) 的三分类统计口径）给你实现。

运行（Linux + GPU 服务器；vllm_utils 里的 pkill / killpg 是 POSIX 专用，Windows 上跑不了）。
两种方式：
  1. 服务端本机跑：LAUNCH_SERVER = True，脚本自己拉起 vLLM 进程；
  2. 远程客户端（Windows 推荐）：服务器上先 `vllm serve ... --port 8000`，
     本机 `ssh -L 8001:127.0.0.1:8000 user@server`，然后 LAUNCH_SERVER = False、PORT = 8001。
在 assignment5-alignment 目录下执行：
    python prompting_baselines.py
"""
# --------------------------- 请求链路备忘 ---------------------------
# 你的 Windows 电脑（本地）
#     ↓ 访问 http://127.0.0.1:8001/v1/completions
#     ↓ 请求交给本地 ssh 程序
#     ↓ SSH 加密隧道（走 GPU3 的 ssh 连接，一般是 22 端口）
# GPU3 服务器
#     ↓ ssh 把数据包转发给 GPU3 本机的 127.0.0.1:8000（vLLM 服务）
# vLLM 服务收到请求，生成结果
#     ↓ 原路返回 → 沿着 ssh 隧道发回 Windows → 程序拿到返回结果
#
# 注意：这段必须写成 # 注释。字符串字面量不能夹在模块 docstring 与
# `from __future__ import annotations` 之间，否则报
# SyntaxError: from __future__ imports must occur at the beginning of the file
# -------------------------------------------------------------------
from __future__ import annotations

import json
import sys
from pathlib import Path

# reference/ 放算法实现；公共模块在 utils/，提示词、数据、产物在仓库根目录
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from utils.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from utils.vllm_utils import VLLMServer

# ------------------------------- CONFIG -------------------------------
MODEL_ID = "allenai/OLMo-2-0425-1B"
LAUNCH_SERVER = False  # True=在本机拉起 vLLM 进程（仅 Linux）；False=连已部署好的远程服务
HOST = "127.0.0.1"  # 走 SSH 隧道就填 127.0.0.1；直连服务器填它的 IP
PORT = 8001  # 隧道本地端口（服务器端监听的是 8000）
GPU = 0  # 仅 LAUNCH_SERVER=True 时生效
MAX_TOKENS = 512  # 文档 3.4：max generation length 512
TEMPERATURE = 1.0  # 文档 3.4：temperature 1.0（top-p 未透传，走 vLLM 默认 1.0）
SEED = 0
BATCH_SIZE = 64  # 每次 HTTP 请求塞多少条 prompt，只影响分块、不影响返回顺序
NUM_EXAMPLES = None  # 先用小样本验证流程；跑正式结果改成 None（全量 1319 条）
REVIEW_K = 100  # 每类导出多少条供人工复查；改成 None 就是全导出

MODES = ("question_only", "r1_zero", "r1_zero_three_shot")
MODE_TO_PROMPT = {
    "question_only": "question_only.prompt",
    "r1_zero": "r1_zero.prompt",
    "r1_zero_three_shot": "r1_zero_three_shot_gsm8k.prompt",
}

DATA_DIR = _ROOT / "data" / "gsm8k"
PROMPT_DIR = _ROOT / "prompts"
OUTPUT_DIR = _ROOT / "outputs" / "prompting_baselines"
# ----------------------------------------------------------------------


def load_gsm8k(path: Path) -> list[dict]:
    """读取 GSM8K jsonl，返回 [{"question": str, "gold": str}, ...]。

    原始 answer 形如 "{rationale} #### {answer}"，grader 只认最后那个答案，
    所以按最后一个 "####" 切开再 strip。
    """
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                examples.append(
                    {
                        "question": record["question"],
                        "gold": record["answer"].rsplit("####", 1)[-1].strip(),
                    }
                )
    return examples


def build_prompt(mode: str, question: str) -> str:
    """读对应模板并填入题目。

    用 str.format 而不是 str.replace：r1_zero 模板正文里有 "The User asks a question"，
    replace 会把这句一起替换掉；而 question_only.prompt 里的 \\boxed{{}} 正是为 format 写的转义。
    """
    template = (PROMPT_DIR / MODE_TO_PROMPT[mode]).read_text(encoding="utf-8")
    # 把模板里的 {question} 替换成 question
    return template.format(question=question)


def make_sampling_params(mode: str) -> dict:
    """构造传给 server.generate_completions 的 sampling_params。

    两个不能省的细节：
      - "n" 是硬取键（generate_completions 内部必定会取 sampling_params["n"]），只采一条也必须显式写 1；
      - stop 只对 r1_zero 系列生效：question_only 没要求输出 </answer>，加了会把正常回答截断。
    """
    sampling_params = {
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "n": 1,
    }
    if mode != "question_only":
        sampling_params["stop"] = ["</answer>"]
        sampling_params["include_stop_str_in_output"] = True
    return sampling_params


def score_response(mode: str, response: str, gold: str) -> dict:
    """给单条模型输出打分。

    grader 的签名是 (response, ground_truth) 两个字符串，返回
    {"format_reward", "answer_reward", "reward"}；question_only 与 r1_zero 系列的格式期望不同，
    必须按 mode 选对应的 reward fn。
    """
    if mode == "question_only":
        return question_only_reward_fn(response, gold)
    return r1_zero_reward_fn(response, gold)


def tally(rows: list[dict]) -> dict:
    """作业 (a) 的统计口径 —— 这一问真正需要你思考的部分。

    输入 rows：每个元素是一条样本的结果，字段为
        mode / question / gold / prompt / response / finish_reason /
        num_tokens / format_reward / answer_reward / reward

    需要产出（键名自定，会直接打印并写入 summary.json）：
      - 三分类各自的条数：(format=1, answer=1) / (format=1, answer=0) / (format=0, answer=0)；
      - 准确率口径：文档 3.3 说总 reward 用 answer reward、不给部分分，想清楚分子分母各是什么；
      - （可选）平均生成长度 num_tokens，用来判断有多少条是被 512 截断的。

    提示：reward 字段是浮点 1.0 / 0.0，直接 == 1 比较即可；三分类条数加起来应当等于 len(rows)。
    """
    res = {}
    res["num_fmt1_ans1"] = 0
    res["num_fmt1_ans0"] = 0
    res["num_fmt0_ans0"] = 0

    tot_tokens = 0
    num_correct = 0
    for row in rows:
        tot_tokens += row["num_tokens"]
        if row["format_reward"] == 1 and row["answer_reward"] == 1: 
            res["num_fmt1_ans1"] += 1
            num_correct += 1
        elif row["format_reward"] == 1 and row["answer_reward"] == 0:  res["num_fmt1_ans0"] += 1
        elif row["format_reward"] == 0 and row["answer_reward"] == 0:  res["num_fmt0_ans0"] += 1
        else: pass

    res["accuracy"] = num_correct / len(rows) if rows else 0.0
    res["average_num_tokens"] = tot_tokens / len(rows) if rows else 0.0

    return res


def save_results(mode: str, tag: str, rows: list[dict], limit: int | None = None) -> None:
    """把一批样本写到 OUTPUT_DIR/<mode>/<tag>.{jsonl,md}，两种格式各一份。

    - .jsonl：给程序读，字段齐全（后续 GRPO 对比、复查脚本都靠它）
    - .md   ：给人眼读，一条样本一段，response 保留真实换行

    limit 只截断落盘的内容，不影响 summary 的统计口径。
    """
    if limit is not None:
        rows = rows[:limit]

    out_dir = OUTPUT_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{tag}.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [f"# {mode} · {tag}", "", f"共 {len(rows)} 条", ""]
    for i, row in enumerate(rows, 1):
        lines.append(
            f"## [{i}] gold={row['gold']} · fmt={int(row['format_reward'])} · "
            f"ans={int(row['answer_reward'])} · tokens={row['num_tokens']} · "
            f"finish={row['finish_reason']}"
        )
        lines += ["", "**Question**", "", "> " + row["question"].replace("\n", " "), ""]
        lines += ["**Response**", ""]
        body = row["response"].splitlines() or ["(空输出)"]
        # 缩进 4 空格 = markdown 代码块，任何符号都不会破坏排版
        lines += ["    " + ln if ln.strip() else "    " for ln in body]
        lines.append("")
    (out_dir / f"{tag}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    examples = load_gsm8k(DATA_DIR / "test.jsonl")
    if NUM_EXAMPLES is not None:
        examples = examples[:NUM_EXAMPLES]
    print(f"评估样本数: {len(examples)}")

    server = VLLMServer(
        model_id=MODEL_ID,
        host=HOST,
        port=PORT,
        gpu=GPU,
        seed=SEED,
        launch_server=LAUNCH_SERVER,
    )
    server.start()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    for mode in MODES:
        # 把所有测试问题组装成带提示词模板的版本放进 prompts 
        prompts = [build_prompt(mode, example["question"]) for example in examples]
        completions = server.generate_completions(prompts, make_sampling_params(mode), BATCH_SIZE)
        assert len(completions) == len(prompts)

        rows = []
        for example, prompt, completion in zip(examples, prompts, completions):
            rows.append(
                {
                    "mode": mode,
                    "question": example["question"],
                    "gold": example["gold"],
                    "prompt": prompt,
                    "response": completion.text,
                    "finish_reason": completion.finish_reason,
                    "num_tokens": len(completion.token_ids),
                    **score_response(mode, completion.text, example["gold"]),
                }
            )

        save_results(mode, "all", rows)
        # 作业 (a) 要人工复查的两类，各自单独出一份，方便连着往下读
        save_results(
            mode,
            "fmt1_ans0",
            [r for r in rows if r["format_reward"] == 1 and r["answer_reward"] == 0],
            REVIEW_K,
        )
        save_results(
            mode,
            "fmt0_ans0",
            [r for r in rows if r["format_reward"] == 0 and r["answer_reward"] == 0],
            REVIEW_K,
        )

        summary[mode] = tally(rows)
        print(f"\n{mode}: {summary[mode]}")

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果目录: {OUTPUT_DIR}")
    server.stop()


if __name__ == "__main__":
    main()
