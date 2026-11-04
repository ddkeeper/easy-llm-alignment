# easy-llm-alignment —— 第二章 后训练：对齐与推理 RL

在已有基座模型上做后训练：先用提示词（零样本 / 少样本 / 思维链）测出基线，再用 GRPO 系列算法让模型学会解数学题，最后比较不同策略梯度估计器的效果。

对应作业文档：
[`cs336_assignment5_alignment.pdf`](cs336_assignment5_alignment.pdf)（必选）、
[`cs336_assignment5_supplement_safety_rlhf.pdf`](cs336_assignment5_supplement_safety_rlhf.pdf)（可选，指令微调与 DPO）。

> `reference/` 是可直接运行的完整实现；讲解 Notebook 与 `scripts/` 下的分步代码在制作中。

## 目录

```
easy-llm-alignment/
├── reference/                 # 参考实现（完整、可运行）
│   ├── grpo.py                # GRPO 各组件（PDF 4.2）
│   ├── prompting_baselines.py # 提示词基线跑 GSM8K（PDF 3）
│   └── grpo_train.py          # GRPO 训练脚本（vLLM / HF 双后端）
├── scripts/                   # 讲解用代码
│   └── prompting_baselines.py # 重构后的提示词基线，配合 2.4 讲解
├── notebooks/                 # 讲解 Notebook
│   └── 2_prompting/           # 2.4 提示词基线实验
├── utils/                     # 公共模块，三层代码共用
│   ├── drgrpo_grader.py       # 数学题答案判定（GSM8K 奖励函数）
│   ├── vllm_utils.py          # 起 vLLM 服务、同步策略权重
│   └── checkpoint.py          # 加载 HF 模型与分词器
├── prompts/                   # 提示词模板
│   └── safety/                # 补充作业用的
├── data/                      # 数据集（已随仓库提供）
├── assets/                    # 讲解配图（drawio 源文件与 png）
├── docs/提示词/               # 写讲解文档用的提示词与模板
├── outputs/                   # 实验产物：rollout、指标、checkpoint（不入库）
├── cs336_assignment5_alignment.pdf
├── cs336_assignment5_supplement_safety_rlhf.pdf
└── environment.yml
```

**代码分三层**：`utils/` 是跑实验的公共工具（奖励函数、vLLM 封装、模型加载），`reference/` 是算法实现的完整版，`scripts/` 是讲解用的分步版。后两层都从 `utils/` import，脚本头部加一行路径即可：

```python
_HERE = Path(__file__).resolve().parent          # reference/ 或 scripts/
sys.path.insert(0, str(_HERE.parent))            # 仓库根目录
from utils.drgrpo_grader import r1_zero_reward_fn
```

## 环境配置

这一章需要 Python 3.12 与较新的 PyTorch（vLLM 的依赖要求），与 [`easy-llm`](https://github.com/ddkeeper/easy-llm) 的环境不通用。

```bash
conda env create -f environment.yml
conda activate alignment
```

**vLLM 只在 Linux / WSL 可用**，Windows 上装不上。Windows 本机用 `--backend hf` 跑通流程，正式实验再上服务器或 WSL：

```bash
pip install vllm    # 仅 Linux / WSL
```

国内网络建议先设镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 数据集

已随仓库提供，解开即可用，无需另行下载：

```
data/
├── gsm8k/                  # 4.7 MB  必选作业：train.jsonl / test.jsonl
├── hh/                     # 72 MB   补充作业：helpful / harmless 偏好数据
├── mmlu/                   # 7.4 MB  补充作业：dev / val / test
├── alpaca_eval/            # 1.9 MB  补充作业：指令遵循评测
└── simple_safety_tests/    # 16 KB   补充作业：安全评测
```

## 运行

```bash
conda activate alignment

# 提示词基线：在 GSM8K 上跑零样本 / 少样本 / 思维链
python scripts/prompting_baselines.py     # 讲解版，先读这个
python reference/prompting_baselines.py   # 参考实现

# GRPO 训练（默认 allenai/OLMo-2-0425-1B，默认超参即 PDF 4.3 建议值）
python reference/grpo_train.py --backend vllm --launch-server     # Linux / WSL，正式实验
python reference/grpo_train.py --backend hf                       # 任意平台，流程验证

# 本机显存不够时，先用小模型 + 小 batch 走通流程
python reference/grpo_train.py --backend hf --dry-run \
    --model-id HuggingFaceTB/SmolLM2-135M-Instruct --attn-implementation eager --device cpu
```

`--backend hf` 直接用 HuggingFace 的 `.generate()` 采样，验证的是流程能否走通，不是算法效果——HF 与 vLLM 的采样实现不同，数值不可迁移到正式结果。

Windows 上跑 `scripts/prompting_baselines.py` 要连远处已经起好的 vLLM，中间走 SSH 隧道；服务怎么起、隧道怎么开，见 2.4 正文。

## 与作业文档的对应

| 作业文档 | 参考实现 | 讲解 |
|---|---|---|
| 3 Prompting | `reference/prompting_baselines.py`、`scripts/prompting_baselines.py`、`prompts/*.prompt` | `notebooks/2_prompting/` |
| 4 Group Relative Policy Optimization | `reference/grpo.py`、`reference/grpo_train.py`、`utils/drgrpo_grader.py` | `notebooks/3_grpo/` |
| 5 RL algorithm variants | `reference/grpo.py` | `notebooks/4_pg_variants/` |
| 6 Off-policy RL | `reference/grpo.py` | `notebooks/5_off_policy/` |
| 7 Try your own | `reference/grpo_train.py`、`utils/vllm_utils.py` | `notebooks/6_experiments/` |
| Supplement: SFT / DPO | `prompts/safety/`、`utils/checkpoint.py` | `notebooks/1_sft_dpo/` |
