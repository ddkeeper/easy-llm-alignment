# easy-llm-alignment

本项目为[**【斯坦福 CS336】零基础手搓大模型系列视频**](https://www.bilibili.com/video/BV1NV8t6AE4T/)第二章「后训练」的配套项目。

第一章从零训练出一个能写童话的小模型；第二章换成别人训练好的基座模型，只做后训练。目前完成的是最轻的一段：不训练，只用提示词（零样本 / 思维链 / 少样本）在 GSM8K 上测出基线，看清基座模型的能力边界。GRPO 与 SFT、RLHF 随视频更新。

## 项目结构

```
easy-llm-alignment/
├── notebooks/
│   └── 2_prompting/
│       └── 2_prompting.ipynb      # 2.2–2.4：提示词模板、奖励函数、数据集、结果统计
├── scripts/
│   └── prompting_baselines.py     # 三种提示词基线跑 GSM8K
├── utils/
│   ├── drgrpo_grader.py           # 答案判定：从输出里抽出答案再比对
│   ├── vllm_utils.py              # 起 vLLM 服务、发请求
│   └── checkpoint.py              # 加载 HF 模型与分词器
├── prompts/
│   ├── question_only.prompt
│   ├── r1_zero.prompt
│   └── r1_zero_three_shot_gsm8k.prompt
├── data/
│   └── gsm8k/                     # train.jsonl 7473 条 / test.jsonl 1319 条
├── cs336_assignment5_alignment.pdf
├── environment.yml
└── README.md
```

## 环境配置

需要 Python 3.12 与较新的 PyTorch，与 [easy-llm](https://github.com/ddkeeper/easy-llm) 的环境不通用。

```bash
conda env create -f environment.yml
conda activate alignment
```

推理走 vLLM，Windows 上装不上，只在 Linux / WSL 里装：

```bash
pip install vllm
```

国内网络建议先设镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 数据集

GSM8K 已随仓库提供，解开即可用，无需另行下载：

```
data/gsm8k/train.jsonl    7473 条
data/gsm8k/test.jsonl     1319 条
```

## 运行

```bash
conda activate alignment

# 三种提示词各跑一遍测试集
python scripts/prompting_baselines.py
```

脚本要连一个已经起好的 vLLM 服务，`LAUNCH_SERVER`、`HOST`、`PORT` 在文件开头。本机连服务器走 SSH 隧道、两端都在服务器、两端都在 WSL2 这三种跑法见 2.4 讲解。结果写到 `outputs/prompting_baselines/`。

## 参考资料

- [CS 336 课程官网](https://cs336.stanford.edu/)
- [CS 336 课程回放（2026 春季）](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
- [CS 336 课程作业文档](https://github.com/stanford-cs336)
