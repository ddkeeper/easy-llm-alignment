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

推理另装 vLLM（原生 Windows 装不上，只能在 Linux / WSL2 里）：

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

## vLLM 服务器启动

推理走一个独立的 vLLM 服务。Linux 和 WSL2 用同一条命令：

```bash
conda activate alignment
vllm serve allenai/OLMo-2-0425-1B \
    --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --enable-prefix-caching \
    --gpu-memory-utilization 0.9 --seed 0
```

`--gpu-memory-utilization` 是这张卡最多占多少比例。卡上没有别人的任务就给 `0.9`；共享卡按空闲显存往下压（我们那张 24 GB 卡、邻居占七八 GB 时用 `0.6`）；WSL2 的 6 GB 单卡给 `0.8` 以下。给高了会直接报 `ValueError: Free memory ... is less than desired ...`。

共享卡上还可能报 `Error in memory profiling`——邻居进程在启动期间释放了显存，vLLM 探测到"越用越多"。加上 `--kv-cache-memory-bytes 8589934592` 固定 KV 缓存大小，跳过探测。

多卡机器在 `vllm serve` 前加 `CUDA_VISIBLE_DEVICES=<卡号>` 指定显卡。首次启动会从 HuggingFace 下载权重，缓存过之后可以设 `HF_HUB_OFFLINE=1` 跳过联网校验。

## 实验

提示词基线：三种模板（零样本 / 思维链 / 少样本）在 GSM8K 测试集 1319 题上各跑一遍，统计格式合格率、答案准确率、平均输出长度和截断数。两种跑法任选一种。

### 方式一：服务起好了，另开终端跑脚本

上一节的 `vllm serve` 留在终端里别关，新开一个：

```bash
conda activate alignment
python scripts/prompting_baselines.py
```

脚本默认 `LAUNCH_SERVER = False`、`PORT = 8000`，直接连本机服务。要反复调脚本、改 `NUM_EXAMPLES` 试跑就用这个方式，模型只加载一次。

### 方式二：脚本自己起服务

改掉脚本开头两个常量，一条命令跑完：

```python
LAUNCH_SERVER = True   # 脚本自己拉起 vLLM
GPU = 0                # 用哪张卡
```

```bash
conda activate alignment
python scripts/prompting_baselines.py
```

跑完脚本会把服务一起关掉。显存被邻居占着时这条路走不通（`utils/vllm_utils.py` 里占比默认 0.9，会直接报 `ValueError`），用方式一自己指定 `--gpu-memory-utilization`。

结果写在 `outputs/prompting_baselines/<提示词名>/`：`summary.json` 是三类输出的汇总，`all.jsonl` 是逐条打分，`review.md` 列出答错的样本。

## 参考资料

- [CS 336 课程官网](https://cs336.stanford.edu/)
- [CS 336 课程回放（2026 春季）](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
- [CS 336 课程作业文档](https://github.com/stanford-cs336)
