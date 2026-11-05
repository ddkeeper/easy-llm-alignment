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

脚本自己拉起 vLLM 服务，三个提示词跑完再关掉，一条命令到底。先改脚本开头两个常量，否则它会去连一个已经跑着的服务，等十分钟才超时：

```python
LAUNCH_SERVER = True   # 脚本自己起 vLLM
GPU = 0                # 用哪张卡
```

```bash
conda activate alignment
python scripts/prompting_baselines.py
```

模型权重首次会自动从 HuggingFace 下载（国内先设 `HF_ENDPOINT`）；已经缓存过就加 `HF_HUB_OFFLINE=1`，跳过启动时的联网校验。

### 共享卡或小显存卡：分两个终端跑

卡上还有别人的任务时，一键模式过不去——`utils/vllm_utils.py` 里显存占比默认 0.9，vLLM 一启动就报 `ValueError: Free memory ... is less than desired ...`。这种情况手动起服务，把占比压到空闲显存以下：

```bash
# 终端 A：起服务
conda activate alignment
vllm serve allenai/OLMo-2-0425-1B \
    --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --enable-prefix-caching \
    --gpu-memory-utilization 0.6 --seed 0
```

```bash
# 终端 B：把 LAUNCH_SERVER 改回 False，脚本只连不启
conda activate alignment
python scripts/prompting_baselines.py
```

`0.6` 是按邻居占掉七八 GB 估的，按自己卡上实际空闲显存调；独占整卡可以直接给 `0.9`，WSL2 里 6 GB 的单卡给 `0.8` 以下。启动时报 `Error in memory profiling`（邻居进程在启动期间释放显存，vLLM 探测到"越用越多"），再加 `--kv-cache-memory-bytes 8589934592` 固定 KV 缓存大小。多卡机器在 `vllm serve` 前加 `CUDA_VISIBLE_DEVICES=<卡号>` 指定显卡。

结果写在 `outputs/prompting_baselines/<提示词名>/`：`summary.json` 是三类输出的汇总，`all.jsonl` 是逐条打分，`review.md` 列出答错的样本。

## 参考资料

- [CS 336 课程官网](https://cs336.stanford.edu/)
- [CS 336 课程回放（2026 春季）](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
- [CS 336 课程作业文档](https://github.com/stanford-cs336)
