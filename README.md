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

推理走一个独立的 vLLM 服务：先起服务，再跑脚本。

### 服务器上跑（Linux + NVIDIA 卡）

```bash
conda activate alignment

# 终端 A：起服务，默认吃 90% 显存
vllm serve allenai/OLMo-2-0425-1B \
    --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --enable-prefix-caching \
    --gpu-memory-utilization 0.9 --seed 0
```

```bash
# 终端 B：三种提示词各跑一遍测试集，1319 道题
conda activate alignment
python scripts/prompting_baselines.py
```

多卡机器在 `vllm serve` 前加 `CUDA_VISIBLE_DEVICES=<卡号>` 指定显卡。

### WSL2 上跑（本机单卡）

WSL2 里装好 vLLM 并 cd 到仓库目录，命令与上面相同，只把显存占比降到 `0.8`（6 GB 的卡要留点余量）：

```bash
vllm serve allenai/OLMo-2-0425-1B \
    --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --enable-prefix-caching \
    --gpu-memory-utilization 0.8 --seed 0
```

首次启动会从 HuggingFace 下载模型权重，国内网络记得先设 `HF_ENDPOINT`。

**显存报错时**：`ValueError: Free memory ... is less than desired ...` 说明卡上已经有别人的任务，把 `--gpu-memory-utilization` 压到空闲显存以下；报 `Error in memory profiling` 再加 `--kv-cache-memory-bytes 8589934592`，固定 KV 缓存大小。

结果写在 `outputs/prompting_baselines/<提示词名>/`：`summary.json` 是三类输出的汇总，`all.jsonl` 是逐条打分，`review.md` 列出答错的样本。

## 参考资料

- [CS 336 课程官网](https://cs336.stanford.edu/)
- [CS 336 课程回放（2026 春季）](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
- [CS 336 课程作业文档](https://github.com/stanford-cs336)
