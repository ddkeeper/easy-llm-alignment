# easy-llm-alignment

本项目为[**【斯坦福 CS336】零基础手搓大模型系列视频**](https://www.bilibili.com/video/BV1NV8t6AE4T/)第二章「后训练」的配套代码。

第一章从零训练一个小语言模型；第二章改用预训练基座模型，只做后训练。当前完成提示词部分：以零样本、思维链、少样本三种模板在 GSM8K 上测量基线，作为后续 GRPO 与 SFT、RLHF 的对照。后续内容随视频更新。

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

与第一章共用同一个环境（Python 3.10 + PyTorch 2.5.1 + CUDA 12.1），已装过 [easy-llm](https://github.com/ddkeeper/easy-llm) 的直接激活：

```bash
conda activate nanogpt
```

新机器按 `environment.yml` 创建（依赖相同，环境名为 `alignment`，下文命令里的 `nanogpt` 换成它）：

```bash
conda env create -f environment.yml
conda activate alignment
```

## 数据集

GSM8K 随仓库提供，无需另行下载：

```
data/gsm8k/train.jsonl    7473 条
data/gsm8k/test.jsonl     1319 条
```

## vLLM 服务器启动

vLLM 仅支持 Linux / WSL2，且 0.19.1 固定依赖 `torch==2.10`，与上面环境的 2.5.1 冲突，单独建一个环境：

```bash
# 1. 新建环境并安装 vLLM
conda create -n vllm python=3.12 -y
conda activate vllm
pip install vllm

# 2. 国内网络先设镜像，首次启动要从 HuggingFace 下权重
export HF_ENDPOINT=https://hf-mirror.com

# 3. 起服务，只监听回环供同机脚本调用
vllm serve allenai/OLMo-2-0425-1B \
    --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.9 \
    --seed 0
```

| 参数 | 说明 |
| --- | --- |
| `--host` / `--port` | 只监听本机回环，供同机脚本调用 |
| `--dtype bfloat16` | 以 bfloat16 加载权重，1B 模型约占 2 GB 显存 |
| `--enable-prefix-caching` | 复用固定模板的 KV 缓存，减少重复计算 |
| `--gpu-memory-utilization` | 显存占用上限：独占整卡取 `0.9`，共享卡按空闲显存下调，6 GB 单卡取 `0.8` 以下 |
| `--seed 0` | 固定采样种子 |

显存相关报错：

- `ValueError: Free memory ... is less than desired ...`：占比超过当前空闲显存，调低 `--gpu-memory-utilization`。
- `Error in memory profiling`：共享卡上其他进程在启动期间释放显存，追加 `--kv-cache-memory-bytes 8589934592` 固定 KV 缓存大小。

多卡机器在命令前加 `CUDA_VISIBLE_DEVICES=<卡号>` 指定显卡。首次启动会下载模型权重；权重已缓存时可设置 `HF_HUB_OFFLINE=1` 跳过联网校验。

## 实验

提示词基线：零样本、思维链、少样本三种模板分别在 GSM8K 测试集（1319 题）上推理，统计格式合格率、答案准确率、平均输出长度与截断数。

vLLM 服务启动后，另开一个终端执行：

```bash
conda activate nanogpt
python scripts/prompting_baselines.py
```

脚本默认连接 `127.0.0.1:8000`（`LAUNCH_SERVER = False`），依次跑完三种模板。调试时可在脚本开头设置 `NUM_EXAMPLES`，只跑前若干条。

结果输出到 `outputs/prompting_baselines/<提示词名>/`：`summary.json` 为汇总统计，`all.jsonl` 为逐条打分，`review.md` 为答错样本。

## 参考资料

- [CS 336 课程官网](https://cs336.stanford.edu/)
- [CS 336 课程回放（2026 春季）](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
- [CS 336 课程作业文档](https://github.com/stanford-cs336)
