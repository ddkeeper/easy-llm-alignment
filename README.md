# easy-llm-alignment

本项目为【跟着斯坦福动手学大模型系列视频】第二期[大模型后训练入门](https://www.bilibili.com/video/BV1NV8t6AE4T/)的配套代码。

## 项目结构

```
easy-llm-alignment/
├── notebooks/                      # 代码讲解文档
├── scripts/
│   └── prompting_baselines.py      # 提示词基线实验
├── utils/                          # 判分、vLLM 封装、模型加载
├── prompts/                        # 三个提示词模板
├── data/                           # 数据集
│   └── gsm8k/
├── environment.yml
└── README.md
```

## 环境配置

使用 [vLLM](https://github.com/vllm-project/vllm) 作为本项目的推理引擎，vLLM 需要跑在 Linux 或 WSL 上。模型训练与 vLLM 推理共用一套环境。

```bash
conda env create -f environment.yml
conda activate alignment
```

> vLLM 与 flash-attn 只支持 Linux / WSL2。Windows 下创建环境时删掉 `environment.yml` 里的这两行，仅作客户端连接远程服务。

## 数据集

推理 RL 部分使用 GSM8K 数据集 `data/gsm8k/`。

## 实验

### 提示词基线实验

**方式一：服务端与客户端分开启动**

```bash
# 服务端（WSL 或 Linux 服务器）
conda activate alignment
vllm serve allenai/OLMo-2-0425-1B \
    --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.9

# 客户端（同机的 WSL 或服务器，或者是本地 Windows）
conda activate alignment
python scripts/prompting_baselines.py
```

| 参数 | 说明 |
| --- | --- |
| `--host` / `--port` | 服务监听的地址与端口，`0.0.0.0` 允许其他机器连接（需放行端口），`127.0.0.1` 仅本机；端口与脚本里的 `PORT` 一致 |
| `--dtype bfloat16` | 以 bfloat16 加载权重，1B 模型约占 2 GB 显存 |
| `--enable-prefix-caching` | 复用固定模板的 KV 缓存，减少重复计算 |
| `--gpu-memory-utilization` | 显存占用上限：独占整卡取 `0.9`，共享卡按空闲显存下调 |

客户端与服务端不在同一台机器时，把脚本开头的 `HOST` 改成服务端 IP。

**方式二：脚本内拉起 vLLM**

```bash
# 修改 scripts/prompting_baselines.py 内参数
LAUNCH_SERVER = True   # 脚本自己拉起 vLLM
GPU = 0                # 用哪张卡

# 终端内运行
conda activate alignment
python scripts/prompting_baselines.py
```

## 参考资料

- [CS 336 课程官网](https://cs336.stanford.edu/)
- [CS 336 课程回放（2026 春季）](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)
- [CS 336 课程作业文档](https://github.com/stanford-cs336)
