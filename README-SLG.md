# nanochat 项目导读与 LLM 学习路线

> 本文用于通过当前仓库学习大语言模型（LLM）的原理与实现。先介绍项目结构和依赖，再给出推荐阅读顺序、关键问题与实验方向。项目会持续演进，具体参数与行为以当前源码为准。

## 一、项目定位与整体架构

nanochat 是使用 PyTorch 实现的 GPT 类语言模型训练与推理项目，不是简单调用现成模型 API 的聊天应用。它覆盖分词器训练、语言模型预训练、监督微调、强化学习、评估和推理等环节。

它适合学习两个层面：

- **模型原理**：文本如何变成 token，Transformer 如何处理上下文，损失如何驱动参数更新，模型如何逐个 token 生成文本。
- **工程实现**：数据如何组批，模型如何保存和加载，如何进行对话微调，以及如何使用 KV Cache、混合精度和分布式计算。

当前版本包含较多面向效果和性能的改进，并不是未经修改的原版 GPT-2，也不是最精简的教科书 Transformer。第一遍学习应抓住主干，暂时跳过高级优化。

### 1.1 完整流程

```text
原始文本
  → 下载、读取数据
  → 训练 BPE 分词器
  → 文本编码为 token ID
  → 构造输入与下一个 token 的预测目标
  → GPT 模型预训练
  → 基础模型评估
  → 对话监督微调（SFT）
  → 对话模型评估与 CLI 交互
  → 可选：基于奖励的强化学习
```

推理时不执行上面的训练流程，而是加载已有模型和分词器：

```text
输入文本 → token ID → 模型输出 logits → 选择下一个 token
                  ↑                          ↓
                  └──────── 拼回上下文 ────────┘

生成的 token ID → 解码为文本
```

### 1.2 代码的组织方式

- `nanochat/`：可复用的核心实现，如模型、分词、数据加载、推理与优化器。
- `scripts/`：具体操作入口，如“训练基础模型”“进行 SFT”“启动命令行对话”。
- `tasks/`：训练或评估用的任务与数据集封装。
- `runs/`：将多个操作入口串起来的实验脚本。
- `tests/`：核心模块的测试。
- `dev/`：研究记录、分析材料和辅助工具，不是主训练流程的必读内容。

典型调用关系是：`runs/` 编排 `scripts/`，`scripts/` 调用 `nanochat/` 和 `tasks/`。这是一种阅读视角，并非所有模块都严格单向依赖。

## 二、目录与文件说明

### 2.1 根目录

| 路径 | 作用 |
| --- | --- |
| `README.md` | 上游项目说明、安装方式、运行示例与性能背景。 |
| `README-SLG.md` | 本文：面向原理学习的项目导读与学习路线。 |
| `pyproject.toml` | Python 版本要求、运行依赖、开发依赖、CPU/GPU 可选环境和 pytest 配置。 |
| `uv.lock` | uv 的依赖锁定文件，记录解析后的依赖版本与来源。 |
| `.python-version` | Python 版本管理工具使用的版本提示。 |
| `.gitignore` | Git 忽略规则。 |
| `LICENSE` | 项目许可证。 |
| `.claude/` | Claude 相关辅助配置与技能；当前包含论文阅读技能，不参与模型训练和推理。 |
| `.git/` | Git 版本控制元数据，不属于模型实现。 |

安装后可能出现的 `.venv/` 是本地 Python 虚拟环境，不是业务源码。

### 2.2 `nanochat/`：核心实现

| 文件 | 作用 | 学习优先级 |
| --- | --- | --- |
| `tokenizer.py` | BPE 分词器封装、编码与解码、特殊 token、对话格式和训练掩码。 | 高 |
| `dataset.py` | 预训练数据分片下载与 Parquet 数据读取。 | 中 |
| `dataloader.py` | 在线分词、文档拼接与裁剪、组批、输入与目标错位、数据读取状态。 | 高 |
| `gpt.py` | GPT 配置、Embedding、Attention、MLP、Transformer Block、损失与朴素生成。 | 最高 |
| `engine.py` | 带 KV Cache 的推理、采样、多样本生成和工具调用处理。 | 高，放在基础生成之后 |
| `loss_eval.py` | 使用 bits per byte（BPB）等逻辑评估语言模型。 | 中 |
| `core_eval.py` | 基础模型的 DCLM CORE 任务评估。 | 中 |
| `checkpoint_manager.py` | 模型检查点的保存、恢复及推理加载。 | 中 |
| `common.py` | 设备与精度选择、分布式初始化、日志、缓存目录等公共工具。 | 按需查阅 |
| `optim.py` | Muon 与 AdamW 组合优化器及分布式实现。 | 后期 |
| `flash_attention.py` | 统一注意力接口，在适用时使用 Flash Attention 3，否则回退到 PyTorch SDPA。 | 后期 |
| `fp8.py` | FP8 线性层与训练转换，用于特定硬件上的加速。 | 后期 |
| `execution.py` | 为代码任务提供受限的 Python 子进程执行能力；不能据此假设可安全执行任意不可信程序。 | 学习工具调用时再读 |
| `__init__.py` | Python 包入口，当前为空。 | 无需专门阅读 |

### 2.3 `scripts/`：操作入口

| 文件 | 作用 |
| --- | --- |
| `tok_train.py` | 从文本训练分词器并保存相关产物。 |
| `tok_eval.py` | 比较分词压缩效果。 |
| `base_train.py` | 基础语言模型预训练，包含初始化、数据加载、训练循环、评估和保存。 |
| `base_eval.py` | 基础模型评估，包括 BPB、CORE 和生成样例。 |
| `chat_sft.py` | 加载预训练模型，使用对话数据进行监督微调。 |
| `chat_eval.py` | 对话模型的选择题与生成式任务评估。 |
| `chat_rl.py` | 在 GSM8K 上使用采样与奖励进行简化的强化学习训练。 |
| `chat_cli.py` | 加载模型，在命令行交互或输入提示生成文本。 |
| `infer_bench.py` | 测量推理延迟、吞吐量及显存等性能指标。 |

通常从项目根目录以模块方式运行，例如 `python -m scripts.chat_cli`。聊天入口需要已有的模型检查点与分词器，并非安装依赖后即可直接聊天。

### 2.4 `tasks/`：任务与数据封装

| 文件 | 作用 |
| --- | --- |
| `common.py` | 数据下载与读取、任务基类、任务混合和顺序组合等公共逻辑。 |
| `smoltalk.py` | SmolTalk 对话数据封装。 |
| `mmlu.py` | MMLU 多领域选择题任务。 |
| `arc.py` | ARC 科学选择题任务。 |
| `gsm8k.py` | GSM8K 数学任务及答案处理，用于相关训练与评估。 |
| `humaneval.py` | 简单 Python 代码生成与执行评估；不要仅凭文件名假设它与其他项目的标准 HumanEval 协议完全一致。 |

注意区分数据集在不同脚本中的用途：某个任务出现在训练混合中，并不意味着它在所有入口里都只用于评估。

### 2.5 `runs/`：实验编排

| 文件 | 作用 |
| --- | --- |
| `runcpu.sh` | CPU/MPS 小规模端到端示例，覆盖数据准备、分词器、预训练与 SFT。 |
| `speedrun.sh` | 面向多 GPU 节点的参考流程，追求达到指定基础模型能力的训练速度。 |
| `miniseries.sh` | 不同模型规模的系列实验。 |
| `scaling_laws.sh` | 缩放规律相关实验。 |

这些脚本是实验配方，不是每次启动项目都必须执行的安装脚本。首次学习建议逐段阅读、按需运行，不要直接执行完整流水线。

### 2.6 `tests/`：行为验证与阅读示例

| 文件 | 作用 |
| --- | --- |
| `test_tokenizer.py` | 分词编码/解码与对话格式测试。 |
| `test_engine.py` | KV Cache、采样、随机种子和生成长度等测试。 |
| `test_attention_fallback.py` | 注意力回退逻辑，以及具备条件时的 FA3/SDPA 对比测试。 |
| `test_optim.py` | 优化器正确性与行为测试，需要 CUDA。 |
| `test_tasks.py` | 任务切片、数据封装与混合逻辑测试。 |
| `test_execution.py` | Python 代码执行限制与行为测试。 |

测试既能验证实现，也能帮助理解接口如何使用。部分测试依赖 GPU、网络或其他运行条件，不能把“测试存在”视为“本机已验证通过”。

### 2.7 `dev/`：研究材料与辅助工具

| 文件 | 作用 |
| --- | --- |
| `LEADERBOARD.md` | 速度排行榜的解释与贡献说明。 |
| `LOG.md` | 开发与实验记录。 |
| `scaling_analysis.ipynb` | 缩放规律分析 Notebook。 |
| `estimate_gpt3_core.ipynb` | GPT-3 CORE 分数估算相关 Notebook。 |
| `repackage_data_reference.py` | 预训练数据重打包参考脚本。 |
| `nanochat.png`、`scaling_laws_jan26.png` | README 使用的项目图片和实验结果图。 |

## 三、依赖包与运行环境

依赖声明以 [pyproject.toml](pyproject.toml) 为准，解析版本以 `uv.lock` 为准。下面区分直接依赖、开发依赖和源码中额外使用的包。

### 3.1 直接运行依赖

| 包 | 当前声明 | 主要用途 |
| --- | --- | --- |
| `torch` | `==2.9.1` | 张量计算、神经网络、自动微分、优化、GPU 与分布式计算；整个模型实现的基础。 |
| `numpy` | `>=1.26.0` | 数值数组与任务数据处理。 |
| `pyarrow` | `>=21.0.0` | 读取和处理 Parquet/Arrow 格式的数据。 |
| `rustbpe` | `>=0.1.0` | 训练 BPE 分词器。 |
| `tiktoken` | `>=0.11.0` | 分词编码、解码与词表运行时表示。 |
| `kernels` | `>=0.11.7` | 获取适用硬件的预编译计算内核，用于 Flash Attention 3 集成。 |
| `wandb` | `>=0.21.3` | 实验指标记录；训练入口可通过 `--run=dummy` 使用项目的空日志实现。 |
| `filelock` | `>=3.19.0` | 下载等操作的文件锁，避免并发冲突。 |
| `psutil` | `>=7.1.0` | 系统与进程资源查询工具；当前主流程未发现直接调用，不必优先研究。 |

### 3.2 开发依赖

`dev` 依赖组包含：

| 包 | 当前声明 | 用途 |
| --- | --- | --- |
| `pytest` | `>=8.0.0` | 运行测试。 |
| `ipykernel` | `>=7.1.0` | 在 Notebook 环境运行 Python。 |
| `matplotlib` | `>=3.10.8` | 绘图与实验分析。 |
| `python-dotenv` | `>=1.2.1` | 加载 `.env` 环境变量文件的工具；并不表示所有训练入口都会自动读取 `.env`。 |

### 3.3 间接依赖与辅助脚本依赖

源码还直接导入了部分未在顶层依赖列表中单独声明的包：

- `requests`：`nanochat/dataset.py` 用于 HTTP 下载。
- `PyYAML`（导入名 `yaml`）：`scripts/base_eval.py` 用于解析评估配置。
- `Jinja2`（导入名 `jinja2`）：`nanochat/core_eval.py` 用于任务模板渲染。

这些包由当前依赖链间接带入。它们与项目的直接依赖不是同一个概念，精简环境或替换依赖时应注意。

`dev/repackage_data_reference.py` 还使用 Hugging Face 的 `datasets`，但当前 `pyproject.toml` 未声明它。该文件是辅助参考脚本，不能假设默认安装后就能运行；主任务加载逻辑使用的是 `tasks/common.py` 中的实现。

### 3.4 Python、uv 与 CPU/GPU 环境

项目要求 Python `>=3.10`，使用 `uv` 管理依赖。以下为项目声明的安装方式，并不表示已在当前机器上执行或验证：

```bash
# CPU 环境，包含测试和 Notebook 等开发工具
uv sync --extra cpu --group dev
source .venv/bin/activate
```

```bash
# NVIDIA CUDA 环境，包含开发工具；与上面的 CPU 环境二选一
uv sync --extra gpu --group dev
source .venv/bin/activate
```

说明：

- `uv` 是环境与依赖管理工具，不是模型算法依赖；上述命令假设已安装 uv。
- CPU 与 GPU extra 在配置中互斥，不要同时选择。
- GPU extra 使用 PyTorch CUDA 12.8 软件源，但这不意味着每次都需要另外安装完整的本地 CUDA Toolkit；仍需兼容的 GPU 和驱动。
- `torchrun` 来自 PyTorch，用于多进程/多 GPU 启动，不是单独的 Python 依赖包。
- FA3 只在适用条件下启用；不能使用时会回退到 PyTorch SDPA，不需要为了入门先解决所有加速内核问题。
- 当前默认计算精度由 `nanochat/common.py` 根据设备选择，也可通过 `NANOCHAT_DTYPE` 覆盖。先用默认值，不必在第一轮学习时研究 FP8。

### 3.5 数据、模型和 WSL 注意事项

- 中间产物默认放在 `~/.cache/nanochat`，可用 `NANOCHAT_BASE_DIR` 修改；数据和检查点不一定出现在仓库目录中。
- 在 WSL 中，GPU 是否可用取决于 Windows 宿主机驱动和 WSL GPU 支持，不能只看是否安装了 GPU 版 PyTorch。
- 当前仓库位于 `/mnt/d/`；遇到大量小文件访问缓慢时，应考虑 Windows 挂载文件系统与 WSL 原生文件系统的性能差异，不要先归因于模型算法。
- 数据、依赖和计算内核可能需要联网下载；下载失败应先区分网络问题、权限问题与程序错误。
- 完整训练会占用磁盘、内存、显存和计算时间。README 的多 H100 训练时间、费用和性能不能直接套用到本机。

## 四、推荐学习路径

总体顺序：

```text
基础知识 → 分词 → 训练样本 → Transformer → 损失与训练
        → 自回归生成 → 对话微调 → KV Cache → 强化学习与性能优化
```

每个阶段都应能用自己的话解释输入、输出及其因果关系，不以“脚本成功启动”代替理解。

### 阶段 0：补齐最少的前置知识

需要掌握：

- Python：函数、类、列表、生成器和模块导入。
- 张量：形状、索引、广播、矩阵乘法、`view`/`reshape`。
- 神经网络：参数、前向传播、损失、梯度、反向传播和学习率。
- 概率基础：softmax、对数概率和交叉熵。

不需要先学完所有深度学习理论。若能解释一个小型神经网络如何通过损失更新参数，就可以开始阅读项目。

### 阶段 1：文本如何变成 token

**阅读入口：**

1. [nanochat/tokenizer.py](nanochat/tokenizer.py)：`RustBPETokenizer` 的 `encode()`、`decode()`、`train_from_iterator()`。
2. [scripts/tok_train.py](scripts/tok_train.py)：分词器训练入口。
3. [tests/test_tokenizer.py](tests/test_tokenizer.py)：编码解码与格式示例。

**要理解的问题：**

- token 为什么不等于一个汉字或一个英文单词？
- 文本、UTF-8 字节、token ID、Embedding 向量分别是什么？
- 词表大小如何影响序列长度和模型输出维度？
- BOS、角色边界、回答结束等特殊 token 有什么作用？

**建议实验：** 对短中文、英文、数字和代码进行编码与解码，观察切分以及长度变化。使用项目训练的分词器前，需要已有分词器产物。

**完成标准：** 能解释“模型接收的是 token ID，不是原始字符串；Embedding 再把 ID 转成向量”。

**边界：** 项目把 BPE 训练委托给 `rustbpe`。读懂这里的封装不等于掌握 BPE 的内部合并算法，后者需要单独学习。

### 阶段 2：预训练样本如何构造

**阅读入口：** [nanochat/dataloader.py](nanochat/dataloader.py)，重点看生成 `inputs`、`targets` 的部分；数据来源再按需看 `dataset.py`。

核心是把同一序列错开一位：

```text
完整 token 序列：[A, B, C, D]
inputs：         [A, B, C]
targets：        [B, C, D]
```

配合因果注意力：

- 位置 A 根据 A 预测 B。
- 位置 B 根据 A、B 预测 C。
- 位置 C 根据 A、B、C 预测 D。

**要理解的问题：**

- 为什么不需要人为为每个位置标注答案？
- 为什么训练时不能看未来 token？
- `B`（批大小）与 `T`（序列长度）分别代表什么？
- 文档拼接、裁剪与序列长度有什么关系？

**建议实验：** 手工构造上述错位序列，确认输入与目标的形状都是 `[B, T]`。第一遍不必研究 best-fit 文档选择算法。

**完成标准：** 能解释自监督训练的数据来源，以及目标错位和因果掩码为什么缺一不可。

### 阶段 3：Transformer 如何完成前向计算

**阅读入口：** [nanochat/gpt.py](nanochat/gpt.py)。建议顺序：

1. `GPTConfig`：模型规模和维度。
2. `GPT.forward()`：整体数据流。
3. `Block.forward()`：残差、Attention 与 MLP 的关系。
4. `CausalSelfAttention.forward()`：Q/K/V、多头与因果注意力。
5. `MLP.forward()`：逐位置的非线性变换。
6. `apply_rotary_emb()` 和 `norm()`：第二遍再深入位置编码与归一化。

先跟踪主干：

```text
输入 token ID [B, T]
  → Embedding [B, T, C]
  → 多层 Transformer Block [B, T, C]
  → 最终归一化
  → 输出投影与 logits 处理 [B, T, V]
```

其中，`C` 是隐藏维度，`V` 是词表大小。logits 是分数，不是已经归一化的概率。

标准缩放点积注意力的核心可以先理解为：

```text
Attention(Q, K, V) = softmax(QKᵀ / √d + mask) V
```

本项目在这一主干上加入了 RoPE、QK Norm、额外缩放和窗口策略，不能把上式视为当前实现的全部细节。

**要理解的问题：**

- Embedding 和最终词表投影分别在做什么？
- Q、K、V 来自哪里，注意力权重在哪个维度归一化？
- 多头注意力如何拆分、合并维度？
- Attention 与 MLP 的分工是什么？
- 残差连接为什么保留输入再叠加新结果？

**建议实验：** 跟踪一个小输入经过各层后的形状，手算一个极小的注意力例子，确认未来位置的权重被屏蔽。

**完成标准：** 能从 token ID 开始，完整讲清楚得到下一个 token 分数的过程。

**第一遍暂时跳过：** Value Embedding、`resid_lambdas`、`x0_lambdas`、`smear`、`backout`、logit softcap 等额外机制。先在概念上分离主干，不必立即删改代码。

### 阶段 4：损失如何让模型学会预测

**阅读入口：**

- [nanochat/gpt.py](nanochat/gpt.py)：`GPT.forward()` 中的交叉熵。
- [scripts/base_train.py](scripts/base_train.py)：训练循环中的前向、反向、优化器更新与清空梯度。

先把训练循环还原成：

```text
准备 inputs、targets
  → 模型预测并计算 loss
  → loss.backward()
  → optimizer.step()
  → 清空梯度
  → 下一批数据
```

**要理解的问题：**

- 交叉熵为什么鼓励正确 token 获得更高概率？
- `backward()` 与 `optimizer.step()` 有什么区别？
- 梯度为什么要清空？梯度累积时为什么缩放 loss？
- 学习率、批大小、训练步数与训练效果有什么关系？
- 训练损失下降为什么不一定代表泛化变好？

**建议实验：** 在后续具备环境与数据条件时，用很小的模型和少量固定文本观察过拟合：训练损失下降，模型更容易复现训练内容。目标是验证学习机制，而不是训练聊天助手。

**完成标准：** 能解释一次参数更新的完整过程，并区分训练与验证。

**暂时跳过：** Muon 的数学细节、多卡通信和自动超参数缩放。先理解“梯度如何用于更新参数”。

### 阶段 5：模型如何生成文本

**阅读入口：** [nanochat/gpt.py](nanochat/gpt.py) 的 `GPT.generate()`；它比完整推理引擎更适合第一次阅读。

```text
已有 token → 前向计算 → 取最后位置的 logits
  → 选择新 token → 追加到已有 token → 重复
```

**要理解的问题：**

- 为什么只用最后一个位置的输出预测紧接着的 token？
- 为什么训练时可以并行计算多个位置，普通自回归生成却有先后依赖？
- temperature、top-k、随机种子分别影响什么？
- 为什么贪心选择、随机采样会得到不同结果？

**建议实验：** 对同一个提示比较 temperature 为 0 和不同正值时的结果，再改变 top-k 与随机种子。应使用已有模型，并注意输入和长度限制。

**完成标准：** 能解释“生成是反复预测下一个 token”，以及“普通推理不通过反向传播更新模型权重”。

### 阶段 6：从基础模型到聊天助手

**阅读入口：**

1. [nanochat/tokenizer.py](nanochat/tokenizer.py)：`render_conversation()`。
2. [scripts/chat_sft.py](scripts/chat_sft.py)：数据混合、目标掩码和训练循环。
3. [tasks/smoltalk.py](tasks/smoltalk.py)：对话数据入口。
4. [scripts/chat_cli.py](scripts/chat_cli.py)：聊天使用方式。

SFT 仍然使用下一个 token 预测和交叉熵，主要变化是数据格式、监督位置和训练配置，不是换成了另一个模型原理。

**要理解的问题：**

- 基础模型为什么擅长续写，却未必遵循指令？
- user、assistant 的角色边界如何编码到序列中？
- 为什么用户提示通常不直接计入目标损失？
- 为什么工具返回结果可以作为上下文，却不一定作为模型需要学习输出的内容？
- `mask=0` 为什么不等于“模型完全看不到这段输入”？

项目把不监督的位置对应的目标设为 `-1`，模型交叉熵通过 `ignore_index=-1` 忽略这些目标；这些位置仍可作为上下文。

**建议实验：** 对一条简短对话列出 token 和 mask，辨认哪些位置参与损失。细节以 `render_conversation()` 为准，并非所有特殊 token 都被忽略，例如回答结束标记也参与监督。

**完成标准：** 能解释预训练与 SFT 在目标上的共性，以及在数据组织和监督范围上的区别。

### 阶段 7：KV Cache 与更高效的推理

**阅读入口：**

- [nanochat/engine.py](nanochat/engine.py)：`KVCache`、`sample_next_token()`、`Engine.generate()`。
- [nanochat/gpt.py](nanochat/gpt.py)：注意力的缓存分支。
- [tests/test_engine.py](tests/test_engine.py)：缓存与采样的行为示例。

**要理解的问题：**

- 历史 token 的哪些计算可以复用？为什么缓存 K/V？
- prefill 与 decode 有什么区别？
- KV Cache 为什么能减少重复计算，又为什么占用显存？
- 上下文长度、层数、KV 头数和数值精度如何影响缓存大小？
- 注意力中“能访问的上下文”和缓存的存储策略是否总是一回事？

**建议实验：** 先阅读测试中的缓存位置和形状断言，再比较朴素生成与带缓存生成的数据流。

**完成标准：** 能解释 KV Cache 是推理计算复用机制，不是新知识库，也不是训练得到的长期记忆。

### 阶段 8：强化学习、评估与性能优化

基础主线理解之后，再按兴趣选择，不必一次学完。

#### 8.1 强化学习

阅读 [scripts/chat_rl.py](scripts/chat_rl.py) 和 [tasks/gsm8k.py](tasks/gsm8k.py)，关注：

- 同一问题如何采样多个回答。
- 回答如何得到奖励。
- 奖励如何转换为 advantage，并影响生成 token 的概率。
- SFT 的示范信号与强化学习的奖励信号有什么区别。

**算法边界：** 文件开头明确说明这里是简化的、接近 REINFORCE 的实现，没有参考模型 KL 正则，也没有 PPO ratio/clip，并对 advantage 和损失归一化作了特定选择。不能把它当成完整 PPO、标准 GRPO 或完整 RLHF 流程。

#### 8.2 评估

阅读 `loss_eval.py`、`core_eval.py`、`scripts/base_eval.py` 和 `scripts/chat_eval.py`，理解：

- 训练损失、验证 BPB 和下游任务分数衡量的是不同方面。
- 不同分词器的 token 级损失不适合不加处理地直接比较，BPB 提供另一种比较口径。
- 模型能流畅生成，并不意味着回答正确。
- 实验必须关注训练/评估数据划分与数据污染。

#### 8.3 性能与规模

最后再读：

- `flash_attention.py`：注意力计算与回退。
- `fp8.py`：低精度计算。
- `optim.py`：Muon/AdamW 与分布式优化。
- `scripts/infer_bench.py`：推理测量。
- `runs/scaling_laws.sh`、`runs/miniseries.sh` 和 `dev/scaling_analysis.ipynb`：规模与计算预算的关系。

**完成标准：** 能区分哪些机制改变模型计算或训练方法，哪些主要改变计算成本与执行效率。

## 五、实践顺序与资源边界

### 5.1 第一轮只建立最小闭环

建议先完成这五件事：

1. 观察一段短文本的 token 编码与解码。
2. 手工构造错开一位的输入和目标。
3. 跟踪模型前向传播的张量形状。
4. 理解一次 loss、反向传播与参数更新。
5. 理解自回归采样，并观察生成参数的影响。

先做到“每一步为什么这样做”，再追求“训练出来的模型有多强”。

### 5.2 不要直接照搬完整训练预算

- `runs/speedrun.sh` 面向多 GPU 节点，不适合作为普通电脑上的入门启动命令。
- `runs/runcpu.sh` 虽然是缩小示例，仍包含下载数据、在约 20 亿字符上训练分词器，以及数千步模型训练；它不是几秒钟的冒烟测试。
- 脚本中的耗时说明来自特定硬件，不能作为当前 WSL 机器的耗时承诺。
- 运行前先确认 CPU/GPU、内存、显存、磁盘空间和网络，再决定序列长度、批大小与训练步数。缩小实验时也要满足脚本对批大小等参数的约束。
- 数据准备、分词器训练、模型预训练和 SFT 具有前后依赖关系，不要把任一入口都当成可独立运行的演示。

本文不提供未经本机验证的完整训练配方。确认硬件后再制定小规模实验参数，更容易控制成本和定位问题。

## 六、学习进度检查

- [ ] 能区分文本、token ID、Embedding 与 logits。
- [ ] 能手工构造 next-token 输入和目标。
- [ ] 能解释因果掩码为什么防止看到未来。
- [ ] 能追踪 `[B, T] → [B, T, C] → [B, T, V]`。
- [ ] 能解释 Q/K/V、多头注意力、MLP 与残差连接。
- [ ] 能解释交叉熵、反向传播与参数更新之间的关系。
- [ ] 能解释训练并行预测与自回归生成的区别。
- [ ] 能解释 temperature、top-k 与随机采样。
- [ ] 能解释预训练与 SFT 的共性和差异。
- [ ] 能区分损失掩码与注意力可见性。
- [ ] 能解释 KV Cache、prefill 与 decode。
- [ ] 能区分当前强化学习示例与完整 RLHF 流程。
- [ ] 能区分语言建模指标、任务能力与推理性能。

**最推荐的第一段代码：`nanochat/gpt.py` 的 `GPT.forward()`。** 以它为主线向下理解 Transformer，向前连接分词和数据，向后连接损失、训练与生成，比按目录逐文件通读更高效。
