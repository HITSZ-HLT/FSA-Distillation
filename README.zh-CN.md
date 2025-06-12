[🇨🇳 中文](README.zh-CN.md) | [🇺🇸 English](README.md)

# 从大型语言模型中蒸馏细粒度情感理解

本仓库包含论文 [Distilling Fine-grained Sentiment Understanding from Large Language Models](https://arxiv.org/abs/2412.18552) 的代码与数据。该方法旨在将大型语言模型（LLMs）中的细粒度情感理解能力蒸馏到小型语言模型（SLMs）中。

## 摘要

细粒度情感分析（FSA）旨在从大量主观文本中提取并总结用户的观点。近期研究表明，大型语言模型（LLMs）在情感理解方面表现出色。然而，直接使用 LLM 进行 FSA 应用成本高昂。因此，本文研究如何将 LLM 的细粒度情感理解能力蒸馏到 SLM 中。我们通过提示 LLM 分析评论的情感并生成解释文本，随后使用这些内容对 SLM 进行预训练。此外，我们还构建了一个全面的 FSA 基准，用于评估 SLM 与 LLM 的性能。大量实验结果表明：（1）蒸馏显著提升了 SLM 在 FSA 任务中的表现，F1 分数提高了 6.00%，蒸馏后的小模型（仅 2.2 亿参数）甚至超过了 Llama-2-7b；（2）蒸馏赋予 SLM 出色的零样本情感分类能力，其性能可与甚至超过教师模型。这些结果表明，基于 LLM 蒸馏的 FSA 方法前景广阔。

## 项目结构

```
.
├── README.md
├── evaluation              # 评估代码
│   ├── acsa.py
│   ├── atsa.py
│   ├── bash
│   ├── output
│   └── utils
├── fsa_datasets            # FSA 数据集
│   └── w_hard
├── parse_performance.ipynb # 结果分析脚本
├── pre-training            # 蒸馏相关代码
│   ├── bash
│   ├── output_model
│   ├── seq2seq.py
│   └── utils
├── pretrained_models       # 预训练模型
├── prompting               # 情感理解语料
│   ├── data
│   └── test
└── requirements.txt        # 环境依赖
```

## 使用方法

### 前置条件

#### 1. 下载情感理解语料

从 HuggingFace 下载 [Gporrt/sentiment-understanding-corpus](https://huggingface.co/datasets/Gporrt/sentiment-understanding-corpus) 并放置到 `./prompting/data` 目录下。

#### 2. 下载 T5-Base 模型

从 HuggingFace 下载 [google-t5/t5-base](https://huggingface.co/google-t5/t5-base) 模型，并放置到 `./pretrained_models` 目录下。

### 训练与评估

#### 3. 运行蒸馏脚本

进入 `evaluation` 目录，执行以下命令：

```bash
cd ./evaluation

# 使用蒸馏语料进行预训练
bash/pt_eval/v7.9.sh -c ${CUDA_IDS}
```

**注意**：在运行之前，请修改 `evaluation/bash/model_name.json` 文件中的模型版本与路径映射。

#### 4. 微调并评估蒸馏模型

运行以下命令进行多种随机种子的并行微调：

```bash
cd ./evaluation
# 使用并行处理微调 T5 模型（n = 并行进程数）
bash/atsa_batch_parallel.sh -c ${CUDA_IDS} -b ${model_version} -n 3
bash/acsa_batch_parallel.sh -c ${CUDA_IDS} -b ${model_version} -n 2
```

### 结果分析

#### 5. 解析模型性能结果

使用 `./parse_performance.ipynb` Notebook 进行结果分析。请务必：

* 替换 `path` 变量为实际路径
* 替换 `version` 变量为你的模型版本

## 预训练模型

我们发布了以下蒸馏后的预训练模型：

| 模型名称               | 基础模型     | 下载链接                                                                |
| ------------------ | -------- | ------------------------------------------------------------------- |
| t5-sentiment-base  | t5-base  | [HuggingFace](https://huggingface.co/zhang-yice/t5-sentiment-base)  |
| t5-sentiment-large | t5-large | [HuggingFace](https://huggingface.co/zhang-yice/t5-sentiment-large) |

## 引用

如果你使用了本项目的框架或数据，请引用如下文献：

```bibtex
@misc{zhang2024distillingfinegrainedsentimentunderstanding, 
      title={Distilling Fine-grained Sentiment Understanding from Large Language Models}, 
      author={Yice Zhang and Guangyu Xie and Hongling Xu and Kaiheng Hou and Jianzhu Bao and Qianlong Wang and Shiwei Chen and Ruifeng Xu}, 
      year={2024}, 
      eprint={2412.18552}, 
      archivePrefix={arXiv}, 
      primaryClass={cs.CL}, 
      url={https://arxiv.org/abs/2412.18552}, 
}
```
