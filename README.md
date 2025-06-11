# Distilling Fine-grained Sentiment Understanding from Large Language Models

This repository contains code and data for the [paper](https://arxiv.org/abs/2412.18552), a method for distilling of fine-grained sentiment understanding from LLMs into small language models (SLMs).

## Abstract

Fine-grained sentiment analysis (FSA) aims to extract and summarize user opinions from vast opinionated text. Recent studies demonstrate that large language models (LLMs) possess exceptional sentiment understanding capabilities. However, directly deploying LLMs for FSA applications incurs high inference costs. Therefore, this paper investigates the distillation of fine-grained sentiment understanding from LLMs into small language models (SLMs). We prompt LLMs to examine and interpret the sentiments of given reviews and then utilize the generated content to pretrain SLMs. Additionally, we develop a comprehensive FSA benchmark to evaluate both SLMs and LLMs. Extensive experiments on this benchmark reveal that: (1) distillation significantly enhances the performance of SLMs in FSA tasks, achieving a 6.00% improvement in F1-score, and the distilled model can outperform Llama-2-7b with only 220M parameters; (2) distillation equips SLMs with excellent zero-shot sentiment classification capabilities, enabling them to match or even exceed their teacher models. These results suggest that distillation from LLMs is a highly promising direction for FSA.

## Project Structure
```
.
├── README.md
├── evaluation # evaluation code
│   ├── acsa_5m23d.py
│   ├── atsa_5m23d.py
│   ├── bash
│   ├── output
│   └── utils
├── fsa_datasets # fsa datasets
│   └── w_hard_v2
├── parse_performance.ipynb
├── pre-training # distillation code
│   ├── bash
│   ├── output_model
│   ├── seq2seq.py
│   └── utils
├── pretrained_models # base model
└── prompting # sentiment corpus
    ├── data
    └── test

```


## Usage

### Prerequisites

#### 1. Download the Sentiment Understanding Corpus

Download the `Gporrt/sentiment-understanding-corpus` corpus from HuggingFace and place it in the `./prompting/data` directory.

#### 2. Download the T5-Base Model

Download the `google-t5/t5-base` model and place it in the `./pretrained_models` directory.

### Training and Evaluation

#### 3. Run Distillation Scripts

Navigate to the evaluation directory and execute the following commands:

```bash
cd ./evaluation

# Run pretraining with distillation corpus
bash/pt_eval/v7.9.sh -c ${CUDA_IDS}
```

**Important**: Before proceeding, update the model version and path mappings in `evaluation/bash/model_name.json`.

#### 4. Fine-tune and Evaluate the Distilled Model

Run the following commands for parallel multi-seed fine-tuning:

```bash
cd ./evaluation
# Fine-tune T5 with parallel processing (n = number of parallel processes)
bash/atsa_batch_parallel.sh -c ${CUDA_IDS} -b ${model_version} -n 3
bash/acsa_batch_parallel.sh -c ${CUDA_IDS} -b ${model_version} -n 2
```

### Results Analysis

#### 5. Parse Performance Results

Use the `./parse_performance.ipynb` notebook to analyze the results. Make sure to:

- Replace the `path` variable with your actual path
- Replace the `version` variable with your model version

