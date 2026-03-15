# ============================================
# 推荐数据集指南
# ============================================

本文档汇总了各训练阶段推荐使用的开源数据集。

---

## 一、LLM 预训练数据集

### 1. 中文通用语料

| 数据集 | 规模 | 特点 | 下载地址 |
|--------|------|------|----------|
| **WuDaoCorpora (悟道)** | 200G (开源) / 5TB (完整) | 中文最大开源语料，覆盖百科、新闻、论坛等50+领域，经20+规则清洗 | https://data.baai.ac.cn/details/WuDaoCorporaText |
| **SkyPile-150B (书生·万卷)** | 150B tokens | 高质量中文网页、书籍、论文，深度清洗 | https://opendatalab.com/ |
| **MNBVC** | 26TB+ | 超大规模中文语料，持续更新，涵盖书籍、网页、对话等 | https://github.com/esbatmop/MNBVC |
| **CLUECorpus2020** | 100GB | 高质量中文预训练语料，需申请 | https://www.cluebenchmarks.com/dataSet_search.html |
| **Firefly-LLaMA2-Chinese** | 22GB | 整合CLUE、维基百科等，含古诗词等特色语料 | 魔搭社区 / HyperAI |
| **CCI (中文互联网语料库)** | 100GB+ | 安全、高质量中文网页数据 | https://data.baai.ac.cn/ |

### 2. 领域专用语料

| 数据集 | 领域 | 说明 |
|--------|------|------|
| **Chinese-Fineweb-Edu-v2** | 教育 | 188M条教育领域文本，约420B tokens |
| **Wanjuan 1.0 (万卷)** | 综合 | 书生·万卷，多领域高质量数据 |
| **LeetCode-Chinese** | 代码 | 编程题解数据 |
| **CNKI 论文摘要** | 学术 | 中文学术论文 (需授权) |

### 3. 小规模测试数据（推荐用于实验）

```bash
# Firefly 预训练数据（22GB，适合实验）
https://huggingface.co/datasets/YeungNLP/firefly-pretrain-dataset

# MNBVC 子集（可按需下载部分）
https://github.com/esbatmop/MNBVC

# WuDao 200G 开源版
https://openi.pcl.ac.cn/BAAI/WuDao-Data
```

---

## 二、LLM SFT 指令微调数据集

### 1. 中文指令数据集

| 数据集 | 规模 | 特点 | 下载地址 |
|--------|------|------|----------|
| **BELLE-train-0.5M** | 500K | 链家开源，多领域指令 | https://huggingface.co/datasets/BelleGroup/train_0.5M_CN |
| **BELLE-train-1M** | 1M | 去重后质量更高 | https://huggingface.co/datasets/BelleGroup/generated_train_1M_CN |
| **BELLE-multiturn-chat** | 800K | 多轮对话数据 | https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M |
| **Alpaca-Chinese** | 52K | 斯坦福Alpaca中文翻译版 | https://github.com/hikariming/alpaca_chinese_dataset |
| **Firefly-train-1.1M** | 1.1M | 23种NLP任务，指令模板丰富 | https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M |
| **COIG** | 191K | 安全、多样、多轮的中文指令 | https://huggingface.co/datasets/BAAI/COIG |
| **COIG-PC** | 321M | 大规模中文指令语料 | https://huggingface.co/datasets/BAAI/COIG-PC |
| **pCLUE** | 1.2M | 中文提示学习数据集 | https://github.com/CLUEbenchmark/pCLUE |
| **MOSS-sft-data** | 1.1M | 复旦MOSS项目，多轮对话 | https://github.com/OpenLMLab/MOSS |
| **GuanacoDataset** | 多语言 | 多语言指令数据（含中文） | https://huggingface.co/datasets/JosephusCheung/GuanacoDataset |

### 2. 英文指令数据集

| 数据集 | 规模 | 说明 |
|--------|------|------|
| **Alpaca** | 52K | 斯坦福经典指令数据集 |
| **ShareGPT** | 90K+ | 真实用户对话 (需清洗) |
| **UltraChat** | 1.5M | 多轮对话，质量高 |
| **OpenOrca** | 5M+ | GPT-4 合成指令 |
| **Databricks-dolly-15k** | 15K | 人工标注指令 |

---

## 三、偏好对齐数据集 (DPO/RLHF)

### 1. 开源偏好数据集

| 数据集 | 规模 | 语言 | 说明 |
|--------|------|------|------|
| **mlabonne/orpo-dpo-mix-40k** | 40K | 多语言 | DPO/ORPO 混合数据集 |
| **Chinese-dpo-pairs** | 10K | 中文 | 精心整理的中文偏好对 |
| **hh-rlhf** | 160K | 英文 | Anthropic RLHF 数据集 |
| **SHP (Stanford Human Preferences)** | 380K | 英文 | 多领域偏好数据 |
| **OpenAssistant/oasst1** | 16K | 多语言 | OA 项目偏好标注 |
| **UltraFeedback** | 340K | 多语言 | 大规模反馈数据集 |

### 2. 自制偏好数据方法

```python
# 方法1: 从 SFT 数据构造简单偏好对
# 使用 SFT 数据作为 chosen，使用较弱模型生成 rejected

# 方法2: 使用 GPT-4 生成对比
# 让 GPT-4 生成好回答，让较小模型生成差回答

# 方法3: 人工标注
# 使用标注工具（如 Argilla）进行人工标注
```

---

## 四、VLM 视觉-语言数据集

### 1. 图文对齐数据集 (预训练)

| 数据集 | 规模 | 语言 | 说明 | 下载 |
|--------|------|------|------|------|
| **LAION-5B** | 5.85B | 多语言 | 最大开源图文数据集，含23.2亿英文对 | https://laion.ai/blog/laion-5b/ |
| **LAION-COCO** | 600M | 英文 | LAION子集，BLIP生成高质量描述 | https://laion.ai/blog/laion-coco/ |
| **CC12M** | 12M | 英文 | Conceptual Captions 12M | https://google.com/research/cc12m |
| **CC3M** | 3.3M | 英文 | Conceptual Captions 3M | https://google.com/research/cc3m |
| **COYO-700M** | 700M | 多语言 | 高质量图文对 | https://github.com/kakaobrain/coyo-dataset |
| **Wukong (悟空)** | 100M | 中文 | 大规模中文图文数据集 | https://wukong-dataset.github.io/ |
| **Chinese-CLIP-Dataset** | 2B | 中文 | 中文图文对 | https://github.com/OFA-Sys/Chinese-CLIP |

### 2. 视觉指令微调数据集

| 数据集 | 规模 | 任务类型 | 说明 |
|--------|------|----------|------|
| **LLaVA-Instruct-150K** | 150K | 多轮对话 | 视觉指令跟随 |
| **ShareGPT4V** | 100K+ | 多轮对话 | GPT-4V 生成 |
| **SVIT (多轮对话)** | 4.2M | 多轮对话 | 中文视觉对话 |
| **Visual-Genome** | 108K | 区域描述 | 区域级图文对齐 |
| **VQA-v2** | 1.1M | 视觉问答 | 经典VQA数据集 |
| **COCO-Caption** | 600K | 图像描述 | COCO 图像描述 |
| **Flickr30k** | 150K | 图像描述 | Flickr 图像描述 |
| **RefCOCO** | 20K | 指代表达 | 区域定位任务 |
| **OCR-VQA** | 200K | OCR问答 | 文档理解 |
| **TextVQA** | 45K | 文本VQA | 图中文字问答 |
| **ChartQA** | 30K | 图表问答 | 图表理解 |

### 3. 中文VLM数据集

| 数据集 | 规模 | 说明 |
|--------|------|------|
| **MMBench-Chinese** | - | 中文多模态评测 |
| **CMB (Chinese Medical Benchmark)** | - | 中文医学多模态 |
| **COCO-CN** | 20K | COCO 中文描述 |

---

## 五、数据集快速下载命令

### 使用 HuggingFace datasets

```python
# LLM 预训练数据
from datasets import load_dataset
dataset = load_dataset("YeungNLP/firefly-pretrain-dataset", split="train")

# SFT 数据
dataset = load_dataset("BelleGroup/train_0.5M_CN", split="train")

# DPO 数据
dataset = load_dataset("mlabonne/orpo-dpo-mix-40k", split="train")

# VLM 图文数据（流式加载）
dataset = load_dataset("laion/laion400m", split="train", streaming=True)
```

### 使用 ModelScope（国内更快）

```python
from modelscope.msdatasets import MsDataset

# 下载 WuDao 数据
dataset = MsDataset.load('wudao', subset_name='200G')

# 下载 BELLE 数据
dataset = MsDataset.load('belle-group/train_0.5M_CN')
```

### 使用 Git LFS

```bash
# 克隆数据集仓库
git lfs install
git clone https://huggingface.co/datasets/BelleGroup/train_0.5M_CN
```

---

## 六、数据集选择建议

### 实验阶段（小模型/少量GPU）

| 阶段 | 推荐数据 | 数据量 |
|------|----------|--------|
| 预训练 | Firefly-Pretrain 子集 | 1-5GB |
| SFT | Alpaca-Chinese | 52K |
| DPO | Chinese-dpo-pairs | 10K |
| VLM对齐 | COCO-Caption | 600K |
| VLM-SFT | LLaVA-Instruct-150K 子集 | 50K |

### 生产阶段（大模型/多机多卡）

| 阶段 | 推荐数据 | 数据量 |
|------|----------|--------|
| 预训练 | WuDao + SkyPile | 200GB+ |
| SFT | BELLE-1M + COIG + Firefly | 3M+ |
| DPO | UltraFeedback + 自制数据 | 300K+ |
| VLM对齐 | LAION-5B 子集 (100M+) | 100M+ |
| VLM-SFT | ShareGPT4V + SVIT + 多任务数据 | 1M+ |

---

## 七、数据质量检查清单

- [ ] 文本长度分布合理（去除过短/过长样本）
- [ ] 语言分布符合需求（中英文比例）
- [ ] 去重处理（避免重复样本）
- [ ] 敏感信息过滤（隐私、有害内容）
- [ ] 格式统一（统一编码、换行符等）
- [ ] 领域平衡（各领域比例适当）
