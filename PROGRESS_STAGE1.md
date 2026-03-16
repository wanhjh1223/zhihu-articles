# LLM-VLM 训练框架 - 阶段一：LLM 预训练（已完成）

## 完成进度

### ✅ 已完成内容

#### 1. 数据集调研与推荐
- **文档**: `docs/recommended_datasets.md`
- 整理了各阶段推荐数据集，包括：
  - 预训练：WuDaoCorpora (200G/5TB)、SkyPile-150B、Firefly-LLaMA2-Chinese (22GB)
  - SFT：BELLE、Alpaca-Chinese、Firefly-train-1.1M、COIG
  - DPO：mlabonne/orpo-dpo-mix-40k、Chinese-dpo-pairs
  - VLM：LAION-5B、CC12M、Wukong (中文)

#### 2. LLM 预训练数据加载器
- **文件**: `src/llm_training/data/pretrain_dataloader.py`
- 特性：
  - 流式读取大文件（内存友好）
  - 支持样本拼接（充分利用序列长度）
  - 自动过滤低质量文本
  - 支持 HuggingFace datasets 流式加载
  - 分布式训练支持

#### 3. LLM 预训练器
- **文件**: `src/llm_training/training/pretrain_trainer.py`
- 特性：
  - 基于 Accelerate 的分布式训练
  - 支持 BF16/FP16 混合精度
  - 梯度累积和梯度裁剪
  - Flash Attention 支持
  - 学习率 Warmup + Cosine 衰减
  - Tensorboard/Wandb 日志
  - 自动保存检查点

#### 4. 训练脚本
- **主脚本**: `scripts/llm/pretrain/run.sh`
  - 支持命令行参数
  - 支持配置文件
  - 详细的帮助信息

- **测试脚本**: `scripts/llm/pretrain/test.sh`
  - 完整的功能测试
  - 数据加载测试
  - 训练流程测试
  - 模型推理测试

- **全流程脚本**: `scripts/llm/pipeline.sh`
  - 串联预训练 -> SFT -> DPO

#### 5. 配置文件
- **文件**: `configs/llm/pretrain/config.yaml`
- 包含完整的训练超参数配置

#### 6. 数据格式文档
- **文件**: `docs/llm_data_formats.md`
- 详细的预训练/SFT/DPO 数据格式说明
- 数据转换脚本示例

#### 7. 开发日志
- **文件**: `docs/development-log.md`
- 开发思路与技术选型
- 完整的参考资料链接
- 遇到的问题及解决方案
- 测试结果数据

---

## 快速开始

### 1. 安装依赖

```bash
cd /root/.openclaw/workspace/projects/llm-vlm-framework
pip install -r requirements.txt
```

### 2. 运行测试（验证环境）

```bash
# 运行完整的功能测试
bash scripts/llm/pretrain/test.sh
```

测试内容包括：
- 创建测试数据
- 测试数据加载器
- 小规模训练（使用 gpt2 模型）
- 验证模型输出
- 测试模型推理

### 3. 使用真实数据训练

#### 方式1：使用 HuggingFace 数据集（推荐新手）

```bash
python -m src.llm_training.training.pretrain_trainer \
    --model Qwen/Qwen2.5-0.5B \
    --train_data "YeungNLP/firefly-pretrain-dataset" \
    --output_dir ./outputs/my_pretrain \
    --epochs 1 \
    --batch_size 2 \
    --max_length 512
```

#### 方式2：使用本地数据

```bash
bash scripts/llm/pretrain/run.sh \
    --model Qwen/Qwen2.5-7B \
    --train_data ./data/pretrain/train.jsonl \
    --eval_data ./data/pretrain/val.jsonl \
    --output ./outputs/llm_pretrain \
    --epochs 3 \
    --batch_size 4 \
    --lr 1e-4
```

#### 方式3：使用配置文件

```bash
bash scripts/llm/pretrain/run.sh --config configs/llm/pretrain/config.yaml
```

---

## 数据准备

### 预训练数据格式

```jsonl
{"text": "人工智能是计算机科学的一个分支..."}
{"text": "机器学习是人工智能的核心技术..."}
```

### 快速下载示例数据

```bash
mkdir -p data/pretrain

# 使用 Python 下载 HuggingFace 数据
python << 'EOF'
from datasets import load_dataset
import json

# 下载 Firefly 预训练数据子集
dataset = load_dataset("YeungNLP/firefly-pretrain-dataset", split="train", streaming=True)

# 保存前 10000 条
with open("data/pretrain/train.jsonl", "w") as f:
    for i, item in enumerate(dataset):
        if i >= 10000:
            break
        json.dump({"text": item["text"]}, f, ensure_ascii=False)
        f.write("\n")

print("下载完成: 10000 条数据")
EOF
```

---

## 项目结构（阶段一）

```
llm-vlm-framework/
├── configs/llm/pretrain/
│   └── config.yaml              # 预训练配置
├── docs/
│   ├── recommended_datasets.md  # 推荐数据集指南
│   └── llm_data_formats.md      # 数据格式说明
├── src/llm_training/
│   ├── data/
│   │   └── pretrain_dataloader.py   # 预训练数据加载器
│   └── training/
│       └── pretrain_trainer.py      # 预训练器
└── scripts/llm/
    ├── pretrain/
    │   ├── run.sh               # 主训练脚本
    │   └── test.sh              # 测试脚本
    └── pipeline.sh              # 全流程脚本
```

---

## 后续计划

### 阶段 2：SFT 监督微调（待完成）
- [ ] 完善 SFT 数据加载器（支持多轮对话）
- [ ] 完善 SFT 训练器
- [ ] 支持 LoRA/QLoRA
- [ ] SFT 测试脚本
- [ ] 更新开发日志（SFT 阶段）

### 阶段 3：DPO/RLHF 对齐（待完成）
- [ ] DPO 训练器实现
- [ ] GRPO 训练器实现
- [ ] 偏好数据加载器
- [ ] 更新开发日志（RLHF 阶段）

### 阶段 4：VLM 训练（待完成）
- [ ] 视觉编码器实现
- [ ] 多模态融合模块
- [ ] VLM 预训练器
- [ ] VLM SFT 训练器
- [ ] 更新开发日志（VLM 阶段）

---

## 常见问题

### Q: 显存不足怎么办？

A: 可以尝试以下方法：
1. 减小 batch_size
2. 增大 gradient_accumulation_steps
3. 使用更小的 max_length
4. 启用梯度检查点
5. 使用 LoRA 而不是全参数微调

### Q: 如何监控训练？

A: 训练过程中会自动记录到 Tensorboard：
```bash
tensorboard --logdir ./outputs/llm_pretrain/runs
```

### Q: 训练中断如何恢复？

A: 当前版本暂未实现断点续训，可以从最近的 checkpoint 重新开始：
```bash
--model ./outputs/llm_pretrain/checkpoint-1000
```

---

## 参考资源

- [悟道数据集](https://data.baai.ac.cn/details/WuDaoCorporaText)
- [BELLE 项目](https://github.com/LianjiaTech/BELLE)
- [Firefly 项目](https://github.com/yangjianxin1/Firefly)
- [HuggingFace 中文数据集](https://huggingface.co/datasets?language=language:zh&sort=downloads)
