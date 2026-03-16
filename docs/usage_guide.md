# LLM-VLM 训练框架使用指南

## 目录

1. [快速开始](#快速开始)
2. [数据收集](#数据收集)
3. [LLM 训练](#llm-训练)
4. [VLM 训练](#vlm-训练)
5. [模型评估](#模型评估)
6. [模型部署](#模型部署)

---

## 快速开始

### 安装依赖

```bash
cd /root/.openclaw/workspace/projects/llm-vlm-framework
pip install -r requirements.txt
```

### 项目结构

```
llm-vlm-framework/
├── configs/          # 配置文件
├── src/              # 源代码
│   ├── data_collection/    # 数据收集
│   ├── llm_training/       # LLM 训练
│   ├── vlm_training/       # VLM 训练
│   └── common/             # 公共组件
├── scripts/          # 训练脚本
├── examples/         # 示例代码
└── docs/             # 文档
```

---

## 数据收集

### 1. 爬取公众号文章

```bash
python -m src.data_collection.wechat.article_crawler \
    --keyword "人工智能" \
    --pages 5 \
    --output ./data/articles.jsonl \
    --fetch-content
```

### 2. 数据清洗

```bash
python -m src.data_collection.preprocessing.clean_data \
    --input ./data/articles.jsonl \
    --output ./data/cleaned.jsonl \
    --remove-html \
    --remove-urls \
    --min-length 50
```

### 3. 构建训练数据集

```bash
python -m src.data_collection.dataset_builder.build_dataset \
    --input ./data/cleaned.jsonl \
    --output ./data/dataset \
    --format conversation \
    --system-prompt "你是一个有用的AI助手。"
```

---

## LLM 训练

### 预训练

```bash
bash scripts/llm_pretrain.sh
```

或手动运行：

```bash
python -m src.llm_training.training.sft_trainer \
    --mode pretrain \
    --model qwen2.5-7b \
    --train-data ./data/pretrain/train.jsonl \
    --output ./outputs/llm_pretrain
```

### SFT 微调

```bash
bash scripts/llm_sft.sh
```

### DPO 对齐训练

```bash
bash scripts/llm_dpo.sh
```

---

## VLM 训练

### 预训练（图文对齐）

```bash
bash scripts/vlm_pretrain.sh
```

### 指令微调

```bash
bash scripts/vlm_sft.sh
```

---

## 模型评估

```bash
python -m src.common.evaluation.evaluate \
    --model ./outputs/llm_sft/final \
    --data ./data/eval/eval.jsonl \
    --output ./results/eval_results.jsonl \
    --type llm
```

---

## 模型部署

### API 服务

```bash
python -m src.common.deployment.api_server \
    --model ./outputs/llm_sft/final \
    --port 8000
```

### Gradio 界面

```bash
python -m src.common.deployment.gradio_ui \
    --model ./outputs/llm_sft/final \
    --port 7860
```

### Python API 调用

```python
from src.llm_training.models.base_model import load_model_for_inference

model = load_model_for_inference("./outputs/llm_sft/final")
response = model.generate("你好！")
print(response)
```

---

## 数据格式说明

### 训练数据格式

#### Conversation 格式（推荐）

```json
{
  "messages": [
    {"role": "system", "content": "你是一个有用的AI助手。"},
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}
  ]
}
```

#### Alpaca 格式

```json
{
  "instruction": "解释人工智能",
  "input": "",
  "output": "人工智能是计算机科学的一个分支..."
}
```

#### DPO 偏好格式

```json
{
  "prompt": "什么颜色代表和平？",
  "chosen": "白色通常代表和平与纯洁。",
  "rejected": "我不知道。"
}
```

#### VLM 多模态格式

```json
{
  "image": "images/example.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\n描述这张图片。"},
    {"from": "gpt", "value": "这是一张美丽的风景照片..."}
  ]
}
```

---

## 支持的模型

### LLM 基础模型
- Qwen2 / Qwen2.5 (推荐)
- LLaMA 2 / LLaMA 3
- Baichuan 2
- ChatGLM 3 / GLM-4
- InternLM 2

### 视觉编码器
- CLIP ViT (OpenAI)
- SigLIP (Google)
- EVA-CLIP (BAAI)
- InternViT (Shanghai AI Lab)

---

## 硬件要求

| 训练阶段 | 显存需求 | 推荐配置 |
|---------|---------|---------|
| LLM SFT (7B) | 16GB+ | RTX 4090 / A100 40G |
| LLM SFT (14B) | 32GB+ | A100 40G / A100 80G |
| LLM Pretrain | 80GB+ | 多卡 A100 80G |
| VLM Pretrain | 40GB+ | A100 40G+ |
| DPO/RLHF | 40GB+ | A100 40G+ |

---

## 常见问题

### 1. 显存不足

解决方案：
- 启用梯度累积
- 使用更小的 batch size
- 启用 8bit/4bit 量化
- 使用 LoRA 微调
- 启用梯度检查点

### 2. 数据加载慢

解决方案：
- 使用更快的存储（NVMe SSD）
- 增加 dataloader 的 num_workers
- 预处理数据并保存为 Arrow 格式

### 3. 训练不稳定

解决方案：
- 降低学习率
- 增加 warmup 步骤
- 使用梯度裁剪
- 检查数据质量
