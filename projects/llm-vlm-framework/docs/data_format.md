# 数据格式说明

## 概述

本文档详细说明训练框架支持的各种数据格式。

---

## LLM 训练数据格式

### 1. Conversation 格式（推荐）

OpenAI 风格的消息列表格式，支持多轮对话。

```json
{
  "messages": [
    {"role": "system", "content": "你是一个有用的AI助手。"},
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}
  ]
}
```

**字段说明：**
- `messages`: 消息列表
- `role`: 角色，可选 `system` | `user` | `assistant`
- `content`: 消息内容

---

### 2. ShareGPT 格式

ShareGPT 对话格式。

```json
{
  "conversations": [
    {"from": "human", "value": "你好！"},
    {"from": "gpt", "value": "你好！有什么我可以帮助你的吗？"}
  ]
}
```

**字段说明：**
- `conversations`: 对话列表
- `from`: 发言者，`human` 或 `gpt`
- `value`: 对话内容

---

### 3. Alpaca 格式

Stanford Alpaca 原始格式。

```json
{
  "instruction": "解释什么是机器学习",
  "input": "用通俗易懂的语言",
  "output": "机器学习是人工智能的一个分支..."
}
```

**字段说明：**
- `instruction`: 指令/问题
- `input`: 额外输入（可选）
- `output`: 期望输出

---

### 4. Instruction 格式

简化的指令-回复格式。

```json
{
  "prompt": "解释什么是深度学习",
  "completion": "深度学习是机器学习的一个子集..."
}
```

**字段说明：**
- `prompt`: 输入提示
- `completion`: 完成/回复内容

---

## 偏好数据格式（DPO/RLHF）

用于对齐训练的偏好对比数据。

```json
{
  "prompt": "什么颜色代表和平？",
  "chosen": "白色通常代表和平与纯洁，象征着纯净和宁静。",
  "rejected": "我不知道，可能是红色？"
}
```

**字段说明：**
- `prompt`: 输入提示
- `chosen`: 偏好的/更好的回复
- `rejected`: 不被偏好的/较差的回复

---

## VLM 多模态数据格式

### 1. 标准 VLM 格式

```json
{
  "image": "images/example.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n描述这张图片。"
    },
    {
      "from": "gpt",
      "value": "这是一张美丽的风景照片，展示了一片山脉和湖泊..."
    }
  ]
}
```

**字段说明：**
- `image`: 图像相对路径
- `conversations`: 对话列表
- `<image>`: 图像占位符，会被替换为图像 token

---

### 2. 多图像格式

```json
{
  "images": ["image1.jpg", "image2.jpg"],
  "messages": [
    {"role": "user", "content": "<image 1><image 2>\n比较这两张图片。"},
    {"role": "assistant", "content": "第一张图片展示...第二张图片展示..."}
  ]
}
```

---

## 预训练数据格式

纯文本格式，用于继续预训练。

```json
{"text": "这是一段用于预训练的文本内容..."}
```

或简单的文本文件（每行一个样本）。

---

## 数据转换

### 从原始数据转换

```python
from src.data_collection.dataset_builder import DatasetBuilder, DatasetConfig, DatasetFormat

config = DatasetConfig(
    format=DatasetFormat.CONVERSATION,
    system_prompt="你是一个有用的AI助手。"
)

builder = DatasetBuilder(config)
builder.build_from_raw(
    input_files=["./raw_data.jsonl"],
    output_dir="./dataset",
    text_key="content"
)
```

---

## 数据集示例

### 查看 `examples/sample_data/` 目录

- `conversation_sample.jsonl`: Conversation 格式示例
- `alpaca_sample.jsonl`: Alpaca 格式示例
- `preference_sample.jsonl`: DPO 偏好数据示例
- `vlm_sample.jsonl`: VLM 多模态数据示例
