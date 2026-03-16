# ============================================
# LLM 训练数据格式详细说明
# ============================================

## 阶段 1: 预训练 (Pre-training)

### 数据格式
纯文本格式，每行一个 JSON 对象。

```jsonl
{"text": "这是一个用于预训练的文本样本..."}
{"text": "预训练数据可以是任何领域的文本内容..."}
{"text": "书籍、论文、网页、代码等都可以作为预训练数据..."}
```

### 字段说明
- `text` (required): 文本内容

### 数据准备示例

```python
import json

# 将原始文本转换为预训练格式
with open("raw_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 按段落分割
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

# 保存为 JSONL
with open("pretrain_data.jsonl", "w", encoding="utf-8") as f:
    for para in paragraphs:
        if len(para) > 100:  # 过滤太短的内容
            json.dump({"text": para}, f, ensure_ascii=False)
            f.write("\n")
```

---

## 阶段 2: 监督微调 (SFT)

### 格式 A: Conversation 格式（推荐）

```jsonl
{
  "messages": [
    {"role": "system", "content": "你是一个有用的 AI 助手。"},
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}
  ]
}
{
  "messages": [
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是人工智能的一个分支..."},
    {"role": "user", "content": "能举个例子吗？"},
    {"role": "assistant", "content: "当然！比如垃圾邮件过滤器..."}
  ]
}
```

### 格式 B: Alpaca 格式

```jsonl
{
  "instruction": "解释什么是深度学习",
  "input": "用通俗易懂的语言",
  "output": "深度学习是机器学习的一个子集..."
}
{
  "instruction": "列举三种机器学习算法",
  "input": "",
  "output": "1. 决策树\n2. 支持向量机\n3. 神经网络"
}
```

### 格式 C: ShareGPT 格式

```jsonl
{
  "conversations": [
    {"from": "human", "value": "你好！"},
    {"from": "gpt", "value": "你好！很高兴见到你。"},
    {"from": "human", "value": "今天天气怎么样？"},
    {"from": "gpt", "value": "我无法获取实时天气信息..."}
  ]
}
```

### 字段说明

**Conversation 格式:**
- `messages`: 消息列表
  - `role`: 角色 (`system` | `user` | `assistant`)
  - `content`: 消息内容

**Alpaca 格式:**
- `instruction` (required): 指令/问题
- `input` (optional): 额外输入上下文
- `output` (required): 期望输出

**ShareGPT 格式:**
- `conversations`: 对话列表
  - `from`: 发言者 (`human` | `gpt`)
  - `value`: 对话内容

---

## 阶段 3: DPO (Direct Preference Optimization)

### 数据格式

```jsonl
{
  "prompt": "什么颜色代表和平？",
  "chosen": "白色通常代表和平与纯洁，象征着纯净和宁静。",
  "rejected": "我不知道，可能是红色？"
}
{
  "prompt": "如何学习编程？",
  "chosen": "学习编程可以遵循以下步骤：\n1. 选择一门编程语言\n2. 学习基础语法\n3. 做小项目练习\n4. 阅读他人代码",
  "rejected": "直接开始写代码就行了，不用学理论。"
}
{
  "prompt": "写一首关于春天的诗",
  "chosen": "春风拂面柳丝长，\n桃花盛开满院香。\n燕子归来寻旧垒，\n人间四月好时光。",
  "rejected": "春天来了，花开了，很好。"
}
```

### 字段说明
- `prompt` (required): 输入提示/问题
- `chosen` (required): 偏好的/更好的回复
- `rejected` (required): 非偏好的/较差的回复

### 数据收集方法
1. **人工标注**: 让人工标注者对同一问题的不同回复进行排序
2. **模型生成对比**: 使用不同温度或不同模型生成多个回复
3. **自动构造**: 使用规则或启发式方法构造对比对

---

## 阶段 4: GRPO / RLHF

### 数据格式
只需要 prompt，模型自行生成回复。

```jsonl
{"prompt": "请解释量子计算的基本原理"}
{"prompt": "写一个 Python 函数计算斐波那契数列"}
{"prompt": "总结人工智能发展的三个主要阶段"}
```

### 字段说明
- `prompt` (required): 输入提示

### 奖励函数配置
在 GRPO/RLHF 中，需要配置奖励函数来评估生成质量：

```yaml
reward_functions:
  - type: "length"           # 长度奖励
    min_length: 50
    max_length: 500
  - type: "format"          # 格式奖励
    regex: "^##

.*##

.*$"  # Markdown 格式
  - type: "custom"          # 自定义奖励
    module: "rewards.custom"
    function: "compute_reward"
```

---

## 完整训练流程示例

### 步骤 1: 准备预训练数据
```bash
# 数据格式: {"text": "..."}
mkdir -p data/pretrain
# 准备 pretrain/train.jsonl 和 pretrain/val.jsonl
```

### 步骤 2: 运行预训练
```bash
bash scripts/llm/pretrain/run.sh
# 输出: outputs/llm_pretrain/final
```

### 步骤 3: 准备 SFT 数据
```bash
# 数据格式: {"messages": [...]}
mkdir -p data/sft
# 准备 sft/train.jsonl 和 sft/val.jsonl
```

### 步骤 4: 运行 SFT
```bash
export PRETRAIN_MODEL=./outputs/llm_pretrain/final
bash scripts/llm/sft/run.sh
# 输出: outputs/llm_sft/final
```

### 步骤 5: 准备偏好数据
```bash
# 数据格式: {"prompt": "...", "chosen": "...", "rejected": "..."}
mkdir -p data/preference
# 准备 preference/train.jsonl
```

### 步骤 6: 运行 DPO
```bash
export SFT_MODEL=./outputs/llm_sft/final
bash scripts/llm/dpo/run.sh
# 输出: outputs/llm_dpo/final
```

---

## 数据格式转换脚本

### Alpaca 转 Conversation
```python
import json

def alpaca_to_conversation(alpaca_file, output_file):
    with open(alpaca_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            item = json.loads(line)
            
            instruction = item["instruction"]
            input_text = item.get("input", "")
            output = item["output"]
            
            # 构建 prompt
            if input_text:
                user_content = f"{instruction}\n\n{input_text}"
            else:
                user_content = instruction
            
            conversation = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output}
                ]
            }
            
            json.dump(conversation, f_out, ensure_ascii=False)
            f_out.write("\n")

alpaca_to_conversation("alpaca_data.jsonl", "conversation_data.jsonl")
```

### 生成 DPO 数据（从 SFT 数据）
```python
import json

def create_dpo_data(sft_file, output_file):
    """
    简单示例：使用不同质量水平的回复构造偏好对
    实际应用中需要更复杂的策略
    """
    with open(sft_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            item = json.loads(line)
            messages = item["messages"]
            
            # 提取用户问题和助手回复
            user_msg = None
            assistant_msg = None
            for msg in messages:
                if msg["role"] == "user":
                    user_msg = msg["content"]
                elif msg["role"] == "assistant":
                    assistant_msg = msg["content"]
            
            if user_msg and assistant_msg:
                dpo_item = {
                    "prompt": user_msg,
                    "chosen": assistant_msg,
                    "rejected": f"对不起，我不知道答案。"  # 示例：简单的 rejected
                }
                json.dump(dpo_item, f_out, ensure_ascii=False)
                f_out.write("\n")
```
