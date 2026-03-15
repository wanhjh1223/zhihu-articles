#!/bin/bash
# ============================================
# LLM 预训练阶段测试脚本
# ============================================
# 使用小规模数据进行全流程测试

set -e

echo "=========================================="
echo "LLM 预训练阶段测试"
echo "=========================================="
echo ""

# 配置
TEST_DIR="./tests/llm_pretrain"
MODEL_NAME="gpt2"  # 使用小模型进行测试
TEST_DATA_SIZE=1000
MAX_LENGTH=512
BATCH_SIZE=2
EPOCHS=1

echo "测试配置:"
echo "  模型: $MODEL_NAME"
echo "  测试数据量: $TEST_DATA_SIZE"
echo "  最大长度: $MAX_LENGTH"
echo ""

# 创建测试目录
mkdir -p "$TEST_DIR"
mkdir -p "$TEST_DIR/data"
mkdir -p "$TEST_DIR/output"

# 步骤 1: 创建测试数据
echo "[1/5] 创建测试数据..."
python3 << 'EOF'
import json
import random
import os

# 测试文本模板
templates = [
    "人工智能是计算机科学的一个分支，致力于创造能够模拟人类智能的系统。{content}",
    "机器学习是人工智能的核心技术之一，它使计算机能够从数据中学习。{content}",
    "深度学习是机器学习的一个子领域，基于神经网络。{content}",
    "自然语言处理让计算机能够理解和生成人类语言。{content}",
    "计算机视觉使机器能够理解和分析视觉信息。{content}",
]

contents = [
    "近年来，这项技术得到了快速发展，在各行各业都有广泛应用。",
    "随着算力的提升和数据的积累，相关技术不断突破。",
    "未来的发展方向包括更高效的算法和更广泛的应用场景。",
    "这项技术正在改变我们的生活方式和工作方式。",
    "研究人员正在探索新的方法来解决现有的挑战。",
]

# 生成训练数据
with open("./tests/llm_pretrain/data/train.jsonl", "w", encoding="utf-8") as f:
    for i in range(1000):
        template = random.choice(templates)
        content = random.choice(contents)
        text = template.format(content=content) * random.randint(1, 5)
        json.dump({"text": text}, f, ensure_ascii=False)
        f.write("\n")

# 生成验证数据
with open("./tests/llm_pretrain/data/val.jsonl", "w", encoding="utf-8") as f:
    for i in range(100):
        template = random.choice(templates)
        content = random.choice(contents)
        text = template.format(content=content) * random.randint(1, 5)
        json.dump({"text": text}, f, ensure_ascii=False)
        f.write("\n")

print(f"创建了 1000 条训练数据")
print(f"创建了 100 条验证数据")
EOF

echo "✓ 测试数据创建完成"
echo ""

# 步骤 2: 测试数据加载器
echo "[2/5] 测试数据加载器..."
python3 << 'EOF'
import sys
sys.path.insert(0, "./src")

from transformers import AutoTokenizer
from llm_training.data.pretrain_dataloader import PretrainDataset, PretrainDataConfig

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = PretrainDataConfig(
    data_path="./tests/llm_pretrain/data/train.jsonl",
    max_length=512,
    concat_samples=True,
)

dataset = PretrainDataset(
    config=config,
    tokenizer=tokenizer,
)

print(f"数据集长度 (估算): {len(dataset)}")

# 测试读取
for i, sample in enumerate(dataset):
    if i >= 2:
        break
    print(f"\nSample {i}:")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")

print("\n✓ 数据加载器测试通过")
EOF

echo ""

# 步骤 3: 运行小规模训练
echo "[3/5] 运行小规模训练测试..."
python -m src.llm_training.training.pretrain_trainer \
    --model "$MODEL_NAME" \
    --train_data "$TEST_DIR/data/train.jsonl" \
    --eval_data "$TEST_DIR/data/val.jsonl" \
    --output_dir "$TEST_DIR/output" \
    --epochs 1 \
    --batch_size 2 \
    --lr 5e-5 \
    --max_length $MAX_LENGTH

echo "✓ 训练测试通过"
echo ""

# 步骤 4: 验证输出
echo "[4/5] 验证模型输出..."
if [ -d "$TEST_DIR/output/final" ]; then
    echo "✓ 最终模型目录存在"
    
    if [ -f "$TEST_DIR/output/final/config.json" ]; then
        echo "✓ 模型配置文件存在"
    fi
    
    if [ -f "$TEST_DIR/output/final/pytorch_model.bin" ] || [ -f "$TEST_DIR/output/final/model.safetensors" ]; then
        echo "✓ 模型权重文件存在"
    fi
    
    if [ -f "$TEST_DIR/output/final/vocab.json" ]; then
        echo "✓ Tokenizer 文件存在"
    fi
else
    echo "✗ 最终模型目录不存在"
    exit 1
fi

echo ""

# 步骤 5: 测试模型加载和推理
echo "[5/5] 测试模型加载和推理..."
python3 << 'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载训练后的模型
model_path = "./tests/llm_pretrain/output/final"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f"模型加载成功")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# 测试生成
inputs = tokenizer("人工智能是", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n生成测试:")
print(f"  输入: 人工智能是")
print(f"  输出: {generated_text}")

print("\n✓ 模型推理测试通过")
EOF

echo ""
echo "=========================================="
echo "所有测试通过!"
echo "=========================================="
echo ""
echo "测试输出位置: $TEST_DIR/output"
echo ""
echo "清理测试数据:"
echo "  rm -rf $TEST_DIR"
