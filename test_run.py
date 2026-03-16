#!/usr/bin/env python3
"""
快速测试 - 验证预训练框架能运行
使用最小模型和最少数据
"""

import sys
sys.path.insert(0, './src')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_training.data.pretrain_dataloader import PretrainDataset, PretrainDataConfig

print("=" * 50)
print("LLM预训练框架测试")
print("=" * 50)

# 1. 加载模型和tokenizer
print("\n[1/5] 加载gpt2模型...")
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"✓ 模型加载完成: {model_name}")
print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# 2. 测试数据加载器
print("\n[2/5] 测试数据加载器...")
config = PretrainDataConfig(
    data_path="./tests/llm_pretrain/data/train.jsonl",
    max_length=128,
    concat_samples=True,
)
dataset = PretrainDataset(config=config, tokenizer=tokenizer)
print(f"✓ 数据加载器创建成功")

# 3. 测试单步训练
print("\n[3/5] 测试单步训练...")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 获取一个样本
sample = next(iter(dataset))
input_ids = sample['input_ids'].unsqueeze(0)
attention_mask = sample['attention_mask'].unsqueeze(0)

print(f"  input_ids shape: {input_ids.shape}")

# 前向传播
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
loss = outputs.loss
print(f"  初始Loss: {loss.item():.4f}")

# 反向传播
loss.backward()
optimizer.step()
optimizer.zero_grad()

print(f"✓ 单步训练成功")

# 4. 测试多步训练
print("\n[4/5] 测试多步训练 (10步)...")
initial_loss = None
final_loss = None

for step in range(10):
    sample = next(iter(dataset))
    input_ids = sample['input_ids'].unsqueeze(0)
    attention_mask = sample['attention_mask'].unsqueeze(0)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    
    if step == 0:
        initial_loss = loss.item()
    if step == 9:
        final_loss = loss.item()
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if step % 3 == 0:
        print(f"  Step {step}: Loss = {loss.item():.4f}")

print(f"✓ 10步训练完成")
print(f"  初始Loss: {initial_loss:.4f}")
print(f"  最终Loss: {final_loss:.4f}")
print(f"  下降: {initial_loss - final_loss:.4f}")

# 5. 保存模型
print("\n[5/5] 保存模型...")
output_dir = "./tests/llm_pretrain/output"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✓ 模型已保存到: {output_dir}")

print("\n" + "=" * 50)
print("✓ 所有测试通过！框架可以正常运行")
print("=" * 50)
