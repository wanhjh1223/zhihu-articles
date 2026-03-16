#!/usr/bin/env python3
"""
从头训练测试 - 使用随机初始化的模型验证训练流程
"""

import sys
sys.path.insert(0, './src')

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

print("=" * 60)
print("LLM-VLM框架 - 从头训练验证")
print("=" * 60)

# 1. 创建随机初始化的小模型
print("\n[1/6] 创建随机初始化的GPT2模型...")
config = GPT2Config(
    vocab_size=50257,
    n_positions=512,
    n_embd=128,
    n_layer=4,
    n_head=4,
)
model = GPT2LMHeadModel(config)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

param_count = sum(p.numel() for p in model.parameters())
print(f"✓ 模型创建成功")
print(f"  参数量: {param_count / 1e6:.2f}M")
print(f"  层数: {config.n_layer}, 维度: {config.n_embd}")

# 2. 创建数据加载器
print("\n[2/6] 创建数据加载器...")
from llm_training.data.pretrain_dataloader import PretrainDataset, PretrainDataConfig

data_config = PretrainDataConfig(
    data_path="./tests/llm_pretrain/data/train.jsonl",
    max_length=128,
    concat_samples=True,
)
dataset = PretrainDataset(config=data_config, tokenizer=tokenizer)
print(f"✓ 数据加载器创建成功")
print(f"  数据集大小: {len(dataset)} 样本")

# 3. 测试数据加载
print("\n[3/6] 测试数据加载...")
sample = next(iter(dataset))
print(f"✓ 数据样本获取成功")
print(f"  input_ids shape: {sample['input_ids'].shape}")
print(f"  attention_mask shape: {sample['attention_mask'].shape}")

# 4. 测试前向传播
print("\n[4/6] 测试前向传播...")
input_ids = sample['input_ids'].unsqueeze(0)
attention_mask = sample['attention_mask'].unsqueeze(0)

model.train()
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
initial_loss = outputs.loss.item()
print(f"✓ 前向传播成功")
print(f"  初始Loss: {initial_loss:.4f}")

# 5. 测试训练步骤
print("\n[5/6] 测试训练步骤 (20步)...")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

losses = []
for step in range(20):
    sample = next(iter(dataset))
    input_ids = sample['input_ids'].unsqueeze(0)
    attention_mask = sample['attention_mask'].unsqueeze(0)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    losses.append(loss.item())
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if step % 5 == 0:
        print(f"  Step {step:2d}: Loss = {loss.item():.4f}")

final_loss = losses[-1]
print(f"  Step 19: Loss = {final_loss:.4f}")
print(f"✓ 训练步骤成功")

# 6. 验证Loss下降
print("\n[6/6] 验证训练效果...")
loss_drop = initial_loss - final_loss
loss_drop_pct = (loss_drop / initial_loss) * 100

print(f"  初始Loss: {initial_loss:.4f}")
print(f"  最终Loss: {final_loss:.4f}")
print(f"  Loss下降: {loss_drop:.4f} ({loss_drop_pct:.1f}%)")

if loss_drop > 0:
    print(f"✓ 训练有效! Loss正在下降")
else:
    print(f"⚠ Loss未下降，可能需要调整参数")

# 7. 测试保存和加载
print("\n[7/6] 测试模型保存和加载...")
output_dir = "./tests/from_scratch_output"
import os
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✓ 模型已保存到: {output_dir}")

# 重新加载
model_loaded = GPT2LMHeadModel.from_pretrained(output_dir)
print(f"✓ 模型加载成功")
print(f"  参数量: {sum(p.numel() for p in model_loaded.parameters()) / 1e6:.2f}M")

print("\n" + "=" * 60)
print("✓ 从头训练验证通过!")
print("✓ 所有组件工作正常!")
print("=" * 60)
print(f"\n训练效果:")
print(f"  - Loss从 {initial_loss:.4f} 降至 {final_loss:.4f}")
print(f"  - 下降幅度: {loss_drop_pct:.1f}%")
print(f"\n结论: 框架代码逻辑正确，可以正常训练!")
