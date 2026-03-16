#!/usr/bin/env python3
"""
带TensorBoard日志的训练测试
"""

import sys
sys.path.insert(0, './src')

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from torch.utils.tensorboard import SummaryWriter

print("=" * 60)
print("TensorBoard训练可视化")
print("=" * 60)

# 创建模型
config = GPT2Config(vocab_size=1000, n_positions=512, n_embd=128, n_layer=4, n_head=4)
model = GPT2LMHeadModel(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

# TensorBoard记录器
log_dir = "./runs/llm_training_demo"
writer = SummaryWriter(log_dir=log_dir)
print(f"✓ TensorBoard日志目录: {log_dir}")

# 模拟数据
class SimpleDataset:
    def __init__(self, num_samples, seq_length, vocab_size):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        torch.manual_seed(idx)
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        attention_mask = torch.ones(self.seq_length, dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

dataset = SimpleDataset(100, 128, config.vocab_size)

# 训练并记录
print("\n开始训练 (100步)...")
model.train()

for step in range(100):
    sample = dataset[step % len(dataset)]
    input_ids = sample['input_ids'].unsqueeze(0)
    attention_mask = sample['attention_mask'].unsqueeze(0)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    
    # 记录到TensorBoard
    writer.add_scalar('Loss/train', loss.item(), step)
    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], step)
    
    # 每10步记录梯度统计
    if step % 10 == 0:
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'gradients/{name}', param.grad, step)
                break  # 只记录第一个参数的梯度，避免日志太多
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if step % 20 == 0:
        print(f"  Step {step:3d}: Loss = {loss.item():.4f}")

writer.close()
print(f"\n✓ 训练完成，TensorBoard日志已保存到: {log_dir}")
print(f"\n查看命令: tensorboard --logdir={log_dir}")
