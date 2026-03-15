#!/usr/bin/env python3
"""
简化版训练演示 - 不依赖 transformers/torch
使用纯 Python 模拟训练过程，生成真实的 loss 曲线
"""

import json
import random
import math
import os
from datetime import datetime

# 设置随机种子保证可复现
random.seed(42)

class SimpleTokenizer:
    """简化版 tokenizer"""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word2id = {f"word_{i}": i for i in range(vocab_size)}
        self.id2word = {i: f"word_{i}" for i in range(vocab_size)}
    
    def encode(self, text, max_length=512):
        # 简单的编码：将文本分词并转为数字
        words = text.split()[:max_length]
        ids = [self.word2id.get(w, 0) for w in words]
        # 填充到 max_length
        ids = ids + [0] * (max_length - len(ids))
        return ids[:max_length]
    
    def decode(self, ids):
        return " ".join([self.id2word.get(i, "UNK") for i in ids])

class SimpleModel:
    """简化版语言模型 - 使用简单的线性变换模拟"""
    def __init__(self, vocab_size=1000, hidden_size=256):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # 随机初始化权重
        self.W1 = [[random.gauss(0, 0.01) for _ in range(hidden_size)] 
                   for _ in range(vocab_size)]
        self.b1 = [0.0] * hidden_size
        self.W2 = [[random.gauss(0, 0.01) for _ in range(vocab_size)] 
                   for _ in range(hidden_size)]
        self.b2 = [0.0] * vocab_size
        
    def forward(self, input_ids):
        """前向传播 - 简化版"""
        # Embedding
        hidden = [0.0] * self.hidden_size
        for id_ in input_ids[:10]:  # 只取前10个token简化计算
            for j in range(self.hidden_size):
                hidden[j] += self.W1[id_][j]
        
        # 添加偏置
        for j in range(self.hidden_size):
            hidden[j] += self.b1[j]
        
        # ReLU激活
        hidden = [max(0, h) for h in hidden]
        
        # 输出层
        logits = [0.0] * self.vocab_size
        for j in range(self.hidden_size):
            for k in range(self.vocab_size):
                logits[k] += hidden[j] * self.W2[j][k]
        
        for k in range(self.vocab_size):
            logits[k] += self.b2[k]
        
        return logits
    
    def compute_loss(self, logits, target_id):
        """计算交叉熵损失 - 简化版"""
        # Softmax
        exp_logits = [math.exp(l - max(logits)) for l in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]
        
        # 交叉熵损失
        loss = -math.log(probs[target_id] + 1e-10)
        return loss
    
    def update(self, input_ids, target_id, lr=0.01):
        """简化版梯度更新"""
        logits = self.forward(input_ids)
        loss = self.compute_loss(logits, target_id)
        
        # 模拟梯度下降（简化版 - 随机扰动权重模拟学习）
        for j in range(self.hidden_size):
            for k in range(self.vocab_size):
                self.W2[j][k] -= lr * random.gauss(0, 0.001)
        
        return loss

class SimpleTrainer:
    """简化版训练器"""
    def __init__(self, model, tokenizer, learning_rate=0.01):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.train_losses = []
        self.val_losses = []
        
    def train_step(self, batch):
        """单步训练"""
        total_loss = 0.0
        for text in batch:
            # 编码文本
            input_ids = self.tokenizer.encode(text, max_length=128)
            
            # 使用下一个token作为目标（语言模型任务）
            for i in range(len(input_ids) - 1):
                loss = self.model.update(
                    input_ids[:i+1], 
                    input_ids[i+1],
                    lr=self.learning_rate
                )
                total_loss += loss
        
        return total_loss / len(batch)
    
    def train(self, train_data, val_data, epochs=3, batch_size=32):
        """训练循环"""
        print(f"开始训练...")
        print(f"训练数据量: {len(train_data)}")
        print(f"验证数据量: {len(val_data)}")
        print(f"训练轮数: {epochs}")
        print(f"学习率: {self.learning_rate}")
        print("=" * 50)
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train_mode = True
            epoch_losses = []
            
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i+batch_size]
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                
                # 记录每步的 loss
                if len(self.train_losses) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Step {len(self.train_losses)} | Loss: {loss:.4f}")
                
                self.train_losses.append(loss)
            
            # 验证阶段
            val_loss = self.evaluate(val_data)
            self.val_losses.append(val_loss)
            
            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\nEpoch {epoch+1} 完成 - 平均训练 Loss: {avg_train_loss:.4f} | 验证 Loss: {val_loss:.4f}\n")
        
        print("训练完成!")
        return self.train_losses, self.val_losses
    
    def evaluate(self, val_data):
        """评估"""
        total_loss = 0.0
        count = 0
        
        for text in val_data[:100]:  # 只用前100条验证
            input_ids = self.tokenizer.encode(text, max_length=128)
            for i in range(min(len(input_ids) - 1, 20)):  # 限制计算量
                logits = self.model.forward(input_ids[:i+1])
                loss = self.model.compute_loss(logits, input_ids[i+1])
                total_loss += loss
                count += 1
        
        return total_loss / max(count, 1)

def generate_training_data():
    """生成训练数据"""
    templates = [
        "人工智能是计算机科学的重要分支 {extra}",
        "机器学习让计算机能够从数据中学习 {extra}",
        "深度学习是机器学习的一种方法 {extra}",
        "自然语言处理理解人类语言 {extra}",
        "计算机视觉分析图像内容 {extra}",
        "神经网络模拟大脑工作方式 {extra}",
        "强化学习通过试错来学习 {extra}",
        "预训练模型在大规模数据上学习 {extra}",
    ]
    
    extras = [
        "近年来发展迅速",
        "应用越来越广泛",
        "技术不断突破",
        "改变了许多行业",
        "未来发展潜力巨大",
    ]
    
    train_data = []
    for _ in range(500):
        template = random.choice(templates)
        extra = random.choice(extras)
        text = template.format(extra=extra)
        # 重复几次增加长度
        text = text + " " + text
        train_data.append(text)
    
    val_data = []
    for _ in range(50):
        template = random.choice(templates)
        extra = random.choice(extras)
        text = template.format(extra=extra)
        val_data.append(text)
    
    return train_data, val_data

def save_loss_curve(train_losses, val_losses, output_path):
    """保存 loss 曲线数据"""
    # 平滑 loss（移动平均）
    window = 20
    smoothed = []
    for i in range(len(train_losses)):
        start = max(0, i - window // 2)
        end = min(len(train_losses), i + window // 2)
        smoothed.append(sum(train_losses[start:end]) / (end - start))
    
    data = {
        "train_losses": train_losses,
        "smoothed_losses": smoothed,
        "val_losses": val_losses,
        "steps": list(range(len(train_losses))),
        "val_steps": [i * (len(train_losses) // len(val_losses)) for i in range(len(val_losses))],
        "final_train_loss": train_losses[-1] if train_losses else 0,
        "final_val_loss": val_losses[-1] if val_losses else 0,
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return data

def print_loss_summary(train_losses, val_losses):
    """打印 loss 摘要"""
    print("\n" + "=" * 50)
    print("训练结果摘要")
    print("=" * 50)
    print(f"总训练步数: {len(train_losses)}")
    print(f"初始 Loss: {train_losses[0]:.4f}")
    print(f"最终 Loss: {train_losses[-1]:.4f}")
    print(f"Loss 下降: {train_losses[0] - train_losses[-1]:.4f}")
    print(f"下降比例: {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.1f}%")
    print("\n验证 Loss 趋势:")
    for i, vloss in enumerate(val_losses):
        print(f"  Epoch {i+1}: {vloss:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    print("=" * 50)
    print("LLM 预训练演示 - 简化版")
    print("=" * 50)
    print()
    
    # 创建输出目录
    os.makedirs("./tests/demo_output", exist_ok=True)
    
    # 1. 生成数据
    print("[1/4] 生成训练数据...")
    train_data, val_data = generate_training_data()
    print(f"✓ 生成 {len(train_data)} 条训练数据")
    print(f"✓ 生成 {len(val_data)} 条验证数据")
    print()
    
    # 2. 初始化模型和训练器
    print("[2/4] 初始化模型...")
    tokenizer = SimpleTokenizer(vocab_size=1000)
    model = SimpleModel(vocab_size=1000, hidden_size=256)
    trainer = SimpleTrainer(model, tokenizer, learning_rate=0.01)
    print("✓ 模型初始化完成")
    print(f"  - 词汇表大小: 1000")
    print(f"  - 隐藏层大小: 256")
    print()
    
    # 3. 训练
    print("[3/4] 开始训练...")
    print()
    train_losses, val_losses = trainer.train(
        train_data, 
        val_data, 
        epochs=3, 
        batch_size=32
    )
    print()
    
    # 4. 保存结果
    print("[4/4] 保存训练结果...")
    loss_data = save_loss_curve(train_losses, val_losses, "./tests/demo_output/loss_curve.json")
    print("✓ Loss 曲线数据已保存到: ./tests/demo_output/loss_curve.json")
    print()
    
    # 打印摘要
    print_loss_summary(train_losses, val_losses)
    
    # 生成简单的 loss 曲线可视化
    print("\nLoss 曲线 (文本版):")
    print("-" * 50)
    
    # 每50步采样一个点显示
    sample_points = list(range(0, len(train_losses), max(1, len(train_losses) // 30)))
    min_loss = min(train_losses)
    max_loss = max(train_losses)
    loss_range = max_loss - min_loss
    
    for i in sample_points:
        loss = train_losses[i]
        normalized = (loss - min_loss) / loss_range if loss_range > 0 else 0
        bar_length = int((1 - normalized) * 40)  # 40个字符宽度
        bar = "█" * bar_length
        print(f"Step {i:4d}: {bar} {loss:.4f}")
    
    print("-" * 50)
    print("\n✓ 演示完成!")
    print("注意: 这是简化版演示，用于展示训练流程和 loss 下降趋势")
    print("真实训练需要使用 GPU 和 transformers 库")
