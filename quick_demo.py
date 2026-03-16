#!/usr/bin/env python3
"""
超快速训练演示 - 生成真实的 loss 下降曲线
使用简化数学模型模拟训练过程
"""

import json
import random
import math
import os

random.seed(42)

print("=" * 60)
print("LLM 预训练 Loss 曲线生成演示")
print("=" * 60)
print()

# 模拟训练参数
num_steps = 150
initial_loss = 4.5
final_loss = 2.1
noise_level = 0.15

print("[1/3] 生成训练数据...")
print("  - 模拟训练步数: 150 steps")
print("  - 初始 Loss: 4.5")
print("  - 目标 Loss: 2.1")
print()

# 生成 loss 曲线
train_losses = []
for step in range(num_steps):
    # 指数衰减 + 噪声
    progress = step / num_steps
    decay = math.exp(-3 * progress)  # 指数衰减
    base_loss = final_loss + (initial_loss - final_loss) * decay
    noise = random.gauss(0, noise_level * (0.5 + 0.5 * decay))  # 噪声随训练减小
    loss = base_loss + noise
    train_losses.append(max(loss, final_loss * 0.9))  # 保证不降得太低

print("[2/3] 模拟验证过程...")
val_steps = [0, 50, 100, 150]
val_losses = []
for step in val_steps:
    progress = step / num_steps
    val_loss = final_loss + (initial_loss - final_loss) * math.exp(-2.5 * progress)
    val_losses.append(val_loss + random.gauss(0, 0.1))
print(f"  - 验证点: {val_steps}")
print(f"  - 验证 Loss: {[f'{v:.3f}' for v in val_losses]}")
print()

# 保存结果
os.makedirs("./tests/demo_output", exist_ok=True)
result = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "val_steps": val_steps,
    "metadata": {
        "total_steps": num_steps,
        "initial_loss": initial_loss,
        "final_loss": train_losses[-1],
        "best_loss": min(train_losses),
        "generated_at": "2026-03-15"
    }
}

with open("./tests/demo_output/loss_curve.json", "w") as f:
    json.dump(result, f, indent=2)

print("[3/3] 保存结果...")
print("✓ Loss 曲线已保存到: ./tests/demo_output/loss_curve.json")
print()

# 打印摘要
print("=" * 60)
print("训练结果摘要")
print("=" * 60)
print(f"总训练步数: {num_steps}")
print(f"初始 Loss: {train_losses[0]:.4f}")
print(f"最终 Loss: {train_losses[-1]:.4f}")
print(f"最佳 Loss: {min(train_losses):.4f}")
print(f"Loss 下降: {train_losses[0] - train_losses[-1]:.4f}")
print(f"下降比例: {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.1f}%")
print()

# 绘制文本版 loss 曲线
print("Loss 曲线可视化:")
print("-" * 60)
print("Loss    | 曲线 (越高越好)")
print("-" * 60)

min_loss = min(train_losses)
max_loss = max(train_losses)
loss_range = max_loss - min_loss

# 每10步显示一个点
for i in range(0, len(train_losses), 10):
    loss = train_losses[i]
    normalized = (loss - min_loss) / loss_range if loss_range > 0 else 0.5
    bar_length = int((1 - normalized) * 35)
    bar = "█" * bar_length + "░" * (35 - bar_length)
    print(f"{loss:.3f} | {bar} Step {i:3d}")

print("-" * 60)
print()

# 显示验证点
print("验证 Loss:")
for step, vloss in zip(val_steps, val_losses):
    print(f"  Step {step:3d}: {vloss:.4f}")

print()
print("=" * 60)
print("✓ 演示完成!")
print("=" * 60)
print()
print("说明:")
print("- 这是使用简化数学模型生成的模拟训练曲线")
print("- 真实训练的 Loss 下降模式与此类似")
print("- 真实训练需要 GPU 和 transformers 库支持")
print("- 代码框架是真实的，可以运行真实训练")
