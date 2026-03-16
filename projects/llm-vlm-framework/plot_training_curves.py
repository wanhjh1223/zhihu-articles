#!/usr/bin/env python3
"""
绘制训练曲线图
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 加载数据
with open('docs/training_data.json', 'r') as f:
    data = json.load(f)

steps = data['steps']
losses = data['loss']
lrs = data['lr']

# 创建图表
fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=150)
fig.suptitle('LLM-VLM Framework Training Demo\n(From Scratch, 100 Steps)', fontsize=14, fontweight='bold')

# Loss曲线
ax1 = axes[0]
ax1.plot(steps, losses, 'b-', linewidth=1.5, alpha=0.8, label='Loss')
ax1.fill_between(steps, losses, alpha=0.2)
ax1.axhline(y=min(losses), color='r', linestyle='--', alpha=0.5, label=f'Min: {min(losses):.4f}')
ax1.set_xlabel('Step', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Training Loss', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 添加起始和结束标注
ax1.annotate(f'Start: {losses[0]:.4f}', xy=(0, losses[0]), xytext=(10, losses[0]+0.01),
            arrowprops=dict(arrowstyle='->', color='green'), fontsize=9, color='green')
ax1.annotate(f'End: {losses[-1]:.4f}', xy=(99, losses[-1]), xytext=(80, losses[-1]+0.01),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')

# 学习率曲线
ax2 = axes[1]
ax2.plot(steps, lrs, 'g-', linewidth=2)
ax2.set_xlabel('Step', fontsize=11)
ax2.set_ylabel('Learning Rate', fontsize=11)
ax2.set_title('Learning Rate', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, max(lrs) * 1.2)

plt.tight_layout()
plt.savefig('docs/images/training_curves.png', dpi=150, bbox_inches='tight')
print(f"✓ 训练曲线图已保存: docs/images/training_curves.png")

# 单独保存Loss大图
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
ax.plot(steps, losses, 'b-', linewidth=2, label='Training Loss')
ax.fill_between(steps, losses, alpha=0.3)
ax.set_xlabel('Step', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('LLM-VLM Framework - Training Loss Curve\n(From Scratch Training Demo)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

# 添加统计信息文本框
stats_text = f"Initial Loss: {losses[0]:.4f}\nFinal Loss: {losses[-1]:.4f}\nMin Loss: {min(losses):.4f}\nSteps: 100"
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('docs/images/loss_curve.png', dpi=150, bbox_inches='tight')
print(f"✓ Loss曲线图已保存: docs/images/loss_curve.png")

plt.close('all')
print("✓ 所有图表生成完成!")
