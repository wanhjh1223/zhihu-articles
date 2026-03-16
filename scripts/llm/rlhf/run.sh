#!/bin/bash
# ============================================
# RLHF (PPO) 训练脚本
# ============================================
# 说明:
# - 基于奖励模型使用 PPO 算法优化
# - 需要训练好的奖励模型
#
# 阶段串联:
# - 输入: SFT 模型 + 奖励模型
# - 输出: PPO 优化模型

set -e

echo "=========================================="
echo "RLHF (PPO) 训练"
echo "=========================================="
echo ""

CONFIG_FILE=${CONFIG_FILE:-"configs/llm/rlhf/config.yaml"}

# 模型配置
SFT_MODEL=${SFT_MODEL:-"./outputs/llm_sft/final"}
REWARD_MODEL=${REWARD_MODEL:-"./outputs/reward_model/final"}

# 数据配置
TRAIN_DATA=${TRAIN_DATA:-"./data/rlhf/train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/llm_ppo"}

# PPO 参数
KL_COEF=${KL_COEF:-0.2}
LEARNING_RATE=${LEARNING_RATE:-1e-5}

echo "训练配置:"
echo "  Actor 模型: $SFT_MODEL"
echo "  奖励模型: $REWARD_MODEL"
echo "  训练数据: $TRAIN_DATA"
echo "  KL 系数: $KL_COEF"
echo ""

# 检查模型
if [ ! -d "$SFT_MODEL" ]; then
    echo "错误: SFT 模型不存在"
    exit 1
fi

if [ ! -d "$REWARD_MODEL" ]; then
    echo "错误: 奖励模型不存在: $REWARD_MODEL"
    echo "请先训练奖励模型"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "开始 PPO 训练..."

python -m src.llm_training.rlhf.ppo_trainer \
    --config "$CONFIG_FILE" \
    --actor_model_path "$SFT_MODEL" \
    --reward_model_path "$REWARD_MODEL" \
    --train_data "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --kl_coef "$KL_COEF" \
    --learning_rate "$LEARNING_RATE"

echo ""
echo "=========================================="
echo "PPO 训练完成!"
echo "模型保存路径: $OUTPUT_DIR/final"
echo "=========================================="
