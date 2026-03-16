#!/bin/bash
# ============================================
# GRPO (Generalized Reward-Penalty Optimization) 训练脚本
# ============================================
# 说明:
# - 基于群组相对策略优化
# - 只需 prompt，模型生成多个回复进行对比
#
# 阶段串联:
# - 输入: SFT 模型
# - 输出: GRPO 优化模型

set -e

echo "=========================================="
echo "GRPO 训练"
echo "=========================================="
echo ""

CONFIG_FILE=${CONFIG_FILE:-"configs/llm/grpo/config.yaml"}

# 模型配置
SFT_MODEL=${SFT_MODEL:-"./outputs/llm_sft/final"}

# 数据配置
TRAIN_DATA=${TRAIN_DATA:-"./data/grpo/train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/llm_grpo"}

# GRPO 参数
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
KL_COEFF=${KL_COEFF:-0.01}

echo "训练配置:"
echo "  SFT 模型: $SFT_MODEL"
echo "  训练数据: $TRAIN_DATA"
echo "  每组生成数: $NUM_GENERATIONS"
echo "  KL 系数: $KL_COEFF"
echo ""

if [ ! -d "$SFT_MODEL" ]; then
    echo "错误: SFT 模型不存在"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "开始 GRPO 训练..."

python -m src.llm_training.rlhf.grpo_trainer \
    --config "$CONFIG_FILE" \
    --model_path "$SFT_MODEL" \
    --train_data "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --num_generations "$NUM_GENERATIONS" \
    --kl_coeff "$KL_COEFF"

echo ""
echo "=========================================="
echo "GRPO 训练完成!"
echo "模型保存路径: $OUTPUT_DIR/final"
echo "=========================================="
