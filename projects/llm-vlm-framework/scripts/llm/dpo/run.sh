#!/bin/bash
# ============================================
# DPO (Direct Preference Optimization) 训练脚本
# ============================================
# 说明:
# - 基于偏好数据对齐模型
# - 无需训练奖励模型
# - 数据格式: {"prompt": "...", "chosen": "...", "rejected": "..."}
#
# 阶段串联:
# - 输入: SFT 模型
# - 输出: DPO 对齐模型

set -e

echo "=========================================="
echo "DPO 对齐训练"
echo "=========================================="
echo ""

CONFIG_FILE=${CONFIG_FILE:-"configs/llm/dpo/config.yaml"}

# 模型配置
SFT_MODEL=${SFT_MODEL:-"./outputs/llm_sft/final"}
REF_MODEL=${REF_MODEL:-""}  # 如果为空，使用 SFT 模型作为参考

# 数据配置
TRAIN_DATA=${TRAIN_DATA:-"./data/preference/train.jsonl"}
EVAL_DATA=${EVAL_DATA:-"./data/preference/val.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/llm_dpo"}

# DPO 参数
BETA=${BETA:-0.1}                   # DPO temperature
LR=${LR:-5e-5}                      # 学习率

# LoRA 配置
USE_LORA=${USE_LORA:-"true"}

echo "训练配置:"
echo "  SFT 模型: $SFT_MODEL"
if [ -n "$REF_MODEL" ]; then
    echo "  参考模型: $REF_MODEL"
else
    echo "  参考模型: 使用 SFT 模型 (无梯度)"
fi
echo "  训练数据: $TRAIN_DATA"
echo "  DPO beta: $BETA"
echo "  学习率: $LR"
echo "  使用 LoRA: $USE_LORA"
echo ""

# 检查模型和数据
if [ ! -d "$SFT_MODEL" ]; then
    echo "错误: SFT 模型不存在: $SFT_MODEL"
    echo "请先运行 SFT 阶段: bash scripts/llm/sft.sh"
    exit 1
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "错误: 偏好数据不存在: $TRAIN_DATA"
    echo "数据格式示例:"
    echo '  {"prompt": "问题", "chosen": "好回答", "rejected": "差回答"}'
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "开始 DPO 训练..."
echo ""

python -m src.llm_training.rlhf.dpo_trainer \
    --config "$CONFIG_FILE" \
    --model_path "$SFT_MODEL" \
    --ref_model_path "$REF_MODEL" \
    --train_data "$TRAIN_DATA" \
    --eval_data "$EVAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --beta "$BETA" \
    --learning_rate "$LR" \
    --use_lora "$USE_LORA" \
    --bf16

echo ""
echo "=========================================="
echo "DPO 训练完成!"
echo "模型保存路径: $OUTPUT_DIR/final"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  python -m src.common.deployment.api_server --model $OUTPUT_DIR/final"
echo ""
