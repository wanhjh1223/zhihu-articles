#!/bin/bash
# LLM SFT 微调脚本

set -e

echo "========================================="
echo "开始 LLM SFT 微调"
echo "========================================="

# 配置
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-7B"}
TRAIN_DATA=${TRAIN_DATA:-"./data/sft/train.jsonl"}
EVAL_DATA=${EVAL_DATA:-"./data/sft/val.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/llm_sft"}

echo "模型: $MODEL_NAME"
echo "训练数据: $TRAIN_DATA"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 运行训练
python -m src.llm_training.training.sft_trainer \
    --mode sft \
    --model "$MODEL_NAME" \
    --train-data "$TRAIN_DATA" \
    --eval-data "$EVAL_DATA" \
    --output "$OUTPUT_DIR"

echo ""
echo "========================================="
echo "LLM SFT 微调完成"
echo "模型保存路径: $OUTPUT_DIR/final"
echo "========================================="
