#!/bin/bash
# LLM DPO 对齐训练脚本

set -e

echo "========================================="
echo "开始 LLM DPO 训练"
echo "========================================="

# 配置
MODEL_PATH=${MODEL_PATH:-"./outputs/llm_sft/final"}
TRAIN_DATA=${TRAIN_DATA:-"./data/preference/train.jsonl"}
EVAL_DATA=${EVAL_DATA:-"./data/preference/val.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/llm_dpo"}

echo "模型: $MODEL_PATH"
echo "训练数据: $TRAIN_DATA"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 运行训练
python -m src.llm_training.rlhf.dpo_trainer \
    --mode dpo \
    --model "$MODEL_PATH" \
    --train-data "$TRAIN_DATA" \
    --eval-data "$EVAL_DATA" \
    --output "$OUTPUT_DIR"

echo ""
echo "========================================="
echo "LLM DPO 训练完成"
echo "模型保存路径: $OUTPUT_DIR/final"
echo "========================================="
