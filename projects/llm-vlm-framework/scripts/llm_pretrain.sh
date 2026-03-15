#!/bin/bash
# LLM 预训练脚本

set -e

echo "========================================="
echo "开始 LLM 预训练"
echo "========================================="

# 配置
MODEL_NAME=${MODEL_NAME:-"qwen2.5-7b"}
TRAIN_DATA=${TRAIN_DATA:-"./data/pretrain/train.jsonl"}
EVAL_DATA=${EVAL_DATA:-"./data/pretrain/val.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/llm_pretrain"}

# 训练参数
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-1e-4}
MAX_LENGTH=${MAX_LENGTH:-2048}

echo "模型: $MODEL_NAME"
echo "训练数据: $TRAIN_DATA"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 运行训练
python -m src.llm_training.training.sft_trainer \
    --mode pretrain \
    --model "$MODEL_NAME" \
    --train-data "$TRAIN_DATA" \
    --eval-data "$EVAL_DATA" \
    --output "$OUTPUT_DIR"

echo ""
echo "========================================="
echo "LLM 预训练完成"
echo "模型保存路径: $OUTPUT_DIR/final"
echo "========================================="
