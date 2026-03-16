#!/bin/bash
# VLM 指令微调脚本

set -e

echo "========================================="
echo "开始 VLM 指令微调"
echo "========================================="

# 配置
LLM_MODEL=${LLM_MODEL:-"./outputs/vlm_pretrain/final"}
VISION_ENCODER=${VISION_ENCODER:-"clip-vit-large"}
TRAIN_DATA=${TRAIN_DATA:-"./data/vlm_sft/train.jsonl"}
IMAGE_FOLDER=${IMAGE_FOLDER:-"./data/images"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/vlm_sft"}

echo "语言模型: $LLM_MODEL"
echo "视觉编码器: $VISION_ENCODER"
echo "训练数据: $TRAIN_DATA"
echo "图像文件夹: $IMAGE_FOLDER"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 运行训练
python -m src.vlm_training.training.trainer \
    --stage sft \
    --llm "$LLM_MODEL" \
    --vision "$VISION_ENCODER" \
    --train-data "$TRAIN_DATA" \
    --image-folder "$IMAGE_FOLDER" \
    --output "$OUTPUT_DIR"

echo ""
echo "========================================="
echo "VLM 指令微调完成"
echo "模型保存路径: $OUTPUT_DIR/final"
echo "========================================="
