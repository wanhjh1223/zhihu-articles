#!/bin/bash
# ============================================
# VLM 视觉-语言对齐脚本 (Alignment/Pre-training)
# ============================================
# 说明:
# - 冻结 LLM 和 Vision Encoder，训练 Projector
# - 建立视觉特征和语言模型的对齐
# - 数据格式: 图文对 (Image-Caption)
#
# 数据准备:
# - JSONL 格式: {"image": "path/to/image.jpg", "caption": "图片描述"}
# - 或 Conversation 格式: {"image": "...", "messages": [...]}
#
# 阶段串联:
# - 输入: LLM 模型 (可选) + Vision Encoder
# - 输出: 对齐后的 VLM 模型

set -e

echo "=========================================="
echo "VLM 视觉-语言对齐阶段 (Alignment)"
echo "=========================================="
echo ""

CONFIG_FILE=${CONFIG_FILE:-"configs/vlm/alignment/config.yaml"}

# 模型配置
LLM_MODEL=${LLM_MODEL:-"Qwen/Qwen2.5-7B"}
VISION_ENCODER=${VISION_ENCODER:-"clip-vit-large-336"}

# 数据配置
TRAIN_DATA=${TRAIN_DATA:-"./data/vlm_alignment/train.jsonl"}
EVAL_DATA=${EVAL_DATA:-"./data/vlm_alignment/val.jsonl"}
IMAGE_FOLDER=${IMAGE_FOLDER:-"./data/images"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/vlm_alignment"}

# 训练参数
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-1e-3}

echo "训练配置:"
echo "  LLM 模型: $LLM_MODEL"
echo "  视觉编码器: $VISION_ENCODER"
echo "  训练数据: $TRAIN_DATA"
echo "  图像目录: $IMAGE_FOLDER"
echo "  输出目录: $OUTPUT_DIR"
echo "  学习率: $LR (较大，只训练 projector)"
echo ""

# 检查数据
if [ ! -f "$TRAIN_DATA" ]; then
    echo "错误: 训练数据不存在: $TRAIN_DATA"
    echo "数据格式示例 (Caption):"
    echo '  {"image": "0001.jpg", "caption": "一只猫在沙发上睡觉"}'
    echo ""
    echo "数据格式示例 (Conversation):"
    echo '  {"image": "0001.jpg", "messages": [{"role": "user", "content": "<image>\n描述这张图片"}, ...]}'
    exit 1
fi

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "警告: 图像目录不存在: $IMAGE_FOLDER"
fi

mkdir -p "$OUTPUT_DIR"

echo "开始对齐训练..."
echo ""

python -m src.vlm_training.training.alignment_trainer \
    --config "$CONFIG_FILE" \
    --llm_model "$LLM_MODEL" \
    --vision_encoder "$VISION_ENCODER" \
    --train_data "$TRAIN_DATA" \
    --eval_data "$EVAL_DATA" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --bf16

echo ""
echo "=========================================="
echo "对齐训练完成!"
echo "模型保存路径: $OUTPUT_DIR/final"
echo "=========================================="
echo ""
echo "关键信息:"
echo "  - LLM 和 Vision Encoder 已冻结"
echo "  - 只训练了 Projector (Connector)"
echo ""
echo "下一步:"
echo "  运行 VLM SFT: bash scripts/vlm/sft/run.sh"
echo ""
