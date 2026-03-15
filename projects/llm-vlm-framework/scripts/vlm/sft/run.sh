#!/bin/bash
# ============================================
# VLM 指令微调脚本 (Instruction Tuning)
# ============================================
# 说明:
# - 基于对齐后的 VLM 模型进行指令微调
# - 解冻 LLM，训练视觉理解能力
# - 数据格式: 多模态对话
#
# 数据准备:
# - JSONL 格式: {"image": "...", "messages": [{"role": "...", "content": "..."}]}
#
# 阶段串联:
# - 输入: VLM Alignment 模型
# - 输出: 最终的 VLM 模型

set -e

echo "=========================================="
echo "VLM 指令微调阶段 (SFT)"
echo "=========================================="
echo ""

CONFIG_FILE=${CONFIG_FILE:-"configs/vlm/sft/config.yaml"}

# 模型配置 - 从对齐阶段加载
ALIGNMENT_MODEL=${ALIGNMENT_MODEL:-"./outputs/vlm_alignment/final"}

# 数据配置
TRAIN_DATA=${TRAIN_DATA:-"./data/vlm_sft/train.jsonl"}
EVAL_DATA=${EVAL_DATA:-"./data/vlm_sft/val.jsonl"}
IMAGE_FOLDER=${IMAGE_FOLDER:-"./data/images"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/vlm_sft"}

# 训练参数
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-2e-5}

# LoRA 配置
USE_LORA=${USE_LORA:-"false"}

echo "训练配置:"
echo "  对齐模型: $ALIGNMENT_MODEL"
echo "  训练数据: $TRAIN_DATA"
echo "  图像目录: $IMAGE_FOLDER"
echo "  输出目录: $OUTPUT_DIR"
echo "  使用 LoRA: $USE_LORA"
echo ""

# 检查模型
if [ ! -d "$ALIGNMENT_MODEL" ]; then
    echo "错误: 对齐模型不存在: $ALIGNMENT_MODEL"
    echo "请先运行对齐阶段: bash scripts/vlm/alignment/run.sh"
    exit 1
fi

# 检查数据
if [ ! -f "$TRAIN_DATA" ]; then
    echo "错误: 训练数据不存在: $TRAIN_DATA"
    echo "数据格式示例:"
    echo '  {'
    echo '    "image": "0001.jpg",'
    echo '    "messages": ['
    echo '      {"role": "user", "content": "<image>\n这张图片里有什么?"},'
    echo '      {"role": "assistant", "content": "图片里有一只猫..."}'
    echo '    ]'
    echo '  }'
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "开始 VLM SFT 训练..."
echo ""

python -m src.vlm_training.training.vlm_sft_trainer \
    --config "$CONFIG_FILE" \
    --model_path "$ALIGNMENT_MODEL" \
    --train_data "$TRAIN_DATA" \
    --eval_data "$EVAL_DATA" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --use_lora "$USE_LORA" \
    --bf16 \
    --gradient_checkpointing

echo ""
echo "=========================================="
echo "VLM 指令微调完成!"
echo "模型保存路径: $OUTPUT_DIR/final"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  python -m src.common.deployment.api_server --model $OUTPUT_DIR/final"
echo ""
