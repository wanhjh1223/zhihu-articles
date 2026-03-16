#!/bin/bash
# ============================================
# LLM 监督微调脚本 (SFT - Supervised Fine-Tuning)
# ============================================
# 说明:
# - 基于预训练模型或基座模型进行指令微调
# - 数据格式: Instruction-following (Conversation/Alpaca)
#
# 数据准备:
# - Conversation 格式: {"messages": [{"role": "user", "content": "..."}, ...]}
# - Alpaca 格式: {"instruction": "...", "input": "...", "output": "..."}
#
# 阶段串联:
# - 输入: 预训练模型 (可选)
# - 输出: SFT 模型，可用于 DPO/RLHF

set -e

echo "=========================================="
echo "LLM 监督微调阶段 (SFT)"
echo "=========================================="
echo ""

# 配置
CONFIG_FILE=${CONFIG_FILE:-"configs/llm/sft/config.yaml"}

# 模型配置 - 可从预训练模型继续
PRETRAIN_MODEL=${PRETRAIN_MODEL:-""}  # 如果为空，使用 config 中的模型
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen2.5-7B"}

TRAIN_DATA=${TRAIN_DATA:-"./data/sft/train.jsonl"}
EVAL_DATA=${EVAL_DATA:-"./data/sft/val.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/llm_sft"}

# 训练参数
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-2e-4}
MAX_LENGTH=${MAX_LENGTH:-2048}

# LoRA 配置
USE_LORA=${USE_LORA:-"true"}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-32}

echo "训练配置:"

# 确定使用的模型
if [ -n "$PRETRAIN_MODEL" ] && [ -d "$PRETRAIN_MODEL" ]; then
    MODEL_NAME="$PRETRAIN_MODEL"
    echo "  从预训练模型继续: $MODEL_NAME"
else
    MODEL_NAME="$BASE_MODEL"
    echo "  基座模型: $MODEL_NAME"
fi

echo "  训练数据: $TRAIN_DATA"
echo "  验证数据: $EVAL_DATA"
echo "  输出目录: $OUTPUT_DIR"
echo "  使用 LoRA: $USE_LORA"
if [ "$USE_LORA" = "true" ]; then
    echo "  LoRA r: $LORA_R, alpha: $LORA_ALPHA"
fi
echo ""

# 检查数据
if [ ! -f "$TRAIN_DATA" ]; then
    echo "错误: 训练数据不存在: $TRAIN_DATA"
    echo "请准备 SFT 数据，格式示例:"
    echo '  {"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}]}'
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "开始 SFT 微调..."
echo ""

python -m src.llm_training.training.sft_trainer \
    --config "$CONFIG_FILE" \
    --model_name "$MODEL_NAME" \
    --train_data "$TRAIN_DATA" \
    --eval_data "$EVAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --max_length "$MAX_LENGTH" \
    --use_lora "$USE_LORA" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --bf16 \
    --gradient_checkpointing

echo ""
echo "=========================================="
echo "SFT 微调完成!"
echo "模型保存路径: $OUTPUT_DIR/final"
echo "=========================================="
echo ""
echo "下一步可选:"
echo "  1. 直接部署: python -m src.common.deployment.api_server --model $OUTPUT_DIR/final"
echo "  2. DPO 对齐: bash scripts/llm/dpo.sh"
echo "  3. RLHF 训练: bash scripts/llm/rlhf.sh"
echo ""
