#!/bin/bash
# ============================================
# LLM 预训练脚本 (Pre-training)
# ============================================
# 使用方式:
#   方式1: 直接运行（使用默认配置）
#     bash scripts/llm/pretrain/run.sh
#
#   方式2: 指定参数
#     bash scripts/llm/pretrain/run.sh \
#       --model Qwen/Qwen2.5-7B \
#       --train_data ./data/pretrain/train.jsonl \
#       --output ./outputs/llm_pretrain
#
#   方式3: 使用配置文件
#     CONFIG_FILE=configs/llm/pretrain/config.yaml bash scripts/llm/pretrain/run.sh

set -e

echo "=========================================="
echo "LLM 预训练阶段 (Pre-training)"
echo "=========================================="
echo ""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --train_data)
            TRAIN_DATA="$2"
            shift 2
            ;;
        --eval_data)
            EVAL_DATA="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "使用方法: bash scripts/llm/pretrain/run.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --model         模型名称或路径 (默认: Qwen/Qwen2.5-7B)"
            echo "  --train_data    训练数据路径 (默认: ./data/pretrain/train.jsonl)"
            echo "  --eval_data     验证数据路径 (可选)"
            echo "  --output        输出目录 (默认: ./outputs/llm_pretrain)"
            echo "  --epochs        训练轮数 (默认: 3)"
            echo "  --batch_size    每设备batch size (默认: 4)"
            echo "  --lr            学习率 (默认: 1e-4)"
            echo "  --max_length    最大序列长度 (默认: 2048)"
            echo "  --config        配置文件路径"
            echo ""
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 默认配置
CONFIG_FILE=${CONFIG_FILE:-""}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-7B"}
TRAIN_DATA=${TRAIN_DATA:-"./data/pretrain/train.jsonl"}
EVAL_DATA=${EVAL_DATA:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/llm_pretrain"}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-1e-4}
MAX_LENGTH=${MAX_LENGTH:-2048}

# 检查数据文件
if [ ! -f "$TRAIN_DATA" ]; then
    echo "错误: 训练数据文件不存在: $TRAIN_DATA"
    echo ""
    echo "请准备预训练数据，格式如下:"
    echo '  {"text": "这里是预训练文本内容..."}'
    echo ""
    echo "推荐数据集:"
    echo "  - WuDaoCorpora (200G): https://data.baai.ac.cn/details/WuDaoCorporaText"
    echo "  - Firefly-Pretrain (22G): https://huggingface.co/datasets/YeungNLP/firefly-pretrain-dataset"
    echo "  - MNBVC (26TB+): https://github.com/esbatmop/MNBVC"
    echo ""
    echo "快速开始测试:"
    echo "  bash scripts/llm/pretrain/test.sh"
    exit 1
fi

echo "训练配置:"
echo "  模型: $MODEL_NAME"
echo "  训练数据: $TRAIN_DATA"
if [ -n "$EVAL_DATA" ]; then
    echo "  验证数据: $EVAL_DATA"
fi
echo "  输出目录: $OUTPUT_DIR"
echo "  训练轮数: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  梯度累积: $GRAD_ACCUM"
echo "  学习率: $LR"
echo "  最大长度: $MAX_LENGTH"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 构建 Python 命令
PYTHON_CMD="python -m src.llm_training.training.pretrain_trainer"

if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    echo "使用配置文件: $CONFIG_FILE"
    PYTHON_CMD="$PYTHON_CMD --config $CONFIG_FILE"
else
    PYTHON_CMD="$PYTHON_CMD \
        --model $MODEL_NAME \
        --train_data $TRAIN_DATA"
    
    if [ -n "$EVAL_DATA" ]; then
        PYTHON_CMD="$PYTHON_CMD --eval_data $EVAL_DATA"
    fi
    
    PYTHON_CMD="$PYTHON_CMD \
        --output_dir $OUTPUT_DIR \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --max_length $MAX_LENGTH"
fi

# 运行训练
echo "开始预训练..."
echo "命令: $PYTHON_CMD"
echo ""
eval $PYTHON_CMD

echo ""
echo "=========================================="
echo "预训练完成!"
echo "模型保存路径: $OUTPUT_DIR/final"
echo "=========================================="
echo ""
echo "下一步建议:"
echo "  1. 运行 SFT 微调: bash scripts/llm/sft/run.sh"
echo "  2. 评估模型效果: python -m src.common.evaluation.evaluate --model $OUTPUT_DIR/final"
echo "  3. 部署模型: python -m src.common.deployment.api_server --model $OUTPUT_DIR/final"
echo ""
