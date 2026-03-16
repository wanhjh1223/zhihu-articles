#!/bin/bash
# ============================================
# LLM 训练全流程串联脚本
# ============================================
# 按顺序执行: 预训练 -> SFT -> DPO
# 每个阶段的输出作为下一阶段的输入

set -e

echo "=========================================="
echo "LLM 训练全流程串联"
echo "=========================================="
echo ""

# 全局配置
PROJECT_NAME=${PROJECT_NAME:-"my_llm_project"}
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen2.5-7B"}
DATA_ROOT=${DATA_ROOT:-"./data"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"./outputs/$PROJECT_NAME"}

# 创建输出目录
mkdir -p "$OUTPUT_ROOT"

echo "项目配置:"
echo "  项目名称: $PROJECT_NAME"
echo "  基座模型: $BASE_MODEL"
echo "  数据目录: $DATA_ROOT"
echo "  输出目录: $OUTPUT_ROOT"
echo ""

# ============================================
# 阶段 1: 预训练
# ============================================
STAGE1_OUTPUT="$OUTPUT_ROOT/llm_pretrain"

if [ -d "$STAGE1_OUTPUT/final" ]; then
    echo "[阶段 1/3] 预训练模型已存在，跳过..."
else
    echo "[阶段 1/3] 开始预训练..."
    echo "=========================================="
    
    # 检查数据
    if [ ! -f "$DATA_ROOT/pretrain/train.jsonl" ]; then
        echo "错误: 预训练数据不存在: $DATA_ROOT/pretrain/train.jsonl"
        echo "请准备数据或运行测试脚本"
        exit 1
    fi
    
    bash scripts/llm/pretrain/run.sh \
        --model "$BASE_MODEL" \
        --train_data "$DATA_ROOT/pretrain/train.jsonl" \
        --eval_data "$DATA_ROOT/pretrain/val.jsonl" \
        --output "$STAGE1_OUTPUT" \
        --epochs 3 \
        --batch_size 4
    
    echo ""
    echo "✓ 预训练完成"
    echo "  模型路径: $STAGE1_OUTPUT/final"
fi

# 设置预训练模型为后续阶段的输入
PRETRAIN_MODEL="$STAGE1_OUTPUT/final"

echo ""

# ============================================
# 阶段 2: SFT 监督微调
# ============================================
STAGE2_OUTPUT="$OUTPUT_ROOT/llm_sft"

if [ -d "$STAGE2_OUTPUT/final" ]; then
    echo "[阶段 2/3] SFT 模型已存在，跳过..."
else
    echo "[阶段 2/3] 开始 SFT 监督微调..."
    echo "=========================================="
    
    # 检查数据
    if [ ! -f "$DATA_ROOT/sft/train.jsonl" ]; then
        echo "错误: SFT 数据不存在: $DATA_ROOT/sft/train.jsonl"
        exit 1
    fi
    
    export PRETRAIN_MODEL
    bash scripts/llm/sft/run.sh \
        --pretrain_model "$PRETRAIN_MODEL" \
        --train_data "$DATA_ROOT/sft/train.jsonl" \
        --eval_data "$DATA_ROOT/sft/val.jsonl" \
        --output "$STAGE2_OUTPUT" \
        --epochs 3 \
        --use_lora true
    
    echo ""
    echo "✓ SFT 完成"
    echo "  模型路径: $STAGE2_OUTPUT/final"
fi

# 设置 SFT 模型为后续阶段的输入
SFT_MODEL="$STAGE2_OUTPUT/final"

echo ""

# ============================================
# 阶段 3: DPO 对齐训练
# ============================================
STAGE3_OUTPUT="$OUTPUT_ROOT/llm_dpo"

if [ -d "$STAGE3_OUTPUT/final" ]; then
    echo "[阶段 3/3] DPO 模型已存在，跳过..."
else
    echo "[阶段 3/3] 开始 DPO 对齐训练..."
    echo "=========================================="
    
    # 检查数据
    if [ ! -f "$DATA_ROOT/preference/train.jsonl" ]; then
        echo "警告: 偏好数据不存在，跳过 DPO 阶段"
        echo "  数据路径: $DATA_ROOT/preference/train.jsonl"
        SFT_MODEL_FINAL="$SFT_MODEL"
    else
        export SFT_MODEL
        bash scripts/llm/dpo/run.sh \
            --sft_model "$SFT_MODEL" \
            --train_data "$DATA_ROOT/preference/train.jsonl" \
            --output "$STAGE3_OUTPUT" \
            --beta 0.1
        
        echo ""
        echo "✓ DPO 完成"
        echo "  模型路径: $STAGE3_OUTPUT/final"
        SFT_MODEL_FINAL="$STAGE3_OUTPUT/final"
    fi
fi

echo ""
echo "=========================================="
echo "LLM 训练全流程完成!"
echo "=========================================="
echo ""
echo "最终模型路径: $SFT_MODEL_FINAL"
echo ""
echo "部署模型:"
echo "  python -m src.common.deployment.api_server --model $SFT_MODEL_FINAL"
echo ""
echo "评估模型:"
echo "  python -m src.common.evaluation.evaluate --model $SFT_MODEL_FINAL --data ./data/eval/eval.jsonl"
echo ""
