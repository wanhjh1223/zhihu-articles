# LLM-VLM 训练框架

一个完整的从数据收集到大型语言模型(LLM)和视觉语言模型(VLM)的训练框架。

## 项目架构

```
llm-vlm-framework/
├── configs/                    # 配置文件目录
│   ├── llm/                   # LLM训练配置
│   ├── vlm/                   # VLM训练配置
│   └── data/                  # 数据处理配置
├── src/                        # 源代码
│   ├── data_collection/       # 数据收集模块
│   │   ├── wechat/           # 微信数据获取
│   │   ├── preprocessing/    # 数据清洗预处理
│   │   └── dataset_builder/  # 训练数据集构建
│   ├── llm_training/          # LLM训练框架
│   │   ├── models/           # 模型定义
│   │   ├── training/         # 训练流程
│   │   └── rlhf/             # 强化学习训练
│   ├── vlm_training/          # VLM训练框架
│   │   ├── vision_encoder/   # 视觉编码器
│   │   ├── multimodal_fusion/# 多模态融合
│   │   └── training/         # 训练流程
│   └── common/                # 公共组件
│       ├── utils/            # 工具函数
│       ├── data_loader/      # 数据加载器
│       ├── evaluation/       # 评估工具
│       └── deployment/       # 部署接口
├── scripts/                    # 训练脚本
├── examples/                   # 示例代码
├── tests/                      # 测试代码
└── docs/                       # 文档
```

## 代码统计

- **Python 源代码**: ~8700 行
- **Shell 脚本**: ~1200 行
- **配置文件**: 10+
- **文档**: 6+
- **总计**: ~10000+ 行代码

## 功能模块

### 1. 数据收集模块 ✅
- **微信数据获取**: 支持聊天记录、公众号文章导出
- **数据清洗**: 文本清洗、去重、质量过滤
- **数据集构建**: 转换为训练所需格式（JSONL、Parquet等）

### 2. LLM 训练框架 ✅
- **基础模型**: 支持 Qwen、LLaMA、Baichuan、ChatGLM 等
- **预训练**: 从头预训练或继续预训练 (515行)
- **微调 (SFT)**: 监督微调训练 (333行)
- **强化学习**: 
  - DPO 对齐 (293行) ✅
  - GRPO 对齐 (457行) ✅
  - PPO RLHF (504行) ✅
  - 奖励模型训练 (395行) ✅

### 3. VLM 训练框架 ✅
- **视觉编码器**: CLIP、SigLIP 等 (312行)
- **多模态融合**: 投影层、连接器 (303行)
- **VLM 训练**: 预训练 + SFT (340+212行)

### 4. 评估与部署 ✅
- **评估模块**: PPL、BLEU、ROUGE、MCQA (491行)
- **部署服务**: API服务、Gradio UI
- **统一Pipeline**: 整合所有训练阶段 (433行)
- **视觉编码器**: CLIP、EVA、SigLIP 等视觉编码器
- **多模态融合**: 投影层、Connector 设计
- **预训练**: 图文对齐预训练
- **指令微调**: 多模态指令跟随训练

### 4. 工程化组件
- **数据加载器**: 高效的多模态数据加载
- **训练脚本**: 完整的训练流程脚本
- **配置文件**: YAML/JSON 配置管理
- **评估工具**: 自动化模型评估
- **部署接口**: API 服务部署

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据收集

```bash
# 导出微信聊天记录
python src/data_collection/wechat/export_chats.py --output ./data/raw/chats

# 数据清洗
python src/data_collection/preprocessing/clean_data.py \
    --input ./data/raw/chats \
    --output ./data/processed/cleaned

# 构建训练数据集
python src/data_collection/dataset_builder/build_dataset.py \
    --input ./data/processed/cleaned \
    --output ./data/datasets/train.jsonl
```

### 统一Pipeline训练（推荐）

使用统一的pipeline脚本进行全流程训练：

```bash
# 1. 预训练
python pipeline.py pretrain \
    --model_name Qwen/Qwen2.5-7B \
    --train_data ./data/pretrain/train.jsonl \
    --output_dir ./outputs/pretrain \
    --epochs 3

# 2. SFT微调
python pipeline.py sft \
    --model_name ./outputs/pretrain/final \
    --train_data ./data/sft/train.jsonl \
    --output_dir ./outputs/sft \
    --epochs 3

# 3a. DPO对齐
python pipeline.py dpo \
    --model_name ./outputs/sft/final \
    --train_data ./data/dpo/train.jsonl \
    --output_dir ./outputs/dpo \
    --beta 0.1

# 3b. GRPO对齐 (DeepSeek-R1风格)
python pipeline.py grpo \
    --model_name ./outputs/sft/final \
    --train_data ./data/grpo/train.jsonl \
    --output_dir ./outputs/grpo \
    --group_size 4

# 3c. 训练奖励模型
python pipeline.py reward \
    --model_name ./outputs/sft/final \
    --train_data ./data/reward/train.jsonl \
    --output_dir ./outputs/reward_model

# 3d. PPO RLHF
python pipeline.py ppo \
    --model_name ./outputs/sft/final \
    --reward_model ./outputs/reward_model/final \
    --train_data ./data/ppo/train.jsonl \
    --output_dir ./outputs/ppo

# 4a. VLM预训练
python pipeline.py vlm_pretrain \
    --model_name ./outputs/sft/final \
    --vision_model openai/clip-vit-base-patch32 \
    --train_data ./data/vlm/pretrain.jsonl \
    --output_dir ./outputs/vlm_pretrain

# 4b. VLM指令微调
python pipeline.py vlm_sft \
    --model_name ./outputs/vlm_pretrain/final \
    --train_data ./data/vlm/sft.jsonl \
    --output_dir ./outputs/vlm_sft

# 5. 模型评估
python pipeline.py eval \
    --model_path ./outputs/sft/final \
    --eval_data ./data/eval/eval.json \
    --eval_tasks perplexity,generation,mcqa \
    --output_dir ./eval_results
```

### 独立脚本训练

## 配置说明

所有训练参数通过配置文件管理，详见 `configs/` 目录：

- `configs/llm/pretrain.yaml` - LLM预训练配置
- `configs/llm/sft.yaml` - LLM微调配置
- `configs/llm/dpo.yaml` - DPO训练配置
- `configs/vlm/pretrain.yaml` - VLM预训练配置
- `configs/vlm/sft.yaml` - VLM指令微调配置

## 支持的模型

### LLM 基础模型
- Qwen2 / Qwen2.5 (推荐)
- LLaMA 2 / LLaMA 3
- Baichuan 2
- ChatGLM 3 / GLM-4
- InternLM 2

### VLM 视觉编码器
- CLIP ViT (OpenAI)
- SigLIP (Google)
- EVA-CLIP (BAAI)
- InternViT (Shanghai AI Lab)

## 硬件要求

| 训练阶段 | 显存需求 | 推荐配置 |
|---------|---------|---------|
| LLM SFT (7B) | 16GB+ | RTX 4090 / A100 40G |
| LLM SFT (14B) | 32GB+ | A100 40G / A100 80G |
| LLM Pretrain | 80GB+ | 多卡 A100 80G |
| VLM Pretrain | 40GB+ | A100 40G+ |
| RLHF/DPO | 40GB+ | A100 40G+ |

## 开发计划

- [x] 基础框架搭建
- [x] 数据收集模块
- [x] LLM 训练框架
- [x] VLM 训练框架
- [x] 工程化组件
- [ ] 分布式训练优化
- [ ] 更多模型支持
- [ ] Web UI 界面

## 许可证

Apache 2.0 License

## 致谢

感谢开源社区的优秀项目：
- Transformers (HuggingFace)
- DeepSpeed (Microsoft)
- LLaMA-Factory
- XTuner
