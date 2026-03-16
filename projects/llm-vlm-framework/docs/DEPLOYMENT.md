# 部署指南

## 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+)
- **Python**: 3.8+
- **内存**: 16GB+ (推荐 32GB)
- **存储**: 50GB+ 可用空间
- **GPU**: 可选，支持 CUDA 11.7+

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/wanhjh1223/llm-vlm-framework.git
cd llm-vlm-framework
```

### 2. 创建虚拟环境（推荐）

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
# CPU版本
pip install -r requirements.txt

# GPU版本（如果有CUDA）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 4. 验证安装

```bash
python test_from_scratch_simple.py
```

## 数据准备

### 预训练数据

```bash
mkdir -p data/pretrain

# 准备JSONL格式数据
cat > data/pretrain/train.jsonl << 'EOF'
{"text": "人工智能是计算机科学的一个分支..."}
{"text": "机器学习是人工智能的核心技术..."}
EOF
```

### SFT数据

```bash
mkdir -p data/sft

# 示例数据已包含在仓库中
cat data/sft/train.jsonl
```

## 运行训练

### 方式1：命令行

```bash
# 预训练
python scripts/train_pretrain.py \
    --model_name gpt2 \
    --train_data_path ./data/pretrain/train.jsonl \
    --output_dir ./outputs/pretrain

# SFT
python scripts/train_sft.py \
    --base_model ./outputs/pretrain \
    --train_data_path ./data/sft/train.jsonl \
    --output_dir ./outputs/sft
```

### 方式2：Web UI

```bash
python web_ui.py
# 访问 http://localhost:7860
```

### 方式3：Docker

```bash
# 构建镜像
docker build -t llm-vlm-framework .

# 运行容器
docker run -p 7860:7860 -v $(pwd)/data:/app/data llm-vlm-framework
```

## 生产环境部署

### 使用Docker Compose

```yaml
version: '3.8'

services:
  llm-training:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 使用Systemd服务

```bash
# 创建服务文件
sudo tee /etc/systemd/system/llm-vlm.service > /dev/null <<EOF
[Unit]
Description=LLM-VLM Training Framework
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/llm-vlm-framework
ExecStart=/home/ubuntu/llm-vlm-framework/venv/bin/python web_ui.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# 启动服务
sudo systemctl enable llm-vlm
sudo systemctl start llm-vlm
```

## 常见问题

### Q: 显存不足怎么办？

A: 使用4-bit量化或LoRA
```python
model = LLMModel(
    model_name_or_path="model_name",
    load_in_4bit=True,  # 启用4-bit量化
)
```

### Q: 训练速度慢？

A: 启用Flash Attention和混合精度
```python
model = LLMModel(
    use_flash_attention=True,  # 启用Flash Attention
    torch_dtype="bf16",        # 使用BF16
)
```

### Q: 如何恢复训练？

A: 从检查点继续
```bash
python scripts/train_pretrain.py \
    --resume_from_checkpoint ./outputs/pretrain/checkpoint-1000
```

## 性能优化

### CPU训练优化

```bash
# 设置线程数
export OMP_NUM_THREADS=8

# 使用Intel MKL
export KMP_AFFINITY=granularity=fine,compact,1,0
```

### GPU训练优化

```bash
# 使用DeepSpeed
deepspeed scripts/train_pretrain.py --deepspeed configs/ds_config.json

# 多卡训练
torchrun --nproc_per_node=4 scripts/train_pretrain.py
```

## 监控和日志

### TensorBoard

```bash
tensorboard --logdir=./runs --host 0.0.0.0 --port 6006
```

### 查看日志

```bash
# 实时查看
tail -f logs/training.log

# 查看最后100行
tail -n 100 logs/training.log
```

## 安全建议

1. **不要上传敏感数据**到GitHub
2. **定期备份**训练好的模型
3. **使用密钥管理**API Token
4. **限制Web UI访问**（使用Nginx反向代理）

## 获取帮助

- GitHub Issues: https://github.com/wanhjh1223/llm-vlm-framework/issues
- 文档: https://github.com/wanhjh1223/llm-vlm-framework/tree/main/docs
