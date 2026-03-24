# HuggingFace Token 设置教程

## 1. 获取 HuggingFace Token

1. 访问 https://huggingface.co/settings/tokens
2. 点击 "New token"
3. 选择权限（建议选 "Read" 即可）
4. 复制生成的 token（格式：hf_xxxxxx）

## 2. 在 OpenClaw 中设置 Token

### 方法一：环境变量（推荐）

```bash
# 设置环境变量（仅当前会话有效）
export HF_TOKEN=hf_你的token

# 验证设置
echo $HF_TOKEN
```

### 方法二：写入配置文件

```bash
# 创建/编辑 huggingface 配置
mkdir -p ~/.cache/huggingface
echo "token: hf_你的token" > ~/.cache/huggingface/token
```

### 方法三：直接告诉我

你可以直接把 token 发给我，我会立即使用，不会保存到文件。

**注意**：Token 只有你我知道，对话结束后就会清除。

## 3. 验证 Token 是否有效

```python
from huggingface_hub import HfApi
api = HfApi()
print(api.whoami())  # 显示当前用户信息
```

## 4. 使用 Token 下载数据集

```python
from datasets import load_dataset
import os

os.environ["HF_TOKEN"] = "hf_你的token"

dataset = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train")
```

---

**获取到 Token 后直接发给我，或者按上面的方法设置好告诉我即可！**
