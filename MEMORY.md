# MEMORY.md - GitHub 仓库速查手册

> ⚠️ **推送前必看！** 每次推送到 GitHub 前，先确认文件应该推送到哪个仓库！

---

## 📦 仓库列表

| 仓库名 | 功能/用途 | 本地路径 | 远程地址 |
|--------|-----------|----------|----------|
| **zhihu-articles** | 技术研究文章（LLM、VLA论文分析） | `/root/.openclaw/workspace/` | `https://github.com/wanhjh1223/zhihu-articles.git` |
| **data_process** | 数据处理项目集合 | `/root/.openclaw/workspace/data_process/` | `https://github.com/wanhjh1223/data_process.git` |

---

## 📁 目录结构

```
/root/.openclaw/workspace/           ← zhihu-articles 仓库根
├── llm-大模型技术/                   ← 论文分析 → zhihu-articles
├── vla-自动驾驶大模型/                ← 论文分析 → zhihu-articles
├── data_process/                    ← ⚠️ 独立仓库！→ data_process
│   └── medical-kg-pretrain/         ← 数据处理项目
│       ├── process_medical_data.py
│       └── README.md
└── memory/                          ← 本地记忆（不推送）
```

---

## ✅ 推送前检查清单

### 如果是技术文章 (*.md 论文分析)
```bash
cd /root/.openclaw/workspace
git add .
git commit -m "xxx"
git push origin main   # 推送到 zhihu-articles
```

### 如果是数据处理脚本
```bash
cd /root/.openclaw/workspace/data_process
git add .
git commit -m "xxx"
git push origin main   # 推送到 data_process
```

### ⚠️ 禁忌
- [x] 不要把 `data_process/` 嵌套进 `zhihu-articles`
- [x] 不要把技术文章放进 `data_process/`
- [x] `zhihu-articles` 已配置 `.gitignore` 排除 `data_process/`

---

## 🔧 常用命令

```bash
# 查看当前仓库信息
git remote -v
git log --oneline -3

# 确认当前目录对应的仓库
pwd && git remote get-url origin
```

---

*最后更新: 2026-03-24*

## 🚨 数据处理任务规范（重要！）

> **每次处理数据前，必须先读取 `data_process/TASK_SPEC.md`！**

### 核心要求

1. **输出格式**：必须是 JSONL，每条数据包含三个字段：
   - `id`: `openclaw_{递增整数}` 格式
   - `type`: 数据类型（pretrain/code/math/instruction等）
   - `text`: 训练文本内容

2. **Token 长度控制**：
   - 最大 4096 tokens
   - 必须使用 tokenizer 计算
   - 超长文本必须切分，不能丢弃

3. **文件分片**：
   - 每个 JSONL 最大 100000 条
   - 命名：`dataset_part_00000.jsonl`

4. **GitHub 发布**：
   - train/test/validation 必须分开创建独立 Release
   - 示例：`v1.0-train`, `v1.0-validation`

5. **仓库结构**：
   ```
   repo/
   ├── scripts/          # 处理脚本
   ├── examples/         # ≤100条示例
   ├── README.md
   └── task_spec.md
   ```

### 参考实现
- `data_process/medical-kg-pretrain/`
- `data_process/huatuo-encyclopedia-qa/`

## 🔑 HuggingFace Token

用于访问 HuggingFace Hub 下载数据集。

**获取方式**：https://huggingface.co/settings/tokens → New token → 选择 Read 权限

**使用方法**：
```python
import os
os.environ["HF_TOKEN"] = "hf_xxx..."  # 运行时从用户获取

from datasets import load_dataset
ds = load_dataset("user/dataset", token=os.environ["HF_TOKEN"])
```

**注意**：Token 仅保存在当前会话，不写入文件。每次需要时询问用户。
