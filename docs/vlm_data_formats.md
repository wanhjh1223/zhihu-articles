# ============================================
# VLM 训练数据格式详细说明
# ============================================

## 阶段 1: 视觉-语言对齐 (Alignment)

### 格式 A: Caption 格式（图文对）

```jsonl
{
  "image": "0001.jpg",
  "caption": "一只橘色的猫正在沙发上睡觉"
}
{
  "image": "0002.jpg",
  "caption": "一座雪山倒映在平静的湖水中，天空湛蓝"
}
{
  "image": "0003.png",
  "caption": "一份代码截图，显示 Python 函数定义"
}
```

### 格式 B: Conversation 格式

```jsonl
{
  "image": "0001.jpg",
  "messages": [
    {"role": "user", "content": "<image>\n描述这张图片"},
    {"role": "assistant", "content": "图片中有一只橘色的猫，正蜷缩在沙发上睡觉。"}
  ]
}
```

### 字段说明
- `image` (required): 图像文件路径（相对于 image_folder）
- `caption` (for format A): 图像描述文本
- `messages` (for format B): 对话消息列表

### 数据准备建议
- 图像数量: 至少 10K-100K 对
- 描述长度: 20-100 词为宜
- 数据来源: LAION, CC12M, 自定义图文对

---

## 阶段 2: 指令微调 (Instruction Tuning)

### 数据格式: 多模态对话

```jsonl
{
  "image": "cat_001.jpg",
  "messages": [
    {"role": "user", "content": "<image>\n这张图片里有什么动物？"},
    {"role": "assistant", "content": "图片里有一只橘色的猫。"},
    {"role": "user", "content": "它在做什么？"},
    {"role": "assistant", "content": "它正在沙发上睡觉，看起来很放松。"}
  ]
}
{
  "image": "chart_001.png",
  "messages": [
    {"role": "user", "content": "<image>\n分析这张图表"},
    {"role": "assistant", "content": "这是一张销售趋势图，显示了2023年各季度的销售额变化。从图中可以看出：\n1. Q1 销售额为 100 万\n2. Q2 增长到 150 万\n3. Q3 达到峰值 200 万\n4. Q4 略有下降，为 180 万\n\n整体来看，2023年呈现上升趋势。"}
  ]
}
{
  "image": "doc_001.jpg",
  "messages": [
    {"role": "user", "content": "<image>\n提取图片中的文字"},
    {"role": "assistant", "content": "图片中的文字内容是：\n'会议时间：2024年1月15日 下午2点\n地点：会议室A\n参会人员：张三、李四、王五'"},
    {"role": "user", "content": "会议是什么时间？"},
    {"role": "assistant", "content": "会议时间是 2024年1月15日 下午2点。"}
  ]
}
```

### 字段说明
- `image` (required): 图像文件路径
- `messages` (required): 多轮对话列表
  - `role`: `user` 或 `assistant`
  - `content`: 消息内容，可包含 `<image>` 占位符

### 图像占位符说明
- `<image>`: 图像位置标记，会被替换为图像特征 token
- 位置: 通常放在 user message 的开头或适当位置

---

## 任务类型示例

### 图像描述 (Image Captioning)

```jsonl
{
  "image": "scene.jpg",
  "messages": [
    {"role": "user", "content": "<image>\n详细描述这张图片"},
    {"role": "assistant", "content": "这是一幅美丽的风景照片..."}
  ]
}
```

### 视觉问答 (Visual Question Answering)

```jsonl
{
  "image": "animal.jpg",
  "messages": [
    {"role": "user", "content": "<image>\n图中有几只动物？"},
    {"role": "assistant", "content": "图中有 3 只动物：2只狗和1只猫。"}
  ]
}
```

### OCR (文字识别)

```jsonl
{
  "image": "document.jpg",
  "messages": [
    {"role": "user", "content": "<image>\n提取图片中的所有文字"},
    {"role": "assistant", "content": "图片中的文字是：..."}
  ]
}
```

### 图表理解 (Chart Understanding)

```jsonl
{
  "image": "chart.png",
  "messages": [
    {"role": "user", "content": "<image>\n这张图表展示了什么趋势？"},
    {"role": "assistant", "content": "图表显示了..."}
  ]
}
```

### 多图对话

```jsonl
{
  "images": ["before.jpg", "after.jpg"],
  "messages": [
    {"role": "user", "content": "<image 1>\n<image 2>\n描述这两张图的区别"},
    {"role": "assistant", "content": "左图显示...右图显示..."}
  ]
}
```

---

## 完整训练流程示例

### 步骤 1: 准备对齐数据
```bash
# 创建目录结构
mkdir -p data/vlm_alignment/train
mkdir -p data/vlm_alignment/val
mkdir -p data/images

# 数据格式: {"image": "...", "caption": "..."}
# 准备 vlm_alignment/train.jsonl
# 准备 vlm_alignment/val.jsonl
# 将图片放入 data/images/
```

### 步骤 2: 运行对齐训练
```bash
bash scripts/vlm/alignment/run.sh
# 输出: outputs/vlm_alignment/final
```

### 步骤 3: 准备 SFT 数据
```bash
mkdir -p data/vlm_sft

# 数据格式: {"image": "...", "messages": [...]}
# 准备 vlm_sft/train.jsonl
# 准备 vlm_sft/val.jsonl
```

### 步骤 4: 运行 VLM SFT
```bash
export ALIGNMENT_MODEL=./outputs/vlm_alignment/final
bash scripts/vlm/sft/run.sh
# 输出: outputs/vlm_sft/final
```

### 步骤 5: 部署测试
```bash
python -m src.common.deployment.api_server \
    --model ./outputs/vlm_sft/final \
    --port 8000
```

---

## 数据准备脚本示例

### 从现有数据集转换

```python
import json
import os
from pathlib import Path

def convert_coco_to_vlm_format(coco_json, image_dir, output_file):
    """将 COCO 格式转换为 VLM 格式"""
    with open(coco_json, "r") as f:
        coco_data = json.load(f)
    
    # 构建 image_id 到 captions 的映射
    image_captions = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        caption = ann["caption"]
        if image_id not in image_captions:
            image_captions[image_id] = []
        image_captions[image_id].append(caption)
    
    # 构建 id 到 file_name 的映射
    id_to_filename = {img["id"]: img["file_name"] 
                      for img in coco_data["images"]}
    
    # 输出 VLM 格式
    with open(output_file, "w", encoding="utf-8") as f:
        for image_id, captions in image_captions.items():
            # 使用第一个 caption
            filename = id_to_filename[image_id]
            
            vlm_item = {
                "image": filename,
                "caption": captions[0]
            }
            json.dump(vlm_item, f, ensure_ascii=False)
            f.write("\n")

# 使用示例
convert_coco_to_vlm_format(
    "coco/annotations/captions_train2017.json",
    "coco/train2017",
    "vlm_train.jsonl"
)
```

### 创建指令微调数据

```python
import json

def create_vlm_sft_data(image_list, output_file):
    """从图像列表创建 VLM SFT 数据模板"""
    templates = [
        {
            "prompt": "描述这张图片",
            "type": "caption"
        },
        {
            "prompt": "这张图片里有什么？",
            "type": "vqa"
        },
        {
            "prompt": "详细描述这张图片的内容",
            "type": "detailed_caption"
        }
    ]
    
    with open(output_file, "w", encoding="utf-8") as f:
        for image_path in image_list:
            for template in templates:
                item = {
                    "image": image_path,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"<image>\n{template['prompt']}"
                        },
                        {
                            "role": "assistant",
                            "content": "[需要人工标注或使用其他模型生成]"
                        }
                    ]
                }
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
```

---

## 数据质量检查清单

- [ ] 图像文件存在且可读取
- [ ] 图像路径正确（相对路径）
- [ ] JSON 格式正确，无语法错误
- [ ] 对话至少包含一轮 user-assistant
- [ ] 图像路径中不包含特殊字符
- [ ] 文本内容不为空
- [ ] 多图对话中图像数量与占位符匹配
