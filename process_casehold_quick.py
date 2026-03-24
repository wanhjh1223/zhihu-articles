#!/usr/bin/env python3
"""
CaseHOLD 数据集快速处理脚本
下载并转换为预训练格式
"""

import os
import json
from pathlib import Path
from datasets import load_dataset


def main():
    output_dir = Path("./casehold_processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 正在下载 CaseHOLD 数据集...")
    dataset = load_dataset("lex_glue", "case_hold")
    
    print(f"✅ 数据集加载完成")
    print(f"   训练集: {len(dataset['train'])} 条")
    print(f"   验证集: {len(dataset['validation'])} 条")
    
    # 处理训练集 - 三联体格式
    print("\n🔄 处理三联体格式...")
    triplet_data = []
    for ex in dataset["train"]:
        triplet = {
            "case_citation": ex["context"].replace("¡HOLDING¿", "").strip(),
            "holding": ex["endings"][ex["label"]],
            "explanation": f"根据案例引用，法院裁决要点为：{ex['endings'][ex['label']]}",
            "legal_principle": "法律原则待提取",
            "distractors": [e for i, e in enumerate(ex["endings"]) if i != ex["label"]],
            "full_text": ex["context"].replace("¡HOLDING¿", ex["endings"][ex["label"]])
        }
        triplet_data.append(triplet)
    
    with open(output_dir / "train_triplet.jsonl", "w", encoding="utf-8") as f:
        for item in triplet_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"   💾 train_triplet.jsonl: {len(triplet_data)} 条")
    
    # 处理验证集 - 三联体格式
    triplet_val = []
    for ex in dataset["validation"]:
        triplet = {
            "case_citation": ex["context"].replace("¡HOLDING¿", "").strip(),
            "holding": ex["endings"][ex["label"]],
            "explanation": f"根据案例引用，法院裁决要点为：{ex['endings'][ex['label']]}",
            "legal_principle": "法律原则待提取",
            "distractors": [e for i, e in enumerate(ex["endings"]) if i != ex["label"]],
            "full_text": ex["context"].replace("¡HOLDING¿", ex["endings"][ex["label"]])
        }
        triplet_val.append(triplet)
    
    with open(output_dir / "validation_triplet.jsonl", "w", encoding="utf-8") as f:
        for item in triplet_val:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"   💾 validation_triplet.jsonl: {len(triplet_val)} 条")
    
    # 处理训练集 - 指令格式
    print("\n🔄 处理指令微调格式...")
    instruction_data = []
    for ex in dataset["train"]:
        context = ex["context"].replace("¡HOLDING¿", "[HOLDING]")
        choices_text = "\n".join([f"{chr(65+i)}. {e}" for i, e in enumerate(ex["endings"])])
        answer = f"{chr(65+ex['label'])}. {ex['endings'][ex['label']]}"
        
        instruction_item = {
            "instruction": "根据以下法院判决引用，确定正确的法律裁决要点：",
            "input": f"{context}\n\n选项：\n{choices_text}",
            "output": f"正确答案是：{answer}",
            "label": ex["label"],
            "choices": ex["endings"]
        }
        instruction_data.append(instruction_item)
    
    with open(output_dir / "train_instruction.jsonl", "w", encoding="utf-8") as f:
        for item in instruction_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"   💾 train_instruction.jsonl: {len(instruction_data)} 条")
    
    # 处理验证集 - 指令格式
    instruction_val = []
    for ex in dataset["validation"]:
        context = ex["context"].replace("¡HOLDING¿", "[HOLDING]")
        choices_text = "\n".join([f"{chr(65+i)}. {e}" for i, e in enumerate(ex["endings"])])
        answer = f"{chr(65+ex['label'])}. {ex['endings'][ex['label']]}"
        
        instruction_item = {
            "instruction": "根据以下法院判决引用，确定正确的法律裁决要点：",
            "input": f"{context}\n\n选项：\n{choices_text}",
            "output": f"正确答案是：{answer}",
            "label": ex["label"],
            "choices": ex["endings"]
        }
        instruction_val.append(instruction_item)
    
    with open(output_dir / "validation_instruction.jsonl", "w", encoding="utf-8") as f:
        for item in instruction_val:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"   💾 validation_instruction.jsonl: {len(instruction_val)} 条")
    
    # 处理训练集 - 对比学习格式
    print("\n🔄 处理对比学习格式...")
    contrastive_data = []
    for ex in dataset["train"]:
        contrastive = {
            "anchor": ex["context"].replace("¡HOLDING¿", "").strip(),
            "positive": ex["endings"][ex["label"]],
            "hard_negatives": [e for i, e in enumerate(ex["endings"]) if i != ex["label"]],
            "full_context": ex["context"].replace("¡HOLDING¿", ex["endings"][ex["label"]])
        }
        contrastive_data.append(contrastive)
    
    with open(output_dir / "train_contrastive.jsonl", "w", encoding="utf-8") as f:
        for item in contrastive_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"   💾 train_contrastive.jsonl: {len(contrastive_data)} 条")
    
    # 处理验证集 - 对比学习格式
    contrastive_val = []
    for ex in dataset["validation"]:
        contrastive = {
            "anchor": ex["context"].replace("¡HOLDING¿", "").strip(),
            "positive": ex["endings"][ex["label"]],
            "hard_negatives": [e for i, e in enumerate(ex["endings"]) if i != ex["label"]],
            "full_context": ex["context"].replace("¡HOLDING¿", ex["endings"][ex["label"]])
        }
        contrastive_val.append(contrastive)
    
    with open(output_dir / "validation_contrastive.jsonl", "w", encoding="utf-8") as f:
        for item in contrastive_val:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"   💾 validation_contrastive.jsonl: {len(contrastive_val)} 条")
    
    # 生成 README
    readme = """# CaseHOLD 数据集处理结果

## 数据说明

CaseHOLD (Case Holdings on Legal Decisions) 是 LexGLUE 基准中的法律多选题问答数据集。

- **来源**: 美国法院判决（Harvard Law Library case law corpus）
- **任务**: 根据案例引用选择正确的法律裁决要点
- **规模**: 训练集 45,000 条 / 验证集 3,900 条

## 文件说明

| 文件名 | 格式 | 样本数 | 用途 |
|--------|------|--------|------|
| `train_triplet.jsonl` | 三联体 | 45,000 | 法律领域预训练 |
| `validation_triplet.jsonl` | 三联体 | 3,900 | 验证 |
| `train_instruction.jsonl` | 指令格式 | 45,000 | 指令微调 (SFT) |
| `validation_instruction.jsonl` | 指令格式 | 3,900 | 验证 |
| `train_contrastive.jsonl` | 对比学习 | 45,000 | Embedding/检索训练 |
| `validation_contrastive.jsonl` | 对比学习 | 3,900 | 验证 |

## 格式详情

### 三联体格式 (triplet)
```json
{
  "case_citation": "案例引用文本",
  "holding": "正确裁决要点",
  "explanation": "解释说明",
  "legal_principle": "法律原则",
  "distractors": ["干扰项1", "干扰项2", "干扰项3", "干扰项4"],
  "full_text": "完整文本"
}
```

### 指令格式 (instruction)
```json
{
  "instruction": "根据以下法院判决引用，确定正确的法律裁决要点：",
  "input": "案例文本\\n\\n选项：A... B...",
  "output": "正确答案是：A...",
  "label": 0,
  "choices": ["选项A", "选项B", ...]
}
```

### 对比学习格式 (contrastive)
```json
{
  "anchor": "案例引用（查询）",
  "positive": "正确的holding",
  "hard_negatives": ["错误选项1", "错误选项2", ...],
  "full_context": "完整上下文"
}
```

## 使用建议

1. **法律领域预训练**: 使用 `*_triplet.jsonl`
2. **指令微调**: 使用 `*_instruction.jsonl`
3. **检索模型训练**: 使用 `*_contrastive.jsonl` (hard_negatives 是高质量难负样本)

## 引用

```bibtex
@inproceedings{zheng2021casehold,
  title={When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset of 53,000+ Legal Holdings},
  author={Zheng, Lucille and Guha, Neel and Anderson, Brandon and Henderson, Peter and Ho, Daniel E},
  booktitle={Proceedings of the Eighteenth International Conference on Artificial Intelligence and Law},
  pages={159--168},
  year={2021}
}
```
"""
    
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    
    print(f"\n✅ 处理完成！")
    print(f"📁 输出目录: {output_dir.absolute()}")
    print(f"📖 查看 README.md 了解数据格式")


if __name__ == "__main__":
    main()
