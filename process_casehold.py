#!/usr/bin/env python3
"""
CaseHOLD 数据集处理脚本
按照预训练数据要求转换格式

用法:
    python process_casehold.py --output_dir ./casehold_processed
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset


def download_casehold():
    """下载 CaseHOLD 数据集"""
    print("📥 正在下载 CaseHOLD 数据集...")
    dataset = load_dataset("lex_glue", "case_hold")
    print(f"✅ 数据集加载完成")
    print(f"   训练集: {len(dataset['train'])} 条")
    print(f"   验证集: {len(dataset['validation'])} 条")
    print(f"   测试集: {len(dataset['test'])} 条")
    return dataset


def format_pure_text(examples: List[Dict]) -> List[str]:
    """
    格式1: 纯文本格式 (用于继续预训练)
    将引用文本中的 ¡HOLDING¿ 替换为正确的 holding
    """
    texts = []
    for ex in examples:
        context = ex["context"]
        correct_holding = ex["endings"][ex["label"]]
        full_text = context.replace("¡HOLDING¿", correct_holding)
        texts.append(full_text)
    return texts


def format_legal_triplet(examples: List[Dict]) -> List[Dict]:
    """
    格式2: 法条-案例-解释三联体 (推荐用于法律领域预训练)
    
    结构:
    {
        "case_citation": "案例引用文本",
        "holding": "正确裁决要点", 
        "explanation": "推理解释",
        "legal_principle": "涉及的法律原则"
    }
    """
    triplets = []
    for ex in examples:
        context = ex["context"]
        endings = ex["endings"]
        label = ex["label"]
        
        correct_holding = endings[label]
        incorrect_holdings = [e for i, e in enumerate(endings) if i != label]
        
        # 构建三联体
        triplet = {
            "case_citation": context.replace("¡HOLDING¿", "").strip(),
            "holding": correct_holding,
            "explanation": f"根据案例引用，法院裁决要点为：{correct_holding}",
            "legal_principle": "从案例中提取的法律原则",
            "distractors": incorrect_holdings,  # 干扰项作为负样本
            "full_text": context.replace("¡HOLDING¿", correct_holding)
        }
        triplets.append(triplet)
    
    return triplets


def format_instruction_tuning(examples: List[Dict]) -> List[Dict]:
    """
    格式3: 指令微调格式 (Alpaca格式)
    """
    instructions = []
    
    # 定义多种指令模板
    templates = [
        {
            "instruction": "根据以下法院判决引用，确定正确的法律裁决要点：",
            "input_template": "{context}\n\n选项：\n{choices}",
            "output_template": "正确答案是：{answer}"
        },
        {
            "instruction": "阅读以下法律案例引用，选择最准确的holding statement：",
            "input_template": "案例引用：{context}\n\n可能的裁决：\n{choices}",
            "output_template": "{answer}"
        },
        {
            "instruction": "作为法律专家，分析以下案例引用并确定法院的裁决要点：",
            "input_template": "{context}",
            "output_template": "法院的裁决要点是：{answer}"
        }
    ]
    
    for ex in examples:
        context = ex["context"].replace("¡HOLDING¿", "[HOLDING]")
        endings = ex["endings"]
        label = ex["label"]
        
        # 构建选项文本
        choices_text = "\n".join([f"{chr(65+i)}. {e}" for i, e in enumerate(endings)])
        answer = f"{chr(65+label)}. {endings[label]}"
        
        # 随机选择一个模板
        template = templates[hash(ex["context"]) % len(templates)]
        
        instruction_item = {
            "instruction": template["instruction"],
            "input": template["input_template"].format(
                context=context,
                choices=choices_text
            ),
            "output": template["output_template"].format(answer=answer),
            "label": label,
            "choices": endings
        }
        instructions.append(instruction_item)
    
    return instructions


def format_contrastive_learning(examples: List[Dict]) -> List[Dict]:
    """
    格式4: 对比学习格式 (用于 Embedding 模型训练)
    anchor: 案例引用（不含holding）
    positive: 正确的 holding
    negatives: 4个错误的 holding（作为难负样本）
    """
    contrastive_pairs = []
    
    for ex in examples:
        context = ex["context"].replace("¡HOLDING¿", "").strip()
        endings = ex["endings"]
        label = ex["label"]
        
        positive = endings[label]
        negatives = [e for i, e in enumerate(endings) if i != label]
        
        contrastive_pairs.append({
            "anchor": context,
            "positive": positive,
            "hard_negatives": negatives,  # CaseHOLD的干扰项是高质量难负样本
            "full_context": ex["context"].replace("¡HOLDING¿", positive)
        })
    
    return contrastive_pairs


def format_cot_reasoning(examples: List[Dict]) -> List[Dict]:
    """
    格式5: Chain-of-Thought 推理格式
    模拟法考推理过程
    """
    cot_examples = []
    
    for ex in examples:
        context = ex["context"].replace("¡HOLDING¿", "[待确定]")
        endings = ex["endings"]
        label = ex["label"]
        
        # 构建 CoT 推理文本
        cot_text = f"""问题：根据以下法院判决引用，确定正确的法律裁决要点：

{context}

选项：
A. {endings[0]}
B. {endings[1]}
C. {endings[2]}
D. {endings[3]}
E. {endings[4]}

逐步推理：
1. 首先，分析案例引用的上下文...
2. 案例涉及的关键法律问题是...
3. 对比各选项：
   - 选项A涉及...
   - 选项B涉及...
   - 选项C涉及...
   - 选项D涉及...
   - 选项E涉及...
4. 根据法律原则，最符合案例情况的裁决是...

最终答案：{chr(65+label)}. {endings[label]}"""
        
        cot_examples.append({
            "text": cot_text,
            "question": context,
            "reasoning_steps": [
                "分析案例上下文",
                "识别关键法律问题",
                "对比各选项",
                "应用法律原则"
            ],
            "answer": endings[label],
            "answer_idx": label
        })
    
    return cot_examples


def format_rag_retrieval(examples: List[Dict]) -> List[Dict]:
    """
    格式6: 检索增强生成 (RAG) 格式
    用于训练法律检索模型
    """
    rag_data = []
    
    for ex in examples:
        context = ex["context"].replace("¡HOLDING¿", "").strip()
        endings = ex["endings"]
        label = ex["label"]
        
        # Query: 案例描述
        # Document: 正确的 holding
        # Hard negatives: 错误的 holdings
        rag_data.append({
            "query": f"案例：{context[:200]}... 的裁决要点是什么？",
            "positive_doc": endings[label],
            "hard_negative_docs": [e for i, e in enumerate(endings) if i != label],
            "case_text": context,
            "metadata": {
                "source": "casehold",
                "task": "legal_holding_retrieval"
            }
        })
    
    return rag_data


def augment_legal_data(examples: List[Dict]) -> List[Dict]:
    """
    数据增强：生成更多训练样本
    """
    augmented = []
    
    for ex in examples:
        context = ex["context"]
        endings = ex["endings"]
        label = ex["label"]
        correct = endings[label]
        
        # 增强1: 不同表述方式
        templates = [
            ("法院判决指出：{holding}", "formal"),
            ("本案的裁决要点是：{holding}", "summary"),
            ("法院认为：{holding}", "opinion"),
            ("根据该案例，{holding}", "reference")
        ]
        
        for template, style in templates:
            text = template.format(holding=correct)
            augmented.append({
                "text": context.replace("¡HOLDING¿", text),
                "style": style,
                "original_label": label
            })
        
        # 增强2: 反向任务（给定holding，生成查询）
        augmented.append({
            "text": f"法律要点：{correct}\n相关案例：{context.replace('¡HOLDING¿', correct)}",
            "style": "reverse",
            "original_label": label
        })
    
    return augmented


def save_dataset(data: List[Any], output_path: Path, format_name: str):
    """保存数据集到文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format_name == "pure_text":
        # 保存为纯文本文件（每行一个样本）
        with open(output_path, "w", encoding="utf-8") as f:
            for line in data:
                f.write(line + "\n")
    else:
        # 保存为 JSONL
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"   💾 已保存: {output_path} ({len(data)} 条)")


def main():
    parser = argparse.ArgumentParser(description="处理 CaseHOLD 数据集")
    parser.add_argument("--output_dir", type=str, default="./casehold_processed",
                        help="输出目录")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"],
                        help="要处理的数据集划分")
    parser.add_argument("--formats", nargs="+", 
                        default=["all"],
                        choices=["all", "pure_text", "triplet", "instruction", 
                                "contrastive", "cot", "rag", "augmented"],
                        help="输出格式")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载数据
    dataset = download_casehold()
    
    # 处理每个划分
    for split in args.splits:
        if split not in dataset:
            print(f"⚠️  跳过不存在的划分: {split}")
            continue
        
        print(f"\n📊 处理 {split} 划分...")
        examples = list(dataset[split])
        
        formats_to_process = (
            ["pure_text", "triplet", "instruction", "contrastive", "cot", "rag", "augmented"]
            if "all" in args.formats else args.formats
        )
        
        for fmt in formats_to_process:
            print(f"   🔄 生成 {fmt} 格式...")
            
            if fmt == "pure_text":
                data = format_pure_text(examples)
                save_dataset(data, output_dir / f"{split}_pure_text.txt", fmt)
                
            elif fmt == "triplet":
                data = format_legal_triplet(examples)
                save_dataset(data, output_dir / f"{split}_triplet.jsonl", fmt)
                
            elif fmt == "instruction":
                data = format_instruction_tuning(examples)
                save_dataset(data, output_dir / f"{split}_instruction.jsonl", fmt)
                
            elif fmt == "contrastive":
                data = format_contrastive_learning(examples)
                save_dataset(data, output_dir / f"{split}_contrastive.jsonl", fmt)
                
            elif fmt == "cot":
                data = format_cot_reasoning(examples)
                save_dataset(data, output_dir / f"{split}_cot.jsonl", fmt)
                
            elif fmt == "rag":
                data = format_rag_retrieval(examples)
                save_dataset(data, output_dir / f"{split}_rag.jsonl", fmt)
                
            elif fmt == "augmented":
                data = augment_legal_data(examples)
                save_dataset(data, output_dir / f"{split}_augmented.jsonl", fmt)
    
    # 生成 README
    readme_content = """# CaseHOLD 数据集处理结果

## 文件说明

| 文件名 | 格式 | 用途 |
|--------|------|------|
| `*_pure_text.txt` | 纯文本 | 继续预训练 |
| `*_triplet.jsonl` | 三联体 | 法条-案例-解释结构 |
| `*_instruction.jsonl` | 指令格式 | 指令微调 (Alpaca风格) |
| `*_contrastive.jsonl` | 对比学习 | Embedding模型训练 |
| `*_cot.jsonl` | CoT推理 | 法律推理能力训练 |
| `*_rag.jsonl` | RAG格式 | 检索增强生成 |
| `*_augmented.jsonl` | 增强数据 | 数据增强后的训练集 |

## 推荐使用

1. **继续预训练**: 使用 `*_pure_text.txt`
2. **法律领域微调**: 使用 `*_triplet.jsonl` + `*_instruction.jsonl`
3. **法律检索模型**: 使用 `*_contrastive.jsonl`
4. **法考推理训练**: 使用 `*_cot.jsonl`

## 数据规模

- 训练集: 45,000 条
- 验证集: 3,900 条
- 测试集: 3,900 条 (建议仅用于最终评估)
"""
    
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"\n✅ 处理完成！输出目录: {output_dir}")
    print("📖 查看 README.md 了解各文件用途")


if __name__ == "__main__":
    main()
