# CaseHOLD 数据集处理示例

本示例展示了按照预训练数据要求处理后的 CaseHOLD 数据集格式。

---

## 原始数据示例

```json
{
  "context": "...see United States v. Wright-Barker, 784 F.2d 161, 176 (3d Cir.1986) (¡HOLDING¿), superseded...",
  "endings": [
    "the Coast Guard may conduct warrantless searches of American vessels",
    "the Coast Guard lacks jurisdiction over foreign vessels", 
    "warrants are required for all maritime inspections",
    "reasonable suspicion is never required for searches",
    "the Fourth Amendment applies differently at sea"
  ],
  "label": 0
}
```

---

## 格式1: 法条-案例-解释三联体 (推荐)

**文件名**: `train_triplet.jsonl`

```json
{
  "case_citation": "...see United States v. Wright-Barker, 784 F.2d 161, 176 (3d Cir.1986), superseded...",
  "holding": "the Coast Guard may conduct warrantless searches of American vessels",
  "explanation": "根据案例引用，法院裁决要点为：the Coast Guard may conduct warrantless searches of American vessels",
  "legal_principle": "从案例中提取的法律原则",
  "distractors": [
    "the Coast Guard lacks jurisdiction over foreign vessels",
    "warrants are required for all maritime inspections",
    "reasonable suspicion is never required for searches",
    "the Fourth Amendment applies differently at sea"
  ],
  "full_text": "...see United States v. Wright-Barker, 784 F.2d 161, 176 (3d Cir.1986) (the Coast Guard may conduct warrantless searches of American vessels), superseded..."
}
```

**用途**: 法律领域继续预训练，构建"案例-裁决-解释"结构

---

## 格式2: 指令微调格式 (Alpaca)

**文件名**: `train_instruction.jsonl`

```json
{
  "instruction": "根据以下法院判决引用，确定正确的法律裁决要点：",
  "input": "...see United States v. Wright-Barker, 784 F.2d 161, 176 (3d Cir.1986) [HOLDING], superseded...\n\n选项：\nA. the Coast Guard may conduct warrantless searches of American vessels\nB. the Coast Guard lacks jurisdiction over foreign vessels\nC. warrants are required for all maritime inspections\nD. reasonable suspicion is never required for searches\nE. the Fourth Amendment applies differently at sea",
  "output": "正确答案是：A. the Coast Guard may conduct warrantless searches of American vessels",
  "label": 0,
  "choices": [
    "the Coast Guard may conduct warrantless searches of American vessels",
    "the Coast Guard lacks jurisdiction over foreign vessels",
    "warrants are required for all maritime inspections",
    "reasonable suspicion is never required for searches",
    "the Fourth Amendment applies differently at sea"
  ]
}
```

**用途**: 指令微调 (SFT)，训练模型理解法律问答

---

## 格式3: 对比学习格式

**文件名**: `train_contrastive.jsonl`

```json
{
  "anchor": "...see United States v. Wright-Barker, 784 F.2d 161, 176 (3d Cir.1986), superseded...",
  "positive": "the Coast Guard may conduct warrantless searches of American vessels",
  "hard_negatives": [
    "the Coast Guard lacks jurisdiction over foreign vessels",
    "warrants are required for all maritime inspections", 
    "reasonable suspicion is never required for searches",
    "the Fourth Amendment applies differently at sea"
  ],
  "full_context": "...see United States v. Wright-Barker, 784 F.2d 161, 176 (3d Cir.1986) (the Coast Guard may conduct warrantless searches of American vessels), superseded..."
}
```

**用途**: 训练法律检索模型 (RAG)、Embedding模型
- `anchor`: 案例引用（查询）
- `positive`: 正确的holding（目标文档）
- `hard_negatives`: 干扰项（高质量难负样本）

---

## 格式4: Chain-of-Thought 推理格式

**文件名**: `train_cot.jsonl`

```json
{
  "text": "问题：根据以下法院判决引用，确定正确的法律裁决要点：\n\n...see United States v. Wright-Barker, 784 F.2d 161, 176 (3d Cir.1986) [待确定], superseded...\n\n选项：\nA. the Coast Guard may conduct warrantless searches of American vessels\nB. the Coast Guard lacks jurisdiction over foreign vessels\nC. warrants are required for all maritime inspections\nD. reasonable suspicion is never required for searches\nE. the Fourth Amendment applies differently at sea\n\n逐步推理：\n1. 首先，分析案例引用的上下文...\n2. 案例涉及的关键法律问题是...\n3. 对比各选项：\n   - 选项A涉及海岸警卫队对美国船只的搜查权...\n   - 选项B涉及对外国船只的管辖权...\n   - 选项C涉及搜查令要求...\n   - 选项D涉及合理怀疑标准...\n   - 选项E涉及第四修正案适用范围...\n4. 根据法律原则，最符合案例情况的裁决是...\n\n最终答案：A. the Coast Guard may conduct warrantless searches of American vessels",
  "question": "...see United States v. Wright-Barker, 784 F.2d 161, 176 (3d Cir.1986) [待确定], superseded...",
  "reasoning_steps": [
    "分析案例上下文",
    "识别关键法律问题", 
    "对比各选项",
    "应用法律原则"
  ],
  "answer": "the Coast Guard may conduct warrantless searches of American vessels",
  "answer_idx": 0
}
```

**用途**: 训练法律推理能力 (模拟法考思维过程)

---

## 格式5: RAG检索格式

**文件名**: `train_rag.jsonl`

```json
{
  "query": "案例：...see United States v. Wright-Barker... 的裁决要点是什么？",
  "positive_doc": "the Coast Guard may conduct warrantless searches of American vessels",
  "hard_negative_docs": [
    "the Coast Guard lacks jurisdiction over foreign vessels",
    "warrants are required for all maritime inspections",
    "reasonable suspicion is never required for searches", 
    "the Fourth Amendment applies differently at sea"
  ],
  "case_text": "...see United States v. Wright-Barker, 784 F.2d 161, 176 (3d Cir.1986), superseded...",
  "metadata": {
    "source": "casehold",
    "task": "legal_holding_retrieval"
  }
}
```

**用途**: 训练法律文档检索系统

---

## 格式6: 纯文本格式 (继续预训练)

**文件名**: `train_pure_text.txt`

```
...see United States v. Wright-Barker, 784 F.2d 161, 176 (3d Cir.1986) (the Coast Guard may conduct warrantless searches of American vessels), superseded...

法院判决指出：the Coast Guard may conduct warrantless searches of American vessels

本案的裁决要点是：the Coast Guard may conduct warrantless searches of American vessels

法院认为：the Coast Guard may conduct warrantless searches of American vessels
```

**用途**: 直接用于语言模型继续预训练

---

## 使用建议

### 场景1: 法律领域继续预训练
```python
# 推荐数据混合比例
- FineWeb-Edu (通用): 70%
- 法律语料 (Legal-BERT等): 20%
- CaseHOLD 三联体: 10%
```

### 场景2: 法律指令微调
```python
# 使用 instruction + cot 格式
- casehold_instruction.jsonl
- casehold_cot.jsonl
```

### 场景3: 法律检索系统
```python
# 使用 contrastive + rag 格式
- casehold_contrastive.jsonl
- casehold_rag.jsonl
```

---

## 运行处理脚本

```bash
# 处理所有格式
python3 process_casehold.py --output_dir ./casehold_processed

# 仅处理指定格式
python3 process_casehold.py --output_dir ./casehold_processed \
    --formats triplet instruction contrastive

# 仅处理训练集
python3 process_casehold.py --output_dir ./casehold_processed \
    --splits train --formats all
```

---

## 注意事项

1. **数据规模**: 
   - 训练集: 45,000 条
   - 验证集: 3,900 条  
   - 测试集: 3,900 条 (建议仅用于最终评估)

2. **难负样本**: CaseHOLD的4个干扰项是**高质量难负样本**，非常适合对比学习

3. **时间划分**: 数据按时间顺序划分，避免数据泄漏

4. **建议**: 
   - 仅用 `train` 划分进行预训练/微调
   - `validation` 用于超参调优
   - `test` 仅用于最终评估
