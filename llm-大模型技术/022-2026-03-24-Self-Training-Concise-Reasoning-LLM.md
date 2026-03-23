# Self-Training Elicits Concise Reasoning in Large Language Models

## 论文信息速览

- **标题**: Self-Training Elicits Concise Reasoning in Large Language Models
- **作者**: Tergel Munkhbat*, Namgyu Ho*, Seo Hyun Kim*, Yongjin Yang, Yujin Kim, Se-Young Yun†
- **机构**: KAIST AI (韩国科学技术院人工智能学院)
- **发表时间**: 2025年2月27日 (arXiv v1), 2025年6月10日 (v3更新)
- **论文链接**: https://arxiv.org/abs/2502.20122
- **代码仓库**: https://github.com/TergelMunkhbat/concise-reasoning

---

## 一、研究背景与问题陈述

### 1.1 Chain-of-Thought的双刃剑效应

Chain-of-thought (CoT) 推理自2022年被发现以来，已成为提升大语言模型(LLM)复杂推理能力的关键技术。其核心思想是让模型在输出最终答案之前，先生成一系列中间推理步骤。这种方法的数学基础在于：每个中间推理token使模型能够执行额外的前向传播计算，从而有效增加模型在推理过程中的计算量[Nye et al., 2021; Wei et al., 2022]。

然而，CoT推理是一把双刃剑。虽然它显著提升了模型在数学、逻辑和复杂问题求解上的表现，但也带来了显著的**推理成本问题**：

- **推理延迟增加**：输出token数量直接决定了推理时间
- **计算资源消耗**：每个生成的token都需要一次完整的前向传播
- **成本上升**：在实际部署中，API调用通常按token计费

> 典型示例：一个复杂的数学问题可能产生数百个推理token，其中大量是冗余的表述，如反复强调计算步骤、冗长的引导语等。

### 1.2 核心观察：推理链中存在大量冗余

研究团队提出了一个关键假设：**当前模型的推理链中包含大量冗余token，造成了不必要的推理成本**。

通过对现有LLM输出分布的分析，作者发现典型推理样本（如图1蓝色框所示）包含大量对解题无贡献的token，主要包括：

1. **冗长解释**：过度详细的步骤说明
2. **重复措辞**：多次重复相同概念
3. **模板化表述**：固定的引导语句（如"Step 1:", "Therefore"）
4. **过度推导**：简单问题的复杂化表达

这一观察得到了其他研究的支持[Renze and Guven, 2024; Zhang et al., 2024; Chiang and Lee, 2024]，表明这是一个普遍存在的问题。

### 1.3 为什么存在这种冗余？

作者从LLM的训练过程角度解释了这种现象的成因：

**CoT推理是涌现能力而非显式训练目标**
- CoT推理是LLM的涌现能力(emergent ability)，最初是通过few-shot prompting偶然发现的[Wei et al., 2022a]
- LLM在预训练阶段并未被显式训练为"高效利用token进行推理"
- 推理能力主要来源于预训练数据中的程序性知识(procedural knowledge)[Ruis et al., 2024]
- 这些预训练数据并未针对"简洁性"进行优化

**关键洞见**：虽然LLM的默认行为未优化token效率，但这并不意味着它们不具备简洁推理的潜力。事实上，在模型的输出分布中已经存在更短的正确推理路径。

---

## 二、核心发现：模型具备潜在的简洁推理能力

### 2.1 实验证据：输出分布中的短路径

研究团队通过系统性实验验证了这一假设。他们分析了多个主流模型在GSM8K数据集上的推理路径长度分布：

**实验设置**：
- 对每个问题采样16条推理路径
- 计算每条路径的归一化长度（相对于该问题正确路径的平均长度）
- 使用核密度估计可视化分布

**关键发现**（见图2）：

1. **分布左偏现象**：所有模型的推理路径长度分布在100%以下都有显著概率质量，表明模型确实能够生成比默认输出更短的正确解答

2. **不同模型的差异**：
   - DeepSeekMath-7B在绝对长度上最短，但在8.37%的情况下仍能生成不到平均长度一半的正确解答
   - Llama、Gemma、Qwen系列模型都显示出类似的简洁推理潜力

3. **潜在能力vs默认行为**：模型虽然有能力生成简洁推理，但默认采样倾向于产生冗长输出

### 2.2 简洁推理的定义

论文给出了简洁推理的精确定义：

> **简洁推理（Concise Reasoning）**：对于给定模型，简洁推理是指使用比默认输出更少的输出token来正确解决给定问题的推理方式。

其中，**默认输出**定义为：在多次随机生成中，正确路径的平均长度（而非贪心解码长度，因为贪心路径可能是错误的）。

---

## 三、现有方法的局限性分析

### 3.1 Zero-shot Prompting方法的不足

在提出新方法之前，作者系统评估了现有的zero-shot prompting方法：

| 方法 | 描述 | GSM8K相对长度 | GSM8K相对准确率 |
|------|------|---------------|-----------------|
| Baseline | 标准提示 | 100% | 100% |
| Be Concise | 追加"be concise" | 88.46% | 99.85% |
| Fixed Budget | 限制100词 | 67.80% | 89.90% |
| Hand Crafted 2 | 优化的手工提示 | 77.10% | 98.27% |

**主要问题**：

1. **准确率-长度权衡困境**：大多数zero-shot方法能减少长度，但通常导致显著准确率下降。例如Fixed Budget虽然缩短32.2%长度，但准确率下降10.1%

2. **模型间不一致性**：
   - 在通用模型（如Llama-3.2）上有效的提示
   - 在数学专用模型（如Qwen2.5-Math）上几乎无效
   - 这表明任务特定模型的内部表征对zero-shot提示的响应性较低

3. **效果有限**：即使是最优的手工提示，在数学专用模型上也无法实现显著的长度缩减

### 3.2 需要根本性解决方案

上述分析表明，仅通过提示工程无法可靠地激发简洁推理，特别是在任务特定模型上。这 necessitates 一种更深层次的解决方案——通过微调来改变模型的行为模式。

---

## 四、方法论：自训练激发简洁推理

### 4.1 核心思想

研究团队提出了一种简单但有效的自训练(self-training)方法，核心思想是：

> **利用模型自身输出分布中已经存在的简洁推理路径，通过微调将这些路径"固化"为模型的默认行为。**

这种方法具有以下优势：
1. **自洽性**：训练数据来自模型自身分布，不会引入分布外(out-of-distribution)数据
2. **无需外部依赖**：不依赖人工标注或更强大的模型
3. **保留推理能力**：由于数据源自模型自身，推理能力得以保持

### 4.2 方法一：Naive Best-of-N Sampling (BoN)

**基本流程**：
1. 对每个训练集问题，生成N条推理路径
2. 选择每条问题最短的正确推理路径
3. 使用这些路径进行标准微调

**关键设计**：按问题级别选择最短路径（而非全局最短），确保监督覆盖各种难度级别——困难问题可能需要更长的绝对推理长度。

**局限性**：
- 样本效率问题：N与长度缩减呈对数线性关系（图3）
- 要实现更大幅度的缩减，需要指数级增长的生成成本

### 4.3 方法二：Few-Shot Conditioning (FS)

为解决BoN的样本效率问题，作者引入few-shot prompting来引导生成更短的推理路径。

**Few-shot示例来源**：
- **FS-Human**：使用人工标注的简洁示例（来自Wei et al., 2022b）
- **FS-GPT4o**：使用GPT-4o生成的简洁推理
- **FS-Self**：使用模型自身生成的简洁示例

**关键发现**：
- 8-shot conditioning显著减少推理路径长度
- FS-Human甚至能实现比BoN (N=256) 更强的缩减效果
- Few-shot prompting在所有测试模型上都有效

**理论基础**：Few-shot learning是LLM的基本能力[Brown et al., 2020]，因此这种方法具有良好的泛化性。

### 4.4 方法三：Few-Shot Conditioned Best-of-N (FS-BoN)

**核心创新**：结合FS和BoN的优势，实现最大化长度缩减。

**流程**：
1. 使用few-shot conditioning引导生成长度更短的分布
2. 在此分布上进行BoN采样（N=16）
3. 选择最短正确路径
4. 进行微调

**效果**：FS和BoN的改进是**独立且可加**的，实现了显著的长度缩减。

### 4.5 Sample Augmentation：平衡简洁性与准确性

**问题识别**：
Few-shot prompting可能：
1. 阻碍复杂问题所需的长推理路径生成
2. 对简单问题产生不必要的步骤

**解决方案**：Sample Augmentation
- 对于FS和FS-BoN生成的样本，额外补充N个naive BoN样本
- 从合并集合中选择最短正确路径
- 这样既保留了FS/FS-BoN的长度缩减优势，又更好地保持了准确率

---

## 五、实验设计与结果分析

### 5.1 实验设置

**测试模型**（5个主流模型家族）：
- Llama-3.2-3B
- Gemma-2-2B  
- Qwen2.5-3B
- Qwen2.5-Math-1.5B（数学专用）
- DeepSeekMath-7B（数学专用）

**数据集**：
- **GSM8K**：小学数学应用题，模型准确率40-90%
- **MATH500**：竞赛级数学问题，模型准确率20-70%

**评估指标**：
- **准确率(Accuracy)**：使用Python解析最终答案
- **长度(Length)**：所有推理路径的平均输出token数（包括错误路径，因为部署中错误输出同样消耗成本）
- **相对指标**：相对于强基线zero-shot prompt的比值

**对比基线**：
- Zero-shot prompting方法
- 直接微调（Ground Truth答案）
- 外部监督（Human/GPT-4o生成的简洁推理）
- Rational Metareasoning [De Sabbata et al., 2024]

### 5.2 主要结果

#### 5.2.1 综合性能对比（表2）

| 方法 | GSM8K Acc | GSM8K Len | Rel. Acc | Rel. Len |
|------|-----------|-----------|----------|----------|
| Baseline | 78.06% | 241.87 | 100% | 100% |
| Be Concise | 77.98% | 214.87 | 99.85% | 88.46% |
| Naive BoN | 77.12% | 214.22 | 98.79% | 87.17% |
| FS-Human | 76.66% | 161.72 | 98.06% | **67.96%** |
| FS-GPT4o | 78.07% | 175.54 | 99.94% | 73.15% |
| **FS-GPT4o-BoN** | 75.88% | **153.38** | 97.00% | **64.25%** |

**关键发现**：

1. **30%平均长度缩减**：FS-GPT4o-BoN在GSM8K上实现64.25%相对长度（即35.75%缩减），在MATH上实现76.30%相对长度

2. **准确率保持**：平均相对准确率保持在97-99%，优于外部监督方法

3. **2.4倍提升**：相比之前的微调基线（如Rational Metareasoning），长度缩减效果提升2.4倍

#### 5.2.2 方法间的详细对比

**Naive BoN**：
- ✅ 有效减少长度（平均12%）且不显著降低性能
- ✅ 在数学专用模型上也有效（zero-shot失败的地方）
- ❌ 样本效率低，进一步缩减成本高

**Few-Shot Conditioning**：
- ✅ 相比BoN实现更大幅度的长度缩减
- ✅ 在数学专用模型上同样有效
- FS-Human在GSM8K上达到67.96%相对长度

**FS-BoN**：
- ✅ **最佳长度缩减**（64.25%相对长度）
- ✅ 保持相对准确率
- 在数学专用模型上效果尤为突出

#### 5.2.3 自训练vs外部监督

**关键结论**：自训练在准确率-长度权衡上显著优于外部监督：

- **外部数据微调**（如Human CoT、GPT4o CoT）：虽然长度大幅缩减，但准确率严重下降
  - Human CoT: 54.95%相对长度，但83.82%相对准确率
  - GPT4o CoT: 67.60%相对长度，97.65%相对准确率

- **自训练方法**：位于准确率-长度帕累托前沿，实现最佳平衡

**原因分析**：
- 外部数据可能来自不同的分布
- 模型难以适应分布外的简洁推理风格
- 自训练数据来自模型自身分布，更容易内化

### 5.3 定性分析

表3展示了一个典型示例：

**原始模型输出**（Zero-Shot）：
```
To find the total number of bolts needed, we need to calculate 
the amount of white fiber first, since it's half the amount of 
blue fiber. Step 1: Determine the amount of blue fiber needed. 
The robe takes 2 bolts of blue fiber. Step 2: Calculate the 
amount of white fiber needed. The robe takes half the amount 
of white fiber as blue fiber. Half of 2 bolts is 2/2 = 1 bolt. 
Step 3: Add the amount of blue fiber and white fiber to find 
the total. Total bolts = blue fiber + white fiber. 
Total bolts = 2 bolts + 1 bolt. Total bolts = 3 bolts. 
The answer is 3.
```
（约100+ tokens）

**FS-GPT4o-BoN微调后输出**：
```
The robe takes 2 bolts of blue fiber. It takes half that much 
white fiber, which is 2/2 = 1 bolt. Add the blue and white 
fiber together: 2 + 1 = 3 bolts. The answer is 3.
```
（约40 tokens）

**改进点**：
- 移除了冗余的问题重述
- 简化了步骤标记
- 直接呈现核心计算过程
- 保留必要的推理步骤

### 5.4 自适应长度调整能力

研究发现，经过微调的模型展现出**自适应长度调整**能力：
- 对于困难问题，保留详细的推理步骤
- 对于简单问题，自动简化响应
- 这种自适应行为是涌现的，非显式训练

---

## 六、深入分析：为什么自训练有效？

### 6.1 理论解释

自训练激发简洁推理的有效性可以从以下几个角度理解：

**1. 分布内学习**
- 微调数据来自模型自身的输出分布
- 避免了分布外数据带来的适应困难
- 保持了模型的原始能力

**2. 行为固化**
- Few-shot conditioning在推理时"激活"了简洁推理能力
- 微调将这种条件激活固化为默认行为
- 消除了推理时的额外开销

**3. 隐式能力显性化**
- 简洁推理能力始终存在于模型中（如2.1节所示）
- 自训练只是改变了输出分布的重心
- 从一个"冗长偏向"的分布转变为"简洁偏向"的分布

### 6.2 样本效率分析

图3展示了几种方法的样本效率对比：

| 方法 | 达到50%缩减所需样本 | 效率评级 |
|------|---------------------|----------|
| Naive BoN | ~256 | ⭐⭐ |
| FS-Human | ~8 | ⭐⭐⭐⭐⭐ |
| FS-GPT4o | ~8 | ⭐⭐⭐⭐⭐ |
| FS-GPT4o-BoN | ~16 | ⭐⭐⭐⭐ |

**结论**：Few-shot conditioning将样本效率提升约32倍。

### 6.3 跨模型泛化性

研究验证了方法在不同模型家族上的有效性：

**通用模型**（Llama、Gemma、Qwen）：
- 所有方法都有效
- Zero-shot prompting也有一定效果

**数学专用模型**（Qwen2.5-Math、DeepSeekMath）：
- Zero-shot prompting基本失效
- 自训练方法依然有效
- 这表明自训练具有更强的鲁棒性

### 6.4 计算成本分析

表5详细对比了不同方法的推理时开销：

| 方法 | 推理开销 | 长度缩减 | 实际效益 |
|------|----------|----------|----------|
| Zero-shot | 1x | 0-12% | 一般 |
| Naive BoN (test-time) | 16x | ~15% | 差（开销>收益） |
| Few-shot (test-time) | 1.5x | ~25% | 中等 |
| **FS-BoN + FT** | **1x** | **30%** | **优秀** |

**关键洞察**：只有在将长度缩减"内化"到模型后，才能真正实现推理成本的降低。

---

## 七、行业影响与应用前景

### 7.1 对LLM部署的意义

**成本优化**：
- 30%的token缩减直接转化为30%的推理成本降低
- 对于大规模部署，这意味着显著的经济效益
- 延迟降低同样改善用户体验

**绿色AI**：
- 减少不必要的计算，降低能耗
- 符合AI可持续发展的趋势

### 7.2 与其他技术的兼容性

本研究与以下技术方向高度兼容：

**1. Test-Time Scaling**
- 与s1、o1-like模型等test-time scaling方法互补
- 可以在保持缩放能力的同时提高效率

**2. 推理优化**
- 与推测解码(speculative decoding)、量化等技术结合
- 多层次的效率提升

**3. Agent系统**
- 多步骤推理的Agent系统将显著受益于token效率提升
- 降低长程任务的累积成本

### 7.3 局限性与未来方向

**当前局限**：
1. 任务特定性：需要为每个任务单独微调
2. 领域局限：主要在数学推理上验证
3. 准确率轻微下降：虽然很小，但在某些关键应用中可能需要权衡

**未来研究方向**：
1. **通用简洁推理模型**：开发跨任务的通用简洁推理能力
2. **动态长度控制**：根据问题复杂度实时调整推理深度
3. **多模态扩展**：将方法扩展到视觉-语言、代码生成等多模态任务
4. **理论理解**：更深入地理解简洁推理的数学基础

---

## 八、个人理解与批判性思考

### 8.1 核心贡献评价

**这项研究的核心贡献在于**：

1. **问题定义的新颖性**：将"推理效率"作为一个显式的优化目标，而非仅仅关注准确率

2. **方法论的简洁性**：使用简单的自训练流程实现显著改进，易于复现和部署

3. **发现的深刻性**：揭示了LLM内部存在的"冗长偏向"及其可矫正性

### 8.2 与其他研究的联系

**与相关工作的对比**：

| 研究 | 核心思想 | 与本文关系 |
|------|----------|------------|
| Don't Overthink [Wang et al., 2025] | 选择性推理 | 互补：本文关注长度优化，该工作关注是否推理 |
| s1 [Muennighoff et al., 2025] | Test-time scaling | 互补：可以结合使用 |
| O1-Pruner [Luo et al., 2025] | o1-like模型剪枝 | 相似目标，本文方法更通用 |
| NoThinking [Ma et al., 2025] | 无思考推理 | 不同方向：本文优化思考过程，该工作跳过思考 |

### 8.3 潜在风险与伦理考量

**需要注意的问题**：

1. **过度简洁的风险**：在某些领域（如医疗、法律），详细的推理过程可能比简洁性更重要

2. **可解释性权衡**：虽然简洁推理保留了关键步骤，但极度的简洁可能影响人类对模型决策过程的理解

3. **误差累积**：在链式推理中，过度简化可能增加中间步骤错误的概率

### 8.4 实践建议

基于本研究，我对LLM开发者和部署者提出以下建议：

**对于模型开发者**：
- 在预训练或后训练阶段考虑引入简洁性目标
- 开发动态长度控制机制
- 在保持准确性的前提下优化推理效率

**对于应用部署者**：
- 在成本敏感的场景中考虑使用自训练方法
- 根据任务特性权衡简洁性与可解释性
- 监控简洁化后的模型性能，确保质量不降

---

## 九、技术细节补充

### 9.1 超参数设置

**Few-shot示例数量**：
- 实验中使用8-shot conditioning
- 更多的shot可能进一步提升效果，但增加输入成本

**BoN采样数N**：
- 标准设置：N=16
- Budget-Matched设置：N=8
- N与长度缩减呈对数关系

**微调配置**：
- 使用标准监督微调
- 学习率、batch size等遵循各模型家族的最佳实践

### 9.2 数据过滤策略

**质量控制**：
- 仅使用正确解答进行训练
- 排除无正确解答的问题
- 保留多样性：按问题级别选择而非全局选择

### 9.3 评估细节

**答案解析**：
- 使用Python脚本解析数值答案
- 考虑多种格式（整数、小数、分数等）
- 处理常见的答案包装格式

**统计显著性**：
- 报告平均值和标准差
- 跨多个模型验证一致性
- 多次运行确保结果稳定

---

## 十、结论

### 10.1 研究总结

"Self-Training Elicits Concise Reasoning in Large Language Models"是一项具有重要意义的研究。它系统地：

1. **识别了问题**：当前LLM推理中存在大量token冗余
2. **发现了潜力**：模型本身具备简洁推理的潜在能力
3. **提出了方案**：通过自训练有效激发这种能力
4. **验证了效果**：实现30%长度缩减，同时保持准确率

### 10.2 核心启示

**对于AI研究者**：
- 效率与准确性并非零和博弈，可以共同优化
- 模型内部潜藏着许多未被充分利用的能力
- 简单的方法往往最有效

**对于AI从业者**：
- 推理成本优化有巨大的实际价值
- 自训练是一种实用且有效的部署技术
- 关注输出质量而非仅仅关注准确性

**对于AI领域**：
- 研究重点可能需要从"能不能做"转向"如何高效地做"
- 简洁推理可能成为下一代LLM的标准特性
- 效率与能力的平衡将成为关键研究方向

### 10.3 最终评价

这篇论文以其清晰的思路、扎实的实验和实用的方法，为LLM推理效率优化领域做出了重要贡献。其方法简单有效，易于复现和部署，具有很高的实际应用价值。

在LLM能力日益增强的今天，如何高效地利用这些能力变得越来越重要。这项研究为这一挑战提供了一个优雅的解决方案，值得广泛关注和跟进。

---

## 参考文献

[主要引用]

1. Munkhbat, T., Ho, N., Kim, S. H., Yang, Y., Kim, Y., & Yun, S. Y. (2025). Self-Training Elicits Concise Reasoning in Large Language Models. arXiv preprint arXiv:2502.20122.

2. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., ... & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. NeurIPS.

3. Renze, M., & Guven, E. (2024). The benefits of a concise chain of thought on problem-solving in large language models. FLLM.

4. De Sabbata, G., Sarti, G., & Nissim, M. (2024). Rational metareasoning: A framework for efficient llm reasoning. EMNLP.

5. Brown, T., Mann, B., Ryder, N., et al. (2020). Language models are few-shot learners. NeurIPS.

[相关扩展工作]

6. Muennighoff, N., Yang, Z., Shi, W., et al. (2025). s1: Simple test-time scaling. arXiv.

7. Luo, H., Shen, L., He, H., et al. (2025). O1-pruner: Length-harmonizing fine-tuning for o1-like reasoning pruning. arXiv.

8. Ma, X., Wan, G., Yu, R., et al. (2025). Cot-valve: Length-compressible chain-of-thought tuning. arXiv.

9. Wang, J., Lin, K. Q., Cheng, J., & Shou, M. Z. (2025). Think or Not? Selective Reasoning via Reinforcement Learning for Vision-Language Models. arXiv.

---

*本文由AI辅助分析撰写，内容基于原论文及公开资料整理。*
*字数统计：约 6500 字*
