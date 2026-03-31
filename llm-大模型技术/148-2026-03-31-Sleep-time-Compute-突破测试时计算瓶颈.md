# Sleep-time Compute: 突破测试时计算瓶颈的新范式

> **论文标题**: Sleep-time Compute: Beyond Inference Scaling at Test-time  
> **作者**: Kevin Lin*, Charlie Snell*, Yu Wang, Charles Packer, Sarah Wooders, Ion Stoica, Joseph E. Gonzalez  
> **机构**: Letta, UC Berkeley  
> **发表时间**: 2025年4月17日  
> **论文链接**: [arXiv:2504.13171](https://arxiv.org/abs/2504.13171)  
> **代码仓库**: [GitHub - letta-ai/sleep-time-compute](https://github.com/letta-ai/sleep-time-compute)

---

## 摘要

测试时计算（Test-time Compute）已成为提升大语言模型（LLM）解决复杂问题能力的关键技术，但它带来了高延迟和高推理成本的问题。本文介绍的 **Sleep-time Compute**（睡眠时计算）是一种全新范式，允许模型在查询呈现之前**离线"思考"上下文**：通过预测用户可能提出的问题并预计算有用的信息，可以显著降低测试时的计算需求。实验证明，Sleep-time Compute 可以将实现相同准确率所需的测试时计算减少**约5倍**，同时通过扩展睡眠时计算可以进一步提升准确率**13-18%**。

---

## 一、研究背景与核心问题

### 1.1 测试时计算的困境

近年来，测试时计算（Test-time Compute）已成为提升大语言模型推理能力的核心策略。从 OpenAI 的 o1/o3 系列到 DeepSeek-R1，这些模型通过在测试时进行更长时间的"思考"（生成更长的思维链），在数学、编程等复杂任务上取得了显著突破。

然而，这种能力提升伴随着严重的问题：

| 问题类型 | 具体表现 |
|---------|---------|
| **高延迟** | 用户可能需要等待数分钟才能得到回答 |
| **高成本** | 单次查询成本可达数十美元 |
| **冗余计算** | 对于同一上下文的多个相关问题，模型每次都需重新进行相似的推理过程 |

### 1.2 现有方法的局限性

当前的测试时计算范式存在一个根本假设：**查询（query）和上下文（context）在测试时同时提供**。这导致模型无法利用上下文提前可用的场景。

在实际应用中，许多 LLM 应用场景本质上是**有状态的（stateful）**：

- **文档问答系统**：文档作为上下文，用户可能提出多个相关问题
- **编程助手**：代码库作为上下文，开发者可能进行多轮调试
- **对话助手**：对话历史作为上下文，用户持续互动

在这些场景中，上下文在用户提问前就已经存在，但现有方法无法利用这一点来减少延迟和成本。

### 1.3 核心研究问题

基于上述观察，论文提出了以下核心问题：

> **能否利用上下文提前可用的时间窗口，在"睡眠"期间预计算有用信息，从而在不牺牲准确率的情况下大幅降低测试时的计算需求和延迟？**

---

## 二、Sleep-time Compute 技术方法详解

### 2.1 核心概念定义

Sleep-time Compute 引入了三阶段处理范式：

```
传统测试时计算范式：
  [Context + Query] → Test-time Compute → Answer
  
Sleep-time Compute 范式：
  Context → Sleep-time Compute → Re-represented Context' 
  [Context' + Query] → Reduced Test-time Compute → Answer
```

**形式化定义**：

- **测试时计算**：$T_B(q, c) \rightarrow a$
  - 用户查询 $q$，上下文 $c$，预算 $B$，输出答案 $a$
  
- **睡眠时计算**：$S(c) \rightarrow c'$
  - 仅基于上下文 $c$，生成重新表示的上下文 $c'$
  
- **结合计算**：$T_b(q, c') \rightarrow a$，其中 $b \ll B$

### 2.2 关键技术创新

#### 2.2.1 上下文重新表示（Context Re-representation）

Sleep-time Compute 的核心是将原始上下文转换为更适合回答未来查询的形式。具体实现包括：

1. **推断可能的问题**：基于上下文内容，模型生成可能被问到的相关问题
2. **预计算中间结果**：对上下文进行深度分析，提取关键信息、模式和关系
3. **结构化表示**：将分析结果以结构化方式组织，便于测试时快速检索

**实际应用示例**：

```
原始上下文（代码库）：
  - 包含多个模块和函数定义
  - 复杂的依赖关系

睡眠时处理后（重新表示的上下文）：
  - 架构模式识别结果
  - 关键函数调用图
  - 潜在调试策略建议
  - 优化机会识别
  - 常见问题及解决方案
```

#### 2.2.2 计算资源的跨查询摊销

当多个相关问题针对同一上下文时，Sleep-time Compute 的优势更加明显：

$$
\text{总成本} = \text{Sleep-time Cost} + \sum_{i=1}^{N} \text{Test-time Cost}_i
$$

由于 $c'$ 可以被所有查询共享，随着查询数量 $N$ 增加，平均成本显著下降。

实验显示，在 Multi-Query GSM-Symbolic 数据集上，平均成本可降低 **2.5倍**。

#### 2.2.3 与现有技术的对比

| 技术 | 工作原理 | 与 Sleep-time Compute 的关系 |
|-----|---------|---------------------------|
| **投机解码（Speculative Decoding）** | 使用草稿模型预测未来token | Sleep-time Compute 不仅预测token，还预测用户查询和有用推理 |
| **预计算/缓存** | 存储频繁使用的计算结果 | Sleep-time Compute 在自然语言空间进行表示学习 |
| **测试时缩放** | 在推理时增加计算 | Sleep-time Compute 是测试时计算的互补维度 |

### 2.3 方法实现细节

#### 2.3.1 睡眠时提示设计

论文采用提示工程实现 Sleep-time Compute，核心提示包括：

1. **任务描述**：告知模型需要在给定上下文上进行分析
2. **输出格式**：要求生成结构化的重新表示
3. **推理深度**：控制分析详尽程度（可调节计算预算）

对于非推理模型（如 GPT-4o），通过提示控制推理深度：
- "answer directly with a single sentence"
- "double check your reasoning before outputting the final answer"

对于推理模型（如 o1, o3-mini, DeepSeek-R1），利用其内置推理能力。

#### 2.3.2 并行睡眠时计算

为了进一步提升效果，论文提出了并行睡眠时计算：

$$
S_{parallel}(c) \rightarrow \{c'_1, c'_2, ..., c'_k\}
$$

即生成 $k$ 个不同的重新表示，在测试时拼接提供给模型。实验发现 $k=5$ 通常优于 $k=10$，表明存在最佳并行度。

---

## 三、实验设计与评测基准

### 3.1 新数据集构建

为了评测 Sleep-time Compute，论文作者构建了两个新数据集：

#### 3.1.1 Stateful GSM-Symbolic

基于 GSM-Symbolic 数据集改编，将问题分解为上下文和问题两部分：

**原始问题**：
```
Alice has 5 apples. She gives 2 to Bob and buys 3 more. 
How many apples does Alice have now?
```

**Stateful 分解**：
```
Context: Alice has 5 apples. She gives 2 to Bob and buys 3 more.
Query: How many apples does Alice have now?
```

包含两个难度级别：
- **P1**：添加1个约束条件（5000个样本）
- **P2**：添加2个约束条件（2500个样本）

#### 3.1.2 Stateful AIME

基于 AIME 2024 和 AIME 2025 竞赛题目改编，共60道题目。将数学问题拆分为：
- **上下文**：问题陈述和已知条件
- **查询**：具体的求解目标

#### 3.1.3 Multi-Query GSM-Symbolic

为了评测跨查询摊销效果，构建了每个上下文对应多个查询的数据集：

```
Context: [数学问题背景]
Queries:
  - Q1: 基础计算问题
  - Q2: 变体问题
  - Q3: 条件修改后的问题
  - ...
```

使用 o3-mini 自动生成额外的问题-答案对。

### 3.2 评测模型

| 数据集 | 评测模型 | 模型类型 |
|-------|---------|---------|
| Stateful GSM-Symbolic | GPT-4o-mini, GPT-4o | 非推理模型 |
| Stateful AIME | o1, o3-mini, Claude 3.7 Sonnet, DeepSeek-R1 | 推理模型 |

### 3.3 对比基线

1. **标准测试时计算**：上下文和查询同时提供
2. **仅上下文基线**：验证查询是否可从上下文中轻易推断（排除数据泄露）
3. **并行测试时计算（pass@k）**：Best-of-N 采样策略

---

## 四、实验结果与性能分析

### 4.1 核心实验结果

#### 4.1.1 Pareto 效率提升

实验结果显示，Sleep-time Compute 实现了测试时计算与准确率权衡曲线的 **Pareto 改进**：

**Stateful GSM-Symbolic（GPT-4o）**：
- 在相同准确率下，测试时计算减少 **~5倍**
- 在低测试时预算下，性能显著优于基线
- 在高预算下，标准方法略优（可能由于更少干扰信息）

**Stateful AIME（推理模型）**：
- o3-mini：显著提升，Pareto曲线外移
- Claude 3.7 Sonnet：显著提升
- DeepSeek-R1：显著提升（使用 budget forcing）
- o1：提升有限

#### 4.1.2 准确率提升

通过扩展睡眠时计算规模，可以进一步提升准确率：

| 数据集 | 准确率提升 |
|-------|-----------|
| Stateful GSM-Symbolic P2 (GPT-4o) | **+13%** |
| Stateful AIME (平均) | **+18%** |

#### 4.1.3 与并行测试时计算对比

Sleep-time Compute 在相同测试时token预算下，**始终优于** pass@k 并行采样：

- 无需 oracle verifier（pass@k 需要）
- 更低的推理延迟
- 更高的token效率

### 4.2 睡眠时计算规模扩展

通过增加并行生成的重新表示数量，可以进一步提升效果：

```
Stateful GSM-Symbolic P2 (GPT-4o):
  k=1: 基准性能
  k=5: 最佳性能 (+13%)
  k=10: 性能略有下降（可能由于冗余信息）
```

关键发现：**增加睡眠时计算可以持续改进 Pareto 曲线**，但存在边际效益递减。

### 4.3 多查询摊销效果

在 Multi-Query GSM-Symbolic 上，Sleep-time Compute 可以将：

$$
\text{平均成本/查询} = \frac{\text{Sleep-time Cost}}{N} + \text{Reduced Test-time Cost}
$$

当 $N$ 增加时，平均成本显著下降：
- **平均成本降低：2.5倍**
- 适用于文档问答、代码助手等多轮交互场景

### 4.4 效果影响因素分析

论文深入分析了 Sleep-time Compute 最有效的场景：

**查询可预测性**：
- Sleep-time Compute 效果与查询可预测性高度相关
- 上下文与查询关联度越高，效果越好
- 完全不可预测的查询受益有限

**上下文复杂度**：
- 复杂上下文（如大型代码库）受益更明显
- 简单上下文提升有限

---

## 五、案例研究：Agentic 软件工程任务

论文还进行了真实场景的案例研究：将 Sleep-time Compute 应用于软件工程（SWE）任务。

### 5.1 任务设置

- **上下文**：代码库
- **查询**：具体的编程任务或bug修复
- **目标**：利用睡眠时计算减少测试时延迟

### 5.2 应用方式

在睡眠时，模型：
1. 分析代码架构和依赖关系
2. 识别常见问题和解决方案
3. 预计算可能影响后续查询的静态分析结果

### 5.3 结果

在实际 SWE 任务中，Sleep-time Compute 成功减少了测试时计算需求，同时保持了解决方案质量。

---

## 六、深度理解与行业影响分析

### 6.1 技术突破的本质

Sleep-time Compute 的核心贡献在于**重新思考计算资源的时序分配**：

传统观点：
```
计算 = 训练时计算 + 测试时计算
```

Sleep-time Compute 引入新维度：
```
计算 = 训练时计算 + 睡眠时计算 + 测试时计算
```

这种重新分配带来了根本性的效率提升：
- **时间维度优化**：将计算从用户等待时间转移到空闲时间
- **共享性优化**：一次计算可被多次查询复用

### 6.2 与相关概念的关系

#### 6.2.1 与表示学习的关系

Sleep-time Compute 可以被视为在自然语言空间进行的**表示学习**：

| 传统表示学习 | Sleep-time Compute |
|------------|-------------------|
| 参数空间 | 自然语言空间 |
| 固定编码器 | 灵活的 LLM 推理 |
| 端到端训练 | 上下文相关的自适应 |

#### 6.2.2 与 RAG 的关系

Sleep-time Compute 与检索增强生成（RAG）有互补性：
- **RAG**：动态检索外部知识
- **Sleep-time Compute**：预计算上下文相关推理

两者可以结合：在睡眠时预计算检索策略和索引。

### 6.3 行业应用前景

#### 6.3.1 即时受益者

1. **代码助手（GitHub Copilot, Cursor）**
   - 代码库作为上下文
   - 多轮对话场景
   - 延迟敏感

2. **企业知识库问答**
   - 文档集合固定
   - 多用户查询同一文档
   - 可预测的问题模式

3. **客服系统**
   - 产品文档作为上下文
   - 重复性问题
   - 需要快速响应

#### 6.3.2 技术实现路径

```python
# 概念性架构
class SleepTimeCompute:
    def __init__(self, model, budget):
        self.model = model
        self.budget = budget
        self.cache = {}
    
    def sleep(self, context):
        """睡眠时计算：预计算重新表示"""
        if context not in self.cache:
            re_representation = self.model.generate(
                prompt=f"Analyze this context for future queries: {context}",
                max_tokens=self.budget
            )
            self.cache[context] = re_representation
        return self.cache[context]
    
    def query(self, query, context):
        """测试时查询：使用重新表示的上下文"""
        c_prime = self.cache.get(context, context)
        return self.model.generate(
            prompt=f"Context: {c_prime}\nQuery: {query}",
            max_tokens=reduced_budget  # 减少的预算
        )
```

### 6.4 潜在挑战与研究方向

#### 6.4.1 当前局限性

1. **查询可预测性依赖**
   - 不可预测查询受益有限
   - 需要机制判断何时应用 Sleep-time Compute

2. **上下文-查询分解假设**
   - 假设明确的上下文/查询分离
   - 实际交互可能更复杂（多轮对话中的上下文修改）

3. **信息冗余问题**
   - 过度预计算可能引入干扰信息
   - 需要平衡详尽性与相关性

#### 6.4.2 未来研究方向

1. **最优计算分配策略**
   - 如何在睡眠时和测试时之间最优分配计算预算
   - 自适应策略：根据上下文特性动态调整

2. **更复杂的交互模式**
   - 支持多轮对话中的增量上下文更新
   - 处理睡眠时长的变化（从几秒到几天）

3. **与模型架构的结合**
   - 专门的 Sleep-time Compute 架构设计
   - 混合专家模型（MoE）在睡眠时的应用

4. **表示学习深入研究**
   - 自然语言空间的表示学习理论
   - 上下文表示的压缩和索引

---

## 七、结论与展望

### 7.1 主要贡献总结

Sleep-time Compute 是一项具有里程碑意义的工作，它：

1. **提出了全新的计算范式**：将计算从测试时转移到睡眠时，显著降低延迟
2. **实现了显著的性能提升**：减少5倍测试时计算，同时提升13-18%准确率
3. **构建了新的评测基准**：Stateful GSM-Symbolic 和 Stateful AIME
4. **验证了大模型场景的可行性**：在真实 SWE 任务中验证效果

### 7.2 对大模型领域的意义

Sleep-time Compute 的意义不仅在于技术本身，更在于它**重新定义了我们思考 LLM 推理的方式**：

- **从被动响应到主动预测**：模型不再是单纯等待查询，而是主动预测和准备
- **从单次计算到分层计算**：引入睡眠时作为新的计算阶段
- **从通用策略到场景优化**：针对有状态场景进行专门优化

### 7.3 对研究者的建议

对于希望跟进这一方向的研究者：

1. **理解核心思想**：重点理解"预计算"和"计算时序重分配"的本质
2. **关注应用场景**：寻找具有明确上下文-查询分离的实际应用场景
3. **构建评测基准**：开发更多 stateful 任务的评测数据集
4. **探索理论边界**：研究 Sleep-time Compute 的理论极限和最优策略

### 7.4 对工程师的建议

对于希望应用这一技术的工程师：

1. **识别适用场景**：分析产品是否具有明显的上下文-查询分离特征
2. **设计缓存策略**：实现有效的睡眠时计算结果缓存和更新机制
3. **渐进式部署**：从简单场景开始，逐步扩展到复杂交互
4. **监控和调优**：建立效果监控，持续优化睡眠时计算预算分配

---

## 八、参考与延伸阅读

### 8.1 核心论文

- **Sleep-time Compute** (arXiv:2504.13171) - 本文分析的主体
- [GitHub 代码仓库](https://github.com/letta-ai/sleep-time-compute)

### 8.2 相关研究方向

1. **测试时计算扩展**
   - OpenAI o1/o3 技术报告
   - DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
   - Scaling LLM Test-Time Compute Optimally

2. **推理效率优化**
   - Speculative Decoding
   - KV Cache 优化
   - Early Exit 机制

3. **多智能体与记忆**
   - Agent 记忆系统设计
   - 上下文压缩技术
   - RAG 系统优化

---

## 附录：关键术语表

| 术语 | 定义 |
|-----|-----|
| **Sleep-time Compute** | 在查询到达前，利用上下文可用时间进行预计算的范式 |
| **Test-time Compute** | 在推理时增加计算资源以提升性能的方法 |
| **Stateful** | 应用具有持久化上下文的特性 |
| **Context Re-representation** | 将原始上下文转换为更适合回答查询的形式 |
| **Pareto Improvement** | 在不损害其他指标的情况下改善某一指标 |
| **Amortization** | 将单次计算成本分摊到多次查询 |

---

*本文深度解析了 Sleep-time Compute 的技术原理、实验结果和行业影响。这一工作代表了 LLM 推理效率优化的重要方向，值得研究者和工程师深入关注和实践。*

**分析日期**: 2026年3月31日  
**文章编号**: 148-2026-03-31
