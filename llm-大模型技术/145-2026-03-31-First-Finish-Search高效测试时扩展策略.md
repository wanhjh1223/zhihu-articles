# First Finish Search: 高效测试时扩展策略革新大模型推理效率

> **论文标题**: First Finish Search: Efficient Test-Time Scaling in Large Language Models  
> **作者**: Aradhye Agarwal, Ayan Sengupta, Tanmoy Chakraborty  
> **机构**: Indian Institute of Technology Delhi (印度理工学院德里分校)  
> **发表时间**: 2025年5月23日  
> **arXiv编号**: arXiv:2505.18149v1

---

## 一、研究背景与核心问题

### 1.1 大模型推理的 scaling law 瓶颈

大型语言模型（LLMs）在复杂推理任务上取得了令人瞩目的成就，主要得益于模型规模的扩大、预训练计算量的增加以及海量训练语料的使用。然而，进一步提升模型性能正变得越来越困难，因为单纯增加模型规模的收益正在递减，而预训练成本却在急剧上升。

近年来，研究界开始探索**测试时干预（Test-Time Intervention）**的可能性，即在不增加训练成本的前提下，通过在推理阶段动态分配计算资源来提升模型性能。这种方法被称为**测试时扩展（Test-Time Scaling, TTS）**，在复杂推理任务上表现出显著效果。

OpenAI 的 o1/o3 系列和 DeepSeek-R1 都是利用 TTS 提升推理能力的典型代表。这些模型通过延长推理链（Chain-of-Thought, CoT）长度，让模型进行更深入的思考，从而在数学、编程和逻辑推理任务上取得了突破。

### 1.2 现有TTS方法的局限性

根据 Zhang 等人提出的分类体系，现有的 TTS 策略可分为三大类：

| 类别 | 代表方法 | 核心思想 | 主要局限 |
|------|---------|---------|---------|
| **并行扩展** | Beam Search, Diverse Beam Search, Majority Voting | 同时生成多个候选输出，通过启发式或评分函数选择最佳响应 | 高token消耗、需要复杂的评估机制、 Majority Voting需要等待所有样本完成 |
| **序列扩展** | Budget Forcing, Thought Interleaving Penalty | 延长单条推理路径，引入延迟token鼓励深入思考 | 计算成本高、引入延迟、不适合实时部署 |
| **混合扩展** | MCTS, Self-Backtracking | 结合并行和序列策略，根据任务难度动态调整 | 实现复杂、需要模型权重访问、API不友好 |

**具体问题包括：**

1. **高token消耗**：并行方法如 Beam Search 和 Majority Voting 需要生成大量token
2. **推理延迟**：序列扩展方法如 Budget Forcing 通过引入"Wait"等人工延迟token来延长思考时间
3. **API不兼容**：许多方法需要访问模型权重或修改logits，无法通过标准API实现
4. **评估困难**：Majority Voting 假设输出可以容易地通过字符串相等性或语义相似性进行比较，这在开放式任务中往往不成立

### 1.3 核心发现：短轨迹更可能正确

本论文提出了一个令人惊讶的实证发现：**对于推理任务，较短的推理轨迹更有可能是正确的**。

作者通过对 QwQ-32B 和 R1-Distill-Qwen 模型的分析发现：
- 正确回答的推理轨迹长度分布明显偏向较短的一侧
- 错误回答的推理轨迹往往更长，包含更多冗余或错误的探索
- Welch 统计量为 16.56，p 值 < 0.001，表明这种差异具有高度统计显著性

这一发现挑战了传统的"更长推理 = 更好结果"的直觉，为设计更高效的 TTS 策略提供了新的思路。

---

## 二、First Finish Search (FFS) 方法详解

### 2.1 核心思想

基于"短轨迹更可能正确"的发现，作者提出了 **First Finish Search (FFS)** —— 一种**无需训练**的并行解码策略：

> **FFS 核心机制**：启动 n 个独立的样本并行生成，一旦任何一个样本完成（生成 EOS token），立即停止所有其他样本并返回该完成的轨迹。

这种方法的直觉是：如果较短的轨迹更可能是正确的，那么第一个完成的样本（即最短的轨迹）有很高的概率是正确的答案。

### 2.2 算法实现

FFS 提供两种实现变体：

#### 2.2.1 Sync-FFS（同步版本）

```python
# 伪代码示意
Algorithm 1: Sync-FFS
Input: Model M, Prompt P, Number of samples n
Output: Completed trace T

1. Initialize n partial sequences from P
2. While no sequence has generated EOS:
   a. Sample one token for each partial sequence in parallel (batch processing)
   b. Check if any token is EOS
3. Return the first completed trace
```

**特点**：
- 单模型副本，批处理所有样本
- 在每一步解码时同时处理 n 个部分序列
- 最大化计算和内存效率
- 适合集中式服务器或单GPU部署

#### 2.2.2 Async-FFS（异步版本）

```python
# 伪代码示意
Algorithm 2: Async-FFS
Input: Model M, Prompt P, Number of samples n
Output: Completed trace T

1. Launch n independent decoding jobs on separate processes/machines
2. Wait for any job to generate EOS or reach token limit
3. Interrupt remaining n-1 jobs
4. Free resources and return the completed trace
```

**特点**：
- 分布式部署，n 个独立解码任务
- 天然适合多worker或多机器环境
- 任一任务完成即可终止其他任务

### 2.3 关键设计决策

#### 2.3.1 轨迹长度测量

虽然核心假设是"短推理轨迹更正确"，但 FFS 实际测量的是**完整轨迹长度**（推理 + 最终答案）。这是因为：
- 答案部分通常很小
- 按完整长度排序几乎总是与按推理长度排序一致
- 简化实现，无需解析推理和答案边界

#### 2.3.2 Beam Size 设置为 1

FFS 故意使用 beam size = 1（即纯采样，无beam搜索），原因是：
- 最大化独立样本数量
- 增加短且正确的轨迹率先完成的概率
- 重复或退化的beam很少率先完成，自然被过滤掉

### 2.4 理论分析

#### 2.4.1 正确性概率分析

作者推导出给定长度轨迹的正确性概率公式：

**结果 1**：假设正确和错误样本分别来自均值为 μ₁、μ₂，标准差为 σ₁、σ₂ 的正态分布。令 α 为正确样本的比例。则随机采样轨迹 T 在长度为 x 时正确的概率为：

$$
\Pr[T \text{ is correct} \mid |T| = x] = \left[1 + \frac{1-\alpha}{\alpha} \cdot \frac{\sigma_1}{\sigma_2} \cdot e^{-\frac{1}{2}\left[\left(\frac{x-\mu_2}{\sigma_2}\right)^2 - \left(\frac{x-\mu_1}{\sigma_1}\right)^2\right]}\right]^{-1}
$$

**关键洞察**：
- 当 σ₁ ≈ σ₂ 时，正确概率取决于 μ₂ > μ₁（错误轨迹平均更长）
- 对于推理模型，这个不等式成立，证实了简洁性与准确性的联系

#### 2.4.2 序列成本分析

作者利用极值理论分析 FFS 成本随样本数增加的变化：

**结果 2**：设 Y₁, Y₂, ..., Yₙ 为 i.i.d. 正态分布随机变量，均值为 μ，标准差为 σ，则当 n → ∞ 时：

$$
\mathbb{E}[\min\{Y_1, Y_2, ..., Y_n\}] = \mu - \sigma\sqrt{2\log n}
$$

$$
\mathbb{E}[\max\{Y_1, Y_2, ..., Y_n\}] = \mu + \sigma\sqrt{2\log n}
$$

**对 FFS 的意义**：
- 最小trace长度（FFS的序列成本）随 n 增加以 O(√log n) 减小
- 最大trace长度（Majority Voting的序列成本）随 n 增加而增加
- **结论**：增加样本数可以降低 FFS 的预期成本，同时提高找到正确答案的概率

---

## 三、实验设置与评估

### 3.1 评估模型

作者在5个模型上评估 FFS：

| 模型 | 类型 | 参数量 | 说明 |
|------|------|--------|------|
| DeepSeek-R1 | 推理模型 | - | 当前最强开源推理模型之一 |
| R1-Distill-Qwen-32B | 蒸馏推理模型 | 32B | DeepSeek-R1蒸馏版本 |
| QwQ-32B | 推理模型 | 32B | Qwen系列推理模型 |
| Phi-4-Reasoning-Plus | 推理模型 | - | 微软推理模型 |
| DeepSeek-V3 | 基础模型 | - | 非推理模型，用于对比 |

### 3.2 评估数据集

| 数据集 | 类型 | 规模 | 难度 |
|--------|------|------|------|
| AIME24 | 数学竞赛 | 30题 | 高中竞赛级 |
| AIME25-I | 数学竞赛 | 15题 | 高中竞赛级 |
| AIME25-II | 数学竞赛 | 15题 | 高中竞赛级 |
| GPQA Diamond | 科学问答 | 198题 | 研究生级 |

**AIME 数据集**：美国数学邀请赛题目，答案为000-999之间的整数，用于测试数学推理能力。

**GPQA Diamond**：由领域专家设计的研究生级多选题，涵盖物理、化学、生物，测试高级概念推理。

### 3.3 评估指标

1. **Accuracy（准确率）**：与ground-truth完全匹配的比例
2. **Total Compute（总计算量）**：所有并行轨迹生成的token总数
3. **Sequential Compute（序列计算量）**：获取最终答案所需的最小顺序token数

### 3.4 对比基线

| 方法 | 类型 | 说明 |
|------|------|------|
| Simple Decoding (SD) | 基线 | 单样本解码 |
| Beam Search (BS) | 并行扩展 | 固定宽度的top-scoring部分序列 |
| Diverse Beam Search (DVBS) | 并行扩展 | 引入多样性惩罚 |
| Majority Voting (MV) | 并行扩展 | 采样N个答案，返回最高频 |
| Budget Forcing (BF) | 序列扩展 | 引入"Wait"token延长推理 |
| Last Finish Search (LFS) | 额外基线 | 作者提出的对比方法，返回最后完成的轨迹 |
| **FFS** | **并行扩展** | **本文方法** |

### 3.5 实现细节

- **采样参数**：top_p = 0.95, temperature = 0.6
- **样本数**：n = 4（对于所有采样方法）
- **最大生成长度**：AIME 32K tokens，GPQA 16K tokens
- **API**：使用 deepinfra.com API

---

## 四、实验结果深度分析

### 4.1 主要结果概览

#### 4.1.1 R1-Distill-Qwen & QwQ-32B

| 方法 | AIME24 Acc | AIME25-I Acc | AIME25-II Acc | GPQA Acc | Total Tokens | Sequential Tokens |
|------|-----------|-------------|--------------|---------|-------------|------------------|
| **FFS** | **53.3%** | 46.7% | 46.7% | 62.6% | **31.3K** | **7.8K** |
| MV | 53.3% | 46.7% | **53.3%** | 60.1% | 45.8K | 13.5K |
| LFS | 40.0% | 40.0% | 33.3% | 55.1% | 45.8K | 16.2K |
| BF | 46.7% | **53.3%** | 46.7% | **64.1%** | 28.4K | 28.4K |
| BS | 46.7% | 46.7% | 40.0% | 55.6% | 40.0K | 40.0K |

#### 4.1.2 DeepSeek-R1 & Phi-4-Reasoning-Plus

| 方法 | AIME24 Acc | AIME25-I Acc | AIME25-II Acc | GPQA Acc | Total Tokens | Sequential Tokens |
|------|-----------|-------------|--------------|---------|-------------|------------------|
| **FFS (R1)** | **86.7%** | **80.0%** | **93.3%** | - | **31.1K** | **7.8K** |
| MV (R1) | 73.3% | 73.3% | 80.0% | - | 42.2K | 11.6K |
| LFS (R1) | 60.0% | 46.7% | 60.0% | - | 42.2K | 15.2K |
| **FFS (Phi-4)** | 73.3% | 73.3% | **86.7%** | **67.2%** | **44.8K** | **11.2K** |
| MV (Phi-4) | 66.7% | 73.3% | 80.0% | 62.6% | 58.8K | 17.0K |

### 4.2 关键发现

#### 4.2.1 准确率提升显著

**DeepSeek-R1 上的突破**：
- FFS 在 AIME 数据集上达到 **82.23%** 的平均准确率
- 相比 DeepSeek-R1 独立推理的准确率提升 **15%**
- **接近 OpenAI o4-mini 的 83.7% 性能**
- 在 AIME25-II 上达到 **93.3%**，超越 Majority Voting 的 80.0%

**QwQ-32B 表现**：
- AIME25-II 达到 78.0%，仅比 MV 低 3.9 个百分点
- 使用 token 减少 25%（47.2K vs 59.7K）

**Phi-4-Reasoning-Plus**：
- GPQA 准确率达到 67.2%，超越 MV 4.5 个百分点
- token 预算减少 24%（44.8K vs 58.8K）

#### 4.2.2 计算效率大幅提升

**总计算量（Total Tokens）对比**：
- FFS vs MV：减少 **26-45%** 的 token 使用
- DeepSeek-R1：31.1K vs 42.2K（-26%）
- R1-Distill-Qwen：31.3K vs 45.8K（-32%）

**序列计算量（Sequential Tokens）对比**：
- FFS 的序列成本最低，因为它在第一个样本完成时立即停止
- DeepSeek-R1：7.8K vs MV 的 11.6K（-33%）
- QwQ-32B：11.8K vs MV 的 13-19K（-22-38%）

**实际意义**：
- 在 API 计费场景下，token 使用量直接转化为成本节省
- 在延迟敏感应用中，序列计算量的减少直接降低响应时间

#### 4.2.3 与 Budget Forcing 的对比

Budget Forcing (BF) 是一种序列扩展方法，通过引入"Wait"token强制延长推理：

| 维度 | FFS | BF |
|------|-----|-----|
| 准确率 | 相当或更好 | 在某些任务上略好 |
| 总token | 相当 | 28.4K |
| 序列token | 显著更低（7.8K） | 28.4K |
| API友好性 | 是 | 否（需要特殊token操作） |
| 实时性 | 更好 | 较差 |

**结论**：FFS 在保持准确率的同时，显著降低了序列延迟，且更易于部署。

### 4.3 模型规模的影响

实验发现 **FFS 从增加的模型容量中获益比任何竞争方法都多**：

- 在 DeepSeek-R1（最大模型）上，FFS 超越所有基线
- 在较小的蒸馏模型上，FFS 优势相对较小
- 这表明 FFS 能够有效利用更大模型的潜在推理能力

### 4.4 非推理模型的表现

在 DeepSeek-V3（非推理模型）上的实验显示了相反的趋势：
- Majority Voting 在所有数据集上都优于 FFS 和 LFS
- FFS 准确率最低（GPQA 50%，AIME25-II 20%）

**洞察**：
- "最短正确轨迹"的偏差特定于已经内化了多步推理训练的模型
- 对于没有显式 CoT 监督的模型，更长的推理可能确实更有帮助
- 这进一步证实了 FFS 特别适合推理增强型模型

### 4.5 扩展性分析

#### 4.5.1 样本数 n 的影响

理论分析表明，FFS 的预期序列成本随样本数 n 增加以 O(√log n) 减小：

- 增加并行样本数可以降低预期最短轨迹长度
- 同时增加找到正确答案的概率
- 这种"双赢"特性使 FFS 具有很强的扩展性

#### 4.5.2 实际扩展表现

实验验证：
- 当 n 从 1 增加到 4，FFS 的准确率和效率都提升
- 相比 Majority Voting，增加 n 会线性增加其序列成本
- FFS 的成本增长 sub-linear，使其在大规模部署时更具优势

---

## 五、方法对比与特性分析

### 5.1 五维对比框架

作者提出了五个关键维度来评估 TTS 方法：

| 维度 | 说明 | FFS | MV | BF | BS | DVBS | 训练方法 |
|------|------|-----|-----|-----|-----|------|---------|
| **Training-free** | 无需额外训练 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| **API-friendly** | 可通过标准API实现 | ✅ | ✅ | ⚠️ | ❌ | ❌ | ❌ |
| **Scalable** | 可通过增加计算持续提升性能 | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| **T-parallelizable** | 并行化可降低总计算成本 | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| **S-parallelizable** | 并行化可降低序列延迟 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

**FFS 的独特优势**：
- **唯一同时满足五个维度的方法**
- 特别是 **S-parallelizable**（序列并行化）是 FFS 独有的特性
- 这意味着增加并行样本不仅提高准确率，还能降低响应延迟

### 5.2 与现有方法的详细对比

#### 5.2.1 vs Majority Voting (MV)

**相似点**：
- 都是并行采样方法
- 都启动 n 个独立样本
- 都无需训练

**差异点**：
| 方面 | FFS | MV |
|------|-----|-----|
| 选择策略 | 第一个完成 | 最高频答案 |
| 终止条件 | 任一完成 | 全部完成 |
| 序列成本 | O(√log n) 递减 | 随 n 线性增加 |
| 评估需求 | 无需 | 需要答案相等性检查 |
| 开放式任务 | 适用 | 可能困难 |

**适用场景**：
- FFS 更适合需要低延迟的应用
- MV 在答案容易比较的场景（如多选题）中仍有用

#### 5.2.2 vs Budget Forcing (BF)

**相似点**：
- 都无需训练
- 都旨在提升推理质量

**差异点**：
| 方面 | FFS | BF |
|------|-----|-----|
| 扩展方向 | 并行 | 序列 |
| 延迟 | 低 | 高 |
| Token效率 | 高 | 中 |
| API实现 | 容易 | 困难（需要logit操作） |
| 核心假设 | 短轨迹更正确 | 长思考更好 |

**洞察**：
- FFS 和 BF 基于对立的假设（短vs长），但实验表明对于推理模型，短轨迹假设更成立

#### 5.2.3 vs Beam Search (BS)

**差异点**：
| 方面 | FFS | BS |
|------|-----|-----|
| 并行策略 | 独立样本 | 共享前缀的beam |
| 多样性 | 高（独立采样） | 低（相似前缀） |
| API友好 | 是 | 否（需要访问中间状态） |
| 早期终止 | 是 | 否 |

### 5.3 可视化对比

论文提供了清晰的图示来对比不同采样策略：

1. **Beam Search (BS)**：同步扩展 k 个部分假设，用模型概率 P(·) 排名
2. **Diverse Beam Search (DVBS)**：启动 g 个独立的单beam组，多样性项保持组间差异
3. **Majority Voting (MV)**：采样 N 个完整答案，用字符串相等测试选择最高频
4. **First Finish Search (FFS)**：启动 n 个随机样本，第一个到达 EOS 时终止

FFS 的关键优势在于**箭头被截断到最小长度**——所有计算在第一个完成时停止。

---

## 六、局限性与未来工作

### 6.1 当前局限性

#### 6.1.1 模型依赖性

- FFS 的有效性依赖于"短轨迹更正确"的假设
- 在非推理模型（如 DeepSeek-V3）上效果不佳
- 需要模型已经内化了有效的多步推理能力

#### 6.1.2 任务适用性

- 在需要开放式生成的任务中，答案比较可能比长度判断更困难
- 某些任务可能需要更长的推理链才能确保正确性

#### 6.1.3 理论假设

- 理论分析假设轨迹长度服从正态分布
- 实际分布可能有长尾（图3显示 Shapiro-Wilk 检验 p 值分布）
- 极端长度的轨迹行为可能与理论预测不同

### 6.2 未来研究方向

#### 6.2.1 混合策略

- 结合 FFS 的并行优势和序列扩展的深度探索
- 开发自适应策略，根据任务难度动态选择方法

#### 6.2.2 动态样本数

- 根据问题复杂度动态调整并行样本数 n
- 早期终止条件优化，平衡准确率和效率

#### 6.2.3 多模态扩展

- 将 FFS 应用于多模态推理任务
- 探索视觉-语言模型中的推理效率优化

#### 6.2.4 与强化学习的结合

- 训练模型生成更短但更准确的推理轨迹
- 将 FFS 的思想整合到 RL 训练目标中

---

## 七、个人理解与行业影响分析

### 7.1 核心洞察

#### 7.1.1 简洁性的价值

FFS 的核心发现——**短推理轨迹更可能正确**——挑战了关于 LLM 推理的一些传统假设：

1. **效率与质量的统一**：传统观点认为需要权衡效率和质量，但 FFS 表明简洁的推理可以同时实现两者
2. **冗余探索的代价**：过长的推理链往往包含不必要的探索、回溯或重复验证
3. **模型的"直觉"能力**：强大的推理模型似乎能够快速找到正确路径，而不需要大量试错

#### 7.1.2 测试时计算的重新思考

FFS 提出了一个关于测试时计算的新视角：

> **不是"用更多计算换更好结果"，而是"用更聪明的计算分配获得更好效率"**

这与传统的 TTS 思路形成对比：
- 传统：增加计算预算（更多token/更多样本）→ 提升性能
- FFS：优化计算分配（并行+早期终止）→ 同时提升性能和效率

### 7.2 对行业的影响

#### 7.2.1 API 成本优化

对于依赖 LLM API 的企业：
- FFS 可减少 26-45% 的 token 消耗
- 直接转化为成本节省
- 无需模型微调或基础设施变更

**实际案例估算**：
- 假设每月 API 成本 $10,000
- 采用 FFS 后可能节省 $2,600-$4,500/月
- 对于大规模部署，节省更可观

#### 7.2.2 实时应用加速

对于延迟敏感的应用：
- 序列计算量减少 22-38%
- 响应时间显著降低
- 特别适合客服机器人、实时助手等场景

#### 7.2.3 推理模型部署策略

FFS 的发现对推理模型的部署有重要影响：
- 强调简洁推理的价值
- 可能影响未来模型的训练目标
- 为推理模型的高效部署提供新思路

### 7.3 研究社区启示

#### 7.3.1 简单方法的力量

论文强调了简单方法的价值：
> "The elegance and simplicity of FFS demonstrate that straightforward TTS strategies can perform remarkably well, revealing the untapped potential of simple approaches at inference time."

这提醒我们：
- 在追求复杂方法之前，先验证简单假设
- 实证观察（短轨迹更正确）可以带来突破
- 工程简洁性往往是实际部署的关键

#### 7.3.2 反向思考的价值

FFS 采用了与主流相反的假设：
- 主流：更长推理 = 更好结果（o1/o3, BF 等）
- FFS：更短推理 = 更好结果

这种反向思考带来了：
- 更好的效率
- 相当或更好的准确率
- 更简单的实现

### 7.4 与其他前沿工作的联系

#### 7.4.1 与推理效率研究的关联

FFS 与以下研究方向形成呼应：

1. **Chain of Draft**：通过写更少来更快思考
2. **TokenSkip**：可控的 CoT 压缩
3. **ThinkPrune**：通过 RL 剪枝长 CoT
4. **Dynamic Early Exit**：推理模型的动态早期退出

这些工作共同指向一个趋势：**推理效率优化正在成为 LLM 研究的重要方向**。

#### 7.4.2 与模型能力的关系

FFS 的有效性依赖于模型已经具备强大的推理能力：
- 在 R1、QwQ、Phi-4-Reasoning 上表现优异
- 在 V3（非推理模型）上表现不佳

这表明：
- **高效利用能力**需要**先有能力**
- 推理模型的训练质量直接影响 FFS 的效果
- 未来模型设计可能需要考虑与 FFS 类方法的兼容性

### 7.5 批评与反思

#### 7.5.1 样本数限制

实验中使用的样本数 n=4 相对较小：
- 更大 n 值下的行为如何？
- 是否存在收益递减点？
- 实际部署中 n 的选择策略是什么？

#### 7.5.2 任务范围

评估主要集中在数学和科学推理：
- 在其他领域（如代码生成、创意写作）的表现如何？
- 开放式任务的评估挑战如何解决？
- 需要更广泛的基准测试

#### 7.5.3 理论基础的完善

虽然提供了理论分析，但仍有改进空间：
- 非正态分布下的行为
- 更精确的成本-收益权衡模型
- 与信息论或最优停止理论的联系

---

## 八、结论

### 8.1 核心贡献总结

1. **实证发现**：揭示了推理任务中"短轨迹更正确"的现象
2. **方法创新**：提出了 First Finish Search，一种简单但高效的 TTS 策略
3. **理论分析**：提供了 FFS 正确性和成本特性的数学解释
4. **实验验证**：在多个模型和数据集上验证了 FFS 的有效性

### 8.2 关键成果

- 在 DeepSeek-R1 上达到 **82.23%** AIME 准确率，接近 OpenAI o4-mini
- 相比 Majority Voting 减少 **26-45%** 的 token 使用
- 序列计算量减少 **22-38%**，显著降低延迟
- **唯一**同时满足 Training-free、API-friendly、Scalable、T-parallelizable、S-parallelizable 的方法

### 8.3 实践建议

**对于开发者**：
- 如果正在使用推理模型（R1、QwQ、o1等），FFS 是一个值得尝试的策略
- 实现简单，无需修改模型或训练
- 可以立即降低 API 成本

**对于研究人员**：
- 关注推理效率优化这一新兴方向
- 考虑简洁性与准确性的关系
- 探索 FFS 与其他方法的结合

**对于模型设计者**：
- 在训练推理模型时考虑简洁性目标
- 设计模型架构以支持高效的测试时策略
- 将 FFS 的思想整合到训练流程中

### 8.4 展望

First Finish Search 代表了一种新的测试时计算范式：**不是通过增加计算来提升性能，而是通过更聪明的计算分配来同时提升性能和效率**。

随着 LLM 应用的大规模部署，计算效率将变得越来越重要。FFS 及其后续研究可能为构建既强大又经济的 AI 系统提供关键思路。

---

## 参考资料

1. Agarwal, A., Sengupta, A., & Chakraborty, T. (2025). First Finish Search: Efficient Test-Time Scaling in Large Language Models. *arXiv preprint arXiv:2505.18149*.

2. DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *arXiv preprint arXiv:2501.12948*.

3. Muennighoff, N., et al. (2025). s1: Simple test-time scaling. *arXiv preprint arXiv:2501.19393*.

4. Snell, C., et al. (2024). Scaling LLM test-time compute optimally can be more effective than scaling model parameters. *arXiv preprint arXiv:2408.03314*.

5. Xu, S., et al. (2025). Chain of draft: Thinking faster by writing less. *arXiv preprint arXiv:2502.18600*.

---

*本文分析基于 arXiv:2505.18149v1，如有理解偏差，请以原文为准。*
