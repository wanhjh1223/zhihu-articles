# MAGRPO: 多智能体强化学习驱动的LLM协作新范式

> **论文标题**: LLM Collaboration With Multi-Agent Reinforcement Learning  
> **作者**: Shuo Liu, Zeyu Liang, Xueguang Lyu, Christopher Amato  
> **机构**: Northeastern University (Khoury College of Computer Sciences)  
> **发表时间**: 2025年8月  
> **arXiv链接**: https://arxiv.org/abs/2508.04652

---

## 一、研究背景与核心问题

### 1.1 多智能体系统的崛起与挑战

随着大语言模型（LLM）能力的快速提升，单一模型已经难以满足复杂任务的需求。多智能体系统（Multi-Agent Systems, MAS）应运而生——通过多个专业化智能体的协作，实现远超单个模型能力上限的系统性智能。这一趋势在2024-2025年尤为明显：从OpenAI的Multi-Agent框架到各类Agent编排工具，多智能体协作已成为大模型应用的核心范式。

然而，当前的多智能体系统面临一个根本性瓶颈：**智能体之间缺乏真正的协同优化**。

现有的多智能体系统主要依赖以下两种模式：

| 模式 | 特点 | 局限性 |
|------|------|--------|
| **提示工程驱动** | 通过精心设计的prompt定义角色和交互规则 | 依赖人工设计，难以扩展，智能体行为不可控 |
| **独立微调** | 每个智能体单独进行SFT或RL训练 | 缺乏联合优化，智能体之间无法形成有效配合 |

这种"拼接式"的架构导致智能体之间经常出现：
- **信息冲突**：不同智能体给出矛盾的回答
- **重复劳动**：多个智能体做相似的工作
- **责任推诿**：复杂任务中谁都不愿承担责任
- **效率低下**：通信开销大，响应时间长

### 1.2 核心研究问题

本论文直面一个关键科学问题：**如何让多个LLM智能体通过端到端的强化学习训练，真正学会协作？**

具体而言，研究者们试图解决以下子问题：

1. **形式化建模**：如何将LLM协作问题转化为可优化的数学框架？
2. **信用分配**：在联合决策中，如何确定每个智能体对最终结果的贡献？
3. **训练稳定性**：多智能体环境的非平稳性（non-stationarity）如何克服？
4. **效率与质量的平衡**：协作是否能同时提升响应质量和效率？

### 1.3 与现有工作的区别

| 现有方法 | MAGRPO |
|----------|--------|
| 基于提示的协调（如AutoGen、MetaGPT） | 参数级联合优化 |
| 独立RL训练（各智能体有自己的奖励） | 集中式优势估计，共享优化目标 |
| 预定义角色和流水线 | 涌现式协作策略 |
| 单回合交互 | 多回合深度协作 |

---

## 二、技术方法详解

### 2.1 问题形式化：Dec-POMDP框架

MAGRPO的核心创新之一是将LLM协作严格建模为**分散式部分可观测马尔可夫决策过程（Decentralized Partially Observable Markov Decision Process, Dec-POMDP）**。

#### Dec-POMDP的形式化定义

一个Dec-POMDP可以表示为七元组 $(\mathcal{I}, \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O})$：

- **$\mathcal{I} = \{1, 2, ..., n\}$**：智能体集合，每个智能体对应一个可训练的LLM
- **$\mathcal{S}$**：环境状态空间
- **$\mathcal{A} = \mathcal{A}_1 \times \mathcal{A}_2 \times ... \times \mathcal{A}_n$**：联合动作空间，每个动作是一个文本响应
- **$\mathcal{T}: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{S})$**：状态转移函数
- **$\mathcal{R}: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$**：全局奖励函数
- **$\Omega = \Omega_1 \times \Omega_2 \times ... \times \Omega_n$**：观测空间
- **$\mathcal{O}: \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\Omega)$**：观测函数

#### LLM协作的特殊性

与传统MARL问题不同，LLM协作具有以下独特特征：

**1. 动作空间的高维离散性**
- 每个动作是词表（vocabulary）上的序列生成
- 动作空间大小为 $|V|^{L}$，其中 $|V|$ 是词表大小，$L$ 是生成长度
- 直接应用传统MARL方法不可行

**2. 部分可观测性的语义特性**
- 每个智能体只能看到分配给它的prompt/历史信息
- 需要通过自然语言通信来共享信息
- 观测是"语义级"的而非传感器数据

**3. 回合制协作的动态性**
- 智能体同步生成响应（simultaneous move）
- 环境根据联合响应演化
- 对话历史随时间累积

### 2.2 MAGRPO算法：多智能体GRPO

MAGRPO的核心是**Multi-Agent Group Relative Policy Optimization**，这是GRPO（Group Relative Policy Optimization）在多智能体场景的自然扩展。

#### 2.2.1 单智能体GRPO回顾

在DeepSeek-R1等工作中，GRPO展示了卓越的训练效率和稳定性。其核心思想是：

1. 对每个输入采样 $G$ 个候选响应（group）
2. 计算相对优势（relative advantage）：$A^{(g)} = R^{(g)} - \text{mean}(R)$
3. 使用PPO-clip更新策略，无需显式价值网络

GRPO的优势在于：
- **无需critic模型**：节省显存和计算
- **降低方差**：相对优势估计比绝对奖励更稳定
- **与RLHF兼容**：可直接使用基于规则的奖励

#### 2.2.2 多智能体扩展：MAGRPO

MAGRPO将GRPO扩展到多智能体场景，核心公式如下：

**集中式优势估计：**

对于每个回合 $t$ 和组内样本 $g$，定义集中式优势为：

$$\hat{A}^{(g)}_t = \frac{R^{(g)}_t - \frac{1}{G}\sum_{g=1}^{G}R^{(g)}_t}{\sigma(R^{\mathcal{G}}_t)}$$

其中：
- $R^{(g)}_t = \sum_{\tau=t}^{H-1} r^{(g)}_{\tau}$ 是从时刻 $t$ 开始的累积回报
- $\sigma(\cdot)$ 是组内奖励的标准差，用于归一化

**关键洞察**：所有智能体共享同一个优势估计 $\hat{A}^{(g)}_t$，这意味着：
- 如果一个组内的响应获得了高奖励，所有参与该响应生成的智能体都得到正反馈
- 如果响应质量差，所有智能体共同承担"责任"
- 这种"绑定"机制迫使智能体学习协调

**分布式策略更新：**

每个智能体 $i$ 独立更新其策略 $\pi_{\theta_i}$：

$$J(\theta_i) = \mathbb{E}_{\mathbf{o}_0 \sim \mathcal{D}, \mathbf{h}^{\mathcal{G}} \sim \boldsymbol{\pi}_{\boldsymbol{\theta}, \text{old}}} \left[ \frac{1}{|B|}\frac{1}{|\mathcal{G}|}\sum_{h_i^{\mathcal{G}} \in B}\sum_{g \in \mathcal{G}} \min\left(\rho^{(g)}_{i,t}\hat{A}^{(g)}_t, \text{clip}(\rho^{(g)}_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}^{(g)}_t\right) \right]$$

其中重要性采样比为：

$$\rho^{(g)}_{i,t} = \frac{\pi_{\theta_i}(a^{(g)}_{i,t}|h^{(g)}_{i,t})}{\pi_{\theta_i, \text{old}}(a^{(g)}_{i,t}|h^{(g)}_{i,t})}$$

**CTDE架构的体现**：
- **集中式（Centralized）**：训练时使用共享的优势估计，所有智能体朝着同一方向优化
- **分散式（Decentralized）**：执行时每个智能体只基于自己的观测和策略生成响应

### 2.3 奖励设计

MAGRPO采用**稀疏的全局奖励**机制，这对于协作学习至关重要。

#### 写作协作任务的奖励

```python
# 伪代码示意
def compute_writing_reward(agent_outputs, reference_text):
    # 1. 内容覆盖度：是否包含关键信息点
    coverage_score = check_coverage(agent_outputs, reference_text)
    
    # 2. 一致性：多个智能体的输出是否逻辑一致
    consistency_score = check_consistency(agent_outputs)
    
    # 3. 流畅性：生成的文本是否自然
    fluency_score = language_model_score(agent_outputs)
    
    return coverage_score + consistency_score + fluency_score
```

#### 代码协作任务的奖励

```python
def compute_coding_reward(agent_outputs, test_cases):
    # 1. 语法正确性
    syntax_correct = check_syntax(agent_outputs)
    
    # 2. 结构合理性：函数定义、模块化等
    structure_score = evaluate_structure(agent_outputs)
    
    # 3. 测试通过率（最重要）
    test_pass_rate = run_tests(agent_outputs, test_cases)
    
    # 4. 协作效率：是否避免了重复代码
    cooperation_score = check_cooperation(agent_outputs)
    
    return test_pass_rate * (syntax_correct + structure_score + cooperation_score)
```

**为什么全局奖励优于个体奖励？**

| 奖励类型 | 优点 | 缺点 |
|----------|------|------|
| **个体奖励** | 信用分配明确 | 需要人工设计每个角色的奖励函数，容易产生"局部最优" |
| **全局奖励** | 鼓励真正的协作，涌现最优分工 | 信用分配问题，需要通过优势估计解决 |

MAGRPO通过组相对优势估计，在不引入复杂信用分配机制的情况下，有效解决了多智能体协作的优化问题。

### 2.4 多回合协作机制

MAGRPO支持**多回合（multi-turn）**协作，这是其与单回合方法的关键区别。

#### 单回合 vs 多回合

| 特征 | 单回合 | 多回合 |
|------|--------|--------|
| 交互深度 | 浅层，一次性响应 | 深层，可迭代改进 |
| 信息流动 | 单向（输入→输出） | 双向（输出→反馈→改进） |
| 适用场景 | 简单任务 | 复杂推理、代码生成 |
| 训练难度 | 较易 | 需要处理长程依赖 |

#### 多回合协作流程

```
Round 1:
  Agent A (Helper) → 生成辅助函数/思路
  Agent B (Main) → 生成主函数/初稿
  
  ↓ 外部评审（如Claude-Sonnet-4）
  
Round 2:
  Agent A ← 获得编辑建议 + 上一轮输出
  Agent B ← 获得编辑建议 + 上一轮输出
  
  Agent A → 改进版辅助函数
  Agent B → 改进版主函数
  
  ↓ 重复直到收敛或达到最大回合数
```

**关键设计**：
- 使用折扣因子 $\gamma = 1.0$（即不减益未来奖励），鼓励长期协作
- 回合间通过对话历史传递信息
- 外部评审者（external critic）提供结构化反馈

---

## 三、实验设计与结果分析

### 3.1 实验设置

#### 模型与任务

| 配置 | 详情 |
|------|------|
| **基础模型** | Qwen2.5-Coder-7B（单智能体基线）<br>Qwen2.5-Coder-3B × 2（多智能体配置） |
| **任务1** | 写作协作（Writing Collaboration） |
| **任务2** | 代码协作（Coding Collaboration） |
| **训练步数** | 写作：1,500步<br>代码（单回合）：1,500步<br>代码（多回合）：2,200步 |

#### 对比基线

| 方法 | 描述 |
|------|------|
| **Single 7B** | 单个大模型（Qwen2.5-Coder-7B）独立完成任务 |
| **Naive Concatenation** | 两个3B模型各自独立生成，结果简单拼接 |
| **Sequential Pipeline** | 辅助智能体先生成，主智能体看到后生成 |
| **One-Round Discussion** | 两个智能体各自生成后交换意见，再各自修改一轮 |
| **MAGRPO (Ours)** | 本文方法，包含单回合和多回合两个变体 |

### 3.2 主要实验结果

#### 代码协作性能（HumanEval / CodeContests）

| 方法 | 速度(tokens/s) | 响应时间(s) | 结构(%) | 语法(%) | 测试通过(%) | 协作得分 | 总回报 |
|------|----------------|-------------|---------|---------|-------------|----------|--------|
| Single 7B | 73.1 | 1.6 | 100.0 | 100.0 | 64.8 | - | - |
| Naive Concat | 99.6 | 2.2 | 98.4 | 96.5 | 56.4 | 35.1 | 63.1 |
| Sequential | 97.4 | 2.0 | 97.5 | 96.3 | 55.2 | 35.2 | 62.5 |
| Discussion | 82.5 | 2.8 | 98.1 | 94.8 | 41.2 | 30.2 | 57.5 |
| **MAGRPO (单回合)** | **190.0** | **1.5** | **100.0** | **97.8** | **61.6** | **83.4** | **83.7** |
| **MAGRPO (多回合)** | **95.2** | **2.8** | **99.9** | **97.3** | **67.9** | **84.9** | **85.8** |

**关键发现**：

1. **效率优势**：MAGRPO单回合版本以2.5倍于7B模型的速度（190 vs 73.1 tokens/s），达到了接近的测试通过率（61.6% vs 64.8%）
2. **质量优势**：MAGRPO多回合版本在测试通过率上超越了单模型（67.9% vs 64.8%）
3. **协作指标**：MAGRPO的协作得分（~84）显著高于所有基线（~30-35），证明智能体真正学会了配合

#### 写作协作性能

| 方法 | 效率 | 内容覆盖 | 一致性 | 流畅性 | 总回报 |
|------|------|----------|--------|--------|--------|
| Single 7B | 65.5 | 75.2 | 82.1 | 88.3 | 78.4 |
| Naive Concat | 48.2 | 68.5 | 45.3 | 82.1 | 56.8 |
| Sequential | 52.1 | 71.2 | 52.8 | 84.5 | 62.3 |
| **MAGRPO** | **88.1** | **82.3** | **89.4** | **91.2** | **87.5** |

### 3.3 定性分析：涌现的协作策略

通过案例研究，研究者发现MAGRPO训练后，智能体涌现出了多种协作模式：

**模式1：专业化分工（Specialization）**
```
Agent A (Helper): 专门负责处理边界情况、错误检查
Agent B (Main): 专注于核心逻辑实现

→ 两个智能体不再重复工作，而是互补
```

**模式2：迭代精化（Iterative Refinement）**
```
Round 1: Agent A 提出算法框架
         Agent B 尝试实现但遇到问题

Round 2: Agent A 根据B的问题调整框架
         Agent B 成功实现

→ 形成类似"产品经理-工程师"的协作关系
```

**模式3：验证与确认（Verification）**
```
Agent A 生成代码后，Agent B 自动扮演代码审查者角色
即使B是"主"智能体，也会先验证A的输出再集成

→ 内置的质量保证机制
```

### 3.4 消融实验

| 变体 | 测试通过率 | 说明 |
|------|------------|------|
| 完整MAGRPO | 67.9% | 本文方法 |
| 无优势归一化 | 58.2% | 去掉分母的标准差归一化，训练不稳定 |
| 个体奖励 | 52.4% | 每个智能体有自己的奖励，缺乏协作 |
| 单智能体GRPO × 2 | 48.7% | 两个智能体独立训练GRPO，无联合优化 |
| 无多回合 | 61.6% | 只有单回合版本 |

**关键洞察**：集中式优势估计是MAGRPO成功的关键，移除它会导致协作显著下降。

---

## 四、核心创新点深度解读

### 4.1 创新点1：Dec-POMDP形式化

将LLM协作建模为Dec-POMDP是理论上的重要突破。

**为什么重要？**

传统上，LLM协作被视为"工程问题"——通过提示工程和角色设计来实现。MAGRPO将其提升为**可优化的数学问题**，这使得：
- 可以应用成熟的MARL理论成果
- 能够分析系统的收敛性和最优性
- 为后续研究提供了严格的理论框架

**与传统MARL的区别**：

| 传统MARL | LLM-based MARL (MAGRPO) |
|----------|-------------------------|
| 动作是低维连续或离散 | 动作是高维文本序列 |
| 观测是传感器数据 | 观测是自然语言指令 |
| 奖励通常是密集的 | 奖励是稀疏的、基于结果的 |
| 环境是物理或游戏 | 环境是代码执行器/质量评估器 |

### 4.2 创新点2：组相对优势的集中式估计

这是MAGRPO最核心的算法创新。

**对比：传统信用分配方法**

| 方法 | 原理 | 适用性 |
|------|------|--------|
| **Independent PPO** | 每个智能体有自己的critic | 适用于独立任务，不适用于协作 |
| **QMIX/VDN** | 通过mixer分解联合Q函数 | 需要离散动作空间，不适合LLM |
| **MAPPO** | 集中式critic，估计联合价值 | 需要训练大型critic网络 |
| **MAGRPO (本文)** | 组相对优势，无需critic | 特别适合LLM的响应生成场景 |

**组相对优势的独特优势**：

1. **计算效率**：无需额外的价值网络
2. **方差降低**：相对优势比绝对奖励更稳定
3. **协作激励**：共享优势迫使智能体"同甘共苦"
4. **可扩展性**：组大小 $G$ 是唯一的额外超参数

### 4.3 创新点3：CTDE在LLM场景的成功应用

CTDE（Centralized Training with Decentralized Execution）是MARL的经典范式，MAGRPO成功将其应用于LLM。

**CTDE在LLM中的实现**：

```
训练阶段（Centralized）:
  ┌─────────────────────────────────────┐
  │  所有智能体同时生成响应组 {a1, a2, ..., aG}  │
  │  计算全局奖励 R                        │
  │  计算组相对优势 Â                      │
  │  所有智能体使用 Â 更新策略              │
  └─────────────────────────────────────┘

执行阶段（Decentralized）:
  Agent 1: o1 → πθ1 → a1  （仅基于自己的观测）
  Agent 2: o2 → πθ2 → a2
  ...
  Agent n: on → πθn → an
```

**为什么CTDE对LLM特别重要？**

- **训练时**：可以利用额外的计算资源（如critic、组采样）来提升学习效果
- **执行时**：保持各智能体的独立性和模块化，便于部署和维护
- **通信效率**：执行时无需智能体间通信，降低延迟

---

## 五、局限性与开放问题

### 5.1 论文指出的局限性

作者诚实地讨论了以下局限：

**1. 短程任务限制**
- 当前实验主要关注单回合和两回合交互
- 长程多回合协作（如10+回合）的训练稳定性待验证

**2. 智能体数量限制**
- 实验主要使用2个智能体
- 扩展到5+智能体时，联合动作空间爆炸问题

**3. 奖励设计依赖**
- 仍需要领域专家设计奖励函数
- 自动奖励学习（如从人类反馈中学习）尚未探索

**4. 计算开销**
- 组采样需要 $G$ 倍的推理计算
- 多回合设置进一步增加计算量

### 5.2 我分析的潜在局限

**1. 基线模型的选择**
- 使用7B vs 3B×2的对比不完全公平
- 更公平的对比应该是：7B vs 3B×2（MAGRPO优化后）vs 7B（MAGRPO单智能体版本）

**2. 任务范围的局限**
- 主要关注写作和代码两个领域
- 在其他领域（如数学推理、科学发现）的有效性待验证

**3. 可解释性**
- 虽然观察到涌现的协作策略，但缺乏系统性的分析工具
- 智能体具体"学到了什么协作知识"尚不清楚

### 5.3 开放研究问题

| 问题 | 重要性 | 可能的解决方向 |
|------|--------|----------------|
| **长程协作训练** | ⭐⭐⭐⭐⭐ | 引入分层RL，将长程协作分解为子任务 |
| **动态智能体数量** | ⭐⭐⭐⭐ | 学习何时添加/移除智能体，类似神经架构搜索 |
| **异构智能体** | ⭐⭐⭐⭐ | 不同架构（如一个Decoder-only，一个Encoder-Decoder）的协作 |
| **跨域迁移** | ⭐⭐⭐ | 将在代码任务上学到的协作策略迁移到写作任务 |
| **人机协作** | ⭐⭐⭐⭐⭐ | 将人类作为智能体之一纳入MAGRPO框架 |

---

## 六、个人理解与行业影响分析

### 6.1 技术层面的洞察

**洞察1：协作是一种"元能力"**

MAGRPO揭示了一个深刻的洞见：**协作能力可以通过RL涌现，而非必须通过人工设计**。

这与传统观点形成对比：
- 传统观点：协作需要预定义角色、通信协议、协调机制
- MAGRPO观点：给定合适的奖励信号和优化目标，协作作为一种策略会自动涌现

**洞察2：规模法则的新维度**

LLM的规模法则（Scaling Law）通常关注：参数数量 × 数据量 × 计算量

MAGRPO引入了第四个维度：**智能体数量 × 协作质量**

```
性能 = f(参数规模, 数据量, 训练计算, 智能体数量, 协作效率)
```

这意味着：与其训练一个100B的模型，不如训练10个10B的模型并让它们高效协作。

**洞察3：从"模型能力"到"系统能力"**

MAGRPO代表了一个重要范式转移：
- 不再只关注单个模型的能力上限
- 而是关注多个模型如何组成一个更强大的系统
- 这与复杂系统理论中的"涌现性"（emergence）概念一致

### 6.2 行业影响预测

**短期影响（6-12个月）**

1. **Agent框架的升级**
   - 现有的Agent编排框架（如LangGraph、AutoGen）可能集成类似MAGRPO的训练机制
   - 从"手工编排"向"学习编排"转变

2. **模型即服务（MaaS）的新形态**
   - 多个小模型协作可能替代单个大模型
   - 降低推理成本（如MAGRPO单回合版本所示，2.5倍速度提升）

3. **代码生成工具的革新**
   - GitHub Copilot等工具可能采用多智能体架构
   - 一个Agent写代码，一个Agent写测试，一个Agent审查

**中期影响（1-3年）**

1. **企业级AI系统的重构**
   - 企业内部的不同AI系统（客服、销售、技术支持）不再孤立运行
   - 通过MAGRPO训练，形成一个协作的企业级AI组织

2. **多模态Agent的普及**
   - 将MAGRPO扩展到多模态：一个Agent处理图像，一个处理文本，一个处理音频
   - 实现真正的"多模态协作理解"

3. **AI研究本身的自动化**
   - 多个研究Agent协作：文献调研Agent、实验设计Agent、论文写作Agent
   - 加速科学发现进程

**长期影响（3-5年）**

1. **AI组织（AI Organization）**
   - 可能出现完全由AI Agent组成的公司/组织
   - 每个Agent有专业分工，通过MAGRPO类算法协调

2. **人机协作的新范式**
   - 人类作为MAGRPO系统中的一个智能体
   - AI Agent学习如何与人类高效协作，而非仅执行命令

3. **AGI路径的多样性**
   - 证明AGI不一定需要单一超级模型
   - 多个专业模型的协作系统可能是一条可行路径

### 6.3 对相关领域的影响

| 领域 | 影响 |
|------|------|
| **MARL** | 开辟LLM-based MARL新子领域，推动理论与应用的结合 |
| **LLM训练** | 从单模型优化扩展到多模型联合优化的新研究分支 |
| **系统工程** | 复杂AI系统的设计原则需要重新思考 |
| **经济学** | AI Agent间的协作与博弈可能产生新的经济模型 |
| **组织行为学** | 人类团队管理理论可能启发AI Agent管理 |

---

## 七、结论

MAGRPO是一项具有里程碑意义的研究，它成功地将多智能体强化学习与大语言模型相结合，开辟了一条全新的技术路径。

### 核心贡献总结

1. **理论贡献**：首次将LLM协作严格建模为Dec-POMDP，建立了可优化的数学框架
2. **算法贡献**：提出MAGRPO算法，将GRPO扩展到多智能体场景，实现集中式训练与分散式执行
3. **实验贡献**：在写作和代码协作任务上验证了方法的有效性，展示了协作带来的效率和质量双提升

### 技术价值

| 维度 | 评价 |
|------|------|
| **创新性** | ⭐⭐⭐⭐⭐ 首次端到端训练多LLM协作 |
| **实用性** | ⭐⭐⭐⭐ 2.5倍效率提升，同时保持质量 |
| **理论深度** | ⭐⭐⭐⭐ Dec-POMDP形式化具有理论价值 |
| **可扩展性** | ⭐⭐⭐ 扩展到更多智能体和更长程协作仍需探索 |
| **影响力** | ⭐⭐⭐⭐⭐ 可能重塑多Agent系统的研究范式 |

### 未来展望

MAGRPO不仅是一个算法，更是一个**研究纲领（Research Program）**的开端。它提出了一个核心问题：**如何训练AI系统使其学会协作？**

这个问题的答案将影响：
- 下一代AI系统的架构设计
- AGI的实现路径
- 人机协作的未来形态

正如论文作者所言："Our approach opens the door to using other MARL methods for LLMs and highlights the associated challenges."

MAGRPO打开了一扇门，门后是一个充满可能性的新世界。

---

## 参考资料

1. Liu, S., Liang, Z., Lyu, X., & Amato, C. (2025). LLM Collaboration With Multi-Agent Reinforcement Learning. arXiv preprint arXiv:2508.04652.

2. Guo, D., et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2501.12948.

3. Oliehoek, F. A., & Amato, C. (2016). A Concise Introduction to Decentralized POMDPs. Springer.

4. Yu, C., et al. (2022). The Surprising Effectiveness of MAPPO in Cooperative Multi-Agent Games. NeurIPS 2022.

5. Albrecht, S. V., Christianos, F., & Schäfer, L. (2024). Multi-Agent Reinforcement Learning: Foundations and Modern Approaches. MIT Press.

---

*本文分析基于MAGRPO论文原文及相关资料，所有实验数据和图表均来自原论文。如有理解偏差，欢迎指正讨论。*
