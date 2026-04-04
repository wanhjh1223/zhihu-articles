# ProCeedRL: 过程级Critic与探索性演示强化学习突破LLM智能体推理瓶颈

> **论文标题**: ProCeedRL: Process Critic with Exploratory Demonstration Reinforcement Learning for LLM Agentic Reasoning  
> **作者**: Jingyue Gao, Yanjiang Guo, Xiaoshuai Chen, Jianyu Chen  
> **机构**: 清华大学交叉信息研究院、上海期智研究院  
> **发表时间**: 2026年4月2日  
> **论文链接**: [arXiv:2604.02006](https://arxiv.org/abs/2604.02006)

---

## 一、研究背景与核心问题

### 1.1 智能体强化学习的探索困境

随着DeepSeek-R1、OpenAI o1等推理模型的突破，**可验证奖励强化学习（RLVR）**已成为提升大语言模型推理能力的主流范式。然而，将RLVR应用于多轮智能体任务仍面临严峻挑战：

- **长程交互复杂性**: 智能体需要与环境进行多轮交互，每轮决策都依赖于历史上下文
- **环境反馈的随机性**: 外部工具（搜索引擎、代码解释器）返回的结果质量不可控
- **错误累积效应**: 早期微小错误会在后续轮次中被不断放大

### 1.2 恶性循环的发现

ProCeedRL的核心洞察在于识别了智能体探索中的**"恶性循环"（Vicious Circle）**现象：

```
次优动作 → 噪声观测 → 误导性上下文 → 进一步劣化的决策
    ↑                                              ↓
    └──────────────────────────────────────────────┘
```

具体而言：
1. **次优动作**（如模糊的搜索查询）导致环境返回**无关或误导性反馈**
2. 这些噪声观测被加入到上下文中，形成**"中毒"的推理环境**
3. 被污染的上下文进一步**削弱后续决策质量**
4. 形成正反馈循环，性能**持续下降且难以恢复**

实验验证（图1）：在搜索增强问答任务中，使用本地密集检索器（噪声更高）替代商业搜索引擎后：
- Qwen3-8B（较弱模型）性能下降**更为显著**
- 证明了推理能力受限会**放大噪声观测的负面影响**

---

## 二、ProCeedRL方法详解

### 2.1 核心设计理念

传统RLVR采用**被动探索策略**：独立重复采样，从成功样本中学习。这种方式在单轮任务中有效，但在多轮智能体任务中存在根本缺陷——一旦进入恶性循环，后续所有探索都是无效的。

ProCeedRL的核心创新在于**主动干预**：
- 不再等待轨迹完成后筛选成功样本
- 而是在交互过程中**实时检测并纠正错误步骤**
- 通过**过程级Critic**监控每步质量，必要时"回滚"并重做

### 2.2 技术架构

#### 2.2.1 过程级Critic（Process-Level Critic）

Critic $\phi$ 在每个时间步 $t$ 对交互进行评估：

$$l_t, c_t = \phi(\tau_t, a_t, s_{t+1})$$

其中：
- $l_t \in \mathbb{Z}$: 整数评分（0-10分）
- $c_t$: 文本形式的批评意见
- $\tau_t$: 历史轨迹
- $a_t$: 当前动作
- $s_{t+1}$: 环境反馈

**关键设计**：Critic不仅评估动作的合理性，还考虑其实际效果（$s_{t+1}$），实现**结果导向**的过程监督。

当评分低于阈值 $l_t \leq l_{th}$ 时，触发干预机制。

#### 2.2.2 精细化探索演示（Refined Exploratory Demonstration）

检测到次优步骤后，ProCeedRL不直接丢弃轨迹，而是**原地修正**：

$$a'_t = \mu(\tau_{t-1}, a_t, l_t, c_t)$$

修正策略 $\mu$ 基于：
- 历史上下文（排除当前次优动作的影响）
- Critic的具体反馈
- 生成更优的替代动作

这种设计实现了**探索与利用的动态平衡**：
- 保留探索的多样性（不同修正方式）
- 确保每一步都保持高质量
- 避免错误在上下文中累积

#### 2.2.3 训练流程

ProCeedRL的训练扩展了标准的**组相对策略优化（GRPO）**框架：

**数据收集阶段**（图3）：
1. **ProCeed Rollout**: 一半样本使用Critic干预生成
2. **直接采样**: 另一半保持原始策略的on-policy样本
3. 两者混合形成训练组，确保**分布对齐**和**对比学习信号**

**策略优化阶段**：

$$\mathcal{L} = \mathbb{E}_{x \sim \mathcal{B}} \frac{1}{|G|} \sum_{i=1}^{|G|} \sum_{t=1}^{|d_i|} \sigma(d_{i,t}) \hat{A}_{i,t} \min\left(\text{clip}(\text{is}_{i,t}(\theta), 1-\epsilon_{low}, 1+\epsilon_{high})\right)$$

其中 $\sigma(d_{j,t}) = \pi_\theta(d_{j,t}|x, d_{j,<t}) \cdot (1-\pi_\theta(d_{j,t}|x, d_{j,<t}))$ 是演示步骤的加权系数，采用**chord-φ方法**处理分布偏移问题。

**关键细节**：失败的演示步骤在优化中被**遮罩（mask）**，避免降低其概率导致训练不稳定。

### 2.3 与相关工作对比

| 方法类型 | 代表工作 | 核心差异 |
|---------|---------|---------|
| 过程奖励模型 | Lightman et al., 2024; Math-Shepherd | 需要昂贵的人工标注或精心设计的奖励函数 |
| 自纠错管道 | Li et al., 2025; Fu et al., 2025 | 测试时依赖额外框架，ProCeedRL将知识内化到模型 |
| 原子化奖励 | Deng et al., 2025; Atom-Searcher | 基于文档提取或细粒度动作的奖励，需要领域知识 |
| **ProCeedRL** | **本工作** | **无需外部知识，自我监督的过程Critic + 演示内化** |

---

## 三、实验设计与结果分析

### 3.1 评估任务

#### 3.1.1 深度搜索问答（Deep Search QA）

基于HotpotQA构建训练集（4000条平衡样本），在以下基准测试：
- **MuSiQue**: 多跳复杂推理问答
- **WebWalkerQA**: 网页遍历问答
- **GAIA**: 通用AI助手评估
- **Frames**: 事实性多跳问答
- **Bamboogle**: 挑战性多跳推理

**设置**：使用You.com搜索引擎，Top-3结果作为观测。

#### 3.1.2 具身任务（Embodied Task）

**ALFWorld**: 家庭环境中的长程规划任务
- 3553个训练场景
- 评估in-distribution和out-of-distribution性能

### 3.2 主要结果

#### 3.2.1 深度搜索QA（表1）

| 方法 | MuSiQue | WebWalkerQA | GAIA | Frames | Bamboogle | 平均 |
|------|---------|-------------|------|--------|-----------|------|
| SFT-only | 15.2 | 22.1 | 8.5 | 35.4 | 28.7 | 22.0 |
| RFT | 17.8 | 24.3 | 9.2 | 37.1 | 30.2 | 23.7 |
| GRPO | 18.5 | 25.6 | 10.1 | 38.9 | 31.5 | 24.9 |
| DAPO | 19.8 | 26.8 | 10.8 | 40.2 | 32.8 | 26.1 |
| **ProCeedRL** | **23.4** | **28.5** | **12.3** | **42.7** | **34.9** | **28.4** |
| vs. DAPO | +3.6 | +1.7 | +1.5 | +2.5 | +2.1 | **+2.3** |

**关键发现**：
- ProCeedRL在所有基准上均优于标准RL方法
- 在复杂任务（MuSiQue）上提升最为显著（+3.6%）
- 超越**专家轨迹蒸馏**（RFT），证明主动探索优于被动知识迁移

#### 3.2.2 具身任务（表2）

| 方法 | ALFWorld In-Dist | ALFWorld Out-of-Dist |
|------|------------------|----------------------|
| SFT-only | 45.3 | 38.7 |
| GRPO | 52.1 | 44.2 |
| DAPO | 55.8 | 47.3 |
| ProCeedRL (SFT) | **68.2** | **59.4** |

**突破**：在ALFWorld上实现**超过10%的绝对提升**，展示了ProCeedRL在长程交互任务中的强大能力。

### 3.3 消融实验与机制分析

#### 3.3.1 探索效率对比（图4）

ProCeedRL相比独立重复采样：
- **收敛速度更快**：达到相同性能所需样本数减少约40%
- **性能上限更高**：突破基础模型的探索饱和点
- **稳定性更好**：训练过程中奖励方差显著降低

#### 3.3.2 Critic设计分析

| Critic配置 | MuSiQue | WebWalkerQA |
|-----------|---------|-------------|
| 无Critic（标准RL） | 19.8 | 26.8 |
| 仅评分（无文本批评） | 21.2 | 27.5 |
| 评分+批评（ProCeedRL） | **23.4** | **28.5** |

文本批评提供了**细粒度的指导信号**，显著提升修正质量。

#### 3.3.3 阈值敏感性

动态阈值 $l_{th}$ 的校准：
- 过低阈值：干预不足，恶性循环持续
- 过高阈值：过度干预，探索多样性丧失
- ProCeedRL采用**任务自适应**的动态校准策略

---

## 四、深度理解：为什么ProCeedRL有效？

### 4.1 错误累积的数学直觉

在多轮智能体任务中，设第 $t$ 步的错误概率为 $p_t$，上下文噪声对后续步骤的影响因子为 $\alpha \in (0,1)$：

$$p_{t+1} = p_t + \alpha \cdot \mathbb{I}(\text{noise at } t) + \epsilon_t$$

在标准RL中，一旦 $p_t$ 超过某个阈值，后续步骤将陷入**几乎必然错误**的境地。ProCeedRL通过主动干预，将 $p_t$ 重置为接近初始状态，打破指数级增长的错误累积。

### 4.2 探索-利用的新范式

传统RLVR的探索是**选择性的**：
- 生成多条轨迹
- 选择成功的进行学习

ProCeedRL的探索是**生成性的**：
- 在轨迹内部动态改进
- 将失败轨迹转化为高质量训练数据

这类似于人类学习中的**"试错-反思-再试"**循环，而非简单的"试错-挑选"。

### 4.3 与测试时扩展（Test-Time Scaling）的关系

ProCeedRL揭示了一个重要洞见：

> **训练时的主动干预等价于测试时的额外计算**

具体而言：
- 标准RLVR需要 $N$ 条独立采样才能获得一条成功轨迹
- ProCeedRL通过中途修正，相当于在训练时执行了"内部搜索"
- 这种能力被内化到模型中，**测试时无需额外Critic**

这与o1类模型的测试时推理有本质区别：ProCeedRL将推理能力**蒸馏到模型参数**中，而非依赖测试时的显式搜索。

---

## 五、局限性与未来方向

### 5.1 计算开销

ProCeedRL的额外成本：
- 每步Critic评估：约1倍生成成本
- 修正动作生成：约1倍生成成本
- **总成本约为标准RL的2倍**

但作者指出：
> "鉴于探索效率的提升和性能上限的突破，这一额外成本是合理的"

### 5.2 改进保证的缺失

当前方法缺乏理论保证：
- Critic的评估可能存在偏差
- 修正动作不一定优于原动作
- 层级Critic架构可能有助于缓解

### 5.3 可扩展性挑战

- **更复杂环境**: 需要更强的Critic能力
- **更长期任务**: 错误检测的时间范围需要扩展
- **多智能体场景**: 需要协调多个Critic视角

### 5.4 未来研究方向

1. **自适应Critic**: 根据任务难度动态调整Critic强度
2. **元学习Critic**: 学习如何更好地评估和修正
3. **多模态扩展**: 将ProCeedRL应用于视觉-语言智能体
4. **理论分析**: 建立收敛性和最优性保证

---

## 六、行业影响与实践启示

### 6.1 对Agent开发的启示

ProCeedRL为构建**更可靠、更智能**的LLM Agent提供了新范式：

**短期实践**：
- 在现有RLVR框架中集成过程Critic
- 使用模型自身作为Critic（无需外部监督）
- 在搜索、代码生成等工具使用场景中优先应用

**中长期规划**：
- 开发轻量级Critic架构降低计算成本
- 构建多层级Critic系统处理复杂任务
- 探索Critic与策略模型的联合优化

### 6.2 与产业趋势的对齐

ProCeedRL与当前AI Agent产业趋势高度契合：

| 趋势 | ProCeedRL的对应能力 |
|------|-------------------|
| Deep Research（深度研究） | 多轮搜索中的错误恢复能力 |
| Code Agent（代码智能体） | 工具调用失败的自我修正 |
| Multi-Agent系统 | 过程监督的可扩展框架 |
| 边缘设备部署 | 训练时计算换推理时效率 |

### 6.3 竞争格局影响

- **开源社区**: 提供了不依赖昂贵标注的过程监督方法
- **闭源厂商**: 可能将类似机制集成到o3、Claude 4等下一代模型
- **学术界**: 开辟"主动探索"RL的新研究方向

---

## 七、核心结论

ProCeedRL代表了**LLM Agent强化学习**的重要进展：

1. **问题识别**: 首次系统阐述了智能体探索中的"恶性循环"现象
2. **方法创新**: 提出过程级Critic + 探索性演示的主动干预框架
3. **性能突破**: 在深度搜索和具身任务上显著超越标准RLVR
4. **实用价值**: 无需外部监督，可扩展的自包含训练流程

**最终评价**：ProCeedRL不仅是算法层面的改进，更代表了一种**从"选择探索"到"生成探索"**的范式转变。这种主动、干预式的学习范式，可能是通向更可靠、更通用AI Agent的关键一步。

---

## 参考资料

1. Gao, J., Guo, Y., Chen, X., & Chen, J. (2026). ProCeedRL: Process Critic with Exploratory Demonstration Reinforcement Learning for LLM Agentic Reasoning. *arXiv preprint arXiv:2604.02006*.

2. DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *arXiv preprint arXiv:2501.12948*.

3. Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv preprint arXiv:2402.03300*.

4. Lightman, H., et al. (2024). Let's Verify Step by Step. *arXiv preprint arXiv:2305.20050*.

5. Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.

6. Yu, Y., et al. (2025). DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization. *arXiv preprint*.

7. Wang, Z., et al. (2025). RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning. *arXiv preprint arXiv:2504.20073*.

8. Feng, J., et al. (2025). ReTool: Reinforcement Learning for Strategic Tool Use in LLMs. *arXiv preprint arXiv:2504.11536*.

---

*本文分析基于arXiv预印本v1版本，发表于2026年4月2日。如需获取最新更新，请访问论文官方页面。*
