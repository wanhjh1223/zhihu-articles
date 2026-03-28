# DAPO: 大规模语言模型强化学习的开源突破

## 论文概述

**论文标题**: DAPO: An Open-Source LLM Reinforcement Learning System at Scale  
**作者**: Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, Mingxuan Wang  
**机构**: ByteDance Seed, 清华大学智能产业研究院(AIR), 香港大学, SIA-Lab of Tsinghua AIR and ByteDance Seed  
**发表时间**: 2025年3月18日  
**论文链接**: https://arxiv.org/abs/2503.14476  
**代码开源**: https://github.com/volcengine/verl  
**项目主页**: https://dapo-sia.github.io/

---

## 一、研究背景与核心问题

### 1.1 推理能力的范式转变

2024年至2025年，大语言模型领域经历了一场深刻的范式转变——从单纯追求模型参数规模的扩展，转向**测试时计算扩展（Test-Time Scaling）**。OpenAI的o1系列和DeepSeek的R1系列模型证明，通过让模型在推理时进行更长时间的"思考"——即生成长链条思维（Long Chain-of-Thought, Long-CoT）——可以显著提升模型在复杂数学推理、编程竞赛等任务上的表现。

这种范式转变的核心技术是**大规模强化学习（Large-Scale Reinforcement Learning, RL）**。强化学习被证明是激发模型复杂推理行为的关键手段，包括自我验证（self-verification）、迭代优化（iterative refinement）和反思（reflection）等高级认知能力。

### 1.2 社区面临的核心困境

尽管测试时扩展展现了巨大的潜力，但研究社区面临着一个严峻的挑战：**关键的技术细节被隐藏**。无论是OpenAI o1的博客文章，还是DeepSeek R1的技术报告，都没有披露完整的训练细节。这种"黑箱"做法导致：

1. **复现困难**: 社区难以复现DeepSeek报告的47分AIME成绩，许多研究团队在尝试复现时只能达到30分左右
2. **技术瓶颈不明**: 训练过程中的熵崩溃（entropy collapse）、奖励噪声（reward noise）、训练不稳定等问题缺乏系统性的解决方案
3. **研究进展受阻**: 没有完整的开源实现，研究人员难以在此基础上进行进一步创新

### 1.3 DAPO的突破性贡献

DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization，解耦裁剪和动态采样策略优化）正是为解决上述问题而生。该论文的主要贡献包括：

1. **算法创新**: 提出DAPO算法，包含四项关键技术，解决大规模RL训练中的核心难题
2. **性能突破**: 在Qwen2.5-32B基础模型上达到AIME 2024 **50分**的成绩，超越DeepSeek-R1-Zero-Qwen-32B的47分，且仅使用**50%的训练步数**
3. **完全开源**: 开源完整的训练代码、数据集和训练日志，确保可复现性
4. **系统性分析**: 详细分析训练过程中的关键指标，为社区提供宝贵的实践经验

---

## 二、技术方法详解

### 2.1 基础：从PPO到GRPO

在深入DAPO之前，我们需要理解其算法基础。

#### 2.1.1 PPO（Proximal Policy Optimization）

PPO是强化学习中的经典算法，通过裁剪的重要性采样比率来限制策略更新幅度：

$$J^{PPO}(\theta) = \mathbb{E}_{(q,a)\sim D, o_{<t}\sim \pi_{\theta_{old}}}\left[ \min\left( r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(o_t|q,o_{<t})}{\pi_{\theta_{old}}(o_t|q,o_{<t})}$ 是重要性采样比率，$\hat{A}_t$ 是优势函数估计，$\epsilon$ 是裁剪范围（通常设为0.2）。

#### 2.1.2 GRPO（Group Relative Policy Optimization）

GRPO是DeepSeek提出的改进版本，**移除了价值函数**，改为在组内相对估计优势：

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}$$

对于每个问题，GRPO采样 $G$ 个回答，然后通过组内奖励归一化计算每个回答的相对优势。这种方法无需训练额外的价值模型，更适合语言生成任务。

### 2.2 DAPO的核心创新

DAPO在GRPO基础上进行四项关键改进：

#### 2.2.1 Clip-Higher：解耦裁剪范围提升探索

**问题分析**: 在标准PPO/GRPO中，对称的裁剪范围（$1-\epsilon$ 和 $1+\epsilon$）会限制低概率token的概率提升空间。例如，当 $\epsilon=0.2$ 时：
- 对于概率为0.9的高频token，最大可提升到 $0.9 \times 1.2 = 1.08$（实际为1.0）
- 对于概率为0.01的低频token，最大只能提升到 $0.01 \times 1.2 = 0.012$

这种不对称性导致模型难以探索低概率但可能更有价值的token，最终导致**熵崩溃**——模型输出趋于单一，缺乏多样性。

**解决方案**: DAPO提出**解耦裁剪（Decoupled Clip）**策略：

$$J^{DAPO}(\theta) = \mathbb{E}\left[ \frac{1}{\sum|o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\epsilon_{low}, 1+\epsilon_{high})\hat{A}_{i,t} \right) \right]$$

其中：
- $\epsilon_{low}$ 保持较小（如0.2），防止高频token概率过度下降
- $\epsilon_{high}$ 设置较大，给低频token更多的概率提升空间

**实验效果**: 图2显示，应用Clip-Higher后，模型熵值保持稳定，避免了熵崩溃现象，同时AIME准确率持续提升。

#### 2.2.2 Dynamic Sampling：动态采样提升训练效率

**问题分析**: 在RL训练中，某些问题的所有采样回答可能全部正确（奖励=1）或全部错误（奖励=-1）。此时组内优势计算为：

$$\hat{A}_{i,t} = \frac{1 - 1}{0} = 0 \text{ 或 } \frac{-1 - (-1)}{0} = 0$$

零优势意味着**零梯度**，这些样本无法对模型更新做出贡献，造成训练效率低下。

**解决方案**: DAPO引入**动态采样**策略：

$$\text{s.t. } 0 < |\{o_i | \text{is\_equivalent}(a, o_i)\}| < G$$

具体做法是：
1. 对每个问题持续采样，直到获得至少一个正确答案和一个错误答案
2. 过滤掉全对或全错的"无效组"
3. 确保批次中所有样本都能提供有效梯度

**实验效果**: 图3b显示，随着训练进行，准确率=1的样本比例持续上升。动态采样确保了有效样本的充分利用，使训练更加稳定高效。

#### 2.2.3 Token-Level Policy Gradient Loss：Token级策略梯度损失

**问题分析**: GRPO使用**样本级损失计算**：先对每个样本内的token损失求平均，再对所有样本的损失求平均。这种做法在Long-CoT场景中存在问题：

1. **长序列贡献被稀释**: 长回答中的每个token对总损失的贡献被其长度平均，导致高质量长序列的学习信号不足
2. **低质量长序列惩罚不足**: 包含重复、无意义内容的过长样本，其不良模式无法在样本级平均中被有效惩罚

**解决方案**: DAPO改用**Token级策略梯度损失**：

$$J^{DAPO}(\theta) = \mathbb{E}\left[ \frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \min(\cdots) \right]$$

关键变化：分母从 $G$（样本数）变为 $\sum|o_i|$（总token数）。这样：
- 长序列对总损失的贡献与其长度成正比
- 每个token的更新信号更加均衡
- 低质量长序列中的不良模式可以被更有效地惩罚

**实验效果**: 图4显示，Token级损失有效控制了响应长度的健康增长，同时维持了熵的稳定。

#### 2.2.4 Overlong Reward Shaping：过长样本的软惩罚

**问题分析**: RL训练通常设置最大输出长度（如16384或20480 token），超长样本会被截断。传统做法是对截断样本施加固定惩罚（如-1），但这会引入**奖励噪声**——有些样本可能包含正确的推理过程，仅仅因为长度限制被截断，却被给予负奖励。

**解决方案**: DAPO提出**Soft Overlong Punishment（软过长惩罚）**：

$$R_{length}(o) = \begin{cases} 
0 & \text{if } |o| \leq L_{safe} \\
-\alpha \cdot \frac{|o| - L_{safe}}{L_{max} - L_{safe}} & \text{if } L_{safe} < |o| \leq L_{max}
\end{cases}$$

其中：
- $L_{safe}$ 是安全长度阈值（如16384）
- $L_{max}$ 是最大长度（如20480）
- $\alpha$ 是惩罚系数

这种渐进式惩罚相比固定惩罚-1更加平滑，减少了奖励噪声。

### 2.3 算法整体流程

DAPO的完整训练流程如下：

```
算法1: DAPO训练流程
输入: 预训练模型 π_θ, 数据集 D, 组大小 G
输出: 优化后的模型 π_θ

1. for iteration = 1, 2, ... do
2.     从D中采样一批问题{(q, a)}
3.     for each question (q, a) do
4.         重复采样直到满足: 0 < |正确回答| < G
5.         生成G个回答 {o_i}_{i=1}^G
6.         计算每个回答的奖励 R_i
7.         对超长回答应用软惩罚
8.     end for
9.     计算Token级策略梯度损失
10.    使用解耦裁剪更新策略: ε_low, ε_high
11.    更新模型参数 θ
12. end for
```

### 2.4 其他关键设计

#### 2.4.1 移除KL散度惩罚

在RLHF中，KL散度惩罚用于防止策略偏离参考模型太远。但在Long-CoT推理训练中，模型分布可以且应该大幅偏离初始模型。DAPO**完全移除KL项**，给予模型更大的学习自由度。

#### 2.4.2 基于规则的奖励模型

为避免奖励黑客（reward hacking）问题，DAPO使用简单的规则计算奖励：

$$R(\hat{y}, y) = \begin{cases} 
1 & \text{if is\_equivalent}(\hat{y}, y) \\
-1 & \text{otherwise}
\end{cases}$$

这种基于准确度的二元奖励简单、无偏、可验证，适用于数学、代码等有标准答案的任务。

---

## 三、实验结果与性能分析

### 3.1 主要实验设置

- **基础模型**: Qwen2.5-32B
- **训练数据**: DAPO-Math-17K（从AoPS网站和官方竞赛页面收集的17K数学问题）
- **评估基准**: AIME 2024（美国数学邀请赛）
- **训练框架**: 基于verl框架
- **硬件配置**: 多节点GPU集群

### 3.2 核心性能结果

DAPO在AIME 2024上取得了突破性成绩：

| 方法 | 模型 | AIME 2024得分 | 训练步数 |
|------|------|--------------|----------|
| DeepSeek-R1-Zero-Qwen-32B | Qwen-32B | 47 | 100% |
| **DAPO** | **Qwen2.5-32B** | **50** | **50%** |

**关键发现**:
1. **绝对性能领先**: DAPO达到50分，超越DeepSeek-R1-Zero 3分
2. **效率领先**: 仅用50%的训练步数达到更优性能
3. **模型无关**: 使用更新的Qwen2.5-32B作为基础，但算法创新是性能提升的核心

### 3.3 训练动态分析

#### 3.3.1 响应长度增长

DAPO训练过程中，模型输出长度呈现**稳定且健康**的增长趋势：
- 初始阶段：响应长度较短，模型刚开始学习推理
- 中期阶段：长度逐渐增加，模型发展出更详细的推理链
- 后期阶段：长度趋于稳定，模型学会在效率和准确性间平衡

这种增长模式与DeepSeek-R1报告中描述的"顿悟时刻"（Aha Moment）一致——模型自主发展出反思、验证等复杂推理行为。

#### 3.3.2 奖励动态

奖励曲线显示：
- 平均奖励随训练稳步提升
- 奖励方差逐渐减小，表明训练趋于稳定
- 动态采样确保了梯度的有效性

#### 3.3.3 熵与生成概率

- **初始阶段**: 熵值较高，模型探索充分
- **中期阶段**: 熵值下降，Clip-Higher策略防止熵崩溃
- **后期阶段**: 熵值稳定在合理范围，保持适度的探索能力

### 3.4 消融实验

论文通过系统的消融实验验证了各项技术的有效性：

| 配置 | AIME 2024得分 | 说明 |
|------|--------------|------|
| 完整DAPO | 50 | 所有技术组合 |
| w/o Clip-Higher | ~42 | 熵崩溃，多样性不足 |
| w/o Dynamic Sampling | ~44 | 梯度效率低，训练慢 |
| w/o Token-Level Loss | ~45 | 长序列学习不足 |
| w/o Overlong Shaping | ~46 | 奖励噪声影响稳定性 |
| GRPO基线 | ~30 | 原始算法性能有限 |

---

## 四、开源贡献与社区影响

### 4.1 开源内容

DAPO团队发布了完整的开源组件：

1. **训练代码**: 基于verl框架的完整实现
2. **数据集**: DAPO-Math-17K，17K精心整理的数学竞赛题
3. **训练日志**: 完整的WandB训练记录
4. **模型检查点**: 达到50+分的关键检查点

### 4.2 可复现性保障

论文详细披露了所有关键超参数：

- **组大小（Group Size）**: 16
- **裁剪范围**: $\epsilon_{low}=0.20$, $\epsilon_{high}=0.28$
- **学习率**: 1e-6
- **批次大小**: 128
- **最大长度**: 20480 tokens
- **安全长度**: 16384 tokens
- **长度惩罚系数**: 0.1

这种透明性确保了其他研究团队可以完全复现结果。

### 4.3 对研究社区的意义

1. **降低研究门槛**: 提供了一个可工作的强基线，研究人员可以在此基础上进行改进
2. **揭示训练细节**: 填补了DeepSeek等技术报告未披露的关键细节
3. **推动标准化**: 开源代码和数据集可能成为RL推理研究的标准测试平台

---

## 五、深入分析与个人见解

### 5.1 技术深度剖析

#### 5.1.1 为什么Clip-Higher如此关键？

Clip-Higher的设计揭示了RL在语言模型中的一个深层问题：**探索与利用的平衡**。传统的PPO设计假设策略更新应该被对称地限制，但这忽略了语言生成中token分布的长尾特性。

高频token（如常见词汇）即使小幅概率提升也可能对输出产生重大影响，而低频token（如专业术语、数学符号）需要更大的概率提升才能被有效探索。Clip-Higher通过不对称裁剪，给予低频token更多的"成长空间"，这对于激发模型的推理多样性至关重要。

#### 5.1.2 Token级损失的本质

Token级策略梯度损失可以被视为一种**注意力重新分配机制**。在样本级平均中，模型对长短序列的关注度是均等的，这可能不适合Long-CoT场景——长序列通常包含更复杂的推理步骤，值得更多的学习权重。

此外，Token级损失隐含地实现了**长度感知正则化**：低质量的长序列（如包含重复内容）会产生更多token，其不良影响被放大，从而被更有效地抑制。

#### 5.1.3 动态采样的统计意义

从统计学习理论角度，动态采样确保了每个训练批次都具有**非零的梯度方差**。在梯度下降优化中，零梯度意味着该批次对参数更新无贡献，相当于浪费了计算资源。动态采样通过过滤"无效组"，提高了数据效率和收敛速度。

### 5.2 局限性与未来方向

#### 5.2.1 当前局限

1. **任务限制**: DAPO主要在数学推理上验证，其在代码生成、定理证明、科学推理等任务上的有效性需进一步验证
2. **奖励设计**: 基于规则的奖励仅适用于有标准答案的任务，开放性任务需要更复杂的奖励设计
3. **计算成本**: 大规模RL训练仍需要大量计算资源，对小型研究团队构成门槛

#### 5.2.2 未来研究方向

1. **多模态扩展**: 将DAPO应用于视觉-语言推理任务
2. **过程奖励**: 结合过程级奖励（process reward）进一步提升训练效率
3. **自适应长度**: 让模型自动学习最优的推理长度，而非固定最大长度
4. **理论分析**: 深入研究DAPO各项技术的理论收敛性质

### 5.3 产业影响分析

#### 5.3.1 对AI基础设施的影响

DAPO的开源将推动以下产业发展：
- **RL训练平台**: 基于DAPO实现的企业级RL训练服务
- **领域定制**: 垂直领域（金融、医疗、法律）的推理模型定制服务
- **教育应用**: 数学辅导、编程教学等教育场景的AI助手

#### 5.3.2 竞争格局变化

DAPO的发布表明，中国AI公司在RL推理领域已达到世界领先水平。ByteDance与清华的合作模式——产业界提供计算资源和工程能力，学术界提供研究洞察——可能成为未来AI大模型研发的新范式。

#### 5.3.3 开源vs闭源的博弈

DAPO选择完全开源，与OpenAI的闭源策略形成鲜明对比。这种"开放领先"策略可能迫使更多公司开源其技术细节，从而加速整个领域的发展。长期来看，开源生态的繁荣将降低AI技术的应用门槛，促进普惠AI的实现。

---

## 六、总结与展望

### 6.1 核心贡献回顾

DAPO论文做出了以下关键贡献：

1. **算法创新**: 提出Clip-Higher、Dynamic Sampling、Token-Level Loss、Overlong Reward Shaping四项关键技术，系统性地解决了大规模RL训练的核心难题
2. **性能突破**: 在AIME 2024上达到50分，仅用50%训练步数超越DeepSeek-R1-Zero
3. **开源生态**: 完全开源算法、代码、数据和训练日志，推动领域标准化
4. **实践经验**: 提供详细的训练动态分析，为社区提供宝贵参考

### 6.2 对LLM推理研究的启示

DAPO的成功揭示了LLM推理研究的几个重要方向：

1. **算法细节至关重要**: 训练细节的微小差异可能导致性能的显著差距，透明和可复现性是科研的基础
2. **开源胜过闭源**: 在快速发展的新兴领域，开源策略能够汇聚更多研究者的智慧，加速技术进步
3. **工程与研究并重**: DAPO的成功不仅是算法的胜利，也体现了强大的工程实现能力

### 6.3 未来展望

随着DAPO等开源工作的推动，我们有理由期待：

1. **更强的推理模型**: 基于DAPO框架，未来的开源模型可能在数学、代码、科学推理等任务上接近甚至超越闭源模型
2. **更低的研究门槛**: 标准化的开源工具和流程将让更多研究者和开发者参与到RL推理研究中
3. **更广泛的应用场景**: 推理能力的提升将解锁AI在科学研究、软件开发、教育辅导等领域的更多应用

DAPO不仅是一篇论文，更是LLM推理研究领域的一个重要里程碑。它证明了开源社区的力量，也为未来的研究者照亮了前行的道路。在这个测试时计算扩展的新时代，DAPO为我们展示了通往更智能AI系统的可行路径。

---

**参考文献**

1. Yu, Q., et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv preprint arXiv:2503.14476.
2. Guo, D., et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2501.12948.
3. OpenAI. (2024). Learning to Reason with LLMs.
4. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.
5. Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300.

---

*本文完成于2026年3月28日，基于DAPO论文v1版本（arXiv:2503.14476v1）撰写。*
