# 门控注意力机制：大语言模型架构的范式革新

> **论文标题**: Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free  
> **作者**: Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin  
> **机构**: 阿里巴巴通义千问（Qwen）团队  
> **发表**: NeurIPS 2025 Best Paper Award  
> **论文链接**: https://arxiv.org/abs/2505.06708  
> **发布时间**: 2025年5月

---

## 一、引言：一个简单修改引发的架构革命

在深度学习领域，偶尔会出现这样一种研究：它不需要颠覆性的理论创新，也不依赖海量的计算资源堆砌，而是通过一个简洁而精妙的架构修改，就能在性能、稳定性和效率三个维度上同时取得突破。阿里巴巴Qwen团队发表于NeurIPS 2025并获得最佳论文奖的《Gated Attention for Large Language Models》正是这样一项研究。

这项研究的核心贡献可以概括为：**在标准缩放点积注意力（SDPA）之后添加一个逐头的Sigmoid门控函数**。这个看似简单的修改——仅增加不到1%的参数量——却带来了训练稳定性的显著提升、学习率的宽容度增加、以及模型scaling特性的改善。更重要的是，它有效缓解了困扰Transformer架构已久的"注意力汇聚"（Attention Sink）问题，并显著增强了长上下文外推能力。

在NeurIPS 2025收到的21,575篇投稿中，这篇论文脱颖而出成为四篇最佳论文之一，评委会给出的评价是："这项研究代表了一项只有借助工业级计算资源才能完成的大量工作，作者分享工作成果的行为将推动整个社区对大语言模型注意力机制的理解，尤其是在大型语言模型相关科学成果开放共享日渐减少的环境下，这种行为尤为可贵。"

---

## 二、研究背景：注意力机制的隐疾

### 2.1 Transformer的皇冠明珠

自2017年Vaswani等人提出"Attention Is All You Need"以来，注意力机制已成为现代大语言模型的核心支柱。Transformer架构凭借其强大的并行计算能力和对长距离依赖的建模能力，彻底改变了自然语言处理领域的格局。从BERT到GPT系列，从T5到Llama，几乎所有主流的大语言模型都建立在Transformer的基础之上。

标准的多头注意力机制（Multi-Head Attention, MHA）计算流程如下：

```
Q = X · W_Q    # Query投影
K = X · W_K    # Key投影  
V = X · W_V    # Value投影
A = softmax(QK^T / √d_k)  # 注意力权重
Output = A · V   # 加权聚合
```

其中，softmax函数将Query-Key的点积分数转换为概率分布，决定了每个位置应该"关注"序列中哪些其他位置的信息。

### 2.2 Attention Sink：注意力机制的痼疾

然而，随着Transformer模型规模的不断扩大和应用场景的不断拓展，研究者们逐渐发现了一个令人困扰的现象：**Attention Sink（注意力汇聚）**。

Attention Sink最早由Xiao等人在2023年的StreamingLLM论文中系统描述。他们观察到，在处理长序列时，模型会将大量注意力权重集中在序列的起始token（通常是BOS或<s>标记）上，即使这些token在语义上与当前预测任务毫不相关。这种现象的直观表现是——模型就像是一个"注意力多动症"患者，无法合理分配注意力资源，将大量计算浪费在无意义的位置上。

从softmax函数的数学特性可以解释这一现象的根源：

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

由于softmax使用指数函数进行归一化，即使输入分数差异很小，输出概率分布也可能极度偏向某一维度。更重要的是，**softmax要求所有输出的概率之和为1**，这意味着即使当前query与所有key都不相关，模型也必须将概率质量分配到某些位置——而序列起始token由于在每个样本中都存在、能够参与完整序列的注意力计算、且梯度相对稳定，自然成为了注意力汇聚的"磁石"。

### 2.3 Attention Sink的连锁危害

Attention Sink不仅仅是一个理论上的好奇现象，它引发了一系列实际问题：

**训练不稳定性**：当注意力过度集中于少数token时，会导致注意力概率矩阵的Frobenius范数急剧增大，进而引发梯度爆炸和损失发散。这在大规模模型训练中尤为致命，可能导致数周的训练工作功亏一篑。

**长上下文退化**：在StreamingLLM的实验中，当序列长度超过训练时的上下文窗口时，模型性能会急剧下降，原因之一就是Attention Sink导致的注意力分配失衡。

**激活值异常**：Attention Sink往往伴随着"Massive Activations"现象——少数token位置的隐藏状态会产生异常大的幅值，这给模型量化和低精度推理带来了巨大挑战。

**信息流动受阻**：当注意力被"困"在起始token时，序列中其他位置之间的信息交互受到抑制，影响了模型对全局上下文的理解能力。

### 2.4 现有解决方案的局限

针对Attention Sink问题，学术界和工业界已经提出了多种解决方案：

**QK-LayerNorm/QKNorm**：在计算Query-Key点积前应用层归一化，控制注意力分数的尺度。

**Softmax-1**：修改softmax的分母为Σexp(x_j) + 1，允许注意力权重之和不为1。

**Sigmoid Attention**：用sigmoid函数替代softmax，消除归一化约束。

**StreamingLLM**：在KV Cache中始终保留起始token，缓解长上下文中的注意力塌陷。

然而，这些方法要么改变了注意力机制的基本特性（如使用sigmoid会丢失概率解释的直观性），要么需要额外的工程技巧（如StreamingLLM），要么效果有限。业界一直在寻找一个既能保留softmax优势、又能缓解其固有缺陷的优雅方案。

---

## 三、技术方法：门控注意力的设计哲学

### 3.1 门控机制：神经网络的"调节阀"

门控（Gating）是深度学习中的经典概念，从早期的LSTM、GRU到Highway Networks，门控机制被广泛用于控制信息流动。在状态空间模型（SSM）如Mamba系列中，门控更是扮演着核心角色。

一个门控单元的基本形式是：

```
Gate(x) = σ(W_g · x + b_g)   # σ通常为sigmoid
Output = Gate(x) ⊙ x         # ⊙表示逐元素乘法
```

门控就像一个"调节阀"，可以学习性地放大、衰减或阻断信息流。当Gate值接近1时，信息自由通过；接近0时，信息被抑制。

### 3.2 Gated Attention的架构设计

Qwen团队的核心创新在于：**将门控机制引入标准的softmax注意力，并系统性地研究了门控的最佳放置位置**。

标准Attention与Gated Attention的对比：

```python
# 标准Attention
A = softmax(QK^T / √d_k)  
Output = A · V

# Gated Attention（SDPA-output gating）
A = softmax(QK^T / √d_k)
G = σ(W_g · [A·V] + b_g)   # 门控计算
Output = G ⊙ (A · V)       # 门控调制
```

关键在于门控的位置选择。Qwen团队系统比较了多种放置方案：

| 门控位置 | 代号 | 效果 |
|---------|------|------|
| Q投影后 | G-Q | 较弱 |
| K投影后 | G-K | 较弱 |
| V投影后 | G-V | 中等 |
| SDPA输出后 | **G1** | **最强** |
| 最终输出后 | G-O | 中等 |

研究发现，**在SDPA输出后添加门控（G1方案）效果最佳**。这是因为该位置的门控能够：

1. **直接控制注意力块的输出**：对softmax注意力的结果进行精细化调节
2. **引入query-dependent稀疏性**：门控分数与当前query相关，实现自适应注意力分配
3. **有效抑制噪声交互**：过滤掉不重要的注意力输出

### 3.3 核心机制解析

#### 3.3.1 非线性增强

标准softmax注意力可以看作是低秩映射：Attention(Q,K,V) = softmax(QK^T)V。这个映射虽然强大，但在表达复杂函数关系时可能存在局限。

门控机制引入了一个额外的非线性变换：

```
Gate(h) = σ(W_g · h + b_g)
```

这个Sigmoid门控为注意力输出增加了逐元素的非线性调制能力，增强了模型的表达能力。论文通过对比实验发现，正是这种非线性特性带来了性能提升。

#### 3.3.2 稀疏性调制

Gated Attention产生的门控分数呈现出显著的**稀疏性特征**——大多数位置的Gate值接近0，只有少数关键位置的Gate值较大。这种稀疏性有以下几个优势：

1. **计算效率**：稀疏激活意味着实际参与后续计算的信息量减少
2. **噪声过滤**：低Gate值有效抑制了不重要的注意力输出
3. **可解释性**：稀疏模式更容易分析模型的关注焦点

论文中的可视化显示，SDPA-output gating的平均Gate值约为0.116，意味着近90%的注意力输出被抑制——这是一个非常稀疏的激活模式。

#### 3.3.3 Attention Sink的消除

这是Gated Attention最令人惊喜的特性。论文通过大量实验证实，Gated Attention能够有效缓解Attention Sink现象：

1. **门控作为后过滤器**：即使softmax将大量概率分配给起始token，门控可以根据query的实际需求选择性地抑制这些输出
2. **打破恶性循环**：消除了"起始token获得大梯度→参数更新偏向起始token→注意力进一步集中"的正反馈
3. **改善长上下文外推**：没有Attention Sink的干扰，模型在处理超出训练长度的序列时表现更加稳定

### 3.4 实现细节

Qwen团队提供了Gated Attention的简化PyTorch实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttention(nn.Module):
    """
    Gated Attention 简化实现
    基于NeurIPS 2025最佳论文的核心思想
    """
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        
        # 标准Q, K, V, O投影
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # 门控层 - 核心创新
        self.w_gate = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 标准投影
        q = self.w_q(x).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_head, self.d_head).transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        
        # 注意力输出
        attn_output = torch.matmul(attn, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # 门控调制 - 核心操作
        gate = torch.sigmoid(self.w_gate(attn_output))
        gated_output = gate * attn_output
        
        # 最终投影
        output = self.w_o(gated_output)
        return output
```

这个实现仅增加了`w_gate`一个线性层（约占总参数量的0.5-1%），却带来了显著的性能提升。

---

## 四、实验验证：规模化的实证研究

### 4.1 实验设置

Qwen团队进行了一项规模空前的对比研究，以确保结论的可靠性和普适性：

- **模型规模**：15B参数的MoE模型和1.7B参数的dense模型
- **训练数据**：3.5万亿token的高质量语料
- **对比变体**：超过30种不同的门控配置
- **评估维度**：性能、训练稳定性、学习率敏感性、scaling特性、长上下文能力

这种系统性的研究设计是论文获得NeurIPS最佳论文奖的重要原因——它提供了工业级规模的实证证据，而非仅仅是理论推导或小规模实验。

### 4.2 核心发现

#### 4.2.1 性能提升

Gated Attention在各项下游任务上 consistently 超越标准注意力：

| 模型配置 | 平均性能 | 相对提升 |
|---------|---------|---------|
| 标准Attention | 基准 | - |
| Gated Attention (SDPA-output) | **+2.1%** | ✓ |
| Gated Attention (Value-path) | +1.3% | ✓ |
| 其他门控位置 | +0.5~1.0% | △ |

值得注意的是，这种提升是在**不增加推理计算成本**（仅增加少量参数）的情况下实现的。

#### 4.2.2 训练稳定性

这是Gated Attention最令人惊喜的特性之一。论文通过系统性的学习率扫描实验发现：

- **标准Attention**：学习率敏感度高，过大学习率会导致训练发散
- **Gated Attention**：对学习率更加宽容，可以在更大范围内稳定训练

这种稳定性源于门控机制对梯度流的调节作用——当某些注意力头产生异常大的激活时，门控可以将其抑制，防止梯度爆炸。

#### 4.2.3 Scaling特性改善

在scaling law研究中，Gated Attention展现出更好的性能-规模曲线：

- **更低的loss floor**：在相同计算预算下达到更低的训练loss
- **更稳定的scaling**：性能随模型规模增长更加可预测
- **更高的数据效率**：在相同数据量下获得更好性能

这些特性对于大规模模型的训练规划具有重要意义。

#### 4.2.4 长上下文外推

Gated Attention在处理超出训练长度的序列时表现优异：

- **Attention Sink消除**：起始token不再"抢夺"过多注意力
- **更稳定的注意力分布**：注意力权重在序列中分布更加均匀
- **更好的外推性能**：在长度外推任务上显著优于标准Attention

实验显示，在32K→128K的上下文长度外推测试中，Gated Attention的困惑度（perplexity）增长明显更平缓。

### 4.3 消融研究

论文通过大量消融实验揭示了Gated Attention有效性的关键因素：

**Q1：门控位置为何重要？**
- SDPA-output位置的稀疏性最强（平均Gate值0.116 vs Value-path的0.342）
- 更强的query依赖性，实现自适应注意力分配
- 直接控制注意力块输出，调节效果最直接

**Q2：门控如何缓解Attention Sink？**
- 作为后注意力过滤器，选择性抑制高激活
- 打破"起始token优势→大梯度→更强优势"的恶性循环
- 允许模型学习"不关注"的能力

**Q3：非线性vs稀疏性哪个更重要？**
- 两者都有贡献，但稀疏性是关键
- 对比实验显示，稀疏门控（SDPA-output）优于稠密门控
- 稀疏性提供了噪声过滤和计算效率的双重优势

---

## 五、产业应用：从论文到产品

### 5.1 Qwen3-Next的架构创新

Gated Attention的研究成果已经直接应用于阿里巴巴最新发布的Qwen3-Next模型（2025年9月）。该模型采用了革命性的架构设计：

- **Gated DeltaNet**：将门控机制与DeltaNet结合
- **Gated Attention**：完全替代标准Attention
- **混合架构**：在不同层使用不同的注意力变体

这种设计带来的实际收益包括：

1. **上下文学习能力提升**：在in-context learning任务上表现显著改善
2. **计算效率优化**：稀疏激活降低了实际计算成本
3. **长对话稳定性**：在超长对话场景中保持连贯性
4. **训练成本降低**：更稳定的学习动态减少了训练失败和重跑

### 5.2 开源与社区贡献

Qwen团队遵循开放科学的精神，已经开源了相关代码和预训练模型：

- **GitHub仓库**：包含Gated Attention的完整实现
- **HuggingFace模型**：可供社区直接下载使用
- **技术报告**：详细的实验设置和分析

这种开放态度在大模型研究日渐封闭的背景下尤为珍贵，也是NeurIPS评委会高度认可的一点。

### 5.3 行业影响预测

评委会在颁奖词中预测："论文的核心建议易于实施，鉴于论文为这一LLM架构修改提供的广泛证据，我们预计这一想法将被广泛采用。"

这种预测基于以下几点：

1. **低实施成本**：仅需修改几行代码，不增加推理成本
2. **普适性**：适用于任何基于Transformer的模型
3. **显著收益**：在多个维度上都有明确改善
4. **可组合性**：可以与其他架构改进（如RoPE、SwiGLU等）结合使用

预计未来我们将在GPT-5、Gemini 2.0、Claude下一代等主流模型中看到Gated Attention或其变体的身影。

---

## 六、深度思考：为什么这个简单的修改如此有效？

### 6.1 对注意力本质的重新审视

Gated Attention的成功促使我们重新思考注意力机制的本质。传统观点认为softmax注意力已经提供了足够的表达能力，但Gated Attention表明：**注意力不仅仅是"关注哪里"，还包括"关注多少"**。

门控机制引入了第二个层次的调节：
- **第一层次（softmax）**：决定在序列中分配注意力的位置
- **第二层次（gate）**：决定整体允许多少注意力信息通过

这种双层调节赋予模型更精细的控制能力。

### 6.2 稀疏性的价值

Gated Attention的一个重要启示是：**稀疏性本身可能就是优势**。

传统观点倾向于认为模型需要充分利用所有可用的信息。但Gated Attention表明，学会"忽略"不重要的信息同样重要——甚至更重要。

这种稀疏性带来了多个好处：
- **计算效率**：实际计算量减少
- **泛化能力**：减少过拟合风险
- **可解释性**：更容易理解模型的决策依据
- **鲁棒性**：对噪声和干扰更加不敏感

### 6.3 Attention Sink的深层原因

Gated Attention的实验也为我们理解Attention Sink提供了新视角。论文证实了Attention Sink源于softmax的归一化约束——**模型需要一个"垃圾桶"来倾倒多余的注意力概率质量**。

门控机制提供了一种优雅的解决方案：允许模型在门控层面"关闭"不想要的注意力输出，而不需要在softmax层面进行复杂的概率重分配。

### 6.4 架构创新的新范式

Gated Attention代表了一种值得关注的架构创新范式：**基于系统实证研究的渐进式改进**。

与颠覆性的架构革命（如Transformer取代RNN）不同，Gated Attention展示了如何通过：
1. 深入理解现有机制的局限
2. 设计针对性的改进方案
3. 进行大规模系统性验证

来实现实质性的性能提升。这种"深入研究+工程验证"的模式可能是未来大模型架构演进的主流路径。

---

## 七、局限与未来方向

### 7.1 当前局限

尽管Gated Attention取得了显著成功，但仍有一些需要进一步研究的问题：

**理论理解的深度**：论文主要基于实证研究，对于为什么门控在特定位置最有效、稀疏性与性能的确切关系等问题的理论解释还不够深入。

**超参数敏感性**：门控机制引入了新的超参数（如门控层的初始化方式），如何最优地设置这些参数还需要更多研究。

**与其他架构的结合**：Gated Attention与MoE、多模态架构、状态空间模型等的最佳结合方式尚待探索。

### 7.2 未来研究方向

基于Gated Attention的成功，以下几个方向值得关注：

**动态门控**：根据输入动态调整门控策略，实现更灵活的注意力控制。

**多头门控差异化**：不同注意力头使用不同的门控策略，实现功能专业化。

**跨层门控协调**：研究不同层之间门控行为的协调性，优化信息流动。

**硬件友好型稀疏性**：利用门控产生的稀疏性进行硬件加速，进一步提升推理效率。

**可解释性研究**：利用门控的稀疏模式深入理解模型的注意力行为。

---

## 八、结论

《Gated Attention for Large Language Models》是一项典范性的深度学习研究。它不需要复杂的数学推导，不依赖海量的计算资源，而是通过一个简洁的架构修改和系统性的实证研究，为Transformer架构带来了实质性的改进。

这项研究的核心启示是：**有时候，问题不在于设计更复杂的解决方案，而在于找到更优雅的问题视角**。Attention Sink问题困扰学界多年，解决方案从修改softmax到添加辅助token不一而足。Gated Attention的独特洞见在于——与其改变softmax本身，不如在softmax之后添加一个"调节阀"，让模型自己学习如何最优地分配注意力资源。

从产业角度看，Gated Attention具有极高的实用价值：实施成本低（<1%参数增加）、推理开销小、训练收益大、普适性强。这些特性使其成为大模型架构改进的"低 hanging fruit"，预计将在短期内被业界广泛采纳。

更重要的是，这项研究展示了中国AI研究的力量。在全球大模型研究竞争日趋激烈的背景下，阿里巴巴Qwen团队凭借扎实的工程能力和开放的研究态度，在NeurIPS这一顶级会议上获得最佳论文奖，标志着中国团队在基础架构创新方面已达到世界领先水平。

随着Gated Attention被集成到越来越多的模型中，我们有理由期待：未来的大语言模型将更加高效、稳定、可解释。而这个变革的起点，只是一个简单的Sigmoid门控。

---

## 参考资源

- **论文原文**: https://arxiv.org/abs/2505.06708
- **开源代码**: https://github.com/QwenLM/Qwen
- **HuggingFace模型**: https://huggingface.co/Qwen
- **NeurIPS 2025官方公告**: https://nips.cc/virtual/2025/awards_detail
- **Qwen3-Next技术报告**: https://qwenlm.github.io/blog/qwen3-next/

---

*本文撰写于2026年3月27日，基于NeurIPS 2025最佳论文的公开资料和Qwen团队的官方技术报告。*
