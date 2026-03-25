# Byte Latent Transformer (BLT): 告别Tokenizer的字节级大模型新架构

> **论文标题**: Byte Latent Transformer: Patches Scale Better Than Tokens  
> **作者**: Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, Srinivasan Iyer  
> **机构**: Meta FAIR, University of Washington, University of Chicago  
> **发表**: ACL 2025 (Outstanding Paper Award)  
> **论文链接**: https://arxiv.org/abs/2412.09871  
> **代码仓库**: https://github.com/facebookresearch/blt

---

## 一、引言：Tokenization的终结者？

自2017年Transformer架构诞生以来，大语言模型（LLM）的发展几乎完全建立在**子词Tokenization**的基础之上。从BPE（Byte-Pair Encoding）到SentencePiece，Tokenizer已成为LLM pipeline中不可或缺的一环。然而，这一看似理所当然的预处理步骤，实际上带来了一系列深层次的问题：

- **OOV（Out-of-Vocabulary）问题**: 任何固定大小的词表都无法覆盖所有可能的字符组合
- **多语言不公平性**: 不同语言被分配到的token数量差异巨大，导致非英语语言能力受限
- **领域敏感性**: 代码、数学公式、特殊符号等需要专门设计的Tokenizer
- **输入噪声脆弱性**: 简单的字符大小写变化或拼写错误可能导致完全不同的token序列
- **计算效率瓶颈**: Tokenizer必须处理每一个字节，而模型本身只能处理离散的token

Meta FAIR团队在这一背景下提出了**Byte Latent Transformer (BLT)**，这是首个在规模上**完全匹敌甚至超越**基于Tokenizer的LLM性能的字节级架构。更重要的是，BLT获得了ACL 2025的**Outstanding Paper Award**（优秀论文奖），这标志着学术界对这一方向的重大认可。

---

## 二、核心问题与研究动机

### 2.1 Tokenization的本质困境

传统LLM的工作流程可以简化为：

```
Raw Bytes → Tokenizer → Token IDs → Embedding → Transformer → Token Logits → Detokenizer → Text
```

在这个流程中，Tokenizer扮演了一个"翻译官"的角色，将原始字节流转换为模型能够处理的离散符号。然而，这种转换是：

1. **有损的**: 字节信息在tokenization过程中丢失，模型永远无法直接访问原始字节
2. **静态的**: Tokenizer在训练前固定，无法根据数据动态调整
3. **启发式的**: BPE等算法基于频率统计，而非语义理解

论文作者尖锐地指出："**Tokenization has previously been essential because directly training LLMs on bytes is prohibitively costly at scale due to long sequence lengths.**"（Tokenizer之所以必不可少，是因为直接训练字节级LLM因序列过长而成本过高。）

### 2.2 计算效率的本质

那么，为什么字节级模型成本高昂？核心在于：

**Transformer的计算复杂度主要由FFN（前馈网络）层决定，而非Attention层。**

对于标准的Transformer，每个token都需要通过完整的FFN层，这占据了约2/3的总计算量。如果直接处理字节（每个token约3-4字节），序列长度将扩大3-4倍，计算成本将呈平方级增长。

**关键洞察**: 计算成本的关键不是序列长度本身，而是**每一步都需要运行大模型的次数**。

### 2.3 BLT的核心命题

BLT提出的解决方案可以概括为三个层次：

1. **动态分块（Dynamic Patching）**: 根据数据复杂度自适应地将字节分组为"Patches"
2. **计算资源动态分配**: 在简单、可预测的数据上节省计算，在复杂数据上增加计算
3. **字节级信息保留**: 通过轻量级的Encoder/Decoder架构，让全局模型既能处理Patches，又能访问原始字节信息

---

## 三、技术架构深度解析

### 3.1 整体架构概览

BLT采用了**三模块架构**：

```
Input Bytes → [Local Encoder] → Patch Representations → [Latent Global Transformer] → Output Patches → [Local Decoder] → Output Bytes
```

![BLT架构示意图](https://raw.githubusercontent.com/facebookresearch/blt/main/assets/architecture.png)

**核心设计哲学**:
- **Local Encoder**: 轻量级，将字节编码为Patch表示
- **Latent Global Transformer**: 重量级，处理Patch级别的自回归建模（消耗大部分计算）
- **Local Decoder**: 轻量级，将Patch表示解码为字节

这种设计使得昂贵的Global Transformer只需要在Patch级别运行，而Patch数量远少于字节数量。

### 3.2 动态分块：Entropy Patching

BLT最核心的创新是**基于熵的动态分块机制**。

#### 3.2.1 为什么用熵？

语言序列的不确定性分布是不均匀的：
- 预测一个词的**首字母**通常很难（高熵）
- 预测后续字母通常很容易（低熵）
- 空格后的第一个字符通常很关键（高熵）
- 重复模式的可预测性很高（低熵）

**核心洞察**: 在不确定性高的位置启动新Patch，让Global Transformer介入；在确定性高的区域使用长Patch，节省计算。

#### 3.2.2 熵模型实现

BLT使用一个**小型字节级语言模型**（约100M参数）计算每个字节位置的熵：

$$H(x_i) = \sum_{v \in \mathcal{V}} p_e(x_i = v | \boldsymbol{x}_{<i}) \log p_e(x_i = v | \boldsymbol{x}_{<i})$$

其中$p_e$是熵模型的输出分布，$\mathcal{V}$是字节词表（大小为256）。

#### 3.2.3 Patch边界判定

论文尝试了两种边界判定策略：

**全局阈值策略**:
$$H(x_t) > \theta_g$$

当某个位置的熵超过全局阈值时，启动新Patch。

**近似单调性策略**:
$$H(x_t) - H(x_{t-1}) > \theta_r$$

当熵相对于前一个位置显著上升时，启动新Patch。这种方法对"熵漂移"（Entropy Drift）更鲁棒。

![熵分块示例](https://raw.githubusercontent.com/facebookresearch/blt/main/assets/entropy_patching.png)

*图：熵分块可视化。红线为全局阈值，垂直灰线为Patch边界。注意"G"和"e"在"George R.R. Martin"中因高熵而成为独立Patch的起点。*

### 3.3 Local Encoder设计

Local Encoder的目标是将字节序列高效地编码为Patch表示。

#### 3.3.1 Hash n-gram Embeddings

为了保留字节级别的上下文信息，BLT引入了**哈希n-gram嵌入**：

对于每个字节位置$i$，构造3-8字节的n-gram：
$$g_{i,n} = \{b_{i-n+1}, \ldots, b_i\}$$

通过滚动多项式哈希将这些n-gram映射到固定大小的嵌入表：
$$e_i = x_i + \sum_{n=3}^{8} E_n^{hash}(\text{Hash}(g_{i,n}))$$

其中$E_n^{hash}$是每个n-gram大小对应的嵌入表（50万条目）。

**为什么用哈希？**
- 8字节n-gram的可能性空间为$256^8 \approx 10^{19}$，无法直接存储
- 哈希冲突在深度学习中被证明是可以接受的（类似于可学习的位置编码）
- 实验表明，哈希n-gram嵌入对性能提升至关重要（约0.04 BPB提升）

#### 3.3.2 Cross-Attention机制

Encoder使用**Perceiver风格的Cross-Attention**将字节表示聚合为Patch表示：

```
Patch Query (initialized by pooling patch bytes)
     ↓
Cross-Attend to Byte Keys/Values
     ↓
Patch Representation → Global Transformer
```

关键设计细节：
- Query通过Max Pooling字节表示初始化
- 每个Patch Query只关注属于该Patch的字节
- 使用Pre-LayerNorm，不使用位置编码

### 3.4 Latent Global Transformer

这是BLT的"大脑"，一个标准的自回归Transformer，但运行在Patch级别：

- 输入：Encoder输出的Patch表示序列
- 处理：标准的因果自注意力 + FFN
- 输出：下一Patch的表示

**Block-Causal Attention Mask**: 与Llama 3类似，使用块因果掩码，限制注意力在当前文档范围内。

### 3.5 Local Decoder设计

Decoder的目标是将Global Transformer的输出转换回字节。

与Encoder对称，但Cross-Attention的方向相反：
```
Byte Query (from encoder's byte representations)
     ↓
Cross-Attend to Patch Keys/Values (from Global Transformer output)
     ↓
Byte Logits (256-way classification)
```

**关键设计**: Decoder是**自回归的**，每个字节的预测依赖于之前已解码的字节。

### 3.6 与Tokenizer的本质区别

| 特性 | BPE Tokenizer | BLT Patches |
|------|--------------|-------------|
| 词表 | 固定（通常32K-128K） | 无固定词表 |
| 分块方式 | 基于频率统计 | 基于熵/复杂度 |
| 信息保留 | 字节信息丢失 | 完整字节信息 |
| 可变性 | 静态 | 动态、上下文相关 |
| 可扩展性 | 受限于词表大小 | 可任意调整Patch大小 |
| 推理成本 | 与Token数量成正比 | 与Patch数量成正比 |

---

## 四、实验结果与性能分析

### 4.1 训练设置

**数据集**:
- Llama 2数据集：2T tokens（用于Scaling Law实验）
- BLT-1T：1T高质量tokens（用于下游任务评估）

**模型规模**:
- 400M, 1B, 2B, 4B, 8B参数
- 最大训练：8B参数 + 4T bytes

**Baseline**:
- Llama 2 Tokenizer（32K词表）
- Llama 3 Tokenizer（128K词表）

### 4.2 Scaling Law：与BPE并驾齐驱

![Scaling Law](https://raw.githubusercontent.com/facebookresearch/blt/main/assets/scaling_law.png)

*图：BLT与BPE模型的Scaling Law对比。左图显示Space Patching基线仍落后于Llama 3；右图显示Entropy Patching结合架构改进后，BLT在计算最优区域与Llama 3持平。*

**关键发现**:
1. BLT-Entropy在1B到8B规模上与Llama 3的BPB（Bits-Per-Byte）基本持平
2. 这是**首个**在规模上匹敌BPE模型的字节级架构
3. 更大的Patch size（如8字节）在更大规模上表现更好

### 4.3 下游任务评估

| 任务 | Llama 3 (1T) | BLT-Space (6T bytes) | BLT-Entropy (4.5T bytes) |
|------|-------------|---------------------|------------------------|
| ARC-Easy | 77.6 | 75.4 | **79.6** |
| ARC-Challenge | **53.3** | 49.8 | 52.1 |
| HellaSwag | 79.1 | 79.6 | **80.6** |
| PIQA | 80.7 | **81.1** | 80.6 |
| MMLU | **58.1** | 54.8 | 57.4 |
| MBPP | 40.2 | 37.6 | **41.8** |
| HumanEval | 31.1 | 27.4 | **35.4** |
| **Average** | 60.0 | 58.0 | **61.1** |

*表：8B模型在BLT-1T数据集上的性能对比。所有模型FLOP匹配。BLT-Entropy平均性能超越Llama 3。*

**关键洞察**:
- BLT-Entropy在7项任务中4项超越Llama 3
- 代码生成任务（MBPP、HumanEval）提升显著
- BLT-Space虽然性能略低，但推理成本降低约30%

### 4.4 推理效率：Patches Scale Better Than Tokens

这是论文标题的核心论断，也是最具颠覆性的发现。

**固定推理成本Scaling实验**:

| 模型 | 参数量 | 相对Llama 2大小 | 平均Patch/Token大小 |
|------|-------|----------------|-------------------|
| Llama 2 | 470M | 1x | 3.7 bytes |
| Llama 3 | 450M | ~1x | 4.4 bytes |
| BLT-Entropy ps=6 | 610M | 1.3x | 6 bytes |
| BLT-Entropy ps=8 | 760M | 1.6x | 8 bytes |

*表：固定推理FLOP的模型配置。ps=Patch size。*

![固定推理成本Scaling](https://raw.githubusercontent.com/facebookresearch/blt/main/assets/inference_scaling.png)

*图：固定推理成本下的Scaling趋势。BLT-Entropy ps=6和ps=8都显示出比BPE更好的Scaling斜率，且在训练数据量超过计算最优点后迅速超越BPE。*

**核心发现**:
1. 在固定推理预算下，BLT可以同时增加**模型大小**和**Patch大小**
2. Patch size=8的模型在大型推理预算下很快成为最优选择
3. 交叉点（Crossover Point）在约2.5-3倍计算最优数据量处，远低于现代LLM的实际训练量

### 4.5 鲁棒性：字节级建模的天然优势

#### 4.5.1 噪声输入测试

| 任务 | Llama 3 (1T) | Llama 3.1 (16T) | BLT (1T) |
|------|-------------|----------------|---------|
| HellaSwag (原始) | 79.1 | **80.7** | 80.6 |
| HellaSwag (噪声平均) | 56.9 | 64.3 | **64.3** |
| - AntSpeak | 45.6 | **61.3** | 57.9 |
| - Drop | 53.8 | 57.3 | **58.2** |
| - RandomCase | 55.3 | 65.0 | **65.7** |
| - Repeat | 57.0 | 61.5 | **66.6** |
| - UpperCase | 72.9 | 76.5 | **77.3** |

*表：噪声HellaSwag评估。BLT仅用1/16的训练数据就达到与Llama 3.1相当的噪声鲁棒性。*

噪声策略包括：
- **AntSpeak**: 全大写、空格分隔
- **Drop**: 随机删除10%字符
- **RandomCase**: 随机大小写
- **Repeat**: 重复20%字符
- **UpperCase**: 全大写

#### 4.5.2 字符级理解：CUTE基准

| 任务类别 | Llama 3 | Llama 3.1 | BLT |
|---------|---------|----------|-----|
| **平均** | 27.5 | 20.0 | **54.1** |
| Contains Char | 0.0 | 0.0 | **55.9** |
| Contains Word | 55.1 | 21.6 | **73.5** |
| Spelling | 1.1 | - | **99.9** |
| Spelling Inverse | 30.1 | 3.6 | **99.9** |
| Substitute Char | 0.4 | 1.2 | **48.7** |
| Substitute Word | 16.4 | 6.8 | **72.8** |

*表：CUTE基准评估。BLT在字符操作任务上超越Tokenizer模型25+个百分点。*

CUTE任务包括：
- **字符操作**: 替换、删除、插入、交换字符
- **单词操作**: 替换、删除、插入单词
- **正字法/语义相似性**: 基于拼写或语义的选择

**关键洞察**: BLT展现出了对字符级结构的"原生"理解，而这是Tokenizer模型几乎无法学习的（即使训练数据量增加16倍）。

#### 4.5.3 低资源机器翻译

| 语言方向 | Llama 3 | BLT | 提升 |
|---------|---------|-----|------|
| **总体平均** | 12.1 / 5.9 | **14.0** / **6.4** | +1.9 / +0.5 |
| 亚美尼亚语 | 1.7 / 0.6 | **6.3** / **0.9** | +4.6 / +0.3 |
| 阿姆哈拉语 | 1.3 / 0.4 | **3.1** / **0.5** | +1.8 / +0.1 |
| 孟加拉语 | 4.7 / 1.7 | **12.7** / **4.1** | +8.0 / +2.4 |
| 格鲁吉亚语 | 1.7 / 1.0 | **7.4** / **2.5** | +5.7 / +1.5 |
| 古吉拉特语 | 2.0 / 1.0 | **5.8** / **2.2** | +3.8 / +1.2 |

*表：FLORES-101低资源语言翻译（BLEU分数）。左列为翻译到英语，右列为从英语翻译。*

**关键洞察**: BLT在低资源语言上的优势尤为明显，这验证了字节级建模对长尾数据分布的更好适应性。

### 4.6 从Llama 3初始化BLT

论文还探索了一个有趣的实验：**将预训练的Tokenizer模型"字节化"**。

方法：
1. 使用Llama 3.1（15T tokens训练）的Global Transformer权重初始化BLT的Global Transformer
2. 冻结Global Transformer，只训练Local Encoder/Decoder
3. 使用降低10倍的学习率微调Global Transformer

结果：
- BLT-from-Llama3在MMLU上超越从头训练的BLT
- 这为"如何将现有模型迁移到字节级架构"提供了可行路径
- 但其他任务仍有差距，需要进一步优化数据混合和超参数

---

## 五、架构消融实验

### 5.1 Entropy模型超参数

![Entropy模型Scaling](https://raw.githubusercontent.com/facebookresearch/blt/main/assets/entropy_scaling.png)

*图：不同Entropy模型规模和上下文长度的Scaling趋势。50M参数+512字节上下文后收益递减。*

**发现**:
- Entropy模型规模从1M增加到100M有显著提升
- 上下文长度从64增加到512有显著提升
- 超过50M参数和512字节后，收益递减

### 5.2 分块策略对比

| 分块策略 | 描述 | 相对性能 | 推理成本 |
|---------|------|---------|---------|
| Strided (k=4) | 每4字节分块 | 最差 | 低 |
| Strided (k=6) | 每6字节分块 | 较差 | 最低 |
| Space Patching | 空格分块 | 中等 | 低 |
| BPE Patching | 使用Tokenizer分块 | 中等 | 中 |
| Entropy Patching | 熵驱动动态分块 | **最佳** | 可调 |

**发现**: Entropy Patching在Scaling趋势和下游任务上都表现最佳。

### 5.3 Cross-Attention配置

| Encoder CA | Decoder CA | Pooling Init | Train Dist BPB |
|-----------|-----------|-------------|---------------|
| - | - | - | 0.866 |
| Last Layer | - | False | 0.886 |
| All Layers | Last Layer | True | **0.846** |
| All Layers | All Layers | True | **0.844** |

**发现**:
- Decoder Cross-Attention最关键
- Encoder Cross-Attention配合Pooling初始化有效
- 所有层使用Cross-Attention效果最佳

### 5.4 n-gram Hash Embeddings

| n-gram大小 | 每n-gram词表 | 总词表 | Train Dist BPB |
|-----------|-------------|-------|---------------|
| - | - | - | 0.850 |
| 6,7,8 | 100K | 300K | 0.842 |
| 3,4,5 | 100K | 300K | 0.837 |
| 3-8 | 400K | 2M | **0.826** |

**发现**:
- Hash n-gram嵌入至关重要（0.024 BPB提升）
- 较小的n-gram（3-5）比较大的n-gram（6-8）更重要
- 总词表大小有收益递减效应

### 5.5 Local模型层数配置

| Encoder层数 | Decoder层数 | Train Dist BPB |
|------------|-------------|---------------|
| 1 | 9 | **0.822** |
| 5 | 5 | 0.843 |
| 1 | 9 (无n-gram) | 0.850 |

**发现**: 配合n-gram嵌入时，极轻量的Encoder（1层）+ 较重的Decoder（9层）配置最佳。

---

## 六、个人理解与深度思考

### 6.1 BLT的革命性意义

**1. 架构范式的转变**

BLT不仅仅是一个"更好的Tokenizer"，它代表了一种全新的架构范式：
- **从离散到连续**: Token是离散的、固定的；Patch是连续的、动态的
- **从静态到动态**: Tokenizer是预处理步骤；Patching是模型的一部分
- **从有损到无损**: Tokenization丢失信息；BLT保留完整字节信息

**2. 计算效率的重新思考**

传统观点：字节级模型 = 序列长度×4 = 成本×16（因为Attention是二次的）

BLT证明：
- 实际成本取决于**Global Transformer的调用次数**（即Patch数量）
- 通过智能分块，可以在保持性能的同时减少Global Transformer的调用
- 更大的Patch size允许更大的模型，形成新的Scaling维度

**3. 对Tokenizer的必要性提出根本质疑**

论文的结论可以被解读为：**Tokenizer在LLM中的必要性是一个历史偶然，而非理论必然。**

Tokenizer最初被引入是因为：
- 早期RNN难以处理长序列
- 计算资源受限
- 需要固定大小的词表来输出概率分布

BLT表明，在现代Transformer架构和充足计算资源下，这些限制可以被绕过。

### 6.2 BLT的局限与挑战

**1. 工程实现复杂度**

- 动态分块导致批次内序列长度不一致，需要特殊的Padding和Packing策略
- Cross-Attention的动态掩码需要Flex Attention等高效实现
- 熵模型增加了训练和推理的复杂性

**2. 训练稳定性**

- 动态分块使得梯度估计更复杂
- Patch边界的离散决策难以端到端优化
- 长Patch可能导致梯度传播问题

**3. 与现有生态的兼容性**

- 现有的高效推理Kernel（如FlashAttention）主要针对固定长度序列优化
- 大多数LLM工具和库假设Tokenized输入
- 部署和服务的复杂性增加

**4. 优化空间**

论文明确指出，BLT的Scaling Law可能还有优化空间：
- 当前的计算最优比例是基于BPE模型计算的，可能不适用于BLT
- 更大的Patch size在更大规模上可能表现更好，但需要验证
- 端到端学习分块策略（而非使用独立的熵模型）是开放问题

### 6.3 对行业的影响

**1. 多语言模型的民主化**

Tokenizer是 multilingual LLM 的最大障碍之一：
- 不同语言需要不同的Token分配
- 非英语语言经常被"压缩"到较少的Token中
- 代码切换（Code-switching）场景性能差

BLT的字节级方法天然支持：
- 所有语言的平等表示
- 无缝代码切换
- 新语言/领域的零样本适应

**2. 鲁棒性关键应用**

对于需要处理噪声输入的场景：
- OCR错误纠正
- 语音识别后处理
- 非正式文本（社交媒体、聊天）处理
- 拼写检查和纠正

BLT的字节级鲁棒性将带来显著优势。

**3. 推理成本的长期趋势**

虽然BLT的当前实现在 wall-clock time 上可能尚未超越高度优化的Tokenizer模型，但：
- 更大的Patch size意味着更少的Global Transformer步骤
- 在固定推理预算下，BLT可以同时增加模型大小和Patch大小
- 随着模型规模增长，这一优势将更加明显

**4. 对未来架构的启示**

BLT的成功可能催生更多"无Tokenizer"架构：
- **Diffusion-based**: 如Zonkey等扩散语言模型
- **State-space models**: 如MambaByte
- **Hierarchical architectures**: 多层级Patching策略

---

## 七、相关工作对比

### 7.1 字节级模型的演进

| 工作 | 年份 | 规模 | 核心方法 | 与BLT的比较 |
|------|------|------|---------|------------|
| ByT5 | 2022 | - | 字节级T5 | 无Patching，计算成本过高 |
| MegaByte | 2023 | 1.3B | 固定大小Patching | 静态分块，性能落后于BPE |
| SpaceByte | 2024 | 1B | 空格分块 | 简单启发式，不如熵分块 |
| MambaByte | 2024 | 350M | Mamba + 字节 | 无Patching，线性复杂度 |
| **BLT** | **2025** | **8B** | **熵分块 + Cross-Attention** | **首个匹敌BPE的字节级模型** |

### 7.2 动态计算分配

| 工作 | 方法 | 与BLT的关系 |
|------|------|------------|
| MoE (Mixture of Experts) | 为每个Token选择专家子集 | BLT在不同粒度（Patch级别）分配计算 |
| Early Exit | 简单Token提前退出 | BLT通过Patch大小控制计算 |
| CoLT5 | 条件计算长序列 | BLT的分块策略可以被视为一种条件计算 |

### 7.3 分块/Patching策略

| 工作 | 分块方法 | 与BLT的比较 |
|------|---------|------------|
| Hourglass Transformer | 学习型下采样 | 规模较小，需要端到端训练分块器 |
| Token Pooling | 动态Token合并 | 在Token级别操作，非字节级别 |
| BLT | 熵驱动分块 | 简单高效，无需端到端训练 |

---

## 八、未来研究方向

### 8.1 端到端学习分块策略

当前的熵模型是独立训练的，未来可以探索：
- 使用强化学习训练分块策略
- Gumbel-Softmax等可微分方法
- 多目标优化（性能 vs 效率）

### 8.2 多模态扩展

BLT的思想可以扩展到：
- **图像**: 像素级Patching
- **音频**: 采样点级Patching
- **视频**: 时空Patching

统一的字节/像素/采样点级处理可能带来跨模态的新突破。

### 8.3 超长上下文

字节级模型天然适合：
- 处理原始二进制数据（代码、PDF等）
- 无损长文档建模
- 细粒度引用和溯源

### 8.4 硬件协同设计

当前的Transformer硬件优化主要针对：
- 固定长度序列
- Token级别的内存访问模式

BLT可能需要：
- 动态序列长度的Kernel优化
- 字节级访问的高效内存布局
- 针对Cross-Attention的专用硬件支持

---

## 九、结论

Byte Latent Transformer (BLT) 代表了大型语言模型架构的重大突破。通过引入**基于熵的动态分块机制**和**三模块架构**，BLT首次证明了字节级模型可以在规模上完全匹敌甚至超越基于Tokenizer的模型。

**核心贡献回顾**:

1. **性能突破**: 8B规模上匹配Llama 3，某些任务（代码生成、鲁棒性测试）显著超越
2. **效率优势**: 固定推理成本下可节省高达50%的FLOP，或用于增加模型规模
3. **鲁棒性**: 对噪声输入和字符级任务展现出Tokenizer模型无法企及的能力
4. **多语言公平性**: 低资源语言翻译性能显著提升
5. **新Scaling维度**: Patch大小成为与模型规模并列的新优化维度

**对行业的意义**:

BLT的成功标志着**Tokenizer可能不再是LLM的必要组件**。虽然短期内基于Tokenizer的模型仍将主导（由于工程生态的惯性），但长期来看，BLT及其后续工作可能推动整个领域向"无Tokenizer"架构迁移。

这不仅是技术上的进步，更代表了一种范式的转变：**从"如何让模型理解离散的Token"到"如何让模型直接理解原始数据"**。

---

## 参考文献与资源

- **论文**: https://arxiv.org/abs/2412.09871
- **代码**: https://github.com/facebookresearch/blt
- **ACL Anthology**: https://aclanthology.org/2025.acl-long.453/

**关键引用**:
```bibtex
@inproceedings{pagnoni2025blt,
  title={Byte Latent Transformer: Patches Scale Better Than Tokens},
  author={Pagnoni, Artidoro and Pasunuru, Ram and Rodriguez, Pedro and Nguyen, John and Muller, Benjamin and Li, Margaret and Zhou, Chunting and Yu, Lili and Weston, Jason and Zettlemoyer, Luke and others},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics},
  year={2025}
}
```

---

*本文分析基于BLT论文公开内容，仅代表个人观点，不代表Meta或任何机构的官方立场。*
