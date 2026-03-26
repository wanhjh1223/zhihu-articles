# ConFu: 让AI"思考未来"的推测解码新范式

## 论文概览

**论文标题**: ConFu: Contemplate the Future for Better Speculative Sampling  
**作者**: Zongyue Qin*, Raghavv Goel*, Mukul Gagrani, Risheek Garrepalli, Mingu Lee, Yizhou Sun  
**机构**: 加州大学洛杉矶分校 (UCLA), 高通AI研究院 (Qualcomm AI Research)  
**发表时间**: 2026年3月9日  
**arXiv编号**: arXiv:2603.08899v1  
**收录会议**: ICLR 2026 Workshop on Latent Implicit Thinking

---

## 一、研究背景与核心问题

### 1.1 大模型推理的"阿喀琉斯之踵"

大型语言模型（LLMs）在自然语言处理领域取得了令人瞩目的成就，从文本生成到复杂推理任务都展现出强大的能力。然而，这些模型的推理效率却成为制约其实际应用的关键瓶颈。由于自回归生成的本质特性，每个解码步骤都需要通过完整模型进行一次前向传播，这导致：

- **高延迟**: 每个token的生成都需要重新加载模型权重和进行内存同步
- **计算资源浪费**: GPU的大量计算能力因序列依赖而处于闲置状态
- **用户体验受损**: 响应时间过长严重影响实时交互场景的应用

特别是在测试时扩展（Test-Time Scaling）成为主流趋势的背景下，模型如ChatGPT o1和DeepSeek-R1需要在生成最终答案前进行长时间的 deliberative reasoning，这进一步加剧了推理成本的挑战。

### 1.2 推测解码：打破序列依赖的曙光

推测解码（Speculative Decoding）作为一种无损加速技术，为解决上述问题提供了新思路。其核心思想借鉴了"主-从"协作模式：

**传统推测解码流程**：
1. **起草阶段**: 轻量级草稿模型（Draft Model）自回归地生成候选token序列
2. **验证阶段**: 目标大模型（Target Model）在单次前向传播中并行验证所有候选
3. **接受/拒绝**: 按顺序检查token，接受的token直接采用，拒绝的token及后续token被丢弃
4. **修正阶段**: 用目标模型在拒绝位置重新采样，继续下一轮迭代

这种方法的理论基础是：**验证比生成便宜得多**。通过将多个草稿token的生成成本"摊销"到单次目标模型验证中，可以实现2-3倍的加速而不损失输出质量。

### 1.3 现有方法的局限性：误差累积困境

尽管EAGLE系列方法代表了当前推测解码的最高水平，但所有现有方法都面临一个根本性缺陷：**草稿模型仅基于当前前缀进行预测**。

这种设计导致了一个严重问题——**误差累积**：

- **初始阶段**: 草稿模型的隐藏表示与目标模型对齐良好，预测准确
- **推进过程**: 小误差逐渐累积，草稿分布逐渐偏离目标分布
- **结果**: token接受率随步骤增加而下降，效率增益被削弱

正如论文中的图1a所示，没有未来预测的草稿模型就像一个人"走一步看一步"，完全没有全局规划，最终必然偏离正确方向。

### 1.4 核心洞察：人类思维vs机器生成

研究团队从一个深刻的人类认知现象获得启发：**当我们说话时，大脑会提前"预演"接下来要说什么**。不是逐字蹦出，而是先在心中构思大致方向，然后才开始表达。

类比到LLM推理：
- **当前方法**: 像一个人写作文时每写一个字都要停下来想下一个字
- **理想方式**: 先构思整段话的主题和结构，再填充具体内容

这种"先思考再说话"的能力，正是ConFu希望赋予AI的。

---

## 二、ConFu核心技术详解

### 2.1 核心概念：Contemplate Tokens（思考Token）

ConFu的核心创新是引入**Contemplate Tokens**（论文中也称为Pause Tokens），这是一种特殊的token，用于编码目标模型的"中间思考状态"。

**关键设计原则**：
1. **连续表示**: 与传统离散token不同，contemplate tokens产生连续的隐藏表示，捕获目标模型当前的"思维"方向
2. **并行处理**: 这些token可以与其他输入token并行处理，增加的推理成本极小
3. **冻结目标模型**: 不修改目标模型参数，保持模型行为一致性

**实现机制**：
```
输入前缀 + Soft Prompt Tokens + [CONTEMPLATE]
    ↓
目标模型（冻结）
    ↓
隐藏表示（未来预测信号）+ 下一个token
```

其中：
- **Soft Prompt Tokens**: 可学习的嵌入向量，作为辅助参数指导目标模型产生未来预测
- **Attention Mask修改**: 仅允许contemplate tokens关注soft prompt tokens，确保输入前缀表示不受影响

### 2.2 动态Contemplate Tokens：MoE赋能上下文感知

静态的contemplate token嵌入难以适应多样化的生成上下文。ConFu进一步提出了**基于混合专家模型（Mixture-of-Experts, MoE）的动态contemplate token机制**。

**动机**：
- 数学推理任务可能需要"我的下一个方程是："这样的指令
- 长文本写作任务则需要"这一段的主题是："的引导
- 单一固定指令无法在所有场景下准确捕获目标模型的思维

**MoE架构设计**：

```
最近接受token的隐藏状态
    ↓
Router（线性层）→ 专家权重（Softmax归一化）
    ↓
Top-K专家选择
    ↓
加权组合 → 最终Contemplate Token嵌入
```

**技术细节**：
- 维护n_expert个可学习的token嵌入作为专家
- 使用最近接受（或生成）token的隐藏状态作为路由输入
- 输出为选定专家嵌入的加权和
- 分别应用于[con]（目标模型输入）和[f]（草稿模型输入）

这是**首次在暂停token中引入动态性**，使contemplate token能够根据当前上下文自适应选择最合适的"指令"。

### 2.3 推理流程：未来感知的推测解码

ConFu的完整推理流程如图2所示：

**第一步：未来预测生成**
1. 输入token序列进入目标模型
2. 同时附加soft prompt tokens和contemplate token
3. 目标模型生成：
   - 下一个输出token
   - 未来预测向量f（contemplate token的隐藏表示）

**第二步：草稿生成**
1. 草稿模型接收输入序列 + 未来预测向量f作为辅助token
2. 基于共享的未来信号f，自回归生成多个候选token
3. f在整个起草过程中保持固定

**第三步：树形验证与并行未来预测**

这是ConFu最具创新性的设计之一。关键挑战是：**未来预测必须对应最终接受的草稿token，而这个token在验证前是未知的**。

解决方案：
1. 在草稿token树中为每个草稿节点插入一个contemplate token
2. 修改树注意力掩码，使目标模型能为每个草稿候选并行生成不同的未来预测
3. 验证完成后，选择与最后接受token关联的未来预测
4. 将该预测传递给草稿模型用于下一轮迭代

**复杂度分析**：
- 设前缀长度为t，soft prompt tokens数量为s（通常s=16），草稿树节点数为T（通常T=30）
- 首次迭代（prefill）：处理t+s+1个token
- 后续迭代：验证2T个token（T个草稿token + T个contemplate tokens）
- 额外开销相比目标模型解码总成本而言很小

### 2.4 训练框架：稳健的Future Prediction学习

ConFu的草稿模型头在架构上与EAGLE-3类似，但关键区别在于将未来预测作为额外输入token。

**训练目标**：
```
L = Σ_{t=1}^N Σ_{i=1}^L KL(P_Mt(x_{t+i}|x_{1:t+i-1}) || P_Md(x_{t+i}|x_{1:t+i-1}, h^Md_{1:t}, h̃_{t+1:t+i-1}))
```

其中：
- KL表示KL散度
- P_Mt和P_Md分别是目标和草稿模型的输出分布
- h^Md_{1:t}是目标模型隐藏表示的下投影
- h̃是草稿模型的隐藏表示

**训练时测试（Train-Time Testing）**：
- 模拟推理时的条件，使训练与推理保持一致
- 解决训练-推理不匹配问题

**Anchor Token Sampling（锚定Token采样）**：
- 稳定训练过程
- 提高未来预测的鲁棒性

**Future Prediction Replication（未来预测复制）**：
- 确保未来预测信号的有效传播
- 增强模型对未来方向的感知能力

---

## 三、实验结果与性能分析

### 3.1 实验设置

**评估基准**: SpecBench (Xia et al., 2024)  
**测试模型**: Llama-3 3B和8B  
**对比基线**: EAGLE-3（当前最先进的推测解码方法）  
**任务覆盖**: 
- 写作 (Writing)
- 问答 (QA)
- 摘要 (Summarization)
- 翻译 (Translation)
- 代码生成 (Coding)
- 数学推理 (Math)

### 3.2 核心性能指标

**Token接受率提升**：
- 相比EAGLE-3平均提升8-11%
- 在所有任务类别上均有提升
- 在数学推理任务上提升尤为显著

**生成速度提升**：
- Llama-3 3B: 速度提升与接受率提升一致
- Llama-3 8B: 同样实现8-11%的端到端加速
- 在不同采样温度下表现稳定

### 3.3 细粒度分析

**不同解码条件的鲁棒性**：
- 采样温度: 从0到1的不同温度设置下均保持稳定提升
- 序列长度: 长序列生成中优势更明显
- 任务类型: 在需要复杂推理的任务上增益最大

**计算效率**：
- 额外开销主要来自2T个token的并行处理（T≈30）
- 相比目标模型完整前向传播成本，增加比例很小
- 草稿模型的额外计算可以忽略不计（单层Transformer）

### 3.4 与相关工作的对比

| 方法 | 核心思想 | 相对EAGLE-3提升 | 主要局限 |
|------|---------|----------------|---------|
| EAGLE-1 | 单层Transformer草稿头 | 基线 | 误差累积 |
| EAGLE-2 | 动态草稿树 | +5-7% | 仍基于当前前缀 |
| EAGLE-3 | 多层特征融合 | +10-15% | 无未来感知 |
| BiTA | 双向调优 | 相当 | 需修改目标模型 |
| **ConFu** | **Contemplate Tokens + MoE** | **+8-11%** | **额外contemplate tokens开销** |

**关键区别**：
- BiTA直接从容思考token解码未来token
- ConFu使用思考token的表示来指导草稿生成
- ConFu保持目标模型冻结，更具实用性

---

## 四、技术创新深度解析

### 4.1 推测解码与隐式推理的首次融合

ConFu最重要的理论贡献是**首次将推测解码与连续隐式推理（Latent Implicit Reasoning）范式连接起来**。

**背景**：
- 近期研究表明，LLM可以通过后训练生成连续的"思考token"作为中间推理状态
- 但这类方法需要目标模型的多次前向传播，成本极高

**ConFu的突破**：
- 利用contemplate tokens编码目标模型的当前"思维"
- 通过单次前向传播获取未来预测信号
- 以极低成本实现类似隐式推理的效果

**意义**：
- 开辟了通过"未来感知"加速LLM推理的新方向
- 为推测解码与推理能力增强的结合提供了范式

### 4.2 MoE在Token级别的创新应用

传统MoE应用于模型层级别，ConFu将其创新性地应用于**Token嵌入级别**：

**设计亮点**：
1. **条件化专家选择**: 基于当前上下文动态选择专家
2. **轻量级路由**: 单层线性层，计算开销极小
3. **双MoE模块**: 分别服务于目标模型和草稿模型

**实际效果**：
- 不同任务类型自动选择最合适的"指令风格"
- 数学任务倾向于选择结构化、逻辑性强的专家
- 创意写作任务倾向于选择开放式、描述性的专家

### 4.3 训练-推理一致性的保障

推测解码方法常面临训练与推理条件不匹配的问题，ConFu通过以下机制解决：

**Anchor Token Sampling**：
- 在训练时模拟推理时的token接受/拒绝过程
- 使用锚定token确保梯度传播稳定

**Future Prediction Replication**：
- 确保未来预测信号在训练batch中的一致性
- 防止因采样随机性导致的训练不稳定

**与EAGLE-3训练框架的兼容性**：
- 继承EAGLE-3的训练时测试范式
- 平滑升级路径，便于实际部署

---

## 五、个人理解与行业影响分析

### 5.1 为什么是"思考未来"？

ConFu的核心理念——让草稿模型"思考未来"——触及了机器生成与人类思维的根本差异：

**人类写作过程**：
1. 构思整体结构和主要观点
2. 规划段落逻辑和论证顺序
3. 逐句展开，保持与整体方向一致

**传统LLM生成**：
1. 仅基于已生成的token
2. 局部最优选择可能偏离全局目标
3. 缺乏"大局观"导致错误累积

ConFu通过contemplate tokens赋予模型"大局观"，这是对自回归生成范式的根本性改进。

### 5.2 对推测解码领域的深远影响

**短期影响（6-12个月）**：
1. **EAGLE-4的潜在方向**: ConFu很可能被整合进EAGLE系列的下一版本
2. **vLLM/SGLang集成**: 推测解码框架将考虑支持contemplate tokens
3. **端侧推理优化**: 高通作为移动芯片巨头，ConFu可能首先在端侧LLM推理中落地

**中期影响（1-2年）**：
1. **多模态推测解码**: 将"未来感知"扩展到视觉-语言模型的推测解码
2. **与推理模型结合**: o1/R1类模型的推测解码将面临新机遇
3. **硬件协同设计**: 针对contemplate tokens优化的专用加速器可能出现

**长期影响（3-5年）**：
1. **新解码范式**: 从"预测下一个token"到"规划未来序列"
2. **认知架构融合**: 推测解码与认知科学中的"前瞻"机制结合
3. **AGI推理基础**: 为未来AGI系统的高效推理提供基础组件

### 5.3 对AI基础设施生态的影响

**云服务商**：
- 降低推理成本，提升API服务利润率
- 支持更长的测试时计算，增强模型能力

**端侧AI**：
- 高通主导的ConFu天然适合端侧部署
- 手机、IoT设备上运行更大模型的可能性

**开源社区**：
- EAGLE系列已经开源，ConFu技术有望跟进
- vLLM等推理引擎将受益

### 5.4 局限性与未来研究方向

**当前局限**：
1. **额外内存开销**: 需要存储contemplate tokens的嵌入
2. **MoE训练复杂度**: 动态专家选择增加了训练难度
3. **长上下文适应性**: 在超出训练上下文长度时的泛化能力待验证

**未来研究方向**：
1. **分层未来预测**: 不仅预测下一步，而是多步未来规划
2. **自适应contemplate深度**: 根据任务复杂度动态调整思考深度
3. **跨模态contemplation**: 将视觉、音频信号纳入未来预测
4. **与强化学习的结合**: 使用RL优化contemplate策略

---

## 六、技术实现细节补充

### 6.1 Contemplate Token的数学形式

设目标模型的隐藏维度为d，contemplate token的嵌入为e_con ∈ R^d。

**标准静态嵌入**：
```
e_con = W_con  (可学习参数)
```

**ConFu的动态MoE嵌入**：
```
g = Softmax(W_router · h_last)  # 路由权重
W_con = Σ_{i=1}^K g_i · E_i    # 加权专家组合
```

其中：
- h_last: 最近接受token的隐藏状态
- W_router: 路由线性层
- E_i: 第i个专家嵌入
- K: 选择的专家数量（Top-K）

### 6.2 树注意力掩码修改

在标准树注意力中，注意力矩阵定义了token间的可见关系。

**ConFu的修改**：
- 每个草稿节点t_i对应一个contemplate token c_i
- c_i可以访问t_i及其所有祖先
- c_i的输出用于下一轮的未来预测
- 注意力矩阵大小从T×T变为2T×2T

### 6.3 与现有推测解码方法的兼容性

ConFu设计为可与以下技术叠加：
- **树形解码**: 与Medusa、EAGLE-2/3的草稿树兼容
- **量化**: 支持INT8/INT4量化降低内存占用
- **连续批处理**: 可与vLLM的PagedAttention结合
- **张量并行**: 支持多GPU分布式推理

---

## 七、结论

ConFu代表了推测解码领域的重要突破，通过引入"思考未来"的能力，解决了现有方法中误差累积的根本性问题。其核心贡献包括：

1. **Contemplate Tokens**: 以极低开销捕获目标模型的未来意图
2. **动态MoE机制**: 实现上下文感知的自适应未来预测
3. **训练框架创新**: Anchor Token Sampling和Future Prediction Replication确保训练稳定性

**实验结果**证明了ConFu的有效性：相比EAGLE-3提升8-11%的token接受率和生成速度，且在所有任务类型上表现一致。

更重要的是，ConFu首次将推测解码与连续隐式推理范式连接起来，为LLM推理加速开辟了新的研究方向。随着测试时扩展成为主流趋势，ConFu这类能够提升推理效率的技术将在AI基础设施中扮演越来越重要的角色。

对于研究者和工程师而言，ConFu不仅是一个具体的算法改进，更是一种思维范式的转变：从"预测下一个token"到"规划未来方向"。这种转变可能预示着自回归生成模型的下一次重大进化。

---

## 参考资料

1. Qin, Z., et al. (2026). ConFu: Contemplate the Future for Better Speculative Sampling. arXiv:2603.08899v1.
2. Li, Y., et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test. arXiv:2503.01840.
3. Leviathan, Y., et al. (2023). Fast Inference from Transformers via Speculative Decoding. ICML.
4. Miao, X., et al. (2024). SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification. ASPLOS.
5. Xia, H., et al. (2024). SpecBench: A Benchmark for Speculative Decoding.
6. Hao, S., et al. (2024). Training Large Language Models to Reason in a Continuous Latent Space.
7. Goyal, S., et al. (2023). Think Before You Speak: Training Language Models with Pause Tokens.
8. Lin, F., et al. (2025). BiTA: Bi-directional Tuning for Lossless Acceleration in Large Language Models.

---

*本文基于arXiv:2603.08899v1论文内容进行分析，部分技术细节参考了相关开源实现和官方解读。*
