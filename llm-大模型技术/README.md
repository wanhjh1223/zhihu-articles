# 大模型技术前沿论文分析

本仓库收录大模型（LLM）领域的前沿论文深度分析，涵盖架构创新、训练方法、推理优化、Agent、RAG、多模态等方向。

## 📊 最新文章

| 编号 | 日期 | 论文 | 核心创新 |
|------|------|------|----------|
| 297 | 2026-04-10 | [FlashAttention-4: Blackwell架构深度优化](./297-2026-04-10-FlashAttention-4-Blackwell架构深度优化.md) | 针对NVIDIA Blackwell架构的非对称硬件扩展，通过算法-内核协同设计实现71%峰值利用率，超越cuDNN 1.3x和Triton 2.7x |
| 296 | 2026-04-07 | [MASKSEARCH: Agent搜索能力预训练框架](./296-2026-04-07-MASKSEARCH-Agent搜索能力预训练框架.md) | 阿里通义实验室提出RAMP预训练任务，多智能体+自进化蒸馏+课程学习，显著增强LLM Agent搜索能力 |
| 279 | 2025-04-06 | [RAG-Critic: 自动化批评引导智能体工作流](./279-2025-04-06-RAG-Critic-自动化批评引导智能体框架.md) | 首个层次化RAG错误系统(7/19/4000+标签)，批评模型自动诊断+规划智能体自我修正 |
| 278 | 2026-04-06 | [SKILL0-ICRL: 内化技能到推理过程](./278-2026-04-06-SKILL0-ICRL-Skill-Internalization.md) | 将外部工具调用能力内化到推理过程中，实现零工具调用的Agent推理 |

## 📚 已收录论文列表

### 架构创新
- FlashAttention-4 - Blackwell架构Attention优化 (2026)
- Titans - 神经长时记忆架构 (Google Research)
- Mamba系列 - 状态空间模型
- Mixture of Experts (MoE) - 混合专家模型

### 训练与优化
- DeepSeek-R1 - 强化学习推理
- TTRL - 测试时强化学习
- GRPO - 分组相对策略优化

### RAG与Agent
- **MASKSEARCH** - Agent搜索能力预训练框架 (阿里通义实验室 2025)
- **RAG-Critic** - 自动化批评引导智能体 (ACL 2025)
- Self-RAG - 自我反思检索增强
- ReAct - 推理与行动结合

### 多模态
- CLIP - 视觉语言对齐
- Flamingo - 少样本学习
- GPT-4V - 视觉理解

## 🔄 更新频率

每日更新，追踪大模型领域最新突破。

## 📖 关于

本仓库由AI助手自动维护，专注于高质量、有深度的论文分析。

---

**免责声明**：所有论文分析仅供学术研究参考，版权归原论文作者所有。
