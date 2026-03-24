# 预训练数据集推荐清单

> 针对您列出的10个低分领域，整理的高质量预训练数据集（非测试集）

---

## 1. 初等与中等数学 (Elementary & High School Mathematics)

### 核心数据集

| 数据集 | 规模 | 特点 | HuggingFace链接 |
|--------|------|------|-----------------|
| **OpenWebMath** | 14.7B tokens | 从Common Crawl提取的数学网页，含LaTeX公式 | https://huggingface.co/datasets/open-web-math/open-web-math |
| **FineMath-3plus/4plus** | 高质量子集 | HuggingFaceTB筛选的教育级数学内容 | https://huggingface.co/datasets/HuggingFaceTB/finemath |
| **InfiMM-WebMath-40B** | 40B tokens | 大规模数学网页语料 | https://huggingface.co/datasets/Infi-MM/InfimWebMath |
| **Proof-Pile-2** | 学术级 | 包含数学证明的语料库 | https://huggingface.co/datasets/EleutherAI/proof-pile-2 |
| **MetaMathQA** | 395K条 | GSM8K+MATH改写的高质量CoT数据 | https://huggingface.co/datasets/meta-math/MetaMathQA |
| **NVIDIA/OpenMathInstruct-1** | 大规模 | 数学指令微调数据 | https://huggingface.co/datasets/nvidia/OpenMathInstruct-1 |
| **MegaMath** | 高级推理 | 合成数学推理数据 | https://huggingface.co/datasets/AI-MO/MegaMath |

### 使用建议
- 优先使用 **OpenWebMath** + **FineMath** 组合
- 合成分步解题CoT数据（参考Nemotron-CC-Math的52B方案）
- 加入常见错误纠正样本（negative examples）

---

## 2. 专业法律 (Professional Law)

### 核心数据集

| 数据集 | 规模 | 特点 | HuggingFace链接 |
|--------|------|------|-----------------|
| **Legal-BERT Corpus** | 大规模 | 英文法律文档预训练语料 | https://huggingface.co/nlpaueb/legal-bert-base-uncased |
| **CaseHOLD** | 53,000+条 | 法律裁决样本 | https://huggingface.co/casehold |
| **LEDGAR** | 大规模 | 合同条款分类语料 | https://huggingface.co/datasets/ledgar |
| **cLegal-QA (中文)** | 数万条 | 中国法律问答数据 | 需通过论文联系作者获取 |
| **InternLM-Law Corpus (中文)** | 大规模 | 中文法律语料+通用数据 | https://huggingface.co/internlm/internlm-law-7b |
| **LawRefBook/Laws (中文)** | 3,500+条 | 中国法律法规 | https://github.com/LawRefBook/Laws |
| **CrimeKgAssitant (中文)** | 20万+条 | 罪名知识+法务问答 | https://github.com/liuhuanyong/CrimeKgAssitant |

### 使用建议
- 中文场景必须补充 **LawRefBook** 和 **CrimeKgAssitant**
- 构建"法条-案例-解释"三联体数据
- 合成法律职业资格考试（法考）推理链条

---

## 3. 伦理道德与复杂推理 (Moral Scenarios & Formal Logic)

### 核心数据集

| 数据集 | 规模 | 特点 | HuggingFace链接 |
|--------|------|------|-----------------|
| **Moral-Reason-QA** | 680条+推理轨迹 | 高歧义道德场景+伦理框架推理 | https://huggingface.co/datasets/zankjhk/Moral-Reason-QA |
| **MoralExceptQA** | 大规模 | 道德例外情况问答 | https://huggingface.co/datasets/feradauto/MoralExceptQA |
| **ETHICS** | 全面覆盖 | 道德情境判断数据集 | https://huggingface.co/datasets/hendrycks/ethics |
| **MFTCXplain** | 多语言 | 多跳仇恨言论解释（含道德推理） | 通过论文获取 |
| **Moralise (VLM)** | 2,481对 | 视觉语言模型道德对齐 | https://huggingface.co/datasets/Ze1025/MORALISE |
| **moral_stories** | 12,000条 | 道德故事+行为后果 | https://huggingface.co/datasets/demelin/moral_stories |

### 使用建议
- 使用 **Moral-Reason-QA** 训练多伦理框架（功利主义/义务论/美德伦理）
- 合成"问题-反思-修正"三元组格式
- 多智能体辩论生成结构化辩论文本

---

## 4. 计算机系统核心 (OS & Computer Networks)

### 核心数据集

| 数据集 | 规模 | 特点 | HuggingFace链接 |
|--------|------|------|-----------------|
| **The Stack v2** | 大规模 | 代码预训练语料（含系统级代码） | https://huggingface.co/datasets/bigcode/the-stack-v2 |
| **Stack-Edu-Python** | 教育级 | 高质量Python代码（含系统编程） | https://huggingface.co/datasets/HuggingFaceTB/python-edu |
| **OpenCoder Annealing Corpus** | 算法+代码 | 包含系统原理解释的合成数据 | https://huggingface.co/datasets/OpenCoder-LLM/opc-annealing-corpus |
| **CommitPackFT** | 92种语言 | Git提交数据（含系统级修改） | https://huggingface.co/datasets/bigcode/commitpackft |
| **StackOverflow-Clean** | 问答级 | 编程问答（含OS/网络问题） | https://huggingface.co/datasets/bigcode/stackoverflow-clean |
| **Glaive-Code-Assistant** | 大规模 | 代码助手训练数据 | https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v3 |

### 使用建议
- 结合代码+系统原理解释的混合文本
- 提取Linux内核、Redis等开源项目源码注释
- 合成"代码实现+原理说明"数据（如简易进程调度器+抢占式多任务解释）

---

## 5. 基础医学与生命科学 (Basic Medical Sciences)

### 核心数据集

| 数据集 | 规模 | 特点 | HuggingFace链接 |
|--------|------|------|-----------------|
| **PubMed Central** | 14M+文献 | 医学全文文献 | https://huggingface.co/datasets/ncbi/pubmed_central |
| **PubMed Abstracts** | 30M+摘要 | 医学文献摘要 | https://huggingface.co/datasets/ncbi/pubmed |
| **Me-LLaMA Corpus** | 129B tokens | 生物医学文献+临床笔记 | https://huggingface.co/clinicalnlplab |
| **MIMIC-IV** | 需申请 | 临床病历数据 | https://physionet.org/content/mimic-iv-note/ |
| **PMC-Patients** | 病例级 | 患者病例报告 | https://huggingface.co/datasets/PMC-Patients |
| **MedCPT Corpus** | 医学检索 | PubMed搜索日志 | https://huggingface.co/ncbi/MedCPT |
| **FOMO300K** | 31万+扫描 | 脑部MRI影像数据 | https://huggingface.co/datasets/fomo300k |

### 使用建议
- **PubMed Central + Abstracts** 是基础医学核心语料
- 构建"症状-诊断-治疗"推理链条
- 合成解剖学3D结构描述（从图谱提取）

---

## 6. 物理学 (High School to College Physics)

### 核心数据集

| 数据集 | 规模 | 特点 | HuggingFace链接 |
|--------|------|------|-----------------|
| **OpenWebMath (Physics子集)** | 筛选后 | 含物理公式的数学网页 | https://huggingface.co/datasets/open-web-math/open-web-math |
| **FineWeb-Edu (Physics)** | 教育级 | 高质量教育网页（含物理） | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu |
| **Proof-Pile-2 (Physics)** | 学术级 | 含物理证明的学术文本 | https://huggingface.co/datasets/EleutherAI/proof-pile-2 |
| **Poseidon PDE Dataset** | 大规模 | 流体动力学PDE求解数据 | https://huggingface.co/collections/camlab-ethz/poseidon |
| **PDEGym Collection** | 多领域 | 偏微分方程求解数据集 | https://huggingface.co/collections/camlab-ethz/pdegym-665472c2b1181f7d10b40651 |
| **Cosmopedia (Physics)** | 合成级 | Mixtral生成的教育内容 | https://huggingface.co/datasets/HuggingFaceTB/cosmopedia |

### 使用建议
- 重点合成"概念物理"机制解释（如卫星轨道+万有引力+圆周运动联合推理）
- "实验设计→现象观察→理论解释→误差分析"完整科研流程
- 与数学数据结合（微积分推导物理公式）

---

## 7. 化学 (Chemistry)

### 核心数据集

| 数据集 | 规模 | 特点 | HuggingFace链接 |
|--------|------|------|-----------------|
| **ChemPile** | 75B+ tokens | 化学基础模型专用数据集（250GB） | NeurIPS 2025发布，即将上线HuggingFace |
| **OpenWebMath (Chemistry)** | 筛选后 | 含化学计算的数学内容 | https://huggingface.co/datasets/open-web-math/open-web-math |
| **FineWeb-Edu (Chemistry)** | 教育级 | 化学教育网页 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu |
| **OMG Dataset** | 3.1T碱基对 | 宏基因组学语料 | https://huggingface.co/datasets/tattabio/OMG |
| **PubMed (Chemistry)** | 文献级 | 生物化学相关文献 | https://huggingface.co/datasets/ncbi/pubmed |

### 使用建议
- 等待 **ChemPile** 正式发布（NeurIPS 2025）
- 重点合成有机化学机制（电子转移、反应路径）
- 分析化学"检测方法→原理→误差来源"流程

---

## 8. 经济学与计量经济学 (Economics & Econometrics)

### 核心数据集

| 数据集 | 规模 | 特点 | HuggingFace链接 |
|--------|------|------|-----------------|
| **BBT-Fin Corpus (中文)** | 大规模 | 中文金融经济语料 | https://huggingface.co/datasets/BBT-Fin/BBT-FinCorpus |
| **Fin-R1 Dataset** | 金融推理 | 金融推理+强化学习数据 | https://huggingface.co/datasets/Fin-R1 |
| **FinGPT Datasets** | 多任务 | 金融情感/关系/问答数据 | https://huggingface.co/FinGPT |
| **FinEval (中文)** | 评估级 | 中文金融知识评测 | https://huggingface.co/datasets/FinEval |
| **Finance-Instruct-500k** | 50万条 | 金融指令数据 | https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k |
| **AlphaFin** | 金融分析 | 股票分析+检索增强 | https://huggingface.co/datasets/AlphaFin |
| **FineWeb-Edu (Economics)** | 教育级 | 经济学教育内容 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu |

### 使用建议
- 中文必须补充 **BBT-Fin** 和 **FinEval**
- 构建"经济现象→数学建模→政策含义"三段论数据
- 合成计量经济学"假设检验+回归结果解读+稳健性检验"学术写作

---

## 9. 历史、古典汉语与文学 (History & Classical Chinese)

### 核心数据集

| 数据集 | 规模 | 特点 | HuggingFace链接 |
|--------|------|------|-----------------|
| **Daizhige (殆知阁)** | 33亿tokens | 最大古典中文语料（四库全书等） | 需通过古籍相关渠道获取 |
| **SikuBERT Corpus** | 高质量 | 四库全书语料 | https://huggingface.co/models?search=sikubert |
| **GuwenBERT Corpus** | 连续训练 | 基于Daizhige的古典中文 | https://huggingface.co/ethanyt/guwenbert-base |
| **WYWEB Benchmark** | 多任务 | 古典中文NLP评测基准 | https://github.com/kyrie-wyx/wyweb |
| **AncientDoc (字节)** | 多模态 | 中文古籍OCR+翻译+QA | https://huggingface.co/datasets/bytedance/AncientDoc |
| **FSPC (古诗情感)** | 古诗级 | 古诗细粒度情感标注 | https://huggingface.co/datasets/THUAIPoet/FSPC |
| **CCMP (古诗匹配)** | 对联级 | 中国古典诗歌匹配 | https://huggingface.co/datasets/ccmp |

### 使用建议
- **Daizhige** 是古典中文核心语料（33亿tokens）
- 合成"古文原文→现代翻译→历史背景→文学手法"四段式数据
- 构建编年体历史事件链（时间→地点→人物→因果→影响）

---

## 10. 会计与商业管理 (Accounting & Business)

### 核心数据集

| 数据集 | 规模 | 特点 | HuggingFace链接 |
|--------|------|------|-----------------|
| **FinBERT Corpus** | 金融财务 | TRC2财务新闻语料 | https://huggingface.co/ProsusAI/finbert |
| **SEC-BERT Corpus** | SEC级 | SEC文件报告语料 | https://huggingface.co/nlpaueb/sec-bert-base |
| **FLANG-BERT Corpus** | 多任务 | 金融NLP基准语料 | https://huggingface.co/SALT-NLP/FLANG-BERT |
| **BBT-Fin (中文)** | 大规模 | 中文金融商业语料 | https://huggingface.co/datasets/BBT-Fin/BBT-FinCorpus |
| **FinEval (中文)** | 评估级 | 含会计/管理/税务 | https://huggingface.co/datasets/FinEval |
| **FineWeb-Edu (Business)** | 教育级 | 商业教育内容 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu |

### 使用建议
- 基于中国会计准则和税法合成数据
- "经济业务描述→会计判断→分录编制→报表影响"决策链
- 针对管理会计（成本控制、预算管理）生成企业案例

---

## 通用高质量预训练语料（跨领域补充）

| 数据集 | 规模 | 特点 | HuggingFace链接 |
|--------|------|------|-----------------|
| **FineWeb** | 15T tokens | 最高质量网页语料 | https://huggingface.co/datasets/HuggingFaceFW/fineweb |
| **FineWeb-Edu** | 1.3T tokens | 教育级高质量子集 | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu |
| **Cosmopedia** | 3000万条 | Mixtral合成教育数据 | https://huggingface.co/datasets/HuggingFaceTB/cosmopedia |
| **SmolLM Corpus** | 多领域 | HuggingFace精选混合 | https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus |
| **DCLM** | 大规模 | 高质量Common Crawl | https://huggingface.co/datasets/mlfoundations/dclm |

---

## 数据合成策略总结

### 数学
- EntiGraph从教材合成CoT解题步骤
- 参考Nemotron-CC-Math 52B token方案

### 法律
- 法条-案例-解释三联体
- PDF-VL提取法律条文结构化知识

### 伦理
- 多智能体辩论生成
- 问题-反思-修正三元组

### 计算机系统
- 代码+系统原理解释混合
- 开源项目源码注释提取

### 医学
- 症状-诊断-治疗推理链
- 解剖学3D结构文本描述

### 物理/化学
- 实验→现象→理论→误差流程
- 概念物理机制解释

### 经济
- 经济→数学→政策三段论
- 计量经济学学术写作体

### 历史/文学
- 古文→翻译→背景→手法四段式
- 编年体事件链

### 会计
- 业务→判断→分录→报表决策链
- 中国企业案例

---

*最后更新: 2026-03-24*
