# 幻觉与评测体系（LLM 八股 22）

> **更新时间**：2026-08-31

> **标签**：幻觉、评测、LLM-as-Judge、数据污染、面试八股

> **一句话**：幻觉源于"下一 token 概率最大化"与"事实正确"之间的目标错位（外加数据噪声、知识边界与对齐副作用），治理要在**数据、训练、推理、系统**四层同时下手；评测则要区分能力评测（benchmark）、对齐评测（人评/Judge）与线上评测（AB + 业务指标），并时刻警惕数据污染。

> **关联阅读**：[[/docs/rag/rag-basics.md]]、[[/docs/llm/rlhf-ppo-dpo.md]]、[[/docs/ml/model-evaluation-metrics.md]]

---

## 1. 幻觉的分类

| 分类维度 | 类型 | 例子 |
|----------|------|------|
| 与输入的关系 | **内在幻觉**（intrinsic） | 与给定上下文矛盾（摘要写出原文没有的数字） |
| | **外在幻觉**（extrinsic） | 上下文没提，模型自行编造（虚构文献、API） |
| 内容类型 | 事实性 | 人物、时间、数据错误 |
| | 引用性 | 编造论文/链接/条款（最容易被抓） |
| | 逻辑性 | 推理步骤自相矛盾 |
| | 指令性 | 不遵守格式/约束（其实是 instruction following 问题） |

---

## 2. 成因（必背五点）

1. **训练目标错位**：语言模型学的是 $P(x_t|x_{<t})$，**流畅 ≠ 正确**；没有"我不知道"的显式监督信号；
2. **数据问题**：语料含错误/过时/矛盾信息；长尾知识出现次数太少无法可靠记忆；
3. **知识边界不可知**：模型难以判断自己是否知道（校准不足），倾向"编一个像样的答案"；
4. **对齐副作用**：RLHF 偏好"有帮助、详细、自信"的回答 → 训练出**过度自信**与讨好倾向（sycophancy）；拒答会被打低分；
5. **解码随机性与长程漂移**：采样引入偏离；长生成中早期小错被后续内容"合理化"（snowballing）。

> 面试高频：**幻觉能彻底消除吗？** → 不能。理论上（如 Kalai & Vempala 等工作）在校准良好的生成模型中，对长尾事实存在不可避免的错误率下界；工程目标是**降低到可接受水平 + 可检测 + 可归因**，而非归零。

---

## 3. 缓解手段（四层）

### 3.1 数据层
- 高质量事实数据、去重与去噪；
- **加入"拒答/澄清"样本**：教模型在上下文不足时说"根据提供资料无法确定"；
- 知识时效标注、来源标注训练。

### 3.2 训练层
- SFT 阶段用 **grounded QA**（答案必须来自给定材料）与引用格式训练；
- 对齐阶段把"**诚实/可归因**"纳入奖励（拒答不再被惩罚）；
- **FactTune / RLAIF 事实性奖励**：用外部检索或一致性作为事实奖励信号；
- 减少 SFT 阶段"硬灌知识"（见 [[/docs/llm/sft-lora-peft.md]]）。

### 3.3 推理层
- **RAG**：把知识外置并要求带引用，是工业界最有效的单一手段；
- 降低温度、约束解码；
- **自一致性检查**：多次采样比对（SelfCheckGPT——同一问题多次生成不一致 → 很可能是幻觉）；
- **Chain-of-Verification（CoVe）**：先答 → 生成核查问题 → 独立回答核查问题 → 修订终答；
- 工具调用：计算器、代码执行、数据库/搜索，把"记忆"换成"查询"。

### 3.4 系统层
- **引用强制 + 可点击溯源**；答案中每个论断都要能定位到 chunk；
- **后置校验**：命名实体/数字与来源文本比对、规则校验、NLI 蕴含判断（答案是否被上下文支持）；
- **不确定性表达**：低置信度时降级为"给线索 + 建议人工核实"；
- 兜底与人工介入路径、幻觉率线上监控与用户反馈闭环。

---

## 4. 评测体系

### 4.1 三层评测

| 层次 | 内容 | 方法 |
|------|------|------|
| **能力评测** | 知识、推理、代码、长文本、多语言 | 公开 benchmark（MMLU/MMLU-Pro、GSM8K/MATH/AIME、HumanEval/LiveCodeBench、RULER、GPQA…） |
| **对齐/体验评测** | 有用性、指令遵循、安全、风格 | 人工评测、Arena（Elo/Bradley-Terry）、MT-Bench/AlpacaEval（LLM-as-Judge）、IFEval（可程序化校验的指令遵循） |
| **线上评测** | 业务效果 | AB 实验、任务完成率、人工抽检、用户反馈/投诉率、成本与延迟 |

### 4.2 常用自动指标及其局限

| 指标 | 用途 | 局限 |
|------|------|------|
| Perplexity | 语言建模质量 | 与下游能力弱相关；跨 tokenizer 不可比 |
| BLEU / ROUGE | 翻译、摘要 | 只看 n-gram 重叠，不认可合理改写 |
| BERTScore / BLEURT | 语义相似 | 依赖打分模型，仍不判事实 |
| Exact Match / Pass@k | QA、代码 | 需要可判定答案；EM 对格式敏感 |
| **RAG 专用**：Faithfulness、Answer Relevance、Context Precision/Recall（RAGAS 类） | RAG 分层诊断 | 多数由 LLM 判定，需校准 |

### 4.3 LLM-as-Judge（重点）

**两种形式**：pointwise 打分（1–10）与 **pairwise 比较**（更可靠，推荐）。

**已知偏差**（必须能说出 3 个以上）：
- **位置偏差**：偏爱靠前/靠后的答案 → 交换顺序做两次取一致结果；
- **长度偏差**：偏爱更长回答 → 控制长度或做长度校正；
- **自我偏好**：偏爱与自己风格相同/自己生成的答案 → 用多个不同家族的 judge 或人评校准；
- **格式偏差**：偏爱 markdown/列表/自信语气；
- **对细粒度正确性不敏感**：数值/事实错误可能被漂亮表述掩盖。

**改进做法**：给明确 rubric + few-shot 锚点、要求先给理由后给结论、多 judge 投票、与人评算一致性（Cohen's Kappa / Spearman）后再信任。

### 4.4 数据污染（Contamination）

- 现象：测试集出现在预训练语料 → 分数虚高；
- 检测：n-gram 重叠扫描、canary 字符串、成员推断、**对比"改写题面/换数字"后的掉分幅度**；
- 对策：用**新鲜/动态**评测集（LiveBench、LiveCodeBench、每月更新的竞赛题）、私有 holdout、多样化 prompt 模板、报告时说明去污染流程。

> 面试高频：**怎么评估一个对话助手模型？** → 分三层答：① 能力 benchmark 做基线体检（含代码/数学/长文本）；② 对齐评测用 pairwise 人评 + LLM-as-Judge（含偏差控制）+ IFEval 类可校验指令；③ 线上 AB 看任务完成率、人工抽检、投诉率与成本延迟。再补一句"必须有自建业务评测集，公开榜单只能参考"。

---

## 5. 手撕/伪代码：SelfCheck 一致性检测

```python
def selfcheck_hallucination(model, question, n=5, judge=None):
    """同一问题多次采样，一致性低 → 疑似幻觉（SelfCheckGPT 思路）"""
    main = model.generate(question, temperature=0.0)          # 待检答案
    samples = [model.generate(question, temperature=0.8) for _ in range(n)]
    # 用 NLI 或 judge 判断 main 中的每个论断是否被多数样本支持
    claims = split_into_claims(main)
    scores = []
    for c in claims:
        support = sum(judge.entails(s, c) for s in samples) / n
        scores.append(support)
    return {"answer": main,
            "risk": 1 - sum(scores) / max(len(scores), 1),     # 越高越可疑
            "claim_support": list(zip(claims, scores))}
```

---

## 6. 面试高频问题速查

1. **什么是幻觉？怎么分类？** → 生成看似合理但不真实/无依据的内容；分内在（与上下文矛盾）与外在（无据编造）。
2. **幻觉的根因？** → 目标错位（流畅≠正确）、数据噪声与长尾、知识边界不可知、RLHF 造成过度自信、解码随机与滚雪球。
3. **RAG 能消除幻觉吗？** → 不能。检索错/不全、上下文冲突、模型仍可能忽略证据；必须加引用校验与拒答机制。
4. **怎么让模型学会说"不知道"？** → 训练数据里加入拒答/澄清样本，对齐阶段不惩罚合理拒答，并用可归因奖励。
5. **CoVe 是什么？** → 先答、再自问核查问题、独立回答后修订，显著降低事实错误。
6. **SelfCheckGPT 的思想？** → 多次采样的一致性作为事实性代理，不需外部知识库。
7. **BLEU/ROUGE 为什么不够？** → 只看 n-gram 重叠，无法判断语义与事实正确性。
8. **LLM-as-Judge 有哪些偏差？** → 位置、长度、自我偏好、格式偏好、对细粒度错误不敏感。
9. **怎么让 Judge 更可靠？** → pairwise + 顺序交换、明确 rubric、先理由后结论、多 judge 投票、与人评算一致性。
10. **什么是数据污染？怎么检测？** → 测试集进了训练语料；用 n-gram 扫描、canary、改写题面掉分对比、动态评测集。
11. **PPL 低就一定好吗？** → 不一定，PPL 与下游能力弱相关且跨 tokenizer 不可比。
12. **RAG 系统怎么评？** → 检索侧（Recall@K、MRR、NDCG、Context Precision）与生成侧（Faithfulness、Answer Relevance）分开评，见 [[/docs/rag/retrieval-optimization-and-graphrag.md]]。
13. **上线后如何持续监控幻觉？** → 引用覆盖率、答案-证据蕴含校验通过率、用户点踩/投诉率、抽检人评、按主题切分的失败样本集。

---

## 参考

- Ji et al., *Survey of Hallucination in Natural Language Generation*, arXiv:2202.03629
- Huang et al., *A Survey on Hallucination in Large Language Models*, arXiv:2311.05232
- Manakul et al., *SelfCheckGPT*, arXiv:2303.08896
- Dhuliawala et al., *Chain-of-Verification Reduces Hallucination*, arXiv:2309.11495
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, arXiv:2306.05685
- Zhou et al., *Instruction-Following Evaluation for Large Language Models (IFEval)*, arXiv:2311.07911
- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation*, arXiv:2309.15217
