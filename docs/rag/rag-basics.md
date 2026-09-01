# RAG 基础：全链路与技术选型（RAG 八股 01）

> **更新时间**：2026-08-31

> **标签**：RAG、检索增强、微调对比、长上下文、面试八股

> **一句话**：RAG = 检索（把相关知识找出来）+ 增强（拼进 prompt）+ 生成（基于证据作答），用外部知识库解决 LLM 的知识过时、私有数据缺失与幻觉不可溯源问题；它与微调、长上下文不是替代关系而是互补。

> **关联阅读**：[[/docs/rag/vector-index-and-database.md]]、[[/docs/rag/retrieval-optimization-and-graphrag.md]]、[[/docs/llm/hallucination-and-evaluation.md]]

---

## 1. 为什么需要 RAG

LLM 的四个硬伤：
1. **知识截止**：训练后的新知识不知道；
2. **私有数据缺失**：企业内部文档不在训练集里；
3. **幻觉不可溯源**：说错了也没有出处；
4. **更新成本高**：改一条知识不可能重训模型。

RAG 把知识从"参数记忆"搬到"外部索引"，因此可**即时更新、可引用溯源、可做权限控制**。

---

## 2. 完整链路（面试要能一口气画出来）

```
【离线】文档 → 解析(parse) → 清洗 → 切分(chunk) → 元数据补全 → Embedding → 写入向量库/倒排索引
【在线】Query → 查询理解/改写 → 多路召回(向量+BM25+图) → 融合(RRF) → Rerank
        → 上下文组装(去重/压缩/排序) → Prompt → LLM 生成(带引用) → 校验/后处理 → 返回
```

**逐环节要点**：

| 环节 | 关键决策 | 常见坑 |
|------|----------|--------|
| 解析 | PDF/表格/图片走 OCR 或版面模型（如 layout 解析），保留标题层级 | 表格被拉平成乱码、页眉页脚污染 |
| 切分 | 大小、overlap、语义/结构边界 | 固定长度硬切割断句意，见 §3 |
| 元数据 | 标题路径、来源、页码、时间、权限标签 | 无元数据 → 无法过滤、无法溯源 |
| Embedding | 模型选型、维度、是否需要指令前缀、中英/领域适配 | 查询与文档编码方式不一致 |
| 索引 | HNSW/IVF-PQ、是否需要 hybrid | 见 [[/docs/rag/vector-index-and-database.md]] |
| 召回 | top-k 大小、多路策略 | 只用向量召回，关键词/编号类查询失败 |
| Rerank | cross-encoder 精排 | 不做 rerank，top-3 里没有正确答案 |
| 组装 | 顺序、压缩、token 预算 | Lost in the Middle、超长截断把答案截掉 |
| 生成 | 强制引用、允许拒答 | 不给"无法回答"出口 → 编答案 |

---

## 3. Chunking 策略

| 策略 | 说明 | 适用 |
|------|------|------|
| 固定长度 + overlap | 最简单（如 512 token，overlap 10–20%） | 通用兜底 |
| **按结构切**（标题/章节/Markdown 层级） | 保留语义完整性，附标题路径 | 技术文档、手册、法规 |
| 按句/段递归切（RecursiveCharacterTextSplitter） | 优先在段落→句子→字符层级切 | 通用首选 |
| **语义切分** | 用 embedding 相似度找语义边界 | 长文、叙事文本 |
| **父子/小到大**（small-to-big） | 用小块检索、返回父块给 LLM | 平衡召回精度与上下文完整性，非常实用 |
| 表格/代码特殊处理 | 表格转 markdown/CSV 并加表头描述；代码按函数切 | 结构化内容 |
| 上下文增强（contextual retrieval） | 给每个 chunk 加一段由 LLM 生成的"该块在全文中的位置/主题"说明再嵌入 | 显著提升召回，成本较高 |

**大小权衡**：块太小 → 语义不完整、上下文碎片化；块太大 → 噪声多、embedding 被稀释、token 浪费。经验起点 **300–800 token + 10–20% overlap**，再用失败样本集调。

---

## 4. RAG vs 微调 vs 长上下文（最高频的选型题）

| 需求 | 首选 | 理由 |
|------|------|------|
| 补充**事实知识**、需时效与溯源 | **RAG** | 索引更新即可，可引用、可做权限 |
| 改变**风格/格式/领域术语/输出结构** | **微调（SFT/LoRA）** | 这类"行为模式"靠提示不稳定 |
| 学习**新技能/新范式**（如特定工具调用规范） | 微调（+少量 RAG） | 需要参数层面固化 |
| 单文档深度理解（合同、长报告） | **长上下文** | 无需切分损失，整体推理 |
| 海量知识 + 频繁更新 + 要引用 | **RAG（+rerank）** | 长上下文成本与中间信息丢失不可接受 |
| 追求极限效果 | **组合**：RAG 提供证据 + 微调优化"如何使用证据" | 工业最终形态 |

> **标准答法**：「知识用 RAG，行为用微调，两者互补；长上下文解决的是'读得深'，RAG 解决的是'找得对'。」再补成本数据：RAG 只喂 top-k（几千 token），长上下文每次 prefill 十万 token，成本与延迟差一个量级。

---

## 5. 相似度与 Embedding 选型

**三种度量**：
- **余弦相似度**：只看方向，最常用（文本长度不同但主题相同应视为相似）；
- **内积（IP）**：兼顾方向与模长，**向量归一化后与余弦等价**；某些模型（如为 MIPS 训练的）要求用 IP；
- **欧氏距离（L2）**：归一化向量下与余弦单调等价（$\|a-b\|^2=2-2\cos$）。

**必须与 embedding 模型的训练目标一致**——用错度量是常见事故。

**Embedding 选型要点**：
- 看 **MTEB / C-MTEB 榜**但必须在**自己数据上重测**（榜单存在过拟合/污染）；
- 中文场景：BGE / GTE / Conan / Qwen-Embedding 等；多语言看专门的多语言版本；
- 是否需要**非对称指令前缀**（如查询加 `Represent this sentence for searching relevant passages:`），用错会掉点；
- 维度与成本：768/1024 常用；更大维度收益递减而索引成本线性增长；可用 **Matryoshka（MRL）** 模型做维度裁剪；
- **微调 embedding**：用业务内的 (query, positive, hard negatives) 三元组做对比学习，通常是 RAG 提升最大的单点投入之一。

---

## 6. 工程与治理

- **权限**：必须在**检索层**做过滤（metadata filter / 多租户隔离），不能靠提示词约束；
- **更新**：增量 upsert + 软删除；换 embedding 模型必须**全量重建索引**（向量空间变了）；chunk 策略变更同样要重建；
- **版本**：同一文档多版本要么只留最新、要么带版本元数据过滤，否则新旧同时召回互相矛盾；
- **缓存**：语义缓存（相似 query 直接复用答案）、前缀缓存（系统提示复用 prefill）；
- **成本**：召回 top-k 越大越贵；rerank 是 cross-encoder，延迟敏感需控候选数（常 50–100 进 rerank，出 3–8 条）；
- **可观测**：记录 query、召回 id 与分数、rerank 分、最终引用、用户反馈，构成**失败样本集**（RAG 优化的起点）。

---

## 7. RAG 范式演进

| 范式 | 特征 |
|------|------|
| Naive RAG | 一次检索 + 拼接生成 |
| Advanced RAG | 查询改写、hybrid 检索、rerank、上下文压缩 |
| Modular RAG | 路由、多索引、条件分支、循环 |
| **Agentic RAG** | LLM 自主决定是否检索、检索几次、用哪个工具（Self-RAG / ReAct 式），见 [[/docs/agent/agent-fundamentals.md]] |
| GraphRAG | 引入知识图谱处理多跳与全局问题，见 [[/docs/rag/retrieval-optimization-and-graphrag.md]] |

---

## 8. 面试高频问题速查

1. **RAG 的完整链路？** → 离线：解析→清洗→切分→元数据→嵌入→索引；在线：改写→多路召回→融合→rerank→组装→生成→校验。
2. **RAG 和微调怎么选？** → 知识/时效/溯源用 RAG，风格/格式/技能用微调，最终常组合。
3. **长上下文会取代 RAG 吗？** → 不会：成本、时效、可引用性、知识规模四点决定互补。
4. **chunk 怎么切？多大合适？** → 优先按结构/语义切 + overlap；起点 300–800 token，用失败样本调；父子块很实用。
5. **余弦、内积、欧氏怎么选？** → 与 embedding 训练目标一致；归一化后三者单调等价。
6. **embedding 模型怎么选？** → 榜单初筛 + 自有数据实测 + 注意指令前缀与维度成本；有数据就微调。
7. **换 embedding 模型要重建索引吗？** → 必须，向量空间不兼容。
8. **RAG 一定不会幻觉吗？** → 不是。检索错/不全、上下文冲突、模型忽略证据都会导致幻觉，需引用校验 + 拒答。
9. **Lost in the Middle 怎么应对？** → 精选 top-k、rerank、把最相关证据放首尾、控制上下文长度。
10. **权限怎么做？** → 检索层元数据过滤 + 多租户隔离，不能靠 prompt。
11. **RAG 怎么评估？** → 检索侧 Recall@K/MRR/NDCG，生成侧 Faithfulness/Answer Relevance，分层定位问题。
12. **什么是 Agentic RAG？** → 让模型自主决策检索时机与次数、可多轮迭代检索与反思。

---

## 参考

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, arXiv:2005.11401
- Gao et al., *Retrieval-Augmented Generation for Large Language Models: A Survey*, arXiv:2312.10997
- Asai et al., *Self-RAG*, arXiv:2310.11511
- Anthropic, *Introducing Contextual Retrieval*, 2024
- JavaGuide《RAG 面试题总结》与卡码笔记《RAG 大厂面试题汇总》（2026）
