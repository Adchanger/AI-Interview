# 检索优化与 GraphRAG（RAG 八股 03）

> **更新时间**：2026-08-31

> **标签**：Rerank、HybridSearch、QueryRewrite、GraphRAG、RAG评测、面试八股

> **一句话**：RAG 效果差要先**分层定位**（解析/切分/嵌入/索引/召回/排序/上下文/生成），再对症下药——查询改写解决"问法与文本不匹配"，hybrid 解决"关键词/专有名词失配"，rerank 解决"召回到了但排不上去"，GraphRAG 解决"跨文档多跳与全局性问题"。

> **关联阅读**：[[/docs/rag/rag-basics.md]]、[[/docs/rag/vector-index-and-database.md]]、[[/docs/llm/long-context-and-flashattention.md]]

---

## 1. 分层定位：RAG 效果差先查哪里

| 症状 | 大概率环节 | 验证方法 |
|------|-----------|----------|
| 正确 chunk **根本不在候选池** | 解析/切分/嵌入/索引 | 直接用答案原文当 query 检索，若仍召回不到 → 入库或嵌入有问题 |
| 正确 chunk 在池里但排在 topK 之外 | 排序 | 扩大 k 观察是否命中；命中 → 加 rerank |
| topK 正确但答案错 | 上下文组装 / 生成 | 检查顺序、截断、指令约束、引用要求 |
| 答案引用了错误片段 | 上下文过长/冲突 | 减少 k、去重、显式标注来源编号 |
| 新旧版本同时出现 | 更新链路 | 检查软删除/版本元数据过滤 |
| 只有最终评分时无法判断 | 评测体系 | 分开评检索与生成（见 §5） |

> **面试标准答法**：先说"我会先建**失败样本集**，逐条标注是召回问题还是生成问题，再按上表定位"，比直接罗列优化技巧高一个层次。

---

## 2. 查询侧优化

| 技术 | 做法 | 解决什么 |
|------|------|----------|
| **Query Rewrite** | LLM 改写为检索友好的表述（补全指代、去口语） | 多轮对话中的省略与指代（"它多少钱"） |
| **Multi-Query / RAG-Fusion** | 生成多个不同角度的 query 并行检索后融合 | 单一表述召回不全 |
| **HyDE** | 先让 LLM 生成一个"假设答案"，用它的向量去检索 | 查询与文档**长度/风格不对称**（短问题 vs 长文档） |
| **Self-Query** | LLM 把自然语言解析成结构化过滤条件 + 语义 query | "2024 年之后的财报里…"这类含元数据约束的问题 |
| **Query 分解** | 复杂问题拆成子问题分别检索（multi-hop） | 需要多个事实组合的问题 |
| **Step-back prompting** | 先抽象成更一般的问题再检索 | 细节问题缺乏直接匹配 |
| **意图路由** | 分类到不同索引/工具（FAQ / 文档 / SQL / 计算） | 单一 RAG 处理不了的混合需求 |

**代价提醒**：每种改写都增加 LLM 调用与延迟，线上要按 query 复杂度路由，不要无脑全开。

---

## 3. 混合检索（Hybrid Search）

### 3.1 为什么需要

- **稠密向量**擅长语义相似，但对**精确 token**（型号 `A100-80G`、错误码 `E1024`、人名、缩写）容易失配；
- **BM25/倒排**擅长精确词匹配与稀有词（IDF 高），但不懂同义/语义。

$$\text{BM25}(q,d)=\sum_{t\in q}\mathrm{IDF}(t)\cdot\frac{f(t,d)\cdot(k_1+1)}{f(t,d)+k_1\big(1-b+b\frac{|d|}{\overline{|d|}}\big)}$$

（$k_1$ 控词频饱和，$b$ 控长度归一化）

### 3.2 融合方式

**① RRF（Reciprocal Rank Fusion）——首选**

$$\text{score}(d)=\sum_{i}\frac{1}{k + \mathrm{rank}_i(d)},\quad k\approx 60$$

只用**排名**不用分数 → 无需归一化、无需调权重、对不同检索器的分数尺度不敏感，工程上最稳。

**② 分数加权融合**：$\alpha\cdot\tilde s_{\text{vec}} + (1-\alpha)\cdot\tilde s_{\text{bm25}}$，需先做 min-max/z-score 归一化，$\alpha$ 要调且不同 query 类型最优值不同。

**③ 学习式稀疏（SPLADE）**：用模型产出带权重的稀疏向量，在一个倒排索引里同时具备语义扩展与精确匹配。

> 面试高频：**为什么推荐 RRF？** → 不同检索器分数不可比（余弦 0~1 vs BM25 无上界），归一化本身引入偏差；RRF 只依赖秩，鲁棒、无超参（仅 k），是工业默认。

---

## 4. Rerank 与上下文工程

### 4.1 为什么必须 rerank

- 向量检索是 **bi-encoder**：query 与 doc **独立编码**，交互只有一次点积 → 快但粗；
- **Cross-encoder rerank**：把 (query, doc) **拼在一起**过 Transformer，token 级充分交互 → 精度高得多但 $O(k)$ 次前向，只能用于少量候选。

**标准流程**：召回 50–200 条（多路 + RRF）→ cross-encoder rerank → 取 top 3–8 给 LLM。

**成本控制**：rerank 模型选小型（如 bge-reranker-base/large、Cohere Rerank），批处理 + 截断文档长度；也可用 **late-interaction**（ColBERT）在精度与速度间折中。

**微调 rerank 模型**：用业务点击/人工标注构造 (query, positive, hard negative)，hard negative 从"召回到但不相关"的样本里挖，收益通常大于换更大 LLM。

### 4.2 上下文组装（Context Engineering）

- **去重/合并**：相邻 chunk 合并、去掉重复段落；
- **排序**：把最相关的放**开头和结尾**（对抗 Lost in the Middle）；
- **压缩**：LLMLingua 类 prompt 压缩、按句抽取式压缩（注意压缩可能删掉关键细节，需评测）；
- **结构化**：给每段加编号 + 来源，要求模型引用编号，便于溯源与校验；
- **token 预算**：预留生成空间，宁可少给几条精准证据也不要塞满。

---

## 5. GraphRAG

### 5.1 解决什么

标准向量 RAG 的两个天花板：
1. **chunk 是信息孤岛**：实体关系跨越多个 chunk，向量相似度无法把它们"连起来"；
2. **全局性问题无法回答**："这份 500 页报告的核心主题有哪些？"——没有任何单个 chunk 与该问题语义相似。

> 面试高频：**向量相似度为什么不擅长多跳？** → 多跳问题的中间实体不在 query 里（"A 的老板的母校在哪"，query 与"老板的母校"那段文本相似度不高），语义相似只能一跳；需要显式的关系结构做遍历。

### 5.2 微软 GraphRAG 的做法

1. **实体与关系抽取**：用 LLM 从每个 chunk 抽 (实体, 关系, 描述)；
2. **图构建 + 实体消歧/合并**；
3. **社区检测**（Leiden 算法）把图分层聚类成社区；
4. **社区摘要**：LLM 为每个社区（自底向上分层）生成摘要；
5. **查询两条路**：
   - **Local search**（局部）：从 query 相关实体出发，取其邻居、关系与相关 chunk → 回答具体事实/多跳问题；
   - **Global search**（全局）：用所有（或选中层级）社区摘要做 map-reduce → 回答主题概括类问题。

### 5.3 成本与取舍

| 维度 | 说明 |
|------|------|
| **构建成本高** | 每个 chunk 都要 LLM 抽取 + 每个社区都要生成摘要，token 消耗是普通 RAG 建库的几十倍 |
| **更新成本高** | 新增文档可能改变图结构与社区划分，增量更新复杂 |
| **权限过滤难** | 图与社区摘要是跨文档聚合的产物，摘要里可能混入无权限内容 → 需按权限分图/分租户构图，或对摘要做权限标注 |
| **LightRAG / 轻量方案** | 只建实体-关系索引 + 双层检索（低层实体、高层主题），大幅降低构建与更新成本 |

**何时适合**：文档间关联密集（企业知识库、法律法规、科研文献、故障根因）、需要多跳与全局洞察、可承担离线成本。
**何时不适合**：FAQ/单文档问答、文档间几乎无关联、更新极频繁、成本敏感。

**成熟系统的答案是组合**：关键词（BM25）+ 稠密向量 + 多向量/ColBERT + 图检索，用路由决定走哪几路，再统一 RRF + rerank。

---

## 6. 评测：检索与生成分开看

| 层 | 指标 | 含义 |
|----|------|------|
| **检索** | **Recall@K** | 正确 chunk 是否进了前 K（RAG 的上限） |
| | **MRR** | 第一个正确结果排名的倒数均值 |
| | **NDCG@K** | 分级相关性 + 位置折损，最全面 |
| | Context Precision | 召回内容中有用比例（噪声度） |
| **生成** | **Faithfulness / Groundedness** | 答案是否**只**由给定上下文支撑（衡量幻觉） |
| | Answer Relevance | 答案是否切题 |
| | Answer Correctness | 与标准答案的一致性 |
| **系统** | 端到端准确率、拒答率、引用覆盖率、延迟/成本 | 上线看这些 |

**为什么必须分开**：端到端一个分数无法定位问题。Recall 低 → 优化检索；Recall 高但 Faithfulness 低 → 优化上下文组装与生成约束。

---

## 7. 手撕代码：RRF 融合 + rerank 管线

```python
def rrf_fuse(rank_lists, k=60, top_n=100):
    """rank_lists: List[List[doc_id]]，各检索器按相关性降序的结果"""
    scores = {}
    for lst in rank_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)[:top_n]

def retrieve_pipeline(query, vec_index, bm25_index, reranker,
                      k_recall=100, k_final=5):
    dense = vec_index.search(query, top_k=k_recall)      # [doc_id, ...]
    sparse = bm25_index.search(query, top_k=k_recall)
    fused = rrf_fuse([dense, sparse], top_n=k_recall)
    pairs = [(query, get_text(d)) for d in fused]
    scores = reranker.predict(pairs)                     # cross-encoder 精排
    order = sorted(range(len(fused)), key=lambda i: scores[i], reverse=True)
    return [fused[i] for i in order[:k_final]]
```

---

## 8. 面试高频问题速查

1. **召回率低怎么排查？** → 用答案原文当 query 反查定位是入库/嵌入问题还是排序问题，再逐层查解析→切分→嵌入→索引→召回。
2. **hybrid search 为什么有效？** → 稠密补语义、稀疏补精确词与稀有词，互补覆盖失配场景。
3. **多路结果怎么融合？** → 首选 RRF（只用秩，无需归一化）；加权融合需归一化且要调参。
4. **rerank 为什么比向量排序准？** → cross-encoder 让 query 与 doc 在 token 级交互，而 bi-encoder 只有一次点积。
5. **rerank 的成本怎么控？** → 限制候选数（50–200）、用小 reranker、截断长度、必要时用 ColBERT 式 late interaction。
6. **HyDE 适合什么场景？** → 短查询 vs 长文档的不对称问题；生成假设答案再检索。
7. **Query Rewrite 解决什么？** → 多轮指代省略、口语化、术语不一致。
8. **上下文压缩有风险吗？** → 有，可能删掉关键细节；需用 Faithfulness/端到端准确率评测把关。
9. **GraphRAG 解决什么？成本在哪？** → 多跳与全局主题问题；成本在 LLM 抽取实体关系与社区摘要，且更新与权限难。
10. **GraphRAG 的 local 与 global search 区别？** → local 从实体邻域取证据答具体问题；global 用社区摘要 map-reduce 答概括性问题。
11. **RAG 怎么评估？为什么分层？** → 检索用 Recall@K/MRR/NDCG，生成用 Faithfulness/Relevance；分层才能定位瓶颈。
12. **Faithfulness 怎么算？** → 把答案拆成论断，用 NLI/LLM 判断每条是否被上下文蕴含，取支持比例。
13. **知识库更新为什么不能只增不删？** → 旧版本仍会被召回，与新版冲突产生矛盾答案；需软删除 + 版本元数据过滤 + 灰度与回滚。

---

## 参考

- Cormack et al., *Reciprocal Rank Fusion*, SIGIR 2009
- Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*, arXiv:2212.10496
- Khattab & Zaharia, *ColBERT*, arXiv:2004.12832
- Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, arXiv:2404.16130
- Guo et al., *LightRAG: Simple and Fast Retrieval-Augmented Generation*, arXiv:2410.05779
- Es et al., *RAGAS*, arXiv:2309.15217
