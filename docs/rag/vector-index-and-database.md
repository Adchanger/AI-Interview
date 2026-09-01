# 向量索引与向量数据库（RAG 八股 02）

> **更新时间**：2026-08-31

> **标签**：向量索引、HNSW、IVF-PQ、ANN、向量数据库、面试八股

> **一句话**：向量检索用 ANN（近似最近邻）在召回率与延迟之间换取数量级的速度提升——**HNSW** 用多层跳表式图做导航（高召回低延迟、内存大），**IVF** 用聚类缩小搜索空间，**PQ** 用乘积量化压缩向量（省内存、精度降），选型本质是"内存 × 延迟 × 召回 × 更新频率"的四维权衡。

> **关联阅读**：[[/docs/rag/rag-basics.md]]、[[/docs/rag/retrieval-optimization-and-graphrag.md]]、[[/docs/ml/unsupervised-clustering-and-dimensionality.md]]

---

## 1. 为什么要 ANN

精确最近邻（Flat / 暴力搜索）复杂度 $O(N\cdot d)$：1 亿条 768 维向量单次查询要扫 768 亿次乘加，不可接受。

**ANN 的核心交换**：牺牲一点召回率（如 95–99%），换取 10–1000× 的速度。

> 面试高频：**为什么可以接受不精确？** → ① 下游是 LLM 生成，top-k 里少一条边缘结果通常不影响答案；② embedding 本身就是语义近似，"精确的近似"意义有限；③ 后面还有 rerank 兜底。真正需要 100% 精确时（小数据、合规审计）就用 Flat。

---

## 2. 索引算法家族

### 2.1 Flat（暴力）

无损、无需训练、支持任意过滤，适合 < 10 万条或要求精确的场景。可用 SIMD/GPU 加速。

### 2.2 IVF（倒排文件，基于聚类）

1. **训练**：KMeans 把向量空间划成 `nlist` 个簇（聚类中心叫 centroid）；
2. **建库**：每个向量归入最近簇的倒排列表；
3. **查询**：只搜最近的 `nprobe` 个簇。

- 加速比约 $\text{nlist}/\text{nprobe}$；
- `nlist` 经验值 $\approx 4\sqrt{N}$ ~ $16\sqrt{N}$；`nprobe` 调大 → 召回↑延迟↑；
- **边界问题**：真正的最近邻可能落在没被搜到的邻簇里 → 靠增大 nprobe 缓解；
- **需要训练**，数据分布漂移后要重训；对高频插入不友好。

### 2.3 PQ（乘积量化）与 IVF-PQ

**PQ**：把 $d$ 维向量切成 $m$ 段，每段用 256（8bit）个聚类中心做码本 → 一个向量压成 $m$ 字节。

- 768 维 fp32（3072 B）→ $m=96$ 时仅 96 B，**压缩 32×**；
- 距离用**非对称距离计算（ADC）**：查询保持原始精度，与码本预计算的距离表查表求和，极快；
- 精度损失明显 → 常配 **re-ranking**：用 PQ 粗筛，再用原始向量（或 SQ8）对候选精算；
- **IVF-PQ** = 聚类缩小范围 + PQ 压缩内存，是十亿级规模的经典方案（Faiss `IVF4096,PQ64`）；
- **OPQ**：先做旋转让各子空间方差均衡，再 PQ，精度更好。

### 2.4 HNSW（分层可导航小世界图）

- 构建**多层近邻图**：上层稀疏（长边，快速跳跃）、底层稠密（短边，精细搜索），类似跳表；
- 查询：从顶层入口贪心走向更近的邻居，逐层下降，底层用大小为 `ef_search` 的候选队列做 beam 式搜索；
- **参数**：
  - `M`（每节点最大出边数，常 16–64）：越大召回越高、内存越大、构建越慢；
  - `ef_construction`（建图时候选队列，常 100–500）：越大图质量越好、建库越慢；
  - **`ef_search`**（查询候选队列，常 50–400）：**唯一在线可调的召回/延迟旋钮**，必须 ≥ top-k；
- 优点：**召回-延迟曲线最好**、支持增量插入、无需训练；
- 缺点：**内存大**（向量 + 图结构，约 $N\times M\times$ 指针开销）、删除是软删除需定期重建、构建慢。

### 2.5 其他

| 索引 | 说明 |
|------|------|
| **ScaNN** | 各向异性量化 + 分区，Google 方案，量化下召回优秀 |
| **DiskANN / SPANN** | 图索引落盘（SSD），支持超大规模、内存友好，延迟略高 |
| LSH | 局部敏感哈希，理论优雅但实际召回/内存效率不如图索引，现较少用 |
| **稀疏向量索引** | BM25/倒排、SPLADE（学习式稀疏），做 hybrid 必备 |
| 二值/SQ 量化 | 标量量化（fp32→int8，4×）几乎无损，**性价比最高的第一步优化**；二值量化 32× 但需 rerank |

### 2.6 选型对照表

| 索引 | 召回 | 延迟 | 内存 | 增量更新 | 需训练 | 适用规模 |
|------|------|------|------|----------|--------|----------|
| Flat | 100% | 慢 | 大（原始） | 好 | 否 | <10 万 / 要求精确 |
| IVF-Flat | 高 | 中 | 大 | 一般 | 是 | 百万级 |
| **HNSW** | **最高** | **最快** | **最大** | 好 | 否 | 百万~千万（内存够） |
| IVF-PQ | 中 | 快 | **最小** | 一般 | 是 | 亿级 |
| DiskANN | 高 | 中 | 小（SSD） | 一般 | 是 | 十亿级 |

---

## 3. 向量数据库

### 3.1 与传统数据库的差异

| 维度 | 传统 DB | 向量 DB |
|------|---------|---------|
| 查询语义 | 精确匹配 / 范围 | **相似度 top-k（近似）** |
| 索引 | B+ 树、哈希、倒排 | HNSW / IVF-PQ / 图索引 |
| 结果 | 确定 | 近似，含召回率概念 |
| 核心难点 | 事务一致性 | 高维距离计算 + 内存/召回权衡 + **带过滤的 ANN** |

### 3.2 vector index / vector DB / vector plugin 的区别

- **index（库）**：Faiss、hnswlib、ScaNN —— 只有索引算法，无持久化/分布式/元数据；
- **原生向量 DB**：Milvus、Qdrant、Weaviate、Vespa —— 分布式、持久化、过滤、多租户、混合检索；
- **插件/扩展**：pgvector、Elasticsearch/OpenSearch kNN、Redis Search、ClickHouse —— 复用已有技术栈，运维简单。

**选型建议**：
- 数据 < 百万、已有 PostgreSQL → **pgvector**（一个库搞定业务数据 + 向量 + 事务 + 权限），最省事；
- 需要重度过滤 + 高 QPS + 亿级 → **Milvus / Qdrant / Vespa**；
- 已重度使用 ES 且需强 BM25 混合 → **ES kNN**；
- 何时该换专业向量库：单机内存/QPS 打满、需要分片与副本、需要按标签强过滤且召回下降明显、需要多索引与在线重建。

### 3.3 过滤（Filtering）的坑（高频）

| 方式 | 问题 |
|------|------|
| **Post-filter**（先 ANN 再过滤） | 过滤条件很严时，top-k 全被过滤掉 → 结果为空/不足 |
| **Pre-filter**（先过滤再暴力搜） | 候选集大时退化成暴力扫描，慢 |
| **Filtered ANN**（图/倒排内部感知过滤） | 现代向量库的做法（如带 label 的 HNSW 遍历、分区 + bitmap），但**高选择性过滤下召回会下降**，需要放大 ef/nprobe 补偿 |

**实践**：高基数强过滤（如 user_id）优先做**物理分区/多集合**，而不是靠标量过滤。

### 3.4 规模化（100 万 → 1 亿的架构变化）

1. 单机 HNSW（内存足够）→ 加 **SQ8 量化** 减内存；
2. 分片（按 tenant/时间/hash）+ 副本，查询做 scatter-gather 合并；
3. 换 **IVF-PQ / DiskANN** 控内存成本；
4. 冷热分层：热数据内存 HNSW、冷数据磁盘索引；
5. 离线批量建索引 + 在线增量 buffer（小的 Flat 段定期 merge）；
6. 引入 rerank 抵消粗召回精度损失；
7. 监控召回率（用 Flat 结果做 ground truth 抽样评估）与 p99 延迟。

---

## 4. 手撕代码：HNSW 单层搜索（贪心 + beam）

```python
import heapq

def search_layer(graph, vectors, query, entry_points, ef, dist):
    """graph: {node: [neighbors]}；返回 ef 个最近候选（简化版单层搜索）"""
    visited = set(entry_points)
    # candidates: 小顶堆（按距离升序，取最近的先扩展）
    candidates = [(dist(query, vectors[p]), p) for p in entry_points]
    heapq.heapify(candidates)
    # results: 大顶堆（用负距离模拟，堆顶是当前最差的结果）
    results = [(-d, p) for d, p in candidates]
    heapq.heapify(results)

    while candidates:
        d, c = heapq.heappop(candidates)
        worst = -results[0][0]
        if d > worst and len(results) >= ef:
            break                              # 最近候选都比现有最差结果远 → 停止
        for nb in graph.get(c, []):
            if nb in visited:
                continue
            visited.add(nb)
            dn = dist(query, vectors[nb])
            if len(results) < ef or dn < -results[0][0]:
                heapq.heappush(candidates, (dn, nb))
                heapq.heappush(results, (-dn, nb))
                if len(results) > ef:
                    heapq.heappop(results)     # 淘汰最差
    return sorted((-d, p) for d, p in results)
```

---

## 5. 面试高频问题速查

1. **为什么用 ANN 而不是精确检索？** → 精确 $O(Nd)$ 不可行；牺牲少量召回换数量级加速，下游有 rerank 兜底。
2. **HNSW 的原理？** → 多层近邻图，上层长边跳跃、底层精搜，贪心 + beam（ef）搜索。
3. **`ef_search` 怎么调？** → 在线召回/延迟旋钮，调大召回升延迟升；必须 ≥ top-k；用抽样 ground truth 找拐点。
4. **`M` 和 `ef_construction` 的影响？** → 图连通度与建图质量，越大越准但内存/建库成本升，且不可在线改。
5. **IVF 的原理与参数？** → KMeans 分簇 + 只搜 nprobe 个簇；nlist 约 $4\sim16\sqrt N$，nprobe 控召回。
6. **PQ 怎么压缩？精度怎么补？** → 分段码本，一段一字节；用 ADC 查表算距离，再用原始向量对候选 rerank。
7. **HNSW 与 IVF-PQ 怎么选？** → 内存充足、要高召回低延迟选 HNSW；亿级、内存受限选 IVF-PQ/DiskANN。
8. **向量库与传统库的核心差别？** → 近似 top-k 相似检索 vs 精确匹配，索引结构与"带过滤 ANN"是难点。
9. **过滤为什么会降召回？怎么办？** → 图/簇的连通性被过滤破坏；对策：filtered-ANN、放大 ef/nprobe、高基数字段改用物理分区。
10. **pgvector 什么时候不够用？** → 数据量/QPS 打满单机、需分片副本、需要复杂过滤与在线重建时。
11. **数据从 100 万涨到 1 亿要做什么？** → 量化降内存、分片 + 副本、换 IVF-PQ/DiskANN、冷热分层、离线建库 + 增量段、加 rerank、监控召回。
12. **怎么评估 ANN 召回率？** → 抽样查询用 Flat 算 ground truth，计算 Recall@K，并与 p99 延迟一起画取舍曲线。
13. **要不要用标量量化？** → fp32→int8 几乎无损、内存降 4×，通常是第一步该做的优化。

---

## 参考

- Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using HNSW*, arXiv:1603.09320
- Jégou et al., *Product Quantization for Nearest Neighbor Search*, TPAMI 2011
- Johnson et al., *Billion-scale similarity search with GPUs (Faiss)*, arXiv:1702.08734
- Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN)*, ICML 2020
- Subramanya et al., *DiskANN*, NeurIPS 2019
- Milvus / Qdrant / pgvector 官方文档
