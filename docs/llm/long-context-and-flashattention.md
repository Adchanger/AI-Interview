# 长上下文与 FlashAttention（LLM 八股 16）

> **更新时间**：2026-08-31

> **标签**：长上下文、FlashAttention、YaRN、稀疏注意力、LostInTheMiddle、面试八股

> **一句话**：长上下文有三个独立瓶颈——**算力（$O(n^2)$）、显存（KV Cache 线性增长）、外推能力（位置编码超出训练分布）**；FlashAttention 用分块 + online softmax 解决前者的 IO 问题（数学结果不变），YaRN/NTK 插值解决外推，稀疏/线性注意力与 KV 压缩解决规模，而"能读进去"不等于"用得好"（Lost in the Middle）。

> **关联阅读**：[[/docs/llm/positional-encoding.md]]、[[/docs/llm/kv-cache.md]]、[[/docs/llm/attention-variants-mha-mqa-gqa.md]]

---

## 1. 三个瓶颈拆解

| 瓶颈 | 表现 | 主要手段 |
|------|------|----------|
| **算力/显存中间量** | 注意力 $O(n^2d)$；朴素实现要物化 $n\times n$ 矩阵（32k 长度、单头就是 10 亿元素） | FlashAttention（精确）、稀疏/滑窗、线性注意力 |
| **KV Cache 显存** | 随 $b\times s$ 线性增长，长上下文时远超权重 | GQA/MLA、KV 量化、PagedAttention、Cache 淘汰 |
| **长度外推** | 超出训练长度后 PPL 暴涨 | RoPE 插值（NTK/YaRN/LongRoPE）、ALiBi、长上下文继续训练 |
| **有效利用** | 能输入但答不准（中间信息被忽略） | 检索/重排 + 上下文重排、提示工程、长上下文对齐训练 |

---

## 2. FlashAttention

### 2.1 核心思想

朴素注意力的致命问题不是 FLOPs，而是 **HBM 读写**：要把 $S=QK^\top$（$n\times n$）写回显存、读出做 softmax、再写回、再读出乘 $V$。FlashAttention 是 **IO-aware** 的精确实现：

1. **Tiling（分块）**：把 Q/K/V 切成能放进 SRAM 的块，在片上完成 $QK^\top\to\text{softmax}\to\times V$；
2. **Online softmax**：边遍历 K/V 块边维护当前最大值 $m$ 与指数和 $\ell$，用重缩放公式增量更新输出，**无需先看到整行**再归一化；
3. **反向重计算**：不保存 $n\times n$ 注意力矩阵，反向时用保存的 $m,\ell$ 重算（用少量额外计算换大量显存）。

**结果**：显存从 $O(n^2)$ 降到 $O(n)$，速度提升 2–4×（长序列更明显），且**数值上是精确注意力**（不是近似！这是高频陷阱点）。

### 2.2 版本演进

| 版本 | 关键改进 |
|------|----------|
| FlashAttention-1 (2022) | 分块 + online softmax + 重计算，IO 复杂度分析 |
| **FlashAttention-2** (2023) | 更好的 work partitioning（减少非矩阵乘操作、序列维并行、warp 级优化），A100 上约 2× 于 FA1 |
| **FlashAttention-3** (2024) | 面向 Hopper：warp-specialization、异步（TMA/WGMMA）、**FP8** 支持 |
| FlashDecoding / FlashDecoding++ | 针对 **decode 阶段**（Q 只有 1 行）在 KV 长度维切分并行，提升长上下文解码吞吐 |

> 面试高频：**FlashAttention 减少了计算量吗？** → 没有，FLOPs 基本不变（反向还略增）；它减少的是 HBM 访问量，属于 **memory-bound 优化**。**它也不减少 KV Cache**，与 GQA/MLA 正交。

### 2.3 online softmax 的数学

遍历第 $j$ 块后维护 $m_j=\max(m_{j-1}, \tilde m_j)$、$\ell_j = e^{m_{j-1}-m_j}\ell_{j-1} + e^{\tilde m_j - m_j}\tilde\ell_j$，输出同样按 $e^{m_{j-1}-m_j}$ 重缩放累加。核心就是"**换基准最大值时对已累积结果做统一缩放**"。

---

## 3. 长度外推

### 3.1 为什么直接外推会崩

RoPE 的旋转角 $\theta_i\cdot pos$ 在超过训练长度后进入未见过的相位区间，注意力 logits 分布漂移 → PPL 爆炸。

### 3.2 主流方案

| 方案 | 思路 | 是否需要训练 |
|------|------|-------------|
| **位置内插（PI）** | 把位置除以缩放因子 $s$，把新长度压回训练区间 | 少量微调 |
| **NTK-aware / Dynamic NTK** | 不均匀缩放：低频维少压、高频维多压，保留局部分辨率 | 可免训练 |
| **YaRN** | NTK-by-parts（按波长分组处理）+ 注意力温度修正，效果最好且训练成本低（论文报告约 1/10 token、1/2.5 步） | 少量微调 |
| **LongRoPE** | 搜索最优非均匀频率缩放 + 渐进扩展，可达 2M 级 | 需要微调 |
| **ALiBi** | 线性距离偏置，训短测长 | 训练时即采用 |
| **改 base（θ）** | 增大 RoPE base（如 1e4 → 1e6/5e5）后继续训练，LLaMA-3 等采用 | 需继续训练 |
| **StreamingLLM** | attention sink + 滑窗，支持"无限流式"但不真正利用全部历史 | 免训练 |

**工业标准流程**：先在 4k/8k 上做主预训练 → **长上下文继续预训练阶段**（改 RoPE base/插值 + 长文档数据，几十 B token）→ 长上下文 SFT（长文档 QA、多文档推理）→ 用 Needle-in-a-Haystack、RULER、LongBench 等评测。

> 面试高频：**为什么不直接用 128k 训练全程？** → 注意力 $O(n^2)$ 使长序列训练极贵，且长文档数据稀缺；分阶段（短序列打基础 + 长序列适配）性价比最高。

---

## 4. 稀疏 / 线性 / 新架构

| 类别 | 代表 | 要点 |
|------|------|------|
| **滑窗/局部** | Longformer、BigBird、Mistral SWA(4096) | 局部窗口 + 少量全局 token；层层叠加可获得较大有效感受野 |
| **块稀疏/可学稀疏** | Reformer(LSH)、Routing Transformer、**NSA**（DeepSeek 原生稀疏注意力） | 让模型自己选择要看的块，兼顾训练与推理效率 |
| **线性注意力** | Performer、Linear Attention、RWKV、RetNet | 用核近似去掉 softmax，复杂度 $O(n)$；长程精度常弱于全注意力 |
| **状态空间模型** | Mamba / Mamba-2、混合 Transformer-Mamba | 选择性 SSM，推理为常数状态、无 KV Cache 线性增长；混合架构在 2025 年后进入实用 |
| **压缩上下文** | 记忆 token、上下文蒸馏、prompt 压缩（LLMLingua） | 用更少 token 表达同样信息 |

---

## 5. 长上下文"能用"≠"用好"

### 5.1 Lost in the Middle（Liu et al. 2023）

给定长上下文，模型对**开头和结尾**的信息利用最好，**中间**明显下降，呈 U 形曲线。实践对策：
- 关键信息前置/后置；检索结果**按相关性重排**并把最相关的放两端；
- 控制上下文长度，不要"能塞就塞"；
- 用 rerank 精选 top-k 而非全塞（见 [[/docs/rag/retrieval-optimization-and-graphrag.md]]）。

### 5.2 长上下文 vs RAG

| 维度 | 长上下文 | RAG |
|------|----------|-----|
| 成本 | prefill 随长度线性（甚至平方）增长，很贵 | 只喂 top-k，成本可控 |
| 时效/更新 | 每次请求都要重新喂 | 索引更新即可 |
| 可解释/引用 | 弱 | 强（可给出来源） |
| 知识规模上限 | 受窗口限制 | 近乎无限 |
| 精度 | 中间信息易丢 | 取决于召回质量 |

**结论**：两者互补——RAG 负责"从海量里挑对的"，长上下文负责"把挑出来的读透"。当前主流工程是 **RAG + 中等长上下文 + rerank**。

---

## 6. 面试高频问题速查

1. **FlashAttention 为什么快？** → 分块 + online softmax 避免物化 $n\times n$ 矩阵，减少 HBM 读写；反向用重计算省显存。
2. **它是近似算法吗？** → 不是，结果与标准注意力精确一致。
3. **FlashAttention 能减小 KV Cache 吗？** → 不能，与 GQA/MLA 是正交的两类优化。
4. **online softmax 怎么工作？** → 维护运行最大值与指数和，换基准时对已累积输出统一重缩放。
5. **FA2/FA3 改了什么？** → FA2 优化并行与非矩阵乘开销；FA3 面向 Hopper 异步与 FP8。
6. **RoPE 直接外推为什么失败？** → 高频维相位进入训练未见区间，logits 分布漂移。
7. **YaRN 的核心思想？** → 按波长分组的非均匀频率插值 + 注意力温度校正，少量微调即可大幅外推。
8. **ALiBi 与 RoPE 插值怎么选？** → ALiBi 训短测长简单但超长精度弱；RoPE+YaRN 是当前主流。
9. **长上下文训练怎么做？** → 短序列主训 → 改 base/插值 + 长文档继续预训练 → 长上下文 SFT → NIAH/RULER 评测。
10. **Lost in the Middle 是什么？怎么缓解？** → 中间信息利用率低；关键信息放两端、rerank 精选、控制长度。
11. **Mamba 类模型解决了什么？** → 推理时常数状态，无 KV Cache 线性增长；但长程精确检索能力通常弱于全注意力，故多用混合架构。
12. **长上下文会取代 RAG 吗？** → 不会，成本、时效性、可引用性与知识规模决定两者互补。

---

## 参考

- Dao et al., *FlashAttention*, arXiv:2205.14135
- Dao, *FlashAttention-2*, arXiv:2307.08691
- Shah et al., *FlashAttention-3*, arXiv:2407.08608
- Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models*, arXiv:2309.00071
- Chen et al., *Extending Context Window via Position Interpolation*, arXiv:2306.15595
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, arXiv:2307.03172
- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752
