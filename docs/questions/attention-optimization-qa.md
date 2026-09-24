# 注意力优化面试题汇总（FlashAttention / MQA / GQA / MLA / KV Cache）

> **更新时间**：2026-09-04

> **标签**：面试题汇总、FlashAttention、MQA、GQA、MLA、KV Cache

> **一句话**：注意力优化是大模型面试的"计算题重灾区"——本文把 18 道高频题按「基础复杂度 → FlashAttention → KV Cache 与注意力变体 → 稀疏/线性注意力 → 手撕」组织，每题给出 30 秒速答 + 解析 + 面试官追问，公式与显存数字均可白板复现。

> **关联阅读**：[[/docs/llm/attention-variants-mha-mqa-gqa.md]]、[[/docs/llm/kv-cache.md]]、[[/docs/llm/long-context-and-flashattention.md]]、[[/docs/llm/mla-multi-head-latent-attention.md]]

---

## 0. 怎么用这份题集

1. **先分清"优化的是哪一层"**：注意力优化有四条独立主线——**IO 效率**（FlashAttention）、**KV Cache 体积**（MQA/GQA/MLA/量化/淘汰）、**渐近复杂度**（稀疏/线性注意力）、**显存管理**（PagedAttention）。面试官最常挖的坑就是让你混淆它们（经典陷阱："FlashAttention 能省 KV Cache 吗？"）。
2. **公式必须能推**：online softmax、KV Cache 显存公式、GQA 的组数换算，这三组是白板计算题源头。
3. **自测 → 查漏 → 补缺**：每题先口述 30 秒速答，卡壳点通过「关联」跳回知识点文档系统补。

> 频率标注：⭐⭐⭐ 几乎必问 ｜ ⭐⭐ 高频 ｜ ⭐ 进阶（答出来是加分项）

---

## 一、注意力基础与复杂度

### Q1. 注意力为什么要除以 √d_k？⭐⭐⭐

> **速答**：假设 q、k 各分量独立、零均值、方差为 1，则点积 $q\cdot k$ 的方差随 $d_k$ **线性增长**（方差 = $d_k$）。$d_k=128$ 时 logits 标准差约 11，softmax 直接进入饱和区——梯度接近 0，训练崩。除以 $\sqrt{d_k}$ 把方差归一到 1，让 softmax 工作在梯度健康的区间。

**追问**：
- 不除会怎样？→ softmax 输出接近 one-hot，反向传播时 $\partial\,\text{softmax}/\partial z \approx 0$，注意力权重学不动。
- 有其他解法吗？→ 初始化时把 $W^Q/W^K$ 缩小 $\sqrt{d_k}$ 倍、QK-Norm（对 Q/K 做归一化）都能达到类似效果；现代模型（如 Qwen3、OLMo2 的一些配置）开始用 QK-Norm 稳定训练。

**关联**：[[/docs/llm/transformer-principle.md]]

### Q2. 自注意力的时间和空间复杂度是多少？⭐⭐⭐

> **速答**：设序列长 $n$、隐维 $d$——计算量 $O(n^2 d)$（$QK^\top$ 与 $\mathrm{Attn}\cdot V$ 各占一半）；朴素实现的中间显存 $O(n^2)$（要物化 $n\times n$ 的分数矩阵和概率矩阵）。长序列场景 $n^2$ 主导一切：$n=32\mathrm{k}$ 时单头的分数矩阵就有 10 亿元素。

**追问**：
- 和 FFN 比呢？→ FFN 是 $O(n d^2)$（逐位置）；短序列时 FFN 更贵，$n$ 超过 $d$ 量级后注意力反超——所以"长上下文优化"基本等于"注意力优化"。
- 显存只有 $O(n^2)$ 这部分吗？→ 还有 KV Cache（推理侧，$O(n)$ 每 token 每层，随 batch 和长度线性涨），见 Q9。

**关联**：[[/docs/llm/long-context-and-flashattention.md]] §1

### Q3. softmax 为什么要做数值稳定处理？怎么做的？⭐⭐

> **速答**：直接算 $e^{z_i}$，当 $z_i$ 大时上溢（float32 约 $z>88$ 即 inf）。利用 softmax 的平移不变性，先减行最大值：$\mathrm{softmax}(z)_i = e^{z_i - m}/\sum_j e^{z_j - m}$，数学结果不变但指数恒 $\le 1$，不溢出。

**解析**：标准的 safe softmax 需要**三趟**遍历（求 max → 求和 → 归一化），每趟都要读写一次数据——这正是 FlashAttention 用 online softmax 把它压成一趟的动机（见 Q6）。

**关联**：[[/docs/llm/long-context-and-flashattention.md]] §2.3

---

## 二、FlashAttention

### Q4. 朴素注意力的瓶颈到底在哪？（为什么不是 FLOPs）⭐⭐⭐

> **速答**：瓶颈是 **HBM（显存）读写次数**，不是计算量。朴素实现要把 $S=QK^\top$（$n\times n$）写回 HBM、读出来做 softmax、写回 $P$、再读出来乘 $V$——$O(n^2)$ 量级的显存搬运。GPU 上片上 SRAM 比 HBM 快一个数量级以上但容量只有几十~几百 KB，注意力在长序列下是典型 **memory-bound** 算子：算力空转，时间在等数据。

**解析**：判断一个算子是 compute-bound 还是 memory-bound 看**算术强度**（FLOPs / 字节数）。逐元素操作（softmax、dropout、mask）算术强度极低，而朴素注意力里这类操作全作用在 $n\times n$ 矩阵上。

**追问**：为什么矩阵乘不慢？→ 矩阵乘算术强度高（$O(n)$ FLOPs/字节），Tensor Core 能跑满；慢的是中间结果在 HBM 和 SRAM 之间的来回搬运。

**关联**：[[/docs/llm/long-context-and-flashattention.md]] §2.1

### Q5. FlashAttention 的原理是什么？⭐⭐⭐

> **速答**：三招——①**Tiling（分块）**：把 Q/K/V 切成能塞进 SRAM 的小块，在片上完成 $QK^\top \to \text{softmax} \to \times V$ 全流程，不物化 $n\times n$ 矩阵；②**Online softmax**：边扫 K/V 块边维护运行最大值 $m$ 与指数和 $\ell$，换基准时对已累积输出统一重缩放，一趟算完精确 softmax；③**反向重计算**：反向不存 $n\times n$ 矩阵，用保存的 $m,\ell$ 现场重算（少量计算换大量显存）。结果：显存 $O(n^2)\!\to\!O(n)$，训练速度 2–4×，且**数值上是精确注意力**。

**追问**（三大经典陷阱）：
- 它是近似算法吗？→ **不是**，结果与标准注意力精确一致（仅浮点误差级差异）。
- 它减少计算量了吗？→ 没有，FLOPs 基本不变（反向重计算还略增）；它减少的是 HBM 访问量。
- 它能减小 KV Cache 吗？→ **不能**，KV Cache 是该省的还得靠 GQA/MLA/量化（见 Q9–Q14），两者正交可叠加。

**关联**：[[/docs/llm/long-context-and-flashattention.md]] §2

### Q6. online softmax 的公式能推一下吗？⭐⭐⭐

> **速答**：扫到第 $j$ 块时，局部最大值 $\tilde m_j$、全局新最大值 $m_j = \max(m_{j-1}, \tilde m_j)$；指数和与输出都按"换基准"统一重缩放：

$$\ell_j = e^{\,m_{j-1}-m_j}\,\ell_{j-1} + \textstyle\sum_{x\in\text{块}j} e^{\,x-m_j},\qquad O_j = e^{\,m_{j-1}-m_j}\,O_{j-1} + \tilde P_j V_j$$

> 核心就一句话：**每次基准最大值变大时，把之前累积的结果乘 $e^{m_{\text{旧}}-m_{\text{新}}}$ 折算到新基准下**。最后输出 $O / \ell$。

**追问**：
- 为什么要从三趟变一趟？→ 省掉对 $n\times n$ 矩阵的两次 HBM 读写；更重要的是分块后根本"看不到整行"，只能增量式维护。
- 除法在哪做？→ 归一化（除以 $\ell$）推迟到最后一次，中间全程只维护未归一化的累积量。

**关联**：[[/docs/llm/long-context-and-flashattention.md]] §2.3

### Q7. FlashAttention-1/2/3 各改了什么？⭐⭐

> **速答**：FA1 确立分块 + online softmax + 重计算的 IO 框架；**FA2** 减少非矩阵乘开销（更少 warp 间通信、softmax 重算优化）并在**序列维并行**（不同 Q 块分给不同 SM），A100 上约 2× FA1；**FA3** 面向 Hopper 架构：warp-specialization 生产者/消费者、TMA + WGMMA 异步重叠计算与搬运、支持 FP8。

**追问**：FlashDecoding 是什么？→ decode 阶段 Q 只有 1 个 token，batch×heads 并行度喂不满 GPU；FlashDecoding(++) 改在 **KV 长度维**切分并行，最后归约合并，专打长上下文解码。

**关联**：[[/docs/llm/long-context-and-flashattention.md]] §2.2

---

## 三、KV Cache 与注意力变体

### Q8. 为什么说 decode 阶段是带宽瓶颈，prefill 不是？⭐⭐⭐

> **速答**：自回归 decode 每步只算 1 个 token，矩阵乘退化为矩阵-向量乘，**算术强度极低**；但每步都要把全部权重 + 全部 KV Cache 从 HBM 完整读一遍。耗时 ≈（权重字节 + KV Cache 字节）/ 带宽——**减小 KV Cache ≈ 直接提升吞吐**。prefill 则是一次性并行算完整个 prompt，矩阵乘计算密度高，是 compute-bound。

**解析**：这是 MQA/GQA/MLA/KV 量化/PagedAttention 一整条优化线的第一性原理，面试时一定要先讲清这个再展开。

**追问**：这对 batch 策略有什么启示？→ decode 时增大 batch 可以摊薄权重读取（权重只读一次喂多个请求），所以 continuous batching 是推理引擎的核心；但 batch 越大 KV Cache 越大，又回到显存问题。

**关联**：[[/docs/llm/attention-variants-mha-mqa-gqa.md]] §1、[[/docs/llm/kv-cache.md]]

### Q9. KV Cache 显存怎么算？（必考计算题）⭐⭐⭐

> **速答**：每 token 每层缓存 K 和 V 两份，所以总字节数 $= 2 \times L_{\text{层}} \times n_{kv} \times d_{\text{head}} \times b \times s \times \text{dtype字节}$。例：7B 模型（$L=32, n_{kv}=32, d_{\text{head}}=128$, fp16）每 token ≈ 0.5 MB；$b=1, s=8\mathrm{k}$ 时约 **4 GB**，已经和小模型的权重同级。

**解析**：心算模板——$2\times32\times32\times128\times2\ \text{B} = 524288\ \text{B} = 0.5\ \text{MB/token}$，再乘 $b\times s$。注意 $n_{kv}$ 是 **KV 头数**：GQA-8 就把 0.5 MB 压到 0.125 MB。

**追问**：
- 为什么说长上下文时 KV Cache 超过权重？→ 它随 $b\times s$ 线性增长：$b=32, s=32\mathrm{k}$ 时上例达 512 GB，远超 7B 权重的 14 GB。
- 怎么省？→ GQA/MLA（减 $n_{kv}$）、KV 量化（减字节）、淘汰/滑窗（减 $s$）、分页管理（减浪费），见 Q10–Q14、Q17。

**关联**：[[/docs/llm/kv-cache.md]]

### Q10. MQA 和 GQA 分别怎么做的？折中了什么？⭐⭐⭐

> **速答**：MQA 让所有 Q 头共享**同一份** K/V（1 个 KV 头），Cache 降 $h$ 倍但质量明显下降、训练不稳；GQA 把 $h$ 个 Q 头分 $g$ 组、每组共享一份 K/V，$g=h$ 退化为 MHA、$g=1$ 退化为 MQA。工业默认 $g=8$（LLaMA-2 70B：64 Q 头 / 8 KV 头）：质量近似无损、Cache 降 8 倍、恰好对齐 TP=8（每卡 1 个 KV 头，免跨卡复制）。

**追问**：
- 已有 MHA 模型能转 GQA 吗？→ 能，GQA 论文给出 uptrain 方案：组内 KV 头均值池化初始化，再用约 5% 预训练算力继续训练。
- KV 头数为什么不取 2 或 4？→ 质量/显存/并行度的平衡点：$g=8$ 时 Cache 已降一个量级，再往下质量损失开始显现；且 8 对齐常见 TP 度。

**关联**：[[/docs/llm/attention-variants-mha-mqa-gqa.md]] §3–4

### Q11. MLA 的原理是什么？和 GQA 的本质差别在哪？⭐⭐⭐

> **速答**：思路完全不同——GQA 是"**减少 KV 头数**"，MLA 是"**把 KV 低秩压缩成一个隐向量**" $c_t^{KV}$（DeepSeek-V2 为 512 维，远小于 $h\cdot d_k$），推理只缓存隐向量，计算时用"矩阵吸收"把上投影吸进 Q 侧，避免真正恢复完整 K/V。Cache 降到 MHA 的约 1/20 量级，而 DeepSeek 报告效果**反而优于 MHA**。

![MHA/GQA/MQA/MLA 对比](../images/mla-comparison-01.png)

图1：MHA / GQA / MQA / MLA 结构对比（来源：DeepSeek-V2 技术报告，arXiv:2405.04434）

**追问**：
- 为什么压缩了还能效果更好？→ 低秩 bottleneck 迫使 KV 表示更紧凑（类正则作用），且所有头共享一个大隐向量，比把表示切碎到几十个小头里表达更充分；代价是训练与算子实现更复杂。
- MLA 只有 DeepSeek 在用吗？→ 由 DeepSeek-V2 提出并沿用至 V3；其"低秩压缩 KV"思想影响了后续一批 KV 压缩工作。

**关联**：[[/docs/llm/mla-multi-head-latent-attention.md]]、[[/docs/llm/attention-variants-mha-mqa-gqa.md]] §5

### Q12. MLA 为什么要"解耦 RoPE"？⭐⭐

> **速答**：矩阵吸收要求 K 能写成 $W^{UK}c_t$ 的形式、把 $W^{UK}$ 折进 Q 侧，但 RoPE 旋转与低秩投影**不可交换**（$R\,W\,c \neq W'\,c$）——位置信息会卡在压缩矩阵里吸不动。DeepSeek 的解法：K 拆两半，压缩部分不带位置，另设一个独立的、携带 RoPE 的共享键 $k_t^R$（64 维）单独缓存，Q 侧同样拼接触带位置的分量。

**追问**：代价是什么？→ 每 token 每层多缓存 64 维；Q/K 计算多一次拼接，算子复杂度上升——这也是"MLA Cache 极小但不是零"的原因。

**关联**：[[/docs/llm/mla-multi-head-latent-attention.md]]、[[/docs/questions/positional-encoding-qa.md]]

### Q13. FlashAttention、GQA、MLA、PagedAttention 能一起用吗？⭐⭐⭐

> **速答**：能，四者**正交**：FlashAttention 改 IO（不改数学、不减 Cache）、GQA/MLA 改架构（减 Cache）、PagedAttention 改显存管理（分页消碎片）。当前工业标配就是组合拳：**RoPE + GQA + FlashAttention + PagedAttention + KV 量化**。

**解析**（高频分类题）：

| 优化 | 作用层 | 需要训练时确定？ | 减 KV Cache？ |
|------|--------|------------------|---------------|
| FlashAttention | 算子 IO | 否（推理侧可开关） | 否 |
| MQA/GQA/MLA | 模型架构 | **是**（权重结构相关） | 是 |
| KV 量化 | 存储格式 | 否（PTQ） | 是（2–4×） |
| PagedAttention | 显存管理 | 否（推理引擎） | 否（减的是碎片浪费） |

**追问**：面试让你给一个"长上下文高并发"场景开优化处方 → 架构用 GQA/MLA + 推理引擎开 PagedAttention/continuous batching + FA + KV int8 + 前缀缓存（RadixAttention），超长再上滑窗/稀疏。

**关联**：[[/docs/llm/attention-variants-mha-mqa-gqa.md]] §7、[[/docs/engineering/inference-serving-optimization.md]]

### Q14. 除了改头数，KV Cache 还能怎么省？⭐⭐

> **速答**：三条路——①**量化**：KV int8/fp8，再省 2–4×，对 decode 带宽直接生效；②**淘汰/滑窗**：H2O 按累计注意力分数只留"重 token"，StreamingLLM 发现 **attention sink** 现象（注意力大量汇聚在开头几个 token），保留 sink + 滑窗即可流式推理；③**跨层/跨请求共享**：CLA 相邻层共享 KV，YOCO 只算一遍全局 KV，Prefix/Radix 缓存复用公共前缀。

**追问**：attention sink 是什么？→ 模型倾向把大量注意力分数投给序列开头少数 token（即使语义不重要），它起到"softmax 归一化的垃圾桶/锚点"作用；直接滑窗丢掉开头 token 会崩，留住 sink 就稳——这是 StreamingLLM 的核心发现。

**关联**：[[/docs/llm/kv-cache.md]]、[[/docs/llm/long-context-and-flashattention.md]]

---

## 四、稀疏注意力与线性注意力

### Q15. 稀疏注意力的代表方案与核心思想？⭐⭐

> **速答**：核心思想是"大多数 token 对本来就不需要互相看"，用结构化稀疏把 $O(n^2)$ 降下来：固定模式的**滑窗+全局**（Longformer、BigBird、Mistral SWA-4096）、可学习的**路由/哈希**（Reformer LSH、Routing Transformer）、以及 DeepSeek 的 **NSA**（原生可训稀疏：压缩粗看 + 块选择细看 + 滑窗保底三分支）。代价是丢掉部分全连接表达力，所以滑窗常配合"层层叠加扩大有效感受野"。

**追问**：滑窗注意力的有效上下文只有窗口大小吗？→ 不是。$L$ 层叠加后信息可传播 $L\times w$ 的距离（类似 CNN 感受野），但精确的长程检索能力仍弱于全注意力。

**关联**：[[/docs/llm/long-context-and-flashattention.md]] §4

### Q16. 线性注意力怎么做？付出了什么代价？⭐⭐

> **速答**：用核函数近似把 softmax 拆开：$\phi(Q)(\phi(K)^\top V)$——先算 $\phi(K)^\top V$（$d\times d$），复杂度从 $O(n^2 d)$ 降到 $O(n d^2)$，且可以写成 RNN 式递推（常数状态、无 KV Cache 增长）。代表：Performer、Linear Attention、RetNet；SSM 一系（Mamba/Mamba-2）可视为同一思想的状态空间版本。代价：**长程精确检索弱**——固定大小的状态本质是有限容量的联想记忆，"大海捞针"类任务明显弱于全注意力，所以 2025 年后实用方案多是 **混合架构**（全注意力层 + 线性/SSM 层混排）。

**追问**：线性注意力能用于 decode 提速吗？→ 能且收益巨大（每步只更新常数状态），这正是 RWKV/Mamba 类的卖点；短板在质量端而非速度端。

**关联**：[[/docs/llm/long-context-and-flashattention.md]] §4

### Q17. PagedAttention 和 RadixAttention 分别解决什么？⭐⭐

> **速答**：PagedAttention（vLLM）：借鉴操作系统虚拟内存，把 KV Cache 按固定大小**分页**（block），逻辑页映射到不连续物理页——消除预分配造成的显存碎片，支持 copy-on-write 共享（beam search、并行采样），显存利用率从不足一半提到接近打满。RadixAttention（SGLang）：用**基数树**缓存所有请求的 KV 前缀，多轮对话、few-shot 模板、多请求共享 system prompt 时自动复用前缀计算，省的是**重复 prefill**。

![PagedAttention 分页管理](../images/sglang-vllm-pagedattention-01.png)

图2：PagedAttention 的 KV Cache 分页管理（来源：vLLM / SGLang 相关技术资料）

**追问**：两者冲突吗？→ 不冲突，一个管"显存怎么摆"，一个管"算过的怎么复用"，SGLang 两者都用。

**关联**：[[/docs/llm/sglang-vs-vllm.md]]、[[/docs/engineering/inference-serving-optimization.md]]

---

## 五、手撕代码

### Q18. 手撕：写一个 GQA 前向（白板 10 分钟）⭐⭐⭐

> **速答**：三个考点——Q/K/V 投影的输出维度不同（$h$ vs $g$ 个头）；**缓存里始终只有 $g$ 个 KV 头**（省显存的本质）；计算时用 `repeat_interleave` 把 KV 头复制到 $h$ 个与 Q 对齐（或等价的 reshape-broadcast）。

```python
import torch, torch.nn as nn, torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.h, self.g = n_heads, n_kv_heads
        self.rep = n_heads // n_kv_heads            # 每个 KV 头被几个 Q 头共享
        self.dk = d_model // n_heads
        self.wq = nn.Linear(d_model, n_heads * self.dk, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.dk, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.dk, bias=False)
        self.wo = nn.Linear(n_heads * self.dk, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.dk).transpose(1, 2)   # (B,h,T,dk)
        k = self.wk(x).view(B, T, self.g, self.dk).transpose(1, 2)   # (B,g,T,dk)
        v = self.wv(x).view(B, T, self.g, self.dk).transpose(1, 2)
        if kv_cache is not None:                    # 增量解码：拼接历史
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)                          # 只缓存 g 个头！
        k = k.repeat_interleave(self.rep, dim=1)    # 计算时才复制到 h 个
        v = v.repeat_interleave(self.rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        return self.wo(out.transpose(1, 2).reshape(B, T, -1)), new_cache
```

**必答追问**：`repeat_interleave` 只是计算视图，不占缓存；生产实现会用 GQA 原生 kernel（FA2 支持）避免物化复制。

**关联**：[[/docs/llm/attention-variants-mha-mqa-gqa.md]] §8、[[/docs/interview/coding-must-write.md]]（MHA 完整版）

---

## 六、一页速记表（考前扫）

| # | 题目 | 一句话答案 |
|---|------|-----------|
| Q1 | 为什么除 √d_k | 点积方差随 d_k 线性涨，缩放防 softmax 饱和 |
| Q2 | 复杂度 | 计算 O(n²d)，朴素中间显存 O(n²)，长序列 n² 主导 |
| Q3 | softmax 稳定化 | 减行最大值，平移不变性防 exp 溢出 |
| Q4 | 朴素瓶颈在哪 | HBM 读写（n×n 矩阵三进三出），memory-bound 非算力 |
| Q5 | FlashAttention 原理 | 分块 + online softmax + 反向重计算；精确、显存 O(n)、快 2–4× |
| Q6 | online softmax | 维护 m/ℓ，换基准乘 e^(m旧-m新) 重缩放，一趟算完 |
| Q7 | FA1/2/3 | FA2 减非矩阵乘+序列维并行；FA3 面向 Hopper 异步+FP8 |
| Q8 | decode 为何带宽瓶颈 | 每步 1 token 算术强度低，全量权重+Cache 读一遍 |
| Q9 | KV Cache 公式 | 2·L·n_kv·d_head·b·s·字节；7B 每 token ≈ 0.5 MB |
| Q10 | MQA/GQA | 共享 KV 降 Cache；g=h 是 MHA、g=1 是 MQA，工业默认 g=8 |
| Q11 | MLA vs GQA | 减头数 vs 低秩压缩隐向量；MLA Cache 约 1/20 且效果更好 |
| Q12 | 解耦 RoPE | 旋转与低秩不可交换，独立 64 维位置键单独缓存 |
| Q13 | 优化能否叠加 | FA/GQA/Paged/量化四者正交，工业标配组合拳 |
| Q14 | Cache 其他省法 | KV 量化、H2O/StreamingLLM（attention sink）、跨层/前缀共享 |
| Q15 | 稀疏注意力 | 滑窗+全局/路由哈希/NSA；牺牲全连接换复杂度 |
| Q16 | 线性注意力 | 核技巧 O(nd²)、常数状态；长程精确检索弱，多用混合架构 |
| Q17 | Paged/Radix | 分页消碎片（vLLM）vs 基数树复用前缀（SGLang） |
| Q18 | 手撕 GQA | 缓存只存 g 个头，计算时 repeat_interleave 对齐 Q |

---

## 参考

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022, [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Dao, *FlashAttention-2: Better Attention with Better Parallelism and Work Partitioning*, 2023, [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Shah et al., *FlashAttention-3*, 2024, [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need (MQA)*, 2019, [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models*, 2023, [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- DeepSeek-AI, *DeepSeek-V2*（MLA）, 2024, [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention (vLLM)*, 2023, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Xiao et al., *Efficient Streaming Language Models with Attention Sinks (StreamingLLM)*, 2023, [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
