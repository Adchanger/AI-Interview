# 注意力变体：MHA / MQA / GQA / MLA（LLM 八股 10）

> **更新时间**：2026-08-31

> **标签**：MHA、MQA、GQA、MLA、KV Cache、面试八股

> **一句话**：MHA 每个头都有独立 K/V，KV Cache 最大；MQA 让所有头共享一份 K/V，Cache 降 h 倍但质量掉；GQA 分组共享（KV 头数 g）是当前工业默认折中；MLA 用低秩隐向量压缩 KV，在更小 Cache 下反而质量更好。

> **关联阅读**：[[/docs/llm/transformer-principle.md]]、[[/docs/llm/kv-cache.md]]、[[/docs/llm/mla-multi-head-latent-attention.md]]

---

## 1. 问题起点：解码阶段的瓶颈不是算力，是带宽

自回归生成每步只算 1 个 token，矩阵乘退化为矩阵-向量乘，**算术强度极低**；而每步都要把整个 KV Cache 从 HBM 读进 SRAM。于是：

$$\text{decode 阶段耗时} \approx \frac{\text{权重字节数} + \text{KV Cache 字节数}}{\text{显存带宽}}$$

**结论**：减小 KV Cache ≈ 直接提升解码吞吐并支持更大 batch / 更长上下文。MQA/GQA/MLA 全部围绕这一件事。

---

## 2. MHA（Multi-Head Attention）

$$\mathrm{head}_i = \mathrm{softmax}\!\left(\frac{Q_iK_i^\top}{\sqrt{d_k}}\right)V_i,\qquad \mathrm{MHA}=\mathrm{Concat}(\mathrm{head}_1..\mathrm{head}_h)W^O$$

- $h$ 个头各自有 $W^Q_i,W^K_i,W^V_i$，$d_k=d_{\text{model}}/h$；
- **多头的意义**：在多个子空间并行建模不同关系（句法/共指/位置模式），类似 CNN 的多通道；总参数量与单头 $d_{\text{model}}$ 维相同，但表达更丰富。

**KV Cache 大小**（每 token 每层）：$2\times h\times d_k \times \text{bytes} = 2\times d_{\text{model}}\times\text{bytes}$。

---

## 3. MQA（Multi-Query Attention，Shazeer 2019）

**所有查询头共享同一份 K/V**（1 个 KV 头）：

- KV Cache 降为原来的 $1/h$ —— 7B 模型 32 头即降 32 倍；
- 解码速度大幅提升（PaLM、Falcon、早期 StarCoder 采用）；
- 代价：**质量下降 + 训练不稳定**（论文与后续 GQA 论文均报告 quality degradation）。

---

## 4. GQA（Grouped-Query Attention，Ainslie et al. 2023）

把 $h$ 个 Q 头分成 $g$ 组，**每组共享一份 K/V**：

$$g=h \Rightarrow \text{MHA},\qquad g=1 \Rightarrow \text{MQA}$$

| 配置 | KV 头数 | KV Cache 相对 MHA | 质量 |
|------|---------|-------------------|------|
| MHA | $h$ | 1× | 基线 |
| GQA-8（如 LLaMA-2 70B：64 Q 头 / 8 KV 头） | 8 | $8/64=1/8$ | 接近 MHA |
| MQA | 1 | $1/h$ | 明显下降 |

**工程细节**：
- 论文提出可从已有 MHA 检查点 **uptrain**（对每组 KV 头做均值池化后用约 5% 预训练算力继续训）得到 GQA 模型，无需从头训；
- KV 头数常取 8，恰好匹配张量并行度（TP=8 时每卡 1 个 KV 头，无需跨卡复制/通信）；
- LLaMA-2 34B/70B、LLaMA-3 全系列、Mistral、Qwen2 之后基本都是 GQA。

> 面试高频：**GQA 的 g 怎么选？** → 经验上 $g=8$ 在质量/显存/并行度上最平衡：Cache 降到 1/8 已经解决大部分显存压力，质量损失可忽略，同时与常见 TP=8 对齐。

---

## 5. MLA（Multi-head Latent Attention，DeepSeek）

思路完全不同：不是减少 KV 头数，而是把 K/V **低秩压缩成一个隐向量** $c^{KV}_t$（维度远小于 $h\cdot d_k$），推理时只缓存这个隐向量，用上投影矩阵在计算中恢复 K/V；配合"矩阵吸收"技巧避免真正物化。

- KV Cache 比 GQA 更小（DeepSeek-V2 报告约为 MHA 的 1/20 量级），而效果**优于 MHA**；
- 由于 RoPE 与低秩压缩不可交换，MLA 采用**解耦 RoPE**：K 拆成"压缩部分 + 携带 RoPE 的独立部分"；
- 细节展开见 [[/docs/llm/mla-multi-head-latent-attention.md]]。

![MLA 与 MHA/GQA/MQA 的 KV Cache 对比](../images/mla-comparison-01.png)

图1：MHA / GQA / MQA / MLA 结构对比（来源：DeepSeek-V2 技术报告，arXiv:2405.04434）

---

## 6. 对比总表（必背）

| 方案 | KV 头数 | 每 token 每层 KV 元素数 | 相对 Cache | 代表模型 |
|------|---------|------------------------|-----------|----------|
| MHA | $h$ | $2h\,d_k$ | 1 | GPT-3、LLaMA-1、Qwen1 |
| GQA | $g$ | $2g\,d_k$ | $g/h$ | LLaMA-2 70B / LLaMA-3、Mistral、Qwen2+ |
| MQA | 1 | $2d_k$ | $1/h$ | PaLM、Falcon、早期 StarCoder |
| MLA | — | $d_c(+d_r)$ | 最小（约 1/20 量级） | DeepSeek-V2/V3 |

---

## 7. 其他注意力优化（区分"省 Cache"与"省算力"）

| 目标 | 技术 |
|------|------|
| **省 KV Cache**（显存/带宽） | MQA、GQA、MLA、跨层共享 KV（CLA/YOCO）、KV 量化（KV int8/fp8）、KV 淘汰（H2O、StreamingLLM 的 attention sink）、Prefix/Radix 前缀复用 |
| **省注意力算力**（$O(n^2)$） | 稀疏/滑窗注意力（Longformer、Mistral SWA）、线性注意力（Performer、RWKV、Mamba）、分块 + IO 优化（FlashAttention，**不改数学结果**） |
| **提升长度泛化** | RoPE 插值（NTK/YaRN）、ALiBi、位置内插 + 微调 |

> 面试高频：**FlashAttention 属于哪一类？** → 它是**精确注意力**的 IO 优化（分块 + online softmax，不物化 $n\times n$ 矩阵），既省显存又省时间，但**不减少 KV Cache**，与 GQA 正交、可叠加。见 [[/docs/llm/long-context-and-flashattention.md]]。

---

## 8. 手撕代码：GQA 前向

```python
import torch, torch.nn as nn, torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.h, self.g = n_heads, n_kv_heads
        self.rep = n_heads // n_kv_heads          # 每个 KV 头被几个 Q 头共享
        self.dk = d_model // n_heads
        self.wq = nn.Linear(d_model, n_heads * self.dk, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.dk, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.dk, bias=False)
        self.wo = nn.Linear(n_heads * self.dk, d_model, bias=False)

    def forward(self, x, kv_cache=None, causal=True):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.dk).transpose(1, 2)   # (B,h,T,dk)
        k = self.wk(x).view(B, T, self.g, self.dk).transpose(1, 2)   # (B,g,T,dk)
        v = self.wv(x).view(B, T, self.g, self.dk).transpose(1, 2)

        if kv_cache is not None:                                     # 增量解码
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)                                           # 只缓存 g 个头

        # 关键：把 g 个 KV 头复制成 h 个，与 Q 对齐
        k = k.repeat_interleave(self.rep, dim=1)
        v = v.repeat_interleave(self.rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal and T > 1)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.wo(out), new_cache
```

要点：`repeat_interleave` 只发生在计算时，**缓存里始终只有 g 个 KV 头**——这是省显存的本质。

---

## 9. 面试高频问题速查

1. **为什么要多头？** → 多子空间并行建模不同关系，提升表达力，参数量与单头等宽相当。
2. **MQA 的做法与代价？** → 所有 Q 头共享 1 份 KV；Cache 降 h 倍，质量下降、训练可能不稳。
3. **GQA 的定义与两个极端？** → 分组共享 KV；$g=h$ 是 MHA，$g=1$ 是 MQA。
4. **GQA 为什么常取 8 个 KV 头？** → 质量几乎无损、Cache 降 8 倍、与 TP=8 对齐避免跨卡复制。
5. **能把已有 MHA 模型改成 GQA 吗？** → 可以，对组内 KV 头均值池化后用少量算力 uptrain（GQA 论文约 5%）。
6. **MLA 与 GQA 的本质差别？** → GQA 是"减少 KV 头数"，MLA 是"低秩压缩 KV 到隐向量"；MLA Cache 更小且质量更好，但实现更复杂（需解耦 RoPE、矩阵吸收）。
7. **KV Cache 怎么算？** → $2\cdot L\cdot n_{kv}\cdot d_{head}\cdot b\cdot s\cdot\text{bytes}$，详见 [[/docs/llm/kv-cache.md]]。
8. **为什么 decode 阶段是带宽瓶颈？** → 每步计算量小但要读全部权重 + KV Cache，算术强度低。
9. **注意力为什么要除 $\sqrt{d_k}$？** → 点积方差随 $d_k$ 增长，缩放避免 softmax 饱和、梯度消失。
10. **这些变体会影响训练还是只影响推理？** → 结构改动在训练时就要确定（MQA/GQA/MLA 都是架构层面）；FlashAttention、KV 量化、PagedAttention 属于实现优化，可只在推理侧启用。

---

## 参考

- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need (MQA)*, arXiv:1911.02150
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, arXiv:2305.13245
- DeepSeek-AI, *DeepSeek-V2*, arXiv:2405.04434（MLA）
- Touvron et al., *Llama 2*, arXiv:2307.09288（34B/70B 使用 GQA）
