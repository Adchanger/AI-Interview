# KV Cache：原理、显存公式与优化（LLM 八股 11）

> **更新时间**：2026-08-31

> **标签**：KV Cache、prefill、decode、显存估算、PagedAttention、面试八股

> **一句话**：自回归解码时每个 token 的 K/V 只取决于自己，因此可以缓存复用，把每步 $O(n^2)$ 的重算降为 $O(n)$；代价是显存随 batch×长度线性膨胀，于是有了 GQA/MLA（减小）、PagedAttention（不浪费）、量化与淘汰（压缩）等一整套优化。

> **关联阅读**：[[/docs/llm/attention-variants-mha-mqa-gqa.md]]、[[/docs/engineering/inference-serving-optimization.md]]、[[/docs/llm/sglang-vs-vllm.md]]

---

## 1. 为什么能缓存

因果注意力下，位置 $t$ 的 $k_t=W^Kx_t,\ v_t=W^Vx_t$ **只依赖 $x_t$**，与后续 token 无关；而 $q_t$ 每步都要新算。所以生成第 $t+1$ 个 token 时：

- 需要：$q_{t+1}$（新算）+ $k_{1..t+1},v_{1..t+1}$（前 $t$ 个可复用）；
- 没有 Cache：每步重算全部 $K,V$，总复杂度 $O(n^3)$ 级；
- 有 Cache：每步只算 1 个新 K/V，总复杂度降一个量级。

> 面试高频：**为什么不缓存 Q？** → $q_t$ 只在生成第 $t$ 个 token 时用一次，之后再不需要；K/V 则被后续所有 token 反复使用。

---

## 2. Prefill 与 Decode 两阶段（必考）

| 阶段 | 输入 | 计算特征 | 瓶颈 | 关键指标 |
|------|------|----------|------|----------|
| **Prefill**（预填充） | 整个 prompt（n 个 token 并行） | 大矩阵乘、算术强度高 | **算力（compute-bound）** | **TTFT**（首 token 延迟） |
| **Decode**（增量解码） | 每步 1 个 token | 矩阵-向量乘、要读全部权重+Cache | **显存带宽（memory-bound）** | **TPOT / ITL**（每 token 延迟） |

由此推出一系列工程结论：
- Prefill 长 → TTFT 高 → 用 **chunked prefill**（切块）与 **prefix cache 复用**（相同系统提示只算一次）；
- Decode 慢且难加速 → 靠**增大 batch**摊薄权重读取（continuous batching）、**减小 KV Cache**、**投机解码**一次多出几个 token；
- Prefill 与 Decode 争抢资源 → 有 **PD 分离**（prefill/decode disaggregation）部署方案。

---

## 3. 显存公式（面试必须能手算）

$$\boxed{\text{KV Cache 字节数} = 2 \times L \times n_{kv} \times d_{head} \times b \times s \times \text{bytes}}$$

- 2：K 和 V 各一份；$L$：层数；$n_{kv}$：KV 头数（MHA 时 = $h$）；$d_{head}$：每头维度；$b$：batch；$s$：序列总长（prompt + 已生成）；bytes：dtype 字节数（fp16/bf16 = 2）。

### 3.1 算例 1：LLaMA-2 7B（MHA）

$L=32,\ h=n_{kv}=32,\ d_{head}=128$，fp16：

**每 token 每层** = $2\times32\times128\times2 = 16{,}384$ B = 16 KB
**每 token（全 32 层）** = $16\text{KB}\times32 = 512$ KB

→ 4096 token 单条序列 ≈ **2 GB**；batch=16 ≈ **32 GB**（比模型权重 13GB 还大！）

### 3.2 算例 2：LLaMA-2 70B（GQA，$n_{kv}=8$）

$L=80,\ d_{head}=128,\ n_{kv}=8$，fp16：

每 token 每层 = $2\times8\times128\times2=4096$ B = 4 KB；每 token = $4\text{KB}\times80=320$ KB
→ 4096 token ≈ **1.25 GB/序列**。若用 MHA（64 头）会是 **10 GB/序列**，直接不可用 —— 这就是 GQA 的价值。

### 3.3 快速估算口诀

$$\text{每 token 每层} = 4\times n_{kv}\times d_{head}\ \text{bytes (fp16)}$$

再乘层数得"每 token 多少 KB"，最后乘 `batch × 长度`。

---

## 4. KV Cache 优化全景

| 方向 | 方法 | 要点 |
|------|------|------|
| **减少头数/维度** | MQA、GQA、**MLA** | 架构级，训练时决定，见 [[/docs/llm/attention-variants-mha-mqa-gqa.md]] |
| **跨层共享** | CLA（Cross-Layer Attention）、YOCO | 多层复用同一份 KV，进一步线性降低 |
| **降低精度** | KV int8 / fp8 / int4 | 收益直接（4bit ≈ 1/4），需 per-channel/per-token 量化控误差；对长上下文精度影响需实测 |
| **不浪费显存** | **PagedAttention**（vLLM） | 把 Cache 分成固定大小 block 按需分配，消除预留式分配的内/外部碎片，显存利用率从 ~20-40% 提到接近 100% |
| **复用前缀** | Prefix Caching、**RadixAttention**（SGLang） | 多请求共享系统提示/few-shot 前缀，命中即跳过 prefill |
| **淘汰/稀疏** | StreamingLLM（attention sink + 滑窗）、H2O（heavy hitter）、SnapKV、Quest | 只保留重要 token 的 KV，长度不再线性增长；有信息损失，需评测 |
| **卸载** | KV offload 到 CPU/NVMe（FlexGen 等） | 换带宽换容量，延迟敏感场景慎用 |

> **StreamingLLM 的经典发现（attention sink）**：模型会把大量注意力分配给**最开始的几个 token**（即使它们语义无关），因此滑窗时必须**保留最初的 4 个 token**，否则 PPL 崩掉。这是一道很能体现深度的面试点。

![PagedAttention 的分块 KV 管理](../images/sglang-vllm-pagedattention-01.png)

图1：PagedAttention 用页表管理 KV block，消除碎片（来源：vLLM / Efficient Memory Management for LLM Serving，arXiv:2309.06180）

---

## 5. 与吞吐的关系（服务视角）

一张 80GB A100/H100 上：可用显存 = 80 − 权重 − 激活/框架开销 → 剩下的都给 KV Cache。

$$\text{最大并发} \approx \frac{\text{可用显存}}{\text{单序列 KV 大小}}$$

- 减小单序列 KV（GQA/MLA/量化）→ 并发上升 → **decode 阶段的权重读取被更多 token 摊薄** → 吞吐近线性提升；
- 这就是"KV Cache 优化 = 吞吐优化"的因果链，面试时能讲清这条链条比背名词重要得多。

---

## 6. 手撕代码：带 KV Cache 的增量解码

```python
import torch

@torch.no_grad()
def generate(model, input_ids, max_new_tokens=64, temperature=1.0, top_p=0.9):
    """model(x, cache) -> (logits, new_cache)；演示 prefill + decode 两阶段"""
    # ---- Prefill：整段 prompt 一次前向，产出初始 cache ----
    logits, cache = model(input_ids, cache=None)
    out = input_ids
    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :] / max(temperature, 1e-5)
        probs = torch.softmax(next_logits, dim=-1)
        # nucleus 采样
        sp, si = torch.sort(probs, descending=True, dim=-1)
        mask = (sp.cumsum(-1) - sp) > top_p
        sp[mask] = 0.0
        sp = sp / sp.sum(-1, keepdim=True)
        nxt = si.gather(-1, torch.multinomial(sp, 1))
        out = torch.cat([out, nxt], dim=1)
        # ---- Decode：只喂 1 个新 token，复用 cache ----
        logits, cache = model(nxt, cache=cache)
    return out
```

关键点：prefill 传全序列，decode 每步只传 1 个 token 且带上 cache；cache 的长度维每步 +1。

---

## 7. 面试高频问题速查

1. **KV Cache 解决什么问题？** → 避免每步重算历史 K/V，把解码从平方级重算降为线性增量。
2. **为什么只缓存 K/V 不缓存 Q？** → K/V 会被后续所有 token 使用，Q 只用一次。
3. **KV Cache 显存怎么算？** → $2\,L\,n_{kv}\,d_{head}\,b\,s\,\text{bytes}$，要能现场算出 7B/70B 的数值。
4. **prefill 和 decode 的瓶颈分别是什么？** → prefill 受算力约束、decode 受显存带宽约束。
5. **TTFT 与 TPOT 分别受什么影响？** → TTFT 受 prompt 长度/prefill 优化/前缀缓存影响；TPOT 受权重+Cache 读取带宽、batch 大小、投机解码影响。
6. **PagedAttention 的核心思想？** → 借操作系统分页思想，KV 分 block 按需分配 + 页表映射，消除碎片、支持共享（copy-on-write）。
7. **怎么在不改架构的情况下减小 KV Cache？** → KV 量化、前缀复用、Cache 淘汰（StreamingLLM/H2O）、offload。
8. **StreamingLLM 为什么要留最初几个 token？** → attention sink 现象，去掉初始 token 会使注意力分布崩坏、PPL 暴涨。
9. **KV Cache 与 batch 的关系？** → Cache 越小可并发越多，decode 的权重读取被摊薄，吞吐提升。
10. **投机解码为什么能加速 decode？** → 用小模型/多头一次提议多个 token，大模型一次前向并行验证，把"带宽受限的多步"变成"一步"，见 [[/docs/engineering/inference-serving-optimization.md]]。
11. **长上下文下 KV Cache 会成为主要显存开销吗？** → 会。128k 上下文时 KV 常远超权重，必须 GQA/MLA + 量化 + 分页联合使用。

---

## 参考

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, arXiv:2309.06180
- Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, arXiv:2309.17453
- Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference*, arXiv:2306.14048
- Pope et al., *Efficiently Scaling Transformer Inference*, arXiv:2211.05102
- Brandon et al., *Reducing Transformer Key-Value Cache Size with Cross-Layer Attention*, arXiv:2405.12981
